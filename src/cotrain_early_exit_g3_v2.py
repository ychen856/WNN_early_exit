# src/train/train_wnn.py
import argparse
import copy
from pathlib import Path
import json
from networkx import sigma
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from src.core.linearExitHead import build_exits_from_ckpt
from src.dataio.data import build_loaders_bits
from src.dataio.mapping import make_tuple_mapping, audit_mapping
from src.early_exit import _head_logits_from_hidden_trainable, _margin_from_logits
from src.exit.analyze_hidden import compute_mu_sigma
from src.prune import *
from src.early_exit import *
from src.tools.utils import _head_logits_from_hidden, print_sweep_table  
from test import *
from src.core.infer import *
from src.core.multiLayerWNN import MultiLayerWNN, build_backbone_from_ckpt, load_ckpt, save_ckpt, save_ckpt_v2
from src.dataio.encode import minmax_normalize, thermometer_encode, dt_thermometer_encode, compute_dt_thresholds
from src.tools.fpga_tools.export_fpga_bundle import export_multilayer_2layer_for_fpga, verify_multilayer_export
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

# from core.decision import tune_decision  #  Step 2

CANONICAL_MAPPING = Path("/Users/yi-chunchen/workspace/WNN_early_exit/models/meta/tuple_mapping.json")

def load_or_create_mapping(bit_len, tiles, num_luts, addr_bits, seed=42, save_path=CANONICAL_MAPPING):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        mapping = json.loads(save_path.read_text())
        # alignment check
        assert len(mapping) == num_luts, "num_luts mismatch with saved mapping"
        return mapping

    mapping = make_tuple_mapping(
        num_luts=num_luts,
        addr_bits=addr_bits,
        bit_len=bit_len,
        tiles=tiles,          #  None or meta["tile_index_ranges"]
        seed=seed
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    return mapping


def get_lr(epoch):
    if epoch < 25:
        return 1e-3
    elif epoch < 55:
        return 3e-4
    else:
        return 1e-4

def compute_accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

# -----------------------------
# 2) Train exit head on cached features
# -----------------------------
def train_exit_head_on_cached(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes=10,
    num_epochs=50,
    lr=3e-3,
    weight_decay=1e-4,
    batch_size=1024,
    device="cpu",
):
    """
    Trains a simple Linear classifier on cached features.
    Returns:
      clf: nn.Linear(K -> num_classes)
      best_state: best weights (loaded into clf)
    """
    K = X_train.size(1)
    clf = nn.Linear(K, num_classes, bias=True).to(device)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # --- train ---
        clf.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = clf(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # --- val ---
        clf.eval()
        v_total_loss = 0.0
        v_total = 0
        v_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = clf(xb)
                loss = F.cross_entropy(logits, yb)
                v_total_loss += loss.item() * xb.size(0)
                pred = logits.argmax(dim=-1)
                v_correct += (pred == yb).sum().item()
                v_total += xb.size(0)

        val_loss = v_total_loss / v_total
        val_acc = v_correct / v_total

        print(
            f"[cached-exit] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}

    if best_state is not None:
        clf.load_state_dict(best_state)

    return clf

def train_exit_head(model, train_loader, val_loader, device,
                    num_epochs=50, base_lr=1e-3, weight_decay=1e-4):
    model.to(device)

    # freeze backbone + final classifier
    for p in model.layers.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False

    # train only exit head
    for p in model.exit1_classifier.parameters():
        p.requires_grad = True

    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable:", trainable)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

    best_state = None
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, exit1_logits, _ = model.forward_with_all_hidden_and_exits(xb)
            loss = F.cross_entropy(exit1_logits, yb)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = eval_exit1_epoch(model, train_loader, device)
        val_loss, val_acc = eval_exit1_epoch(model, val_loader, device)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
              f"train_acc={train_acc*100:.2f}% | val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


import torch
import torch.nn.functional as F

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

@torch.no_grad()
def eval_final_only(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)  # final only
        loss = F.cross_entropy(logits, yb)
        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        loss_sum += loss.item() * yb.numel()
    return loss_sum / total, correct / total

@torch.no_grad()
def eval_exit1_only(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        _, exit1_logits, _ = model.forward_with_all_hidden_and_exits(xb)
        loss = F.cross_entropy(exit1_logits, yb)
        pred = exit1_logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        loss_sum += loss.item() * yb.numel()
    return loss_sum / total, correct / total

@torch.no_grad()
def eval_with_gate(model, loader, device, thr=2.0):
    """
    Use your gating rule (logit margin) on exit1 logits.
    Returns: overall_acc, exit_rate
    """
    model.eval()
    total, correct, exited = 0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        final_logits, exit1_logits, _ = model.forward_with_all_hidden_and_exits(xb)

        # margin on logits (NOT softmax prob) — matches your earlier approach
        top2 = torch.topk(exit1_logits, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        exit_mask = margin > thr

        logits = final_logits.clone()
        logits[exit_mask] = exit1_logits[exit_mask]

        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        exited += exit_mask.sum().item()

    overall_acc = correct / total
    exit_rate = exited / total
    return overall_acc, exit_rate

# -------------------------
# G2 training
# -------------------------

def dump_trainable(model, tag=""):
    trainable = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable.append((n, tuple(p.shape)))
    print(f"[{tag}] trainable params ({len(trainable)}):")
    for n, shp in trainable[:40]:
        print("  ", n, shp)
    if len(trainable) > 40:
        print("  ...")

def freeze_g2(model, freeze_exit=True):
    # freeze layer1
    for p in model.layers[0].parameters():
        p.requires_grad = False

    # train layer2 + final
    for p in model.layers[1].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    # exit head normally freeze in G2
    for p in model.exit1_classifier.parameters():
        p.requires_grad = (not freeze_exit)

@dataclass
class G2Config:
    num_epochs: int = 20
    lr_layer2: float = 3e-4
    lr_final: float = 3e-4
    lr_exit: float = 0.0           # 0 = freeze exit head
    lambda_exit: float = 0.0       # start with 0; can set 0.05~0.1
    weight_decay: float = 1e-3
    grad_clip: Optional[float] = 1.0
    thr_eval: float = 2.0



import copy
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from typing import List, Sequence, Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# assumes you already have:
# - set_requires_grad(module, bool)
# - eval_cascade_multi_exit(...)
# - _head_logits_from_hidden_trainable(...)
# - _margin_from_logits(...)
# - model.forward_with_all_hidden(xb)

def _build_optimizer(
    params_backbone, params_classifier, params_exits,
    lr_backbone, lr_classifier, lr_exits, weight_decay
):
    return torch.optim.AdamW(
        [
            {"params": params_backbone,   "lr": lr_backbone,   "weight_decay": weight_decay},
            {"params": params_classifier, "lr": lr_classifier, "weight_decay": weight_decay},
            {"params": params_exits,      "lr": lr_exits,      "weight_decay": weight_decay},
        ]
    )

def _collect_params(model, exit_heads, train_layer_indices):
    params_backbone = []
    for li in train_layer_indices:
        params_backbone += [p for p in model.layers[li].parameters() if p.requires_grad]
    params_classifier = [p for p in model.classifier.parameters() if p.requires_grad]
    params_exits = []
    for h in exit_heads:
        params_exits += [p for p in h.parameters() if p.requires_grad]
    return params_backbone, params_classifier, params_exits


def cotrain_g3_multi_exit_staged(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    *,
    # exits / cfg
    exit_heads: List[nn.Module],
    payload_exit_cfg: List[dict],
    thrs: Sequence[float],

    # loss
    lambda_final: float = 1.0,
    lambda_exits: Optional[Sequence[float]] = None,  # length=num_exits

    # gate-weighting
    use_gate_weighting: bool = True,
    use_prob_margin: bool = False,
    gate_T: Optional[Sequence[float]] = None,   # e.g. [3.5, 1.25]

    # optim base
    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,

    # staged schedule
    stage_cfgs: List[Dict[str, Any]] = None,
    # each stage cfg keys (suggested):
    # {
    #   "name": "A",
    #   "epochs": 10,
    #   "train_layers": [],             # backbone layers to unfreeze (empty => none)
    #   "freeze_layers": "all",         # or explicit list
    #   "train_exit_ids": (1,),         # exits to unfreeze
    #   "lr_backbone": 1e-5,
    #   "lr_classifier": 3e-4,
    #   "lr_exits": 5e-4,
    # }

    # best selection
    best_metric: str = "val_overall_acc",  # "val_overall_acc" | "val_final_only"
):
    """
    g3 staged: same joint loss as g3, but unfreeze progressively.
    """

    model = model.to(device)
    exit_heads = [h.to(device) for h in exit_heads]

    assert len(exit_heads) == len(payload_exit_cfg)
    num_exits = len(exit_heads)
    assert len(thrs) == num_exits

    if lambda_exits is None:
        lambda_exits = [0.05] * num_exits
    else:
        assert len(lambda_exits) == num_exits

    if gate_T is None:
        # keep your original default
        gate_T = [3.5, 1.25]
    assert len(gate_T) == num_exits, f"gate_T must match num_exits, got {len(gate_T)} vs {num_exits}"

    if stage_cfgs is None or len(stage_cfgs) == 0:
        raise ValueError("stage_cfgs must be provided (list of stage configs).")

    best = {"metric": -1.0, "stage": None, "epoch": None, "state_model": None, "state_exits": None}

    eps = 1e-8
    global_epoch = 0

    def _apply_freeze(stage_cfg):
        # 0) freeze all
        set_requires_grad(model, False)
        for h in exit_heads:
            set_requires_grad(h, False)

        # 1) backbone layer freeze policy
        if stage_cfg.get("freeze_layers", "all") == "all":
            # already frozen
            pass
        else:
            # freeze specific layers (redundant but explicit)
            for li in stage_cfg["freeze_layers"]:
                if 0 <= li < len(model.layers):
                    set_requires_grad(model.layers[li], False)

        # 2) unfreeze train layers
        train_layers = stage_cfg.get("train_layers", [])
        for li in train_layers:
            if li < 0 or li >= len(model.layers):
                raise ValueError(f"train_layers contains out-of-range layer {li}")
            set_requires_grad(model.layers[li], True)

        # 3) classifier always trainable in g3 (unless you want otherwise)
        set_requires_grad(model.classifier, True)

        # 4) unfreeze selected exits
        train_exit_ids = tuple(stage_cfg.get("train_exit_ids", ()))
        for i in train_exit_ids:
            set_requires_grad(exit_heads[i], True)

        return train_layers, train_exit_ids

    for si, stage_cfg in enumerate(stage_cfgs):
        stage_name = stage_cfg.get("name", f"S{si}")
        stage_epochs = int(stage_cfg.get("epochs", 0))
        if stage_epochs <= 0:
            continue

        train_layers, train_exit_ids = _apply_freeze(stage_cfg)

        # build optimizer for this stage
        params_backbone, params_classifier, params_exits = _collect_params(model, exit_heads, train_layers)

        lr_backbone = float(stage_cfg.get("lr_backbone", 1e-5))
        lr_classifier = float(stage_cfg.get("lr_classifier", 3e-4))
        lr_exits = float(stage_cfg.get("lr_exits", 5e-4))

        optimizer = _build_optimizer(
            params_backbone, params_classifier, params_exits,
            lr_backbone, lr_classifier, lr_exits, weight_decay
        )

        print(f"[g3-staged:{stage_name}] epochs={stage_epochs} "
              f"train_layers={train_layers} train_exit_ids={train_exit_ids} "
              f"lr(backbone/cls/exits)={lr_backbone}/{lr_classifier}/{lr_exits} "
              f"params(backbone/cls/exits)={sum(p.numel() for p in params_backbone)}/"
              f"{sum(p.numel() for p in params_classifier)}/"
              f"{sum(p.numel() for p in params_exits)}")

        # ---- train stage ----
        for e in range(stage_epochs):
            model.train()
            for h in exit_heads:
                h.train()

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)

                final_logits, h_list = model.forward_with_all_hidden(xb)
                ce_final = F.cross_entropy(final_logits, yb, reduction="none")  # [B]

                if use_gate_weighting:
                    ce_exit_list = []
                    m_list = []
                    for i in range(num_exits):
                        li = int(payload_exit_cfg[i]["layer_idx"])
                        logits_i = _head_logits_from_hidden_trainable(exit_heads[i], h_list[li], device)
                        ce_i = F.cross_entropy(logits_i, yb, reduction="none")
                        ce_exit_list.append(ce_i)
                        m_i = _margin_from_logits(logits_i, use_prob=use_prob_margin)
                        m_list.append(m_i)

                    u = torch.ones_like(ce_final)
                    loss_exit_sum = 0.0

                    for i in range(num_exits):
                        thr_i = float(thrs[i])
                        Ti = float(gate_T[i])
                        w_i = torch.sigmoid((m_list[i] - thr_i) / Ti)
                        take_i = u * w_i
                        take_i_det = take_i.detach()

                        # only count exit loss if that exit is trainable in THIS stage
                        if i in train_exit_ids:
                            loss_i = (take_i_det * ce_exit_list[i]).sum() / (take_i_det.sum() + eps)
                            loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * loss_i

                        u = u * (1.0 - w_i)

                    u_det = u.detach()
                    loss_final = (u_det * ce_final).sum() / (u_det.sum() + eps)

                    loss = float(lambda_final) * loss_final + loss_exit_sum
                else:
                    loss_final = ce_final.mean()
                    loss_exit_sum = 0.0
                    for i in range(num_exits):
                        if i not in train_exit_ids:
                            continue
                        li = int(payload_exit_cfg[i]["layer_idx"])
                        logits_i = _head_logits_from_hidden_trainable(exit_heads[i], h_list[li], device)
                        loss_exit_i = F.cross_entropy(logits_i, yb)
                        loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * loss_exit_i
                    loss = float(lambda_final) * loss_final + loss_exit_sum

                loss.backward()

                if grad_clip is not None:
                    if params_backbone:
                        torch.nn.utils.clip_grad_norm_(params_backbone, grad_clip)
                    if params_classifier:
                        torch.nn.utils.clip_grad_norm_(params_classifier, grad_clip)
                    if params_exits:
                        torch.nn.utils.clip_grad_norm_(params_exits, grad_clip)

                optimizer.step()

            # ---- eval each epoch ----
            model.eval()
            for h in exit_heads:
                h.eval()

            out_val = eval_cascade_multi_exit(
                model, val_loader, device,
                exit_heads=exit_heads,
                exit_cfg_list=payload_exit_cfg,
                thrs=thrs,
                use_prob_margin=use_prob_margin,
                log_margins=False,
            )
            va_overall = float(out_val["overall_acc"])
            va_final_only = float(out_val.get("final_acc", 0.0))  # if your eval provides it

            if best_metric == "val_overall_acc":
                metric = va_overall
            elif best_metric == "val_final_only":
                metric = va_final_only
            else:
                raise ValueError("best_metric must be 'val_overall_acc' or 'val_final_only'")

            print(
                f"[G3-staged:{stage_name}] Ep{global_epoch:03d} (stage_e={e:03d}) "
                f"| overall@{tuple(thrs)} va={va_overall*100:.2f} "
                f"| final_rate={out_val['final_rate']:.4f} exit_rates={out_val['exit_rates']} "
                f"| final_only={va_final_only*100:.2f}"
            )

            if metric > best["metric"]:
                best["metric"] = metric
                best["stage"] = stage_name
                best["epoch"] = global_epoch
                best["state_model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best["state_exits"] = [{k: v.detach().cpu().clone() for k, v in h.state_dict().items()} for h in exit_heads]

            global_epoch += 1

    # ---- restore best ----
    if best["state_model"] is not None:
        model.load_state_dict(best["state_model"], strict=True)
        for i, h in enumerate(exit_heads):
            h.load_state_dict(best["state_exits"][i], strict=True)

    return model, exit_heads, best


def cotrain_g3_multi_exit_staged_backup(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs: int = 30,

    # g3: train all layers by default
    train_layer_indices: Optional[Sequence[int]] = None,   # None => all
    freeze_layer_indices: Sequence[int] = (),              # usually empty

    exit_heads: List[nn.Module],
    payload_exit_cfg: List[dict],

    # gating thresholds used for BOTH training loss and cascade eval
    thrs: Sequence[float] = (227.0, 18.0),
    use_prob_margin: bool = False,

    # loss weights
    lambda_final: float = 1.0,
    lambda_exits: Optional[Sequence[float]] = None,

    # optim
    lr_backbone: float = 1e-4,
    lr_classifier: float = 3e-4,
    lr_exits: float = 5e-4,
    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,

    # gate weighting
    use_gate_weighting: bool = True,
    # NOTE: 你原本寫死的 T，如果你想保留就保留
    # 建議之後改成依 margin 尺度自動設定
    gate_T: Optional[Sequence[float]] = (3.5, 1.25),

    # ---- staged training controls ----
    warmup_epochs: int = 10,                 # Phase A: freeze all exits
    train_exit_ids: Sequence[int] = (1,),    # Phase B: unfreeze which exits (default: only exit1)

    # best selection
    best_metric: str = "val_overall_acc",    # "val_overall_acc" / "val_final_only"
):
    """
    g3 staged:
      Phase A (epoch < warmup_epochs): train backbone+classifier only, exits frozen.
      Phase B (epoch >= warmup_epochs): train backbone+classifier + selected exits jointly.

    The loss follows your existing gate-weighting logic (cascade-style soft weighting),
    but only trains exit heads after warmup.
    """
    eps = 1e-8
    model = model.to(device)

    assert len(exit_heads) == len(payload_exit_cfg)
    num_exits = len(exit_heads)
    assert len(thrs) == num_exits, f"len(thrs) must match num_exits ({num_exits})"

    if lambda_exits is None:
        lambda_exits = [0.3] * num_exits
    else:
        assert len(lambda_exits) == num_exits

    if gate_T is not None:
        assert len(gate_T) == num_exits, f"gate_T must match num_exits ({num_exits})"

    # -----------------------
    # Helpers
    # -----------------------
    def set_requires_grad(mod: nn.Module, flag: bool):
        for p in mod.parameters():
            p.requires_grad = flag

    def build_optimizer(train_exits: bool) -> Tuple[torch.optim.Optimizer, list, list, list]:
        # collect params
        params_backbone = []
        for li in train_layer_indices_eff:
            params_backbone += [p for p in model.layers[li].parameters() if p.requires_grad]

        params_classifier = [p for p in model.classifier.parameters() if p.requires_grad]

        params_exits = []
        if train_exits:
            for i, h in enumerate(exit_heads):
                if i in train_exit_ids:
                    params_exits += [p for p in h.parameters() if p.requires_grad]

        print(
            f"[g3-staged] build_optimizer(train_exits={train_exits}) "
            f"backbone={sum(p.numel() for p in params_backbone)} "
            f"classifier={sum(p.numel() for p in params_classifier)} "
            f"exits={sum(p.numel() for p in params_exits)}"
        )

        groups = [
            {"params": params_backbone,   "lr": lr_backbone,   "weight_decay": weight_decay},
            {"params": params_classifier, "lr": lr_classifier, "weight_decay": weight_decay},
        ]
        if train_exits:
            groups.append({"params": params_exits, "lr": lr_exits, "weight_decay": weight_decay})

        optimizer = torch.optim.AdamW(groups)
        return optimizer, params_backbone, params_classifier, params_exits

    # -----------------------
    # 0) Decide train layers
    # -----------------------
    if train_layer_indices is None:
        train_layer_indices_eff = tuple(range(len(model.layers)))
    else:
        train_layer_indices_eff = tuple(train_layer_indices)

    # -----------------------
    # 1) Setup: freeze all, then unfreeze backbone/classifier (exits handled per phase)
    # -----------------------
    set_requires_grad(model, False)

    # freeze specified layers (usually none in g3)
    for li in freeze_layer_indices:
        if 0 <= li < len(model.layers):
            set_requires_grad(model.layers[li], False)

    # unfreeze train layers
    for li in train_layer_indices_eff:
        if li < 0 or li >= len(model.layers):
            raise ValueError(f"train_layer_indices contains out-of-range layer {li}")
        set_requires_grad(model.layers[li], True)

    # unfreeze classifier
    if not hasattr(model, "classifier"):
        raise ValueError("model has no classifier")
    set_requires_grad(model.classifier, True)

    # move exits to device
    exit_heads = [h.to(device) for h in exit_heads]

    # -----------------------
    # 2) Phase A: exits frozen
    # -----------------------
    for h in exit_heads:
        set_requires_grad(h, False)

    optimizer, params_backbone, params_classifier, params_exits = build_optimizer(train_exits=False)

    best = {
        "metric": -1.0,
        "epoch": -1,
        "state_model": None,
        "state_exits": None,
    }

    # -----------------------
    # 3) Train loop
    # -----------------------
    for epoch in range(num_epochs):
        # Phase switch
        if epoch == warmup_epochs:
            # Phase B: unfreeze selected exits + rebuild optimizer
            for i, h in enumerate(exit_heads):
                set_requires_grad(h, i in train_exit_ids)

            optimizer, params_backbone, params_classifier, params_exits = build_optimizer(train_exits=True)

        warmup = (epoch < warmup_epochs)

        model.train()
        for h in exit_heads:
            # even frozen exits can be in eval mode; but train() is fine (no grads anyway)
            h.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            final_logits, h_list = model.forward_with_all_hidden(xb)
            ce_final = F.cross_entropy(final_logits, yb, reduction="none")  # [B]

            # ---- exit logits / margin ----
            if use_gate_weighting:
                ce_exit_list = []
                m_list = []

                for i in range(num_exits):
                    li = int(payload_exit_cfg[i]["layer_idx"])
                    h_i = h_list[li]
                    # IMPORTANT: use the same head forward you used elsewhere
                    logits_i = _head_logits_from_hidden_trainable(exit_heads[i], h_i, device)  # [B,C]
                    ce_i = F.cross_entropy(logits_i, yb, reduction="none")
                    m_i = _margin_from_logits(logits_i, use_prob=use_prob_margin)
                    ce_exit_list.append(ce_i)
                    m_list.append(m_i)

                # cascade-style soft weighting
                u = torch.ones_like(ce_final)
                loss_exit_sum = 0.0

                for i in range(num_exits):
                    thr_i = float(thrs[i])

                    # temperature
                    if gate_T is None:
                        Ti = 1.0
                    else:
                        Ti = float(gate_T[i])

                    w_i = torch.sigmoid((m_list[i] - thr_i) / Ti)  # [B]
                    take_i = u * w_i                                # [B]
                    take_i_det = take_i.detach()

                    # only train exit loss after warmup AND only for train_exit_ids
                    if (not warmup) and (i in train_exit_ids) and any(p.requires_grad for p in exit_heads[i].parameters()):
                        loss_i = (take_i_det * ce_exit_list[i]).sum() / (take_i_det.sum() + eps)
                        loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * loss_i

                    # update undecided prob
                    u = u * (1.0 - w_i)

                u_det = u.detach()
                loss_final = (u_det * ce_final).sum() / (u_det.sum() + eps)

                loss = float(lambda_final) * loss_final + loss_exit_sum

            else:
                # simpler fallback
                loss_final = ce_final.mean()
                loss_exit_sum = 0.0
                if not warmup:
                    for i in range(num_exits):
                        if i not in train_exit_ids:
                            continue
                        li = int(payload_exit_cfg[i]["layer_idx"])
                        logits_i = _head_logits_from_hidden_trainable(exit_heads[i], h_list[li], device)
                        loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * F.cross_entropy(logits_i, yb)

                loss = float(lambda_final) * loss_final + loss_exit_sum

            loss.backward()

            if grad_clip is not None:
                if params_backbone:
                    torch.nn.utils.clip_grad_norm_(params_backbone, grad_clip)
                if params_classifier:
                    torch.nn.utils.clip_grad_norm_(params_classifier, grad_clip)
                if params_exits:
                    torch.nn.utils.clip_grad_norm_(params_exits, grad_clip)

            optimizer.step()

        # -----------------------
        # 4) Eval
        # -----------------------
        model.eval()
        for h in exit_heads:
            h.eval()

        out_val = eval_cascade_multi_exit(
            model, val_loader, device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=thrs,
            use_prob_margin=use_prob_margin,
            log_margins=False,
        )

        va_overall = float(out_val["overall_acc"])
        va_final_only = float(out_val.get("final_acc", 0.0))

        print(
            f"[G3-staged] Ep{epoch:03d} | warmup={warmup} | thrs={tuple(thrs)} "
            f"| overall={va_overall*100:.2f} | final_only={va_final_only*100:.2f} "
            f"| exit_rates={out_val['exit_rates']} final_rate={out_val['final_rate']:.4f}"
        )

        if best_metric == "val_overall_acc":
            metric = va_overall
        elif best_metric == "val_final_only":
            metric = va_final_only
        else:
            raise ValueError("best_metric must be 'val_overall_acc' or 'val_final_only'")

        if metric > best["metric"]:
            best["metric"] = metric
            best["epoch"] = epoch
            best["state_model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["state_exits"] = [{k: v.detach().cpu().clone() for k, v in h.state_dict().items()} for h in exit_heads]

    # -----------------------
    # 5) Restore best
    # -----------------------
    if best["state_model"] is not None:
        model.load_state_dict(best["state_model"], strict=True)
        for i, h in enumerate(exit_heads):
            h.load_state_dict(best["state_exits"][i], strict=True)

    return model, exit_heads, best


def cotrain_g3_multi_exit(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs: int = 30,

    # g3: train all layers (default)
    train_layer_indices: Optional[Sequence[int]] = None,   # None => all
    freeze_layer_indices: Sequence[int] = (),              # g3: usually empty

    exit_heads: List[nn.Module],
    payload_exit_cfg: List[dict],

    thrs: Sequence[float] = (1.0, 1.5),
    use_prob_margin: bool = False,

    lambda_final: float = 1.0,
    lambda_exits: Optional[Sequence[float]] = None,

    lr_backbone: float = 3e-4,
    lr_classifier: float = 3e-4,
    lr_exits: float = 3e-3,

    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,

    gate_temp: float = 1.0,
    use_gate_weighting: bool = True,

    best_metric: str = "val_overall_acc",
):
    """
    g3: update ALL backbone layers + classifier + all exit heads jointly.
    """
    model = model.to(device)

    assert len(exit_heads) == len(payload_exit_cfg)
    num_exits = len(exit_heads)
    assert len(thrs) == num_exits

    if lambda_exits is None:
        lambda_exits = [0.3] * num_exits
    else:
        assert len(lambda_exits) == num_exits

    # -----------------------
    # 0) Freeze all first
    # -----------------------
    set_requires_grad(model, False)

    # decide which layers to train
    if train_layer_indices is None:
        train_layer_indices = tuple(range(len(model.layers)))

    # freeze specified layers (usually none in g3)
    for li in freeze_layer_indices:
        if 0 <= li < len(model.layers):
            set_requires_grad(model.layers[li], False)

    # unfreeze train layers (g3 => all)
    for li in train_layer_indices:
        if li < 0 or li >= len(model.layers):
            raise ValueError(f"train_layer_indices contains out-of-range layer {li}")
        set_requires_grad(model.layers[li], True)

    # unfreeze final classifier
    set_requires_grad(model.classifier, True)

    # exit heads trainable
    exit_heads = [h.to(device) for h in exit_heads]
    for h in exit_heads:
        set_requires_grad(h, True)

    # -----------------------
    # 1) Build optimizer param groups
    # -----------------------
    params_backbone = []
    for li in train_layer_indices:
        params_backbone += [p for p in model.layers[li].parameters() if p.requires_grad]

    params_classifier = [p for p in model.classifier.parameters() if p.requires_grad]

    params_exits = []
    for h in exit_heads:
        params_exits += [p for p in h.parameters() if p.requires_grad]

    print(f"[g3] trainable params: backbone={sum(p.numel() for p in params_backbone)} "
          f"classifier={sum(p.numel() for p in params_classifier)} "
          f"exits={sum(p.numel() for p in params_exits)}")

    optimizer = torch.optim.AdamW(
        [
            {"params": params_backbone,   "lr": lr_backbone,   "weight_decay": weight_decay},
            {"params": params_classifier, "lr": lr_classifier, "weight_decay": weight_decay},
            {"params": params_exits,      "lr": lr_exits,      "weight_decay": weight_decay},
        ]
    )

    best = {"val_overall_acc": -1.0, "state": None}

    # -----------------------
    # 2) Train loop (loss 幾乎照搬 g2)
    # -----------------------
    eps = 1e-8
    for epoch in range(num_epochs):
        model.train()
        for h in exit_heads:
            h.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            final_logits, h_list = model.forward_with_all_hidden(xb)
            ce_final = F.cross_entropy(final_logits, yb, reduction="none")  # [B]

            if use_gate_weighting:
                ce_exit_list = []
                m_list = []
                for i in range(num_exits):
                    cfg = payload_exit_cfg[i]
                    layer_idx = int(cfg["layer_idx"])
                    h_i = h_list[layer_idx]
                    logits_i = _head_logits_from_hidden_trainable(exit_heads[i], h_i, device)  # [B,C]
                    ce_i = F.cross_entropy(logits_i, yb, reduction="none")  # [B]
                    ce_exit_list.append(ce_i)
                    m_i = _margin_from_logits(logits_i, use_prob=use_prob_margin)
                    m_list.append(m_i)

                # cascade-style soft weighting (你原本那段)
                u = torch.ones_like(ce_final)  # undecided prob
                loss_exit_sum = 0.0

                # 你原本 T 設定保留（也可改成 gate_temp shared）
                # NOTE: 這裡如果 exit 數不固定，記得提供對應長度
                T = [3.5, 1.25]  # <-- 你原本寫死的
                assert len(T) == num_exits, f"T must match num_exits, got {len(T)} vs {num_exits}"

                for i in range(num_exits):
                    thr_i = float(thrs[i])
                    Ti = float(T[i])
                    w_i = torch.sigmoid((m_list[i] - thr_i) / Ti)   # [B]
                    take_i = u * w_i                                # [B]
                    take_i_det = take_i.detach()                    # gate detach

                    loss_i = (take_i_det * ce_exit_list[i]).sum() / (take_i_det.sum() + eps)
                    loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * loss_i

                    # undecided update（你原本有試 detach / 不 detach；先沿用你現在版本）
                    u = u * (1.0 - w_i)

                u_det = u.detach()
                loss_final = (u_det * ce_final).sum() / (u_det.sum() + eps)

                loss = float(lambda_final) * loss_final + loss_exit_sum
            else:
                loss_final = ce_final.mean()
                loss_exit_sum = 0.0
                for i in range(num_exits):
                    cfg = payload_exit_cfg[i]
                    layer_idx = int(cfg["layer_idx"])
                    h_i = h_list[layer_idx]
                    logits_i = _head_logits_from_hidden_trainable(exit_heads[i], h_i, device)
                    loss_exit_i = F.cross_entropy(logits_i, yb)
                    loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * loss_exit_i
                loss = float(lambda_final) * loss_final + loss_exit_sum

            loss.backward()

            if grad_clip is not None:
                if params_backbone:
                    torch.nn.utils.clip_grad_norm_(params_backbone, grad_clip)
                if params_classifier:
                    torch.nn.utils.clip_grad_norm_(params_classifier, grad_clip)
                if params_exits:
                    torch.nn.utils.clip_grad_norm_(params_exits, grad_clip)

            optimizer.step()

        # -----------------------
        # 3) Eval (same as g2)
        # -----------------------
        out_val = eval_cascade_multi_exit(
            model, val_loader, device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=thrs,
            use_prob_margin=use_prob_margin,
            log_margins=False,
        )
        va_overall = out_val["overall_acc"]

        print(
            f"[G3] Ep{epoch:03d} | overall@{tuple(thrs)} va={va_overall*100:.2f} "
            f"| exit_rates={out_val['exit_rates']} final_rate={out_val['final_rate']:.4f}"
        )

        if va_overall > best["val_overall_acc"]:
            best["val_overall_acc"] = va_overall
            best["state"] = {
                "model": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                "exits": [copy.deepcopy({k: v.detach().cpu() for k, v in h.state_dict().items()}) for h in exit_heads],
            }

    # -----------------------
    # 4) Restore best
    # -----------------------
    if best["state"] is not None:
        model.load_state_dict(best["state"]["model"], strict=True)
        for i, h in enumerate(exit_heads):
            h.load_state_dict(best["state"]["exits"][i], strict=True)

    return model, exit_heads, best




# -----------------------------
# 4) End-to-end driver function
# -----------------------------
def run_cached_exit_pipeline(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    exit1_keep_idx,
    num_classes=10,
    cache_batches_train=None,
    cache_batches_val=None,
    cache_batches_test=None,
    normalize=True,
    num_epochs=50,
    lr=3e-3,
    weight_decay=1e-4,
    thr_list=(0.0, 0.5, 1.0, 2.0, 4.0),
):
    # 1) Cache features
    X_tr, y_tr, mu, sigma = cache_exit1_features(
        model, train_loader, device, exit1_keep_idx,
        max_batches=cache_batches_train, normalize=normalize
    )
    X_va, y_va, _, _ = cache_exit1_features(
        model, val_loader, device, exit1_keep_idx,
        max_batches=cache_batches_val, normalize=normalize
    )
    # For test, we only need mu/sigma from train if normalize=True
    X_te, y_te, _, _ = cache_exit1_features(
        model, test_loader, device, exit1_keep_idx,
        max_batches=cache_batches_test, normalize=normalize
    )

    print(f"[cache] train X={tuple(X_tr.shape)}, val X={tuple(X_va.shape)}, test X={tuple(X_te.shape)}")

    # 2) Train classifier on cached
    clf = train_exit_head_on_cached(
        X_tr, y_tr, X_va, y_va,
        num_classes=num_classes,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=1024,
        device=device,
    )

    # 3) Evaluate metrics on test
    res = eval_cached_exit_metrics(
        model, clf, test_loader, device,
        exit1_keep_idx=exit1_keep_idx,
        mu=mu if normalize else None,
        sigma=sigma if normalize else None,
        thr_list=thr_list
    )
    print("[cached-exit metrics]", res)
    return clf, res

@torch.no_grad()
def collect_hidden_activations(model, data_loader, device):
    model.eval()
    all_h = []
    all_y = []

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits, h_last = model(xb, return_hidden=True)  # the forwarding need to be able to support return_hidden
        all_h.append(h_last.cpu())
        all_y.append(yb.cpu())

    H = torch.cat(all_h, dim=0)
    Y = torch.cat(all_y, dim=0)
    return H, Y



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--backbone_ckpt", type=str, required=True)
    parser.add_argument("--path_out", type=str, required=True, help="Save ckpt with exit_config list")

    parser.add_argument("--exit_layers", type=str, default="0", help='e.g. "0" or "0,1"')
    parser.add_argument("--k", type=str, default="256", help='e.g. "256" or "256,512" (broadcast ok)')
    parser.add_argument("--keep_mode", type=str, default="p*(1-p)*std", help='broadcast ok')
    parser.add_argument("--exit_tau", type=str, default="1.0", help='broadcast ok')

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size_cached", type=int, default=512)
    parser.add_argument("--use_norm", action="store_true", default=True)
    parser.add_argument("--thr", type=str, default="1.0,1.5",
                    help="comma-separated thresholds per exit, e.g. 1.0,1.5")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders
    train_loader, val_loader, test_loader, in_bits, C, ds_meta = build_loaders_bits(
        dataset=args.dataset,
        root="/Users/yi-chunchen/workspace/WNN_early_exit/datasets/",
        batch_size_train=256,
        batch_size_eval=512,
        val_ratio=0.1,
        seed=3,
        z=32,
        device_for_encoding=device,
        shuffle_train=False,
    )

    backbone, bb_cfg, extra = build_backbone_from_ckpt(args.backbone_ckpt, device)
    backbone.eval()
    C = int(bb_cfg["num_classes"])

    exit_heads, exit_cfg_list = build_exits_from_ckpt(args.backbone_ckpt, device, num_classes=C)

    test_loss, test_acc = eval_epoch(backbone, test_loader, device)
    print("[final-only] test_acc", test_acc)
    # 之後直接用 backbone + exit_heads + exit_cfg_list 做 cascade eval / g1 training
    out = eval_cascade_multi_exit(
        backbone, test_loader, device,
        exit_heads=exit_heads,
        exit_cfg_list=[ec.to_payload() for ec in exit_cfg_list],  # 或你也可以把 eval 改成吃 ExitConfig 物件
        thrs=[1.0, 1.5],
        use_prob_margin=False,
    )
    print(out)


    # group 1 co-train
    # thrs 由系統輸入 "1.0,1.5"
    thr_list = [float(x) for x in args.thr.split(",")]
    assert len(thr_list) == 2

    payload_exit_cfg = [ec.to_payload() for ec in exit_cfg_list]

    '''# g2: train layer1+layer2 + classifier + exits
    backbone, exit_heads, best = cotrain_g3_multi_exit(
        model=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,

        # g3: train all, so either omit or:
        train_layer_indices=None,
        freeze_layer_indices=(),

        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
        thrs=thr_list,

        lambda_final=1.0,
        lambda_exits=(0.05, 0.10),

        lr_backbone=1e-4,      # 你可以先沿用 g2 的 lr
        lr_classifier=3e-4,
        lr_exits=5e-4,
        weight_decay=1e-3,
    )'''
    '''backbone, exit_heads, best = cotrain_g3_multi_exit_staged(
        model=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,

        train_layer_indices=None,
        freeze_layer_indices=(),

        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,

        thrs=thr_list,                 # 例如 (227.0, 18.0)

        lambda_final=1.0,
        lambda_exits=(0.05, 0.10),

        lr_backbone=1e-4,
        lr_classifier=3e-4,
        lr_exits=5e-4,
        weight_decay=1e-3,

        warmup_epochs=10,              # Phase A: 只訓練 final branch
        train_exit_ids=(1,),           # Phase B: 只解凍 exit1（exit0 保護住）
        best_metric="val_overall_acc", # 或 "val_final_only"
    )'''
    L = len(backbone.layers)
    half = L // 2

    stage_cfgs = [
        # A: head warmup (no cascade weighting)
        dict(name="A", epochs=5,
        train_layers=[],
        train_exit_ids=(1,),
        lr_backbone=0.0,
        lr_classifier=5e-4,   # 或 1e-3
        lr_exits=5e-4,
        disable_cascade_weighting=True),

        # B: adapt deeper rep for exit1/final
        dict(name="B", epochs=20,
            train_layers=list(range(half, L)),
            train_exit_ids=(1,),
            lr_backbone=1e-4,
            lr_classifier=5e-4,
            lr_exits=5e-4),

        # C: overall-ish, but controlled
        # option C-1 (safer): only deeper layers
        dict(name="C", epochs=10,
            train_layers=list(range(half, L)),
            train_exit_ids=(1,),
            lr_backbone=3e-5,
            lr_classifier=3e-4,
            lr_exits=3e-4),

        # optional D (true joint tiny): add exit0 tiny lr to compensate drift
        dict(name="D", epochs=3,
            train_layers=list(range(0, L)),
            train_exit_ids=(0,1),
            lr_backbone=1e-5,
            lr_classifier=1e-4,
            lr_exits=1e-4),
    ]
    backbone, exit_heads, best = cotrain_g3_multi_exit_staged(
        model=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
        thrs=thr_list,
            lambda_final=1.0,
        lambda_exits=(0.05, 0.10),
        weight_decay=1e-3,
        use_gate_weighting=True,
        gate_T=[3.5, 1.25],
        best_metric="val_overall_acc",
        stage_cfgs=stage_cfgs,
    )
    #print("Best val overall acc:", best["val_overall_acc"])

    # 最後存成一個 ckpt：backbone_cfg 不動 + backbone weights + exit_cfg_list
    payload_exit_cfg = [ec.to_payload() for ec in exit_cfg_list]


    save_ckpt_v2(
        args.path_out,
        backbone,                 # backbone model
        exit_heads,
        bb_cfg,          # backbone cfg 不動
        exit_cfg_list=payload_exit_cfg,  # <-- exit cfg list
        extra={"dataset": args.dataset}
    )

    print("\nSaved:", args.path_out)
    print("Exit cfg list length:", len(payload_exit_cfg))
    
    test_loss, test_acc = eval_epoch(backbone, test_loader, device)
    print("[final-only] test_acc", test_acc)

    thrs = [0.0, 0.5, 1.0, 2.0, 4.0]
    for thr in thrs:
        out = eval_overall_at_thr_multi_exit(
            backbone, test_loader, device,
            thr=thr,
            exit_id=0,
            exit_cfg_list=payload_exit_cfg,   # <-- 用 ExitConfig list
            exit_heads=exit_heads,
            use_prob_margin=False,
        )
        print(thr, out["exit_rate"], out["overall_acc"], out["exited_acc"], out["non_exited_acc"],
              out["margin_mean"], out["margin_p95"])
        print_eval_profile(f"G3-v2 exit0@thr={thr}", out)
    
    print('=======================================')
    #thrs0 = [0.0, 0.25, 0.5, 0.75, 1.0]
    #thrs1 = [1.2, 1.5, 1.8, 2.0]
    thrs0 = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    thrs1 = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]

    for thr0 in thrs0:
        for thr1 in thrs1:
            out = eval_cascade_multi_exit(
                    backbone, test_loader, device,
                    exit_heads=exit_heads,
                    exit_cfg_list=payload_exit_cfg,   
                    thrs=[thr0, thr1],
                    use_prob_margin=False,
                )
            s = sum(out["exit_rates"]) + out["final_rate"]
            assert abs(s - 1.0) < 1e-6, s
            '''print(thr0, thr1,
                out["overall_acc"], out["exit_rates"], out["final_rate"],
                out["exit_accs"],
                out["final_acc"])'''

            r0, r1 = out["exit_rates"]
            rF = out["final_rate"]

            exp_layers = 1*r0 + 2*r1 + 3*rF
            compute_ratio = exp_layers / 3.0

            print(
                f"{thr0:>4} {thr1:>4} | "
                f"overall={out['overall_acc']:.4f} | "
                f"r0={out['exit_rates'][0]:.4f} a0={out['exit_accs'][0]:.4f} | "
                f"r1={out['exit_rates'][1]:.4f} a1={out['exit_accs'][1]:.4f} | "
                f"rf={out['final_rate']:.4f} af={out['final_acc']:.4f}"
            )
            m0 = out["margin_stats"][0]
            m1 = out["margin_stats"][1]
            print(f" | m0f={m0['mean']:.2f}/{m0['p95']:.2f} m1={m1['mean']:.2f}/{m1['p95']:.2f}")
            m0_detail = out['margin_stats'][2]
            m1_detail = out['margin_stats'][3]
            print(f" | m0_undecided={m0_detail['undecided_mean']:.2f} m0_undecided_p95={m0_detail['undecided_p95']:.2f} m0_taken_mean={m0_detail['taken_mean']:.2f} m0_taken_p95={m0_detail['taken_p95']:.2f}")
            print(f" | m1_undecided={m1_detail['undecided_mean']:.2f} m1_undecided_p95={m1_detail['undecided_p95']:.2f} m1_taken_mean={m1_detail['taken_mean']:.2f} m1_taken_p95={m1_detail['taken_p95']:.2f}")
            print_eval_profile(f"G3-v2 cascade@({thr0},{thr1})", out)


    print('=======================================')
    best, dbg = sweep_cascade_by_quantile(
        model=backbone,
        val_loader=val_loader,
        device=device,
        exit_heads=exit_heads,
        exit_cfg_list=payload_exit_cfg
    )

    @torch.no_grad()
    def debug_logit_scales(model, loader, device, exit_heads, payload_exit_cfg, use_prob_margin=False, n_batches=3):
        model.eval()
        for h in exit_heads:
            h.eval()

        for bi, (xb, yb) in enumerate(loader):
            if bi >= n_batches:
                break
            xb, yb = xb.to(device), yb.to(device)

            final_logits, h_list = model.forward_with_all_hidden(xb)

            # ---- final logits stats ----
            print("[final] abs mean/max:",
                final_logits.abs().mean().item(),
                final_logits.abs().max().item())

            # ---- each exit logits stats ----
            for i, head in enumerate(exit_heads):
                layer_idx = int(payload_exit_cfg[i]["layer_idx"])
                h_i = h_list[layer_idx]

                # head logits (你原本就有這個 helper；沒有就 head(h_i))
                exit_logits = _head_logits_from_hidden(head, h_i, device)  # [B,C]

                # margin on logits (same as your cascade gate)
                top2 = torch.topk(exit_logits, k=2, dim=-1).values
                margin = top2[:, 0] - top2[:, 1]

                print(f"[exit{i}] abs mean/max:",
                    exit_logits.abs().mean().item(),
                    exit_logits.abs().max().item(),
                    "| margin mean/p95:",
                    margin.mean().item(),
                    torch.quantile(margin, 0.95).item())

    debug_logit_scales(
        model=backbone,
        loader=test_loader,
        device=device,
        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
        use_prob_margin=False,
        n_batches=5,
    )


    
