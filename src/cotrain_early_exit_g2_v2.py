# src/train/train_wnn.py
import argparse
import copy
import math
from pathlib import Path
import json
from networkx import sigma
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn.functional as F

from dataclasses import dataclass

from torchgen import model
from src.core.linearExitHead import build_exits_from_ckpt
from src.cotrain_early_exit_g1_v2 import eval_exit_only
from src.dataio.data import build_loaders_bits
from src.dataio.mapping import make_tuple_mapping, audit_mapping
from src.early_exit import _calibrate_thr_from_margins, _head_logits_from_hidden_trainable, _margin_from_logits
from src.tools.utils import _head_logits_from_hidden
from src.exit.analyze_hidden import compute_mu_sigma
from src.prune import *
from src.early_exit import *
from src.tools.utils import print_sweep_table  
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


@torch.no_grad()
def _calibrate_thrs_from_val(
    model: nn.Module,
    val_loader,
    device,
    *,
    exit_heads: List[nn.Module],
    payload_exit_cfg: List[dict],
    use_prob_margin: bool,
    r0_target: float,
    r1_target: float,
    max_batches: int = 2,
) -> Tuple[float, float]:
    """
    用 val 的少量 batch 自動選 thr0/thr1，使得：
      r0 ≈ r0_target
      r1 ≈ r1_target（在 undecided-after-exit0 的子集上）
    注意：這裡假設 gate 規則是 margin > thr => exit
    """
    model.eval()
    for h in exit_heads:
        h.eval()

    # 只支援 2 exits（你現在 FMNIST multi-exit 就是 2 個）
    assert len(exit_heads) == 2
    assert len(payload_exit_cfg) == 2

    m0_all = []
    m1_all = []

    nb = 0
    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        final_logits, h_list = model.forward_with_all_hidden(xb)

        # exit0
        cfg0 = payload_exit_cfg[0]
        li0 = int(cfg0["layer_idx"])
        logits0 = _head_logits_from_hidden_trainable(exit_heads[0], h_list[li0], device)
        m0 = _margin_from_logits(logits0, use_prob=use_prob_margin)  # [B]
        m0_all.append(m0.detach().cpu())

        nb += 1
        if nb >= max_batches:
            break

    m0_cat = torch.cat(m0_all, dim=0)  # [N]
    # r0_target = P(m0 > thr0)  => thr0 = quantile(m0, 1 - r0_target)
    thr0 = torch.quantile(m0_cat, q=float(1.0 - r0_target)).item()

    # 再跑一次收集 exit1 margin，但只在 “没被 exit0 拿走” 的子集上
    nb = 0
    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        _, h_list = model.forward_with_all_hidden(xb)

        cfg0 = payload_exit_cfg[0]
        li0 = int(cfg0["layer_idx"])
        logits0 = _head_logits_from_hidden_trainable(exit_heads[0], h_list[li0], device)
        m0 = _margin_from_logits(logits0, use_prob=use_prob_margin)
        undecided = (m0 <= thr0)  # [B]

        cfg1 = payload_exit_cfg[1]
        li1 = int(cfg1["layer_idx"])
        logits1 = _head_logits_from_hidden_trainable(exit_heads[1], h_list[li1], device)
        m1 = _margin_from_logits(logits1, use_prob=use_prob_margin)  # [B]

        if undecided.any():
            m1_all.append(m1[undecided].detach().cpu())

        nb += 1
        if nb >= max_batches:
            break

    if len(m1_all) == 0:
        # 如果 exit0 太強導致 undecided 幾乎沒樣本，就把 thr0 調高一點（更嚴格）保證 tail
        # 這裡先保守 fallback
        thr1 = float("inf")
    else:
        m1_cat = torch.cat(m1_all, dim=0)
        thr1 = torch.quantile(m1_cat, q=float(1.0 - r1_target)).item()

    return thr0, thr1

@torch.no_grad()
def _eval_final_only(model, loader, device, max_batches: Optional[int] = None):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for bi, (xb, yb) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        loss_sum += loss.item() * yb.numel()
    return (loss_sum / max(1, total)), (correct / max(1, total))


@torch.no_grad()
def _eval_exit_only(model, exit_head, payload_exit_cfg_i, loader, device, use_prob_margin: bool,
                    max_batches: Optional[int] = None):
    model.eval()
    exit_head.eval()
    layer_idx = int(payload_exit_cfg_i["layer_idx"])
    total, correct, loss_sum = 0, 0, 0.0
    m_all = []
    for bi, (xb, yb) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        xb, yb = xb.to(device), yb.to(device)
        _, h_list = model.forward_with_all_hidden(xb)  # assumes available
        logits = _head_logits_from_hidden_trainable(exit_head, h_list[layer_idx], device)
        loss = F.cross_entropy(logits, yb)
        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        loss_sum += loss.item() * yb.numel()

        m = _margin_from_logits(logits, use_prob=use_prob_margin)
        m_all.append(m.detach().float().cpu())

    m_cat = torch.cat(m_all, dim=0) if len(m_all) else torch.zeros(0)
    m_mean = float(m_cat.mean().item()) if m_cat.numel() else float("nan")
    m_p95 = float(torch.quantile(m_cat, 0.95).item()) if m_cat.numel() else float("nan")
    return (loss_sum / max(1, total)), (correct / max(1, total)), {"margin_mean": m_mean, "margin_p95": m_p95}



import copy
from typing import List, Sequence, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


@torch.no_grad()
def _logits_margin(logits: torch.Tensor) -> torch.Tensor:
    # logits: [B, C]
    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]  # [B]


def _head_logits_from_hidden(head, h, device):
    """
    h: [B, D_layer] on device
    return logits: [B, C] on device
    """
    x = h[:, head.exit_keep_idx.to(device)]
    if getattr(head, "use_norm", False):
        mu = head.mu.to(device)
        sigma = head.sigma.to(device)
        x = (x - mu) / sigma
    return head.classifier(x) / float(getattr(head, "exit_tau", 1.0))


import copy
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 你已有：set_requires_grad, eval_cascade_multi_exit, _head_logits_from_hidden, _logits_margin

import torch
import torch.nn.functional as F

@torch.no_grad()
def eval_final_only_on_tail(
    model,
    val_loader,
    device,
    *,
    exit_heads,
    payload_exit_cfg,
    thrs_tail=(227.0, 18.0),
    use_prob_margin=False,   # 你目前不用 prob margin 就留 False
):
    """
    Compute final classifier accuracy on tail samples defined by cascade gating
    using thrs_tail, i.e., samples NOT taken by any early exit.
    """
    model.eval()
    for h in exit_heads:
        h.eval()

    num_exits = len(exit_heads)
    assert len(thrs_tail) == num_exits

    correct = 0
    total = 0

    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        final_logits, h_list = model.forward_with_all_hidden(xb)

        # tail mask: survive all exits
        tail = torch.ones_like(yb, dtype=torch.bool)

        for i in range(num_exits):
            cfg = payload_exit_cfg[i]
            li = int(cfg["layer_idx"])
            thr_i = float(thrs_tail[i])

            logits_i = _head_logits_from_hidden(exit_heads[i], h_list[li], device)
            margin_i = _margin_from_logits(logits_i, use_prob=use_prob_margin)
            take_i = tail & (margin_i > thr_i)
            tail = tail & (~take_i)

        if tail.any():
            pred = final_logits[tail].argmax(dim=1)
            correct += (pred == yb[tail]).sum().item()
            total += int(tail.sum().item())

    acc = (correct / total) if total > 0 else float("nan")
    return {"final_tail_acc": float(acc), "tail_count": int(total)}

def cotrain_g2_multi_exit_v4(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs: int = 50,

    # g2: train deeper backbone + classifier
    train_layer_indices: Sequence[int] = (1, 2),
    freeze_layer_indices: Sequence[int] = (0,),

    # exits
    exit_heads: List[nn.Module],
    payload_exit_cfg: List[dict],  # each has {"layer_idx": ...}

    # ---- NEW: split thresholds ----
    thrs_train: Sequence[float] = (227.0, 18.0),  # always use stable gate distribution in training
    thrs_eval_list: Optional[Sequence[Sequence[float]]] = None,  # e.g. [(227,18),(227,10)]
    best_eval_idx: int = 0,  # choose which eval threshold set decides "best"

    # which exits to TRAIN (default: only exit1)
    train_exit_ids: Sequence[int] = (1,),

    # optionally compute exit_i loss only on tail samples
    # (if True: exit i loss uses samples not taken by previous exits under thrs_train)
    exit_only_on_tail: bool = False,

    # loss weights
    lambda_final: float = 1.0,
    lambda_exits: Optional[Sequence[float]] = None,  # length=num_exits; only used for trainable exits
    beta_tail: float = 0.0,  # optional: upweight final CE on tail samples under thrs_train

    # optim
    lr_backbone: float = 1e-4,
    lr_classifier: float = 1e-3,
    lr_exits: float = 5e-4,
    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,

    # warmup: first N epochs only train final branch (no exit losses)
    warmup_epochs: int = 0,

    # best metric
    best_metric: str = "val_overall_acc",  # "val_overall_acc" or "val_final_only"
):
    """
    G2-v4 (gate-decoupled):
      - training gate uses thrs_train (stable)
      - eval can run multiple threshold sets thrs_eval_list, and you pick which one selects best

    Notes:
      - Frozen exits are excluded from exit loss entirely (no gradient path).
      - Even if an exit head is frozen, backbone can still learn from final loss / other exits.
    """
    model = model.to(device)
    exit_heads = [h.to(device) for h in exit_heads]

    num_exits = len(exit_heads)
    assert len(payload_exit_cfg) == num_exits
    assert len(thrs_train) == num_exits, f"len(thrs_train) must == num_exits ({num_exits})"

    if thrs_eval_list is None:
        thrs_eval_list = [tuple(thrs_train)]
    else:
        thrs_eval_list = [tuple(x) for x in thrs_eval_list]
        for x in thrs_eval_list:
            assert len(x) == num_exits, "each thrs_eval must have length=num_exits"
    assert 0 <= best_eval_idx < len(thrs_eval_list)

    # lambda_exits default
    if lambda_exits is None:
        lambda_exits = [0.1] * num_exits
    else:
        assert len(lambda_exits) == num_exits

    # -----------------------
    # 0) Freeze all
    # -----------------------
    set_requires_grad(model, False)
    for h in exit_heads:
        set_requires_grad(h, False)

    # freeze specified layers
    for li in freeze_layer_indices:
        if 0 <= li < len(model.layers):
            set_requires_grad(model.layers[li], False)

    # unfreeze specified train layers
    for li in train_layer_indices:
        if li < 0 or li >= len(model.layers):
            raise ValueError(f"train_layer_indices contains out-of-range layer {li}")
        set_requires_grad(model.layers[li], True)

    # unfreeze classifier
    if not hasattr(model, "classifier"):
        raise ValueError("model has no classifier; g2 expects final classifier exists.")
    set_requires_grad(model.classifier, True)

    # unfreeze only selected exits
    train_exit_ids = tuple(train_exit_ids)
    for i in train_exit_ids:
        if i < 0 or i >= num_exits:
            raise ValueError(f"train_exit_ids contains out-of-range exit {i}")
        set_requires_grad(exit_heads[i], True)

    # -----------------------
    # 1) Optimizer
    # -----------------------
    params_backbone = []
    for li in train_layer_indices:
        params_backbone += [p for p in model.layers[li].parameters() if p.requires_grad]

    params_classifier = [p for p in model.classifier.parameters() if p.requires_grad]

    params_exits = []
    for i in train_exit_ids:
        params_exits += [p for p in exit_heads[i].parameters() if p.requires_grad]

    print(
        f"[G2-v4] trainable params: backbone={sum(p.numel() for p in params_backbone)} "
        f"classifier={sum(p.numel() for p in params_classifier)} "
        f"exits(train_ids={train_exit_ids})={sum(p.numel() for p in params_exits)}"
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": params_backbone, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": params_classifier, "lr": lr_classifier, "weight_decay": weight_decay},
            {"params": params_exits, "lr": lr_exits, "weight_decay": weight_decay},
        ]
    )

    best = {
        "metric": -1.0,
        "epoch": -1,
        "state_model": None,
        "state_exits": None,
        "best_thrs_eval": thrs_eval_list[best_eval_idx],
    }

    eps = 1e-8

    # -----------------------
    # helper: compute tail mask under thrs_train
    # -----------------------
    def _compute_tail_mask(h_list, yb):
        alive = torch.ones_like(yb, dtype=torch.bool)
        for j in range(num_exits):
            cfgj = payload_exit_cfg[j]
            lj = int(cfgj["layer_idx"])
            thrj = float(thrs_train[j])

            logits_j = _head_logits_from_hidden(exit_heads[j], h_list[lj], device)
            margin_j = _logits_margin(logits_j)
            take_j = alive & (margin_j > thrj)
            alive = alive & (~take_j)
        return alive

    # -----------------------
    # 2) Train loop
    # -----------------------
    for epoch in range(num_epochs):
        model.train()
        for h in exit_heads:
            h.train()

        warmup = epoch < warmup_epochs

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            final_logits, h_list = model.forward_with_all_hidden(xb)

            # ---- final loss ----
            ce_final = F.cross_entropy(final_logits, yb, reduction="none")  # [B]
            ce_all = ce_final.mean()

            if beta_tail > 0.0:
                with torch.no_grad():
                    tail_mask = _compute_tail_mask(h_list, yb)
                if tail_mask.any():
                    ce_tail = F.cross_entropy(final_logits[tail_mask], yb[tail_mask])
                else:
                    ce_tail = 0.0 * ce_all
                loss_final = ce_all + float(beta_tail) * ce_tail
            else:
                loss_final = ce_all

            # ---- exit losses (ONLY for trainable exits) ----
            loss_exit_sum = 0.0
            if not warmup:
                for i in range(num_exits):
                    if i not in train_exit_ids:
                        continue  # keep frozen exits totally out of gradient path

                    cfg = payload_exit_cfg[i]
                    li = int(cfg["layer_idx"])
                    logits_i = _head_logits_from_hidden(exit_heads[i], h_list[li], device)

                    if exit_only_on_tail and i > 0:
                        # train exit i only on samples that survive previous exits under thrs_train
                        with torch.no_grad():
                            alive = torch.ones_like(yb, dtype=torch.bool)
                            for j in range(i):
                                cfgj = payload_exit_cfg[j]
                                lj = int(cfgj["layer_idx"])
                                thrj = float(thrs_train[j])
                                logits_j = _head_logits_from_hidden(exit_heads[j], h_list[lj], device)
                                margin_j = _logits_margin(logits_j)
                                take_j = alive & (margin_j > thrj)
                                alive = alive & (~take_j)

                        if alive.any():
                            loss_i = F.cross_entropy(logits_i[alive], yb[alive])
                        else:
                            loss_i = 0.0 * loss_final
                    else:
                        loss_i = F.cross_entropy(logits_i, yb)

                    loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * loss_i

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
        # 3) Eval: run multiple thrs_eval
        # -----------------------
        model.eval()
        for h in exit_heads:
            h.eval()

        eval_records = []
        for k, thrs_eval in enumerate(thrs_eval_list):
            out_val = eval_cascade_multi_exit(
                model, val_loader, device,
                exit_heads=exit_heads,
                exit_cfg_list=payload_exit_cfg,
                thrs=thrs_eval,
                use_prob_margin=False,
                log_margins=False,
            )
            tail_stats = eval_final_only_on_tail(
                model, val_loader, device,
                exit_heads=exit_heads,
                payload_exit_cfg=payload_exit_cfg,
                thrs_tail=(227.0, 18.0),
                #use_prob_margin=use_prob_margin,
            )
            final_tail_only = float(tail_stats["final_tail_acc"])
            tail_n = int(tail_stats["tail_count"])

            overall = float(out_val["overall_acc"])
            final_only = float(out_val.get("final_acc", 0.0))  # 你 eval 若有 final_acc
            eval_records.append((overall, final_only, out_val, final_tail_only))

            print(
                f"[G2-v4] Ep{epoch:03d} | warmup={warmup} | eval#{k} thrs={tuple(thrs_eval)} "
                f"| overall={overall*100:.2f} | final_rate={out_val['final_rate']:.4f} "
                f'| final_only={final_only*100:.2f} '
                f"| exit_rates={out_val['exit_rates']}"
            )
            print(
                f"[G2-v4] Ep{epoch:03d} | tail@{(227.0,18.0)} "
                f"final_tail_only={final_tail_only*100:.2f} | tail_n={tail_n}"
            )

        # choose best by best_eval_idx
        overall_sel, final_only_sel, out_sel, final_tail_only_sel = eval_records[best_eval_idx]
        if best_metric == "val_overall_acc":
            metric = overall_sel
        elif best_metric == "val_final_only":
            metric = final_only_sel
        elif best_metric == "val_final_tail_only":
            metric = final_tail_only_sel
        else:
            raise ValueError("best_metric must be 'val_overall_acc' or 'val_final_only'")

        if metric > best["metric"]:
            best["metric"] = metric
            best["epoch"] = epoch
            best["state_model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["state_exits"] = [{k: v.detach().cpu().clone() for k, v in h.state_dict().items()} for h in exit_heads]

    # -----------------------
    # 4) Restore best
    # -----------------------
    if best["state_model"] is not None:
        model.load_state_dict(best["state_model"], strict=True)
        for i, h in enumerate(exit_heads):
            h.load_state_dict(best["state_exits"][i], strict=True)

    return model, exit_heads, best


def cotrain_g2_multi_exit_v4_temp(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs: int = 50,

    # g2: train deeper backbone + classifier
    train_layer_indices: Sequence[int] = (1, 2),
    freeze_layer_indices: Sequence[int] = (0,),

    # exits
    exit_heads: List[nn.Module],
    payload_exit_cfg: List[dict],     # each has {"layer_idx": ...}
    thrs: Sequence[float] = (227.0, 10.0),

    # which exits to TRAIN (default: only exit1)
    train_exit_ids: Sequence[int] = (1,),
    # optionally compute exit1 loss only on tail samples
    exit1_only_on_tail: bool = False,

    # loss weights
    lambda_final: float = 1.0,
    lambda_exits: Optional[Sequence[float]] = None,  # length=num_exits; used only for trainable exits
    beta_tail: float = 1.0,

    # optim
    lr_backbone: float = 1e-4,
    lr_classifier: float = 1e-3,
    lr_exits: float = 5e-4,
    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,

    # warmup (optional): first N epochs only train final branch (no exit losses)
    warmup_epochs: int = 0,

    # eval metric
    best_metric: str = "val_overall_acc",  # or "val_final_only"
):
    """
    g2: freeze early exit (exit0) + earlier backbone (layer0),
        train deeper backbone (layer1,layer2,...) + final classifier,
        optionally train deeper exit head (exit1) without touching exit0.

    returns: (model, exit_heads, best)
    """
    model = model.to(device)
    exit_heads = [h.to(device) for h in exit_heads]

    num_exits = len(exit_heads)
    assert len(payload_exit_cfg) == num_exits
    assert len(thrs) == num_exits

    if lambda_exits is None:
        lambda_exits = [0.0] * num_exits
        for i in range(num_exits):
            lambda_exits[i] = 0.1  # default
    else:
        assert len(lambda_exits) == num_exits

    train_exit_ids = set(train_exit_ids)

    # -----------------------
    # 0) Freeze/unfreeze
    # -----------------------
    set_requires_grad(model, False)
    for h in exit_heads:
        set_requires_grad(h, False)

    # freeze early layers explicitly
    for li in freeze_layer_indices:
        if 0 <= li < len(model.layers):
            set_requires_grad(model.layers[li], False)

    # train deeper layers
    for li in train_layer_indices:
        if li < 0 or li >= len(model.layers):
            raise ValueError(f"train_layer_indices out of range: {li}")
        set_requires_grad(model.layers[li], True)

    # train final classifier
    if not hasattr(model, "classifier"):
        raise ValueError("model must have .classifier for g2")
    set_requires_grad(model.classifier, True)

    # train selected exits (default: only exit1)
    for i in range(num_exits):
        if i in train_exit_ids:
            set_requires_grad(exit_heads[i], True)
        else:
            set_requires_grad(exit_heads[i], False)

    # -----------------------
    # 1) Optimizer groups
    # -----------------------
    params_backbone = []
    for li in train_layer_indices:
        params_backbone += [p for p in model.layers[li].parameters() if p.requires_grad]

    params_classifier = [p for p in model.classifier.parameters() if p.requires_grad]

    params_exits = []
    for i in range(num_exits):
        if i in train_exit_ids:
            params_exits += [p for p in exit_heads[i].parameters() if p.requires_grad]

    print(f"[g2-v4] trainable params: "
          f"backbone={sum(p.numel() for p in params_backbone)} "
          f"classifier={sum(p.numel() for p in params_classifier)} "
          f"exits(train_ids={sorted(list(train_exit_ids))})={sum(p.numel() for p in params_exits)}")

    optimizer = torch.optim.AdamW(
        [
            {"params": params_backbone,   "lr": lr_backbone,   "weight_decay": weight_decay},
            {"params": params_classifier, "lr": lr_classifier, "weight_decay": weight_decay},
            {"params": params_exits,      "lr": lr_exits,      "weight_decay": weight_decay},
        ]
    )

    best = {"metric": -1.0, "state": None}

    # -----------------------
    # 2) Train loop
    # -----------------------
    for epoch in range(num_epochs):
        model.train()
        for h in exit_heads:
            h.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            final_logits, h_list = model.forward_with_all_hidden(xb)

            # ---------- final loss ----------
            ce_final = F.cross_entropy(final_logits, yb, reduction="none")  # [B]
            ce_all = ce_final.mean()

            if epoch < warmup_epochs:
                # ✅ warmup: final 看全分佈，先把 decision boundary 釘住
                loss_final = ce_all
            else:
                # ✅ after warmup: 才開始 tail penalty（用 thrs/exit heads 找 tail）
                with torch.no_grad():
                    tail_mask = torch.ones_like(yb, dtype=torch.bool)
                    for j in range(num_exits):   # 或只用 exit0/exit1 你想的版本
                        cfgj = payload_exit_cfg[j]
                        lj = int(cfgj["layer_idx"])
                        thrj = float(thrs[j])
                        logits_j = _head_logits_from_hidden(exit_heads[j], h_list[lj], device)
                        margin_j = _logits_margin(logits_j)
                        take_j = tail_mask & (margin_j > thrj)
                        tail_mask = tail_mask & (~take_j)

                if tail_mask.any():
                    ce_tail = F.cross_entropy(final_logits[tail_mask], yb[tail_mask])
                else:
                    ce_tail = 0.0 * ce_all

                loss_final = ce_all + beta_tail * ce_tail


            # ---------- exit losses (ONLY for trainable exits) ----------
            loss_exit_sum = 0.0
            if epoch >= warmup_epochs:
                for i in range(num_exits):
                    if i not in train_exit_ids:
                        continue

                    cfg = payload_exit_cfg[i]
                    li = int(cfg["layer_idx"])
                    logits_i = _head_logits_from_hidden(exit_heads[i], h_list[li], device)

                    if exit1_only_on_tail:
                        # alive recompute (你的版本 OK)
                        with torch.no_grad():
                            alive = torch.ones_like(yb, dtype=torch.bool)
                            for j in range(i):
                                cfgj = payload_exit_cfg[j]
                                lj = int(cfgj["layer_idx"])
                                thrj = float(thrs[j])
                                logits_j = _head_logits_from_hidden(exit_heads[j], h_list[lj], device)
                                margin_j = _logits_margin(logits_j)
                                take_j = alive & (margin_j > thrj)
                                alive = alive & (~take_j)
                        if alive.any():
                            loss_i = F.cross_entropy(logits_i[alive], yb[alive])
                        else:
                            loss_i = 0.0 * loss_final
                    else:
                        loss_i = F.cross_entropy(logits_i, yb)

                    loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * loss_i

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
        # 3) Eval (you already have these helpers; call your own)
        # -----------------------
        model.eval()
        for h in exit_heads:
            h.eval()

        # You likely already have:
        # - eval_cascade_multi_exit(model, val_loader, device, exit_heads, exit_cfg_list, thrs, use_prob_margin=False)
        # - eval_final_only(model, val_loader, device)
        # - eval_exit_only(model, exit_head_i, val_loader, device, layer_idx)

        out = eval_cascade_multi_exit(
            model, val_loader, device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=thrs,
            use_prob_margin=False,
            log_margins=False,
        )
        va_overall = float(out["overall_acc"])

        va_final_loss, va_final_acc = eval_final_only(model, val_loader, device)

        # exit-only accs (helpful debug)
        ex_accs = []
        for i in range(num_exits):
            li = int(payload_exit_cfg[i]["layer_idx"])
            _, ex_acc = eval_exit_only(model, exit_heads[i], val_loader, device, li)
            ex_accs.append(ex_acc)

        print(f"[G2-v4] Ep{epoch:03d} | warmup={epoch < warmup_epochs} "
              f"| thr0={thrs[0]:.3f} thr1={thrs[1]:.3f} "
              f"| overall={va_overall*100:.2f} final_only={va_final_acc*100:.2f} "
              f"| ex0_only={ex_accs[0]*100:.2f} ex1_only={ex_accs[1]*100:.2f} "
              f"| rates={out['exit_rates']} final_rate={out['final_rate']:.4f}")

        if best_metric == "val_final_only":
            metric = float(va_final_acc)
        else:
            metric = float(va_overall)

        if metric > best["metric"]:
            best["metric"] = metric
            best["state"] = {
                "model": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                "exits": [copy.deepcopy({k: v.detach().cpu() for k, v in h.state_dict().items()}) for h in exit_heads],
            }

    # restore best
    if best["state"] is not None:
        model.load_state_dict(best["state"]["model"], strict=True)
        for i, h in enumerate(exit_heads):
            h.load_state_dict(best["state"]["exits"][i], strict=True)

    return model, exit_heads, best

def cotrain_g2_multi_exit(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs: int = 30,
    train_layer_indices=(1, 2),
    freeze_layer_indices=(0,),
    exit_heads=None,
    payload_exit_cfg=None,
    thrs=(1.0, 1.5),
    use_prob_margin: bool = False,
    lambda_final: float = 1.0,
    lambda_exits=None,
    lr_backbone: float = 3e-4,
    lr_classifier: float = 3e-4,
    lr_exits: float = 3e-3,
    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,
    use_gate_weighting: bool = True,
    gate_temp: float = 1.0,
    beta_tail: float = 0.1,

    # ✅ NEW: freeze policy for exits
    freeze_exit0: bool = True,     # G2: earlier exit fixed
    train_exit1: bool = True,      # G2: deeper exit co-train
):
    model = model.to(device)
    assert exit_heads is not None
    assert payload_exit_cfg is not None
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

    # freeze specified layers
    for li in freeze_layer_indices:
        if 0 <= li < len(model.layers):
            set_requires_grad(model.layers[li], False)

    # unfreeze train layers
    for li in train_layer_indices:
        if li < 0 or li >= len(model.layers):
            raise ValueError(f"train_layer_indices contains out-of-range layer {li}")
        set_requires_grad(model.layers[li], True)

    # unfreeze final classifier
    if not hasattr(model, "classifier"):
        raise ValueError("model has no classifier; g2 expects final classifier exists.")
    set_requires_grad(model.classifier, True)

    # -----------------------
    # 0.5) Exit heads freeze policy (IMPORTANT)
    # -----------------------
    exit_heads = [h.to(device) for h in exit_heads]

    # default: freeze all
    for h in exit_heads:
        set_requires_grad(h, False)

    # exit0
    if not freeze_exit0:
        set_requires_grad(exit_heads[0], True)

    # exit1 (if exists)
    if num_exits >= 2 and train_exit1:
        set_requires_grad(exit_heads[1], True)

    # if you have >2 exits, extend here as needed

    # -----------------------
    # 1) Build optimizer param groups (NO accidental params)
    # -----------------------
    params_backbone = []
    for li in train_layer_indices:
        params_backbone += [p for p in model.layers[li].parameters() if p.requires_grad]

    params_classifier = [p for p in model.classifier.parameters() if p.requires_grad]

    params_exits = []
    for h in exit_heads:
        params_exits += [p for p in h.parameters() if p.requires_grad]  # only those you unfreezed

    print(
        f"[g2] trainable params: backbone={sum(p.numel() for p in params_backbone)} "
        f"classifier={sum(p.numel() for p in params_classifier)} "
        f"exits={sum(p.numel() for p in params_exits)} | "
        f"freeze_exit0={freeze_exit0} train_exit1={train_exit1}"
    )

    groups = [
        {"params": params_backbone,   "lr": lr_backbone,   "weight_decay": weight_decay},
        {"params": params_classifier, "lr": lr_classifier, "weight_decay": weight_decay},
    ]
    if len(params_exits) > 0:
        groups.append({"params": params_exits, "lr": lr_exits, "weight_decay": weight_decay})

    optimizer = torch.optim.AdamW(groups)

    # -----------------------
    # 2) Train loop
    # -----------------------
    best = {"val_overall_acc": -1.0, "state": None}

    for epoch in range(num_epochs):
        model.train()
        for h in exit_heads:
            h.train()  # even if frozen, train/eval mode only affects dropout/bn; should be harmless

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            final_logits, h_list = model.forward_with_all_hidden(xb)

            # ---- your loss logic here ----
            # e.g. final CE + optional tail penalty + exit losses (only for trainable exits)
            # IMPORTANT: when you compute loss_exit_sum, you can still compute for all exits,
            # but only the ones with requires_grad=True will backprop into their head parameters.
            # (it will still backprop into backbone through h_list if backbone layers are trainable)

            # Example: final loss (your current style)
            ce_final = F.cross_entropy(final_logits, yb, reduction="none")
            ce_all = ce_final.mean()
            loss_final = ce_all  # + beta_tail * tail_ce (keep your existing tail code)

            # exit losses (keep your existing gate-weighting code; unchanged)
            # ... compute loss_exit_sum ...
            loss_exit_sum = 0.0
            # (placeholder) use your original code to compute loss_exit_sum
            # loss = lambda_final * loss_final + loss_exit_sum

            loss = lambda_final * loss_final + loss_exit_sum
            loss.backward()

            if grad_clip is not None:
                if params_backbone:
                    torch.nn.utils.clip_grad_norm_(params_backbone, grad_clip)
                if params_classifier:
                    torch.nn.utils.clip_grad_norm_(params_classifier, grad_clip)
                if params_exits:
                    torch.nn.utils.clip_grad_norm_(params_exits, grad_clip)

            optimizer.step()

        # ---- eval ----
        out_val = eval_cascade_multi_exit(
            model, val_loader, device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=thrs,
            use_prob_margin=use_prob_margin,
            log_margins=False,
        )
        va_overall = float(out_val["overall_acc"])
        print(f"[G2] Ep{epoch:03d} | overall@{tuple(thrs)} va={va_overall*100:.2f}")

        if va_overall > best["val_overall_acc"]:
            best["val_overall_acc"] = va_overall
            best["state"] = {
                "model": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                "exits": [copy.deepcopy({k: v.detach().cpu() for k, v in h.state_dict().items()}) for h in exit_heads],
            }

    # restore best
    if best["state"] is not None:
        model.load_state_dict(best["state"]["model"], strict=True)
        for i, h in enumerate(exit_heads):
            h.load_state_dict(best["state"]["exits"][i], strict=True)

    return model, exit_heads, best

'''def cotrain_g2_multi_exit_v3(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs: int = 50,

    # g2: train deeper layers + classifier
    train_layer_indices: Sequence[int] = (1, 2),
    freeze_layer_indices: Sequence[int] = (0,),

    exit_heads: List[nn.Module],
    payload_exit_cfg: List[dict],   # each has "layer_idx"

    # --- thresholds ---
    # If auto_calibrate_thrs=True, these are initial values only.
    thrs: Sequence[float] = (227.0, 15.0),
    auto_calibrate_thrs: bool = True,
    # calibrate to desired exit rates (roughly)
    target_r0: float = 0.30,
    target_r1: float = 0.20,
    calibrate_batches: int = 3,    # how many batches per epoch to estimate margins

    # margin definition
    use_prob_margin: bool = False,

    # loss weights
    lambda_final: float = 1.0,
    lambda_exits: Optional[Sequence[float]] = (0.0, 0.10),  # g2: keep exit0 fixed => lambda_exits[0]=0

    # optimizer
    lr_backbone: float = 1e-4,
    lr_classifier: float = 1e-3,
    lr_exits: float = 5e-4,
    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,

    # tail emphasis for final
    beta_tail: float = 1.0,

    # choose best by cascade overall under current thrs
    best_metric: str = "val_overall_acc",
):
    """
    g2 v3:
      - Freeze: early layer(s) + exit0 head
      - Train: deeper layers + final classifier + exit1 head (and optionally other deeper exits)
      - Training loss is HARD-GATE aligned with eval:
          route each sample to exit0 / exit1 / final using (margin > thr)
          compute CE on the routed subset(s)
      - Optionally calibrate thr0/thr1 each epoch to match target exit rates.
    """
    model = model.to(device)
    exit_heads = [h.to(device) for h in exit_heads]
    assert len(exit_heads) == len(payload_exit_cfg)
    assert len(thrs) == len(exit_heads) == 2, "v3 assumes 2 exits for now (exit0, exit1)"

    if lambda_exits is None:
        lambda_exits = (0.0, 0.10)
    assert len(lambda_exits) == 2

    # -----------------------
    # Freeze / unfreeze
    # -----------------------
    set_requires_grad(model, False)

    for li in freeze_layer_indices:
        if 0 <= li < len(model.layers):
            set_requires_grad(model.layers[li], False)

    for li in train_layer_indices:
        set_requires_grad(model.layers[li], True)

    set_requires_grad(model.classifier, True)

    # exits: freeze exit0, train exit1
    set_requires_grad(exit_heads[0], False)
    set_requires_grad(exit_heads[1], True)

    params_backbone = []
    for li in train_layer_indices:
        params_backbone += [p for p in model.layers[li].parameters() if p.requires_grad]
    params_classifier = [p for p in model.classifier.parameters() if p.requires_grad]
    params_exits = [p for p in exit_heads[1].parameters() if p.requires_grad]  # only exit1

    print(f"[g2-v3] trainable params: backbone={sum(p.numel() for p in params_backbone)} "
          f"classifier={sum(p.numel() for p in params_classifier)} "
          f"exit1={sum(p.numel() for p in params_exits)}")

    optimizer = torch.optim.AdamW(
        [
            {"params": params_backbone,   "lr": lr_backbone,   "weight_decay": weight_decay},
            {"params": params_classifier, "lr": lr_classifier, "weight_decay": weight_decay},
            {"params": params_exits,      "lr": lr_exits,      "weight_decay": weight_decay},
        ]
    )

    thr0, thr1 = float(thrs[0]), float(thrs[1])
    best = {"val_overall_acc": -1.0, "state": None, "thrs": (thr0, thr1)}

    def _forward_logits_and_margins(xb):
        final_logits, h_list = model.forward_with_all_hidden(xb)

        # exit0
        li0 = int(payload_exit_cfg[0]["layer_idx"])
        logits0 = _head_logits_from_hidden_trainable(exit_heads[0], h_list[li0], device)
        m0 = _margin_from_logits(logits0, use_prob=use_prob_margin)

        # exit1
        li1 = int(payload_exit_cfg[1]["layer_idx"])
        logits1 = _head_logits_from_hidden_trainable(exit_heads[1], h_list[li1], device)
        m1 = _margin_from_logits(logits1, use_prob=use_prob_margin)

        return final_logits, logits0, logits1, m0, m1

    # -----------------------
    # Train loop
    # -----------------------
    for epoch in range(num_epochs):
        model.train()
        exit_heads[1].train()

        # ---- auto calibrate thrs per-epoch (fast) ----
        if auto_calibrate_thrs:
            m0_all, m1_all = [], []
            seen = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                _, _, _, m0, m1 = _forward_logits_and_margins(xb)
                m0_all.append(m0.detach().float().cpu())
                m1_all.append(m1.detach().float().cpu())
                seen += 1
                if seen >= calibrate_batches:
                    break
            m0_cat = torch.cat(m0_all, dim=0)
            m1_cat = torch.cat(m1_all, dim=0)

            thr0 = _calibrate_thr_from_margins(m0_cat, target_r0)
            # exit1 只對「沒被 exit0 拿走的」樣本定 thr 比較合理
            mask_tail0 = (m0_cat <= thr0)
            if mask_tail0.any():
                thr1 = _calibrate_thr_from_margins(m1_cat[mask_tail0], target_r1 / max(1e-6, float(mask_tail0.float().mean().item())))
            else:
                thr1 = float("inf")

        # ---- train over full epoch ----
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            final_logits, logits0, logits1, m0, m1 = _forward_logits_and_margins(xb)

            # hard routing (cascade)
            take0 = (m0 > thr0)
            take1 = (~take0) & (m1 > thr1)
            takeF = (~take0) & (~take1)

            # exit losses (only exit1 is trainable, exit0 frozen => we can skip loss0 or keep it as monitor)
            loss = 0.0

            # final loss: all samples mean + tail emphasis
            ce_all = F.cross_entropy(final_logits, yb)
            if takeF.any():
                ce_tail = F.cross_entropy(final_logits[takeF], yb[takeF])
            else:
                ce_tail = 0.0 * ce_all
            loss_final = ce_all + float(beta_tail) * ce_tail
            loss = loss + float(lambda_final) * loss_final

            # exit1 loss: only on take1 samples (ensure not empty)
            if float(lambda_exits[1]) > 0.0 and take1.any():
                loss_exit1 = F.cross_entropy(logits1[take1], yb[take1])
                loss = loss + float(lambda_exits[1]) * loss_exit1
            else:
                loss_exit1 = torch.zeros((), device=device)

            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                if params_backbone:
                    torch.nn.utils.clip_grad_norm_(params_backbone, grad_clip)
                if params_classifier:
                    torch.nn.utils.clip_grad_norm_(params_classifier, grad_clip)
                if params_exits:
                    torch.nn.utils.clip_grad_norm_(params_exits, grad_clip)
            optimizer.step()

        # -----------------------
        # Eval (cascade overall under current thrs)
        # -----------------------
        model.eval()
        for h in exit_heads:
            h.eval()

        out_val = eval_cascade_multi_exit(
            model, val_loader, device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=[thr0, thr1],
            use_prob_margin=use_prob_margin,
            log_margins=False,
        )
        va_overall = float(out_val["overall_acc"])

        print(
            f"[G2-v3] Ep{epoch:03d} | thr0={thr0:.3f} thr1={thr1:.3f} "
            f"| overall={va_overall*100:.2f} "
            f"| exit_rates={out_val['exit_rates']} final_rate={out_val['final_rate']:.4f}"
        )

        if va_overall > best["val_overall_acc"]:
            best["val_overall_acc"] = va_overall
            best["thrs"] = (thr0, thr1)
            best["state"] = {
                "model": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                "exits": [copy.deepcopy({k: v.detach().cpu() for k, v in h.state_dict().items()}) for h in exit_heads],
            }

    # restore best
    if best["state"] is not None:
        model.load_state_dict(best["state"]["model"], strict=True)
        for i, h in enumerate(exit_heads):
            h.load_state_dict(best["state"]["exits"][i], strict=True)

    return model, exit_heads, best'''


def cotrain_g2_multi_exit_v2(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs: int = 30,

    # g2: train deeper layers + classifier
    train_layer_indices: Sequence[int] = (1, 2),
    freeze_layer_indices: Sequence[int] = (0,),

    exit_heads: List[nn.Module],
    payload_exit_cfg: List[dict],

    # thrs: 初始值；之后每 epoch 可自动校准
    thrs: Sequence[float] = (1.0, 1.5),
    auto_calibrate_thrs: bool = True,
    r0_target: float = 0.70,
    r1_target: float = 0.15,
    cali_val_batches: int = 2,

    use_prob_margin: bool = False,

    # loss weights
    lambda_final: float = 1.0,
    lambda_exits: Optional[Sequence[float]] = None,  # e.g. (0.0, 0.1)

    # lr
    lr_backbone: float = 3e-4,
    lr_classifier: float = 3e-4,
    lr_exits: float = 5e-4,

    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,

    # tail emphasis
    beta_tail: float = 0.1,

    # gate-aware weighting
    use_gate_weighting: bool = True,
    gate_temp: float = 1.0,

    # best selection metric
    best_metric: str = "val_overall_acc",
):
    """
    ✅ g2 (as you defined):
      - freeze earlier exit (exit0) completely
      - train deeper backbone layers + exit1 + final classifier
    """

    model = model.to(device)
    exit_heads = [h.to(device) for h in exit_heads]

    assert len(exit_heads) == len(payload_exit_cfg)
    num_exits = len(exit_heads)
    assert num_exits == 2, "目前这版假设 2-exit（FMNIST 你的设置）"

    if lambda_exits is None:
        # ✅ g2: exit0 fixed -> lambda 设 0；exit1 给一点
        lambda_exits = (0.0, 0.1)
    else:
        assert len(lambda_exits) == num_exits

    # -----------------------
    # 0) Freeze / unfreeze
    # -----------------------
    def set_requires_grad(module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    set_requires_grad(model, False)
    for li in freeze_layer_indices:
        if 0 <= li < len(model.layers):
            set_requires_grad(model.layers[li], False)

    for li in train_layer_indices:
        if li < 0 or li >= len(model.layers):
            raise ValueError(f"train_layer_indices contains out-of-range layer {li}")
        set_requires_grad(model.layers[li], True)

    if not hasattr(model, "classifier"):
        raise ValueError("model has no classifier; g2 expects final classifier exists.")
    set_requires_grad(model.classifier, True)

    # exits: ✅ freeze exit0, train exit1 only
    for h in exit_heads:
        set_requires_grad(h, True)
    set_requires_grad(exit_heads[0], False)  # ✅ critical

    set_requires_grad(exit_heads[0], False)

    # -----------------------
    # 1) Optimizer param groups
    # -----------------------
    params_backbone = []
    for li in train_layer_indices:
        params_backbone += [p for p in model.layers[li].parameters() if p.requires_grad]

    params_classifier = [p for p in model.classifier.parameters() if p.requires_grad]

    params_exit1 = [p for p in exit_heads[1].parameters() if p.requires_grad]  # ✅ only exit1

    print(
        f"[g2] trainable params: backbone={sum(p.numel() for p in params_backbone)} "
        f"classifier={sum(p.numel() for p in params_classifier)} "
        f"exit1={sum(p.numel() for p in params_exit1)} "
        f"(exit0 frozen)"
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": params_backbone,   "lr": lr_backbone,   "weight_decay": weight_decay},
            {"params": params_classifier, "lr": lr_classifier, "weight_decay": weight_decay},
            {"params": params_exit1,      "lr": lr_exits,      "weight_decay": weight_decay},
        ]
    )

    best = {"val_overall_acc": -1.0, "state": None}

    thr0, thr1 = float(thrs[0]), float(thrs[1])

    # -----------------------
    # 2) Train loop
    # -----------------------
    for epoch in range(num_epochs):
        # ✅ 每个 epoch 先校准 thrs，保证 exit1 不会饿死
        if auto_calibrate_thrs:
            thr0, thr1 = _calibrate_thrs_from_val(
                model, val_loader, device,
                exit_heads=exit_heads,
                payload_exit_cfg=payload_exit_cfg,
                use_prob_margin=use_prob_margin,
                r0_target=r0_target,
                r1_target=r1_target,
                max_batches=cali_val_batches,
            )

        model.train()
        for h in exit_heads:
            h.train()
        exit_heads[0].eval()  # ✅ frozen exit0：用 eval 也更稳（无 dropout/bn）

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            final_logits, h_list = model.forward_with_all_hidden(xb)
            ce_final = F.cross_entropy(final_logits, yb, reduction="none")  # [B]

            # --------- compute margins for gating (exit0 frozen + exit1 trainable) ---------
            cfg0 = payload_exit_cfg[0]
            li0 = int(cfg0["layer_idx"])
            logits0 = _head_logits_from_hidden_trainable(exit_heads[0], h_list[li0], device)
            m0 = _margin_from_logits(logits0, use_prob=use_prob_margin)  # [B]
            take0 = (m0 > thr0)  # [B] bool

            cfg1 = payload_exit_cfg[1]
            li1 = int(cfg1["layer_idx"])
            logits1 = _head_logits_from_hidden_trainable(exit_heads[1], h_list[li1], device)
            m1 = _margin_from_logits(logits1, use_prob=use_prob_margin)
            take1 = (~take0) & (m1 > thr1)  # [B]

            tail = (~take0) & (~take1)

            # --------- final loss (all + tail upweight) ---------
            ce_all = ce_final.mean()
            if tail.any():
                ce_tail = F.cross_entropy(final_logits[tail], yb[tail])
            else:
                ce_tail = 0.0 * ce_all
            loss_final = ce_all + beta_tail * ce_tail

            # --------- exit1 loss (gate-weighted but avoid starving) ---------
            # 只对 exit1 训练：exit0 frozen
            if use_gate_weighting:
                # soft weight for exit1 only（对 take1 做 detach，避免 gate 被 CE 拉爆）
                w1 = torch.sigmoid((m1 - thr1) / gate_temp)  # [B]
                # 只在未被 exit0 拿走的样本上训练 exit1（更符合 cascade）
                w1 = w1 * (~take0).float()
                w1_det = w1.detach()

                ce1 = F.cross_entropy(logits1, yb, reduction="none")  # [B]
                eps = 1e-8
                loss_exit1 = (w1_det * ce1).sum() / (w1_det.sum() + eps)
            else:
                # naive: 只在 ~take0 的样本上训练 exit1（避免 exit0 抢走全部）
                if (~take0).any():
                    loss_exit1 = F.cross_entropy(logits1[~take0], yb[~take0])
                else:
                    loss_exit1 = 0.0 * loss_final

            loss = lambda_final * loss_final + float(lambda_exits[1]) * loss_exit1

            loss.backward()

            if grad_clip is not None:
                if params_backbone:
                    torch.nn.utils.clip_grad_norm_(params_backbone, grad_clip)
                if params_classifier:
                    torch.nn.utils.clip_grad_norm_(params_classifier, grad_clip)
                if params_exit1:
                    torch.nn.utils.clip_grad_norm_(params_exit1, grad_clip)

            optimizer.step()

        # -----------------------
        # 3) Eval (cascade)
        # -----------------------
        out_val = eval_cascade_multi_exit(
            model, val_loader, device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=(thr0, thr1),
            use_prob_margin=use_prob_margin,
            log_margins=False,
        )
        va_overall = float(out_val["overall_acc"])

        print(
            f"[G2] Ep{epoch:03d} | thr0={thr0:.3f} thr1={thr1:.3f} "
            f"| overall={va_overall*100:.2f} "
            f"| final={out_val['final_acc']*100:.2f} "
            f'| exit1={out_val["exit_accs"][1]*100:.2f} '
            f'| exit0={out_val["exit_accs"][0]*100:.2f} '
            f"| exit_rates={out_val['exit_rates']} final_rate={out_val['final_rate']:.4f}"
        )

        if va_overall > best["val_overall_acc"]:
            best["val_overall_acc"] = va_overall
            best["state"] = {
                "model": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                "exits": [copy.deepcopy({k: v.detach().cpu() for k, v in h.state_dict().items()}) for h in exit_heads],
                "thrs": (thr0, thr1),
            }

    # -----------------------
    # 4) Restore best
    # -----------------------
    if best["state"] is not None:
        model.load_state_dict(best["state"]["model"], strict=True)
        for i, h in enumerate(exit_heads):
            h.load_state_dict(best["state"]["exits"][i], strict=True)

    return model, exit_heads, best


def cotrain_g2_multi_exit_temp(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs: int = 30,

    # 哪些 layer 要 train（通常 layer1, layer2）
    train_layer_indices: Sequence[int] = (1, 2),

    # freeze 哪些 layer（g2 常見 freeze layer0）
    freeze_layer_indices: Sequence[int] = (0,),


    # 你已 build 好的 exits
    exit_heads: List[nn.Module],
    payload_exit_cfg: List[dict],   # list[dict]，包含 layer_idx / exit_tau...（給 eval 用）

    # cascade gate thresholds（長度 = num_exits）
    thrs: Sequence[float] = (1.0, 1.5),
    use_prob_margin: bool = False,

    # loss weights
    lambda_final: float = 1.0,
    lambda_exits: Optional[Sequence[float]] = None,  # e.g. (0.3, 0.3)

    # lr
    lr_backbone: float = 3e-4,
    lr_classifier: float = 3e-4,
    lr_exits: float = 3e-3,

    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,

    # --- new (gate-aware) ---
    gate_temp: float = 1.0,          # sigmoid 溫度，0.5~1.0 都可
    use_gate_weighting: bool = True,  # 開關

    # best selection metric
    best_metric: str = "val_overall_acc",  # 目前只用 cascade overall
):
    """
    g2: update layer1+layer2 + classifier + all exit heads
        (freeze layer0)

    Best model selected by eval_cascade_multi_exit(val, thrs).
    """
    beta_tail = 0.1
    model = model.to(device)

    assert len(exit_heads) == len(payload_exit_cfg), "exit_heads and payload_exit_cfg must align"
    num_exits = len(exit_heads)
    assert len(thrs) == num_exits, "len(thrs) must equal num_exits"

    if lambda_exits is None:
        lambda_exits = [0.3] * num_exits
    else:
        assert len(lambda_exits) == num_exits

    # -----------------------
    # 0) Freeze all first
    # -----------------------
    set_requires_grad(model, False)

    # freeze specified layers (redundant but explicit)
    for li in freeze_layer_indices:
        if 0 <= li < len(model.layers):
            set_requires_grad(model.layers[li], False)

    # unfreeze specified train layers
    for li in train_layer_indices:
        if li < 0 or li >= len(model.layers):
            raise ValueError(f"train_layer_indices contains out-of-range layer {li}")
        set_requires_grad(model.layers[li], True)

    # unfreeze final classifier (g2 必須動 classifier)
    if not hasattr(model, "classifier"):
        raise ValueError("model has no classifier; g2 expects final classifier exists.")
    set_requires_grad(model.classifier, True)

    # exit heads trainable
    exit_heads = [h.to(device) for h in exit_heads]
    for h in exit_heads:
        set_requires_grad(h, True)
    # freeze ex0
    #set_requires_grad(exit_heads[0], False)

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
    # only accept ex1
    #params_exits += [p for p in exit_heads[1].parameters() if p.requires_grad]

    print(f"[g2] trainable params: backbone={sum(p.numel() for p in params_backbone)} "
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
    # 2) Train loop
    # -----------------------
    epoch_idx = 0
    for epoch in range(num_epochs):
        model.train()
        for h in exit_heads:
            h.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            
            final_logits, h_list = model.forward_with_all_hidden(xb)

            # per-sample CE
            ce_final = F.cross_entropy(final_logits, yb, reduction="none")  # [B]

            # Mirror g2.py: penalize final head on all samples, then upweight
            # the "tail" samples that are not taken by any early exit.
            tail_mask = torch.ones_like(yb, dtype=torch.bool)
            for i in range(num_exits):
                cfg = payload_exit_cfg[i]
                layer_idx = int(cfg["layer_idx"])
                thr_i = float(thrs[i])

                h_i = h_list[layer_idx]
                logits_i = _head_logits_from_hidden_trainable(exit_heads[i], h_i, device)
                margin_i = _margin_from_logits(logits_i, use_prob=use_prob_margin)
                exit_mask_i = tail_mask & (margin_i > thr_i)
                tail_mask = tail_mask & (~exit_mask_i)

            ce_all = ce_final.mean()
            if tail_mask.any():
                ce_tail = F.cross_entropy(final_logits[tail_mask], yb[tail_mask])
            else:
                ce_tail = 0.0 * ce_all
            loss_final = ce_all + beta_tail * ce_tail

            # -------- gate-aware weighting --------
            if use_gate_weighting:
                # soft gate weights w_i = sigmoid((margin_i - thr_i)/T)
                w_list = []
                ce_exit_list = []
                m_list = []
                for i in range(num_exits):
                    cfg = payload_exit_cfg[i]
                    layer_idx = int(cfg["layer_idx"])
                    thr_i = float(thrs[i])

                    h_i = h_list[layer_idx]
                    logits_i = _head_logits_from_hidden_trainable(exit_heads[i], h_i, device)  # [B,C]

                    # per-sample exit CE
                    ce_i = F.cross_entropy(logits_i, yb, reduction="none")  # [B]
                    ce_exit_list.append(ce_i)

                    # margin (logits margin) -> [B]
                    m_i = _margin_from_logits(logits_i, use_prob=use_prob_margin)
                    m_list.append(m_i)

                    # soft gate weight
                    w_i = torch.sigmoid((m_i - thr_i) / gate_temp)  # [B] in (0,1)
                    w_list.append(w_i)

                '''# final weight = prob of "still undecided"
                # 用連乘近似：w_final = Π (1 - w_i)
                w_final = torch.ones_like(ce_final)
                for w_i in w_list:
                    w_final = w_final * (1.0 - w_i)

                # weighted final loss
                loss_final = (w_final * ce_final).mean()'''

                '''# weighted exit losses
                loss_exit_sum = 0.0
                for i in range(num_exits):
                    w_i = w_list[i].detach()  # ✅ 這裡建議 detach，避免 gate 自己被 CE 拉到極端
                    loss_exit_i = (w_i * ce_exit_list[i]).mean()
                    loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * loss_exit_i

                loss = float(lambda_final) * loss_final + loss_exit_sum'''
                # ce_final: [B]
                # ce_i: [B] for each exit

                eps = 1e-8
                u = torch.ones_like(ce_final)  # [B], undecided prob
                loss_exit_sum = 0.0
                #T0 ∈ {2.5, 3.0, 3.5}
	            #T1 ∈ {1.0, 1.25, 1.5}
                # (T0,T1) = (3.0,1.0), (3.0,1.25), (3.0,1.5), (3.5,1.25) 
                # T1 ∈ {1.10, 1.25, 1.40, 1.50, 1.60, 1.75}
                T = [3.5, 1.25]
                for i in range(num_exits):
                    m_i = m_list[i]                  # [B]
                    ce_i = ce_exit_list[i]           # [B]
                    thr_i = float(thrs[i])
                    Ti = float(T[i])                 # or shared T

                    w_i = torch.sigmoid((m_i - thr_i) / Ti)   # [B] in (0,1)
                    take_i = u * w_i                           # [B]

                    # gate 不想被 CE 拉爆：detach weights
                    take_i_det = take_i.detach()

                    # ✅ weighted average (NOT mean)
                    loss_i = (take_i_det * ce_i).sum() / (take_i_det.sum() + eps)
                    loss_exit_sum = loss_exit_sum + lambda_exits[i] * loss_i

                    # 更新 undecided（這裡建議也用 detach 避免數值怪）
                    #u = u * (1.0 - w_i.detach())
                    u = u * (1.0 - w_i)

                loss = lambda_final * loss_final + loss_exit_sum

            else:
                # fallback: 原本的 naive loss
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
        # 3) Eval (use your existing cascade eval)
        # -----------------------
        # 你已有 eval_cascade_multi_exit
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
            f"[G2] Ep{epoch:03d} | overall@{tuple(thrs)} va={va_overall*100:.2f} "
            f"| exit_rates={out_val['exit_rates']} final_rate={out_val['final_rate']:.4f}"
        )

        if va_overall > best["val_overall_acc"]:
            best["val_overall_acc"] = va_overall
            # 存 backbone + exits
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

    parser.add_argument("--epochs", type=int, default=30)
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

    # g2: train layer1+layer2 + classifier + exits
    '''backbone, exit_heads, best = cotrain_g2_multi_exit_v3(
        model=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=100,

        train_layer_indices=(1, 2),
        freeze_layer_indices=(0,),

        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
        thrs=[227, 15], #thr_list,

        lambda_final=1.0,
        lambda_exits=(0.05, 0.10),

        lr_backbone=1e-4,
        lr_classifier=1e-3,
        lr_exits=5e-4,
        weight_decay=1e-3,
        auto_calibrate_thrs=False,  # 先關掉自動校準，看看固定 thrs 的效果；之后可以再开看看
        beta_tail=1.0,
    )

    print("Best val overall acc:", best["val_overall_acc"])'''
    '''backbone, exit_heads, best = cotrain_g2_multi_exit(
        model=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=100,

        train_layer_indices=(1, 2),
        freeze_layer_indices=(0,),      # 你原本的 g2

        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
        thrs=[227, 10],

        lambda_final=1.0,
        lambda_exits=(0.0, 0.0),       # ✅ g2 常见：exit0 固定，exit1 可训

        lr_backbone=1e-4,
        lr_classifier=3e-4,             # ✅ 降
        lr_exits=5e-4,
        weight_decay=1e-3,

        beta_tail=0.2,                  # ✅ 降
        #warmup_epochs=10,               # ✅ 新增
        #warmup_train_layers=(),         # warmup 不训 backbone
        #freeze_exit0_in_warmup=True,
        #freeze_exit0_after=True,
        #train_exit_ids_after=(1,),      # 只训 exit1
    )'''

    # G2 recommended: warmup + co-train final + exit1 (full batch)
    '''backbone, exit_heads, best = cotrain_g2_multi_exit_v4(
        model=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=100,

        train_layer_indices=(1,2),
        freeze_layer_indices=(0,),

        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,

        # IMPORTANT: training thresholds (NOT necessarily deployment thresholds)
        thrs=(227.0, 18.0),          # <-- 把 thr1 拉高，避免 exit1 吃光
        train_exit_ids=(1,),          # exit0 frozen, train exit1 only
        exit1_only_on_tail=False,     # <- 先別用 tail 版本

        warmup_epochs=8,              # 前 8 epoch 只訓練 final 分支（方法 I）

        lambda_final=1.0,
        lambda_exits=(0.0, 0.03),     # exit1 權重先小一點；0.02~0.05 都可試

        lr_backbone=1e-4,
        lr_classifier=1e-3,
        lr_exits=5e-4,

        beta_tail=1.0,                # 你原本的 final tail penalty 還是可以留
    )'''

    backbone, exit_heads, best = cotrain_g2_multi_exit_v4(
        model=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=100,

        train_layer_indices=(1, 2),
        freeze_layer_indices=(0,),

        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,

        thrs_train=thr_list,
        thrs_eval_list=[thr_list, thr_list],
        best_eval_idx=0,                  # 用 (227,18) 選 best
        #best_metric="val_overall_acc",
        best_metric="val_final_tail_only",

        train_exit_ids=(1,),              # 只 train exit1
        exit_only_on_tail=False,          # 先不要 tail-only（你之前說會餓死）

        lambda_final=1.0,
        lambda_exits=(0.0, 0.05),         # exit0 frozen => lambda 0
        beta_tail=1.0,                    # 你想用 tail penalty 的話就留著

        lr_backbone=3e-4,
        lr_classifier=1e-3,
        lr_exits=5e-4,
        weight_decay=1e-3,
        warmup_epochs=10,
    )

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
        print_eval_profile(f"G2-v2 exit0@thr={thr}", out)
    
    print('=======================================')
    thrs0 = [0.0, 0.25, 0.5, 0.75, 1.0, math.inf]
    thrs1 = [1.2, 1.5, 1.8, 2.0, math.inf]

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
            print_eval_profile(f"G2-v2 cascade@({thr0},{thr1})", out)


    # 1) 走 model(x) 的 final-only
    vl1, va1 = eval_final_only(backbone, val_loader, device)

    # 2) 走 forward_with_all_hidden 拿 final_logits 的 final-only
    @torch.no_grad()
    def eval_final_only_via_hidden(model, loader, device):
        model.eval()
        correct = total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            final_logits, _ = model.forward_with_all_hidden(xb)
            pred = final_logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
        return correct / total

    va2 = eval_final_only_via_hidden(backbone, val_loader, device)

    print("final_only via model(x):", va1)
    print("final_only via forward_with_all_hidden:", va2)

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



    
