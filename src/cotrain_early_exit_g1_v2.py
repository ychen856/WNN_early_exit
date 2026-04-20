# src/train/train_wnn.py
import argparse
import copy
import math
from pathlib import Path
import json
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn.functional as F
from src.core.linearExitHead import build_exits_from_ckpt
from src.core.multiLayerWNN import build_backbone_from_ckpt, save_ckpt_v2
from src.dataio.data import build_loaders_bits
from src.dataio.mapping import make_tuple_mapping, audit_mapping
from src.early_exit import _collect_margins_exit0, _head_logits_from_hidden, _weighted_avg_loss
from src.exit.ckpt_exit import ExitConfig
from src.prune import *
from src.early_exit import *
from src.tools.utils import print_sweep_table  
from test import *
from src.core.infer import *
from src.dataio.encode import minmax_normalize, thermometer_encode, dt_thermometer_encode, compute_dt_thresholds
from src.tools.fpga_tools.export_fpga_bundle import export_multilayer_2layer_for_fpga, verify_multilayer_export
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader


# from core.decision import tune_decision  #  Step 2

CANONICAL_MAPPING = Path("/Users/yi-chunchen/workspace/WNN_early_exit/models/meta/tuple_mapping.json")

def _parse_list(s, cast=int):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]

def _broadcast(xs, n):
    if len(xs) == 1:
        return xs * n
    if len(xs) == n:
        return xs
    raise ValueError(f"Need 1 or {n} values, got {len(xs)}")

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


def cotrain_g1_layer1_exit1_only(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=30,
    layer1_idx=1,          # 這個很重要：layer1 是 index 1
    lr_layer1=3e-4,
    lr_exit1=3e-3,
    lambda_exit=0.3,
    thrs=(1.0, 1.5),       # (thr0, thr1) cascade 用
    weight_decay=1e-3,
    grad_clip=1.0,
    exit_heads=None,       # 若你用 list head，就傳進來
    payload_exit_cfg=None,    # 同上（cascade eval 需要）
    use_prob_margin=False, # 你目前 margin 是用 logits margin
):
    model.to(device)

    # -----------------------
    # 0) Freeze everything
    # -----------------------
    set_requires_grad(model, False)

    # -----------------------
    # 1) Unfreeze layer1 + exit1 only
    # -----------------------
    if layer1_idx >= len(model.layers):
        raise ValueError(f"layer1_idx={layer1_idx} out of range, layers={len(model.layers)}")

    set_requires_grad(model.layers[layer1_idx], True)

    # exit1 classifier / head
    # 你有兩種可能：
    # A) model.exit1_classifier
    # B) exit_heads[1]
    if getattr(model, "exit1_classifier", None) is not None:
        exit1_module = model.exit1_classifier
    else:
        assert exit_heads is not None, "Need model.exit1_classifier or exit_heads[1]"
        exit1_module = exit_heads[1].to(device)

    set_requires_grad(exit1_module, True)

    # final classifier 明確保持 frozen（g2 才動）
    if hasattr(model, "classifier"):
        set_requires_grad(model.classifier, False)

    # -----------------------
    # 2) Optimizer
    # -----------------------
    params_layer1 = [p for p in model.layers[layer1_idx].parameters() if p.requires_grad]
    params_exit1  = [p for p in exit1_module.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": params_layer1, "lr": lr_layer1, "weight_decay": weight_decay},
            {"params": params_exit1,  "lr": lr_exit1,  "weight_decay": weight_decay},
        ]
    )

    best = {"val_overall_acc": -1.0, "state": None}

    for epoch in range(num_epochs):
        # -----------------------
        # train
        # -----------------------
        model.train()
        if getattr(model, "exit1_classifier", None) is None and exit_heads is not None:
            # 若 exit1 是外部 head，記得也設 train()
            exit1_module.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            final_logits, h_list = model.forward_with_all_hidden(xb)

            # exit1 logits：從 layer1 hidden 接 head
            h1 = h_list[layer1_idx]  # [B, D1]

            # 若你 exit1 head 是 ExitHead 物件（有 keep_idx/mu/sigma），用你已經寫好的 helper：
            if hasattr(exit1_module, "exit_keep_idx"):
                exit1_logits = _head_logits_from_hidden(exit1_module, h1, device)

            else:
                # 如果只是 nn.Linear，代表你已經把 keep_idx/normalize 做在 model.forward 內部
                exit1_logits = exit1_module(h1)

            loss_final = F.cross_entropy(final_logits, yb)
            loss_exit1 = F.cross_entropy(exit1_logits, yb)
            loss = loss_final + lambda_exit * loss_exit1


            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params_layer1, grad_clip)
                torch.nn.utils.clip_grad_norm_(params_exit1, grad_clip)
            optimizer.step()

        # -----------------------
        # eval (cascade overall)
        # -----------------------
        model.eval()
        if getattr(model, "exit1_classifier", None) is None and exit_heads is not None:
            exit1_module.eval()

        # 這裡用你已經驗證過的 cascade eval
        # thrs = (thr0, thr1)
        thr0, thr1 = float(thrs[0]), float(thrs[1])

        # 你既然要 “multi-exit cascade”，val_overall 直接用 cascade
        # exit_heads / cfg_list 請務必按順序：exit0 是 0, exit1 是 1
        assert exit_heads is not None and payload_exit_cfg is not None, "Need exit_heads + payload_exit_cfg for cascade eval"

        out = eval_cascade_multi_exit(
            model, val_loader, device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=[thr0, thr1],
            use_prob_margin=use_prob_margin,
        )
        va_overall_acc = out["overall_acc"]
        exit_rates = out["exit_rates"]
        final_rate = out["final_rate"]

        print(
            f"[G1] Ep{epoch:03d} "
            f"| overall@({thr0},{thr1}) va={va_overall_acc*100:.2f} "
            f"| exit_rates={exit_rates} final_rate={final_rate:.4f}"
        )

        if va_overall_acc > best["val_overall_acc"]:
            best["val_overall_acc"] = va_overall_acc
            best["state"] = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)

    return model, best


import torch
import torch.nn.functional as F



@torch.no_grad()
def eval_exit_only(model, exit_head, loader, device, layer_idx: int):
    model.eval()
    exit_head.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        final_logits, h_list = model.forward_with_all_hidden(xb)
        h = h_list[layer_idx]
        logits = _head_logits_from_hidden(exit_head, h, device)
        loss = F.cross_entropy(logits, yb)
        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        loss_sum += loss.item() * yb.numel()
    return loss_sum / total, correct / total


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
import torch.nn.functional as F

import copy
import torch
import torch.nn.functional as F

import copy
import torch
import torch.nn.functional as F


def cotrain_g1_stage_v3(
    model,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs=30,
    layer_idx: int,
    exit_id: int,
    lr_layer=3e-4,
    lr_exit=3e-3,
    lambda_exit=1.0,

    # g1 典型：不動 final classifier（你要 g2/g3 才動）
    use_final_loss=False,
    lambda_final=1.0,

    # cascade eval 只拿來 report，不拿來選 best（stage1 用 exit-only 選 best）
    thrs=(1.0, 1.5),
    weight_decay=1e-3,
    grad_clip=1.0,

    exit_heads=None,          # list[ExitHead]
    payload_exit_cfg=None,    # list[dict]
    use_prob_margin=False,    # keep for eval function signature

    # ---- NEW for Stage1 residual training ----
    enable_soft_residual: bool = True,
    target_exit0_rate: float = 0.70,     # keep exit0 taking ~70% during stage1
    gate_temp0: float = 30.0,            # IMPORTANT: margin scale大(你現在~100~400)，T要大避免變hard mask
    thr0_update_batches: int = 20,       # 用多少個train batch估thr0（快一點）
):
    """
    G1 stage-wise co-train:
      - update: model.layers[layer_idx] + exit_heads[exit_id]
      - freeze: everything else (other layers, final classifier, other exit heads)

    v3 change:
      - if exit_id==1 and enable_soft_residual:
          train exit1 on residual distribution using SOFT weights from exit0 margin gate.
          thr0 is updated each epoch to keep exit0_rate about target_exit0_rate.
    """
    assert exit_heads is not None
    assert payload_exit_cfg is not None
    assert 0 <= layer_idx < len(model.layers)
    assert 0 <= exit_id < len(exit_heads)

    model = model.to(device)
    exit_heads = [h.to(device) for h in exit_heads]
    exit_module = exit_heads[exit_id]

    # ---- freeze all ----
    set_requires_grad(model, False)
    for h in exit_heads:
        set_requires_grad(h, False)

    # ---- unfreeze current layer + current exit head ----
    set_requires_grad(model.layers[layer_idx], True)
    set_requires_grad(exit_module, True)

    # ---- keep final classifier frozen in g1 ----
    if hasattr(model, "classifier"):
        set_requires_grad(model.classifier, False)

    params_layer = [p for p in model.layers[layer_idx].parameters() if p.requires_grad]
    params_exit  = [p for p in exit_module.parameters() if p.requires_grad]
    assert params_layer and params_exit

    optimizer = torch.optim.AdamW(
        [
            {"params": params_layer, "lr": lr_layer, "weight_decay": weight_decay},
            {"params": params_exit,  "lr": lr_exit,  "weight_decay": weight_decay},
        ]
    )

    # best selection: stage-wise objective
    best = {"val_metric": -1.0, "state": None, "exit_states": None}

    # ---- Stage1: maintain a dynamic thr0 to prevent starvation ----
    thr0_dyn = float(thrs[0])  # initial
    exit0_head = exit_heads[0] if len(exit_heads) > 0 else None

    for epoch in range(num_epochs):
        # ---------- (A) update thr0 for stage1 ----------
        if enable_soft_residual and exit_id == 1 and exit0_head is not None:
            # compute margins on some train batches
            m0_all = _collect_margins_exit0(
                model, train_loader, device,
                exit0_head=exit0_head, layer0_idx=0,
                max_batches=thr0_update_batches
            )
            if m0_all.numel() > 0:
                # want P(m0 > thr0) ~= target_exit0_rate  => thr0 = quantile(m0, 1-target_exit0_rate)? NO
                # because exit when m0 > thr0. So exit rate = 1 - CDF(thr0).
                # set thr0 at (1 - target_exit0_rate) quantile.
                q = max(0.0, min(1.0, 1.0 - float(target_exit0_rate)))
                thr0_dyn = float(torch.quantile(m0_all, q).item())

        # ---------- (B) train ----------
        model.train()
        exit_module.train()
        # exit0 head should stay frozen/eval-ish (for stable gate)
        if exit0_head is not None:
            exit0_head.eval()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            final_logits, h_list = model.forward_with_all_hidden(xb)
            h = h_list[layer_idx]
            exit_logits = _head_logits_from_hidden(exit_module, h, device)

            # per-sample CE
            ce_exit = F.cross_entropy(exit_logits, yb, reduction="none")  # [B]

            # -------- soft residual weighting for stage1 only --------
            if enable_soft_residual and exit_id == 1 and exit0_head is not None:
                h0 = h_list[0]
                logits0 = _head_logits_from_hidden(exit0_head, h0, device)  # [B,C]
                top2 = torch.topk(logits0, k=2, dim=-1).values
                m0 = top2[:, 0] - top2[:, 1]  # [B]

                # g0 = prob exit0 takes; r = residual prob
                g0 = torch.sigmoid((m0 - thr0_dyn) / float(gate_temp0))
                r = (1.0 - g0).detach()  # detach to avoid gate being pushed by CE
                #print(f'r_mean: {(1 - g0).mean().item():.3f}')
                #print(f'exit 0 rate: {(g0 > 0.5).float().mean():.4f}')

                loss_exit = _weighted_avg_loss(ce_exit, r)
            else:
                loss_exit = ce_exit.mean()

            loss = float(lambda_exit) * loss_exit

            # optional: final loss (normally False for g1)
            if use_final_loss:
                loss_final = F.cross_entropy(final_logits, yb)
                loss = loss + float(lambda_final) * loss_final

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params_layer, grad_clip)
                torch.nn.utils.clip_grad_norm_(params_exit, grad_clip)
            optimizer.step()

        # ---------- (C) eval ----------
        model.eval()
        for hhh in exit_heads:
            hhh.eval()

        # stage-wise metric: exit-only acc on its own layer
        va_exit_loss, va_exit_acc = eval_exit_only(model, exit_module, val_loader, device, layer_idx)

        # report cascade metric (optional)
        if exit_id == 1:
            thrs_eval = (float("inf"), float(thrs[1]))
        else:
            thrs_eval = thrs

        out = eval_cascade_multi_exit(
            model, val_loader, device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=thrs_eval,
            use_prob_margin=use_prob_margin,
        )
        va_overall_acc = float(out["overall_acc"])

        # print debug
        extra = ""
        if enable_soft_residual and exit_id == 1:
            extra = f" | thr0_dyn={thr0_dyn:.3f} T0={gate_temp0} target_r={1-target_exit0_rate:.2f}"

        print(
            f"[G1-stage-v3] layer={layer_idx} exit={exit_id} Ep{epoch:03d}"
            f" | exit_only_acc={va_exit_acc*100:.2f}"
            f" | overall@{thrs_eval} va={va_overall_acc*100:.2f}"
            f" | exit_rates={out['exit_rates']} final_rate={out['final_rate']:.4f}"
            f"{extra}"
        )

        # choose best by stage-wise metric (exit-only)
        metric = float(va_exit_acc)
        if metric > best["val_metric"]:
            best["val_metric"] = metric
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["exit_states"] = [{k: v.detach().cpu().clone() for k, v in hhh.state_dict().items()}
                                   for hhh in exit_heads]

    # restore best
    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
        for hhh, sd in zip(exit_heads, best["exit_states"]):
            hhh.load_state_dict(sd, strict=True)

    return model, best



def cotrain_g1_stage_v4(
    model,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs=30,
    layer_idx: int,
    exit_id: int,
    lr_layer=3e-4,
    lr_exit=3e-3,
    weight_decay=1e-3,
    grad_clip=1.0,

    # IMPORTANT: G1 for head accuracy only
    lambda_exit=1.0,          # 直接 1.0，別縮小到 0.3（會讓 head 學很慢）
    label_smoothing=0.0,      # 先關掉，確認 head acc 正常再加回去

    # logging/reporting only
    exit_heads=None,
    payload_exit_cfg=None,
    thrs=(1.0, 1.5),
    use_prob_margin=False,
):
    """
    Stage-wise co-train (head-accuracy first):
      - update: model.layers[layer_idx] + exit_heads[exit_id]
      - freeze: everything else (other layers, final classifier, other exit heads)
    Best selection: EXIT-ONLY accuracy on val set.
    Cascade eval: reporting only; stage1 uses (inf, thr1) to avoid starving exit1.
    """
    assert exit_heads is not None
    assert payload_exit_cfg is not None
    assert 0 <= layer_idx < len(model.layers)
    assert 0 <= exit_id < len(exit_heads)

    model = model.to(device)
    exit_heads = [h.to(device) for h in exit_heads]
    exit_module = exit_heads[exit_id]

    # ---- freeze everything ----
    set_requires_grad(model, False)
    for h in exit_heads:
        set_requires_grad(h, False)

    # ---- unfreeze target layer + target exit ----
    set_requires_grad(model.layers[layer_idx], True)
    set_requires_grad(exit_module, True)

    # ---- keep final classifier frozen in G1 ----
    if hasattr(model, "classifier"):
        set_requires_grad(model.classifier, False)

    params_layer = [p for p in model.layers[layer_idx].parameters() if p.requires_grad]
    params_exit  = [p for p in exit_module.parameters() if p.requires_grad]
    assert params_layer and params_exit

    optimizer = torch.optim.AdamW(
        [
            {"params": params_layer, "lr": lr_layer, "weight_decay": weight_decay},
            {"params": params_exit,  "lr": lr_exit,  "weight_decay": weight_decay},
        ]
    )

    best = {"val_exit_acc": -1.0, "state": None, "exit_states": None}

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        exit_module.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            # must keep grad through backbone to layer_idx
            _, h_list = model.forward_with_all_hidden(xb)
            h = h_list[layer_idx]

            exit_logits = _head_logits_from_hidden(exit_module, h, device)
            loss_exit = F.cross_entropy(exit_logits, yb, label_smoothing=label_smoothing)
            loss = lambda_exit * loss_exit

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params_layer, grad_clip)
                torch.nn.utils.clip_grad_norm_(params_exit, grad_clip)
            optimizer.step()

        # ---- eval: EXIT-ONLY (selection metric) ----
        va_exit_loss, va_exit_acc = eval_exit_only(model, exit_module, val_loader, device, layer_idx)

        # ---- eval: cascade (report only) ----
        model.eval()
        for h in exit_heads:
            h.eval()

        if exit_id == 1 and len(thrs) >= 2:
            thrs_eval = (float("inf"), float(thrs[1]))
        else:
            thrs_eval = thrs

        out = eval_cascade_multi_exit(
            model, val_loader, device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=thrs_eval,
            use_prob_margin=use_prob_margin,
        )

        print(
            f"[G1-v3] layer={layer_idx} exit={exit_id} Ep{epoch:03d} | "
            f"exit_only_acc={va_exit_acc*100:.2f} exit_only_loss={va_exit_loss:.4f} | "
            f"cascade_overall={out['overall_acc']*100:.2f} exit_rates={out['exit_rates']} rf={out['final_rate']:.4f}"
        )

        # ---- select best by EXIT-ONLY ----
        if va_exit_acc > best["val_exit_acc"]:
            best["val_exit_acc"] = va_exit_acc
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["exit_states"] = [{k: v.detach().cpu().clone() for k, v in h.state_dict().items()}
                                   for h in exit_heads]

    # restore best
    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
        for h, sd in zip(exit_heads, best["exit_states"]):
            h.load_state_dict(sd, strict=True)

    return model, best




def cotrain_g1_stage_2(
    model,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs=30,
    layer_idx: int,
    exit_id: int,
    lr_layer=3e-4,
    lr_exit=3e-3,
    lambda_exit=0.3,
    use_final_loss=False,     # 建議 stage-wise 預設 False
    lambda_final=1.0,
    thrs=(1.0, 1.5),
    weight_decay=1e-3,
    grad_clip=1.0,
    exit_heads=None,          # list[ExitHead]
    payload_exit_cfg=None,    # list[dict] (for cascade eval)
    use_prob_margin=False,

    # ---- new knobs ----
    warmup_head_epochs: int = 8,     # ✅ 建議 5~10
    warmup_lr_exit: float = None,    # None -> use lr_exit
    best_on: str = "exit_only_acc",  # "exit_only_acc" or "overall_acc"（不建議用 overall）
    report_cascade: bool = True,     # 只用來 print / 觀察
):
    """
    G1 stage-wise co-train (improved):
      - Warmup: train ONLY exit head for a few epochs (stabilize feature->logit alignment)
      - Cotrain: train backbone layer[layer_idx] + that exit head
      - Best selection: by exit-only metric (default), to avoid starving the exit.

    Notes:
      - Everything else is frozen (other backbone layers, final classifier, other exit heads)
      - Cascade eval is only for reporting; stage selection uses exit-only metric.
    """
    assert exit_heads is not None, "exit_heads required"
    assert payload_exit_cfg is not None, "payload_exit_cfg required for cascade eval"
    assert 0 <= layer_idx < len(model.layers)
    assert 0 <= exit_id < len(exit_heads)

    model = model.to(device)
    exit_heads = [h.to(device) for h in exit_heads]
    exit_module = exit_heads[exit_id]

    # -----------------------
    # helpers
    # -----------------------
    def _snapshot_state():
        return {
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "exits": [{k: v.detach().cpu().clone() for k, v in h.state_dict().items()} for h in exit_heads],
        }

    def _restore_state(state):
        model.load_state_dict(state["model"], strict=False)
        for h, sd in zip(exit_heads, state["exits"]):
            h.load_state_dict(sd, strict=True)

    def _thrs_eval_for_reporting():
        # keep your rule: during stage training of exit1, don't let exit0 take samples in cascade reporting
        if exit_id == 1 and len(thrs) >= 2:
            return (float("inf"), float(thrs[1]))
        return thrs

    def _compute_losses(final_logits, exit_logits, yb):
        loss_exit = F.cross_entropy(exit_logits, yb)
        if use_final_loss:
            loss_final = F.cross_entropy(final_logits, yb)
            loss = lambda_final * loss_final + lambda_exit * loss_exit
        else:
            loss_final = None
            loss = lambda_exit * loss_exit
        return loss, loss_exit, loss_final

    # -----------------------
    # 0) Freeze everything first
    # -----------------------
    set_requires_grad(model, False)
    for h in exit_heads:
        set_requires_grad(h, False)

    # keep final classifier frozen in g1
    if hasattr(model, "classifier"):
        set_requires_grad(model.classifier, False)

    # -----------------------
    # 1) Warmup: train ONLY exit head
    # -----------------------
    best = {"val_metric": -1.0, "state": None, "val_overall_acc": None, "val_exit_acc": None}

    warmup_epochs = int(max(0, warmup_head_epochs))
    warmup_lr = float(lr_exit if warmup_lr_exit is None else warmup_lr_exit)

    if warmup_epochs > 0:
        set_requires_grad(exit_module, True)

        params_exit = [p for p in exit_module.parameters() if p.requires_grad]
        assert len(params_exit) > 0, "No trainable params in selected exit head (warmup)"

        opt_warm = torch.optim.AdamW(
            [{"params": params_exit, "lr": warmup_lr, "weight_decay": weight_decay}]
        )

        for epoch in range(warmup_epochs):
            model.train()
            exit_module.train()

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt_warm.zero_grad(set_to_none=True)

                # forward backbone -> hidden
                final_logits, h_list = model.forward_with_all_hidden(xb)
                h = h_list[layer_idx]
                exit_logits = _head_logits_from_hidden(exit_module, h, device)

                loss, _, _ = _compute_losses(final_logits, exit_logits, yb)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(params_exit, grad_clip)
                opt_warm.step()

            # ---- eval ----
            model.eval()
            for hmod in exit_heads:
                hmod.eval()

            va_exit_loss, va_exit_acc = eval_exit_only(model, exit_module, val_loader, device, layer_idx)

            va_overall_acc = None
            if report_cascade:
                out = eval_cascade_multi_exit(
                    model, val_loader, device,
                    exit_heads=exit_heads,
                    exit_cfg_list=payload_exit_cfg,
                    thrs=_thrs_eval_for_reporting(),
                    use_prob_margin=use_prob_margin,
                )
                va_overall_acc = float(out["overall_acc"])

            print(
                f"[G1-warmup] layer={layer_idx} exit={exit_id} Ep{epoch:03d} "
                f"| exit_only_acc={va_exit_acc*100:.2f}%"
                + (f" | overall@{_thrs_eval_for_reporting()} va={va_overall_acc*100:.2f}%" if va_overall_acc is not None else "")
            )

            metric = va_exit_acc if best_on == "exit_only_acc" else (va_overall_acc if va_overall_acc is not None else va_exit_acc)
            if metric > best["val_metric"]:
                best["val_metric"] = metric
                best["val_exit_acc"] = va_exit_acc
                best["val_overall_acc"] = va_overall_acc
                best["state"] = _snapshot_state()

        # restore best after warmup (stabilize before cotrain)
        if best["state"] is not None:
            _restore_state(best["state"])

    # -----------------------
    # 2) Cotrain: train backbone layer + exit head
    # -----------------------
    # re-freeze everything, then unfreeze selected layer + exit
    set_requires_grad(model, False)
    for h in exit_heads:
        set_requires_grad(h, False)
    if hasattr(model, "classifier"):
        set_requires_grad(model.classifier, False)

    set_requires_grad(model.layers[layer_idx], True)
    set_requires_grad(exit_module, True)

    params_layer = [p for p in model.layers[layer_idx].parameters() if p.requires_grad]
    params_exit = [p for p in exit_module.parameters() if p.requires_grad]
    assert len(params_layer) > 0, "No trainable params in selected layer (cotrain)"
    assert len(params_exit) > 0, "No trainable params in selected exit head (cotrain)"

    opt = torch.optim.AdamW(
        [
            {"params": params_layer, "lr": lr_layer, "weight_decay": weight_decay},
            {"params": params_exit,  "lr": lr_exit,  "weight_decay": weight_decay},
        ]
    )

    cotrain_epochs = int(max(0, num_epochs))
    for epoch in range(cotrain_epochs):
        model.train()
        exit_module.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)

            final_logits, h_list = model.forward_with_all_hidden(xb)
            h = h_list[layer_idx]
            exit_logits = _head_logits_from_hidden(exit_module, h, device)

            loss, _, _ = _compute_losses(final_logits, exit_logits, yb)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params_layer, grad_clip)
                torch.nn.utils.clip_grad_norm_(params_exit, grad_clip)
            opt.step()

        # ---- eval ----
        model.eval()
        for hmod in exit_heads:
            hmod.eval()

        va_exit_loss, va_exit_acc = eval_exit_only(model, exit_module, val_loader, device, layer_idx)

        va_overall_acc = None
        out = None
        if report_cascade:
            out = eval_cascade_multi_exit(
                model, val_loader, device,
                exit_heads=exit_heads,
                exit_cfg_list=payload_exit_cfg,
                thrs=_thrs_eval_for_reporting(),
                use_prob_margin=use_prob_margin,
            )
            va_overall_acc = float(out["overall_acc"])

        print(
            f"[G1-cotrain] layer={layer_idx} exit={exit_id} Ep{epoch:03d} "
            f"| exit_only_acc={va_exit_acc*100:.2f}%"
            + (f" | overall@{_thrs_eval_for_reporting()} va={va_overall_acc*100:.2f}% "
               f"| exit_rates={out['exit_rates']} final_rate={out['final_rate']:.4f}"
               if out is not None else "")
        )

        metric = va_exit_acc if best_on == "exit_only_acc" else (va_overall_acc if va_overall_acc is not None else va_exit_acc)
        if metric > best["val_metric"]:
            best["val_metric"] = metric
            best["val_exit_acc"] = va_exit_acc
            best["val_overall_acc"] = va_overall_acc
            best["state"] = _snapshot_state()

    # -----------------------
    # 3) Restore best
    # -----------------------
    if best["state"] is not None:
        _restore_state(best["state"])

    return model, best

def cotrain_g1_stage_2(
    model,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs=30,
    layer_idx: int,
    exit_id: int,
    lr_layer=3e-4,
    lr_exit=3e-3,
    lambda_exit=0.3,
    use_final_loss=True,     # <-- 新增
    lambda_final=1.0,
    thrs=(1.0, 1.5),
    weight_decay=1e-3,
    grad_clip=1.0,
    exit_heads=None,          # list[ExitHead]
    payload_exit_cfg=None,    # list[dict] (for cascade eval)
    use_prob_margin=False,
):
    """
    G1 stage-wise co-train:
      - update: model.layers[layer_idx] + exit_heads[exit_id]
      - freeze: everything else (other layers, final classifier, other exit heads)
    """
    assert exit_heads is not None, "exit_heads required"
    assert payload_exit_cfg is not None, "payload_exit_cfg required for cascade eval"
    assert 0 <= layer_idx < len(model.layers)
    assert 0 <= exit_id < len(exit_heads)

    model.to(device)

    # ---- freeze backbone first ----
    set_requires_grad(model, False)

    # ---- unfreeze one backbone layer ----
    set_requires_grad(model.layers[layer_idx], True)

    # ---- freeze all exits, then unfreeze selected exit head ----
    for h in exit_heads:
        set_requires_grad(h, False)

    exit_module = exit_heads[exit_id].to(device)
    set_requires_grad(exit_module, True)

    # ---- keep final classifier frozen in g1 ----
    if hasattr(model, "classifier"):
        set_requires_grad(model.classifier, False)

    # ---- optimizer param groups ----
    params_layer = [p for p in model.layers[layer_idx].parameters() if p.requires_grad]
    params_exit  = [p for p in exit_module.parameters() if p.requires_grad]
    assert len(params_layer) > 0, "No trainable params in selected layer"
    assert len(params_exit) > 0,  "No trainable params in selected exit head"

    optimizer = torch.optim.AdamW(
        [
            {"params": params_layer, "lr": lr_layer, "weight_decay": weight_decay},
            {"params": params_exit,  "lr": lr_exit,  "weight_decay": weight_decay},
        ]
    )

    #best = {"val_overall_acc": -1.0, "state": None}
    best = {"val_metric": -1.0, "state": None}

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        exit_module.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            # 在 train loop 裡（optimizer.zero_grad 之後、backward 之前）
            
                

            
            final_logits, h_list = model.forward_with_all_hidden(xb)
            h = h_list[layer_idx]
            exit_logits = _head_logits_from_hidden(exit_module, h, device)

            loss_exit  = F.cross_entropy(exit_logits, yb)

            '''if use_final_loss:
                loss_final = F.cross_entropy(final_logits, yb)
                loss = lambda_final * loss_final + lambda_exit * loss_exit
            else:
                loss = lambda_exit * loss_exit'''
            
            # 讓 logits 不要無限放大
            loss_scale = 1e-4 * (exit_logits.pow(2).mean())   # 1e-5 ~ 1e-4 sweep
            loss = lambda_exit * loss_exit + loss_scale

            
            if use_final_loss:
                loss_final = F.cross_entropy(final_logits, yb)
                if exit_id == 1 and epoch < num_epochs // 3:
                    lam_e, lam_f = 1.0, 0.3
                else:
                    lam_e, lam_f = lambda_exit, lambda_final
                loss = lam_f * loss_final + lam_e * loss_exit
            else:
                #loss = lambda_exit * loss_exit
                loss = lambda_exit * loss_exit + loss_scale

            


            '''# final logits + all hidden (must require grad)
            final_logits, h_list = model.forward_with_all_hidden(xb)

            # exit logits from the chosen layer's hidden
            h = h_list[layer_idx]  # [B, D_layer]
            exit_logits = _head_logits_from_hidden(exit_module, h, device)  # NO no_grad!

            loss_final = F.cross_entropy(final_logits, yb)
            loss_exit  = F.cross_entropy(exit_logits, yb)
            loss = loss_final + lambda_exit * loss_exit'''

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params_layer, grad_clip)
                torch.nn.utils.clip_grad_norm_(params_exit, grad_clip)
            optimizer.step()
            

        # ---- eval: cascade overall on val ----
        model.eval()
        for h in exit_heads:
            h.eval()

        '''out = eval_cascade_multi_exit(
            model, val_loader, device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=thrs,
            use_prob_margin=use_prob_margin,
        )

        va_overall_acc = float(out["overall_acc"])
        print(
            f"[G1-stage] layer={layer_idx} exit={exit_id} Ep{epoch:03d} "
            f"| overall@{thrs} va={va_overall_acc*100:.2f} "
            f"| exit_rates={out['exit_rates']} final_rate={out['final_rate']:.4f}"
        )

        if va_overall_acc > best["val_overall_acc"]:
            best["val_overall_acc"] = va_overall_acc
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # exits 也要存！不然你只存 backbone 會漏掉 exit head 的更新
            best["exit_states"] = [ {k: v.detach().cpu().clone() for k, v in h.state_dict().items()}
                                   for h in exit_heads ]'''
        
        if exit_id == 1:
            thrs_eval = (float("inf"), thrs[1])
        else:
            thrs_eval = thrs
        # ---- eval: (1) exit-only metric (for stage-wise selection) ----
        va_exit_loss, va_exit_acc = eval_exit_only(model, exit_module, val_loader, device, layer_idx)

        # ---- eval: (2) cascade metric (for reporting) ----
        out = eval_cascade_multi_exit(
            model, val_loader, device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=thrs_eval,
            use_prob_margin=use_prob_margin,
        )
        va_overall_acc = float(out["overall_acc"])

        print(
            f"[G1-stage] layer={layer_idx} exit={exit_id} Ep{epoch:03d} "
            f"| exit_only_acc={va_exit_acc*100:.2f} "
            f"| overall@{thrs} va={va_overall_acc*100:.2f} "
            f"| exit_rates={out['exit_rates']} final_rate={out['final_rate']:.4f}"
        )

        # ---- choose best: IMPORTANT ----
        # Stage0 用 overall 也可以；Stage1 必须用 exit-only 才不会把 exit1 饿死
        '''if exit_id == 0:
            metric = va_overall_acc
        else:
            metric = va_exit_acc'''
        metric = va_exit_acc

        if metric > best["val_metric"]:
            best["val_metric"] = metric
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["exit_states"] = [{k: v.detach().cpu().clone() for k, v in h.state_dict().items()}
                           for h in exit_heads]

    # restore best
    if best.get("state") is not None:
        model.load_state_dict(best["state"], strict=False)
        for h, sd in zip(exit_heads, best["exit_states"]):
            h.load_state_dict(sd, strict=True)

    return model, best




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
        seed=42,
        z=32,
        device_for_encoding=device,
        shuffle_train=False,
    )



    #model, bb_cfg, ex_cfg, _ = load_ckpt("/Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_w_exit_g0_v1.pth", device)
    

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

    '''model, best0 = cotrain_g1_stage(
        model=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=30,
        layer_idx=0,
        exit_id=0,
        lr_layer=3e-4,
        lr_exit=3e-3,
        lambda_exit=0.3,
        use_final_loss=False,
        thrs=(1.0, 1.5),
        weight_decay=1e-3,
        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
        use_prob_margin=False,
    )
    #print("Stage0 best:", best0["val_overall_acc"])
    #print("Stage0 best:", best0["val_metric"])


    w0_before = {k: v.detach().cpu().clone() for k,v in model.layers[0].state_dict().items()}
    w_exit0_before = {k: v.detach().cpu().clone() for k,v in exit_heads[0].state_dict().items()}

    model, best1 = cotrain_g1_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=30,
        layer_idx=1,
        exit_id=1,
        lr_layer=3e-4,
        lr_exit=3e-3,
        lambda_exit=0.3,
        use_final_loss=False,
        thrs=(1.0, 1.5),
        weight_decay=1e-3,
        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
        use_prob_margin=False,
    )'''


    #print("Stage1 best:", best1["val_overall_acc"])
    #print("Stage1 best:", best1["val_metric"])

    '''# stage0
    model, best0 = cotrain_g1_stage_v3(
        model=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=100,
        layer_idx=0,
        exit_id=0,
        lr_layer=3e-4,
        lr_exit=3e-3,
        lambda_exit=1.0,
        label_smoothing=0.0,
        thrs=(1.0, 1.5),
        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
    )

    # stage1
    model, best1 = cotrain_g1_stage_v3(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=100,
        layer_idx=1,
        exit_id=1,
        lr_layer=3e-4,
        lr_exit=3e-3,
        lambda_exit=1.0,
        label_smoothing=0.0,
        thrs=(1.0, 1.5),
        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
    )'''
    model, best0 = cotrain_g1_stage_v3(
        model=backbone, train_loader=train_loader, val_loader=val_loader, device=device,
        num_epochs=60, layer_idx=0, exit_id=0,
        lr_layer=3e-4, lr_exit=3e-3,
        lambda_exit=1.0,
        use_final_loss=False,
        thrs=(1.0, 1.5),
        weight_decay=1e-3,
        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
        enable_soft_residual=False,   # ✅ stage0 不用
    )
    print("Stage0 best metric:", best0["val_metric"])


    model, best1 = cotrain_g1_stage_v3(
        model=model, train_loader=train_loader, val_loader=val_loader, device=device,
        num_epochs=60, layer_idx=1, exit_id=1,
        lr_layer=3e-4, lr_exit=3e-3,
        lambda_exit=1.0,
        #use_final_loss=False,
        thrs=(1.0, 1.5),
        weight_decay=1e-3,
        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
        enable_soft_residual=True,     # ✅ stage1 打開
        target_exit0_rate=0.70,        # exit0 約接走 70%
        gate_temp0=30.0,               # 你的 margin 很大，這個很重要

        use_final_loss=True,
        lambda_final=0.1,
    )
    print("Stage1 best metric:", best1["val_metric"])



    def max_abs_diff(sd0, sd1):
        m = 0.0
        for k in sd0.keys():
            m = max(m, (sd0[k] - sd1[k].cpu()).abs().max().item())
        return m

    '''d0 = max_abs_diff(w0_before, model.layers[0].state_dict())
    d1 = max_abs_diff(w_exit0_before, exit_heads[0].state_dict())
    print("delta layer0:", d0, "delta exit0:", d1)
    assert d0 < 1e-9 and d1 < 1e-9'''



    # 最後存成一個 ckpt：backbone_cfg 不動 + backbone weights + exit_cfg_list
    payload_exit_cfg = [ec.to_payload() for ec in exit_cfg_list]


    save_ckpt_v2(
        args.path_out,
        model,                 # backbone model
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
            model, test_loader, device,
            thr=thr,
            exit_id=0,
            exit_cfg_list=payload_exit_cfg,   # <-- 用 ExitConfig list
            exit_heads=exit_heads,
            use_prob_margin=False,
        )
        print(thr, out["exit_rate"], out["overall_acc"], out["exited_acc"], out["non_exited_acc"],
              out["margin_mean"], out["margin_p95"])
        print_eval_profile(f"G1-v2 exit0@thr={thr}", out)
    
    print('=======================================')
    thrs0 = [0.5, 1.0, 1.5, math.inf]
    thrs1 = [1.5, 2.0, 2.5, math.inf]

    for thr0 in thrs0:
        for thr1 in thrs1:
            out = eval_cascade_multi_exit(
                    model, test_loader, device,
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
            print_eval_profile(f"G1-v2 cascade@({thr0},{thr1})", out)


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

   
