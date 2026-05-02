# src/train/train_wnn.py
import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import json
from typing import List, Tuple
from networkx import sigma
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.linearExitHead import ExitHead
from src.dataio.data import build_loaders_bits
from src.dataio.mapping import make_tuple_mapping, audit_mapping
from src.exit.analyze_hidden import analyze_hidden_for_exit, compute_mu_sigma, select_exit_keep_idx
from src.exit.ckpt_exit import ExitConfig
from src.prune import *
from src.early_exit import *
from src.tools.utils import print_sweep_table
from test import *
from src.core.infer import *
from src.core.multiLayerWNN import MultiLayerWNN, load_ckpt, save_ckpt, save_ckpt_v2
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


def margin_from_logits(logits):
    top2 = logits.topk(2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def exit1_optimization_loss(
    final_logits,
    exit1_logits,
    yb,
    *,
    use_kd=True,
    kd_only_on_final_correct=False,
    use_margin=True,
    use_quota=False,
    kd_T=2.0,
    lambda_kd=0.3,
    lambda_margin_pos=0.05,
    lambda_margin_neg=0.10,
    lambda_quota=0.02,
    target_margin_correct=6.0,
    target_margin_wrong=2.0,
    target_exit_rate=0.20,
    quota_thr=6.0,
    quota_temp=1.25,
):
    loss_ce = F.cross_entropy(exit1_logits, yb)

    loss = loss_ce
    logs = {"loss_ce": float(loss_ce.detach())}

    if use_kd:
        with torch.no_grad():
            teacher_prob = F.softmax(final_logits / kd_T, dim=-1)
            final_pred = final_logits.argmax(dim=-1)
            final_correct = (final_pred == yb)

        student_log_prob = F.log_softmax(exit1_logits / kd_T, dim=-1)

        if kd_only_on_final_correct:
            if final_correct.any():
                loss_kd = F.kl_div(
                    student_log_prob[final_correct],
                    teacher_prob[final_correct],
                    reduction="batchmean"
                ) * (kd_T * kd_T)
            else:
                loss_kd = exit1_logits.new_zeros(())
        else:
            loss_kd = F.kl_div(
                student_log_prob,
                teacher_prob,
                reduction="batchmean"
            ) * (kd_T * kd_T)

        loss = loss + lambda_kd * loss_kd
        logs["loss_kd"] = float(loss_kd.detach())
        logs["final_correct_rate"] = float(final_correct.float().mean().detach())

    margin = margin_from_logits(exit1_logits)
    pred = exit1_logits.argmax(dim=-1)
    correct = (pred == yb)

    if use_margin:
        if correct.any():
            loss_margin_pos = F.relu(
                target_margin_correct - margin[correct]
            ).mean()
        else:
            loss_margin_pos = exit1_logits.new_zeros(())

        if (~correct).any():
            loss_margin_neg = F.relu(
                margin[~correct] - target_margin_wrong
            ).mean()
        else:
            loss_margin_neg = exit1_logits.new_zeros(())

        loss = (
            loss
            + lambda_margin_pos * loss_margin_pos
            + lambda_margin_neg * loss_margin_neg
        )

        logs["loss_margin_pos"] = float(loss_margin_pos.detach())
        logs["loss_margin_neg"] = float(loss_margin_neg.detach())

    if use_quota:
        take_prob = torch.sigmoid((margin - quota_thr) / quota_temp)
        soft_exit_rate = take_prob.mean()
        loss_quota = (soft_exit_rate - target_exit_rate) ** 2

        loss = loss + lambda_quota * loss_quota

        logs["loss_quota"] = float(loss_quota.detach())
        logs["soft_exit_rate"] = float(soft_exit_rate.detach())

    logs["loss_total"] = float(loss.detach())
    logs["exit1_acc"] = float(correct.float().mean().detach())
    return loss, logs

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
                    num_epochs=50, base_lr=1e-3, weight_decay=1e-4,
                    use_advanced_loss=False, loss_cfg=None):
    model.to(device)
    loss_cfg = loss_cfg or {}

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
            final_logits, exit1_logits, _ = model.forward_with_all_hidden_and_exits(xb)
            if use_advanced_loss:
                loss, _ = exit1_optimization_loss(
                    final_logits,
                    exit1_logits,
                    yb,
                    **loss_cfg,
                )
            else:
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





def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_exits(s: str) -> List[Tuple[int, int]]:
    """
    Parse: "0:256,1:512,2:256" -> [(0,256),(1,512),(2,256)]
    """
    out = []
    for item in parse_csv_list(s):
        if ":" not in item:
            raise ValueError(f"Bad --exits item: {item}, expected layer:K")
        li, k = item.split(":")
        out.append((int(li), int(k)))
    return out


def broadcast_or_match(values: List[str], n: int, name: str) -> List[str]:
    """
    If len(values)==1 -> broadcast to n
    If len(values)==n -> keep
    else -> error
    """
    if len(values) == 1:
        return values * n
    if len(values) == n:
        return values
    raise ValueError(f"--{name} expects 1 value or {n} values, got {len(values)}")


@dataclass
class ExitSpec:
    layer_idx: int
    K: int
    keep_mode: str
    exit_tau: float


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)




############################################
# V2
############################################
def _parse_list(s, cast=int):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]

def _broadcast(xs, n):
    if len(xs) == 1:
        return xs * n
    if len(xs) == n:
        return xs
    raise ValueError(f"Need 1 or {n} values, got {len(xs)}")


def build_exit_loss_cfg(
    loss_mode,
    *,
    kd_T=2.0,
    lambda_kd=0.3,
    lambda_margin_pos=0.05,
    lambda_margin_neg=0.10,
    lambda_quota=0.02,
    target_margin_correct=6.0,
    target_margin_wrong=2.0,
    target_exit_rate=0.20,
    quota_thr=6.0,
    quota_temp=1.25,
):
    mode_cfg = {
        "baseline": {
            "use_advanced_loss": False,
            "loss_cfg": {},
        },
        "kd": {
            "use_advanced_loss": True,
            "loss_cfg": {
                "use_kd": True,
                "kd_only_on_final_correct": False,
                "use_margin": False,
                "use_quota": False,
            },
        },
        "kd_final_correct": {
            "use_advanced_loss": True,
            "loss_cfg": {
                "use_kd": True,
                "kd_only_on_final_correct": True,
                "use_margin": False,
                "use_quota": False,
            },
        },
        "kd_margin": {
            "use_advanced_loss": True,
            "loss_cfg": {
                "use_kd": True,
                "kd_only_on_final_correct": False,
                "use_margin": True,
                "use_quota": False,
            },
        },
        "kd_margin_quota": {
            "use_advanced_loss": True,
            "loss_cfg": {
                "use_kd": True,
                "kd_only_on_final_correct": False,
                "use_margin": True,
                "use_quota": True,
            },
        },
    }
    if loss_mode not in mode_cfg:
        raise ValueError(f"Unsupported exit_loss_mode={loss_mode}")

    cfg = dict(mode_cfg[loss_mode])
    loss_cfg = dict(cfg["loss_cfg"])
    loss_cfg.update({
        "kd_T": kd_T,
        "lambda_kd": lambda_kd,
        "lambda_margin_pos": lambda_margin_pos,
        "lambda_margin_neg": lambda_margin_neg,
        "lambda_quota": lambda_quota,
        "target_margin_correct": target_margin_correct,
        "target_margin_wrong": target_margin_wrong,
        "target_exit_rate": target_exit_rate,
        "quota_thr": quota_thr,
        "quota_temp": quota_temp,
    })
    cfg["loss_cfg"] = loss_cfg
    return cfg


def resolve_exit_loss_setup(exit_layers):
    """
    Fixed per-exit loss setup.
    Key by layer_idx so each exit head can keep its own recipe while the
    training loop still runs one head at a time.
    """
    exit_loss_defaults = {
        "kd_T": 2.0,
        "lambda_kd": 0.3,
        "lambda_margin_pos": 0.05,
        "lambda_margin_neg": 0.10,
        "lambda_quota": 0.02,
        "target_margin_correct": 6.0,
        "target_margin_wrong": 2.0,
        "target_exit_rate": 0.20,
        "quota_thr": 6.0,
        "quota_temp": 1.25,
    }

    # Example:
    # exit_loss_by_layer = {
    #     0: {"mode": "baseline"},
    #     1: {"mode": "kd_margin_quota", "override": {"target_exit_rate": 0.15}},
    # }
    '''exit_loss_by_layer = {
        layer_idx: {"mode": "baseline", "override": {}}
        for layer_idx in exit_layers
    }'''
    exit_loss_by_layer = {
        0: {"mode": "baseline"},
        #1: {'mode': 'kd', 'override': {
        #    "kd_T": 4.0,
        #    "lambda_kd": 0.7,
        #}},
        #1: {'mode': 'kd_final_correct', 'override': {
        #    'kd_T': 2.0,
        #    'lambda_kd': 0.7,
        #}},
        1: {'mode': 'kd_margin', 'override': {
            'kd_T': 2.0,
            'lambda_kd': 0.5,
            'lambda_margin_pos': 0.1,
            'lambda_margin_neg': 0.2,
        }},
    }

    resolved = []
    for layer_idx in exit_layers:
        spec = exit_loss_by_layer.get(layer_idx, {"mode": "baseline", "override": {}})
        loss_mode = spec.get("mode", "baseline")
        override = spec.get("override", {})
        head_loss_cfg = build_exit_loss_cfg(
            loss_mode,
            **{**exit_loss_defaults, **override},
        )
        resolved.append((loss_mode, override, head_loss_cfg))

    return resolved

@torch.no_grad()
def cache_exit_features(model, loader, device, layer_idx, keep_idx, mu, sigma, use_norm: bool, return_final_logits: bool = False):
    model.eval()
    Xs, ys = [], []
    final_logits_list = []
    for xb, yb in loader:
        xb = xb.to(device)
        final_logits, h_list = model.forward_with_all_hidden(xb)
        h = h_list[layer_idx][:, keep_idx]
        if use_norm:
            h = (h - mu.to(h.device)) / sigma.to(h.device)
        Xs.append(h.detach().cpu())
        ys.append(yb.detach().cpu())
        if return_final_logits:
            final_logits_list.append(final_logits.detach().cpu())

    X = torch.cat(Xs, 0)
    y = torch.cat(ys, 0)
    if return_final_logits:
        return X, y, torch.cat(final_logits_list, 0)
    return X, y


def train_one_exit_cached(
    head,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    epochs=50,
    lr=3e-3,
    wd=1e-3,
    bs=512,
    final_logits_train=None,
    final_logits_val=None,
    use_advanced_loss=False,
    loss_cfg=None,
):
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    loss_cfg = loss_cfg or {}

    best = None
    best_val = 0.0
    N = X_train.size(0)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(N)
        tot_loss = 0.0
        corr = 0
        tot = 0
        metric_sums = {}

        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)

            opt.zero_grad()
            logits = head.classifier(xb) / head.exit_tau  # cached 已經是 [N,k]

            if use_advanced_loss:
                if final_logits_train is None:
                    raise ValueError("final_logits_train is required when use_advanced_loss=True")
                teacher_logits = final_logits_train[idx].to(device)
                loss, loss_logs = exit1_optimization_loss(
                    teacher_logits,
                    logits,
                    yb,
                    **loss_cfg,
                )
            else:
                loss = F.cross_entropy(logits, yb)
                loss_logs = {
                    "loss_total": float(loss.detach()),
                    "loss_ce": float(loss.detach()),
                    "exit1_acc": float((logits.argmax(-1) == yb).float().mean().detach()),
                }

            loss.backward()
            opt.step()

            tot_loss += loss.item() * yb.size(0)
            corr += (logits.argmax(-1) == yb).sum().item()
            tot += yb.size(0)
            for k, v in loss_logs.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v * yb.size(0)

        head.eval()
        with torch.no_grad():
            v_logits = head.classifier(X_val.to(device)) / head.exit_tau
            v_acc = (v_logits.argmax(-1).cpu() == y_val).float().mean().item()
            if use_advanced_loss:
                if final_logits_val is None:
                    raise ValueError("final_logits_val is required when use_advanced_loss=True")
                v_loss, v_logs = exit1_optimization_loss(
                    final_logits_val.to(device),
                    v_logits,
                    y_val.to(device),
                    **loss_cfg,
                )
                v_loss = float(v_loss.detach())
            else:
                v_loss = F.cross_entropy(v_logits, y_val.to(device)).item()
                v_logs = {
                    "loss_total": v_loss,
                    "loss_ce": v_loss,
                    "exit1_acc": v_acc,
                }

        train_logs = {k: v / tot for k, v in metric_sums.items()}
        train_ce = train_logs.get("loss_ce", tot_loss / tot)
        train_kd = train_logs.get("loss_kd", 0.0)
        train_margin_pos = train_logs.get("loss_margin_pos", 0.0)
        train_margin_neg = train_logs.get("loss_margin_neg", 0.0)
        train_quota = train_logs.get("loss_quota", 0.0)
        train_exit_rate = train_logs.get("soft_exit_rate", float("nan"))

        print(f"[exit layer] ep{ep:03d} train_loss={tot_loss/tot:.4f} train_acc={corr/tot*100:.2f}% "
              f"| train_ce={train_ce:.4f} train_kd={train_kd:.4f} "
              f"| train_margin+={train_margin_pos:.4f} train_margin-={train_margin_neg:.4f} "
              f"| train_quota={train_quota:.4f} train_soft_exit_rate={train_exit_rate:.4f} "
              f"| val_loss={v_loss:.4f} val_acc={v_acc*100:.2f}%")

        if v_acc > best_val:
            best_val = v_acc
            best = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best is not None:
        head.load_state_dict(best)
    return head, best_val













if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--backbone_ckpt", type=str, required=True)
    parser.add_argument("--path_out", type=str, required=True, help="Save ckpt with exit_config list")

    parser.add_argument("--exit_layers", type=str, default="0", help='e.g. "0" or "0,1"')
    parser.add_argument("--k", type=str, default="256", help='e.g. "256" or "256,512" (broadcast ok)')
    parser.add_argument("--keep_mode", type=str, default="p*(1-p)*std", help='broadcast ok')
    parser.add_argument("--exit_tau", type=str, default="1.0", help='broadcast ok')
    parser.add_argument("--thr", type=str, default="0.5", help='broadcast ok (for future online routing)')

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size_cached", type=int, default=512)
    parser.add_argument("--use_norm", action="store_true", default=True)

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

    # backbone cfg 不動：從 ckpt 讀
    model, bb_cfg, ex_cfg, extra = load_ckpt(args.backbone_ckpt, device)

    # 這支 script 是「從 backbone 建 exit heads」，不應該吃到既有 exit cfg
    if ex_cfg is not None:
        print("[warn] backbone_ckpt already contains exit_config; will ignore and rebuild exits from scratch.")

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 之後就用 bb_cfg 當 backbone_cfg（保存時原樣寫回）
    backbone_cfg = bb_cfg


    exit_layers = _parse_list(args.exit_layers, int)
    ks = _broadcast(_parse_list(args.k, int), len(exit_layers))
    keep_modes = _broadcast([x.strip() for x in args.keep_mode.split(",")], len(exit_layers))
    exit_taus = _broadcast([float(x.strip()) for x in args.exit_tau.split(",")], len(exit_layers))
    thrs = _broadcast([float(x.strip()) for x in args.thr.split(",")], len(exit_layers))
    exit_loss_specs = resolve_exit_loss_setup(exit_layers)

    exit_heads = []
    exit_cfg_list = []

    for layer_idx, k, kmode, exit_tau, thr, loss_spec in zip(
        exit_layers, ks, keep_modes, exit_taus, thrs, exit_loss_specs
    ):
        exit_loss_mode, exit_loss_override, head_loss_cfg = loss_spec
        print("\n" + "="*80)
        print(f"Build/Train exit @ layer {layer_idx} | k={k} mode={kmode} exit_tau={exit_tau} thr={thr}")
        print(f"Loss mode: {exit_loss_mode} | loss_override={exit_loss_override}")
        print("="*80)

        mean_d, std_d, p1_d, bias = analyze_hidden_for_exit(model, train_loader, device, layer_idx=layer_idx)
        exit_keep_idx = select_exit_keep_idx(mean_d, std_d, p1_d, bias, k=k, keep_mode=kmode)

        mu, sigma = compute_mu_sigma(model, train_loader, device, layer_idx=layer_idx, exit_keep_idx=exit_keep_idx)

        # cache (optional normalization)
        X_train, y_train, final_logits_train = cache_exit_features(
            model, train_loader, device, layer_idx, exit_keep_idx, mu, sigma, args.use_norm, return_final_logits=True
        )
        X_val, y_val, final_logits_val = cache_exit_features(
            model, val_loader, device, layer_idx, exit_keep_idx, mu, sigma, args.use_norm, return_final_logits=True
        )
        X_test, y_test = cache_exit_features(
            model, test_loader, device, layer_idx, exit_keep_idx, mu, sigma, args.use_norm
        )
        print(f"[cache] train {tuple(X_train.shape)} val {tuple(X_val.shape)} test {tuple(X_test.shape)}")
        print(f"[loss] mode={exit_loss_mode} advanced={head_loss_cfg['use_advanced_loss']}")

        # head from scratch (but classifier trained on cached X)
        head = ExitHead(k=k, num_classes=C, exit_tau=exit_tau,
                        exit_keep_idx=exit_keep_idx, mu=mu, sigma=sigma,
                        use_norm=args.use_norm)

        # 只訓練 classifier.weight（因為 keep_idx/mu/sigma 是 buffer）
        head, best_val = train_one_exit_cached(
            head, X_train, y_train, X_val, y_val,
            device,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.weight_decay,
            bs=args.batch_size_cached,
            final_logits_train=final_logits_train,
            final_logits_val=final_logits_val,
            use_advanced_loss=head_loss_cfg["use_advanced_loss"],
            loss_cfg=head_loss_cfg["loss_cfg"],
        )

        # 存 cfg（list item）
        exit_cfg_list.append(ExitConfig(
            layer_idx=layer_idx,
            k=k,
            keep_mode=kmode,
            thr=thr,
            exit_tau=exit_tau,
            exit_keep_idx=exit_keep_idx.cpu(),
            mu=mu.cpu(),
            sigma=sigma.cpu(),
            use_norm=args.use_norm,
        ))

        exit_heads.append(head.cpu())

        # quick test acc of this exit alone
        with torch.no_grad():
            logits = (head.classifier(X_test.to(device)) / head.exit_tau).cpu()
            acc = (logits.argmax(-1) == y_test).float().mean().item()
        print(f"[exit@layer{layer_idx}] test_exit_acc={acc*100:.2f}% | best_val={best_val*100:.2f}%")

    

    # 最後存成一個 ckpt：backbone_cfg 不動 + backbone weights + exit_cfg_list
    payload_exit_cfg = [ec.to_payload() for ec in exit_cfg_list]


    save_ckpt_v2(
        args.path_out,
        model,                 # backbone model
        exit_heads,
        backbone_cfg,          # backbone cfg 不動
        exit_cfg_list=payload_exit_cfg,  # <-- exit cfg list
        extra={"dataset": args.dataset}
    )

    print("\nSaved:", args.path_out)
    print("Exit cfg list length:", len(payload_exit_cfg))
    print('exit 0:')
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
        print_eval_profile(f"G0-v2 exit0@thr={thr}", out)

    print('exit 1:')
    thrs = [0.0, 0.5, 1.0, 2.0, 4.0]
    for thr in thrs:
        out = eval_overall_at_thr_multi_exit(
            model, test_loader, device,
            thr=thr,
            exit_id=1,
            exit_cfg_list=payload_exit_cfg,   # <-- 用 ExitConfig list
            exit_heads=exit_heads,
            use_prob_margin=False,
        )
        print(thr, out["exit_rate"], out["overall_acc"], out["exited_acc"], out["non_exited_acc"],
              out["margin_mean"], out["margin_p95"])
        print_eval_profile(f"G0-v2 exit0@thr={thr}", out)
    
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

            r0, r1 = out["exit_rates"]
            rF = out["final_rate"]

            exp_layers = 1*r0 + 2*r1 + 3*rF
            compute_ratio = exp_layers / 3.0


            print(thr0, thr1, out)
            print_eval_profile(f"G0-v2 cascade@({thr0},{thr1})", out)

    print('=======================================')
    best, dbg = sweep_cascade_by_quantile(
        model=model,
        val_loader=val_loader,
        device=device,
        exit_heads=exit_heads,
        exit_cfg_list=payload_exit_cfg
    )
    
