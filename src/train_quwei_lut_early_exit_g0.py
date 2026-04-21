import argparse
import itertools
import os
import time
from dataclasses import fields
from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from src.core.linearExitHead import ExitHead as LinearExitHead
from src.core.multiLayerWNN import save_ckpt_v2
from src.exit.ckpt_exit import ExitConfig
from src.train_quweit_lut_backbone_v2 import (
    QuWeiTViT,
    ResizeWithCIFARStats,
    TrainConfig,
    get_model_profile,
)


def _parse_csv(s: str, cast=float) -> List:
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def _broadcast(values: Sequence, n: int, name: str) -> List:
    if len(values) == 1:
        return list(values) * n
    if len(values) == n:
        return list(values)
    raise ValueError(f"--{name} expects 1 value or {n} values, got {len(values)}")


def _parse_threshold_groups(s: str, num_exits: int) -> List[List[float]]:
    if not s.strip():
        return []
    groups = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if chunk:
            groups.append(_parse_csv(chunk, float))
    return _broadcast(groups, num_exits, "cascade_thr_grid") if groups else []


def _ensure_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _progress_every(total_batches: int) -> int:
    return max(1, total_batches // 10)


def _progress_prefix(stage: str, layer_idx: int) -> str:
    return f"[progress][layer{layer_idx}][{stage}]"


def load_quweit_backbone_ckpt(path: str, device, use_ema: bool = True):
    ckpt = torch.load(path, map_location=device)
    if "config" not in ckpt or "model_state" not in ckpt:
        raise ValueError(
            "Expected a checkpoint saved by train_quweit_lut_backbone_v2.py "
            "(missing config/model_state)."
        )

    allowed = {f.name for f in fields(TrainConfig)}
    cfg_dict = {k: v for k, v in ckpt["config"].items() if k in allowed}
    cfg = TrainConfig(**cfg_dict)
    cfg.use_exit = False

    model = QuWeiTViT(cfg).to(device)
    state = ckpt.get("model_ema_state") if use_ema and ckpt.get("model_ema_state") is not None else ckpt["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[load_quweit_backbone_ckpt] missing:", missing)
    print("[load_quweit_backbone_ckpt] unexpected:", unexpected)
    return model.eval(), cfg, ckpt


def build_clean_cifar_loaders(cfg: TrainConfig, *, batch_size_train: int, batch_size_eval: int, val_ratio: float, seed: int):
    transform = ResizeWithCIFARStats(cfg.image_size)
    dataset_name = cfg.dataset.lower()
    if dataset_name == "cifar10":
        train_set = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "cifar100":
        train_set = datasets.CIFAR100(cfg.data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(cfg.data_dir, train=False, download=True, transform=transform)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset in backbone config: {cfg.dataset}")

    val_size = int(len(train_set) * val_ratio)
    train_size = len(train_set) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_clean_set, val_set = random_split(train_set, [train_size, val_size], generator=gen)

    train_clean_loader = DataLoader(train_clean_set, batch_size=batch_size_train, shuffle=False, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size_eval, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size_eval, shuffle=False, num_workers=0, drop_last=False)
    return train_clean_loader, val_loader, test_loader, num_classes


@torch.no_grad()
def forward_with_all_hidden(model: QuWeiTViT, x: torch.Tensor):
    out = model(x, return_intermediate=True)
    final_logits = out["logits"]
    h_list = [h[:, 0, :].detach() for h in out["intermediates"]]
    return final_logits, h_list


@torch.no_grad()
def analyze_hidden_for_exit(model, loader, device, layer_idx: int, thr_bin: float = 0.0):
    hs = []
    model.eval()
    total_batches = len(loader)
    started_at = time.time()
    print(f"{_progress_prefix('analyze_hidden', layer_idx)} start total_batches={total_batches}")
    progress_every = _progress_every(total_batches)
    for batch_idx, (xb, _) in enumerate(loader, start=1):
        xb = xb.to(device)
        _, h_list = forward_with_all_hidden(model, xb)
        hs.append(h_list[layer_idx - 1].cpu())
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % progress_every == 0:
            elapsed = time.time() - started_at
            print(
                f"{_progress_prefix('analyze_hidden', layer_idx)} "
                f"batch={batch_idx}/{total_batches} elapsed={elapsed:.1f}s"
            )
    h = torch.cat(hs, dim=0)
    mean_per_dim = h.mean(dim=0)
    std_per_dim = h.std(dim=0)
    p1_per_dim = (h > thr_bin).float().mean(dim=0)
    bias = (p1_per_dim - 0.5).abs()
    print(
        f"{_progress_prefix('analyze_hidden', layer_idx)} done "
        f"samples={h.shape[0]} dim={h.shape[1]} elapsed={time.time() - started_at:.1f}s"
    )
    return mean_per_dim, std_per_dim, p1_per_dim, bias


def select_exit_keep_idx(mean_per_dim, std_per_dim, p1_per_dim, bias, k: int, keep_mode: str):
    if keep_mode == "bias":
        score = bias
    elif keep_mode == "bias*std":
        score = bias * std_per_dim
    elif keep_mode == "p*(1-p)*std":
        score = (p1_per_dim * (1 - p1_per_dim)) * std_per_dim
    else:
        raise ValueError(f"Unknown keep_mode={keep_mode}")
    return torch.topk(score, k=k).indices


@torch.no_grad()
def compute_mu_sigma(model, loader, device, layer_idx: int, exit_keep_idx: torch.Tensor):
    hs = []
    model.eval()
    total_batches = len(loader)
    started_at = time.time()
    print(
        f"{_progress_prefix('compute_mu_sigma', layer_idx)} start "
        f"total_batches={total_batches} k={exit_keep_idx.numel()}"
    )
    progress_every = _progress_every(total_batches)
    keep_idx_device = exit_keep_idx.to(device)
    for batch_idx, (xb, _) in enumerate(loader, start=1):
        xb = xb.to(device)
        _, h_list = forward_with_all_hidden(model, xb)
        hs.append(h_list[layer_idx - 1][:, keep_idx_device].cpu())
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % progress_every == 0:
            elapsed = time.time() - started_at
            print(
                f"{_progress_prefix('compute_mu_sigma', layer_idx)} "
                f"batch={batch_idx}/{total_batches} elapsed={elapsed:.1f}s"
            )
    h = torch.cat(hs, dim=0)
    print(
        f"{_progress_prefix('compute_mu_sigma', layer_idx)} done "
        f"samples={h.shape[0]} k={h.shape[1]} elapsed={time.time() - started_at:.1f}s"
    )
    return h.mean(dim=0), h.std(dim=0).clamp_min(1e-6)


@torch.no_grad()
def cache_exit_features(model, loader, device, layer_idx, keep_idx, mu, sigma, use_norm: bool):
    xs = []
    ys = []
    model.eval()
    total_batches = len(loader)
    started_at = time.time()
    print(
        f"{_progress_prefix('cache_features', layer_idx)} start "
        f"total_batches={total_batches} use_norm={use_norm}"
    )
    progress_every = _progress_every(total_batches)
    keep_idx_device = keep_idx.to(device)
    mu_device = mu.to(device)
    sigma_device = sigma.to(device)
    for batch_idx, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device)
        _, h_list = forward_with_all_hidden(model, xb)
        h = h_list[layer_idx - 1][:, keep_idx_device]
        if use_norm:
            h = (h - mu_device) / sigma_device
        xs.append(h.cpu())
        ys.append(yb.cpu())
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % progress_every == 0:
            elapsed = time.time() - started_at
            print(
                f"{_progress_prefix('cache_features', layer_idx)} "
                f"batch={batch_idx}/{total_batches} elapsed={elapsed:.1f}s"
            )
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    print(
        f"{_progress_prefix('cache_features', layer_idx)} done "
        f"x_shape={tuple(x.shape)} y_shape={tuple(y.shape)} elapsed={time.time() - started_at:.1f}s"
    )
    return x, y


def train_one_exit_cached(
    head,
    x_train,
    y_train,
    x_val,
    y_val,
    device,
    *,
    epochs=20,
    lr=3e-3,
    wd=1e-4,
    batch_size=512,
    patience=15,
):
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    best_state = None
    best_val_acc = -1.0
    epochs_since_improve = 0
    n = x_train.size(0)

    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        total = 0
        correct = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head.classifier(xb) / head.exit_tau
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(dim=-1) == yb).sum().item()
            total += yb.size(0)

        head.eval()
        with torch.no_grad():
            val_logits = head.classifier(x_val.to(device)) / head.exit_tau
            val_loss = F.cross_entropy(val_logits, y_val.to(device)).item()
            val_acc = (val_logits.argmax(dim=-1).cpu() == y_val).float().mean().item()

        print(
            f"[exit-train] epoch={epoch:03d} "
            f"train_loss={total_loss / max(total, 1):.4f} "
            f"train_acc={correct / max(total, 1) * 100:.2f}% "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc * 100:.2f}%"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(
                    f"[exit-train] early stop at epoch={epoch:03d} "
                    f"(no val_acc improvement for {patience} epochs)"
                )
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head.cpu(), best_val_acc


@torch.no_grad()
def evaluate_cached_exit_head(head, x, y, device):
    head = head.to(device).eval()
    logits = head.classifier(x.to(device)) / head.exit_tau
    acc = (logits.argmax(dim=-1).cpu() == y).float().mean().item()
    top2 = torch.topk(logits, k=2, dim=-1).values
    margins = (top2[:, 0] - top2[:, 1]).detach().cpu()
    return {
        "acc": acc,
        "margin_mean": float(margins.mean().item()),
        "margin_p95": float(torch.quantile(margins, 0.95).item()),
    }


def _count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters()) if module is not None else 0


def _linear_flops_and_macs(linear: torch.nn.Linear):
    macs = float(linear.in_features * linear.out_features)
    bias_flops = float(linear.out_features if linear.bias is not None else 0.0)
    flops = 2.0 * macs + bias_flops
    return flops, macs


def get_external_exit_profile(model: QuWeiTViT, exit_heads: List[torch.nn.Module], exit_cfg_list: List[dict]):
    profile = get_model_profile(model)
    exit_profiles = []
    total_exit_head_params = 0

    for head, cfg in zip(exit_heads, exit_cfg_list):
        cls_flops, cls_macs = _linear_flops_and_macs(head.classifier)
        norm_flops = float(2 * head.k) if getattr(head, "use_norm", False) else 0.0
        head_flops = norm_flops + cls_flops
        head_macs = cls_macs
        head_params = _count_parameters(head)
        total_exit_head_params += head_params
        exit_profiles.append(
            {
                "layer_idx": int(cfg["layer_idx"]),
                "flops": head_flops,
                "macs": head_macs,
                "params": head_params,
            }
        )

    profile["exit_heads"] = exit_profiles
    profile["total_exit_head_params"] = float(total_exit_head_params)
    profile["param_overhead_ratio"] = (
        float(total_exit_head_params) / float(profile["backbone_params"])
        if float(profile["backbone_params"]) > 0 else float("nan")
    )
    return profile


@torch.no_grad()
def eval_overall_at_thr(model, loader, device, thr: float, *, exit_id: int, exit_cfg_list: List[dict], exit_heads: List[torch.nn.Module]):
    profile = get_external_exit_profile(model, exit_heads, exit_cfg_list)
    cfg = exit_cfg_list[exit_id]
    layer_idx = int(cfg["layer_idx"])
    exit_tau = float(cfg.get("exit_tau", 1.0))
    head = exit_heads[exit_id].to(device).eval()
    exit_profile = profile["exit_heads"][exit_id]

    total = 0
    correct_overall = 0
    exited = 0
    correct_exited = 0
    non_exited = 0
    correct_non_exited = 0
    margins = []
    total_flops = 0.0
    total_macs = 0.0
    total_layers = 0.0

    model.eval()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        final_logits, h_list = forward_with_all_hidden(model, xb)
        h = h_list[layer_idx - 1]
        exit_logits = head(h) / exit_tau
        top2 = torch.topk(exit_logits, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        exit_mask = margin > thr
        margins.append(margin.cpu())

        mixed = final_logits.clone()
        mixed[exit_mask] = exit_logits[exit_mask]
        pred = mixed.argmax(dim=-1)
        correct_overall += (pred == yb).sum().item()
        total += yb.numel()

        if exit_mask.any():
            exited += int(exit_mask.sum().item())
            correct_exited += (exit_logits.argmax(dim=-1)[exit_mask] == yb[exit_mask]).sum().item()
        if (~exit_mask).any():
            non_exited += int((~exit_mask).sum().item())
            correct_non_exited += (final_logits.argmax(dim=-1)[~exit_mask] == yb[~exit_mask]).sum().item()

        batch_size = yb.numel()
        exited_batch = float(exit_mask.sum().item())
        non_exited_batch = float(batch_size - exited_batch)
        prefix_flops = profile["patch_embed"]["flops"] + sum(layer["flops"] for layer in profile["layers"][:layer_idx])
        prefix_macs = profile["patch_embed"]["macs"] + sum(layer["macs"] for layer in profile["layers"][:layer_idx])
        tail_flops = sum(layer["flops"] for layer in profile["layers"][layer_idx:]) + profile["final_head"]["flops"]
        tail_macs = sum(layer["macs"] for layer in profile["layers"][layer_idx:]) + profile["final_head"]["macs"]
        total_flops += batch_size * (prefix_flops + exit_profile["flops"]) + non_exited_batch * tail_flops
        total_macs += batch_size * (prefix_macs + exit_profile["macs"]) + non_exited_batch * tail_macs
        total_layers += batch_size * float(layer_idx) + non_exited_batch * float(profile["num_backbone_layers"] - layer_idx)

    margins = torch.cat(margins, dim=0)
    avg_flops_per_sample = total_flops / max(total, 1)
    avg_macs_per_sample = total_macs / max(total, 1)
    avg_layers_executed_per_sample = total_layers / max(total, 1)
    backbone_full_flops = float(profile["backbone_full_flops"])
    compute_overhead_ratio = (avg_flops_per_sample / backbone_full_flops) if backbone_full_flops > 0 else float("nan")
    return {
        "overall_acc": correct_overall / max(total, 1),
        "exit_rate": exited / max(total, 1),
        "exited_acc": correct_exited / max(exited, 1),
        "non_exited_acc": (correct_non_exited / non_exited) if non_exited > 0 else float("nan"),
        "margin_mean": float(margins.mean().item()),
        "margin_p95": float(torch.quantile(margins, 0.95).item()),
        "exited": exited,
        "non_exited": non_exited,
        "total": total,
        "avg_flops_per_sample": avg_flops_per_sample,
        "avg_macs_per_sample": avg_macs_per_sample,
        "avg_layers_executed_per_sample": avg_layers_executed_per_sample,
        "backbone_params": float(profile["backbone_params"]),
        "total_exit_head_params": float(profile["total_exit_head_params"]),
        "param_overhead_ratio": float(profile["param_overhead_ratio"]),
        "compute_overhead_ratio": compute_overhead_ratio,
        "compute_saving_ratio": 1.0 - compute_overhead_ratio if compute_overhead_ratio == compute_overhead_ratio else float("nan"),
    }


@torch.no_grad()
def eval_cascade(model, loader, device, *, exit_heads: List[torch.nn.Module], exit_cfg_list: List[dict], thrs: List[float]):
    profile = get_external_exit_profile(model, exit_heads, exit_cfg_list)
    total = 0
    correct = 0
    num_exits = len(exit_heads)
    n_exit = [0] * num_exits
    c_exit = [0] * num_exits
    n_final = 0
    c_final = 0
    margin_stats = []
    total_flops = 0.0
    total_macs = 0.0
    total_layers = 0.0

    model.eval()
    for h in exit_heads:
        h.eval()

    all_taken = [[] for _ in range(num_exits)]
    all_undecided = [[] for _ in range(num_exits)]

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        final_logits, h_list = forward_with_all_hidden(model, xb)
        bsz = yb.size(0)
        undecided = torch.ones(bsz, dtype=torch.bool, device=device)
        preds = torch.empty(bsz, dtype=torch.long, device=device)

        for i in range(num_exits):
            if not undecided.any():
                break
            cfg = exit_cfg_list[i]
            head = exit_heads[i].to(device)
            logits_i = head(h_list[int(cfg["layer_idx"]) - 1]) / float(cfg.get("exit_tau", 1.0))
            top2 = torch.topk(logits_i, k=2, dim=-1).values
            margins = top2[:, 0] - top2[:, 1]
            all_undecided[i].append(margins[undecided].detach().cpu())

            take_i = undecided & (margins > float(thrs[i]))
            if take_i.any():
                preds[take_i] = logits_i.argmax(dim=-1)[take_i]
                n_exit[i] += int(take_i.sum().item())
                c_exit[i] += (preds[take_i] == yb[take_i]).sum().item()
                all_taken[i].append(margins[take_i].detach().cpu())
                undecided = undecided & (~take_i)

        if undecided.any():
            preds[undecided] = final_logits.argmax(dim=-1)[undecided]
            n_final += int(undecided.sum().item())
            c_final += (preds[undecided] == yb[undecided]).sum().item()

        correct += (preds == yb).sum().item()
        total += bsz

        route_taken = torch.full((bsz,), fill_value=num_exits, dtype=torch.long, device=device)
        undecided = torch.ones(bsz, dtype=torch.bool, device=device)
        for i in range(num_exits):
            cfg = exit_cfg_list[i]
            head = exit_heads[i].to(device)
            logits_i = head(h_list[int(cfg["layer_idx"]) - 1]) / float(cfg.get("exit_tau", 1.0))
            top2 = torch.topk(logits_i, k=2, dim=-1).values
            margins = top2[:, 0] - top2[:, 1]
            take_i = undecided & (margins > float(thrs[i]))
            route_taken[take_i] = i
            undecided = undecided & (~take_i)

        for route in range(num_exits):
            count = float((route_taken == route).sum().item())
            if count == 0:
                continue
            layer_idx = int(profile["exit_heads"][route]["layer_idx"])
            flops = profile["patch_embed"]["flops"] + sum(layer["flops"] for layer in profile["layers"][:layer_idx])
            macs = profile["patch_embed"]["macs"] + sum(layer["macs"] for layer in profile["layers"][:layer_idx])
            flops += sum(h["flops"] for h in profile["exit_heads"][: route + 1])
            macs += sum(h["macs"] for h in profile["exit_heads"][: route + 1])
            total_flops += count * flops
            total_macs += count * macs
            total_layers += count * float(layer_idx)

        final_count = float((route_taken == num_exits).sum().item())
        if final_count > 0:
            final_route_flops = profile["backbone_full_flops"] + sum(h["flops"] for h in profile["exit_heads"])
            final_route_macs = profile["backbone_full_macs"] + sum(h["macs"] for h in profile["exit_heads"])
            total_flops += final_count * final_route_flops
            total_macs += final_count * final_route_macs
            total_layers += final_count * float(profile["num_backbone_layers"])

    for i in range(num_exits):
        mu_u = float(torch.cat(all_undecided[i]).mean().item()) if all_undecided[i] else float("nan")
        p95_u = float(torch.quantile(torch.cat(all_undecided[i]), 0.95).item()) if all_undecided[i] else float("nan")
        mu_t = float(torch.cat(all_taken[i]).mean().item()) if all_taken[i] else float("nan")
        p95_t = float(torch.quantile(torch.cat(all_taken[i]), 0.95).item()) if all_taken[i] else float("nan")
        margin_stats.append({
            "undecided_mean": mu_u,
            "undecided_p95": p95_u,
            "taken_mean": mu_t,
            "taken_p95": p95_t,
        })

    avg_flops_per_sample = total_flops / max(total, 1)
    avg_macs_per_sample = total_macs / max(total, 1)
    avg_layers_executed_per_sample = total_layers / max(total, 1)
    backbone_full_flops = float(profile["backbone_full_flops"])
    compute_overhead_ratio = (avg_flops_per_sample / backbone_full_flops) if backbone_full_flops > 0 else float("nan")

    return {
        "overall_acc": correct / max(total, 1),
        "total": total,
        "exit_rates": [n / max(total, 1) for n in n_exit],
        "exit_accs": [(c / n) if n > 0 else float("nan") for c, n in zip(c_exit, n_exit)],
        "final_rate": n_final / max(total, 1),
        "final_acc": (c_final / n_final) if n_final > 0 else float("nan"),
        "margin_stats": margin_stats,
        "avg_flops_per_sample": avg_flops_per_sample,
        "avg_macs_per_sample": avg_macs_per_sample,
        "avg_layers_executed_per_sample": avg_layers_executed_per_sample,
        "backbone_params": float(profile["backbone_params"]),
        "total_exit_head_params": float(profile["total_exit_head_params"]),
        "param_overhead_ratio": float(profile["param_overhead_ratio"]),
        "compute_overhead_ratio": compute_overhead_ratio,
        "compute_saving_ratio": 1.0 - compute_overhead_ratio if compute_overhead_ratio == compute_overhead_ratio else float("nan"),
    }


def print_single_exit_sweep(title: str, rows: List[dict]):
    print(f"\n=== {title} ===")
    header = (
        "thr    exit%   overall%  exit_acc%  non_exit_acc%  m_mean  m_p95  "
        "avgFLOPs   avgMACs avgLayers bbParams exitParams overhead"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        non_exit_acc = row["non_exited_acc"] * 100.0 if row["non_exited_acc"] == row["non_exited_acc"] else float("nan")
        print(
            f"{row['thr']:>4.2f}  "
            f"{row['exit_rate'] * 100:>6.2f}  "
            f"{row['overall_acc'] * 100:>8.2f}  "
            f"{row['exited_acc'] * 100:>9.2f}  "
            f"{non_exit_acc:>13.2f}  "
            f"{row['margin_mean']:>6.2f}  "
            f"{row['margin_p95']:>6.2f}  "
            f"{row['avg_flops_per_sample']:>9.0f} "
            f"{row['avg_macs_per_sample']:>9.0f} "
            f"{row['avg_layers_executed_per_sample']:>9.3f} "
            f"{int(row['backbone_params']):>8d} "
            f"{int(row['total_exit_head_params']):>10d} "
            f"{row['compute_overhead_ratio']:>8.4f}"
        )


def print_cascade_sweep(title: str, rows: List[dict]):
    print(f"\n=== {title} ===")
    for row in rows:
        thr_text = ",".join(f"{thr:.2f}" for thr in row["thrs"])
        exit_rate_text = ",".join(f"{r * 100:.2f}" for r in row["exit_rates"])
        exit_acc_text = ",".join(f"{a * 100:.2f}" if a == a else "nan" for a in row["exit_accs"])
        final_acc = row["final_acc"] * 100.0 if row["final_acc"] == row["final_acc"] else float("nan")
        print(
            f"thrs=[{thr_text}] "
            f"overall={row['overall_acc'] * 100:.2f}% "
            f"exit_rates=[{exit_rate_text}] "
            f"final_rate={row['final_rate'] * 100:.2f}% "
            f"exit_accs=[{exit_acc_text}] "
            f"final_acc={final_acc:.2f}% "
            f"avgFLOPs={row['avg_flops_per_sample']:.0f} "
            f"avgMACs={row['avg_macs_per_sample']:.0f} "
            f"avgLayers={row['avg_layers_executed_per_sample']:.3f} "
            f"bbParams={int(row['backbone_params'])} "
            f"exitParams={int(row['total_exit_head_params'])} "
            f"overhead={row['compute_overhead_ratio']:.4f} "
            f"margin_stats={row['margin_stats']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Train cached external early-exit heads from train_quweit_lut_backbone_v2.py checkpoints.")
    parser.add_argument("--backbone_ckpt", type=str, required=True, help="Checkpoint produced by train_quweit_lut_backbone_v2.py")
    parser.add_argument("--path_out", type=str, required=True)
    parser.add_argument("--use_ema_backbone", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--exit_layers", type=str, default="2,4,6,8")
    parser.add_argument("--k", type=str, default="256")
    parser.add_argument("--keep_mode", type=str, default="p*(1-p)*std")
    parser.add_argument("--exit_tau", type=str, default="1.0")
    parser.add_argument("--init_thr", type=str, default="0.5")
    parser.add_argument("--use_norm", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size_train", type=int, default=128)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--batch_size_cached", type=int, default=512)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--single_thr_list", type=str, default="0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0")
    parser.add_argument("--cascade_thr_grid", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model, backbone_cfg, raw_ckpt = load_quweit_backbone_ckpt(args.backbone_ckpt, device, use_ema=args.use_ema_backbone)
    print("[info] building clean CIFAR loaders matched to train_quweit_lut_backbone_v2.py preprocessing")
    train_clean_loader, val_loader, test_loader, num_classes = build_clean_cifar_loaders(
        backbone_cfg,
        batch_size_train=args.batch_size_train,
        batch_size_eval=args.batch_size_eval,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    exit_layers = _parse_csv(args.exit_layers, int)
    bad_layers = [layer_idx for layer_idx in exit_layers if layer_idx < 1 or layer_idx > backbone_cfg.depth]
    if bad_layers:
        raise ValueError(f"Invalid --exit_layers {bad_layers}; QuWeiT depth is {backbone_cfg.depth} and layers are 1-based.")

    ks = _broadcast(_parse_csv(args.k, int), len(exit_layers), "k")
    keep_modes = _broadcast([x.strip() for x in args.keep_mode.split(",") if x.strip()], len(exit_layers), "keep_mode")
    exit_taus = _broadcast(_parse_csv(args.exit_tau, float), len(exit_layers), "exit_tau")
    init_thrs = _broadcast(_parse_csv(args.init_thr, float), len(exit_layers), "init_thr")
    single_thr_list = _parse_csv(args.single_thr_list, float)
    cascade_thr_grid = _parse_threshold_groups(args.cascade_thr_grid, len(exit_layers))

    exit_heads = []
    exit_cfg_list = []

    for layer_idx, k, keep_mode, exit_tau, init_thr in zip(exit_layers, ks, keep_modes, exit_taus, init_thrs):
        print("\n" + "=" * 88)
        print(f"build/train exit layer={layer_idx} k={k} keep_mode={keep_mode} exit_tau={exit_tau} init_thr={init_thr}")
        print("=" * 88)

        mean_d, std_d, p1_d, bias = analyze_hidden_for_exit(model, train_clean_loader, device, layer_idx=layer_idx)
        exit_keep_idx = select_exit_keep_idx(mean_d, std_d, p1_d, bias, k=k, keep_mode=keep_mode)
        mu, sigma = compute_mu_sigma(model, train_clean_loader, device, layer_idx=layer_idx, exit_keep_idx=exit_keep_idx)

        x_train, y_train = cache_exit_features(model, train_clean_loader, device, layer_idx, exit_keep_idx, mu, sigma, args.use_norm)
        x_val, y_val = cache_exit_features(model, val_loader, device, layer_idx, exit_keep_idx, mu, sigma, args.use_norm)
        x_test, y_test = cache_exit_features(model, test_loader, device, layer_idx, exit_keep_idx, mu, sigma, args.use_norm)
        print(f"[cache] train={tuple(x_train.shape)} val={tuple(x_val.shape)} test={tuple(x_test.shape)}")

        head = LinearExitHead(
            k=k,
            num_classes=num_classes,
            exit_tau=exit_tau,
            exit_keep_idx=exit_keep_idx.cpu(),
            mu=mu.cpu(),
            sigma=sigma.cpu(),
            use_norm=args.use_norm,
        )
        head, best_val = train_one_exit_cached(
            head,
            x_train,
            y_train,
            x_val,
            y_val,
            device,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.weight_decay,
            batch_size=args.batch_size_cached,
            patience=args.patience,
        )

        val_metrics = evaluate_cached_exit_head(head, x_val, y_val, device)
        test_metrics = evaluate_cached_exit_head(head, x_test, y_test, device)
        print(
            f"[exit@layer{layer_idx}] "
            f"best_val={best_val * 100:.2f}% "
            f"val_acc={val_metrics['acc'] * 100:.2f}% "
            f"val_margin_mean={val_metrics['margin_mean']:.2f} "
            f"val_margin_p95={val_metrics['margin_p95']:.2f} "
            f"test_acc={test_metrics['acc'] * 100:.2f}% "
            f"test_margin_mean={test_metrics['margin_mean']:.2f} "
            f"test_margin_p95={test_metrics['margin_p95']:.2f}"
        )

        exit_cfg_list.append(
            ExitConfig(
                layer_idx=layer_idx,
                k=k,
                keep_mode=keep_mode,
                thr=init_thr,
                exit_tau=exit_tau,
                exit_keep_idx=exit_keep_idx.cpu(),
                mu=mu.cpu(),
                sigma=sigma.cpu(),
                use_norm=args.use_norm,
            )
        )
        exit_heads.append(head.cpu())

    payload_exit_cfg = [cfg.to_payload() for cfg in exit_cfg_list]
    _ensure_dir(args.path_out)
    save_ckpt_v2(
        args.path_out,
        model,
        exit_heads,
        {
            "source": "train_quweit_lut_backbone_v2.py",
            "backbone_ckpt": args.backbone_ckpt,
            "config": raw_ckpt["config"],
        },
        exit_cfg_list=payload_exit_cfg,
        extra={
            "dataset": backbone_cfg.dataset,
            "train_mode": "clean_cached_exit_training",
            "note": "Hidden analysis, mu/sigma, cached feature training, and sweeps all use clean CIFAR transforms.",
        },
    )
    print(f"\n[saved] {args.path_out}")

    for exit_id, layer_idx in enumerate(exit_layers):
        rows_val = []
        rows_test = []
        for thr in single_thr_list:
            rows_val.append({
                "thr": thr,
                **eval_overall_at_thr(model, val_loader, device, thr, exit_id=exit_id, exit_cfg_list=payload_exit_cfg, exit_heads=exit_heads),
            })
            rows_test.append({
                "thr": thr,
                **eval_overall_at_thr(model, test_loader, device, thr, exit_id=exit_id, exit_cfg_list=payload_exit_cfg, exit_heads=exit_heads),
            })
        print_single_exit_sweep(f"VAL single-exit sweep @ layer {layer_idx}", rows_val)
        print_single_exit_sweep(f"TEST single-exit sweep @ layer {layer_idx}", rows_test)

    if cascade_thr_grid:
        rows_val = []
        rows_test = []
        for thrs in itertools.product(*cascade_thr_grid):
            thrs = list(thrs)
            rows_val.append({"thrs": thrs, **eval_cascade(model, val_loader, device, exit_heads=exit_heads, exit_cfg_list=payload_exit_cfg, thrs=thrs)})
            rows_test.append({"thrs": thrs, **eval_cascade(model, test_loader, device, exit_heads=exit_heads, exit_cfg_list=payload_exit_cfg, thrs=thrs)})
        rows_val.sort(key=lambda row: row["overall_acc"], reverse=True)
        rows_test.sort(key=lambda row: row["overall_acc"], reverse=True)
        print_cascade_sweep("VAL cascade sweep", rows_val)
        print_cascade_sweep("TEST cascade sweep", rows_test)


if __name__ == "__main__":
    main()
