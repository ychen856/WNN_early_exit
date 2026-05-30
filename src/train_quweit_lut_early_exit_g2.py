import argparse
import copy
import itertools
import os
from dataclasses import fields
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from src.core.linearExitHead import build_exit_heads_from_cfg
from src.core.multiLayerWNN import save_ckpt_v2
from src.early_exit import _head_logits_from_hidden_trainable, _margin_from_logits
from src.exit.ckpt_exit import ExitConfig
from src.train_quweit_lut_backbone_v2 import QuWeiTViT, TrainConfig
from src.train_quweit_lut_early_exit_g0_ce import build_clean_cifar_loaders, get_external_exit_profile


def _parse_csv(s: str, cast=float) -> List:
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def _broadcast(values: Sequence, n: int, name: str) -> List:
    if len(values) == 1:
        return list(values) * n
    if len(values) == n:
        return list(values)
    raise ValueError(f"--{name} expects 1 value or {n} values, got {len(values)}")


def _ensure_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _parse_threshold_groups(s: str, num_exits: int, name: str) -> List[List[float]]:
    if not s.strip():
        return []
    groups = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if chunk:
            groups.append(_parse_csv(chunk, float))
    return _broadcast(groups, num_exits, name) if groups else []


def _cfg_from_payload(cfg_payload: dict) -> TrainConfig:
    allowed = {f.name for f in fields(TrainConfig)}
    cfg_dict = {k: v for k, v in cfg_payload.items() if k in allowed}
    cfg = TrainConfig(**cfg_dict)
    cfg.use_exit = False
    return cfg


def load_quweit_model_with_exits(path: str, device):
    ckpt = torch.load(path, map_location=device)
    if "model_state_dict" not in ckpt or "exit_cfg" not in ckpt or "exits_state_dict" not in ckpt:
        raise ValueError("Expected a save_ckpt_v2 checkpoint containing model_state_dict / exit_cfg / exits_state_dict.")

    backbone_cfg_payload = ckpt.get("backbone_cfg", {})
    cfg_payload = backbone_cfg_payload["config"] if isinstance(backbone_cfg_payload, dict) and "config" in backbone_cfg_payload else backbone_cfg_payload
    cfg = _cfg_from_payload(cfg_payload)

    model = QuWeiTViT(cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("[load_quweit_model_with_exits] backbone missing:", missing)
    print("[load_quweit_model_with_exits] backbone unexpected:", unexpected)

    exit_cfg_list = [ExitConfig.from_payload(item) for item in ckpt["exit_cfg"]]
    exit_heads = build_exit_heads_from_cfg(exit_cfg_list, num_classes=cfg.num_classes, device=device)
    exits_state_dict = ckpt["exits_state_dict"]
    if len(exits_state_dict) != len(exit_heads):
        raise ValueError("Checkpoint exit heads/state length mismatch.")
    for head, sd in zip(exit_heads, exits_state_dict):
        head.load_state_dict(sd, strict=True)

    return model.eval(), cfg, ckpt, exit_heads, exit_cfg_list


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def forward_with_all_hidden(model: QuWeiTViT, x: torch.Tensor):
    out = model(x, return_intermediate=True)
    final_logits = out["logits"]
    h_list = [h[:, 0, :] for h in out["intermediates"]]
    return final_logits, h_list


@torch.no_grad()
def eval_cascade_quweit(
    model,
    loader,
    device,
    *,
    exit_heads: List[torch.nn.Module],
    exit_cfg_list: List[dict],
    thrs: Sequence[float],
    use_prob_margin: bool = False,
):
    assert len(exit_heads) == len(exit_cfg_list)
    assert len(thrs) == len(exit_heads)

    model.eval()
    exit_heads = [head.to(device).eval() for head in exit_heads]

    total = 0
    correct = 0
    num_exits = len(exit_heads)
    n_exit = [0] * num_exits
    c_exit = [0] * num_exits
    n_final = 0
    c_final = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        final_logits, h_list = forward_with_all_hidden(model, xb)

        bsz = yb.size(0)
        undecided = torch.ones(bsz, dtype=torch.bool, device=device)
        preds = torch.empty(bsz, dtype=torch.long, device=device)

        for exit_id, (head, cfg, thr) in enumerate(zip(exit_heads, exit_cfg_list, thrs)):
            if not undecided.any():
                break

            layer_idx_1based = int(cfg["layer_idx"])
            logits = head(h_list[layer_idx_1based - 1])
            if use_prob_margin:
                top2 = torch.topk(torch.softmax(logits, dim=-1), k=2, dim=-1).values
            else:
                top2 = torch.topk(logits, k=2, dim=-1).values
            margins = top2[:, 0] - top2[:, 1]

            take = undecided & (margins > float(thr))
            if take.any():
                preds[take] = logits[take].argmax(dim=-1)
                n_exit[exit_id] += int(take.sum().item())
                c_exit[exit_id] += int((preds[take] == yb[take]).sum().item())
                undecided = undecided & (~take)

        if undecided.any():
            preds[undecided] = final_logits[undecided].argmax(dim=-1)
            n_final += int(undecided.sum().item())
            c_final += int((preds[undecided] == yb[undecided]).sum().item())

        correct += int((preds == yb).sum().item())
        total += bsz

    return {
        "overall_acc": correct / max(total, 1),
        "exit_rates": [n / max(total, 1) for n in n_exit],
        "exit_accs": [(c / n) if n > 0 else float("nan") for c, n in zip(c_exit, n_exit)],
        "final_rate": n_final / max(total, 1),
        "final_acc": (c_final / n_final) if n_final > 0 else float("nan"),
    }


@torch.no_grad()
def collect_cascade_cache_quweit(
    model,
    loader,
    device,
    *,
    exit_heads: List[torch.nn.Module],
    exit_cfg_list: List[dict],
    use_prob_margin: bool,
    profile: dict,
):
    num_exits = len(exit_heads)
    model.eval()
    exit_heads = [head.to(device).eval() for head in exit_heads]
    labels_parts = []
    final_pred_parts = []
    exit_pred_parts = [[] for _ in range(num_exits)]
    margins_per_exit = [[] for _ in range(num_exits)]

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        final_logits, h_list = forward_with_all_hidden(model, xb)
        labels_parts.append(yb.cpu())
        final_pred_parts.append(final_logits.argmax(dim=-1).cpu())
        for exit_id in range(num_exits):
            layer_idx = int(exit_cfg_list[exit_id]["layer_idx"])
            logits = exit_heads[exit_id](h_list[layer_idx - 1])
            exit_pred_parts[exit_id].append(logits.argmax(dim=-1).cpu())
            if use_prob_margin:
                top2 = torch.topk(torch.softmax(logits, dim=-1), k=2, dim=-1).values
            else:
                top2 = torch.topk(logits, k=2, dim=-1).values
            margins = top2[:, 0] - top2[:, 1]
            margins_per_exit[exit_id].append(margins.detach().cpu())

    return {
        "profile": profile,
        "labels": torch.cat(labels_parts, dim=0) if labels_parts else torch.empty(0, dtype=torch.long),
        "final_pred": torch.cat(final_pred_parts, dim=0) if final_pred_parts else torch.empty(0, dtype=torch.long),
        "exit_pred": [torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.long) for parts in exit_pred_parts],
        "margins": [torch.cat(parts, dim=0) if parts else torch.empty(0) for parts in margins_per_exit],
    }


def eval_cascade_cached_quweit(cache: dict, thrs: Sequence[float]):
    profile = cache["profile"]
    labels = cache["labels"]
    preds = cache["final_pred"].clone()
    exit_preds = cache["exit_pred"]
    margins = cache["margins"]
    num_exits = len(exit_preds)
    total = int(labels.numel())
    undecided = torch.ones(total, dtype=torch.bool)
    route_taken = torch.full((total,), fill_value=num_exits, dtype=torch.long)
    n_exit = [0] * num_exits
    c_exit = [0] * num_exits

    for exit_id in range(num_exits):
        take = undecided & (margins[exit_id] > float(thrs[exit_id]))
        if take.any():
            preds[take] = exit_preds[exit_id][take]
            n_exit[exit_id] = int(take.sum().item())
            c_exit[exit_id] = int((exit_preds[exit_id][take] == labels[take]).sum().item())
            route_taken[take] = exit_id
            undecided = undecided & (~take)

    n_final = int(undecided.sum().item())
    c_final = int((preds[undecided] == labels[undecided]).sum().item()) if n_final > 0 else 0
    correct = int((preds == labels).sum().item())
    total_flops = 0.0
    total_macs = 0.0
    total_layers = 0.0
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

    avg_flops_per_sample = total_flops / max(total, 1)
    avg_macs_per_sample = total_macs / max(total, 1)
    avg_layers_executed_per_sample = total_layers / max(total, 1)
    backbone_full_flops = float(profile["backbone_full_flops"])
    compute_overhead_ratio = (avg_flops_per_sample / backbone_full_flops) if backbone_full_flops > 0 else float("nan")
    return {
        "overall_acc": correct / max(total, 1),
        "exit_rates": [n / max(total, 1) for n in n_exit],
        "exit_accs": [(c / n) if n > 0 else float("nan") for c, n in zip(c_exit, n_exit)],
        "final_rate": n_final / max(total, 1),
        "final_acc": (c_final / n_final) if n_final > 0 else float("nan"),
        "avg_flops_per_sample": avg_flops_per_sample,
        "avg_macs_per_sample": avg_macs_per_sample,
        "avg_layers_executed_per_sample": avg_layers_executed_per_sample,
        "backbone_params": float(profile["backbone_params"]),
        "total_exit_head_params": float(profile["total_exit_head_params"]),
        "param_overhead_ratio": float(profile["param_overhead_ratio"]),
        "compute_overhead_ratio": compute_overhead_ratio,
        "compute_saving_ratio": 1.0 - compute_overhead_ratio if compute_overhead_ratio == compute_overhead_ratio else float("nan"),
    }


def _unique_quantile_values(values: torch.Tensor, quantiles: List[float]) -> List[float]:
    if values.numel() == 0:
        return [0.0]
    out = []
    for q in quantiles:
        out.append(float(torch.quantile(values, q).item()))
    uniq = sorted(set(out))
    return uniq if uniq else [0.0]


def sweep_cascade_by_quantile(val_cache: dict, test_cache: dict, quantile_groups: List[List[float]], test_top_k: int):
    margin_groups = val_cache["margins"]
    thr_groups = [
        _unique_quantile_values(margins, quantiles)
        for margins, quantiles in zip(margin_groups, quantile_groups)
    ]
    print("[quantile-sweep] threshold groups:", thr_groups)
    num_combinations = int(torch.tensor([len(g) for g in thr_groups], dtype=torch.long).prod().item()) if thr_groups else 0
    print(f"[quantile-sweep] evaluating {num_combinations} combinations on VAL; TEST will use top {test_top_k}")

    rows_val = []
    for thrs in itertools.product(*thr_groups):
        thrs = list(thrs)
        rows_val.append({"thrs": thrs, **eval_cascade_cached_quweit(val_cache, thrs)})

    rows_val.sort(key=lambda row: (-row["overall_acc"], row["avg_flops_per_sample"]))
    rows_test = []
    for row in rows_val[:max(0, test_top_k)]:
        thrs = list(row["thrs"])
        rows_test.append({"thrs": thrs, **eval_cascade_cached_quweit(test_cache, thrs)})
    rows_test.sort(key=lambda row: (-row["overall_acc"], row["avg_flops_per_sample"]))
    return rows_val, rows_test, thr_groups, num_combinations


def print_cascade_quantile_sweep(title: str, rows: List[dict], top_k: int = 20):
    print(f"\n=== {title} ===")
    if not rows:
        print("(empty)")
        return

    num_exits = len(rows[0]["thrs"])
    header_parts = [f"thr{i}" for i in range(num_exits)]
    header_parts += ["overall%"]
    for i in range(num_exits):
        header_parts += [f"exit{i}_rate%", f"exit{i}_acc%"]
    header_parts += ["final_rate%", "avgFLOPs", "avgMACs", "avgLayers", "overhead"]
    header = "  ".join(f"{item:>11s}" for item in header_parts)
    print(header)
    print("-" * len(header))

    for row in rows[:top_k]:
        values = [f"{thr:>11.4f}" for thr in row["thrs"]]
        values.append(f"{row['overall_acc'] * 100:>11.2f}")
        for rate, acc in zip(row["exit_rates"], row["exit_accs"]):
            acc_text = f"{acc * 100:>11.2f}" if acc == acc else f"{'nan':>11s}"
            values.append(f"{rate * 100:>11.2f}")
            values.append(acc_text)
        values.append(f"{row['final_rate'] * 100:>11.2f}")
        values.append(f"{row['avg_flops_per_sample']:>11.0f}")
        values.append(f"{row['avg_macs_per_sample']:>11.0f}")
        values.append(f"{row['avg_layers_executed_per_sample']:>11.3f}")
        values.append(f"{row['compute_overhead_ratio']:>11.4f}")
        print("  ".join(values))


def cotrain_g2_quweit(
    model,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs: int,
    train_block_indices: Sequence[int],
    freeze_block_indices: Sequence[int],
    exit_heads: List[torch.nn.Module],
    payload_exit_cfg: List[dict],
    thrs: Sequence[float],
    use_prob_margin: bool,
    lambda_final: float,
    lambda_exits: Sequence[float],
    lr_backbone: float,
    lr_classifier: float,
    lr_exits: float,
    weight_decay: float,
    grad_clip: Optional[float],
    gate_temps: Sequence[float],
    use_gate_weighting: bool,
):
    model = model.to(device)
    assert len(exit_heads) == len(payload_exit_cfg)
    assert len(thrs) == len(exit_heads)
    assert len(lambda_exits) == len(exit_heads)
    assert len(gate_temps) == len(exit_heads)

    set_requires_grad(model, False)
    for idx in freeze_block_indices:
        if idx < 0 or idx >= len(model.blocks):
            raise ValueError(f"freeze block index out of range: {idx}")
        set_requires_grad(model.blocks[idx], False)
    for idx in train_block_indices:
        if idx < 0 or idx >= len(model.blocks):
            raise ValueError(f"train block index out of range: {idx}")
        set_requires_grad(model.blocks[idx], True)

    set_requires_grad(model.head, True)
    if hasattr(model, "norm"):
        set_requires_grad(model.norm, True)

    exit_heads = [head.to(device) for head in exit_heads]
    for head in exit_heads:
        set_requires_grad(head, True)

    params_backbone = []
    for idx in train_block_indices:
        params_backbone.extend([p for p in model.blocks[idx].parameters() if p.requires_grad])
    params_classifier = [p for p in model.head.parameters() if p.requires_grad]
    if hasattr(model, "norm"):
        params_classifier.extend([p for p in model.norm.parameters() if p.requires_grad])
    params_exits = []
    for head in exit_heads:
        params_exits.extend([p for p in head.parameters() if p.requires_grad])

    print(
        f"[g2] trainable params: backbone={sum(p.numel() for p in params_backbone)} "
        f"classifier={sum(p.numel() for p in params_classifier)} "
        f"exits={sum(p.numel() for p in params_exits)}"
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": params_backbone, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": params_classifier, "lr": lr_classifier, "weight_decay": weight_decay},
            {"params": params_exits, "lr": lr_exits, "weight_decay": weight_decay},
        ]
    )

    best = {"val_overall_acc": -1.0, "state": None}

    for epoch in range(num_epochs):
        model.train()
        for head in exit_heads:
            head.train()

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            final_logits, h_list = forward_with_all_hidden(model, xb)
            ce_final = F.cross_entropy(final_logits, yb, reduction="none")

            if use_gate_weighting:
                eps = 1e-8
                u = torch.ones_like(ce_final)
                loss_exit_sum = 0.0

                for i, (cfg, head) in enumerate(zip(payload_exit_cfg, exit_heads)):
                    layer_idx = int(cfg["layer_idx"]) - 1
                    logits_i = _head_logits_from_hidden_trainable(head, h_list[layer_idx], device)
                    ce_i = F.cross_entropy(logits_i, yb, reduction="none")
                    m_i = _margin_from_logits(logits_i, use_prob=use_prob_margin)
                    w_i = torch.sigmoid((m_i - float(thrs[i])) / float(gate_temps[i]))
                    take_i = u * w_i
                    take_i_det = take_i.detach()
                    loss_i = (take_i_det * ce_i).sum() / (take_i_det.sum() + eps)
                    loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * loss_i
                    u = u * (1.0 - w_i)

                u_det = u.detach()
                loss_final = (u_det * ce_final).sum() / (u_det.sum() + eps)
                loss = float(lambda_final) * loss_final + loss_exit_sum
            else:
                loss_final = ce_final.mean()
                loss_exit_sum = 0.0
                for i, (cfg, head) in enumerate(zip(payload_exit_cfg, exit_heads)):
                    layer_idx = int(cfg["layer_idx"]) - 1
                    logits_i = _head_logits_from_hidden_trainable(head, h_list[layer_idx], device)
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

        out_val = eval_cascade_quweit(
            model,
            val_loader,
            device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=thrs,
            use_prob_margin=use_prob_margin,
        )
        va_overall = out_val["overall_acc"]
        print(
            f"[G2] Ep{epoch:03d} | overall@{tuple(float(x) for x in thrs)} va={va_overall * 100:.2f} "
            f"| exit_rates={out_val['exit_rates']} final_rate={out_val['final_rate']:.4f}"
        )

        if va_overall > best["val_overall_acc"]:
            best["val_overall_acc"] = float(va_overall)
            best["state"] = {
                "model": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                "exits": [copy.deepcopy({k: v.detach().cpu() for k, v in head.state_dict().items()}) for head in exit_heads],
            }

    if best["state"] is not None:
        model.load_state_dict(best["state"]["model"], strict=True)
        for i, head in enumerate(exit_heads):
            head.load_state_dict(best["state"]["exits"][i], strict=True)

    return model, exit_heads, best


def main():
    parser = argparse.ArgumentParser(description="QuWeiT g2 co-train from a g1 checkpoint.")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Checkpoint produced by train_quweit_lut_early_exit_g1.py")
    parser.add_argument("--path_out", type=str, required=True)

    parser.add_argument("--batch_size_train", type=int, default=128)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--train_blocks", type=str, default="", help='0-based block indices to train; empty means all blocks after the earliest exit block')
    parser.add_argument("--freeze_blocks", type=str, default="", help='0-based block indices to freeze; empty means blocks up to and including the earliest exit block')
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_classifier", type=float, default=3e-4)
    parser.add_argument("--lr_exits", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--thr", type=str, default="", help="comma-separated thresholds per exit; empty means use checkpoint values")
    parser.add_argument("--lambda_final", type=float, default=1.0)
    parser.add_argument("--lambda_exits", type=str, default="0.05")
    parser.add_argument("--gate_temps", type=str, default="1.0")
    parser.add_argument("--use_gate_weighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_prob_margin", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--single_thr_list", type=str, default="0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0")
    parser.add_argument("--cascade_thr_grid", type=str, default="")
    parser.add_argument("--cascade_quantiles", type=str, default="0.0,0.25,0.5,0.75,0.9,0.95")
    parser.add_argument("--sweep_top_k", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model, backbone_cfg, raw_ckpt, exit_heads, exit_cfg_list = load_quweit_model_with_exits(args.model_ckpt, device)
    if not exit_heads:
        raise ValueError("No exit heads found in --model_ckpt")

    train_loader, val_loader, test_loader, num_classes = build_clean_cifar_loaders(
        backbone_cfg,
        batch_size_probe=args.batch_size_train,
        batch_size_eval=args.batch_size_eval,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    if num_classes != backbone_cfg.num_classes:
        raise ValueError(f"Dataset num_classes mismatch: loaders={num_classes}, cfg={backbone_cfg.num_classes}")

    payload_exit_cfg = [cfg.to_payload() for cfg in exit_cfg_list]
    if args.thr.strip():
        override_thrs = _broadcast(_parse_csv(args.thr, float), len(payload_exit_cfg), "thr")
        for cfg, thr in zip(payload_exit_cfg, override_thrs):
            cfg["thr"] = float(thr)
    thrs = [float(cfg["thr"]) for cfg in payload_exit_cfg]
    single_thr_list = _parse_csv(args.single_thr_list, float)
    cascade_thr_grid = _parse_threshold_groups(args.cascade_thr_grid, len(payload_exit_cfg), "cascade_thr_grid")
    cascade_quantile_groups = _parse_threshold_groups(args.cascade_quantiles, len(payload_exit_cfg), "cascade_quantiles")

    lambda_exits = _broadcast(_parse_csv(args.lambda_exits, float), len(payload_exit_cfg), "lambda_exits")
    gate_temps = _broadcast(_parse_csv(args.gate_temps, float), len(payload_exit_cfg), "gate_temps")

    exit_layers_0based = [int(cfg["layer_idx"]) - 1 for cfg in payload_exit_cfg]
    earliest_exit_block = min(exit_layers_0based)
    if args.train_blocks.strip():
        train_blocks = _parse_csv(args.train_blocks, int)
    else:
        train_blocks = list(range(min(earliest_exit_block + 1, len(model.blocks)), len(model.blocks)))
    if args.freeze_blocks.strip():
        freeze_blocks = _parse_csv(args.freeze_blocks, int)
    else:
        freeze_blocks = list(range(earliest_exit_block + 1))

    if not train_blocks:
        raise ValueError("No train blocks selected.")
    for idx in train_blocks + freeze_blocks:
        if idx < 0 or idx >= len(model.blocks):
            raise ValueError(f"Block index out of range: {idx}")
    if set(train_blocks) & set(freeze_blocks):
        raise ValueError(f"train_blocks and freeze_blocks overlap: {sorted(set(train_blocks) & set(freeze_blocks))}")

    print("[info] loader settings "
          f"train_batch={args.batch_size_train} eval_batch={args.batch_size_eval} "
          f"num_workers={args.num_workers} pin_memory={args.pin_memory}")
    print(
        f"[info] g2 plan train_blocks={train_blocks} freeze_blocks={freeze_blocks} "
        f"thrs={thrs} lambda_exits={lambda_exits} gate_temps={gate_temps}"
    )

    model, exit_heads, best = cotrain_g2_quweit(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=args.epochs,
        train_block_indices=train_blocks,
        freeze_block_indices=freeze_blocks,
        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
        thrs=thrs,
        use_prob_margin=args.use_prob_margin,
        lambda_final=args.lambda_final,
        lambda_exits=lambda_exits,
        lr_backbone=args.lr_backbone,
        lr_classifier=args.lr_classifier,
        lr_exits=args.lr_exits,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        gate_temps=gate_temps,
        use_gate_weighting=args.use_gate_weighting,
    )
    print(f"[g2] best val overall acc = {best['val_overall_acc'] * 100:.2f}%")

    _ensure_dir(args.path_out)
    save_ckpt_v2(
        args.path_out,
        model.cpu(),
        [head.cpu() for head in exit_heads],
        raw_ckpt["backbone_cfg"],
        exit_cfg_list=[ExitConfig.from_payload(cfg).to_payload() for cfg in payload_exit_cfg],
        extra={
            "dataset": backbone_cfg.dataset,
            "train_mode": "g2_cotrain",
            "source_ckpt": args.model_ckpt,
            "train_blocks": train_blocks,
            "freeze_blocks": freeze_blocks,
            "lambda_final": float(args.lambda_final),
            "lambda_exits": [float(x) for x in lambda_exits],
            "gate_temps": [float(x) for x in gate_temps],
            "use_gate_weighting": bool(args.use_gate_weighting),
            "eval_thrs": thrs,
        },
    )

    model = model.to(device)
    exit_heads = [head.to(device) for head in exit_heads]
    profile = get_external_exit_profile(model, exit_heads, payload_exit_cfg)
    val_cache = collect_cascade_cache_quweit(
        model,
        val_loader,
        device,
        exit_heads=exit_heads,
        exit_cfg_list=payload_exit_cfg,
        use_prob_margin=args.use_prob_margin,
        profile=profile,
    )
    test_cache = collect_cascade_cache_quweit(
        model,
        test_loader,
        device,
        exit_heads=exit_heads,
        exit_cfg_list=payload_exit_cfg,
        use_prob_margin=args.use_prob_margin,
        profile=profile,
    )
    print(f"\n[saved] {args.path_out}")

    val_out = eval_cascade_cached_quweit(val_cache, thrs)
    test_out = eval_cascade_cached_quweit(test_cache, thrs)
    print(
        f"[VAL] overall={val_out['overall_acc'] * 100:.2f}% "
        f"exit_rates={[round(x, 4) for x in val_out['exit_rates']]} final_rate={val_out['final_rate']:.4f}"
    )
    print(
        f"[TEST] overall={test_out['overall_acc'] * 100:.2f}% "
        f"exit_rates={[round(x, 4) for x in test_out['exit_rates']]} final_rate={test_out['final_rate']:.4f}"
    )

    for exit_id, cfg in enumerate(payload_exit_cfg):
        layer_idx = int(cfg["layer_idx"])
        print(f"\n[VAL single-exit scan] exit={exit_id} layer={layer_idx}")
        for thr in single_thr_list:
            scan_thrs = [thr if i == exit_id else thrs[i] for i in range(len(thrs))]
            out = eval_cascade_cached_quweit(val_cache, scan_thrs)
            exit_acc = out["exit_accs"][exit_id]
            exit_acc_text = f"{exit_acc * 100:.2f}%" if exit_acc == exit_acc else "nan"
            print(f"  thr={thr:.2f} overall={out['overall_acc'] * 100:.2f}% exit_rate={out['exit_rates'][exit_id] * 100:.2f}% exit_acc={exit_acc_text}")

    if cascade_thr_grid:
        rows_val = []
        rows_test = []
        for grid_thrs in itertools.product(*cascade_thr_grid):
            grid_thrs = list(grid_thrs)
            rows_val.append({"thrs": grid_thrs, **eval_cascade_cached_quweit(val_cache, grid_thrs)})
            rows_test.append({"thrs": grid_thrs, **eval_cascade_cached_quweit(test_cache, grid_thrs)})
        rows_val.sort(key=lambda row: row["overall_acc"], reverse=True)
        rows_test.sort(key=lambda row: row["overall_acc"], reverse=True)
        print_cascade_quantile_sweep("VAL cascade grid sweep", rows_val, top_k=args.sweep_top_k)
        print_cascade_quantile_sweep("TEST cascade grid sweep", rows_test, top_k=args.sweep_top_k)

    if cascade_quantile_groups:
        rows_val, rows_test, thr_groups, num_combinations = sweep_cascade_by_quantile(val_cache, test_cache, cascade_quantile_groups, args.sweep_top_k)
        print(f"[quantile-sweep] num_combinations={num_combinations}")
        print_cascade_quantile_sweep("VAL cascade quantile sweep", rows_val, top_k=args.sweep_top_k)
        print_cascade_quantile_sweep("TEST cascade quantile sweep", rows_test, top_k=args.sweep_top_k)


if __name__ == "__main__":
    main()
