import argparse
import copy
import itertools
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.core.linearExitHead import build_exits_from_ckpt
from src.core.multiLayerWNN import save_ckpt_v2
from src.early_exit import _head_logits_from_hidden_trainable
from src.exit.ckpt_exit import ExitConfig
from src.train_quweit_lut_backbone_v2 import QuWeiTViT
from src.train_quweit_lut_early_exit_g0 import build_clean_cifar_loaders, get_external_exit_profile, load_quweit_backbone_ckpt


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


def load_quweit_exit_pack(path: str, device, num_classes: int):
    exit_heads, exit_cfg_list = build_exits_from_ckpt(path, device, num_classes=num_classes)
    return exit_heads, exit_cfg_list


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


@dataclass
class AlternatingPhaseConfig:
    cycles: int = 3
    epochs_F: int = 1
    epochs_H: int = 1
    final_train_layers: Sequence[int] = ()
    final_train_classifier: bool = True
    train_exit_ids: Sequence[int] = ()
    lr_backbone_F: float = 1e-5
    lr_classifier_F: float = 1e-5
    lr_exits_H: float = 5e-5


def _clone_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _safe_float(x: float, fallback: float = 0.0) -> float:
    x = float(x)
    return fallback if x != x else x


def _validate_backbone_layer_indices(layer_ids: Sequence[int], *, num_blocks: int, name: str) -> List[int]:
    resolved = []
    for layer_id in layer_ids:
        layer_id = int(layer_id)
        if layer_id < 0 or layer_id >= num_blocks:
            raise ValueError(f"Invalid {name} {layer_id}; expected 0-based index in [0, {num_blocks - 1}]")
        resolved.append(layer_id)
    return resolved


def _resolve_exit_ids_to_indices(train_exit_ids: Sequence[int], payload_exit_cfg: List[dict]) -> Tuple[List[int], List[int]]:
    exit_layer_ids = [int(cfg["layer_idx"]) for cfg in payload_exit_cfg]
    layer_id_to_index = {layer_id: idx for idx, layer_id in enumerate(exit_layer_ids)}

    resolved_indices = []
    resolved_layer_ids = []
    for exit_id in train_exit_ids:
        exit_id = int(exit_id)
        if exit_id in layer_id_to_index:
            idx = layer_id_to_index[exit_id]
        elif 0 <= exit_id < len(payload_exit_cfg):
            idx = exit_id
        else:
            raise ValueError(
                f"Invalid train exit id {exit_id}; expected one of layer ids {exit_layer_ids}"
            )
        resolved_indices.append(idx)
        resolved_layer_ids.append(exit_layer_ids[idx])
    return resolved_indices, resolved_layer_ids


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
    thrs: List[float],
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
            h = h_list[layer_idx_1based - 1]
            logits = head(h)

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


def eval_cascade_cached_quweit(cache: dict, thrs: List[float]):
    profile = cache["profile"]
    labels = cache["labels"]
    final_pred = cache["final_pred"].clone()
    exit_preds = cache["exit_pred"]
    margins = cache["margins"]
    num_exits = len(exit_preds)

    total = int(labels.numel())
    preds = final_pred
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


@torch.no_grad()
def eval_final_only_on_tail_quweit(
    model,
    loader,
    device,
    *,
    exit_heads: List[torch.nn.Module],
    exit_cfg_list: List[dict],
    thrs_tail_anchor: List[float],
    use_prob_margin: bool = False,
):
    model.eval()
    exit_heads = [head.to(device).eval() for head in exit_heads]

    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        final_logits, h_list = forward_with_all_hidden(model, xb)

        undecided = torch.ones_like(yb, dtype=torch.bool)
        for exit_id, (head, cfg, thr) in enumerate(zip(exit_heads, exit_cfg_list, thrs_tail_anchor)):
            layer_idx_1based = int(cfg["layer_idx"])
            logits = head(h_list[layer_idx_1based - 1])
            if use_prob_margin:
                top2 = torch.topk(torch.softmax(logits, dim=-1), k=2, dim=-1).values
            else:
                top2 = torch.topk(logits, k=2, dim=-1).values
            margins = top2[:, 0] - top2[:, 1]
            take = undecided & (margins > float(thr))
            undecided = undecided & (~take)

        if undecided.any():
            pred = final_logits[undecided].argmax(dim=-1)
            correct += int((pred == yb[undecided]).sum().item())
            total += int(undecided.sum().item())

    return {"final_tail_acc": float(correct / total) if total > 0 else float("nan"), "tail_count": int(total)}


def evaluate_multi_exit_bundle_quweit(
    model,
    val_loader,
    device,
    *,
    exit_heads: List[torch.nn.Module],
    exit_cfg_list: List[dict],
    thrs_eval_list: Sequence[Sequence[float]],
    thrs_tail_anchor: Sequence[float],
    best_eval_idx: int = 0,
    use_prob_margin: bool = False,
):
    eval_records = []
    thrs_eval_list = [tuple(x) for x in thrs_eval_list]

    for k, thrs_eval in enumerate(thrs_eval_list):
        out_val = eval_cascade_quweit(
            model,
            val_loader,
            device,
            exit_heads=exit_heads,
            exit_cfg_list=exit_cfg_list,
            thrs=list(thrs_eval),
            use_prob_margin=use_prob_margin,
        )
        tail_stats = eval_final_only_on_tail_quweit(
            model,
            val_loader,
            device,
            exit_heads=exit_heads,
            exit_cfg_list=exit_cfg_list,
            thrs_tail_anchor=list(thrs_tail_anchor),
            use_prob_margin=use_prob_margin,
        )
        eval_records.append(
            {
                "eval_idx": k,
                "thrs": tuple(thrs_eval),
                "overall_acc": float(out_val["overall_acc"]),
                "final_acc": float(out_val["final_acc"]),
                "final_tail_acc": float(tail_stats["final_tail_acc"]),
                "tail_count": int(tail_stats["tail_count"]),
                "final_rate": float(out_val["final_rate"]),
                "exit_rates": list(out_val["exit_rates"]),
                "exit_accs": list(out_val["exit_accs"]),
                "raw": out_val,
            }
        )

    selected = eval_records[best_eval_idx]
    return {"records": eval_records, "selected": selected}


def compute_selection_metric(
    record: Dict[str, Any],
    best_metric: str,
    combo_metric_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    min_exit_accs: Optional[Sequence[Optional[float]]] = None,
    min_tail_count: int = 0,
    baseline_overall: Optional[float] = None,
    baseline_tail_acc: Optional[float] = None,
    max_overall_drop: float = 0.005,
    max_tail_drop: float = 0.01,
) -> float:
    overall = float(record["overall_acc"])
    final_only = _safe_float(record["final_acc"], fallback=-1.0)
    final_tail = _safe_float(record["final_tail_acc"], fallback=-1.0)
    normalized_exit_rate = float(sum(record.get("exit_rates", [])))

    if best_metric == "val_overall_acc":
        metric = overall
    elif best_metric == "val_final_only":
        metric = final_only
    elif best_metric == "val_final_tail_only":
        metric = final_tail
    elif best_metric == "val_combo":
        w_overall, w_tail, w_exit = combo_metric_weights
        metric = w_overall * overall + w_tail * final_tail + w_exit * normalized_exit_rate
    else:
        raise ValueError("best_metric must be one of ['val_overall_acc', 'val_final_only', 'val_final_tail_only', 'val_combo']")

    if best_metric == "val_final_tail_only" and int(record["tail_count"]) < int(min_tail_count):
        return -1.0
    if min_exit_accs is not None and record.get("exit_accs") is not None:
        exit_accs = record["exit_accs"]
        for i, min_acc in enumerate(min_exit_accs):
            if min_acc is None:
                continue
            if i >= len(exit_accs):
                continue
            if exit_accs[i] < min_acc:
                return -1.0
    if baseline_overall is not None and overall < float(baseline_overall) - float(max_overall_drop):
        return -1.0
    if baseline_tail_acc is not None and final_tail < float(baseline_tail_acc) - float(max_tail_drop):
        return -1.0
    return metric


def compute_exit_loss_by_layer(
    *,
    layer_idx: int,
    exit_logits: torch.Tensor,
    final_logits: torch.Tensor,
    yb: torch.Tensor,
    cfg: Dict[str, Any],
):
    mode = cfg.get("mode", "baseline")
    override = dict(cfg.get("override", {}))
    kd_T = float(override.get("kd_T", 2.0))
    lambda_kd = float(override.get("lambda_kd", 0.7))

    loss_ce = F.cross_entropy(exit_logits, yb)
    loss = loss_ce

    if mode in {"kd", "kd_final_correct"}:
        with torch.no_grad():
            teacher_prob = F.softmax(final_logits / kd_T, dim=-1)
            final_pred = final_logits.argmax(dim=-1)
            final_correct = final_pred.eq(yb)
        student_log_prob = F.log_softmax(exit_logits / kd_T, dim=-1)
        if mode == "kd_final_correct":
            if final_correct.any():
                loss_kd = F.kl_div(
                    student_log_prob[final_correct],
                    teacher_prob[final_correct],
                    reduction="batchmean",
                ) * (kd_T * kd_T)
            else:
                loss_kd = exit_logits.new_zeros(())
        else:
            loss_kd = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (kd_T * kd_T)
        loss = loss + lambda_kd * loss_kd
    elif mode != "baseline":
        raise ValueError(f"Unsupported exit loss mode at layer {layer_idx}: {mode}")

    return loss


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


def configure_trainable_params_quweit(
    model,
    exit_heads,
    *,
    train_layers: Sequence[int],
    train_exit_ids: Sequence[int],
    train_classifier: bool,
):
    set_requires_grad(model, False)
    for head in exit_heads:
        set_requires_grad(head, False)

    train_layers = tuple(int(x) for x in train_layers)
    train_exit_ids = tuple(int(x) for x in train_exit_ids)

    for layer_idx in train_layers:
        if layer_idx < 0 or layer_idx >= len(model.blocks):
            raise ValueError(f"train_layers contains out-of-range layer {layer_idx}")
        set_requires_grad(model.blocks[layer_idx], True)

    if train_classifier:
        if hasattr(model, "norm"):
            set_requires_grad(model.norm, True)
        if hasattr(model, "head"):
            set_requires_grad(model.head, True)

    for exit_id in train_exit_ids:
        if exit_id < 0 or exit_id >= len(exit_heads):
            raise ValueError(f"train_exit_ids contains out-of-range exit {exit_id}")
        set_requires_grad(exit_heads[exit_id], True)


def build_optimizer_quweit(
    model,
    exit_heads,
    *,
    train_layers: Sequence[int],
    train_exit_ids: Sequence[int],
    lr_backbone: float,
    lr_classifier: float,
    lr_exits: float,
    weight_decay: float,
):
    train_layers = tuple(int(x) for x in train_layers)
    train_exit_ids = tuple(int(x) for x in train_exit_ids)

    params_backbone = [p for layer_idx in train_layers for p in model.blocks[layer_idx].parameters() if p.requires_grad]
    params_classifier = []
    if hasattr(model, "norm"):
        params_classifier.extend([p for p in model.norm.parameters() if p.requires_grad])
    if hasattr(model, "head"):
        params_classifier.extend([p for p in model.head.parameters() if p.requires_grad])
    params_exits = [p for exit_id in train_exit_ids for p in exit_heads[exit_id].parameters() if p.requires_grad]

    groups = []
    if params_backbone:
        groups.append({"params": params_backbone, "lr": lr_backbone, "weight_decay": weight_decay})
    if params_classifier:
        groups.append({"params": params_classifier, "lr": lr_classifier, "weight_decay": weight_decay})
    if params_exits:
        groups.append({"params": params_exits, "lr": lr_exits, "weight_decay": weight_decay})
    if not groups:
        raise ValueError("No trainable parameters found for alternating phase.")

    return torch.optim.AdamW(groups)


def cotrain_quweit_alternating(
    model,
    train_loader,
    val_loader,
    device,
    *,
    exit_heads: List[torch.nn.Module],
    payload_exit_cfg: List[dict],
    alt_cfg: AlternatingPhaseConfig,
    thrs_eval_list: Sequence[Sequence[float]],
    thrs_tail_anchor: Sequence[float],
    exit_loss_by_layer: Dict[int, Dict[str, Any]],
    baseline_overall: float,
    baseline_tail_acc: float,
    best_metric: str = "val_combo",
    best_eval_idx: int = 0,
    combo_metric_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    min_exit_accs: Optional[Sequence[Optional[float]]] = None,
    max_overall_drop: float = 0.005,
    max_tail_drop: float = 0.01,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    use_prob_margin: bool = False,
):
    model = model.to(device)
    exit_heads = [head.to(device) for head in exit_heads]

    initial_eval = evaluate_multi_exit_bundle_quweit(
        model,
        val_loader,
        device,
        exit_heads=exit_heads,
        exit_cfg_list=payload_exit_cfg,
        thrs_eval_list=thrs_eval_list,
        thrs_tail_anchor=thrs_tail_anchor,
        best_eval_idx=best_eval_idx,
        use_prob_margin=use_prob_margin,
    )
    initial_selected = copy.deepcopy(initial_eval["selected"])
    initial_metric = compute_selection_metric(
        initial_selected,
        best_metric=best_metric,
        combo_metric_weights=combo_metric_weights,
        min_exit_accs=min_exit_accs,
        baseline_overall=baseline_overall,
        baseline_tail_acc=baseline_tail_acc,
        max_overall_drop=max_overall_drop,
        max_tail_drop=max_tail_drop,
    )
    best = {
        "metric": initial_metric,
        "cycle": -1,
        "selected": initial_selected,
        "state_model": _clone_state_dict(model),
        "state_exits": [_clone_state_dict(head) for head in exit_heads],
    }

    final_train_layers = tuple(int(x) for x in alt_cfg.final_train_layers)
    train_exit_ids = tuple(int(x) for x in alt_cfg.train_exit_ids)

    for cycle in range(int(alt_cfg.cycles)):
        configure_trainable_params_quweit(
            model,
            exit_heads,
            train_layers=final_train_layers,
            train_exit_ids=(),
            train_classifier=bool(alt_cfg.final_train_classifier),
        )
        optimizer_F = build_optimizer_quweit(
            model,
            exit_heads,
            train_layers=final_train_layers,
            train_exit_ids=(),
            lr_backbone=float(alt_cfg.lr_backbone_F),
            lr_classifier=float(alt_cfg.lr_classifier_F),
            lr_exits=0.0,
            weight_decay=weight_decay,
        )

        for epoch_F in range(int(alt_cfg.epochs_F)):
            model.train()
            for head in exit_heads:
                head.eval()

            running_loss = 0.0
            batch_count = 0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                final_logits, _ = forward_with_all_hidden(model, xb)
                loss = F.cross_entropy(final_logits, yb)

                optimizer_F.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer_F.param_groups for p in group["params"]],
                        grad_clip,
                    )
                optimizer_F.step()

                running_loss += float(loss.detach().item())
                batch_count += 1

            print(f"[alt:F] cycle={cycle:02d} epoch={epoch_F:02d} | loss={running_loss / max(batch_count, 1):.4f}")

        configure_trainable_params_quweit(
            model,
            exit_heads,
            train_layers=(),
            train_exit_ids=train_exit_ids,
            train_classifier=False,
        )
        optimizer_H = build_optimizer_quweit(
            model,
            exit_heads,
            train_layers=(),
            train_exit_ids=train_exit_ids,
            lr_backbone=0.0,
            lr_classifier=0.0,
            lr_exits=float(alt_cfg.lr_exits_H),
            weight_decay=weight_decay,
        )

        for epoch_H in range(int(alt_cfg.epochs_H)):
            model.eval()
            for head in exit_heads:
                head.train()

            running_loss = 0.0
            batch_count = 0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with torch.no_grad():
                    final_logits, h_list = forward_with_all_hidden(model, xb)

                loss_sum = None
                trained_exit_count = 0
                for exit_id, cfg in enumerate(payload_exit_cfg):
                    if exit_id not in train_exit_ids:
                        continue
                    layer_idx_1based = int(cfg["layer_idx"])
                    h_i = h_list[layer_idx_1based - 1].detach()
                    exit_logits = _head_logits_from_hidden_trainable(exit_heads[exit_id], h_i, device)
                    loss_i = compute_exit_loss_by_layer(
                        layer_idx=layer_idx_1based - 1,
                        exit_logits=exit_logits,
                        final_logits=final_logits.detach(),
                        yb=yb,
                        cfg=exit_loss_by_layer[layer_idx_1based - 1],
                    )
                    loss_sum = loss_i if loss_sum is None else loss_sum + loss_i
                    trained_exit_count += 1

                if loss_sum is None or trained_exit_count == 0:
                    raise ValueError("No exit loss was accumulated in H phase.")
                loss = loss_sum / float(trained_exit_count)

                optimizer_H.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer_H.param_groups for p in group["params"]],
                        grad_clip,
                    )
                optimizer_H.step()

                running_loss += float(loss.detach().item())
                batch_count += 1

            print(f"[alt:H] cycle={cycle:02d} epoch={epoch_H:02d} | loss={running_loss / max(batch_count, 1):.4f}")

        eval_stats = evaluate_multi_exit_bundle_quweit(
            model,
            val_loader,
            device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs_eval_list=thrs_eval_list,
            thrs_tail_anchor=thrs_tail_anchor,
            best_eval_idx=best_eval_idx,
            use_prob_margin=use_prob_margin,
        )
        selected = eval_stats["selected"]
        metric = compute_selection_metric(
            selected,
            best_metric=best_metric,
            combo_metric_weights=combo_metric_weights,
            min_exit_accs=min_exit_accs,
            baseline_overall=baseline_overall,
            baseline_tail_acc=baseline_tail_acc,
            max_overall_drop=max_overall_drop,
            max_tail_drop=max_tail_drop,
        )

        print(
            f"[alt:eval] cycle={cycle:02d} "
            f"| overall={selected['overall_acc'] * 100:.2f} "
            f"| final={selected['final_acc'] * 100:.2f} "
            f"| final_tail={_safe_float(selected['final_tail_acc']) * 100:.2f} "
            f"| final_rate={selected['final_rate']:.4f} "
            f"| exit_rates={selected['exit_rates']} "
            f"| exit_accs={selected.get('exit_accs', None)} "
            f"| metric={metric:.4f}"
        )

        if metric > best["metric"]:
            best["metric"] = metric
            best["cycle"] = cycle
            best["selected"] = copy.deepcopy(selected)
            best["state_model"] = _clone_state_dict(model)
            best["state_exits"] = [_clone_state_dict(head) for head in exit_heads]

    if best["state_model"] is not None:
        model.load_state_dict(best["state_model"], strict=False)
        for head, sd in zip(exit_heads, best["state_exits"]):
            head.load_state_dict(sd, strict=True)

    return model, exit_heads, best


def main():
    parser = argparse.ArgumentParser(description="QuWeiT g1 co-train: use g0 exit heads + backbone_v2 backbone.")
    parser.add_argument("--backbone_ckpt", type=str, required=True, help="Checkpoint produced by train_quweit_lut_backbone_v2.py")
    parser.add_argument("--exit_ckpt", type=str, required=True, help="Checkpoint produced by train_quweit_lut_early_exit_g0.py")
    parser.add_argument("--path_out", type=str, required=True)
    parser.add_argument("--use_ema_backbone", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--batch_size_train", type=int, default=128)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--epochs_F", type=int, default=1)
    parser.add_argument("--epochs_H", type=int, default=1)
    parser.add_argument("--final_train_layers", type=str, default="11", help='0-based backbone block indices for F phase, e.g. "11" or "10,11"')
    parser.add_argument("--train_exit_ids", type=str, default="", help='exit ids following backbone layer naming, e.g. "2,4,6,8"; empty means all exits')
    parser.add_argument("--train_classifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr_backbone_F", type=float, default=1e-7)
    parser.add_argument("--lr_classifier_F", type=float, default=1e-5)
    parser.add_argument("--lr_exits_H", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--thr", type=str, default="", help='comma-separated thresholds per exit; empty means use g0 checkpoint values')
    parser.add_argument("--use_prob_margin", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--best_metric", type=str, default="val_combo")
    parser.add_argument("--combo_metric_weights", type=str, default="0.5,0.3,0.2")
    parser.add_argument("--max_overall_drop", type=float, default=0.005)
    parser.add_argument("--max_tail_drop", type=float, default=0.01)
    parser.add_argument("--min_exit_accs", type=str, default="0.98,0.98")
    parser.add_argument("--single_thr_list", type=str, default="0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0")
    parser.add_argument("--cascade_thr_grid", type=str, default="")
    parser.add_argument("--cascade_quantiles", type=str, default="0.0,0.25,0.5,0.75,0.9,0.95")
    parser.add_argument("--sweep_top_k", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    backbone, backbone_cfg, raw_backbone_ckpt = load_quweit_backbone_ckpt(
        args.backbone_ckpt,
        device,
        use_ema=args.use_ema_backbone,
    )

    exit_heads, exit_cfg_list = load_quweit_exit_pack(
        args.exit_ckpt,
        device,
        num_classes=backbone_cfg.num_classes,
    )
    if not exit_heads:
        raise ValueError("No exit heads found in --exit_ckpt")

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
    for cfg in payload_exit_cfg:
        layer_idx_1based = int(cfg["layer_idx"])
        if layer_idx_1based < 1 or layer_idx_1based > backbone_cfg.depth:
            raise ValueError(f"Invalid exit layer {layer_idx_1based} for depth={backbone_cfg.depth}")

    if args.thr.strip():
        override_thrs = _broadcast(_parse_csv(args.thr, float), len(payload_exit_cfg), "thr")
        for cfg, thr in zip(payload_exit_cfg, override_thrs):
            cfg["thr"] = float(thr)

    eval_thrs = [float(cfg["thr"]) for cfg in payload_exit_cfg]
    single_thr_list = _parse_csv(args.single_thr_list, float)
    cascade_thr_grid = _parse_threshold_groups(args.cascade_thr_grid, len(payload_exit_cfg), "cascade_thr_grid")
    cascade_quantile_groups = _parse_threshold_groups(args.cascade_quantiles, len(payload_exit_cfg), "cascade_quantiles")

    if args.train_exit_ids.strip():
        train_exit_ids_input = _parse_csv(args.train_exit_ids, int)
    else:
        train_exit_ids_input = [int(cfg["layer_idx"]) for cfg in payload_exit_cfg]
    if not train_exit_ids_input:
        raise ValueError("No exits selected for H phase.")
    train_exit_indices, train_exit_layer_ids = _resolve_exit_ids_to_indices(train_exit_ids_input, payload_exit_cfg)

    if args.final_train_layers.strip():
        final_train_layers_input = _parse_csv(args.final_train_layers, int)
    else:
        final_train_layers_input = []
    final_train_layers = _validate_backbone_layer_indices(
        final_train_layers_input,
        num_blocks=len(backbone.blocks),
        name="F-phase layer",
    )

    combo_metric_weights = tuple(_parse_csv(args.combo_metric_weights, float))
    if len(combo_metric_weights) != 3:
        raise ValueError("--combo_metric_weights must have exactly 3 values")

    min_exit_accs_raw = _broadcast(_parse_csv(args.min_exit_accs, float), len(exit_heads), "min_exit_accs")
    min_exit_accs = tuple(float(x) for x in min_exit_accs_raw)
    exit_loss_by_layer = {
        int(cfg["layer_idx"]) - 1: {"mode": "kd_final_correct", "override": {"kd_T": 2.0, "lambda_kd": 0.7}}
        for cfg in payload_exit_cfg
    }
    alt_cfg = AlternatingPhaseConfig(
        cycles=args.cycles,
        epochs_F=args.epochs_F,
        epochs_H=args.epochs_H,
        final_train_layers=final_train_layers,
        final_train_classifier=bool(args.train_classifier),
        train_exit_ids=tuple(train_exit_indices),
        lr_backbone_F=args.lr_backbone_F,
        lr_classifier_F=args.lr_classifier_F,
        lr_exits_H=args.lr_exits_H,
    )

    print("[info] loader settings "
          f"train_batch={args.batch_size_train} eval_batch={args.batch_size_eval} "
          f"num_workers={args.num_workers} pin_memory={args.pin_memory}")
    print("[info] alternating plan")
    print(
        f"  cycles={alt_cfg.cycles} epochs_F={alt_cfg.epochs_F} epochs_H={alt_cfg.epochs_H} "
        f"final_layers={list(final_train_layers_input)} train_classifier={alt_cfg.final_train_classifier} "
        f"train_exit_ids={list(train_exit_layer_ids)} "
        f"lr_backbone_F={alt_cfg.lr_backbone_F} lr_classifier_F={alt_cfg.lr_classifier_F} "
        f"lr_exits_H={alt_cfg.lr_exits_H}"
    )

    baseline_eval = evaluate_multi_exit_bundle_quweit(
        backbone,
        val_loader,
        device,
        exit_heads=exit_heads,
        exit_cfg_list=payload_exit_cfg,
        thrs_eval_list=[tuple(eval_thrs)],
        thrs_tail_anchor=tuple(eval_thrs),
        best_eval_idx=0,
        use_prob_margin=args.use_prob_margin,
    )
    baseline_selected = baseline_eval["selected"]
    baseline_overall = float(baseline_selected["overall_acc"])
    baseline_tail_acc = _safe_float(baseline_selected["final_tail_acc"], fallback=0.0)
    print(
        f"[before training] "
        f"| overall={baseline_selected['overall_acc'] * 100:.2f} "
        f"| final={baseline_selected['final_acc'] * 100:.2f} "
        f"| final_tail={_safe_float(baseline_selected['final_tail_acc']) * 100:.2f} "
        f"| final_rate={baseline_selected['final_rate']:.4f} "
        f"| exit_rates={baseline_selected['exit_rates']} "
        f"| exit_accs={baseline_selected.get('exit_accs', None)}"
    )

    backbone, exit_heads, best = cotrain_quweit_alternating(
        backbone,
        train_loader,
        val_loader,
        device,
        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
        alt_cfg=alt_cfg,
        thrs_eval_list=[tuple(eval_thrs)],
        thrs_tail_anchor=tuple(eval_thrs),
        exit_loss_by_layer=exit_loss_by_layer,
        baseline_overall=baseline_overall,
        baseline_tail_acc=baseline_tail_acc,
        best_metric=args.best_metric,
        best_eval_idx=0,
        combo_metric_weights=combo_metric_weights,
        min_exit_accs=min_exit_accs,
        max_overall_drop=args.max_overall_drop,
        max_tail_drop=args.max_tail_drop,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_prob_margin=args.use_prob_margin,
    )
    print(f"[best] cycle={best['cycle']} metric={best['metric']:.4f} record={best['selected']}")

    _ensure_dir(args.path_out)
    save_ckpt_v2(
        args.path_out,
        backbone.cpu(),
        [head.cpu() for head in exit_heads],
        {
            "source": "train_quweit_lut_early_exit_g1.py",
            "backbone_ckpt": args.backbone_ckpt,
            "exit_ckpt": args.exit_ckpt,
            "config": raw_backbone_ckpt["config"],
        },
        exit_cfg_list=[ExitConfig.from_payload(cfg).to_payload() for cfg in payload_exit_cfg],
        extra={
            "dataset": backbone_cfg.dataset,
            "train_mode": "alternating_cotrain",
            "cycles": int(args.cycles),
            "epochs_F": int(args.epochs_F),
            "epochs_H": int(args.epochs_H),
            "final_train_layers": list(final_train_layers_input),
            "final_train_layer_indices": list(final_train_layers),
            "train_classifier": bool(args.train_classifier),
            "train_exit_ids": list(train_exit_layer_ids),
            "train_exit_indices": list(train_exit_indices),
            "lr_backbone_F": float(args.lr_backbone_F),
            "lr_classifier_F": float(args.lr_classifier_F),
            "lr_exits_H": float(args.lr_exits_H),
            "best_metric_name": args.best_metric,
            "best_metric_value": float(best["metric"]),
            "best_eval_record": best["selected"],
            "eval_thrs": eval_thrs,
        },
    )

    backbone = backbone.to(device)
    exit_heads = [head.to(device) for head in exit_heads]
    profile = get_external_exit_profile(backbone, exit_heads, payload_exit_cfg)
    val_cache = collect_cascade_cache_quweit(
        backbone,
        val_loader,
        device,
        exit_heads=exit_heads,
        exit_cfg_list=payload_exit_cfg,
        use_prob_margin=args.use_prob_margin,
        profile=profile,
    )
    test_cache = collect_cascade_cache_quweit(
        backbone,
        test_loader,
        device,
        exit_heads=exit_heads,
        exit_cfg_list=payload_exit_cfg,
        use_prob_margin=args.use_prob_margin,
        profile=profile,
    )
    print(f"\n[saved] {args.path_out}")

    val_out = eval_cascade_cached_quweit(val_cache, eval_thrs)
    test_out = eval_cascade_cached_quweit(test_cache, eval_thrs)
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
            thrs = [thr if i == exit_id else eval_thrs[i] for i in range(len(eval_thrs))]
            out = eval_cascade_cached_quweit(val_cache, thrs)
            exit_acc = out["exit_accs"][exit_id]
            exit_acc_text = f"{exit_acc * 100:.2f}%" if exit_acc == exit_acc else "nan"
            print(f"  thr={thr:.2f} overall={out['overall_acc'] * 100:.2f}% exit_rate={out['exit_rates'][exit_id] * 100:.2f}% exit_acc={exit_acc_text}")

    if cascade_thr_grid:
        rows_val = []
        rows_test = []
        for thrs in itertools.product(*cascade_thr_grid):
            thrs = list(thrs)
            rows_val.append({"thrs": thrs, **eval_cascade_cached_quweit(val_cache, thrs)})
            rows_test.append({"thrs": thrs, **eval_cascade_cached_quweit(test_cache, thrs)})
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
