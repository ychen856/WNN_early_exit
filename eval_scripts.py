import argparse
import itertools
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from src.core.linearExitHead import build_exits_from_ckpt
from src.early_exit import eval_backbone_profile
from src.train_quweit_lut_early_exit_g0_ce import (
    build_clean_cifar_loaders,
    collect_cascade_cache,
    eval_cascade_cached,
    eval_overall_at_thr_cached,
    get_external_exit_profile,
    load_quweit_backbone_ckpt,
)


def parse_csv(s: str, cast=float) -> List:
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def parse_threshold_groups(s: str, num_exits: int, name: str) -> List[List[float]]:
    if not s.strip():
        return []
    groups = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if chunk:
            groups.append(parse_csv(chunk, float))
    if len(groups) == 1:
        return groups * num_exits
    if len(groups) != num_exits:
        raise ValueError(f"{name} expects 1 group or {num_exits} groups, got {len(groups)}")
    return groups


def unique_quantile_values(values: torch.Tensor, quantiles: List[float]) -> List[float]:
    if values.numel() == 0:
        return [0.0]
    out = [float(torch.quantile(values, q).item()) for q in quantiles]
    uniq = sorted(set(out))
    return uniq if uniq else [0.0]


@torch.no_grad()
def eval_epoch_quweit(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = F.cross_entropy(logits, yb)
        total_loss += float(loss.item()) * yb.size(0)
        total_correct += int((logits.argmax(dim=-1) == yb).sum().item())
        total += int(yb.size(0))
    return total_loss / max(total, 1), total_correct / max(total, 1)


def load_quweit_exits(exit_ckpt: str, device, num_classes: int):
    exit_heads, exit_cfg_list = build_exits_from_ckpt(exit_ckpt, device, num_classes=num_classes)
    payload_exit_cfg = [cfg.to_payload() for cfg in exit_cfg_list]
    return exit_heads, payload_exit_cfg


def sort_rows_for_efficiency(rows: List[dict]) -> List[dict]:
    return sorted(
        rows,
        key=lambda row: (
            row["avg_flops_per_sample"],
            row["avg_layers_executed_per_sample"],
            -row["overall_acc"],
        ),
    )


def get_efficiency_rows(rows: List[dict], baseline_overall: float, drop_pp: float) -> tuple[List[dict], float]:
    cutoff = float(baseline_overall) - float(drop_pp) / 100.0
    filtered = [row for row in rows if row["overall_acc"] >= cutoff]
    return sort_rows_for_efficiency(filtered), cutoff


def find_row_by_thrs(rows: List[dict], thrs: Sequence[float], tol: float = 1e-6):
    for row in rows:
        if len(row["thrs"]) != len(thrs):
            continue
        if all(abs(float(a) - float(b)) <= tol for a, b in zip(row["thrs"], thrs)):
            return row
    return None


def print_backbone_eval(name: str, loss: float, acc: float, avg_flops: float, avg_macs: float):
    print(
        f"[{name}] loss={loss:.4f} acc={acc * 100:.2f}% "
        f"avgFLOPs={avg_flops:.0f} avgMACs={avg_macs:.0f}"
    )


def print_cascade_summary(name: str, out: dict, thrs: Sequence[float]):
    print(
        f"[{name}] thrs={[round(float(x), 4) for x in thrs]} "
        f"overall={out['overall_acc'] * 100:.2f}% "
        f"exit_rates={[round(float(x), 4) for x in out['exit_rates']]} "
        f"exit_accs={[round(float(x), 4) if x == x else float('nan') for x in out['exit_accs']]} "
        f"final_rate={out['final_rate']:.4f} "
        f"final_acc={out['final_acc'] * 100:.2f}% "
        f"avgFLOPs={out['avg_flops_per_sample']:.0f} "
        f"avgMACs={out['avg_macs_per_sample']:.0f} "
        f"avgLayers={out['avg_layers_executed_per_sample']:.3f} "
        f"overhead={out['compute_overhead_ratio']:.4f}"
    )


def print_topk_table(title: str, rows: List[dict], top_k: int):
    print(f"\n=== {title} ===")
    if not rows:
        print("(empty)")
        return
    num_exits = len(rows[0]["thrs"])
    header_parts = [f"thr{i}" for i in range(num_exits)]
    header_parts += ["overall%"]
    for i in range(num_exits):
        header_parts += [f"exit{i}_rate%", f"exit{i}_acc%"]
    header_parts += ["final_rate%", "final_acc%", "avgFLOPs", "avgMACs", "avgLayers", "overhead"]
    header = "  ".join(f"{item:>11s}" for item in header_parts)
    print(header)
    print("-" * len(header))
    for row in rows[:top_k]:
        values = [f"{float(thr):>11.4f}" for thr in row["thrs"]]
        values.append(f"{row['overall_acc'] * 100:>11.2f}")
        for rate, acc in zip(row["exit_rates"], row["exit_accs"]):
            values.append(f"{rate * 100:>11.2f}")
            values.append(f"{acc * 100:>11.2f}" if acc == acc else f"{'nan':>11s}")
        values.append(f"{row['final_rate'] * 100:>11.2f}")
        values.append(f"{row['final_acc'] * 100:>11.2f}" if row["final_acc"] == row["final_acc"] else f"{'nan':>11s}")
        values.append(f"{row['avg_flops_per_sample']:>11.0f}")
        values.append(f"{row['avg_macs_per_sample']:>11.0f}")
        values.append(f"{row['avg_layers_executed_per_sample']:>11.3f}")
        values.append(f"{row['compute_overhead_ratio']:>11.4f}")
        print("  ".join(values))


def print_single_exit_scan_table(title: str, rows: List[dict]):
    print(f"\n=== {title} ===")
    if not rows:
        print("(empty)")
        return
    header_parts = ["thr", "overall%", "exit_rate%", "exited_acc%", "non_exit_acc%", "margin_mean", "margin_p95", "avgFLOPs", "avgMACs"]
    header = "  ".join(f"{item:>12s}" for item in header_parts)
    print(header)
    print("-" * len(header))
    for row in rows:
        values = [
            f"{row['thr']:>12.4f}",
            f"{row['overall_acc'] * 100:>12.2f}",
            f"{row['exit_rate'] * 100:>12.2f}",
            f"{row['exited_acc'] * 100:>12.2f}",
            f"{row['non_exited_acc'] * 100:>12.2f}" if row["non_exited_acc"] == row["non_exited_acc"] else f"{'nan':>12s}",
            f"{row['margin_mean']:>12.4f}",
            f"{row['margin_p95']:>12.4f}",
            f"{row['avg_flops_per_sample']:>12.0f}",
            f"{row['avg_macs_per_sample']:>12.0f}",
        ]
        print("  ".join(values))


def run_single_exit_scan(val_cache, test_cache, profile: dict, payload_exit_cfg: List[dict], thr_values: List[float]):
    for exit_id, cfg in enumerate(payload_exit_cfg):
        layer_idx = int(cfg["layer_idx"])
        rows_val = [{"thr": float(thr), **eval_overall_at_thr_cached(val_cache, profile, thr, exit_id=exit_id)} for thr in thr_values]
        rows_test = [{"thr": float(thr), **eval_overall_at_thr_cached(test_cache, profile, thr, exit_id=exit_id)} for thr in thr_values]
        rows_val.sort(key=lambda row: (-row["overall_acc"], row["thr"]))
        rows_test.sort(key=lambda row: (-row["overall_acc"], row["thr"]))
        print_single_exit_scan_table(f"G0 Single-Exit Scan VAL | exit={exit_id} layer={layer_idx}", rows_val)
        print_single_exit_scan_table(f"G0 Single-Exit Scan TEST | exit={exit_id} layer={layer_idx}", rows_test)


def run_grid_sweep(val_cache, test_cache, profile: dict, threshold_groups: List[List[float]], top_k: int, baseline_val: float, baseline_test: float):
    rows_val = []
    rows_test = []
    for thrs in itertools.product(*threshold_groups):
        thrs = list(thrs)
        rows_val.append({"thrs": thrs, **eval_cascade_cached(val_cache, profile, thrs)})
        rows_test.append({"thrs": thrs, **eval_cascade_cached(test_cache, profile, thrs)})
    rows_val.sort(key=lambda row: (-row["overall_acc"], row["avg_flops_per_sample"]))
    rows_test.sort(key=lambda row: (-row["overall_acc"], row["avg_flops_per_sample"]))
    print_topk_table("VAL Cascade Grid Sweep", rows_val, top_k)
    print_topk_table("TEST Cascade Grid Sweep", rows_test, top_k)
    val_eff_05, cut_val_05 = get_efficiency_rows(rows_val, baseline_val, 0.5)
    val_eff_10, cut_val_10 = get_efficiency_rows(rows_val, baseline_val, 1.0)
    test_eff_05, cut_test_05 = get_efficiency_rows(rows_test, baseline_test, 0.5)
    test_eff_10, cut_test_10 = get_efficiency_rows(rows_test, baseline_test, 1.0)
    print_topk_table(f"VAL Efficiency Top-K @ overall >= {cut_val_05 * 100:.2f}%", val_eff_05, top_k)
    print_topk_table(f"VAL Efficiency Top-K @ overall >= {cut_val_10 * 100:.2f}%", val_eff_10, top_k)
    print_topk_table(f"TEST Efficiency Top-K @ overall >= {cut_test_05 * 100:.2f}%", test_eff_05, top_k)
    print_topk_table(f"TEST Efficiency Top-K @ overall >= {cut_test_10 * 100:.2f}%", test_eff_10, top_k)


def run_quantile_sweep(val_cache, test_cache, profile: dict, quantile_groups: List[List[float]], top_k: int, max_combinations: int, baseline_val: float, baseline_test: float):
    thr_groups = [unique_quantile_values(margins, qs) for margins, qs in zip(val_cache["margins"], quantile_groups)]
    print("[quantile-sweep] threshold groups:", thr_groups)
    num_combinations = 1
    for group in thr_groups:
        num_combinations *= max(len(group), 1)
    print(f"[quantile-sweep] num_combinations={num_combinations}")
    if num_combinations > max_combinations:
        raise ValueError(
            f"Too many threshold combinations: {num_combinations} > max_combinations={max_combinations}. "
            "Reduce --cascade_quantiles."
        )
    rows_val = []
    rows_test = []
    for thrs in itertools.product(*thr_groups):
        thrs = list(thrs)
        rows_val.append({"thrs": thrs, **eval_cascade_cached(val_cache, profile, thrs)})
        rows_test.append({"thrs": thrs, **eval_cascade_cached(test_cache, profile, thrs)})
    rows_val.sort(key=lambda row: (-row["overall_acc"], row["avg_flops_per_sample"]))
    rows_test.sort(key=lambda row: (-row["overall_acc"], row["avg_flops_per_sample"]))
    print_topk_table("VAL Quantile Sweep Top-K", rows_val, top_k)
    print_topk_table("TEST Quantile Sweep Top-K", rows_test, top_k)
    val_eff_05, cut_val_05 = get_efficiency_rows(rows_val, baseline_val, 0.5)
    val_eff_10, cut_val_10 = get_efficiency_rows(rows_val, baseline_val, 1.0)
    test_eff_05, cut_test_05 = get_efficiency_rows(rows_test, baseline_test, 0.5)
    test_eff_10, cut_test_10 = get_efficiency_rows(rows_test, baseline_test, 1.0)
    print_topk_table(f"VAL Efficiency Top-K @ overall >= {cut_val_05 * 100:.2f}%", val_eff_05, top_k)
    print_topk_table(f"VAL Efficiency Top-K @ overall >= {cut_val_10 * 100:.2f}%", val_eff_10, top_k)
    print_topk_table(f"TEST Efficiency Top-K @ overall >= {cut_test_05 * 100:.2f}%", test_eff_05, top_k)
    print_topk_table(f"TEST Efficiency Top-K @ overall >= {cut_test_10 * 100:.2f}%", test_eff_10, top_k)
    if rows_val:
        best_val = rows_val[0]
        best_test = find_row_by_thrs(rows_test, best_val["thrs"])
        print(
            f"\n[best-val-thrs] thrs={[round(float(x), 4) for x in best_val['thrs']]} "
            f"val_overall={best_val['overall_acc'] * 100:.2f}% "
            f"val_avgFLOPs={best_val['avg_flops_per_sample']:.0f}"
        )
        if best_test is not None:
            print(
                f"[best-val-thrs:test] overall={best_test['overall_acc'] * 100:.2f}% "
                f"avgFLOPs={best_test['avg_flops_per_sample']:.0f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate QuWeiT backbone + external exit-head checkpoints.")
    parser.add_argument("--backbone_ckpt", type=str, required=True, help="Checkpoint produced by train_quweit_lut_backbone_v2.py")
    parser.add_argument("--exit_ckpt", type=str, required=True, help="Checkpoint produced by train_quweit_lut_early_exit_g0_ce.py or similar")
    parser.add_argument("--use_ema_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size_probe", type=int, default=128)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--single_thr_list", type=str, default="0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0")
    parser.add_argument("--cascade_thr_grid", type=str, default="")
    parser.add_argument("--cascade_quantiles", type=str, default="0.0,0.25,0.5,0.75,0.9,0.95")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_combinations", type=int, default=2000)
    parser.add_argument("--sweep_selection_baseline_overall_val", type=float, default=None)
    parser.add_argument("--sweep_selection_baseline_overall_test", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    backbone, backbone_cfg, _ = load_quweit_backbone_ckpt(args.backbone_ckpt, device, use_ema=args.use_ema_backbone)
    exit_heads, payload_exit_cfg = load_quweit_exits(args.exit_ckpt, device, num_classes=backbone_cfg.num_classes)
    if not exit_heads:
        raise ValueError("No exit heads found in --exit_ckpt")

    _, val_loader, test_loader, num_classes = build_clean_cifar_loaders(
        backbone_cfg,
        batch_size_probe=args.batch_size_probe,
        batch_size_eval=args.batch_size_eval,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    if num_classes != backbone_cfg.num_classes:
        raise ValueError(f"Dataset num_classes mismatch: loaders={num_classes}, cfg={backbone_cfg.num_classes}")

    print(f"[info] backbone_ckpt={args.backbone_ckpt}")
    print(f"[info] exit_ckpt={args.exit_ckpt}")
    print(f"[info] dataset={backbone_cfg.dataset}")
    print(f"[info] num_exits={len(exit_heads)} exit_layers={[int(cfg['layer_idx']) for cfg in payload_exit_cfg]}")

    val_loss, val_acc = eval_epoch_quweit(backbone, val_loader, device)
    test_loss, test_acc = eval_epoch_quweit(backbone, test_loader, device)
    val_profile = eval_backbone_profile(backbone, val_loader, device)
    test_profile = eval_backbone_profile(backbone, test_loader, device)
    print_backbone_eval(
        "backbone:val",
        val_loss,
        val_acc,
        float(val_profile["avg_flops_per_sample"]),
        float(val_profile["avg_macs_per_sample"]),
    )
    print_backbone_eval(
        "backbone:test",
        test_loss,
        test_acc,
        float(test_profile["avg_flops_per_sample"]),
        float(test_profile["avg_macs_per_sample"]),
    )

    baseline_val = float(args.sweep_selection_baseline_overall_val) / 100.0 if args.sweep_selection_baseline_overall_val is not None else float(val_acc)
    baseline_test = float(args.sweep_selection_baseline_overall_test) / 100.0 if args.sweep_selection_baseline_overall_test is not None else float(test_acc)
    print(
        f"[info] VAL sweep baseline overall={baseline_val * 100:.2f}% "
        f"(cuts: {(baseline_val - 0.005) * 100:.2f}%, {(baseline_val - 0.010) * 100:.2f}%)"
    )
    print(
        f"[info] TEST sweep baseline overall={baseline_test * 100:.2f}% "
        f"(cuts: {(baseline_test - 0.005) * 100:.2f}%, {(baseline_test - 0.010) * 100:.2f}%)"
    )

    profile = get_external_exit_profile(backbone, exit_heads, payload_exit_cfg)
    val_cache = collect_cascade_cache(backbone, val_loader, device, exit_heads=exit_heads, exit_cfg_list=payload_exit_cfg)
    test_cache = collect_cascade_cache(backbone, test_loader, device, exit_heads=exit_heads, exit_cfg_list=payload_exit_cfg)

    ckpt_thrs = [float(cfg["thr"]) for cfg in payload_exit_cfg if "thr" in cfg]
    if len(ckpt_thrs) == len(payload_exit_cfg):
        val_out = eval_cascade_cached(val_cache, profile, ckpt_thrs)
        test_out = eval_cascade_cached(test_cache, profile, ckpt_thrs)
        print_cascade_summary("cascade:val@ckpt_thr", val_out, ckpt_thrs)
        print_cascade_summary("cascade:test@ckpt_thr", test_out, ckpt_thrs)
    else:
        print("[info] exit_ckpt does not contain full per-exit threshold config; skipping ckpt-threshold cascade eval.")

    single_thr_list = parse_csv(args.single_thr_list, float)
    if single_thr_list:
        run_single_exit_scan(val_cache, test_cache, profile, payload_exit_cfg, single_thr_list)

    cascade_thr_grid = parse_threshold_groups(args.cascade_thr_grid, len(exit_heads), "cascade_thr_grid")
    if cascade_thr_grid:
        run_grid_sweep(val_cache, test_cache, profile, cascade_thr_grid, args.top_k, baseline_val, baseline_test)

    cascade_quantiles = parse_threshold_groups(args.cascade_quantiles, len(exit_heads), "cascade_quantiles")
    if cascade_quantiles:
        run_quantile_sweep(
            val_cache,
            test_cache,
            profile,
            cascade_quantiles,
            args.top_k,
            args.max_combinations,
            baseline_val,
            baseline_test,
        )


if __name__ == "__main__":
    main()
