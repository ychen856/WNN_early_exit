import argparse
import itertools
from typing import List, Optional, Sequence

import torch

from src.core.linearExitHead import build_exits_from_ckpt
from src.core.multiLayerWNN import build_backbone_from_ckpt
from src.dataio.data import build_loaders_bits
from src.early_exit import (
    collect_exit_margins,
    eval_backbone_profile,
    eval_cascade_multi_exit,
    eval_overall_at_thr_multi_exit,
    make_thr_candidates_from_quantiles,
)
from test.eval import eval_epoch


def parse_float_list(s: str, n: Optional[int] = None, name: str = "list") -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if n is not None and len(vals) != n:
        raise ValueError(f"{name} must have length {n}, got {len(vals)}: {vals}")
    return vals


def parse_threshold_groups(s: str, num_exits: int, name: str) -> List[List[float]]:
    if not s.strip():
        return []
    groups = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if chunk:
            groups.append(parse_float_list(chunk, name=name))
    if len(groups) == 1:
        return groups * num_exits
    if len(groups) != num_exits:
        raise ValueError(f"{name} expects 1 group or {num_exits} groups, got {len(groups)}")
    return groups


def load_backbone_and_exits(ckpt_path: str, device):
    backbone, bb_cfg, extra = build_backbone_from_ckpt(ckpt_path, device)
    num_classes = int(bb_cfg["num_classes"])
    exit_heads, exit_cfg_list = build_exits_from_ckpt(ckpt_path, device, num_classes=num_classes)
    payload_exit_cfg = [cfg.to_payload() for cfg in exit_cfg_list]
    return backbone, bb_cfg, extra, exit_heads, payload_exit_cfg


def build_wnn_loaders(
    dataset: str,
    data_root: str,
    batch_size_eval: int,
    val_ratio: float,
    seed: int,
    z: int,
    device,
):
    train_loader, val_loader, test_loader, _, _, _ = build_loaders_bits(
        dataset=dataset,
        root=data_root,
        batch_size_train=256,
        batch_size_eval=batch_size_eval,
        val_ratio=val_ratio,
        seed=seed,
        z=z,
        device_for_encoding=device,
        shuffle_train=True,
    )
    return train_loader, val_loader, test_loader


def evaluate_thresholds(model, loader, device, exit_heads, payload_exit_cfg, thrs, use_prob_margin: bool):
    return eval_cascade_multi_exit(
        model,
        loader,
        device,
        exit_heads=exit_heads,
        exit_cfg_list=payload_exit_cfg,
        thrs=thrs,
        use_prob_margin=use_prob_margin,
        log_margins=False,
    )


def compute_exp_layers(out: dict) -> float:
    exit_rates = list(out["exit_rates"])
    final_rate = float(out["final_rate"])
    if len(exit_rates) >= 2:
        return float(exit_rates[0]) * 1.0 + float(exit_rates[1]) * 2.0 + final_rate * 2.0
    if len(exit_rates) == 1:
        return float(exit_rates[0]) * 1.0 + final_rate * 2.0
    return final_rate


def sort_rows_for_efficiency(rows: List[dict]) -> List[dict]:
    return sorted(rows, key=lambda row: (row["exp_layers"], row["final_rate"], -row["overall_acc"]))


def get_efficiency_rows(rows: List[dict], baseline_overall: float, drop_pp: float) -> tuple[List[dict], float]:
    cutoff = float(baseline_overall) - float(drop_pp) / 100.0
    filtered = [row for row in rows if row["overall_acc"] >= cutoff]
    return sort_rows_for_efficiency(filtered), cutoff


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
        f"exp_layers={out.get('exp_layers', float('nan')):.4f} "
        f"avgFLOPs={out.get('avg_flops_per_sample', float('nan')):.0f} "
        f"avgMACs={out.get('avg_macs_per_sample', float('nan')):.0f}"
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
    header_parts += ["final_rate%", "final_acc%", "expLayers", "avgFLOPs", "avgMACs"]
    header = "  ".join(f"{item:>11s}" for item in header_parts)
    print(header)
    print("-" * len(header))

    for row in rows[:top_k]:
        values = [f"{float(thr):>11.4f}" for thr in row["thrs"]]
        values.append(f"{row['overall_acc'] * 100:>11.2f}")
        for rate, acc in zip(row["exit_rates"], row["exit_accs"]):
            acc_text = f"{acc * 100:>11.2f}" if acc == acc else f"{'nan':>11s}"
            values.append(f"{rate * 100:>11.2f}")
            values.append(acc_text)
        final_acc = row["final_acc"]
        final_acc_text = f"{final_acc * 100:>11.2f}" if final_acc == final_acc else f"{'nan':>11s}"
        values.append(f"{row['final_rate'] * 100:>11.2f}")
        values.append(final_acc_text)
        values.append(f"{row.get('exp_layers', float('nan')):>11.4f}")
        values.append(f"{row.get('avg_flops_per_sample', float('nan')):>11.0f}")
        values.append(f"{row.get('avg_macs_per_sample', float('nan')):>11.0f}")
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
        non_exit_acc = row["non_exited_acc"]
        non_exit_acc_text = f"{non_exit_acc * 100:>12.2f}" if non_exit_acc == non_exit_acc else f"{'nan':>12s}"
        values = [
            f"{row['thr']:>12.4f}",
            f"{row['overall_acc'] * 100:>12.2f}",
            f"{row['exit_rate'] * 100:>12.2f}",
            f"{row['exited_acc'] * 100:>12.2f}",
            non_exit_acc_text,
            f"{row['margin_mean']:>12.4f}",
            f"{row['margin_p95']:>12.4f}",
            f"{row.get('avg_flops_per_sample', float('nan')):>12.0f}",
            f"{row.get('avg_macs_per_sample', float('nan')):>12.0f}",
        ]
        print("  ".join(values))


def run_single_exit_scan(
    model,
    val_loader,
    test_loader,
    device,
    *,
    exit_heads,
    payload_exit_cfg,
    thr_values: List[float],
):
    for exit_id, cfg in enumerate(payload_exit_cfg):
        layer_idx = int(cfg["layer_idx"])
        rows_val = []
        rows_test = []
        for thr in thr_values:
            out_val = eval_overall_at_thr_multi_exit(
                model,
                val_loader,
                device,
                thr=thr,
                exit_id=exit_id,
                exit_cfg_list=payload_exit_cfg,
                exit_heads=exit_heads,
                use_prob_margin=False,
            )
            out_test = eval_overall_at_thr_multi_exit(
                model,
                test_loader,
                device,
                thr=thr,
                exit_id=exit_id,
                exit_cfg_list=payload_exit_cfg,
                exit_heads=exit_heads,
                use_prob_margin=False,
            )
            rows_val.append({"thr": float(thr), **out_val})
            rows_test.append({"thr": float(thr), **out_test})

        rows_val.sort(key=lambda row: (-row["overall_acc"], row["thr"]))
        rows_test.sort(key=lambda row: (-row["overall_acc"], row["thr"]))
        print_single_exit_scan_table(f"G0 Single-Exit Scan VAL | exit={exit_id} layer={layer_idx}", rows_val)
        print_single_exit_scan_table(f"G0 Single-Exit Scan TEST | exit={exit_id} layer={layer_idx}", rows_test)


def find_row_by_thrs(rows: List[dict], thrs: Sequence[float], tol: float = 1e-6):
    for row in rows:
        if len(row["thrs"]) != len(thrs):
            continue
        if all(abs(float(a) - float(b)) <= tol for a, b in zip(row["thrs"], thrs)):
            return row
    return None


def run_quantile_sweep(
    model,
    val_loader,
    test_loader,
    device,
    *,
    exit_heads,
    payload_exit_cfg,
    quantile_groups: List[List[float]],
    use_prob_margin: bool,
    top_k: int,
    max_combinations: int,
    baseline_overall_val: float,
    baseline_overall_test: float,
):
    margins = collect_exit_margins(
        model=model,
        loader=val_loader,
        device=device,
        exit_heads=exit_heads,
        exit_cfg_list=payload_exit_cfg,
    )
    thr_groups = [
        make_thr_candidates_from_quantiles(margin, qs=tuple(qs))
        for margin, qs in zip(margins, quantile_groups)
    ]
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
        out_val = evaluate_thresholds(model, val_loader, device, exit_heads, payload_exit_cfg, thrs, use_prob_margin)
        out_test = evaluate_thresholds(model, test_loader, device, exit_heads, payload_exit_cfg, thrs, use_prob_margin)
        rows_val.append(
            {
                "thrs": thrs,
                "overall_acc": float(out_val["overall_acc"]),
                "exit_rates": list(out_val["exit_rates"]),
                "exit_accs": list(out_val["exit_accs"]),
                "final_rate": float(out_val["final_rate"]),
                "final_acc": float(out_val["final_acc"]),
                "exp_layers": compute_exp_layers(out_val),
                "avg_flops_per_sample": float(out_val.get("avg_flops_per_sample", float("nan"))),
                "avg_macs_per_sample": float(out_val.get("avg_macs_per_sample", float("nan"))),
            }
        )
        rows_test.append(
            {
                "thrs": thrs,
                "overall_acc": float(out_test["overall_acc"]),
                "exit_rates": list(out_test["exit_rates"]),
                "exit_accs": list(out_test["exit_accs"]),
                "final_rate": float(out_test["final_rate"]),
                "final_acc": float(out_test["final_acc"]),
                "exp_layers": compute_exp_layers(out_test),
                "avg_flops_per_sample": float(out_test.get("avg_flops_per_sample", float("nan"))),
                "avg_macs_per_sample": float(out_test.get("avg_macs_per_sample", float("nan"))),
            }
        )

    rows_val.sort(key=lambda row: (-row["overall_acc"], row["exp_layers"]))
    rows_test.sort(key=lambda row: (-row["overall_acc"], row["exp_layers"]))
    print_topk_table("VAL Quantile Sweep Top-K", rows_val, top_k=top_k)
    print_topk_table("TEST Quantile Sweep Top-K", rows_test, top_k=top_k)

    efficiency_rows_val_05, cutoff_val_05 = get_efficiency_rows(rows_val, baseline_overall_val, 0.5)
    efficiency_rows_val_10, cutoff_val_10 = get_efficiency_rows(rows_val, baseline_overall_val, 1.0)
    print_topk_table(f"VAL Efficiency Top-K @ overall >= {cutoff_val_05 * 100:.2f}%", efficiency_rows_val_05, top_k=top_k)
    print_topk_table(f"VAL Efficiency Top-K @ overall >= {cutoff_val_10 * 100:.2f}%", efficiency_rows_val_10, top_k=top_k)

    efficiency_rows_test_05, cutoff_test_05 = get_efficiency_rows(rows_test, baseline_overall_test, 0.5)
    efficiency_rows_test_10, cutoff_test_10 = get_efficiency_rows(rows_test, baseline_overall_test, 1.0)
    print_topk_table(
        f"TEST Efficiency Top-K @ overall >= {cutoff_test_05 * 100:.2f}%",
        efficiency_rows_test_05,
        top_k=top_k,
    )
    print_topk_table(
        f"TEST Efficiency Top-K @ overall >= {cutoff_test_10 * 100:.2f}%",
        efficiency_rows_test_10,
        top_k=top_k,
    )

    if rows_val:
        best_val = rows_val[0]
        matched_test = find_row_by_thrs(rows_test, best_val["thrs"])
        print(
            f"\n[best-val-thrs] thrs={[round(float(x), 4) for x in best_val['thrs']]} "
            f"val_overall={best_val['overall_acc'] * 100:.2f}% "
            f"val_exp_layers={best_val['exp_layers']:.4f}"
        )
        if matched_test is not None:
            print(
                f"[best-val-thrs:test] overall={matched_test['overall_acc'] * 100:.2f}% "
                f"exp_layers={matched_test['exp_layers']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate WNN backbone and multi-exit ckpt on val/test, including quantile sweep.")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint saved by save_ckpt_v2 containing backbone and exit heads.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name. If omitted, try ckpt extra['dataset'], else FMNIST.")
    parser.add_argument("--data_root", type=str, default="/Users/yi-chunchen/workspace/WNN_early_exit/datasets/")
    parser.add_argument("--batch_size_eval", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z", type=int, default=32)
    parser.add_argument("--use_prob_margin", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cascade_quantiles", type=str, default="0.0,0.25,0.5,0.75,0.9,0.95")
    parser.add_argument("--single_thr_list", type=str, default="0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_combinations", type=int, default=2000)
    parser.add_argument("--sweep_selection_baseline_overall_val", type=float, default=None,
                        help="VAL baseline overall accuracy in percent for efficiency top-k cutoff. If omitted, use backbone val acc.")
    parser.add_argument("--sweep_selection_baseline_overall_test", type=float, default=None,
                        help="TEST baseline overall accuracy in percent for efficiency top-k cutoff. If omitted, use backbone test acc.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, _, extra, exit_heads, payload_exit_cfg = load_backbone_and_exits(args.ckpt, device)
    dataset = args.dataset or extra.get("dataset") or "FMNIST"

    print(f"[info] ckpt={args.ckpt}")
    print(f"[info] dataset={dataset}")
    print(f"[info] num_exits={len(exit_heads)}")
    print(f"[info] exit_layers={[int(cfg['layer_idx']) for cfg in payload_exit_cfg]}")

    _, val_loader, test_loader = build_wnn_loaders(
        dataset=dataset,
        data_root=args.data_root,
        batch_size_eval=args.batch_size_eval,
        val_ratio=args.val_ratio,
        seed=args.seed,
        z=args.z,
        device=device,
    )

    val_loss, val_acc = eval_epoch(backbone, val_loader, device)
    test_loss, test_acc = eval_epoch(backbone, test_loader, device)
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
    sweep_selection_baseline_overall_val = (
        float(args.sweep_selection_baseline_overall_val) / 100.0
        if args.sweep_selection_baseline_overall_val is not None
        else float(val_acc)
    )
    sweep_selection_baseline_overall_test = (
        float(args.sweep_selection_baseline_overall_test) / 100.0
        if args.sweep_selection_baseline_overall_test is not None
        else float(test_acc)
    )
    print(
        f"[info] VAL sweep baseline overall={sweep_selection_baseline_overall_val * 100:.2f}% "
        f"(cuts: {(sweep_selection_baseline_overall_val - 0.005) * 100:.2f}%, {(sweep_selection_baseline_overall_val - 0.010) * 100:.2f}%)"
    )
    print(
        f"[info] TEST sweep baseline overall={sweep_selection_baseline_overall_test * 100:.2f}% "
        f"(cuts: {(sweep_selection_baseline_overall_test - 0.005) * 100:.2f}%, {(sweep_selection_baseline_overall_test - 0.010) * 100:.2f}%)"
    )

    ckpt_thrs = []
    has_ckpt_thrs = True
    for cfg in payload_exit_cfg:
        if "thr" not in cfg:
            has_ckpt_thrs = False
            break
        ckpt_thrs.append(float(cfg["thr"]))

    if has_ckpt_thrs:
        out_val = evaluate_thresholds(backbone, val_loader, device, exit_heads, payload_exit_cfg, ckpt_thrs, args.use_prob_margin)
        out_test = evaluate_thresholds(backbone, test_loader, device, exit_heads, payload_exit_cfg, ckpt_thrs, args.use_prob_margin)
        out_val["exp_layers"] = compute_exp_layers(out_val)
        out_test["exp_layers"] = compute_exp_layers(out_test)
        print_cascade_summary("cascade:val@ckpt_thr", out_val, ckpt_thrs)
        print_cascade_summary("cascade:test@ckpt_thr", out_test, ckpt_thrs)
    else:
        print("[info] ckpt does not contain per-exit 'thr'; skipping direct cascade eval at ckpt thresholds.")

    single_thr_list = parse_float_list(args.single_thr_list, name="single_thr_list")
    if single_thr_list:
        run_single_exit_scan(
            backbone,
            val_loader,
            test_loader,
            device,
            exit_heads=exit_heads,
            payload_exit_cfg=payload_exit_cfg,
            thr_values=single_thr_list,
        )

    quantile_groups = parse_threshold_groups(args.cascade_quantiles, len(exit_heads), "cascade_quantiles")
    if quantile_groups:
        run_quantile_sweep(
            backbone,
            val_loader,
            test_loader,
            device,
            exit_heads=exit_heads,
            payload_exit_cfg=payload_exit_cfg,
            quantile_groups=quantile_groups,
            use_prob_margin=args.use_prob_margin,
            top_k=args.top_k,
            max_combinations=args.max_combinations,
            baseline_overall_val=sweep_selection_baseline_overall_val,
            baseline_overall_test=sweep_selection_baseline_overall_test,
        )


if __name__ == "__main__":
    main()
