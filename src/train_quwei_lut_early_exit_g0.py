import argparse
import itertools
import os
from typing import List, Sequence

import torch
import torch.nn.functional as F

from src.core.linearExitHead import ExitHead
from src.core.multiLayerWNN import load_ckpt, save_ckpt_v2
from src.dataio.data import build_loaders_bits
from src.early_exit import eval_cascade_multi_exit, eval_overall_at_thr_multi_exit
from src.exit.analyze_hidden import analyze_hidden_for_exit, compute_mu_sigma, select_exit_keep_idx
from src.exit.ckpt_exit import ExitConfig


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


@torch.no_grad()
def cache_exit_features(model, loader, device, layer_idx, keep_idx, mu, sigma, use_norm: bool):
    model.eval()
    xs = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(device)
        _, h_list = model.forward_with_all_hidden(xb)
        h = h_list[layer_idx][:, keep_idx.to(h_list[layer_idx].device)]
        if use_norm:
            h = (h - mu.to(h.device)) / sigma.to(h.device)
        xs.append(h.detach().cpu())
        ys.append(yb.detach().cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


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
):
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    best_state = None
    best_val_acc = -1.0
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


def print_single_exit_sweep(title: str, rows: List[dict]):
    print(f"\n=== {title} ===")
    header = (
        "thr    exit%   overall%  exit_acc%  non_exit_acc%  "
        "m_mean  m_p95  avg_flops  avg_macs  avg_layers  "
        "exit_params  param_ovh  compute_ovh  saving  exited non_exited total"
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
            f"{row['avg_flops_per_sample']:>9.2f}  "
            f"{row['avg_macs_per_sample']:>8.2f}  "
            f"{row['avg_layers_executed_per_sample']:>10.4f}  "
            f"{int(row['total_exit_head_params']):>11d}  "
            f"{row['param_overhead_ratio']:>9.6f}  "
            f"{row['compute_overhead_ratio']:>11.6f}  "
            f"{row['compute_saving_ratio']:>6.4f}  "
            f"{row['exited']:>6d} "
            f"{row['non_exited']:>10d} "
            f"{row['total']:>5d}"
        )


def print_cascade_sweep(title: str, rows: List[dict]):
    print(f"\n=== {title} ===")
    header = (
        "thrs  overall%  exit_rates  final_rate  exit_accs  final_acc  "
        "avg_flops  avg_macs  avg_layers  exit_params  param_ovh  compute_ovh  saving"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        thr_text = "[" + ",".join(f"{thr:.2f}" for thr in row["thrs"]) + "]"
        exit_rate_text = "[" + ",".join(f"{r * 100:.2f}" for r in row["exit_rates"]) + "]"
        exit_acc_text = "[" + ",".join(f"{a * 100:.2f}" if a == a else "nan" for a in row["exit_accs"]) + "]"
        final_acc = row["final_acc"] * 100.0 if row["final_acc"] == row["final_acc"] else float("nan")
        print(
            f"{thr_text:<18} "
            f"{row['overall_acc'] * 100:>8.2f}  "
            f"{exit_rate_text:<22} "
            f"{row['final_rate'] * 100:>10.2f}  "
            f"{exit_acc_text:<20} "
            f"{final_acc:>9.2f}  "
            f"{row['avg_flops_per_sample']:>9.2f}  "
            f"{row['avg_macs_per_sample']:>8.2f}  "
            f"{row['avg_layers_executed_per_sample']:>10.4f}  "
            f"{int(row['total_exit_head_params']):>11d}  "
            f"{row['param_overhead_ratio']:>9.6f}  "
            f"{row['compute_overhead_ratio']:>11.6f}  "
            f"{row['compute_saving_ratio']:>6.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Train cached multi-exit heads for CIFAR10/CIFAR100 WNN backbones.")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--data_root", type=str, default="/Users/yi-chunchen/workspace/WNN_early_exit/datasets/")
    parser.add_argument("--backbone_ckpt", type=str, required=True)
    parser.add_argument("--path_out", type=str, required=True)

    parser.add_argument("--exit_layers", type=str, default="0")
    parser.add_argument("--k", type=str, default="256")
    parser.add_argument("--keep_mode", type=str, default="p*(1-p)*std")
    parser.add_argument("--exit_tau", type=str, default="1.0")
    parser.add_argument("--init_thr", type=str, default="0.5")
    parser.add_argument("--use_norm", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size_train", type=int, default=256)
    parser.add_argument("--batch_size_eval", type=int, default=512)
    parser.add_argument("--batch_size_cached", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z", type=int, default=32)

    parser.add_argument("--single_thr_list", type=str, default="0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0")
    parser.add_argument("--cascade_thr_grid", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print("[info] building clean bit-encoded loaders for exit training/eval")
    train_clean_loader, val_loader, test_loader, _, num_classes, _ = build_loaders_bits(
        dataset=args.dataset,
        root=args.data_root,
        batch_size_train=args.batch_size_train,
        batch_size_eval=args.batch_size_eval,
        val_ratio=args.val_ratio,
        seed=args.seed,
        z=args.z,
        device_for_encoding=device,
        shuffle_train=False,
    )

    model, backbone_cfg, ex_cfg, _ = load_ckpt(args.backbone_ckpt, device)
    if ex_cfg is not None:
        print("[warn] input checkpoint already has exit_config; this script will rebuild exits from scratch.")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    exit_layers = _parse_csv(args.exit_layers, int)
    num_hidden_layers = len(model.layers)
    bad_layers = [layer_idx for layer_idx in exit_layers if layer_idx < 0 or layer_idx >= num_hidden_layers]
    if bad_layers:
        raise ValueError(
            f"Invalid --exit_layers {bad_layers}; backbone exposes hidden layers [0, {num_hidden_layers - 1}]."
        )

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
        print(
            f"build/train exit layer={layer_idx} "
            f"k={k} keep_mode={keep_mode} exit_tau={exit_tau} init_thr={init_thr}"
        )
        print("=" * 88)

        mean_d, std_d, p1_d, bias = analyze_hidden_for_exit(
            model,
            train_clean_loader,
            device,
            layer_idx=layer_idx,
        )
        exit_keep_idx = select_exit_keep_idx(mean_d, std_d, p1_d, bias, k=k, keep_mode=keep_mode)
        mu, sigma = compute_mu_sigma(
            model,
            train_clean_loader,
            device,
            layer_idx=layer_idx,
            exit_keep_idx=exit_keep_idx,
        )

        x_train, y_train = cache_exit_features(model, train_clean_loader, device, layer_idx, exit_keep_idx, mu, sigma, args.use_norm)
        x_val, y_val = cache_exit_features(model, val_loader, device, layer_idx, exit_keep_idx, mu, sigma, args.use_norm)
        x_test, y_test = cache_exit_features(model, test_loader, device, layer_idx, exit_keep_idx, mu, sigma, args.use_norm)
        print(f"[cache] train={tuple(x_train.shape)} val={tuple(x_val.shape)} test={tuple(x_test.shape)}")

        head = ExitHead(
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
        backbone_cfg,
        exit_cfg_list=payload_exit_cfg,
        extra={
            "dataset": args.dataset,
            "train_mode": "clean_cached_exit_training",
            "note": "Exit analysis, feature caching, mu/sigma, and sweeps use clean train loader only.",
        },
    )
    print(f"\n[saved] {args.path_out}")

    for exit_id, layer_idx in enumerate(exit_layers):
        rows_val = []
        rows_test = []
        for thr in single_thr_list:
            rows_val.append(
                {
                    "thr": thr,
                    **eval_overall_at_thr_multi_exit(
                        model,
                        val_loader,
                        device,
                        thr=thr,
                        exit_id=exit_id,
                        exit_cfg_list=payload_exit_cfg,
                        exit_heads=exit_heads,
                        use_prob_margin=False,
                    ),
                }
            )
            rows_test.append(
                {
                    "thr": thr,
                    **eval_overall_at_thr_multi_exit(
                        model,
                        test_loader,
                        device,
                        thr=thr,
                        exit_id=exit_id,
                        exit_cfg_list=payload_exit_cfg,
                        exit_heads=exit_heads,
                        use_prob_margin=False,
                    ),
                }
            )

        print_single_exit_sweep(f"VAL single-exit sweep @ layer {layer_idx}", rows_val)
        print_single_exit_sweep(f"TEST single-exit sweep @ layer {layer_idx}", rows_test)

    if cascade_thr_grid:
        rows_val = []
        rows_test = []
        for thrs in itertools.product(*cascade_thr_grid):
            thrs = list(thrs)
            rows_val.append(
                {
                    "thrs": thrs,
                    **eval_cascade_multi_exit(
                        model,
                        val_loader,
                        device,
                        exit_heads=exit_heads,
                        exit_cfg_list=payload_exit_cfg,
                        thrs=thrs,
                        use_prob_margin=False,
                    ),
                }
            )
            rows_test.append(
                {
                    "thrs": thrs,
                    **eval_cascade_multi_exit(
                        model,
                        test_loader,
                        device,
                        exit_heads=exit_heads,
                        exit_cfg_list=payload_exit_cfg,
                        thrs=thrs,
                        use_prob_margin=False,
                    ),
                }
            )

        rows_val.sort(key=lambda row: row["overall_acc"], reverse=True)
        rows_test.sort(key=lambda row: row["overall_acc"], reverse=True)
        print_cascade_sweep("VAL cascade sweep", rows_val)
        print_cascade_sweep("TEST cascade sweep", rows_test)


if __name__ == "__main__":
    main()
