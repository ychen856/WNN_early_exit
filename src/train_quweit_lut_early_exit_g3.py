import argparse
import copy
import itertools
from typing import List

import torch
import torch.nn.functional as F

from src.core.multiLayerWNN import save_ckpt_v2
from src.early_exit import _head_logits_from_hidden_trainable, lambda_schedule_linear
from src.exit.ckpt_exit import ExitConfig
from src.train_quweit_lut_early_exit_g0_ce import build_clean_cifar_loaders, get_external_exit_profile
from src.train_quweit_lut_early_exit_g2 import (
    _broadcast,
    _ensure_dir,
    _parse_csv,
    _parse_threshold_groups,
    collect_cascade_cache_quweit,
    eval_cascade_cached_quweit,
    eval_cascade_quweit,
    load_quweit_model_with_exits,
    print_cascade_quantile_sweep,
    sweep_cascade_by_quantile,
    forward_with_all_hidden,
    set_requires_grad,
)


def train_g3_quweit(
    model,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs: int,
    lr_backbone: float,
    lr_classifier: float,
    lr_exits: float,
    final_lambda_exit: float,
    lambda_warmup: int,
    weight_decay: float,
    grad_clip: float,
    exit_heads: List[torch.nn.Module],
    payload_exit_cfg: List[dict],
    thrs: List[float],
    use_prob_margin: bool = False,
    lambda_exits: List[float] | None = None,
):
    model = model.to(device)
    num_exits = len(exit_heads)
    assert len(payload_exit_cfg) == num_exits
    assert len(thrs) == num_exits

    if lambda_exits is None:
        lambda_exits = [1.0] * num_exits
    else:
        assert len(lambda_exits) == num_exits

    set_requires_grad(model, True)
    exit_heads = [head.to(device) for head in exit_heads]
    for head in exit_heads:
        set_requires_grad(head, True)

    params_backbone = [p for block in model.blocks for p in block.parameters() if p.requires_grad]
    params_classifier = [p for p in model.head.parameters() if p.requires_grad]
    if hasattr(model, "norm"):
        params_classifier.extend([p for p in model.norm.parameters() if p.requires_grad])
    params_exits = [p for head in exit_heads for p in head.parameters() if p.requires_grad]

    print(
        f"[g3] trainable params: backbone={sum(p.numel() for p in params_backbone)} "
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
        lam = lambda_schedule_linear(epoch, warmup=lambda_warmup, final_lambda=final_lambda_exit)
        model.train()
        for head in exit_heads:
            head.train()

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            final_logits, h_list = forward_with_all_hidden(model, xb)
            loss_final = F.cross_entropy(final_logits, yb)

            loss_exit_sum = 0.0
            for i, (cfg, head) in enumerate(zip(payload_exit_cfg, exit_heads)):
                layer_idx = int(cfg["layer_idx"]) - 1
                logits_i = _head_logits_from_hidden_trainable(head, h_list[layer_idx], device)
                loss_exit_i = F.cross_entropy(logits_i, yb)
                loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * loss_exit_i

            loss = loss_final + float(lam) * loss_exit_sum
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], grad_clip)
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
        print(
            f"[G3] Ep{epoch:03d} lambda={lam:.3f} "
            f"| overall@{tuple(float(x) for x in thrs)} va={out_val['overall_acc'] * 100:.2f} "
            f"| exit_rates={out_val['exit_rates']} final_rate={out_val['final_rate']:.4f}"
        )

        if out_val["overall_acc"] > best["val_overall_acc"]:
            best["val_overall_acc"] = float(out_val["overall_acc"])
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
    parser = argparse.ArgumentParser(description="QuWeiT g3 joint co-train from a g2 checkpoint.")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Checkpoint produced by train_quweit_lut_early_exit_g2.py")
    parser.add_argument("--path_out", type=str, required=True)

    parser.add_argument("--batch_size_train", type=int, default=128)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_classifier", type=float, default=3e-4)
    parser.add_argument("--lr_exits", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--final_lambda_exit", type=float, default=0.3)
    parser.add_argument("--lambda_warmup", type=int, default=10)
    parser.add_argument("--lambda_exits", type=str, default="1.0")

    parser.add_argument("--thr", type=str, default="", help="comma-separated thresholds per exit; empty means use checkpoint values")
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

    lambda_exits = _broadcast(_parse_csv(args.lambda_exits, float), len(payload_exit_cfg), "lambda_exits")
    single_thr_list = _parse_csv(args.single_thr_list, float)
    cascade_thr_grid = _parse_threshold_groups(args.cascade_thr_grid, len(payload_exit_cfg), "cascade_thr_grid")
    cascade_quantile_groups = _parse_threshold_groups(args.cascade_quantiles, len(payload_exit_cfg), "cascade_quantiles")

    print("[info] loader settings "
          f"train_batch={args.batch_size_train} eval_batch={args.batch_size_eval} "
          f"num_workers={args.num_workers} pin_memory={args.pin_memory}")
    print(
        f"[info] g3 plan thrs={thrs} final_lambda_exit={args.final_lambda_exit} "
        f"lambda_warmup={args.lambda_warmup} lambda_exits={lambda_exits}"
    )

    model, exit_heads, best = train_g3_quweit(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_classifier=args.lr_classifier,
        lr_exits=args.lr_exits,
        final_lambda_exit=args.final_lambda_exit,
        lambda_warmup=args.lambda_warmup,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
        thrs=thrs,
        use_prob_margin=args.use_prob_margin,
        lambda_exits=lambda_exits,
    )
    print(f"[g3] best val overall acc = {best['val_overall_acc'] * 100:.2f}%")

    _ensure_dir(args.path_out)
    save_ckpt_v2(
        args.path_out,
        model.cpu(),
        [head.cpu() for head in exit_heads],
        raw_ckpt["backbone_cfg"],
        exit_cfg_list=[ExitConfig.from_payload(cfg).to_payload() for cfg in payload_exit_cfg],
        extra={
            "dataset": backbone_cfg.dataset,
            "train_mode": "g3_joint_cotrain",
            "source_ckpt": args.model_ckpt,
            "final_lambda_exit": float(args.final_lambda_exit),
            "lambda_warmup": int(args.lambda_warmup),
            "lambda_exits": [float(x) for x in lambda_exits],
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
