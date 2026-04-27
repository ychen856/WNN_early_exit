import argparse
import os
from typing import List, Sequence

import torch
import torch.nn.functional as F

from src.core.linearExitHead import build_exits_from_ckpt
from src.core.multiLayerWNN import save_ckpt_v2
from src.early_exit import _head_logits_from_hidden_trainable
from src.exit.ckpt_exit import ExitConfig
from src.train_quweit_lut_backbone_v2 import QuWeiTViT
from src.train_quweit_lut_early_exit_g0 import build_clean_cifar_loaders, load_quweit_backbone_ckpt


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


def load_quweit_exit_pack(path: str, device, num_classes: int):
    exit_heads, exit_cfg_list = build_exits_from_ckpt(path, device, num_classes=num_classes)
    return exit_heads, exit_cfg_list


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


def cotrain_g1_stage_quweit(
    model,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs: int,
    layer_idx: int,
    exit_id: int,
    lr_layer: float,
    lr_exit: float,
    lambda_exit: float,
    use_final_loss: bool,
    lambda_final: float,
    thrs: List[float],
    weight_decay: float,
    grad_clip: float,
    exit_heads: List[torch.nn.Module],
    payload_exit_cfg: List[dict],
    use_prob_margin: bool = False,
):
    assert len(exit_heads) == len(payload_exit_cfg)
    assert 0 <= layer_idx < len(model.blocks)
    assert 0 <= exit_id < len(exit_heads)

    model = model.to(device)

    set_requires_grad(model, False)
    set_requires_grad(model.blocks[layer_idx], True)
    for head in exit_heads:
        set_requires_grad(head, False)
    exit_heads[exit_id] = exit_heads[exit_id].to(device)
    set_requires_grad(exit_heads[exit_id], True)

    if hasattr(model, "head"):
        set_requires_grad(model.head, False)
    if hasattr(model, "norm"):
        set_requires_grad(model.norm, False)
    if hasattr(model, "patch_embed"):
        set_requires_grad(model.patch_embed, False)

    params_layer = [p for p in model.blocks[layer_idx].parameters() if p.requires_grad]
    params_exit = [p for p in exit_heads[exit_id].parameters() if p.requires_grad]
    if not params_layer:
        raise ValueError(f"No trainable params in model.blocks[{layer_idx}]")
    if not params_exit:
        raise ValueError(f"No trainable params in exit_heads[{exit_id}]")

    optimizer = torch.optim.AdamW(
        [
            {"params": params_layer, "lr": lr_layer, "weight_decay": weight_decay},
            {"params": params_exit, "lr": lr_exit, "weight_decay": weight_decay},
        ]
    )

    best = {"val_overall_acc": -1.0, "state": None, "exit_states": None}

    for epoch in range(num_epochs):
        model.train()
        exit_heads[exit_id].train()

        loss_sum = 0.0
        total = 0
        correct_exit = 0
        correct_final = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            final_logits, h_list = forward_with_all_hidden(model, xb)
            exit_logits = _head_logits_from_hidden_trainable(exit_heads[exit_id], h_list[layer_idx], device)

            loss_exit = F.cross_entropy(exit_logits, yb)
            loss = lambda_exit * loss_exit
            if use_final_loss:
                loss_final = F.cross_entropy(final_logits, yb)
                loss = loss + lambda_final * loss_final

            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params_layer, grad_clip)
                torch.nn.utils.clip_grad_norm_(params_exit, grad_clip)
            optimizer.step()

            bsz = yb.size(0)
            loss_sum += float(loss.item()) * bsz
            total += bsz
            correct_exit += int((exit_logits.argmax(dim=-1) == yb).sum().item())
            correct_final += int((final_logits.argmax(dim=-1) == yb).sum().item())

        out = eval_cascade_quweit(
            model,
            val_loader,
            device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=thrs,
            use_prob_margin=use_prob_margin,
        )

        print(
            f"[G1-stage] layer={layer_idx + 1} exit={exit_id} Ep{epoch:03d} "
            f"| train_loss={loss_sum / max(total, 1):.4f} "
            f"| train_exit_acc={correct_exit / max(total, 1) * 100:.2f}% "
            f"| train_final_acc={correct_final / max(total, 1) * 100:.2f}% "
            f"| overall@{[round(float(x), 4) for x in thrs]} va={out['overall_acc'] * 100:.2f} "
            f"| exit_rates={[round(x, 4) for x in out['exit_rates']]} final_rate={out['final_rate']:.4f}"
        )

        if out["overall_acc"] > best["val_overall_acc"]:
            best["val_overall_acc"] = float(out["overall_acc"])
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["exit_states"] = [{k: v.detach().cpu().clone() for k, v in head.state_dict().items()} for head in exit_heads]

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
        for head, sd in zip(exit_heads, best["exit_states"]):
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

    parser.add_argument("--stage_exit_ids", type=str, default="", help='0-based exit ids, e.g. "0,1"; empty means all exits')
    parser.add_argument("--epochs_per_stage", type=str, default="30")
    parser.add_argument("--lr_layer", type=str, default="3e-4")
    parser.add_argument("--lr_exit", type=str, default="3e-3")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_exit", type=float, default=0.3)
    parser.add_argument("--lambda_final", type=float, default=1.0)
    parser.add_argument("--use_final_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--thr", type=str, default="", help='comma-separated thresholds per exit; empty means use g0 checkpoint values')
    parser.add_argument("--use_prob_margin", action=argparse.BooleanOptionalAction, default=False)
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

    if args.stage_exit_ids.strip():
        stage_exit_ids = _parse_csv(args.stage_exit_ids, int)
    else:
        stage_exit_ids = list(range(len(exit_heads)))
    if not stage_exit_ids:
        raise ValueError("No stages to run.")
    for exit_id in stage_exit_ids:
        if exit_id < 0 or exit_id >= len(exit_heads):
            raise ValueError(f"Invalid stage exit id {exit_id}; num_exits={len(exit_heads)}")

    epochs_per_stage = _broadcast(_parse_csv(args.epochs_per_stage, int), len(stage_exit_ids), "epochs_per_stage")
    lr_layers = _broadcast(_parse_csv(args.lr_layer, float), len(stage_exit_ids), "lr_layer")
    lr_exits = _broadcast(_parse_csv(args.lr_exit, float), len(stage_exit_ids), "lr_exit")

    print("[info] loader settings "
          f"train_batch={args.batch_size_train} eval_batch={args.batch_size_eval} "
          f"num_workers={args.num_workers} pin_memory={args.pin_memory}")
    print("[info] stage plan")
    for stage_idx, exit_id in enumerate(stage_exit_ids, start=1):
        layer_idx_1based = int(payload_exit_cfg[exit_id]["layer_idx"])
        print(
            f"  stage {stage_idx}: exit_id={exit_id} "
            f"layer={layer_idx_1based} epochs={epochs_per_stage[stage_idx - 1]} "
            f"lr_layer={lr_layers[stage_idx - 1]} lr_exit={lr_exits[stage_idx - 1]} thr={eval_thrs[exit_id]}"
        )

    for stage_idx, exit_id in enumerate(stage_exit_ids):
        layer_idx = int(payload_exit_cfg[exit_id]["layer_idx"]) - 1
        backbone, exit_heads, best = cotrain_g1_stage_quweit(
            backbone,
            train_loader,
            val_loader,
            device,
            num_epochs=epochs_per_stage[stage_idx],
            layer_idx=layer_idx,
            exit_id=exit_id,
            lr_layer=lr_layers[stage_idx],
            lr_exit=lr_exits[stage_idx],
            lambda_exit=args.lambda_exit,
            use_final_loss=args.use_final_loss,
            lambda_final=args.lambda_final,
            thrs=eval_thrs,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            exit_heads=exit_heads,
            payload_exit_cfg=payload_exit_cfg,
            use_prob_margin=args.use_prob_margin,
        )
        print(f"[stage {stage_idx + 1}] best val overall acc = {best['val_overall_acc'] * 100:.2f}%")

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
            "train_mode": "g1_stagewise_cotrain",
            "stage_exit_ids": stage_exit_ids,
            "use_final_loss": bool(args.use_final_loss),
            "lambda_exit": float(args.lambda_exit),
            "lambda_final": float(args.lambda_final),
            "eval_thrs": eval_thrs,
        },
    )

    backbone = backbone.to(device)
    exit_heads = [head.to(device) for head in exit_heads]
    print(f"\n[saved] {args.path_out}")

    val_out = eval_cascade_quweit(
        backbone,
        val_loader,
        device,
        exit_heads=exit_heads,
        exit_cfg_list=payload_exit_cfg,
        thrs=eval_thrs,
        use_prob_margin=args.use_prob_margin,
    )
    test_out = eval_cascade_quweit(
        backbone,
        test_loader,
        device,
        exit_heads=exit_heads,
        exit_cfg_list=payload_exit_cfg,
        thrs=eval_thrs,
        use_prob_margin=args.use_prob_margin,
    )
    print(
        f"[VAL] overall={val_out['overall_acc'] * 100:.2f}% "
        f"exit_rates={[round(x, 4) for x in val_out['exit_rates']]} final_rate={val_out['final_rate']:.4f}"
    )
    print(
        f"[TEST] overall={test_out['overall_acc'] * 100:.2f}% "
        f"exit_rates={[round(x, 4) for x in test_out['exit_rates']]} final_rate={test_out['final_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
