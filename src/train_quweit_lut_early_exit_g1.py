import argparse
import itertools
import os
from dataclasses import fields
from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from src.core.linearExitHead import build_exit_heads_from_cfg
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


def _trainable_summary(named_params):
    return [(name, p.numel()) for name, p in named_params if p.requires_grad]


def _count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters()) if module is not None else 0


def _linear_flops_and_macs(linear: torch.nn.Linear):
    macs = float(linear.in_features * linear.out_features)
    bias_flops = float(linear.out_features if linear.bias is not None else 0.0)
    flops = 2.0 * macs + bias_flops
    return flops, macs


def _cfg_from_payload(cfg_payload: dict) -> TrainConfig:
    allowed = {f.name for f in fields(TrainConfig)}
    cfg_dict = {k: v for k, v in cfg_payload.items() if k in allowed}
    cfg = TrainConfig(**cfg_dict)
    cfg.use_exit = False
    return cfg


def load_quweit_model_and_exits(path: str, device, use_ema_backbone: bool = True):
    ckpt = torch.load(path, map_location=device)

    if "config" in ckpt and "model_state" in ckpt:
        cfg = _cfg_from_payload(ckpt["config"])
        model = QuWeiTViT(cfg).to(device)
        state = ckpt.get("model_ema_state") if use_ema_backbone and ckpt.get("model_ema_state") is not None else ckpt["model_state"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("[load_quweit_model_and_exits] backbone missing:", missing)
        print("[load_quweit_model_and_exits] backbone unexpected:", unexpected)
        return model.eval(), cfg, ckpt, [], []

    if "model_state_dict" in ckpt and "backbone_cfg" in ckpt:
        backbone_cfg = ckpt["backbone_cfg"]
        cfg_payload = backbone_cfg["config"] if isinstance(backbone_cfg, dict) and "config" in backbone_cfg else backbone_cfg
        cfg = _cfg_from_payload(cfg_payload)
        model = QuWeiTViT(cfg).to(device)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("[load_quweit_model_and_exits] backbone missing:", missing)
        print("[load_quweit_model_and_exits] backbone unexpected:", unexpected)

        payload_exit_cfg = ckpt.get("exit_cfg", []) or []
        exit_cfg_list = [ExitConfig.from_payload(item) if isinstance(item, dict) else item for item in payload_exit_cfg]
        exit_heads = []
        if exit_cfg_list:
            exit_heads = build_exit_heads_from_cfg(exit_cfg_list, num_classes=cfg.num_classes, device=device)
            exits_state_dict = ckpt.get("exits_state_dict", [])
            if len(exits_state_dict) != len(exit_heads):
                raise ValueError("Checkpoint exit heads/state length mismatch.")
            for head, sd in zip(exit_heads, exits_state_dict):
                head.load_state_dict(sd, strict=True)
            exit_heads = [head.cpu() for head in exit_heads]
        return model.eval(), cfg, ckpt, exit_heads, exit_cfg_list

    raise ValueError("Unsupported checkpoint format.")


def build_clean_cifar_loaders(
    cfg: TrainConfig,
    *,
    batch_size_train: int,
    batch_size_eval: int,
    val_ratio: float,
    seed: int,
    num_workers: int,
):
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
        raise ValueError(f"Unsupported dataset in checkpoint config: {cfg.dataset}")

    val_size = int(len(train_set) * val_ratio)
    train_size = len(train_set) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_set, [train_size, val_size], generator=gen)

    def create_loader(dataset, *, batch_size: int, shuffle: bool):
        try:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=cfg.pin_memory,
                drop_last=False,
            )
        except RuntimeError as e:
            if num_workers > 0 and ("shared memory" in str(e).lower() or "shm" in str(e).lower()):
                print(f"[WARNING] SHM error with num_workers={num_workers}. Retrying with num_workers=0...")
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=0,
                    pin_memory=cfg.pin_memory,
                    drop_last=False,
                )
            raise

    train_loader = create_loader(train_subset, batch_size=batch_size_train, shuffle=True)
    val_loader = create_loader(val_subset, batch_size=batch_size_eval, shuffle=False)
    test_loader = create_loader(test_set, batch_size=batch_size_eval, shuffle=False)
    return train_loader, val_loader, test_loader, num_classes


@torch.no_grad()
def forward_with_all_hidden(model: QuWeiTViT, x: torch.Tensor):
    out = model(x, return_intermediate=True)
    final_logits = out["logits"]
    h_list = [h[:, 0, :].detach() for h in out["intermediates"]]
    return final_logits, h_list


def forward_with_all_hidden_trainable(model: QuWeiTViT, x: torch.Tensor):
    out = model(x, return_intermediate=True)
    final_logits = out["logits"]
    h_list = [h[:, 0, :] for h in out["intermediates"]]
    return final_logits, h_list


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def configure_stage_trainable_params(model: QuWeiTViT, exit_heads: List[torch.nn.Module], block_indices: List[int], exit_id: int):
    model.to(next(model.parameters()).device)
    set_requires_grad(model, False)
    for idx in block_indices:
        set_requires_grad(model.blocks[idx], True)

    for head in exit_heads:
        set_requires_grad(head, False)
    exit_heads[exit_id] = exit_heads[exit_id].to(next(model.parameters()).device)
    set_requires_grad(exit_heads[exit_id], True)

    if hasattr(model, "head"):
        set_requires_grad(model.head, False)
    if hasattr(model, "norm"):
        set_requires_grad(model.norm, False)
    if hasattr(model, "patch_embed"):
        set_requires_grad(model.patch_embed, False)

    trainable = _trainable_summary(model.named_parameters())
    trainable += [(f"exit_heads.{exit_id}.{name}", p.numel()) for name, p in exit_heads[exit_id].named_parameters() if p.requires_grad]
    print("[stage] trainable params:", trainable)


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

    model.eval()
    exit_heads = [head.to(device).eval() for head in exit_heads]

    all_taken = [[] for _ in range(num_exits)]
    all_undecided = [[] for _ in range(num_exits)]

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        final_logits, h_list = forward_with_all_hidden(model, xb)
        bsz = yb.size(0)
        undecided = torch.ones(bsz, dtype=torch.bool, device=device)
        preds = torch.empty(bsz, dtype=torch.long, device=device)
        route_taken = torch.full((bsz,), fill_value=num_exits, dtype=torch.long, device=device)

        for i in range(num_exits):
            if not undecided.any():
                break
            cfg = exit_cfg_list[i]
            layer_idx = int(cfg["layer_idx"])
            logits_i = exit_heads[i](h_list[layer_idx - 1])
            top2 = torch.topk(logits_i, k=2, dim=-1).values
            margins = top2[:, 0] - top2[:, 1]
            all_undecided[i].append(margins[undecided].detach().cpu())

            take_i = undecided & (margins > float(thrs[i]))
            if take_i.any():
                preds[take_i] = logits_i.argmax(dim=-1)[take_i]
                n_exit[i] += int(take_i.sum().item())
                c_exit[i] += (preds[take_i] == yb[take_i]).sum().item()
                all_taken[i].append(margins[take_i].detach().cpu())
                route_taken[take_i] = i
                undecided = undecided & (~take_i)

        if undecided.any():
            preds[undecided] = final_logits.argmax(dim=-1)[undecided]
            n_final += int(undecided.sum().item())
            c_final += (preds[undecided] == yb[undecided]).sum().item()

        correct += (preds == yb).sum().item()
        total += bsz

        for route in range(num_exits):
            count = float((route_taken == route).sum().item())
            if count == 0:
                continue
            layer_idx = int(profile["exit_heads"][route]["layer_idx"])
            flops = profile["patch_embed"]["flops"] + sum(layer["flops"] for layer in profile["layers"][:layer_idx])
            macs = profile["patch_embed"]["macs"] + sum(layer["macs"] for layer in profile["layers"][:layer_idx])
            flops += sum(head["flops"] for head in profile["exit_heads"][: route + 1])
            macs += sum(head["macs"] for head in profile["exit_heads"][: route + 1])
            total_flops += count * flops
            total_macs += count * macs

        final_count = float((route_taken == num_exits).sum().item())
        if final_count > 0:
            final_route_flops = profile["backbone_full_flops"] + sum(head["flops"] for head in profile["exit_heads"])
            final_route_macs = profile["backbone_full_macs"] + sum(head["macs"] for head in profile["exit_heads"])
            total_flops += final_count * final_route_flops
            total_macs += final_count * final_route_macs

    for i in range(num_exits):
        mu_u = float(torch.cat(all_undecided[i]).mean().item()) if all_undecided[i] else float("nan")
        p95_u = float(torch.quantile(torch.cat(all_undecided[i]), 0.95).item()) if all_undecided[i] else float("nan")
        mu_t = float(torch.cat(all_taken[i]).mean().item()) if all_taken[i] else float("nan")
        p95_t = float(torch.quantile(torch.cat(all_taken[i]), 0.95).item()) if all_taken[i] else float("nan")
        margin_stats.append(
            {
                "undecided_mean": mu_u,
                "undecided_p95": p95_u,
                "taken_mean": mu_t,
                "taken_p95": p95_t,
            }
        )

    avg_flops_per_sample = total_flops / max(total, 1)
    avg_macs_per_sample = total_macs / max(total, 1)
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
        "backbone_params": float(profile["backbone_params"]),
        "total_exit_head_params": float(profile["total_exit_head_params"]),
        "param_overhead_ratio": float(profile["param_overhead_ratio"]),
        "compute_overhead_ratio": compute_overhead_ratio,
        "compute_saving_ratio": 1.0 - compute_overhead_ratio if compute_overhead_ratio == compute_overhead_ratio else float("nan"),
    }


@torch.no_grad()
def collect_exit_margins(model, loader, device, *, exit_heads: List[torch.nn.Module], exit_cfg_list: List[dict]):
    num_exits = len(exit_heads)
    margins_per_exit = [[] for _ in range(num_exits)]
    model.eval()
    exit_heads = [head.to(device).eval() for head in exit_heads]

    for xb, _ in loader:
        xb = xb.to(device)
        _, h_list = forward_with_all_hidden(model, xb)
        for exit_id in range(num_exits):
            layer_idx = int(exit_cfg_list[exit_id]["layer_idx"])
            logits = exit_heads[exit_id](h_list[layer_idx - 1])
            top2 = torch.topk(logits, k=2, dim=-1).values
            margins = top2[:, 0] - top2[:, 1]
            margins_per_exit[exit_id].append(margins.detach().cpu())

    out = []
    for parts in margins_per_exit:
        if not parts:
            out.append(torch.empty(0))
        else:
            out.append(torch.cat(parts, dim=0))
    return out


def _unique_quantile_values(values: torch.Tensor, quantiles: List[float]) -> List[float]:
    if values.numel() == 0:
        return [0.0]
    out = []
    for q in quantiles:
        thr = float(torch.quantile(values, q).item())
        out.append(thr)
    uniq = sorted(set(out))
    return uniq if uniq else [0.0]


def sweep_cascade_by_quantile(model, val_loader, test_loader, device, *, exit_heads: List[torch.nn.Module], exit_cfg_list: List[dict], quantile_groups: List[List[float]]):
    margin_groups = collect_exit_margins(model, val_loader, device, exit_heads=exit_heads, exit_cfg_list=exit_cfg_list)
    thr_groups = [
        _unique_quantile_values(margins, quantiles)
        for margins, quantiles in zip(margin_groups, quantile_groups)
    ]
    print("[quantile-sweep] threshold groups:", thr_groups)

    rows_val = []
    rows_test = []
    for thrs in itertools.product(*thr_groups):
        thrs = list(thrs)
        rows_val.append({"thrs": thrs, **eval_cascade(model, val_loader, device, exit_heads=exit_heads, exit_cfg_list=exit_cfg_list, thrs=thrs)})
        rows_test.append({"thrs": thrs, **eval_cascade(model, test_loader, device, exit_heads=exit_heads, exit_cfg_list=exit_cfg_list, thrs=thrs)})

    rows_val.sort(key=lambda row: (-row["overall_acc"], row["avg_flops_per_sample"]))
    rows_test.sort(key=lambda row: (-row["overall_acc"], row["avg_flops_per_sample"]))
    return rows_val, rows_test, thr_groups


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
    header_parts += ["avgFLOPs", "avgMACs", "overhead"]
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
        values.append(f"{row['avg_flops_per_sample']:>11.0f}")
        values.append(f"{row['avg_macs_per_sample']:>11.0f}")
        values.append(f"{row['compute_overhead_ratio']:>11.4f}")
        print("  ".join(values))


def parse_stage_block_groups(s: str) -> List[List[int]]:
    groups = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        groups.append([int(x.strip()) for x in chunk.split(",") if x.strip()])
    return groups


def branchwise_train_stage(
    model,
    train_loader,
    val_loader,
    device,
    *,
    stage_id: int,
    block_indices: List[int],
    exit_id: int,
    exit_heads: List[torch.nn.Module],
    payload_exit_cfg: List[dict],
    epochs: int,
    lr_blocks: float,
    lr_exit: float,
    weight_decay: float,
    lambda_exit: float,
    lambda_final: float,
    use_final_loss: bool,
    grad_clip: float,
):
    if not block_indices:
        raise ValueError(f"Stage {stage_id} has empty block group.")
    if exit_id < 0 or exit_id >= len(exit_heads):
        raise ValueError(f"Invalid exit_id={exit_id} for stage {stage_id}.")

    model = model.to(device)
    configure_stage_trainable_params(model, exit_heads, block_indices, exit_id)

    block_params = []
    for idx in block_indices:
        block_params.extend([p for p in model.blocks[idx].parameters() if p.requires_grad])
    exit_heads[exit_id] = exit_heads[exit_id].to(device)
    exit_params = [p for p in exit_heads[exit_id].parameters() if p.requires_grad]
    if not block_params:
        raise ValueError(f"Stage {stage_id} has no trainable block params.")
    if not exit_params:
        raise ValueError(f"Stage {stage_id} has no trainable exit params.")

    optimizer = torch.optim.AdamW(
        [
            {"params": block_params, "lr": lr_blocks, "weight_decay": weight_decay},
            {"params": exit_params, "lr": lr_exit, "weight_decay": weight_decay},
        ]
    )

    eval_thrs = [float(cfg["thr"]) for cfg in payload_exit_cfg]
    best = {"val_overall_acc": -1.0, "model_state": None, "exit_states": None}
    target_layer_idx = int(payload_exit_cfg[exit_id]["layer_idx"])

    for epoch in range(epochs):
        model.train()
        exit_heads[exit_id].train()
        loss_sum = 0.0
        total = 0
        correct_exit = 0
        correct_final = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            final_logits, h_list = forward_with_all_hidden_trainable(model, xb)
            exit_logits = exit_heads[exit_id](h_list[target_layer_idx - 1])

            loss_exit = F.cross_entropy(exit_logits, yb)
            loss = lambda_exit * loss_exit
            if use_final_loss:
                loss_final = F.cross_entropy(final_logits, yb)
                loss = loss + lambda_final * loss_final

            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(block_params, grad_clip)
                torch.nn.utils.clip_grad_norm_(exit_params, grad_clip)
            optimizer.step()

            bsz = yb.size(0)
            loss_sum += float(loss.item()) * bsz
            total += bsz
            correct_exit += (exit_logits.argmax(dim=-1) == yb).sum().item()
            correct_final += (final_logits.argmax(dim=-1) == yb).sum().item()

        val_out = eval_cascade(model, val_loader, device, exit_heads=exit_heads, exit_cfg_list=payload_exit_cfg, thrs=eval_thrs)
        print(
            f"[stage {stage_id}] epoch={epoch:03d} "
            f"blocks={block_indices} exit={exit_id} "
            f"train_loss={loss_sum / max(total, 1):.4f} "
            f"train_exit_acc={correct_exit / max(total, 1) * 100:.2f}% "
            f"train_final_acc={correct_final / max(total, 1) * 100:.2f}% "
            f"val_overall={val_out['overall_acc'] * 100:.2f}% "
            f"exit_rates={[round(x, 4) for x in val_out['exit_rates']]} "
            f"final_rate={val_out['final_rate']:.4f}"
        )

        if val_out["overall_acc"] > best["val_overall_acc"]:
            best["val_overall_acc"] = float(val_out["overall_acc"])
            best["model_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["exit_states"] = [{k: v.detach().cpu().clone() for k, v in head.state_dict().items()} for head in exit_heads]

    if best["model_state"] is not None:
        model.load_state_dict(best["model_state"], strict=False)
        for head, sd in zip(exit_heads, best["exit_states"]):
            head.load_state_dict(sd, strict=True)

    return model, exit_heads, best


def main():
    parser = argparse.ArgumentParser(description="Branch-wise early-exit fine-tuning starting from a g0 checkpoint.")
    parser.add_argument("--backbone_ckpt", type=str, required=True, help="Checkpoint produced by train_quwei_lut_early_exit_g0.py or train_quweit_lut_backbone_v2.py")
    parser.add_argument("--path_out", type=str, required=True)
    parser.add_argument("--use_ema_backbone", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--batch_size_train", type=int, default=128)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--stage_block_groups", type=str, default="0,1,2;3,4;5,6;7,8")
    parser.add_argument("--epochs_per_stage", type=str, default="20")
    parser.add_argument("--lr_blocks", type=str, default="3e-4")
    parser.add_argument("--lr_exit", type=str, default="3e-3")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_exit", type=float, default=0.3)
    parser.add_argument("--lambda_final", type=float, default=1.0)
    parser.add_argument("--use_final_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--single_thr_list", type=str, default="0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0")
    parser.add_argument("--cascade_quantiles", type=str, default="0.0,0.25,0.5,0.75,0.9,0.95")
    parser.add_argument("--sweep_top_k", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model, backbone_cfg, raw_ckpt, exit_heads, exit_cfg_list = load_quweit_model_and_exits(
        args.backbone_ckpt,
        device,
        use_ema_backbone=args.use_ema_backbone,
    )
    if not exit_heads or not exit_cfg_list:
        raise ValueError("g1 branch-wise training expects a checkpoint that already contains exit heads (e.g. g0 output).")

    print("[info] building clean CIFAR loaders matched to checkpoint preprocessing")
    train_loader, val_loader, test_loader, num_classes = build_clean_cifar_loaders(
        backbone_cfg,
        batch_size_train=args.batch_size_train,
        batch_size_eval=args.batch_size_eval,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    if num_classes != backbone_cfg.num_classes:
        raise ValueError(f"Dataset num_classes mismatch: loaders={num_classes}, cfg={backbone_cfg.num_classes}")

    payload_exit_cfg = [cfg.to_payload() for cfg in exit_cfg_list]
    stage_block_groups = parse_stage_block_groups(args.stage_block_groups)
    if len(stage_block_groups) != len(exit_heads):
        raise ValueError(f"--stage_block_groups expects {len(exit_heads)} groups, got {len(stage_block_groups)}")

    for group in stage_block_groups:
        for idx in group:
            if idx < 0 or idx >= backbone_cfg.depth:
                raise ValueError(f"Block index {idx} is out of range for depth={backbone_cfg.depth}")

    epochs_per_stage = _broadcast(_parse_csv(args.epochs_per_stage, int), len(stage_block_groups), "epochs_per_stage")
    lr_blocks = _broadcast(_parse_csv(args.lr_blocks, float), len(stage_block_groups), "lr_blocks")
    lr_exit = _broadcast(_parse_csv(args.lr_exit, float), len(stage_block_groups), "lr_exit")

    print("[info] branch-wise stage plan:")
    for stage_id, (block_group, cfg) in enumerate(zip(stage_block_groups, payload_exit_cfg), start=1):
        print(
            f"  stage {stage_id}: blocks={block_group} "
            f"exit={stage_id - 1} exit_layer={int(cfg['layer_idx'])}"
        )

    for stage_idx, block_group in enumerate(stage_block_groups):
        model, exit_heads, best = branchwise_train_stage(
            model,
            train_loader,
            val_loader,
            device,
            stage_id=stage_idx + 1,
            block_indices=block_group,
            exit_id=stage_idx,
            exit_heads=exit_heads,
            payload_exit_cfg=payload_exit_cfg,
            epochs=epochs_per_stage[stage_idx],
            lr_blocks=lr_blocks[stage_idx],
            lr_exit=lr_exit[stage_idx],
            weight_decay=args.weight_decay,
            lambda_exit=args.lambda_exit,
            lambda_final=args.lambda_final,
            use_final_loss=args.use_final_loss,
            grad_clip=args.grad_clip,
        )
        print(f"[stage {stage_idx + 1}] best val overall acc = {best['val_overall_acc'] * 100:.2f}%")

    _ensure_dir(args.path_out)
    save_ckpt_v2(
        args.path_out,
        model.cpu(),
        [head.cpu() for head in exit_heads],
        {
            "source": "train_quweit_lut_early_exit_g1.py",
            "backbone_ckpt": args.backbone_ckpt,
            "config": raw_ckpt["backbone_cfg"]["config"] if "backbone_cfg" in raw_ckpt and isinstance(raw_ckpt["backbone_cfg"], dict) and "config" in raw_ckpt["backbone_cfg"] else raw_ckpt.get("config", backbone_cfg.__dict__),
        },
        exit_cfg_list=payload_exit_cfg,
        extra={
            "dataset": backbone_cfg.dataset,
            "train_mode": "branchwise_g1_finetune",
            "stage_block_groups": stage_block_groups,
            "use_final_loss": bool(args.use_final_loss),
            "lambda_exit": float(args.lambda_exit),
            "lambda_final": float(args.lambda_final),
        },
    )
    model = model.to(device)
    exit_heads = [head.to(device) for head in exit_heads]
    print(f"\n[saved] {args.path_out}")

    single_thr_list = _parse_csv(args.single_thr_list, float)
    for exit_id, cfg in enumerate(payload_exit_cfg):
        layer_idx = int(cfg["layer_idx"])
        rows = []
        for thr in single_thr_list:
            out = eval_cascade(
                model,
                val_loader,
                device,
                exit_heads=exit_heads,
                exit_cfg_list=payload_exit_cfg,
                thrs=[thr if i == exit_id else float(payload_exit_cfg[i]["thr"]) for i in range(len(payload_exit_cfg))],
            )
            rows.append((thr, out["overall_acc"], out["exit_rates"][exit_id], out["exit_accs"][exit_id]))
        print(f"\n[VAL single-exit scan] exit={exit_id} layer={layer_idx}")
        for thr, overall_acc, exit_rate, exit_acc in rows:
            exit_acc_text = f"{exit_acc * 100:.2f}%" if exit_acc == exit_acc else "nan"
            print(f"  thr={thr:.2f} overall={overall_acc * 100:.2f}% exit_rate={exit_rate * 100:.2f}% exit_acc={exit_acc_text}")

    quantile_groups = _parse_threshold_groups(args.cascade_quantiles, len(exit_heads), "cascade_quantiles")
    rows_val, rows_test, thr_groups = sweep_cascade_by_quantile(
        model,
        val_loader,
        test_loader,
        device,
        exit_heads=exit_heads,
        exit_cfg_list=payload_exit_cfg,
        quantile_groups=quantile_groups,
    )
    print(f"[quantile-sweep] num_combinations={int(torch.tensor([len(g) for g in thr_groups]).prod().item())}")
    print_cascade_quantile_sweep("VAL cascade quantile sweep", rows_val, top_k=args.sweep_top_k)
    print_cascade_quantile_sweep("TEST cascade quantile sweep", rows_test, top_k=args.sweep_top_k)


if __name__ == "__main__":
    main()
