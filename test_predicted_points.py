import argparse
import copy
from dataclasses import fields
from pathlib import Path
from typing import List

import torch

from src.core.linearExitHead import build_exits_from_ckpt
from src.train_quweit_lut_backbone_v2 import QuWeiTViT, TrainConfig
from src.train_quweit_lut_early_exit_g0_ce import (
    build_clean_cifar_loaders,
    collect_cascade_cache,
    eval_cascade_cached,
    get_external_exit_profile,
)


ROOT = Path("/ychen-storage-fast/WNN_early_exit")
DEFAULT_DATA_ROOT = ROOT / "datasets"

RHO_100 = ROOT / "model/weightless_all_v2_final_u_alt_constrain_192_1.pth"
RHO_75 = ROOT / "model/weightless_all_v2_final_u_alt_constrain_144_1.pth"
RHO_50 = ROOT / "model/weightless_all_v2_final_u_alt_constrain_96_1.pth"
RHO_25 = ROOT / "model/weightless_all_v2_final_u_alt_constrain_48_1.pth"

RHO_MODEL_PATHS = {
    100: RHO_100,
    75: RHO_75,
    50: RHO_50,
    25: RHO_25,
}


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_like_rhos(text: str) -> List[int]:
    vals = parse_float_list(text)
    out = []
    for val in vals:
        ival = int(round(val))
        if abs(val - ival) > 1e-6:
            raise ValueError(f"rho must be integer-like, got {val}")
        out.append(ival)
    return out


def _cfg_from_payload(cfg_payload: dict) -> TrainConfig:
    allowed = {f.name for f in fields(TrainConfig)}
    cfg_dict = {k: v for k, v in cfg_payload.items() if k in allowed}
    cfg = TrainConfig(**cfg_dict)
    cfg.use_exit = False
    return cfg


def load_quweit_cascade_bundle(exit_ckpt: Path, device):
    ckpt = torch.load(exit_ckpt, map_location=device)
    if "model_state_dict" not in ckpt or "backbone_cfg" not in ckpt:
        raise ValueError(f"Checkpoint missing backbone bundle fields: {exit_ckpt}")

    backbone_cfg_payload = ckpt["backbone_cfg"]
    cfg_payload = backbone_cfg_payload["config"] if isinstance(backbone_cfg_payload, dict) and "config" in backbone_cfg_payload else backbone_cfg_payload
    cfg = _cfg_from_payload(cfg_payload)

    model = QuWeiTViT(cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("[load_quweit_cascade_bundle] missing:", missing)
    print("[load_quweit_cascade_bundle] unexpected:", unexpected)

    exit_heads, exit_cfg_list = build_exits_from_ckpt(str(exit_ckpt), device, num_classes=cfg.num_classes)
    payload_exit_cfg = [cfg_item.to_payload() for cfg_item in exit_cfg_list]
    return model.eval(), cfg, exit_heads, payload_exit_cfg


def resolve_ckpt_for_rho(rho: int) -> Path:
    if rho not in RHO_MODEL_PATHS:
        raise KeyError(f"Unsupported rho={rho}. Available rhos: {sorted(RHO_MODEL_PATHS)}")
    path = RHO_MODEL_PATHS[rho]
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint for rho={rho} not found: {path}")
    return path


def load_bundle_for_rho(rho: int, device):
    ckpt_path = resolve_ckpt_for_rho(rho)
    model, cfg, exit_heads, payload_exit_cfg = load_quweit_cascade_bundle(ckpt_path, device)
    return {
        "rho": rho,
        "path": ckpt_path,
        "model": model,
        "cfg": cfg,
        "exit_heads": exit_heads,
        "exit_cfg": payload_exit_cfg,
    }


def validate_bundle_compatibility(bundles):
    ref = bundles[0]
    ref_depth = int(ref["cfg"].depth)
    ref_exit_layers = [int(cfg["layer_idx"]) for cfg in ref["exit_cfg"]]
    ref_embed_dim = int(ref["cfg"].embed_dim)
    ref_num_classes = int(ref["cfg"].num_classes)

    for bundle in bundles[1:]:
        if int(bundle["cfg"].depth) != ref_depth:
            raise ValueError("All rho checkpoints must have the same transformer depth.")
        if [int(cfg["layer_idx"]) for cfg in bundle["exit_cfg"]] != ref_exit_layers:
            raise ValueError("All rho checkpoints must have the same exit layer layout.")
        if int(bundle["cfg"].embed_dim) != ref_embed_dim:
            raise ValueError("All rho checkpoints must have the same embed_dim.")
        if int(bundle["cfg"].num_classes) != ref_num_classes:
            raise ValueError("All rho checkpoints must have the same num_classes.")


def assemble_hybrid_model(rhos: List[int], device):
    bundles = [load_bundle_for_rho(rho, device) for rho in rhos]
    validate_bundle_compatibility(bundles)

    assembled_model = copy.deepcopy(bundles[0]["model"]).to(device)
    assembled_exit_heads = [None] * len(bundles[0]["exit_heads"])
    assembled_exit_cfg = [None] * len(bundles[0]["exit_cfg"])

    exit_layer_ids = [int(cfg["layer_idx"]) for cfg in bundles[0]["exit_cfg"]]
    prev_layer = 0

    for exit_id, (rho, exit_layer) in enumerate(zip(rhos, exit_layer_ids)):
        bundle = bundles[exit_id]
        for block_idx in range(prev_layer, exit_layer):
            assembled_model.blocks[block_idx] = copy.deepcopy(bundle["model"].blocks[block_idx]).to(device)

        assembled_exit_heads[exit_id] = copy.deepcopy(bundle["exit_heads"][exit_id]).to(device)
        assembled_exit_cfg[exit_id] = copy.deepcopy(bundle["exit_cfg"][exit_id])
        prev_layer = exit_layer

    final_bundle = bundles[-1]
    for block_idx in range(prev_layer, len(assembled_model.blocks)):
        assembled_model.blocks[block_idx] = copy.deepcopy(final_bundle["model"].blocks[block_idx]).to(device)

    assembled_model.norm = copy.deepcopy(final_bundle["model"].norm).to(device)
    assembled_model.head = copy.deepcopy(final_bundle["model"].head).to(device)
    return assembled_model.eval(), assembled_exit_heads, assembled_exit_cfg, bundles


def print_metrics(split_name: str, out: dict, thrs: List[float]):
    print(
        f"[{split_name}] "
        f"thrs={[round(float(x), 4) for x in thrs]} "
        f"overall={out['overall_acc'] * 100:.2f}% "
        f"exit_rates={[round(float(x), 4) for x in out['exit_rates']]} "
        f"exit_accs={[round(float(x), 4) if x == x else float('nan') for x in out['exit_accs']]} "
        f"final_rate={out['final_rate']:.4f} "
        f"final_acc={out['final_acc'] * 100:.2f}% "
        f"expLayers={out['avg_layers_executed_per_sample']:.4f} "
        f"avgFLOPs={out['avg_flops_per_sample']:.0f} "
        f"avgMACs={out['avg_macs_per_sample']:.0f}"
    )


def print_assembly_summary(rhos: List[int], bundles: List[dict], exit_cfg: List[dict]):
    print(f"rhos={rhos}")
    for exit_id, cfg in enumerate(exit_cfg):
        bundle = bundles[exit_id]
        print(
            f"exit{exit_id}: rho={bundle['rho']} path={bundle['path']} "
            f"layer={int(cfg['layer_idx'])} k={int(cfg['k'])}"
        )
    print(f"final_branch: rho={bundles[-1]['rho']} path={bundles[-1]['path']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--rhos", type=str, required=True, help="Comma-separated rho for each exit head, e.g. 100,75,50,25")
    parser.add_argument("--thrs", type=str, required=True, help="Comma-separated threshold for each exit head")
    parser.add_argument("--batch-size-probe", type=int, default=128)
    parser.add_argument("--batch-size-eval", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rhos = parse_int_like_rhos(args.rhos)
    thrs = parse_float_list(args.thrs)
    model, exit_heads, payload_exit_cfg, bundles = assemble_hybrid_model(rhos, device)

    if len(rhos) != len(exit_heads):
        raise ValueError(f"Expected {len(exit_heads)} rho values, got {len(rhos)}")
    if len(thrs) != len(exit_heads):
        raise ValueError(f"Expected {len(exit_heads)} threshold values, got {len(thrs)}")

    cfg = copy.deepcopy(bundles[0]["cfg"])
    cfg.dataset = args.dataset
    cfg.data_dir = str(args.data_root)
    _, val_loader, test_loader, num_classes = build_clean_cifar_loaders(
        cfg,
        batch_size_probe=args.batch_size_probe,
        batch_size_eval=args.batch_size_eval,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    if int(num_classes) != int(cfg.num_classes):
        raise ValueError(f"Dataset num_classes={num_classes} != checkpoint num_classes={cfg.num_classes}")

    profile = get_external_exit_profile(model, exit_heads, payload_exit_cfg)
    val_cache = collect_cascade_cache(model, val_loader, device, exit_heads=exit_heads, exit_cfg_list=payload_exit_cfg)
    test_cache = collect_cascade_cache(model, test_loader, device, exit_heads=exit_heads, exit_cfg_list=payload_exit_cfg)

    print(f"dataset={args.dataset}")
    print_assembly_summary(rhos, bundles, payload_exit_cfg)

    val_out = eval_cascade_cached(val_cache, profile, thrs)
    test_out = eval_cascade_cached(test_cache, profile, thrs)
    print_metrics("val", val_out, thrs)
    print_metrics("test", test_out, thrs)


if __name__ == "__main__":
    main()
