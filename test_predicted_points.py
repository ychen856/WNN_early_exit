import argparse
import copy
from pathlib import Path

import torch

from src.core.linearExitHead import build_exits_from_ckpt
from src.core.multiLayerWNN import build_backbone_from_ckpt
from src.dataio.data import build_loaders_bits
from src.early_exit import eval_cascade_multi_exit


ROOT = Path("/Users/yi-chunchen/workspace/WNN_early_exit")
DEFAULT_DATA_ROOT = ROOT / "datasets"

RHO_100 = ROOT / "model/F_wnn_w_exit_FMNIST_alt_constrain_8000_1.pth"
RHO_75 = ROOT / "model/F_wnn_w_exit_FMNIST_alt_constrain_6000_1.pth"
RHO_50 = ROOT / "model/F_wnn_w_exit_FMNIST_alt_constrain_4000_1.pth"
RHO_25 = ROOT / "model/F_wnn_w_exit_FMNIST_alt_constrain_2000_1.pth"

RHO_MODEL_PATHS = {
    100: RHO_100,
    75: RHO_75,
    50: RHO_50,
    25: RHO_25,
}


def parse_float_list(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_like_rhos(text: str):
    vals = parse_float_list(text)
    out = []
    for val in vals:
        ival = int(round(val))
        if abs(val - ival) > 1e-6:
            raise ValueError(f"rho must be integer-like, got {val}")
        out.append(ival)
    return out


def build_loaders(dataset, data_root, batch_size_eval, val_ratio, seed, z, device):
    train_loader, val_loader, test_loader, _, _, _ = build_loaders_bits(
        dataset=dataset,
        root=str(data_root),
        batch_size_train=256,
        batch_size_eval=batch_size_eval,
        val_ratio=val_ratio,
        seed=seed,
        z=z,
        device_for_encoding=device,
        shuffle_train=True,
    )
    return train_loader, val_loader, test_loader


def load_backbone_and_exits(ckpt_path: Path, device):
    backbone, bb_cfg, extra = build_backbone_from_ckpt(str(ckpt_path), device)
    num_classes = int(bb_cfg["num_classes"])
    exit_heads, exit_cfg_list = build_exits_from_ckpt(str(ckpt_path), device, num_classes=num_classes)
    payload_exit_cfg = [cfg.to_payload() for cfg in exit_cfg_list]
    return backbone, bb_cfg, extra, exit_heads, payload_exit_cfg


def resolve_ckpt_for_rho(rho: int) -> Path:
    if rho not in RHO_MODEL_PATHS:
        raise KeyError(f"Unsupported rho={rho}. Available rhos: {sorted(RHO_MODEL_PATHS)}")
    path = RHO_MODEL_PATHS[rho]
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint for rho={rho} not found: {path}")
    return path


def load_ckpt_bundle_for_rho(rho: int, device):
    ckpt_path = resolve_ckpt_for_rho(rho)
    model, bb_cfg, extra, exit_heads, payload_exit_cfg = load_backbone_and_exits(ckpt_path, device)
    return {
        "rho": rho,
        "path": ckpt_path,
        "model": model,
        "bb_cfg": bb_cfg,
        "extra": extra,
        "exit_heads": exit_heads,
        "exit_cfg": payload_exit_cfg,
    }


def validate_bundle_compatibility(bundles):
    ref = bundles[0]
    ref_num_layers = len(ref["model"].layers)
    ref_exit_layers = [int(cfg["layer_idx"]) for cfg in ref["exit_cfg"]]
    ref_hidden_luts = tuple(ref["bb_cfg"].get("hidden_luts", ()))

    for bundle in bundles[1:]:
        if len(bundle["model"].layers) != ref_num_layers:
            raise ValueError("All rho checkpoints must have the same number of backbone layers.")
        if [int(cfg["layer_idx"]) for cfg in bundle["exit_cfg"]] != ref_exit_layers:
            raise ValueError("All rho checkpoints must have the same exit layer layout.")
        if tuple(bundle["bb_cfg"].get("hidden_luts", ())) != ref_hidden_luts:
            raise ValueError("All rho checkpoints must have the same hidden_luts config.")


def assemble_hybrid_model(rhos, device):
    bundles = [load_ckpt_bundle_for_rho(rho, device) for rho in rhos]
    validate_bundle_compatibility(bundles)

    assembled_model = copy.deepcopy(bundles[0]["model"]).to(device)
    assembled_exit_heads = [None] * len(bundles[0]["exit_heads"])
    assembled_exit_cfg = [None] * len(bundles[0]["exit_cfg"])

    exit_layer_ids = [int(cfg["layer_idx"]) for cfg in bundles[0]["exit_cfg"]]
    prev_layer = -1

    for exit_id, (rho, exit_layer) in enumerate(zip(rhos, exit_layer_ids)):
        bundle = bundles[exit_id]
        for layer_id in range(prev_layer + 1, exit_layer + 1):
            assembled_model.layers[layer_id] = copy.deepcopy(bundle["model"].layers[layer_id]).to(device)
            assembled_model.layer_in_bits[layer_id] = bundle["model"].layer_in_bits[layer_id]
            assembled_model.layer_out_luts[layer_id] = bundle["model"].layer_out_luts[layer_id]

        assembled_exit_heads[exit_id] = copy.deepcopy(bundle["exit_heads"][exit_id]).to(device)
        assembled_exit_cfg[exit_id] = copy.deepcopy(bundle["exit_cfg"][exit_id])
        prev_layer = exit_layer

    final_bundle = bundles[-1]
    for layer_id in range(prev_layer + 1, len(assembled_model.layers)):
        assembled_model.layers[layer_id] = copy.deepcopy(final_bundle["model"].layers[layer_id]).to(device)
        assembled_model.layer_in_bits[layer_id] = final_bundle["model"].layer_in_bits[layer_id]
        assembled_model.layer_out_luts[layer_id] = final_bundle["model"].layer_out_luts[layer_id]

    assembled_model.classifier = copy.deepcopy(final_bundle["model"].classifier).to(device)
    if hasattr(final_bundle["model"], "keep_idx"):
        assembled_model._buffers["keep_idx"] = final_bundle["model"].keep_idx.detach().clone().to(device)

    return assembled_model, assembled_exit_heads, assembled_exit_cfg, bundles


def evaluate(model, loader, device, exit_heads, payload_exit_cfg, thrs):
    return eval_cascade_multi_exit(
        model,
        loader,
        device,
        exit_heads=exit_heads,
        exit_cfg_list=payload_exit_cfg,
        thrs=thrs,
        use_prob_margin=False,
        log_margins=False,
    )


def print_metrics(split_name, out, thrs):
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


def print_assembly_summary(rhos, bundles, exit_cfg):
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
    parser.add_argument("--dataset", type=str, default="FMNIST")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--rhos", type=str, required=True, help="Comma-separated rho for each exit head, e.g. 100,50")
    parser.add_argument("--thrs", type=str, required=True, help="Comma-separated threshold for each exit head")
    parser.add_argument("--batch-size-eval", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, test_loader = build_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size_eval=args.batch_size_eval,
        val_ratio=args.val_ratio,
        seed=args.seed,
        z=args.z,
        device=device,
    )

    rhos = parse_int_like_rhos(args.rhos)
    thrs = parse_float_list(args.thrs)

    model, exit_heads, payload_exit_cfg, bundles = assemble_hybrid_model(rhos, device)

    if len(thrs) != len(exit_heads):
        raise ValueError(f"Expected {len(exit_heads)} threshold values, got {len(thrs)}")
    if len(rhos) != len(exit_heads):
        raise ValueError(f"Expected {len(exit_heads)} rho values, got {len(rhos)}")

    print(f"dataset={args.dataset}")
    print_assembly_summary(rhos, bundles, payload_exit_cfg)

    val_out = evaluate(model, val_loader, device, exit_heads, payload_exit_cfg, thrs)
    test_out = evaluate(model, test_loader, device, exit_heads, payload_exit_cfg, thrs)

    print_metrics("val", val_out, thrs)
    print_metrics("test", test_out, thrs)


if __name__ == "__main__":
    main()
