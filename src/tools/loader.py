from src.core.infer import *
import os, json, glob, numpy as np

def load_profile_bundle(in_dir: str) -> dict:
    """
    from bundle, recover profile：read ralpha and num_classes from manifest.json
    """
    # --- manifest.json ---
    with open(os.path.join(in_dir, "manifest.json"), "r") as f:
        manifest = json.load(f)
    num_classes = int(manifest["num_classes"])
    alpha = float(manifest["alpha"])
    bit_order = str(manifest.get("bit_order", "lsb"))

    # --- kept_bits.json ---
    with open(os.path.join(in_dir, "kept_bits.json"), "r") as f:
        kept_bits = json.load(f)

    # --- addr_bits_per_lut.json ---
    with open(os.path.join(in_dir, "addr_bits_per_lut.json"), "r") as f:
        m_list = json.load(f)

    # --- LUT tables ---
    lut_paths = sorted(glob.glob(os.path.join(in_dir, "luts", "lut_*.npy")))
    lut_tables = [np.load(p) for p in lut_paths]
    assert len(lut_tables) == len(kept_bits) == len(m_list), "LUT/kept_bits/m_list 長度不一致"

    prof = dict(
        lut_tables=lut_tables,
        kept_global_bits_per_lut=kept_bits,
        addr_bits_per_lut=m_list,
        num_classes=num_classes,
        alpha=alpha,
        bit_order=bit_order,
    )

    # optional: read keep_ids / tuple_mapping_pruned if needed
    keep_ids_path = os.path.join(in_dir, "keep_ids.json")
    if os.path.exists(keep_ids_path):
        with open(keep_ids_path, "r") as f:
            prof["keep_ids"] = json.load(f)

    tuple_map_path = os.path.join(in_dir, "tuple_mapping_pruned.json")
    if os.path.exists(tuple_map_path):
        with open(tuple_map_path, "r") as f:
            prof["tuple_mapping_pruned"] = json.load(f)

    return prof


