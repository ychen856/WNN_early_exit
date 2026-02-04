from __future__ import annotations

import os, json, csv, numpy as np

def write_coe_for_lut(lut_table: np.ndarray, out_path: str, radix: int = 10):
    C, A = lut_table.shape
    with open(out_path, "w") as f:
        f.write(f"memory_initialization_radix={radix};\n")
        f.write("memory_initialization_vector=\n")
        lines = []
        for addr in range(A):
            vals = ",".join(str(int(lut_table[c, addr])) for c in range(C))
            lines.append(vals)
        f.write(",\n".join(lines))
        f.write(";\n")

def export_profile_bundle(
    profile: dict,
    out_dir: str,
    *,
    keep_ids: list | None = None,
    tuple_mapping_pruned: list[list[int]] | None = None,
    include_coe: bool = False,
    coe_radix: int = 10,
):
    """
    output dir structure:
    out_dir/
      manifest.json
      kept_bits.json
      addr_bits_per_lut.json
      keep_ids.json
      tuple_mapping_pruned.json
      meta.csv
      luts/
        lut_000.npy, ...
      coe/
        lut_000.coe, ...
    """
    os.makedirs(out_dir, exist_ok=True)

    # important parameters
    num_classes = int(profile["num_classes"])
    alpha = float(profile["alpha"])
    bit_order = str(profile.get("bit_order", "lsb"))

    lut_tables = profile["lut_tables"]
    kept_bits = profile["kept_global_bits_per_lut"]
    m_list = profile["addr_bits_per_lut"]
    L = len(lut_tables)

    # --- manifest.json ---
    manifest = {
        "num_classes": num_classes,
        "alpha": alpha,
        "bit_order": bit_order,
        "num_luts": L
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # --- kept_bits.json ---
    with open(os.path.join(out_dir, "kept_bits.json"), "w") as f:
        json.dump(kept_bits, f)

    # --- addr_bits_per_lut.json ---
    with open(os.path.join(out_dir, "addr_bits_per_lut.json"), "w") as f:
        json.dump(m_list, f)

    # --- keep_ids.json (optional) ---
    if keep_ids is not None:
        with open(os.path.join(out_dir, "keep_ids.json"), "w") as f:
            json.dump(list(map(int, keep_ids)), f)

    # --- tuple_mapping_pruned.json (optional) ---
    if tuple_mapping_pruned is not None:
        with open(os.path.join(out_dir, "tuple_mapping_pruned.json"), "w") as f:
            json.dump(tuple_mapping_pruned, f)

    # --- meta.csv (optional if present) ---
    if "meta" in profile and profile["meta"]:
        meta_path = os.path.join(out_dir, "meta.csv")
        keys = ["lut", "k_base", "k_final", "H", "U"]
        with open(meta_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in profile["meta"]:
                w.writerow({k: row.get(k, "") for k in keys})

    # --- LUT tables ---
    lut_dir = os.path.join(out_dir, "luts")
    os.makedirs(lut_dir, exist_ok=True)
    for i, tab in enumerate(lut_tables):
        np.save(os.path.join(lut_dir, f"lut_{i:03d}.npy"), np.asarray(tab, dtype=np.float32))

    # --- COE (optional) ---
    if include_coe:
        coe_dir = os.path.join(out_dir, "coe")
        os.makedirs(coe_dir, exist_ok=True)
        for i, tab in enumerate(lut_tables):
            write_coe_for_lut(np.asarray(tab), os.path.join(coe_dir, f"lut_{i:03d}.coe"), radix=coe_radix)
