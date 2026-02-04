from pathlib import Path
import numpy as np
import torch


def export_fpga_bundle_generic(out_dir, tuple_mapping, addr_bits, lut_data, mode="class"):
    """
    general bundle exporter

    Args:
      out_dir: export directory (Path or str)
      tuple_mapping: list of iterable[int], each LUT uses which bit indices
      addr_bits: address length for each LUT (currently all LUTs are the same)
      lut_data:
        - mode="bit"  : numpy array [num_luts, 2^addr_bits], each entry is 0/1
        - mode="class": numpy array [num_luts, num_classes, 2^addr_bits], each entry is int (voting weight)
      mode:
        - "bit"  : intermediate layer, output bit LUT
        - "class": final layer, output class-wise fused LUT
    """
    out_dir = Path(out_dir)
    lut_dir = out_dir / "luts"
    lut_dir.mkdir(parents=True, exist_ok=True)

    if mode == "bit":
        assert lut_data.ndim == 2, "bit mode expects lut_data shape [num_luts, 2^k]"
        num_luts, M = lut_data.shape
        num_classes = None
    elif mode == "class":
        assert lut_data.ndim == 3, "class mode expects lut_data shape [num_luts, num_classes, 2^k]"
        num_luts, num_classes, M = lut_data.shape
    else:
        raise ValueError(f"Unknown mode={mode}")

    assert (1 << addr_bits) == M, "addr_bits not matching LUT depth"

    # 1) addr_bits_per_lut.mem
    addr_bits_per_lut = [addr_bits] * num_luts
    with open(out_dir / "addr_bits_per_lut.mem", "w") as f:
        for val in addr_bits_per_lut:
            f.write(f"{val:x}\n")

    # 2) kept_bits.mem (each line: bit indices used by a LUT, in hex, space-separated)
    with open(out_dir / "kept_bits.mem", "w") as f:
        for bits in tuple_mapping:
            hex_str = " ".join(f"{int(b):x}" for b in bits)
            f.write(hex_str + "\n")

    # 3) keep_ids.mem: list of LUT IDs
    keep_ids = list(range(num_luts))
    with open(out_dir / "keep_ids.mem", "w") as f:
        for kid in keep_ids:
            f.write(f"{kid:x}\n")

    # 4) luts/lut_xxx.mem
    for l in range(num_luts):
        if mode == "bit":
            # each line: output bit (0/1) for one address
            lut_vals = lut_data[l]  # [2^k]
            with open(lut_dir / f"lut_{l:03d}.mem", "w") as f:
                for v in lut_vals:
                    # write as a hex: 0 or 1
                    f.write(f"{int(v):x}\n")
        else:  # mode == "class"
            # each line: voting weights for all classes at this address (space-separated hex)
            lut_vals = lut_data[l]  # [num_classes, 2^k]
            lut_vals_T = lut_vals.T  # [2^k, num_classes]
            with open(lut_dir / f"lut_{l:03d}.mem", "w") as f:
                for addr_row in lut_vals_T:
                    hex_row = " ".join(f"{int(val):x}" for val in addr_row)
                    f.write(hex_row + "\n")

    print(f"[export_fpga_bundle_generic] mode={mode}, num_luts={num_luts}, out_dir={out_dir}")


# for layer 0
def export_bit_layer_from_model_layer(layer, out_dir, addr_bits=None, threshold=0.5):
    """
    layer: your MultiLayerWNN's layer0 / layer1 etc. (WNNLUTLayer)
    out_dir: output directory for this layer, e.g., out_root/layer0
    addr_bits: expected LUT input bits (if None, use layer.lut_input_size)
    """
    table = layer.table.detach().cpu().numpy()  # [num_luts, 2^K]
    conn_idx = layer.conn_idx.detach().cpu().numpy()  # [num_luts, K]

    num_luts, M = table.shape
    k = layer.lut_input_size
    if addr_bits is None:
        addr_bits = k
    assert (1 << addr_bits) == M, "addr_bits not match layer.table width"
    assert conn_idx.shape == (num_luts, addr_bits)

    # bit fuse: sigmoid + threshold
    bit_table = np.zeros_like(table, dtype=np.uint8)
    for j in range(num_luts):
        for addr in range(M):
            v = 1.0 / (1.0 + np.exp(-table[j, addr]))
            bit_table[j, addr] = 1 if v > threshold else 0

    # mapping directly use conn_idx
    tuple_mapping = [list(conn_idx[j, :]) for j in range(num_luts)]

    export_fpga_bundle_generic(out_dir, tuple_mapping, addr_bits, bit_table, mode="bit")


# for final layer
def export_fused_class_layer(model, layer_idx, out_dir, addr_bits=None, scale=255.0):
    """
    model: MultiLayerWNN (already pruned / finetuned)
    layer_idx: layer index to fuse (usually the final layer)
    out_dir: output directory for this layer, e.g., out_root/layer1
    addr_bits: LUT address bits (if None, use layer.lut_input_size)
    scale: quantization scale factor, convert float to int

    Output format:
      - addr_bits_per_lut.mem
      - kept_bits.mem
      - keep_ids.mem
      - luts/lut_xxx.mem  (each line: num_classes hex values)
    """
    layer = model.layers[layer_idx]
    classifier = model.classifier

    table = layer.table.detach().cpu().numpy()    # [num_luts, 2^K]
    conn_idx = layer.conn_idx.detach().cpu().numpy()  # [num_luts, K]

    num_luts, M = table.shape
    k = layer.lut_input_size
    if addr_bits is None:
        addr_bits = k
    assert (1 << addr_bits) == M
    assert conn_idx.shape == (num_luts, addr_bits)

    W = classifier.weight.detach().cpu().numpy()  # [num_classes, num_luts]
    num_classes, num_luts_cls = W.shape
    assert num_luts_cls == num_luts, "classifier in_features != num_luts of last WNN layer"

    # 1) LUT → sigmoid
    lut_act = 1.0 / (1.0 + np.exp(-table))  # [num_luts, 2^K]

    # 2) fuse: classifier_weight * sigmoid(LUT)
    fused = np.zeros((num_luts, num_classes, M), dtype=np.float32)
    for j in range(num_luts):
        for addr in range(M):
            h = lut_act[j, addr]
            fused[j, :, addr] = W[:, j] * h

    # 3) quantize
    w_min, w_max = fused.min(), fused.max()
    print(f"[export_fused_class_layer] Fused range = [{w_min:.4f}, {w_max:.4f}]")
    # reuse scale or adjust according to range
    if w_max - w_min < 1e-6:
        scale_eff = 1.0
    else:
        scale_eff = scale / (w_max - w_min)
    fused_int = np.round((fused - w_min) * scale_eff).astype(np.int32)

    tuple_mapping = [list(conn_idx[j, :]) for j in range(num_luts)]

    export_fpga_bundle_generic(out_dir, tuple_mapping, addr_bits, fused_int, mode="class")


# bundle exporter full WNN model
def export_multilayer_2layer_for_fpga(model_pruned, out_root, addr_bits_list=None):
    """
    For two-layer WNN:
      - layer0: bit fuse export (sigmoid+threshold -> 0/1 LUT)
      - layer1: class fuse export (Classifier_Weight * Sigmoid(LUT) -> int LUT)

    out_root:
      out_root/layer0/...
      out_root/layer1/...
    """
    if addr_bits_list is None:
        addr_bits_list = [
            model_pruned.layers[0].lut_input_size,
            model_pruned.layers[1].lut_input_size,
        ]

    out_root = Path(out_root)
    assert len(model_pruned.layers) == 2, "Currently only implemented for two layers"

    # Layer 0: bit fuse
    export_bit_layer_from_model_layer(
        layer=model_pruned.layers[0],
        out_dir=out_root / "layer0",
        addr_bits=addr_bits_list[0],
        threshold=0.5,
    )

    # Layer 1: class fused (fuse with classifier)
    export_fused_class_layer(
        model=model_pruned,
        layer_idx=1,
        out_dir=out_root / "layer1",
        addr_bits=addr_bits_list[1],
        scale=255.0,
    )

    print(f"[export_multilayer_2layer_for_fpga] Done. Root dir: {out_root}")


#####################################
# load LUT model 
#####################################
# --- The following are tools for "loading .mem + simulating FPGA pipeline + verification" ---
def _load_layer_bit_bundle(layer_dir: Path):
    """
    Load bit-layer bundle:
      - addr_bits_per_lut.mem
      - kept_bits.mem
      - keep_ids.mem
      - luts/lut_XXX.mem  (each line one hex 0/1)
    Returns:
      mapping:  list[num_luts][addr_bits]  (bit index)
      addr_bits: int
      bit_table: [num_luts, 2^addr_bits] (uint8 0/1)
    """
    layer_dir = Path(layer_dir)
    lut_dir = layer_dir / "luts"

    # addr_bits
    with open(layer_dir / "addr_bits_per_lut.mem") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # the current setting make addr_bits the same in each LUT
    addr_bits = int(lines[0], 16)

    # kept_bits
    mapping = []
    with open(layer_dir / "kept_bits.mem") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            bits = [int(tok, 16) for tok in ln.split()]
            mapping.append(bits)
    num_luts = len(mapping)

    # Read the number of lines in the first LUT to infer 2^addr_bits
    first_lut_path = lut_dir / "lut_000.mem"
    with open(first_lut_path) as f:
        lut_lines = [ln.strip() for ln in f if ln.strip()]
    M = len(lut_lines)
    assert (1 << addr_bits) == M, "Bit-layer LUT depth mismatch addr_bits"

    bit_table = np.zeros((num_luts, M), dtype=np.uint8)

    for j in range(num_luts):
        lut_path = lut_dir / f"lut_{j:03d}.mem"
        with open(lut_path) as f:
            rows = [ln.strip() for ln in f if ln.strip()]
        assert len(rows) == M
        vals = [int(r, 16) for r in rows]
        bit_table[j, :] = np.array(vals, dtype=np.uint8)

    return mapping, addr_bits, bit_table


def _load_layer_class_bundle(layer_dir: Path):
    """
    Load class-fused bundle:
      - addr_bits_per_lut.mem
      - kept_bits.mem
      - keep_ids.mem
      - luts/lut_XXX.mem  (each line: num_classes hex values)
    Returns:
      mapping:   list[num_luts][addr_bits]
      addr_bits: int
      fused:     [num_luts, num_classes, 2^addr_bits] (int32)
    """
    layer_dir = Path(layer_dir)
    lut_dir = layer_dir / "luts"

    # addr_bits
    with open(layer_dir / "addr_bits_per_lut.mem") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    addr_bits = int(lines[0], 16)

    # kept_bits
    mapping = []
    with open(layer_dir / "kept_bits.mem") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            bits = [int(tok, 16) for tok in ln.split()]
            mapping.append(bits)
    num_luts = len(mapping)

    # Read the first line of the first LUT to determine num_classes and depth
    first_lut_path = lut_dir / "lut_000.mem"
    with open(first_lut_path) as f:
        raw_rows = [ln.strip() for ln in f if ln.strip()]
    M = len(raw_rows)
    assert (1 << addr_bits) == M, "Class-layer LUT depth mismatch addr_bits"

    tokens = raw_rows[0].split()
    num_classes = len(tokens)

    fused = np.zeros((num_luts, num_classes, M), dtype=np.int32)

    for j in range(num_luts):
        lut_path = lut_dir / f"lut_{j:03d}.mem"
        with open(lut_path) as f:
            rows = [ln.strip() for ln in f if ln.strip()]
        assert len(rows) == M
        for addr, row in enumerate(rows):
            toks = row.split()
            assert len(toks) == num_classes
            fused[j, :, addr] = np.array([int(t, 16) for t in toks], dtype=np.int32)

    return mapping, addr_bits, fused


def simulate_multilayer_pipeline_from_files(
    x_bits: torch.Tensor,
    layer0_dir: Path,
    layer1_dir: Path,
):
    """
    Simulate the FPGA 2-layer pipeline using the exported .mem files:

      x_bits (0/1) -> [layer0 bit LUT] -> hidden bits -> [layer1 class LUT] -> scores

    Returns:
      scores: numpy array [N, num_classes]
    """
    layer0_dir = Path(layer0_dir)
    layer1_dir = Path(layer1_dir)

    # Load two layers
    mapping0, addr_bits0, bit_table0 = _load_layer_bit_bundle(layer0_dir)
    mapping1, addr_bits1, fused_luts = _load_layer_class_bundle(layer1_dir)

    x_np = x_bits.detach().cpu().numpy()
    x_np = (x_np > 0.5).astype(np.int32)

    N, in_bits = x_np.shape
    num_luts0 = len(mapping0)
    num_luts1 = len(mapping1)
    num_classes = fused_luts.shape[1]

    # ---- Layer 0: bit LUT ----
    powers0 = (2 ** np.arange(addr_bits0 - 1, -1, -1)).astype(np.int32)
    h0 = np.zeros((N, num_luts0), dtype=np.int32)

    for j in range(num_luts0):
        idxs = np.array(mapping0[j], dtype=np.int32)
        bits_j = x_np[:, idxs]             # [N, addr_bits0]
        addrs = (bits_j * powers0).sum(axis=1)  # [N]
        h0[:, j] = bit_table0[j, addrs]

    # ---- Layer 1: class fused LUT ----
    powers1 = (2 ** np.arange(addr_bits1 - 1, -1, -1)).astype(np.int32)
    scores = np.zeros((N, num_classes), dtype=np.int64)

    for j in range(num_luts1):
        idxs = np.array(mapping1[j], dtype=np.int32)
        bits_j = h0[:, idxs]                # [N, addr_bits1]
        addrs = (bits_j * powers1).sum(axis=1)  # [N]

        lut_j = fused_luts[j]               # [C, 2^k1]
        # Extract each sample's voting contribution from this LUT
        contrib = lut_j[:, addrs].T         # [N, C]
        scores += contrib.astype(np.int64)

    return scores


def verify_multilayer_export(
    model,
    export_root,
    x_bits: torch.Tensor,
    y: torch.Tensor,
    sample_limit: int = 1000,
):
    """
    Verify:
      1) The result of PyTorch model_pruned(x_bits)
      2) The result of simulate_multilayer_pipeline from loading .mem files in export_root/layer0 & layer1

    Compare the accuracy and mismatch rate of the two.
    """
    export_root = Path(export_root)
    layer0_dir = export_root / "layer0"
    layer1_dir = export_root / "layer1"

    device = next(model.parameters()).device
    model.eval()

    # ---- PyTorch reference results ----
    with torch.no_grad():
        logits = model(x_bits.to(device))
        preds_ref = logits.argmax(dim=1).cpu().numpy()
    y_np = y.detach().cpu().numpy()

    N = x_bits.size(0)
    if sample_limit is not None and N > sample_limit:
        idx = np.random.choice(N, size=sample_limit, replace=False)
        x_bits_sub = x_bits[idx]
        preds_ref = preds_ref[idx]
        y_np = y_np[idx]
    else:
        x_bits_sub = x_bits

    # ---- FPGA pipeline simulation results ----
    scores_hw = simulate_multilayer_pipeline_from_files(
        x_bits_sub,
        layer0_dir=layer0_dir,
        layer1_dir=layer1_dir,
    )
    preds_hw = scores_hw.argmax(axis=1)

    acc_ref = (preds_ref == y_np).mean()
    acc_hw = (preds_hw == y_np).mean()
    mismatch_rate = (preds_ref != preds_hw).mean()

    print(f"[Verify export] PyTorch accuracy   = {acc_ref*100:.2f}%  on {len(y_np)} samples")
    print(f"[Verify export] FPGA-sim accuracy = {acc_hw*100:.2f}%  on {len(y_np)} samples")
    print(f"[Verify export] mismatch rate     = {mismatch_rate*100:.2f}%  (ref vs FPGA path)")

    return acc_ref, acc_hw, mismatch_rate
