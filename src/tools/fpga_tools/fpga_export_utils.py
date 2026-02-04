# fpga_export_utils.py

import numpy as np
import torch
from pathlib import Path


def export_wnn_for_fpga(
    model,
    export_path,
    dt_thresholds=None,
    dt_xmin=None,
    dt_xmax=None,
    quant_bits=None,
    device="cpu",
):
    """
    Export MultiLayerWNN to a single .npz file for FPGA use.
    Additionally stores DT encoder parameters if provided.

    Args:
        model:         trained (and pruned) MultiLayerWNN (PyTorch module)
        export_path:   path to .npz
        dt_thresholds: torch.Tensor or np.ndarray, shape [z] or [num_bins]
        dt_xmin:       torch.Tensor or np.ndarray, scalar or [1]
        dt_xmax:       torch.Tensor or np.ndarray, scalar or [1]
        quant_bits:    if not None, quantize LUT tables to int16 (or similar)
        device:        "cpu" / "cuda"
    """
    export_path = Path(export_path)
    model = model.to(device)
    model.eval()

    # given model.num_layers, model.layers, model.classifier
    num_layers = len(model.layers)
    input_bits = model.layers[0].in_bits
    num_classes = model.classifier.out_features

    arrays = {}
    arrays["num_layers"] = np.int32(num_layers)
    arrays["input_bits"] = np.int32(input_bits)
    arrays["num_classes"] = np.int32(num_classes)

    # classifier weight (without bias)
    W_cls = model.classifier.weight.detach().cpu().numpy().astype(np.float32)
    arrays["classifier_weight"] = W_cls

    # optional: classifier bias
    if model.classifier.bias is not None:
        arrays["classifier_bias"] = (
            model.classifier.bias.detach().cpu().numpy().astype(np.float32)
        )

    # per-layer export
    for l, layer in enumerate(model.layers):
        prefix = f"layer{l}_"

        in_bits_l = layer.in_bits
        num_luts_l = layer.num_luts
        k_l = layer.lut_input_size

        conn_idx = layer.conn_idx.detach().cpu().numpy().astype(np.int32)
        table = layer.table.detach().cpu().numpy().astype(np.float32)

        arrays[prefix + "in_bits"] = np.int32(in_bits_l)
        arrays[prefix + "num_luts"] = np.int32(num_luts_l)
        arrays[prefix + "lut_input_size"] = np.int32(k_l)
        arrays[prefix + "conn_idx"] = conn_idx

        if quant_bits is not None:
            # simple version: linear quantize to int16
            max_abs = np.max(np.abs(table)) + 1e-8
            qmax = (1 << (quant_bits - 1)) - 1  # e.g., 32767
            scale = max_abs / qmax
            table_q = np.round(table / scale).astype(np.int16)
            arrays[prefix + "table_q"] = table_q
            arrays[prefix + "table_scale"] = np.float32(scale)
        else:
            arrays[prefix + "table"] = table

    # --- encoder parameters (DT) ---
    if dt_thresholds is not None:
        arrays["dt_thresholds"] = (
            dt_thresholds.detach().cpu().numpy().astype(np.float32)
            if isinstance(dt_thresholds, torch.Tensor)
            else np.asarray(dt_thresholds, dtype=np.float32)
        )
    if dt_xmin is not None:
        arrays["dt_xmin"] = (
            dt_xmin.detach().cpu().numpy().astype(np.float32)
            if isinstance(dt_xmin, torch.Tensor)
            else np.asarray(dt_xmin, dtype=np.float32)
        )
    if dt_xmax is not None:
        arrays["dt_xmax"] = (
            dt_xmax.detach().cpu().numpy().astype(np.float32)
            if isinstance(dt_xmax, torch.Tensor)
            else np.asarray(dt_xmax, dtype=np.float32)
        )

    np.savez(export_path, **arrays)
    print(f"[export_wnn_for_fpga] Saved to {export_path}")


def _load_layer_table_from_npz(data, layer_idx, prefer_quantized=True):
    prefix = f"layer{layer_idx}_"
    num_luts = int(data[prefix + "num_luts"])
    k = int(data[prefix + "lut_input_size"])

    if prefer_quantized and (prefix + "table_q") in data.files:
        table_q = data[prefix + "table_q"].astype(np.int16)
        scale = float(data[prefix + "table_scale"])
        return table_q, scale, num_luts, k, True
    else:
        table = data[prefix + "table"].astype(np.float32)
        # Here you can quantize if needed; for now, assume float or external processing
        return table, 1.0, num_luts, k, False


def export_lut_init_files(
    npz_path,
    out_dir,
    fmt="mem",
    radix=16,
    fixed_frac_bits=None,
    prefer_quantized=True,
):
    """
    Generate LUT init files from wnn_pruned_fpga.npz.

    Args:
        npz_path:         path to .npz
        out_dir:          output directory
        fmt:              "mem" or "coe"
        radix:            10 or 16 (for coe/mem)
        fixed_frac_bits:  if using float table and want to convert to Qm.n, provide n; if None, cast directly to int
        prefer_quantized: prefer using table_q/table_scale
    Generated file naming: layer{l}_lut{j}.{fmt}
      One file per LUT, containing 2^k entries.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    num_layers = int(data["num_layers"])

    print(f"[export_lut_init_files] num_layers = {num_layers}")

    for l in range(num_layers):
        table, scale, num_luts, k, is_quant = _load_layer_table_from_npz(
            data, l, prefer_quantized=prefer_quantized
        )

        print(
            f"  Layer {l}: num_luts={num_luts}, k={k}, "
            f"{'quantized' if is_quant else 'float'}"
        )

        # table shape: [num_luts, 2^k]
        entries_per_lut = table.shape[1]

        for j in range(num_luts):
            lut_vals = table[j, :]  # [2^k]

            # if float and you want to convert to fixed-point
            if not is_quant and fixed_frac_bits is not None:
                factor = 1 << fixed_frac_bits
                lut_vals_int = np.round(lut_vals * factor).astype(np.int32)
            else:
                # quantized or direct cast
                lut_vals_int = lut_vals.astype(np.int32)

            if fmt == "mem":
                _write_mem_file(out_dir, l, j, lut_vals_int, radix)
            elif fmt == "coe":
                _write_coe_file(out_dir, l, j, lut_vals_int, radix)
            else:
                raise ValueError(f"Unknown fmt={fmt}")

    print(f"[export_lut_init_files] Done. Files in {out_dir}")


def _write_mem_file(out_dir, layer_idx, lut_idx, vals_int, radix=16):
    """
    Generate .mem files for Verilog $readmemh or $readmemb.
    """
    suffix = "hex" if radix == 16 else "dec"
    fname = out_dir / f"layer{layer_idx}_lut{lut_idx}.{suffix}.mem"
    with open(fname, "w") as f:
        if radix == 16:
            for v in vals_int:
                if v < 0:
                    # 32-bit two's complement example, you can adjust bit width as needed
                    v_twos = (v + (1 << 32)) & 0xFFFFFFFF
                    f.write(f"{v_twos:08X}\n")
                else:
                    f.write(f"{v:08X}\n")
        else:
            for v in vals_int:
                f.write(f"{int(v)}\n")
    # print(f"    wrote {fname}")


def _write_coe_file(out_dir, layer_idx, lut_idx, vals_int, radix=16):
    """
    Generate Xilinx .coe BRAM init file.
    """
    fname = out_dir / f"layer{layer_idx}_lut{lut_idx}.coe"
    with open(fname, "w") as f:
        f.write(f"memory_initialization_radix={radix};\n")
        f.write("memory_initialization_vector=\n")
        if radix == 16:
            lines = []
            for v in vals_int:
                if v < 0:
                    v_twos = (v + (1 << 32)) & 0xFFFFFFFF
                    lines.append(f"{v_twos:08X}")
                else:
                    lines.append(f"{v:08X}")
        else:
            lines = [str(int(v)) for v in vals_int]
        f.write(",\n".join(lines))
        f.write(";\n")
    # print(f"    wrote {fname}")