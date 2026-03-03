import os
import torch.nn as nn
from typing import List, Tuple, Optional, List
import numpy as np



#from test import eval_with_profile_varm

# ---------------------------
# Utilities
# ---------------------------
import torch


'''def save_checkpoint(path, model, extra: dict = None):
    # exit info
    exit_enabled = (model.exit1_classifier is not None)
    exit_K = None
    exit_bias = None
    if exit_enabled:
        exit_K = model.exit1_classifier.in_features
        exit_bias = (model.exit1_classifier.bias is not None)

    ckpt = {
        "model_state": model.state_dict(),   # includes exit buffers if present
        "config": {
            "in_bits": model.layer_in_bits[0],
            "num_classes": model.classifier.out_features,
            "lut_input_size": model.layers[0].lut_input_size,
            "hidden_luts": tuple(model.layer_out_luts),
            "tau": float(model.tau),

            # exit head metadata
            "exit_enabled": exit_enabled,
            "exit_K": exit_K,
            "exit_bias": exit_bias,
            "exit_tau": float(getattr(model, "exit_tau", 1.0)),
        }
    }
    if extra is not None:
        ckpt["extra"] = extra

    torch.save(ckpt, path)'''

from typing import List, Sequence, Union

def _parse_float_list(s: Union[str, float, None]) -> List[float]:
    """
    Accept:
      - "0.1" -> [0.1]
      - "0.1,0.0,0.0" -> [0.1,0.0,0.0]
      - "0.1x2,0.0x2" -> [0.1,0.1,0.0,0.0]
      - None -> []
    """
    if s is None:
        return []
    if isinstance(s, (float, int)):
        return [float(s)]

    s = str(s).strip()
    if not s:
        return []

    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if "x" in part:
            v_str, n_str = part.split("x")
            v = float(v_str.strip())
            n = int(n_str.strip())
            out.extend([v] * n)
        else:
            out.append(float(part))
    return out


def make_dropout_schedule(dropout_spec: Union[str, float, None], num_layers: int) -> List[float]:
    """
    Returns per-layer dropout probabilities length == num_layers.

    Rules:
      - if one value: broadcast to all layers
      - if shorter list: pad with last value
      - if longer list: truncate
    """
    vals = _parse_float_list(dropout_spec)
    if len(vals) == 0:
        return [0.0] * num_layers

    if len(vals) == 1:
        return vals * num_layers

    if len(vals) < num_layers:
        vals = vals + [vals[-1]] * (num_layers - len(vals))

    return vals[:num_layers]


# -------------------------
# Utils: exit feature prep
# -------------------------
@torch.no_grad()
def _has_buf(t: Optional[torch.Tensor]) -> bool:
    return (t is not None) and isinstance(t, torch.Tensor) and (t.numel() > 0)

def get_exit1_features(model: nn.Module, h1: torch.Tensor) -> torch.Tensor:
    """
    h1: [B, D1] (output of first LUT layer)
    Return: h1_exit: [B, K] or [B, D1] depending on keep_idx
    Applies optional keep_idx selection and optional (mu/sigma) normalization.
    """
    h = h1
    if hasattr(model, "exit1_keep_idx") and _has_buf(model.exit1_keep_idx):
        h = h[:, model.exit1_keep_idx]

    # optional norm if buffers exist
    if hasattr(model, "exit1_mu") and hasattr(model, "exit1_sigma"):
        if _has_buf(model.exit1_mu) and _has_buf(model.exit1_sigma):
            h = (h - model.exit1_mu) / (model.exit1_sigma + 1e-8)

    return h



def _assert_power_of_two(a: int):
    if a <= 0 or (a & (a - 1)) != 0:
        raise ValueError(f"Address dimension A={a} is not a power of two.")

def _addr_from_bits(bit_vec, ordered_global_bits): # LSB first
    v = 0
    for i, b in enumerate(ordered_global_bits): v |= ((1 if bit_vec[b] else 0) << i)
    return v

def _budget_entries_for_addr_ratio(addr_budget_ratio: float, n_full: int, L_kept: int) -> int:
    # goal：∑ 2^{m_l} ≤ L_kept * (addr_budget_ratio * 2^n)  -> round
    return int(round(L_kept * (addr_budget_ratio * (1 << n_full))))


def lut_addr_stats(X_bits: np.ndarray, kept_global_bits_per_lut: List[List[int]]):
    stats = []
    for l, gbits in enumerate(kept_global_bits_per_lut):
        addrs = np.zeros(X_bits.shape[0], dtype=np.int64)
        for i in range(X_bits.shape[0]):
            addrs[i] = _addr_from_bits(X_bits[i], gbits)
        uniq, cnt = np.unique(addrs, return_counts=True)
        p = cnt / cnt.sum()
        H = -(p * np.log2(p + 1e-12)).sum()
        stats.append(dict(lut=l, unique=int(len(uniq)), entropy=float(H)))
    return stats

# ------------------ estimate the LUT address entropy on validation set ------------------
def _lut_addr_entropy_unique(gbits_ordered: List[int],
                             X_bits_val: np.ndarray,
                             max_samples: int = 4000) -> Tuple[float, int]:
    N = min(max_samples, X_bits_val.shape[0])
    addrs = np.zeros(N, dtype=np.int64)
    for i in range(N):
        addrs[i] = _addr_from_bits(X_bits_val[i], gbits_ordered)
    uniq, cnt = np.unique(addrs, return_counts=True)
    p = cnt / cnt.sum()
    H = -(p * np.log2(p + 1e-12)).sum()
    return float(H), int(len(uniq))


def _score_lut_utility_entropy(H: float, U: int) -> float:
    return H + 0.001 * np.log2(max(U, 1))

def make_per_lut_kcap(
    lut_priority: np.ndarray,
    *,
    top_ratio: float = 0.20,   #  20% high contribution
    low_ratio: float = 0.30,   #  30% low contribution
    top_cap: int = 7,          # the most important LUT upper bound
    mid_cap: int = 5,          # the middle important LUT upper bound
    low_cap: int = 4           # the least important LUT upper bound
) -> np.ndarray:
    """return per-LUT k_cap array。"""
    L = len(lut_priority)
    order = np.argsort(-lut_priority)
    caps = np.empty(L, dtype=np.int32)

    n_top = int(round(L * top_ratio))
    n_low = int(round(L * low_ratio))
    top_idx = order[:n_top]
    low_idx = order[-n_low:] if n_low > 0 else np.array([], dtype=int)
    mid_mask = np.ones(L, dtype=bool)
    if n_top > 0: mid_mask[top_idx] = False
    if n_low > 0: mid_mask[low_idx] = False
    mid_idx = np.where(mid_mask)[0]

    caps[top_idx] = top_cap
    caps[mid_idx] = mid_cap
    if n_low > 0:
        caps[low_idx] = low_cap
    return caps


def _resolve_kcap(k_cap, n_addr_bits: int, L: int) -> np.ndarray:
    """
    k_cap:
      - None  →  n_addr_bits
      - int   → clamp ~ [1, n_addr_bits]
      - array → length L per-LUT cap (each clamp ~ [1, n_addr_bits]）
    return per-LUT k_cap: np.ndarray[int] (L,)
    """
    if k_cap is None:
        return np.full(L, n_addr_bits, dtype=np.int32)

    if isinstance(k_cap, (int, np.integer)):
        val = int(k_cap)
        val = max(1, min(val, n_addr_bits))
        return np.full(L, val, dtype=np.int32)

    kcap = np.asarray(k_cap, dtype=np.int32)
    assert kcap.shape[0] == L, f"k_cap length {kcap.shape[0]} != L {L}"
    kcap = np.clip(kcap, 1, n_addr_bits)
    return kcap

def _clean_adaptive_kwargs(adaptive_kwargs: dict) -> dict:
    RESERVED = {
        "model", "tuple_mapping", "bit_priority", "bits_keep_ratio", "X_bits_val"
    }
    return {k: v for k, v in (adaptive_kwargs or {}).items() if k not in RESERVED}

def print_sweep_table(all_metrics):
    print("\nthr    exit%   overall%  exit_acc%  non_exit_acc%  m_mean  m_p95   m_exit_p95  m_non_exit_p95  exited  non_exited")
    print("-"*86)
    for m in all_metrics:
        print(
            f"{m['thr']:<5.2f}  "
            f"{m['exit_rate']*100:>6.2f}  "
            f"{m['overall_acc']*100:>8.2f}  "
            f"{m['exited_acc']*100:>9.2f}  "
            f"{m['non_exited_acc']*100:>13.2f}  "
            f"{m['margin_mean']:>6.2f}  "
            f"{m['margin_p95']:>6.2f}  "
            f"{m['margin_exit_p95']:>11.2f}  "
            f"{m['margin_non_exit_p95']:>15.2f}  "
            f"{m['exited_total']:>7d}  "
            f"{m['non_exited_total']:>10d}"
        )
    print()




import torch

@torch.no_grad()
def debug_xbits_layout(x_bits: torch.Tensor, C: int, H: int, W: int, Z_or_B: int, *, mode: str):
    """
    mode:
      - "thermo_feature_major": idx = ((c*H*W + p)*Z + t)
      - "thermo_threshold_major": idx = (t*(C*H*W) + (c*H*W + p))
      - "bitplane": idx = (((c*H*W + p)*8) + b)   (assume 8 bits)
    """
    assert x_bits.dim() == 2
    B, D = x_bits.shape
    print(f"[xbits] B={B} D={D} ones_rate={x_bits.float().mean().item():.4f}")

    # sample a few pixels across channels
    sample = [
        (0, 0, 0),      # c,y,x
        (1, 0, 0),
        (2, 0, 0),
        (0, 16, 16),
        (1, 16, 16),
        (2, 16, 16),
        (0, 31, 31),
        (1, 31, 31),
        (2, 31, 31),
    ]

    def idx_of(c, y, x, t_or_b):
        p = y * W + x
        if mode == "thermo_feature_major":
            return ((c * H * W + p) * Z_or_B + t_or_b)
        elif mode == "thermo_threshold_major":
            return (t_or_b * (C * H * W) + (c * H * W + p))
        elif mode == "bitplane":
            return ((c * H * W + p) * 8 + t_or_b)
        else:
            raise ValueError(mode)

    # For each sampled pixel: print mean over batch for each level/bit
    for (c, y, x) in sample:
        vals = []
        for k in range(Z_or_B):
            idx = idx_of(c, y, x, k)
            if idx >= D:
                vals.append(float("nan"))
            else:
                vals.append(x_bits[:, idx].float().mean().item())
        vals_str = " ".join([f"{v:.2f}" for v in vals[:min(8, len(vals))]])
        print(f"[pixel c{c} y{y} x{x}] first levels/bits mean: {vals_str} ...")

@torch.no_grad()
def debug_conn_idx(conn_idx: torch.Tensor, in_bits: int, name="conn"):
    assert conn_idx.dtype == torch.long
    mn = int(conn_idx.min().item())
    mx = int(conn_idx.max().item())
    neg = int((conn_idx < 0).sum().item())
    oob = int((conn_idx >= in_bits).sum().item())
    print(f"[{name}] shape={tuple(conn_idx.shape)} min={mn} max={mx} in_bits={in_bits} neg={neg} oob={oob}")

    # coverage: how much of [0..in_bits) is used
    flat = conn_idx.view(-1)
    uniq = flat.unique()
    print(f"[{name}] unique={uniq.numel()} / total={flat.numel()} (dup_rate={(1-uniq.numel()/flat.numel()):.3f})")
    # where does it concentrate
    print(f"[{name}] uniq min/max = {int(uniq.min())}/{int(uniq.max())}")


import torch

def summarize_conn0(conn0: torch.Tensor, total_bits: int = None, max_show: int = 10):
    """
    conn0: [num_luts, k] LongTensor
    """
    assert conn0.dim() == 2
    num_luts, k = conn0.shape

    # oob / neg sanity
    neg = (conn0 < 0).sum().item()
    oob = 0
    if total_bits is not None:
        oob = (conn0 >= total_bits).sum().item()

    # per-LUT unique count (effective-k)
    uniq_counts = torch.tensor([len(set(row.tolist())) for row in conn0], dtype=torch.long)

    # stats
    print(f"[conn0] shape=({num_luts},{k}) neg={neg} oob={oob}")
    print(f"[conn0] effective-k: mean={uniq_counts.float().mean().item():.3f} "
          f"min={uniq_counts.min().item()} p5={int(torch.quantile(uniq_counts.float(), 0.05).item())} "
          f"p50={int(torch.quantile(uniq_counts.float(), 0.50).item())} "
          f"p95={int(torch.quantile(uniq_counts.float(), 0.95).item())} "
          f"max={uniq_counts.max().item()}")

    # how many LUTs have duplicates
    dup_luts = (uniq_counts < k).sum().item()
    print(f"[conn0] LUTs with duplicates: {dup_luts}/{num_luts} ({dup_luts/num_luts:.3%})")

    # show a few examples with worst uniq counts
    worst = torch.argsort(uniq_counts)[:max_show].tolist()
    print("[conn0] worst LUT examples (idx: uniq_count, row):")
    for i in worst:
        row = conn0[i].tolist()
        print(f"  {i}: {uniq_counts[i].item()}  {row}")

# 用法：
# summarize_conn0(conn0, total_bits=40960)