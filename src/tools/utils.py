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


