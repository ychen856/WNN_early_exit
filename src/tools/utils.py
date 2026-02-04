import os
from typing import List, Tuple
import numpy as np



#from test import eval_with_profile_varm

# ---------------------------
# Utilities
# ---------------------------
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

