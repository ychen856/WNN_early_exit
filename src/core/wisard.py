# examples/wisard_mnist_minimal.py
# Minimal WiSARD example with 4x4 tiling + thermometer(8)
# Author: you :)  — drop this into your repo and adapt paths as needed.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
import json
from pathlib import Path

# -------------------------------
# Encoding: 4x4 tiles + thermometer(8)
# -------------------------------

def _normalize(x: np.ndarray, input_range: Optional[Tuple[float, float]] = (0, 255)) -> np.ndarray:
    x = x.astype(np.float32)
    if input_range is None:
        a, b = float(x.min()), float(x.max())
    else:
        a, b = input_range
    if b == a:
        return np.zeros_like(x, dtype=np.float32)
    z = (x - a) / (b - a)
    return np.clip(z, 0.0, 1.0)

def split_into_tiles(img: np.ndarray, tiles: Tuple[int, int]) -> List[np.ndarray]:
    H, W = img.shape
    tr, tc = tiles
    assert H % tr == 0 and W % tc == 0, "Image size must be divisible by tile grid."
    th, tw = H // tr, W // tc
    out = []
    for r in range(tr):
        for c in range(tc):
            out.append(img[r*th:(r+1)*th, c*tw:(c+1)*tw])
    return out

def thermometer_encode_array(arr: np.ndarray, levels: int = 8, input_range=(0,255), flatten=True) -> np.ndarray:
    arr01 = _normalize(arr, input_range)
    H, W = arr01.shape
    thresholds = (np.arange(levels, dtype=np.float32)[None, None, :] / levels)
    bits = (arr01[:, :, None] > thresholds).astype(np.uint8)
    return bits.reshape(-1) if flatten else bits  # shape: (H*W*levels,)

def encode_image_thermo_tiled(
    img: np.ndarray, tiles=(4,4), levels=8, input_range=(0,255)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return (bit_vector, meta). bit_vector is uint8 {0,1} of length total_bits."""
    assert img.ndim == 2, "expect grayscale 2D image"
    H, W = img.shape
    tile_list = split_into_tiles(img, tiles)
    tr, tc = tiles
    th, tw = H // tr, W // tc

    encoded_tiles = []
    ranges = []
    cur = 0
    for t in tile_list:
        b = thermometer_encode_array(t, levels=levels, input_range=input_range, flatten=True)
        encoded_tiles.append(b)
        ranges.append((cur, cur + b.size))
        cur += b.size

    vec = np.concatenate(encoded_tiles, axis=0).astype(np.uint8)
    meta = dict(
        tiles=tiles,
        tile_size=(th, tw),
        levels=levels,
        bits_per_tile=th*tw*levels,
        total_bits=int(vec.size),
        tile_index_ranges=ranges,
        input_shape=(H, W),
        input_range=input_range
    )
    return vec, meta

# -------------------------------
# Tuple mapping: build & audit
# -------------------------------

def make_tuple_mapping(
    num_luts: int, addr_bits: int, bit_len: int,
    tiles: Optional[List[Tuple[int,int]]] = None, seed: int = 42
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    mapping = []
    for l in range(num_luts):
        if tiles:
            t0, t1 = tiles[l % len(tiles)]
            pool = np.arange(t0, t1)
        else:
            pool = np.arange(bit_len)
        sel = rng.choice(pool, size=addr_bits, replace=False)
        mapping.append(sel.tolist())
    return mapping

def audit_mapping(mapping: List[List[int]], bit_len: int) -> Dict[str, float]:
    hits = np.zeros(bit_len, dtype=np.int32)
    for sel in mapping:
        hits[sel] += 1
    return dict(
        min_hits=int(hits.min()),
        max_hits=int(hits.max()),
        mean_hits=float(hits.mean()),
        std_hits=float(hits.std()),
        total_bits=int(bit_len),
    )

# -------------------------------
# WiSARD core
# -------------------------------

@dataclass
class WiSARD:
    num_classes: int
    num_luts_per_class: int
    address_bits: int
    tuple_mapping: List[List[int]]        # length = num_luts_per_class, each of len address_bits
    value_dtype: np.dtype = np.uint16     # counting RAM cells
    endianness: str = "little"

    def __post_init__(self):
        C, L, n = self.num_classes, self.num_luts_per_class, self.address_bits
        self.table = np.zeros((C, L, 1 << n), dtype=self.value_dtype)  # [C, L, 2^n]
        # precompute bit weights for address compose
        if self.endianness == "little":
            self._w = (1 << np.arange(n, dtype=np.uint32))
        else:
            self._w = (1 << np.arange(n-1, -1, -1, dtype=np.uint32))

    def _addresses_for_sample(self, bit_vec: np.ndarray) -> np.ndarray:
        """
        Return addresses for all LUTs: shape (L,)
        address = dot(bit_vec[mapping[l]], weights) over {0,1}
        """
        L, n = self.num_luts_per_class, self.address_bits
        addr = np.empty((L,), dtype=np.uint32)
        for l in range(L):
            idx = self.tuple_mapping[l]
            bits = bit_vec[idx]  # length n
            addr[l] = int(bits.astype(np.uint32) @ self._w)
        return addr

    def fit(self, X_bits: np.ndarray, y: np.ndarray, batch: int = 512):
        """
        X_bits: shape (N, B) uint8 {0,1}
        y:      shape (N,) in [0..C-1]
        """
        C, L = self.num_classes, self.num_luts_per_class
        N = X_bits.shape[0]
        for i0 in range(0, N, batch):
            i1 = min(i0+batch, N)
            for i in range(i0, i1):
                c = int(y[i])
                addr = self._addresses_for_sample(X_bits[i])
                # increment all LUT cells for class c
                self.table[c, np.arange(L), addr] += 1

    def score_vector(self, bit_vec: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        L, n = self.num_luts_per_class, self.address_bits
        addr = self._addresses_for_sample(bit_vec)  # (L,)
        # GET COUNTS： shape (C, L)
        votes = self.table[np.arange(self.num_classes)[:, None],
                           np.arange(L)[None, :],
                           addr[None, :]].astype(np.float32)
        # FOR ALL LUT, CALCLUCATE p(c|addr_l) ~ (count_c + α) / (sum_over_classes + C*α)
        alpha = 0.5
        denom = votes.sum(axis=0, keepdims=True) + self.num_classes * alpha
        post = (votes + alpha) / denom

        # USE log PROBABILITIES SUM AS SCORE（OR odds/logit）
        scores = np.log(post + 1e-9).sum(axis=1)  # shape (C,)


        return int(np.argmax(scores))


    def compute_lut_weights(self, model, X_val_bits, y_val, alpha: float = 1.0):
            """
            given validation data
            return:
                w_lut: shape (L,) weight(in float32), represent the Importance of a LUT

            weight:
                take average over samples' margin:
                margin_lut = post[true_class, lut] - max_{c!=true} post[c, lut]
            if margin negative (LUT is misleading), it will reduce the weight of the lut
            """

            C = model.num_classes
            L = model.num_luts_per_class

            # accumulate margin
            lut_margin_sum = np.zeros((L,), dtype=np.float64)
            lut_margin_cnt = np.zeros((L,), dtype=np.int64)

            for i in range(X_val_bits.shape[0]):
                bit_vec = X_val_bits[i]
                true_c = int(y_val[i])

                # addresses for all LUTs for this sample
                addr = model._addresses_for_sample(bit_vec)  # shape (L,)

                # votes[c, l] = self.table[c, l, addr[l]]
                votes = model.table[
                    np.arange(C)[:, None],  # (C,1)
                    np.arange(L)[None, :],  # (1,L)
                    addr[None, :]  # (1,L)
                ].astype(np.float32)  # -> shape (C,L)

                # posterior per LUT: p(class=c | address from this LUT)
                denom = votes.sum(axis=0, keepdims=True) + C * alpha  # shape (1,L)
                post = (votes + alpha) / denom  # shape (C,L)

                # margin for each LUT = post[true_c, l] - best_other(l)
                # best_other(l): max over c != true_c
                # we can do this by temporarily zeroing/ignoring the true class
                # but simpler: take full max and handle the case if true_c is the max.
                # Let's compute two-best trick:
                sorted_post = np.sort(post, axis=0)[::-1, :]  # desc sort along classes: shape (C,L)
                best_vals = sorted_post[0, :]  # shape (L,)
                second_vals = sorted_post[1, :] if C > 1 else np.zeros_like(best_vals)

                # For LUTs where true_c is not the top class, margin will be negative or small.
                # To get the margin w.r.t. true class specifically:
                # margin_true = post[true_c,l] - max_{c != true_c} post[c,l]
                # Let's do this directly.
                post_true = post[true_c, :]  # shape (L,)

                # max over others:
                # we can mask out true_c using -inf trick
                mask = np.ones_like(post, dtype=bool)
                mask[true_c, :] = False
                post_others_max = np.max(np.where(mask, post, -1e9), axis=0)

                margin_true = post_true - post_others_max  # shape (L,)

                # accumulate
                lut_margin_sum += margin_true
                lut_margin_cnt += 1

            # average margin per LUT
            avg_margin = lut_margin_sum / np.maximum(lut_margin_cnt, 1)

            # normalize to something sane (0~1-ish, no negatives)
            # we clip at a tiny floor then rescale
            # idea: if avg_margin < 0, that LUT is actively confusing → weight near 0
            avg_margin = np.maximum(avg_margin, 0.0)

            # avoid all-zero
            if np.all(avg_margin == 0):
                w_lut = np.ones((L,), dtype=np.float32)
            else:
                w_lut = (avg_margin / (avg_margin.mean() + 1e-9)).astype(np.float32)

            return w_lut  # shape (L,)





