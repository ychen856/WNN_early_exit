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




#=========================
# Example usage with CIFAR10
#=========================

import torch

def build_cifar10_layer0_mapping(
    num_luts: int,
    k: int,
    z: int,
    H: int = 32,
    W: int = 32,
    C: int = 3,
    patch: int = 5,        # local window size (odd)
    seed: int = 42,
    device: str = "cpu",
    per_conn_random_t: bool = True,
):
    """
    Return:
      conn_idx: [num_luts, k] long tensor, each entry in [0, D*z)
    """
    assert patch % 2 == 1
    g = torch.Generator(device="cpu").manual_seed(seed)

    D = H * W * C
    half = patch // 2

    # precompute pixel index table: (c,y,x) -> p
    # p = c*(H*W) + y*W + x
    # We'll sample anchors and neighbors in (c,y,x).
    conn = torch.empty((num_luts, k), dtype=torch.long)

    # sample anchors
    anchor_c = torch.randint(0, C, (num_luts,), generator=g)
    anchor_y = torch.randint(0, H, (num_luts,), generator=g)
    anchor_x = torch.randint(0, W, (num_luts,), generator=g)

    for i in range(num_luts):
        coords = []
        # 1) anchor always included
        coords.append((int(anchor_c[i]), int(anchor_y[i]), int(anchor_x[i])))

        # 2) sample k-1 neighbors in local patch (allow same channel or force same channel)
        for _ in range(k - 1):
            dy = int(torch.randint(-half, half + 1, (1,), generator=g))
            dx = int(torch.randint(-half, half + 1, (1,), generator=g))
            cy = int(anchor_y[i]) + dy
            cx = int(anchor_x[i]) + dx
            # clamp to image boundary
            cy = max(0, min(H - 1, cy))
            cx = max(0, min(W - 1, cx))
            cc = int(anchor_c[i])  # channel-aware (same channel); you can randomize if you want
            coords.append((cc, cy, cx))

        # coords -> pixel indices
        pix = []
        for (cc, cy, cx) in coords:
            p = cc * (H * W) + cy * W + cx  # [0..D-1]
            pix.append(p)
        pix = torch.tensor(pix, dtype=torch.long)

        # pixel -> bit
        if per_conn_random_t:
            t = torch.randint(0, z, (k,), generator=g)
        else:
            t = torch.zeros((k,), dtype=torch.long)
        bit = pix * z + t  # [0..D*z-1]

        conn[i] = bit

    conn = conn.to(device)
    return conn





import torch
from typing import Tuple, Optional

def build_patch_local_conn_idx_chw(
    num_luts: int,
    lut_input_size: int,
    H: int = 32,
    W: int = 32,
    C: int = 3,
    patch: Tuple[int, int] = (4, 4),
    global_frac: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Return conn_idx: [num_luts, lut_input_size] long
    Indexing assumes CHW flatten: idx = c*H*W + y*W + x
    global_frac: fraction of bits sampled globally (rest from local patch)
    """
    assert 0.0 <= global_frac < 1.0
    ph, pw = patch
    assert ph > 0 and pw > 0

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    in_bits = C * H * W
    conn = torch.empty((num_luts, lut_input_size), dtype=torch.long)

    # how many local vs global per LUT
    k_global = int(round(lut_input_size * global_frac))
    k_local = lut_input_size - k_global

    # pre-sample anchors uniformly over spatial + channel
    # choose anchor pixel index in [0, H*W) and channel in [0, C)
    anchors_xy = torch.randint(0, H * W, (num_luts,), generator=g)
    anchors_c = torch.randint(0, C, (num_luts,), generator=g)

    for i in range(num_luts):
        y0 = int(anchors_xy[i].item() // W)
        x0 = int(anchors_xy[i].item() % W)
        c0 = int(anchors_c[i].item())

        # local patch bounds
        y1 = max(0, y0 - ph // 2)
        y2 = min(H, y1 + ph)
        x1 = max(0, x0 - pw // 2)
        x2 = min(W, x1 + pw)

        # if hit boundary, shift back to keep size
        if (y2 - y1) < ph:
            y1 = max(0, H - ph)
            y2 = H
        if (x2 - x1) < pw:
            x1 = max(0, W - pw)
            x2 = W

        # candidate local indices (same channel by default)
        ys = torch.arange(y1, y2)
        xs = torch.arange(x1, x2)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        local_idx = (c0 * H * W + yy * W + xx).reshape(-1)  # [ph*pw]

        # sample local bits
        if k_local > 0:
            sel = torch.randint(0, local_idx.numel(), (k_local,), generator=g)
            picked_local = local_idx[sel]
        else:
            picked_local = torch.empty((0,), dtype=torch.long)

        # sample global bits
        if k_global > 0:
            picked_global = torch.randint(0, in_bits, (k_global,), generator=g, dtype=torch.long)
        else:
            picked_global = torch.empty((0,), dtype=torch.long)

        picked = torch.cat([picked_local, picked_global], dim=0)
        conn[i] = picked

    return conn.to(device)


import torch

@torch.no_grad()
def build_patch_local_conn_idx_bitmajor(
    *,
    num_luts: int,
    lut_input_size: int,
    H: int,
    W: int,
    C: int,
    bits_per_channel: int = 8,     # CIFAR uint8 -> 8
    patch=(4, 4),
    global_frac: float = 0.1,
    seed: int = 42,
    device="cpu",
):
    """
    x_bits layout (bit-major):
      idx = b*(C*H*W) + c*(H*W) + (y*W + x)
    returns conn_idx: [num_luts, lut_input_size] in [0, C*H*W*bits_per_channel)
    """
    ph, pw = patch
    assert ph <= H and pw <= W
    in_bits = C * H * W * bits_per_channel
    HW = H * W
    BCHW = C * HW

    g = torch.Generator(device="cpu").manual_seed(seed)

    # sample patch anchors for each lut
    y0 = torch.randint(0, H - ph + 1, (num_luts,), generator=g)
    x0 = torch.randint(0, W - pw + 1, (num_luts,), generator=g)

    conn = torch.empty((num_luts, lut_input_size), dtype=torch.long)

    for i in range(num_luts):
        chosen = set()
        for j in range(lut_input_size):
            use_global = (torch.rand((), generator=g).item() < global_frac)

            if use_global:
                # fully random over in_bits
                idx = int(torch.randint(0, in_bits, (1,), generator=g).item())
            else:
                # patch-local: choose (b,c,y,x) but y,x constrained in patch
                b = int(torch.randint(0, bits_per_channel, (1,), generator=g).item())
                c = int(torch.randint(0, C, (1,), generator=g).item())
                yy = int(y0[i].item()) + int(torch.randint(0, ph, (1,), generator=g).item())
                xx = int(x0[i].item()) + int(torch.randint(0, pw, (1,), generator=g).item())
                p = yy * W + xx
                idx = b * BCHW + c * HW + p

            # ensure unique within this LUT
            tries = 0
            while idx in chosen:
                tries += 1
                if tries > 50:
                    # fallback: global random
                    idx = int(torch.randint(0, in_bits, (1,), generator=g).item())
                    break
                idx = int(torch.randint(0, in_bits, (1,), generator=g).item())
            chosen.add(idx)
            conn[i, j] = idx

    return conn.to(device)


import torch
from typing import Tuple, Optional

def build_conn0_rgb_sobel(
    *,
    num_luts: int,
    k: int,
    rgb_in_bits: int,
    sobel_in_bits: int,
    sobel_frac: float = 0.25,     # fraction of k that must come from sobel
    sobel_mode: str = "global",   # "global" or "patch"
    sobel_hw: Tuple[int, int] = (32, 32),  # sobel image size (gray)
    patch_hw: Tuple[int, int] = (8, 8),
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """
    x_bits = concat([rgb_bitplane_bits, sobel_bits])  -> length = rgb_in_bits + sobel_in_bits

    Return conn_idx: [num_luts, k] long, each LUT:
      - picks k_sobel bits from sobel block
      - picks k_rgb   bits from rgb block
    """
    assert rgb_in_bits > 0 and sobel_in_bits > 0
    assert 0.0 <= sobel_frac <= 1.0
    assert sobel_mode in ("global", "patch")

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    k_sobel = int(round(k * sobel_frac))
    k_sobel = max(1, k_sobel) if sobel_frac > 0 else 0
    k_sobel = min(k, k_sobel)
    k_rgb = k - k_sobel

    # helpers
    def rand_unique(low: int, high: int, n: int) -> torch.Tensor:
        # sample with replacement then unique-enough; for k<=10 it's fine
        if n == 0:
            return torch.empty((0,), dtype=torch.long)
        return torch.randint(low=low, high=high, size=(n,), generator=g, dtype=torch.long)

    # sobel candidate sampling
    if sobel_mode == "global":
        def sample_sobel_bits(n: int) -> torch.Tensor:
            # sobel block indices in [rgb_in_bits, rgb_in_bits+sobel_in_bits)
            return rand_unique(rgb_in_bits, rgb_in_bits + sobel_in_bits, n)
    else:
        H, W = sobel_hw
        ph, pw = patch_hw
        assert H * W == sobel_in_bits, f"sobel_in_bits should be H*W, got {sobel_in_bits} vs {H*W}"
        assert ph <= H and pw <= W

        def sample_sobel_bits(n: int) -> torch.Tensor:
            if n == 0:
                return torch.empty((0,), dtype=torch.long)
            # choose a random patch location
            y0 = int(torch.randint(0, H - ph + 1, (1,), generator=g).item())
            x0 = int(torch.randint(0, W - pw + 1, (1,), generator=g).item())
            # flatten indices inside patch
            ys = torch.arange(y0, y0 + ph, dtype=torch.long)
            xs = torch.arange(x0, x0 + pw, dtype=torch.long)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            patch_ids = (grid_y * W + grid_x).reshape(-1)  # [ph*pw]
            # choose n positions from patch
            pick = torch.randint(0, patch_ids.numel(), (n,), generator=g, dtype=torch.long)
            sobel_local = patch_ids[pick]  # [n] in [0, H*W)
            return sobel_local + rgb_in_bits

    # rgb sampling (global)
    def sample_rgb_bits(n: int) -> torch.Tensor:
        return rand_unique(0, rgb_in_bits, n)

    conn = torch.empty((num_luts, k), dtype=torch.long)
    for i in range(num_luts):
        a = sample_sobel_bits(k_sobel)
        b = sample_rgb_bits(k_rgb)
        idx = torch.cat([a, b], dim=0)
        # shuffle within LUT
        if k > 1:
            perm = torch.randperm(k, generator=g)
            idx = idx[perm]
        conn[i] = idx

    return conn.to(device)

import torch

def build_conn0_rgb_thermo_sobel(
    num_luts: int,
    k: int,
    H: int = 32,
    W: int = 32,
    C: int = 3,
    rgb_bits_per_channel: int = 8,
    thermo_levels: int = 8,
    patch=(4, 4),
    frac_thermo: float = 0.33,   # k=9 -> 3 bits
    frac_sobel: float = 0.22,    # k=9 -> 2 bits
    seed: int = 42,
    device="cpu",
):
    """
    concat layout (your current):
      [0 : rgb_in_bits)                  -> rgb bitplane
      [rgb_in_bits : rgb_in_bits+8192)   -> gray thermo (pixel-major, 8 levels)
      [rgb_in_bits+8192 : end)           -> gray sobel (1 bit per pixel)

    Returns:
      conn_idx: LongTensor [num_luts, k], values in [0, 33792)
    """
    ph, pw = patch
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    rgb_in_bits = C * H * W * rgb_bits_per_channel
    thermo_in_bits = H * W * thermo_levels         # 8192
    sobel_in_bits = H * W                          # 1024

    offset_rgb = 0
    offset_thermo = rgb_in_bits
    offset_sobel = rgb_in_bits + thermo_in_bits
    total_bits = rgb_in_bits + thermo_in_bits + sobel_in_bits

    # decide how many bits from each block per LUT
    k_sobel = int(round(k * frac_sobel))
    k_thermo = int(round(k * frac_thermo))
    k_rgb = k - k_thermo - k_sobel
    if k_rgb < 1:
        # keep at least 1 rgb bit
        k_rgb = 1
        # take from thermo first, then sobel
        while k_thermo + k_sobel > k - 1:
            if k_thermo > 0:
                k_thermo -= 1
            elif k_sobel > 0:
                k_sobel -= 1
            else:
                break

    def rgb_bit_index(c, y, x, b):
        # pixel-major within channel, then bit
        # idx = (((c*H + y)*W + x)*BPC + b)
        return offset_rgb + (((c * H + y) * W + x) * rgb_bits_per_channel + b)

    def thermo_bit_index(y, x, level):
        # pixel-major then level
        pix = y * W + x
        return offset_thermo + pix * thermo_levels + level

    def sobel_bit_index(y, x):
        pix = y * W + x
        return offset_sobel + pix

    conn = torch.empty((num_luts, k), dtype=torch.long)

    for i in range(num_luts):
        # anchor for rgb and gray (you can share anchor too; here share y,x)
        y0 = int(torch.randint(0, H, (1,), generator=g).item())
        x0 = int(torch.randint(0, W, (1,), generator=g).item())

        # sample coords within patch around (y0,x0)
        ys = torch.randint(max(0, y0 - ph // 2), min(H, y0 + (ph + 1) // 2), (k_rgb + k_thermo + k_sobel,), generator=g)
        xs = torch.randint(max(0, x0 - pw // 2), min(W, x0 + (pw + 1) // 2), (k_rgb + k_thermo + k_sobel,), generator=g)

        ptr = 0
        # RGB bits
        for _ in range(k_rgb):
            y = int(ys[ptr].item()); x = int(xs[ptr].item()); ptr += 1
            c = int(torch.randint(0, C, (1,), generator=g).item())
            b = int(torch.randint(0, rgb_bits_per_channel, (1,), generator=g).item())
            conn[i, _] = rgb_bit_index(c, y, x, b)

        # THERMO bits
        for j in range(k_thermo):
            y = int(ys[ptr].item()); x = int(xs[ptr].item()); ptr += 1
            lvl = int(torch.randint(0, thermo_levels, (1,), generator=g).item())
            conn[i, k_rgb + j] = thermo_bit_index(y, x, lvl)

        # SOBEL bits
        for j in range(k_sobel):
            y = int(ys[ptr].item()); x = int(xs[ptr].item()); ptr += 1
            conn[i, k_rgb + k_thermo + j] = sobel_bit_index(y, x)

    # sanity (optional)
    assert conn.min().item() >= 0 and conn.max().item() < total_bits, (conn.min().item(), conn.max().item(), total_bits)

    return conn.to(device)





import torch


def build_conn0_rgb_thermo_sobel_v2(
    num_luts: int,
    k: int,
    H: int = 32,
    W: int = 32,
    C: int = 3,
    rgb_bits_per_channel: int = 8,
    thermo_levels: int = 8,
    patch=(4, 4),

    frac_thermo: float = 0.33,   # k=9 -> ~3
    frac_sobel: float = 0.22,    # k=9 -> ~2
    sobel_jitter_p: float = 0.25,      # 讓一部分 LUT: k_sobel + 1
    sobel_global_frac: float = 0.5,    # sobel bits 裡有多少比例用 global 抽樣 (其餘用 patch-local)

    seed: int = 42,
    device="cpu",
    ensure_unique_per_lut: bool = True,
):
    """
    concat layout (your current):
      [0 : rgb_in_bits)                    -> rgb bitplane
      [rgb_in_bits : rgb_in_bits+H*W*L)    -> gray thermo (pixel-major, thermo_levels)
      [rgb_in_bits+H*W*L : end)            -> gray sobel (1 bit per pixel)

    Returns:
      conn_idx: LongTensor [num_luts, k], values in [0, total_bits)
    """
    ph, pw = patch

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    rgb_in_bits   = C * H * W * rgb_bits_per_channel
    thermo_in_bits = H * W * thermo_levels
    sobel_in_bits  = H * W

    offset_rgb   = 0
    offset_thermo = rgb_in_bits
    offset_sobel  = rgb_in_bits + thermo_in_bits
    total_bits    = rgb_in_bits + thermo_in_bits + sobel_in_bits

    # -------- helpers (indexing) --------
    def rgb_bit_index(c, y, x, b):
        return offset_rgb + (((c * H + y) * W + x) * rgb_bits_per_channel + b)

    def thermo_bit_index(y, x, level):
        pix = y * W + x
        return offset_thermo + pix * thermo_levels + level

    def sobel_bit_index(y, x):
        pix = y * W + x
        return offset_sobel + pix

    # -------- decide per-LUT counts --------
    k_sobel_base  = int(round(k * frac_sobel))
    k_thermo_base = int(round(k * frac_thermo))
    print('k_sobel_base:', k_sobel_base, 'k_thermo_base:', k_thermo_base)

    # 保底：至少 1 個 rgb bit
    def clamp_counts(k_sobel, k_thermo):
        k_rgb = k - k_sobel - k_thermo
        if k_rgb < 1:
            k_rgb = 1
            # 優先從 thermo 扣，再扣 sobel
            while k_thermo + k_sobel > k - 1:
                if k_thermo > 0:
                    k_thermo -= 1
                elif k_sobel > 0:
                    k_sobel -= 1
                else:
                    break
        return k_rgb, k_thermo, k_sobel

    # -------- build conn --------
    conn = torch.empty((num_luts, k), dtype=torch.long)

    for i in range(num_luts):
        # jitter sobel count: some LUTs use +1 sobel bit
        k_sobel = k_sobel_base
        if sobel_jitter_p > 0:
            if torch.rand((), generator=g).item() < sobel_jitter_p:
                k_sobel = min(k_sobel_base + 1, k)  # cap

        k_thermo = k_thermo_base
        k_rgb, k_thermo, k_sobel = clamp_counts(k_sobel, k_thermo)

        # anchor (shared)
        y0 = int(torch.randint(0, H, (1,), generator=g).item())
        x0 = int(torch.randint(0, W, (1,), generator=g).item())

        # patch-local coordinate sampler
        y_lo = max(0, y0 - ph // 2)
        y_hi = min(H, y0 + (ph + 1) // 2)
        x_lo = max(0, x0 - pw // 2)
        x_hi = min(W, x0 + (pw + 1) // 2)

        used = set()
        pos = 0

        # ---- RGB bits (patch-local) ----
        for _ in range(k_rgb):
            for _try in range(50):
                y = int(torch.randint(y_lo, y_hi, (1,), generator=g).item())
                x = int(torch.randint(x_lo, x_hi, (1,), generator=g).item())
                c = int(torch.randint(0, C, (1,), generator=g).item())
                b = int(torch.randint(0, rgb_bits_per_channel, (1,), generator=g).item())
                idx = rgb_bit_index(c, y, x, b)
                if (not ensure_unique_per_lut) or (idx not in used):
                    used.add(idx)
                    conn[i, pos] = idx
                    pos += 1
                    break
            else:
                # fallback (allow duplicate)
                conn[i, pos] = idx
                pos += 1

        # ---- THERMO bits (patch-local) ----
        for _ in range(k_thermo):
            for _try in range(50):
                y = int(torch.randint(y_lo, y_hi, (1,), generator=g).item())
                x = int(torch.randint(x_lo, x_hi, (1,), generator=g).item())
                lvl = int(torch.randint(0, thermo_levels, (1,), generator=g).item())
                idx = thermo_bit_index(y, x, lvl)
                if (not ensure_unique_per_lut) or (idx not in used):
                    used.add(idx)
                    conn[i, pos] = idx
                    pos += 1
                    break
            else:
                conn[i, pos] = idx
                pos += 1

        # ---- SOBEL bits (mix local/global) ----
        for _ in range(k_sobel):
            for _try in range(50):
                if sobel_global_frac > 0 and torch.rand((), generator=g).item() < sobel_global_frac:
                    # global
                    y = int(torch.randint(0, H, (1,), generator=g).item())
                    x = int(torch.randint(0, W, (1,), generator=g).item())
                else:
                    # local
                    y = int(torch.randint(y_lo, y_hi, (1,), generator=g).item())
                    x = int(torch.randint(x_lo, x_hi, (1,), generator=g).item())

                idx = sobel_bit_index(y, x)
                if (not ensure_unique_per_lut) or (idx not in used):
                    used.add(idx)
                    conn[i, pos] = idx
                    pos += 1
                    break
            else:
                conn[i, pos] = idx
                pos += 1

        assert pos == k, f"pos={pos} != k={k}"

    # sanity
    mn = conn.min().item()
    mx = conn.max().item()
    assert 0 <= mn and mx < total_bits, (mn, mx, total_bits)

    return conn.to(device)






import torch

def build_conn0_mixed_blocks(
    num_luts: int,
    k: int,
    D_rgb: int,
    D_post: int,
    D_sobel: int = 1024,
    k_sobel: int = 2,          # ✅ 這就是你要調的「sobel 的 k」
    k_rgb: int = None,         # 可不填，會自動補
    seed: int = 42,
) -> torch.Tensor:
    """
    return conn_idx: [num_luts, k] in [0, D_total)
    layout: [rgb_bitplane | thermo | sobel]
    """
    assert 0 <= k_sobel <= k
    g = torch.Generator().manual_seed(seed)

    D_total = D_rgb + D_post
    D_thermo = D_post - D_sobel
    assert D_thermo >= 0

    # ranges
    rgb_lo, rgb_hi = 0, D_rgb
    thermo_lo, thermo_hi = D_rgb, D_rgb + D_thermo
    sobel_lo, sobel_hi = D_rgb + D_thermo, D_total

    if k_rgb is None:
        # 先把剩下的給 rgb（你也可以改成 thermo 優先）
        k_rgb = k - k_sobel
    k_rgb = int(k_rgb)
    k_thermo = k - k_rgb - k_sobel
    assert k_thermo >= 0

    # sample indices
    conn = torch.empty((num_luts, k), dtype=torch.long)
    for i in range(num_luts):
        cols = []
        if k_rgb > 0:
            cols.append(torch.randint(rgb_lo, rgb_hi, (k_rgb,), generator=g))
        if k_thermo > 0 and D_thermo > 0:
            cols.append(torch.randint(thermo_lo, thermo_hi, (k_thermo,), generator=g))
        if k_sobel > 0:
            cols.append(torch.randint(sobel_lo, sobel_hi, (k_sobel,), generator=g))

        v = torch.cat(cols, dim=0)

        # shuffle within LUT so bits aren't block-ordered
        perm = torch.randperm(k, generator=g)
        conn[i] = v[perm]

    return conn





import torch
from collections import defaultdict

def build_conn0_from_buckets(
    *,
    num_luts: int,
    k: int,
    total_bits: int,
    bucketizer,                 # function: idx -> bucket_id (string)
    frac_sobel: float = 0.22,
    frac_thermo: float = 0.25,
    seed: int = 42,
    device: str = "cpu",
    allow_dup_within_lut: bool = False,
):
    """
    先把 [0,total_bits) 全部分桶: bucket -> list[bit_idx]
    再依照 frac 配置，每個 LUT 抽 k_rgb/k_th/k_sb 個 idx。
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # 1) bucket -> indices
    buckets = defaultdict(list)
    for idx in range(total_bits):
        bid = bucketizer(idx)
        buckets[bid].append(idx)

    # 2) split bucket names into groups
    rgb_buckets = [b for b in buckets.keys() if b.startswith("RGB_")]
    th_buckets  = [b for b in buckets.keys() if b.startswith("TH_")]
    sb_buckets  = [b for b in buckets.keys() if b.startswith("SB2_")]

    assert len(rgb_buckets) > 0 and len(th_buckets) > 0 and len(sb_buckets) > 0

    print(len(rgb_buckets), len(th_buckets), len(sb_buckets))
    avg_size = np.mean([len(buckets[b]) for b in rgb_buckets])
    print("avg rgb bucket size:", avg_size)
    avg_size = np.mean([len(buckets[b]) for b in th_buckets])
    print("avg th bucket size:", avg_size)
    avg_size = np.mean([len(buckets[b]) for b in sb_buckets])
    print("avg sb bucket size:", avg_size)

    # 3) decide counts
    k_sb = int(round(k * frac_sobel))
    k_th = int(round(k * frac_thermo))
    k_rgb = k - k_sb - k_th
    if k_rgb < 1:
        k_rgb = 1
        while k_sb + k_th > k - 1:
            if k_th > 0:
                k_th -= 1
            elif k_sb > 0:
                k_sb -= 1
            else:
                break

    # helper: sample one index from a random bucket in that group
    def sample_from_group(group_buckets):
        b = group_buckets[int(torch.randint(0, len(group_buckets), (1,), generator=g).item())]
        lst = buckets[b]
        return lst[int(torch.randint(0, len(lst), (1,), generator=g).item())]

    conn = torch.empty((num_luts, k), dtype=torch.long)

    for i in range(num_luts):
        chosen = []
        for _ in range(k_rgb):
            chosen.append(sample_from_group(rgb_buckets))
        for _ in range(k_th):
            chosen.append(sample_from_group(th_buckets))
        for _ in range(k_sb):
            chosen.append(sample_from_group(sb_buckets))

        if not allow_dup_within_lut:
            # de-dup within lut by resampling duplicates (cheap)
            s = set()
            for t in range(len(chosen)):
                tries = 0
                while chosen[t] in s and tries < 20:
                    # re-sample from same group based on position
                    if t < k_rgb:
                        chosen[t] = sample_from_group(rgb_buckets)
                    elif t < k_rgb + k_th:
                        chosen[t] = sample_from_group(th_buckets)
                    else:
                        chosen[t] = sample_from_group(sb_buckets)
                    tries += 1
                s.add(chosen[t])

        conn[i] = torch.tensor(chosen, dtype=torch.long)

    # sanity
    assert conn.min().item() >= 0 and conn.max().item() < total_bits
    return conn.to(device)


import torch, random
from collections import defaultdict

from collections import defaultdict
import re
import torch


def build_conn0_from_buckets_2(
    num_luts: int,
    k: int,
    total_bits: int,
    bucketizer,  # callable(idx:int)->str  e.g. "RGB0_T00_b1", "TH_T77_H_l31", "SB2_T00_l0"
    frac_sobel: float = 0.22,
    frac_thermo: float = 0.25,
    seed: int = 42,
    device: str = "cpu",
    *,
    rgb_prefix: str = "RGB",      # "RGB" 或 "RGB_"
    thermo_prefix: str = "TH_",   # 你目前用 "TH_"
    sobel_prefix: str = "SB2_",   # 你目前用 "SB2_"
    tile_regex: str = r"(T\d\d)", # 8x8 tiles -> T00..T77
    enforce_all_blocks_per_lut: bool = True,  # True: 保證每個 LUT 至少 1 個 RGB/TH/SB bit（若 k 允許）
):
    """
    Build conn_idx for layer0 using bucket sampling with per-LUT tile anchor.

    Assumptions:
      - bucketizer(idx) returns a bucket id string that includes a tile token like "T00".."T77"
      - buckets are partitionable by prefix: RGB / TH_ / SB2_

    Returns:
      conn_idx: LongTensor [num_luts, k], values in [0, total_bits)
    """
    assert k >= 1
    assert 0.0 <= frac_sobel <= 1.0
    assert 0.0 <= frac_thermo <= 1.0

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    tile_pat = re.compile(tile_regex)

    # -----------------------------
    # 1) bucket -> indices
    # -----------------------------
    buckets = defaultdict(list)  # bid -> [idx...]
    for idx in range(total_bits):
        bid = bucketizer(int(idx))
        buckets[bid].append(int(idx))

    # -----------------------------
    # 2) split bucket names into groups
    # -----------------------------
    rgb_bucket_names = [b for b in buckets.keys() if b.startswith(rgb_prefix)]
    th_bucket_names  = [b for b in buckets.keys() if b.startswith(thermo_prefix)]
    sb_bucket_names  = [b for b in buckets.keys() if b.startswith(sobel_prefix)]

    if len(rgb_bucket_names) == 0 or len(th_bucket_names) == 0 or len(sb_bucket_names) == 0:
        raise ValueError(
            f"Empty bucket group(s): rgb={len(rgb_bucket_names)}, th={len(th_bucket_names)}, sb={len(sb_bucket_names)}. "
            f"Check prefixes rgb_prefix={rgb_prefix} thermo_prefix={thermo_prefix} sobel_prefix={sobel_prefix}."
        )

    # group bucket names by tile
    def group_by_tile(names):
        out = defaultdict(list)  # tile -> [bucket_name...]
        for b in names:
            m = tile_pat.search(b)
            if m is None:
                continue
            out[m.group(1)].append(b)
        return out

    rgb_by_tile = group_by_tile(rgb_bucket_names)
    th_by_tile  = group_by_tile(th_bucket_names)
    sb_by_tile  = group_by_tile(sb_bucket_names)

    common_tiles = sorted(set(rgb_by_tile.keys()) & set(th_by_tile.keys()) & set(sb_by_tile.keys()))
    if len(common_tiles) == 0:
        raise ValueError(
            f"No common tiles across blocks. "
            f"rgb_tiles={len(rgb_by_tile)}, th_tiles={len(th_by_tile)}, sb_tiles={len(sb_by_tile)}. "
            f"Check tile token in bucketizer (expected match {tile_regex})."
        )

    # -----------------------------
    # 3) decide k splits
    # -----------------------------
    k_sb = int(round(k * frac_sobel))
    k_th = int(round(k * frac_thermo))
    k_rgb = k - k_th - k_sb

    if enforce_all_blocks_per_lut and k >= 3:
        # make sure each block has >=1
        if k_rgb < 1:
            k_rgb = 1
        if k_th < 1:
            k_th = 1
        if k_sb < 1:
            k_sb = 1
        # adjust down if overflow
        while k_rgb + k_th + k_sb > k:
            # reduce the largest one (prefer reducing thermo then sobel then rgb)
            if k_th >= k_sb and k_th >= k_rgb and k_th > 1:
                k_th -= 1
            elif k_sb >= k_th and k_sb >= k_rgb and k_sb > 1:
                k_sb -= 1
            elif k_rgb > 1:
                k_rgb -= 1
            else:
                break
    else:
        # if not enforcing, still ensure not negative
        if k_rgb < 0:
            # steal from th/sb
            while k_rgb < 0 and k_th > 0:
                k_th -= 1
                k_rgb += 1
            while k_rgb < 0 and k_sb > 0:
                k_sb -= 1
                k_rgb += 1
            if k_rgb < 0:
                k_rgb = 0

    assert k_rgb + k_th + k_sb == k, (k_rgb, k_th, k_sb, k)

    # -----------------------------
    # 4) sampling helper (tile-conditioned)
    # -----------------------------
    def sample_from_tile(tile: str, block: str) -> int:
        if block == "rgb":
            bnames = rgb_by_tile[tile]
        elif block == "th":
            bnames = th_by_tile[tile]
        elif block == "sb":
            bnames = sb_by_tile[tile]
        else:
            raise ValueError(block)

        # choose a bucket name within tile
        b = bnames[int(torch.randint(0, len(bnames), (1,), generator=g).item())]
        lst = buckets[b]
        # choose an index within that bucket
        return int(lst[int(torch.randint(0, len(lst), (1,), generator=g).item())])

    # -----------------------------
    # 5) build conn
    # -----------------------------
    conn = torch.empty((num_luts, k), dtype=torch.long)

    for i in range(num_luts):
        tile = common_tiles[int(torch.randint(0, len(common_tiles), (1,), generator=g).item())]

        ptr = 0
        for _ in range(k_rgb):
            conn[i, ptr] = sample_from_tile(tile, "rgb")
            ptr += 1
        for _ in range(k_th):
            conn[i, ptr] = sample_from_tile(tile, "th")
            ptr += 1
        for _ in range(k_sb):
            conn[i, ptr] = sample_from_tile(tile, "sb")
            ptr += 1

        # optional: shuffle within LUT so blocks not grouped
        perm = torch.randperm(k, generator=g)
        conn[i] = conn[i, perm]

    # final sanity
    mn = int(conn.min().item())
    mx = int(conn.max().item())
    if mn < 0 or mx >= total_bits:
        raise RuntimeError(f"conn out of range: min={mn}, max={mx}, total_bits={total_bits}")

    return conn.to(device)



from collections import defaultdict
import re
import torch


def build_conn0_from_buckets_3(
    num_luts: int,
    k: int,
    total_bits: int,
    bucketizer,
    *,
    frac_sobel: float = 0.22,
    frac_thermo: float = 0.25,
    seed: int = 42,
    device: str = "cpu",
    # optional global fallback (default off, since you said it hurts)
    global_frac_rgb: float = 0.0,
    global_frac_th: float = 0.0,
    global_frac_sb: float = 0.0,
):
    """
    Build conn_idx [num_luts, k] from bucketizer(idx)->bucket_id.

    Key behavior:
      - For each LUT i:
          1) sample a tile anchor (tx,ty)
          2) sample k_rgb from RGB buckets in that tile
             sample k_th  from TH  buckets in that tile
             sample k_sb  from SB2 buckets in that tile
      - This enforces "shared receptive field" across RGB/TH/SB2 inside each LUT.

    bucketizer must return strings with prefixes: "RGB_", "TH_", "SB2_"
    and include "T{tx}{ty}" inside the string (e.g. RGB_C0_T12_b1).
    """

    # -------------------------
    # 1) Build buckets: bucket_id -> [indices]
    # -------------------------
    buckets = defaultdict(list)
    for idx in range(total_bits):
        bid = bucketizer(idx)
        buckets[bid].append(idx)

    rgb_bucket_names = [b for b in buckets.keys() if b.startswith("RGB_")]
    th_bucket_names  = [b for b in buckets.keys() if b.startswith("TH_")]
    sb_bucket_names  = [b for b in buckets.keys() if b.startswith("SB2_")]

    assert len(rgb_bucket_names) > 0, "No RGB_ buckets found. Check bucketizer prefix."
    assert len(th_bucket_names)  > 0, "No TH_ buckets found. Check bucketizer prefix."
    assert len(sb_bucket_names)  > 0, "No SB2_ buckets found. Check bucketizer prefix."

    # -------------------------
    # 2) Group bucket names by tile (tx,ty)
    # -------------------------
    # Expect "..._T{tx}{ty}..." where tx,ty are digits 0..7 for CIFAR-10 (patch=4 => 8x8 tiles)
    tile_pat = re.compile(r"_T(\d)(\d)_")

    def tile_of(bucket_name: str):
        m = tile_pat.search(bucket_name)
        if m is None:
            return None
        return (int(m.group(1)), int(m.group(2)))

    rgb_by_tile = defaultdict(list)
    th_by_tile  = defaultdict(list)
    sb_by_tile  = defaultdict(list)

    for b in rgb_bucket_names:
        t = tile_of(b)
        if t is not None:
            rgb_by_tile[t].append(b)
    for b in th_bucket_names:
        t = tile_of(b)
        if t is not None:
            th_by_tile[t].append(b)
    for b in sb_bucket_names:
        t = tile_of(b)
        if t is not None:
            sb_by_tile[t].append(b)

    # Tiles that have all three blocks available
    valid_tiles = sorted(set(rgb_by_tile.keys()) & set(th_by_tile.keys()) & set(sb_by_tile.keys()))
    assert len(valid_tiles) > 0, "No valid tiles with RGB/TH/SB2 simultaneously. Check bucket naming."

    # -------------------------
    # 3) Decide per-LUT counts from each block
    # -------------------------
    k_sb = int(round(k * frac_sobel))
    k_th = int(round(k * frac_thermo))
    k_rgb = k - k_th - k_sb

    # keep at least 1 rgb bit
    if k_rgb < 1:
        k_rgb = 1
        while (k_th + k_sb) > (k - 1):
            if k_th > 0:
                k_th -= 1
            elif k_sb > 0:
                k_sb -= 1
            else:
                break

    # -------------------------
    # 4) Sampling helpers
    # -------------------------
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    def rand_choice(seq):
        # seq is python list
        j = int(torch.randint(0, len(seq), (1,), generator=g).item())
        return seq[j]


    def jitter_tile(t, rad):
        print('jittering tile', t, 'with radius', rad)
        tx, ty = t
        tx2 = max(0, min(7, tx + int(torch.randint(-rad, rad+1, (1,), generator=g).item())))
        ty2 = max(0, min(7, ty + int(torch.randint(-rad, rad+1, (1,), generator=g).item())))
        return (tx2, ty2)

    def sample_from_tile(block: str, tile):
        """
        block in {"rgb","th","sb"}
        Sample one global idx from that tile's bucket lists.
        With probability global_frac_*: fallback to global uniform among that block's indices.
        """
        if block == "rgb":
            if global_frac_rgb > 0 and torch.rand((), generator=g).item() < global_frac_rgb:
                # global fallback: pick any rgb bucket then any idx inside it
                b = rand_choice(rgb_bucket_names)
                return rand_choice(buckets[b])
            b = rand_choice(rgb_by_tile[tile])
            return rand_choice(buckets[b])

        if block == "th":
            if global_frac_th > 0 and torch.rand((), generator=g).item() < global_frac_th:
                b = rand_choice(th_bucket_names)
                return rand_choice(buckets[b])
            b = rand_choice(th_by_tile[tile])
            return rand_choice(buckets[b])

        if block == "sb":
            if global_frac_sb > 0 and torch.rand((), generator=g).item() < global_frac_sb:
                b = rand_choice(sb_bucket_names)
                return rand_choice(buckets[b])
            b = rand_choice(sb_by_tile[tile])
            return rand_choice(buckets[b])

        raise ValueError(block)

    # -------------------------
    # 5) Build conn_idx
    # -------------------------
    conn = torch.empty((num_luts, k), dtype=torch.long)

    for i in range(num_luts):
        tile = rand_choice(valid_tiles)  # shared tile anchor

        # fill in order: RGB, TH, SB2
        ptr = 0
        for _ in range(k_rgb):
            conn[i, ptr] = int(sample_from_tile("rgb", tile))
            ptr += 1
        for _ in range(k_th):
            conn[i, ptr] = int(sample_from_tile("th", tile))
            ptr += 1
        for _ in range(k_sb):
            conn[i, ptr] = int(sample_from_tile("sb", tile))
            ptr += 1

    # final sanity
    assert int(conn.min().item()) >= 0 and int(conn.max().item()) < total_bits, (
        int(conn.min().item()), int(conn.max().item()), total_bits
    )

    return conn.to(device)






import re
from collections import defaultdict
import torch

def build_conn0_from_buckets_4(
    num_luts: int,
    k: int,
    total_bits: int,
    bucketizer,
    *,
    frac_sobel: float = 0.22,
    frac_thermo: float = 0.25,
    seed: int = 42,
    device: str = "cpu",

    # optional global fallback (default off)
    global_frac_rgb: float = 0.0,
    global_frac_th: float = 0.0,
    global_frac_sb: float = 0.0,

    # ---- jitter controls (the thing you want) ----
    p_rgb_same: float = 1.0,   # RGB 通常最 local
    p_th_same: float = 0.7,    # TH: mostly same tile, some jitter
    p_sb_same: float = 0.4,    # SB2: more jitter
    rad_rgb: int = 0,
    rad_th: int = 1,
    rad_sb: int = 2,
):
    """
    bucketizer(idx)->bucket_id must return prefixes: "RGB_", "TH_", "SB2_"
    and bucket_id must include substring "_T{tx}{ty}_" (tx,ty are digits 0..7).

    This builds conn_idx [num_luts, k] by:
      - anchor a tile0 per LUT
      - sample k_rgb/k_th/k_sb indices from buckets in (tile0 or jittered tile)
      - fallback if jittered tile has no buckets for that block
    """

    # -------------------------
    # 1) Build buckets: bucket_id -> [indices]
    # -------------------------
    buckets = defaultdict(list)
    for idx in range(total_bits):
        bid = bucketizer(idx)
        buckets[bid].append(idx)

    rgb_bucket_names = [b for b in buckets.keys() if b.startswith("RGB_")]
    th_bucket_names  = [b for b in buckets.keys() if b.startswith("TH_")]
    sb_bucket_names  = [b for b in buckets.keys() if b.startswith("SB2_")]

    assert len(rgb_bucket_names) > 0, "No RGB_ buckets found. Check bucketizer prefix."
    assert len(th_bucket_names)  > 0, "No TH_ buckets found. Check bucketizer prefix."
    assert len(sb_bucket_names)  > 0, "No SB2_ buckets found. Check bucketizer prefix."

    # -------------------------
    # 2) Group bucket names by tile (tx,ty)
    # -------------------------
    tile_pat = re.compile(r"_T(\d)(\d)_")

    def tile_of(bucket_name: str):
        m = tile_pat.search(bucket_name)
        if m is None:
            return None
        return (int(m.group(1)), int(m.group(2)))

    rgb_by_tile = defaultdict(list)
    th_by_tile  = defaultdict(list)
    sb_by_tile  = defaultdict(list)

    for b in rgb_bucket_names:
        t = tile_of(b)
        if t is not None:
            rgb_by_tile[t].append(b)
    for b in th_bucket_names:
        t = tile_of(b)
        if t is not None:
            th_by_tile[t].append(b)
    for b in sb_bucket_names:
        t = tile_of(b)
        if t is not None:
            sb_by_tile[t].append(b)

    # Tiles that have all three blocks available (for anchor choice)
    valid_tiles = sorted(set(rgb_by_tile.keys()) & set(th_by_tile.keys()) & set(sb_by_tile.keys()))
    assert len(valid_tiles) > 0, "No valid tiles with RGB/TH/SB2 simultaneously. Check bucket naming."

    # -------------------------
    # 3) Decide per-LUT counts from each block
    # -------------------------
    k_sb = int(round(k * frac_sobel))
    k_th = int(round(k * frac_thermo))
    k_rgb = k - k_th - k_sb

    if k_rgb < 1:
        k_rgb = 1
        while (k_th + k_sb) > (k - 1):
            if k_th > 0:
                k_th -= 1
            elif k_sb > 0:
                k_sb -= 1
            else:
                break

    # -------------------------
    # 4) RNG helpers
    # -------------------------
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    def rand_choice(lst):
        return lst[int(torch.randint(0, len(lst), (1,), generator=g).item())]

    def jitter_tile(t, rad):
        if rad <= 0:
            return t
        tx, ty = t
        tx2 = max(0, min(7, tx + int(torch.randint(-rad, rad + 1, (1,), generator=g).item())))
        ty2 = max(0, min(7, ty + int(torch.randint(-rad, rad + 1, (1,), generator=g).item())))
        return (tx2, ty2)

    def choose_tile_for_block(block: str, tile0):
        if block == "rgb":
            p_same, rad, by_tile = p_rgb_same, rad_rgb, rgb_by_tile
        elif block == "th":
            p_same, rad, by_tile = p_th_same, rad_th, th_by_tile
        elif block == "sb":
            p_same, rad, by_tile = p_sb_same, rad_sb, sb_by_tile
        else:
            raise ValueError(block)

        # pick tile candidate
        tile = tile0 if torch.rand((), generator=g).item() < p_same else jitter_tile(tile0, rad)

        # fallback if tile not available for that block
        if tile not in by_tile or len(by_tile[tile]) == 0:
            if tile0 in by_tile and len(by_tile[tile0]) > 0:
                return tile0
            # last resort: pick some tile that has that block
            any_tiles = list(by_tile.keys())
            return rand_choice(any_tiles)

        return tile

    def sample_from_tile(block: str, tile0):
        # optional global fallback
        if block == "rgb" and global_frac_rgb > 0 and torch.rand((), generator=g).item() < global_frac_rgb:
            b = rand_choice(rgb_bucket_names)
            return rand_choice(buckets[b])
        if block == "th" and global_frac_th > 0 and torch.rand((), generator=g).item() < global_frac_th:
            b = rand_choice(th_bucket_names)
            return rand_choice(buckets[b])
        if block == "sb" and global_frac_sb > 0 and torch.rand((), generator=g).item() < global_frac_sb:
            b = rand_choice(sb_bucket_names)
            return rand_choice(buckets[b])

        tile = choose_tile_for_block(block, tile0)

        if block == "rgb":
            b = rand_choice(rgb_by_tile[tile])
        elif block == "th":
            b = rand_choice(th_by_tile[tile])
        elif block == "sb":
            b = rand_choice(sb_by_tile[tile])
        else:
            raise ValueError(block)

        return rand_choice(buckets[b])

    # -------------------------
    # 5) Build conn_idx
    # -------------------------
    conn = torch.empty((num_luts, k), dtype=torch.long)

    for i in range(num_luts):
        tile0 = rand_choice(valid_tiles)  # shared anchor for this LUT

        ptr = 0
        for _ in range(k_rgb):
            conn[i, ptr] = int(sample_from_tile("rgb", tile0)); ptr += 1
        for _ in range(k_th):
            conn[i, ptr] = int(sample_from_tile("th", tile0)); ptr += 1
        for _ in range(k_sb):
            conn[i, ptr] = int(sample_from_tile("sb", tile0)); ptr += 1

    assert int(conn.min().item()) >= 0 and int(conn.max().item()) < total_bits, (
        int(conn.min().item()), int(conn.max().item()), total_bits
    )

    return conn.to(device)






    import re
import torch
from collections import defaultdict
from typing import Callable, Tuple, Dict, List

def build_conn0_hybrid_from_buckets(
    num_luts: int,
    k: int,
    total_bits: int,
    bucketizer: Callable[[int], str],
    *,
    frac_sobel: float = 0.22,
    frac_thermo: float = 0.25,
    seed: int = 42,
    device: str = "cpu",

    # hybrid control: fraction of LUTs using bucket/tile structure
    p_struct: float = 0.35,

    # bucket sampling locality controls (only used when struct LUT is chosen)
    p_rgb_same: float = 1.0,
    p_th_same: float = 0.7,
    p_sb_same: float = 0.4,
    rad_rgb: int = 0,
    rad_th: int = 1,
    rad_sb: int = 2,

    # IMPORTANT: de-duplicate indices inside each LUT to keep effective k
    dedup_within_lut: bool = True,
):
    """
    Hybrid conn builder:
      - Build buckets from bucketizer(idx)->bucket_id, bucket_id includes "_T{tx}{ty}_".
      - For each LUT:
          with prob p_struct:
              sample k_rgb/k_th/k_sb using tile-anchored bucket selection (+ optional jitter)
          else:
              sample k_rgb/k_th/k_sb uniformly from each block's indices (global within block)
      - Optionally enforce no duplicate indices within each LUT (recommended).
    """

    # -------------------------
    # 0) RNG
    # -------------------------
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    def randu():
        return float(torch.rand((), generator=g).item())

    def randint(a: int, b: int) -> int:
        # inclusive a..b
        return int(torch.randint(a, b + 1, (1,), generator=g).item())

    def rand_choice(lst: List):
        return lst[int(torch.randint(0, len(lst), (1,), generator=g).item())]

    # -------------------------
    # 1) buckets: bucket_id -> [indices]
    # -------------------------
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx in range(total_bits):
        bid = bucketizer(idx)
        buckets[bid].append(idx)

    rgb_bucket_names = [b for b in buckets.keys() if b.startswith("RGB_")]
    th_bucket_names  = [b for b in buckets.keys() if b.startswith("TH_")]
    sb_bucket_names  = [b for b in buckets.keys() if b.startswith("SB2_")]

    assert len(rgb_bucket_names) > 0, "No RGB_ buckets found. Check bucketizer prefix."
    assert len(th_bucket_names)  > 0, "No TH_ buckets found. Check bucketizer prefix."
    assert len(sb_bucket_names)  > 0, "No SB2_ buckets found. Check bucketizer prefix."

    # -------------------------
    # 2) group buckets by tile
    # -------------------------
    tile_pat = re.compile(r"_T(\d)(\d)_")

    def tile_of(bucket_name: str):
        m = tile_pat.search(bucket_name)
        if m is None:
            return None
        return (int(m.group(1)), int(m.group(2)))

    rgb_by_tile = defaultdict(list)
    th_by_tile  = defaultdict(list)
    sb_by_tile  = defaultdict(list)

    for b in rgb_bucket_names:
        t = tile_of(b)
        if t is not None:
            rgb_by_tile[t].append(b)
    for b in th_bucket_names:
        t = tile_of(b)
        if t is not None:
            th_by_tile[t].append(b)
    for b in sb_bucket_names:
        t = tile_of(b)
        if t is not None:
            sb_by_tile[t].append(b)

    valid_tiles = sorted(set(rgb_by_tile.keys()) & set(th_by_tile.keys()) & set(sb_by_tile.keys()))
    assert len(valid_tiles) > 0, "No valid tiles with RGB/TH/SB2 simultaneously. Check bucket naming."

    # -------------------------
    # 3) decide k per block
    # -------------------------
    k_sb = int(round(k * frac_sobel))
    k_th = int(round(k * frac_thermo))
    k_rgb = k - k_th - k_sb

    # keep at least 1 rgb bit
    if k_rgb < 1:
        k_rgb = 1
        while (k_th + k_sb) > (k - 1):
            if k_th > 0:
                k_th -= 1
            elif k_sb > 0:
                k_sb -= 1
            else:
                break

    # -------------------------
    # 4) infer block ranges (IMPORTANT)
    # We infer D_rgb/D_th/D_sb by scanning a few sentinel points.
    # But you already know total_bits = D_rgb + D_th + D_sb.
    #
    # The safest is to pass these explicitly, BUT since you didn't,
    # we derive them from bucket prefixes:
    #   - min idx in TH_ buckets gives offset_TH (=D_rgb)
    #   - min idx in SB2_ buckets gives offset_SB (=D_rgb + D_th)
    #   - then D_sb = total_bits - offset_SB
    # -------------------------
    min_th = min(min(buckets[b]) for b in th_bucket_names)
    min_sb = min(min(buckets[b]) for b in sb_bucket_names)

    D_rgb = min_th
    D_th  = min_sb - min_th
    D_sb  = total_bits - min_sb

    # sanity (matches your printed: 6144/32768/2048/40960)
    assert D_rgb > 0 and D_th > 0 and D_sb > 0, (D_rgb, D_th, D_sb, total_bits)

    # -------------------------
    # 5) helpers: jitter tile + sample from bucket list
    # -------------------------
    def jitter_tile(tile: Tuple[int, int], rad: int):
        if rad <= 0:
            return tile
        tx, ty = tile
        tx2 = max(0, min(7, tx + randint(-rad, rad)))
        ty2 = max(0, min(7, ty + randint(-rad, rad)))
        return (tx2, ty2)

    def pick_bucket_in_tile(block: str, tile: Tuple[int, int], p_same: float, rad: int):
        # choose tile: same with prob p_same else jittered
        t2 = tile if randu() < p_same else jitter_tile(tile, rad)

        if block == "rgb":
            lst = rgb_by_tile[t2]
        elif block == "th":
            lst = th_by_tile[t2]
        elif block == "sb":
            lst = sb_by_tile[t2]
        else:
            raise ValueError(block)

        return rand_choice(lst)

    def sample_structured_one(block: str, tile: Tuple[int, int]) -> int:
        if block == "rgb":
            b = pick_bucket_in_tile("rgb", tile, p_rgb_same, rad_rgb)
        elif block == "th":
            b = pick_bucket_in_tile("th", tile, p_th_same, rad_th)
        elif block == "sb":
            b = pick_bucket_in_tile("sb", tile, p_sb_same, rad_sb)
        else:
            raise ValueError(block)
        return int(rand_choice(buckets[b]))

    def sample_unstructured_one(block: str) -> int:
        # uniform inside each block's index range (fast and diverse)
        if block == "rgb":
            return int(torch.randint(0, D_rgb, (1,), generator=g).item())
        if block == "th":
            return D_rgb + int(torch.randint(0, D_th, (1,), generator=g).item())
        if block == "sb":
            return D_rgb + D_th + int(torch.randint(0, D_sb, (1,), generator=g).item())
        raise ValueError(block)

    # -------------------------
    # 6) build conn
    # -------------------------
    conn = torch.empty((num_luts, k), dtype=torch.long)

    for i in range(num_luts):
        structured = (randu() < p_struct)
        tile = rand_choice(valid_tiles)  # anchor tile (used only if structured)

        chosen = []
        def add_one(idx: int):
            if not dedup_within_lut:
                chosen.append(idx)
                return
            # dedup
            if idx in chosen:
                return
            chosen.append(idx)

        # fill target counts
        targets = [("rgb", k_rgb), ("th", k_th), ("sb", k_sb)]
        for block, cnt in targets:
            tries = 0
            while sum(1 for x in chosen if True) < k and cnt > 0:
                if structured:
                    idx = sample_structured_one(block, tile)
                else:
                    idx = sample_unstructured_one(block)

                add_one(idx)
                if len(chosen) >= (k - sum(c for _, c in targets if False)):
                    # no-op, kept for readability
                    pass

                # decrement only when we successfully add (so counts stay correct under dedup)
                if (not dedup_within_lut) or (idx in chosen):
                    cnt -= 1

                tries += 1
                # avoid infinite loop under aggressive dedup
                if tries > 50 * k:
                    break

        # if dedup caused shortfall, backfill globally (any block) until length k
        tries = 0
        while len(chosen) < k:
            # backfill from any block, but prefer rgb to keep at least 1
            block = "rgb" if randu() < 0.5 else ("th" if randu() < 0.5 else "sb")
            idx = sample_unstructured_one(block)
            add_one(idx)
            tries += 1
            if tries > 200 * k:
                break

        conn[i, :] = torch.tensor(chosen[:k], dtype=torch.long)

    # final sanity
    assert int(conn.min().item()) >= 0 and int(conn.max().item()) < total_bits, (
        int(conn.min().item()), int(conn.max().item()), total_bits
    )
    return conn.to(device)