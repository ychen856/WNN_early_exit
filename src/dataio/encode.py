# Thermometer (8-level) tiling + encoding utilities for MNIST-like images.
# You can paste this into your repo (e.g., src/dataio/thermo_encode.py).
# Demo at the bottom shows how to use it on a dummy 28x28 image.
import numpy as np
import pandas as pd
import json
from pathlib import Path
from os.path import join
from datasets.LoadDatasets import MnistDataloader

import matplotlib
matplotlib.use('TkAgg')

from typing import Tuple, Dict, Any, List, Optional, Union
from scipy import ndimage
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]

############################
# helpers: unify tensor/numpy
############################

def _to_numpy_image_2d(x: ArrayLike) -> np.ndarray:
    """
    Accepts:
      - torch.Tensor (H,W) or (1,H,W)
      - np.ndarray   (H,W) or (1,H,W)
    Returns:
      numpy ndarray shape (H,W), dtype float32 or uint8 depending on source
    Does NOT normalize to [0,1]; just converts and squeezes channel dim if needed.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)  # (1,H,W) -> (H,W)
        x = x.numpy()
    else:
        x = np.asarray(x)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]         # (1,H,W) -> (H,W)

    assert x.ndim == 2, f"Expected 2D grayscale image, got shape {x.shape}"
    return x


############################
# normalization
############################

def _normalize(x: ArrayLike, input_range: Optional[Tuple[float, float]] = (0, 255)) -> np.ndarray:
    """
    Normalize to [0,1].
    Works for torch.Tensor or np.ndarray.
    Returns numpy float32 array.
    """
    x_np = _to_numpy_image_2d(x).astype(np.float32)

    if input_range is None:
        a, b = float(x_np.min()), float(x_np.max())
    else:
        a, b = input_range

    if b == a:
        return np.zeros_like(x_np, dtype=np.float32)

    z = (x_np - a) / (b - a)
    z = np.clip(z, 0.0, 1.0)
    return z.astype(np.float32)

############################
# 
############################
def minmax_normalize(x, eps=1e-8):
    """
    x: [N, D]
    """
    xmin = x.min(dim=0, keepdim=True)[0]
    xmax = x.max(dim=0, keepdim=True)[0]
    return (x - xmin) / (xmax - xmin + eps)

############################
# tiling + thermometer
############################

def _split_into_tiles(img2d: np.ndarray, tiles: Tuple[int, int]) -> List[np.ndarray]:
    H, W = img2d.shape
    tr, tc = tiles
    assert H % tr == 0 and W % tc == 0, "Image size must be divisible by tile grid."
    th, tw = H // tr, W // tc
    out = []
    for r in range(tr):
        for c in range(tc):
            out.append(img2d[r*th:(r+1)*th, c*tw:(c+1)*tw])
    return out

def _thermometer_encode_tile(
    tile2d: np.ndarray,
    levels: int = 8,
    input_range=(0,255),
) -> np.ndarray:
    """
    tile2d: (th, tw) uint8 or float
    Returns uint8 bits of shape (th*tw*levels,)
    """
    arr01 = _normalize(tile2d, input_range=input_range)  # (th,tw) float32 in [0,1]
    th, tw = arr01.shape
    # thresholds[k] = k/levels
    thresholds = (np.arange(levels, dtype=np.float32)[None, None, :] / levels)
    bits = (arr01[:, :, None] > thresholds).astype(np.uint8)  # (th,tw,levels)
    return bits.reshape(-1)  # flatten -> (th*tw*levels,)

def encode_image_thermo_tiled(
    img: ArrayLike,
    tiles: Tuple[int, int] = (4,4),
    levels: int = 8,
    input_range=(0,255),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Classic thermometer tiling:
    - Split into tiles (e.g. 4x4 -> 16 tiles, each 7x7 for MNIST)
    - Encode each tile using thermometer levels bits/pixel
    Returns:
      vec: uint8 bit vector (length total_bits)
      meta: info about tiling layout
    """
    img2d = _to_numpy_image_2d(img)  # (H,W)
    H, W = img2d.shape
    tile_list = _split_into_tiles(img2d, tiles)

    tr, tc = tiles
    th, tw = H // tr, W // tc

    encoded_tiles = []
    ranges = []
    cur = 0
    for t in tile_list:
        b = _thermometer_encode_tile(
            t,
            levels=levels,
            input_range=input_range,
        )  # shape (th*tw*levels,)
        encoded_tiles.append(b)
        ranges.append((cur, cur + b.size))
        cur += b.size

    thermo_vec = np.concatenate(encoded_tiles, axis=0).astype(np.uint8)

    thermo_meta = dict(
        tiles=tiles,
        tile_size=(th, tw),
        levels=levels,
        bits_per_tile=th*tw*levels,
        total_bits=int(thermo_vec.size),
        tile_index_ranges=ranges,
        input_shape=(H, W),
        input_range=input_range,
    )
    return thermo_vec, thermo_meta


def encode_batch(images: np.ndarray, tiles=(4,4), levels=8) -> Tuple[np.ndarray, Dict[str,Any]]:
    """Encode a batch of 2D images to bit-vectors. Returns (X_bits, meta)."""
    X_bits_list = []
    meta_ref = None
    for img in images:
        img =np.array(img)
        vec, meta = encode_image_thermo_tiled(img, tiles=tiles, levels=levels, input_range=(0,255))
        X_bits_list.append(vec)
        if meta_ref is None:
            meta_ref = meta
    X_bits = np.stack(X_bits_list, axis=0)  # (N, total_bits)
    return X_bits, meta_ref

############################
#
############################
def compute_dt_thresholds(x_train, z=32, eps=1e-8, max_elems=2_000_000):
    """
    x_train: [N, 28, 28] or [N, D]
    return:
      thresholds: [z]  (global quantiles, on CPU)
      xmin, xmax: [1, D] (for normalize; on CPU)
    """
    # 1) flatten -> [N, D]
    if x_train.dim() > 2:
        x = x_train.view(x_train.size(0), -1)
    else:
        x = x_train

    x = x.float().cpu()
    # 2) feature-wise's min/max, normalize should use the same set
    xmin = x.min(dim=0, keepdim=True)[0]
    xmax = x.max(dim=0, keepdim=True)[0]
    x_norm = (x - xmin) / (xmax - xmin + eps)

    # 3) flatten to a global quantile row
    flat = x_norm.view(-1)  # [N*D]
    n = flat.numel()

    if n > max_elems:
        # sampling
        step = max(1, n // max_elems)
        idx = torch.arange(0, n, step=step)
        if idx.numel() > max_elems:
            idx = idx[:max_elems]
        flat_sample = flat[idx]
    else:
        flat_sample = flat

    # 4) calcualte z scores（remove 0 and 1）
    q = torch.linspace(0, 1, steps=z+2)[1:-1]  # [z]
    thresholds = torch.quantile(flat_sample, q)

    print("x_flat D:", x.view(x.size(0), -1).shape[1])
    print("thresholds shape:", thresholds.shape)
    print(f'xmin: {xmin.min().item():.4f}, xmax: {xmax.max().item():.4f}, thresholds min: {thresholds.min().item():.4f}, max: {thresholds.max().item():.4f}')
    return thresholds, xmin, xmax

def dt_thermometer_encode(x, thresholds, xmin, xmax, eps=1e-8):
    """
    x: [B, 28, 28] or [B, D]
    thresholds, xmin, xmax: from compute_dt_thresholds (in CPU)
    """
    device = x.device

    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    x = x.float()

    xmin_dev = xmin.to(device)
    xmax_dev = xmax.to(device)
    th_dev   = thresholds.to(device)
    print(f'th_dev shape: {th_dev.shape}, dtype: {th_dev.dtype}')

    x = (x - xmin_dev) / (xmax_dev - xmin_dev + eps)  
    B, D = x.shape

    x_exp = x.unsqueeze(-1)            # [B, D, 1]
    bits = (x_exp > th_dev).float()   # [B, D, z]
    return bits.view(B, D * len(th_dev))


def thermometer_encode(x, z=32, eps=1e-8):
    """
    x: [B, D] or [B, C, H, W] / [B, H, W]
    auto do:
      1) flatten to [B, D]
      2) normalize to [0, 1]
      3) do thermometer encoding -> [B, D*z]
    """
    # 1) if is image (3D/4D), flatten
    if x.dim() > 2:
        x = x.view(x.size(0), -1)     # [B, D]

    # 2) convert into float and normalize to [0,1]
    x = x.float()
    xmin = x.min(dim=0, keepdim=True)[0]
    xmax = x.max(dim=0, keepdim=True)[0]
    x = (x - xmin) / (xmax - xmin + eps)

    # 3) thermometer encoding
    B, D = x.shape
    device = x.device
    thresholds = torch.linspace(0, 1, steps=z+1, device=device)[1:]  # [z]
    x_exp = x.unsqueeze(-1)                    # [B, D, 1]
    bits = (x_exp > thresholds).float()       # [B, D, z]
    return bits.view(B, D * z)                # [B, D*z]


############################
# sobel edge bits
############################

def _sobel_edge_bits(
    img: ArrayLike,
    threshold_ratio: float = 0.2,
) -> np.ndarray:
    """
    Compute binary edge map using Sobel magnitude.
    1. sobel x/y
    2. mag = |gx| + |gy|
    3. threshold at 'threshold_ratio' * max(mag)
    Return uint8 flat vector of {0,1}.
    """
    img2d = _to_numpy_image_2d(img).astype(np.float32)

    gx = ndimage.sobel(img2d, axis=1, mode='reflect')
    gy = ndimage.sobel(img2d, axis=0, mode='reflect')
    mag = np.abs(gx) + np.abs(gy)

    max_mag = float(mag.max()) if mag.max() > 0 else 1.0
    thr = threshold_ratio * max_mag
    edge_bin = (mag >= thr).astype(np.uint8)  # (H,W)
    return edge_bin.reshape(-1)  # flatten to (H*W,)


############################
# combined encoding: thermo + sobel
############################

def encode_image_thermo_plus_sobel(
    img: ArrayLike,
    tiles: Tuple[int, int] = (4,4),
    levels: int = 8,
    input_range=(0,255),
    sobel_threshold_ratio: float = 0.2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    1. thermometer-encoded tiled bits
    2. sobel edge bits (global over whole image)
    concat -> full_vec

    Returns:
      full_vec: uint8 bit vector of length total_bits
      meta_ext: {
         "thermo": {...},
         "sobel": {
            "bits_per_image": H*W,
            "index_range": (start, end),
            "threshold_ratio": sobel_threshold_ratio
         },
         "feature_blocks": [
             {"name":"thermo","index_range":(...)},
             {"name":"sobel","index_range":(...)}
         ],
         "total_bits": ...,
         "input_shape": (H,W),
         "tiles": ...,
         "tile_index_ranges": ...,
         "levels": ...,
         "tile_size": ...,
         "input_range": ...
      }
    """
    thermo_vec, thermo_meta = encode_image_thermo_tiled(
        img,
        tiles=tiles,
        levels=levels,
        input_range=input_range,
    )

    sobel_vec = _sobel_edge_bits(
        img,
        threshold_ratio=sobel_threshold_ratio,
    ).astype(np.uint8)

    start_sobel = thermo_vec.size
    end_sobel   = start_sobel + sobel_vec.size

    full_vec = np.concatenate([thermo_vec, sobel_vec], axis=0).astype(np.uint8)

    meta_ext = {
        "thermo": thermo_meta,
        "sobel": {
            "bits_per_image": int(sobel_vec.size),
            "index_range": (int(start_sobel), int(end_sobel)),
            "threshold_ratio": sobel_threshold_ratio,
        },
        "feature_blocks": [
            {
                "name": "thermo",
                "index_range": (0, int(thermo_vec.size))
            },
            {
                "name": "sobel",
                "index_range": (int(start_sobel), int(end_sobel))
            }
        ],
        "total_bits": int(full_vec.size),
        "input_shape": thermo_meta["input_shape"],
        "tiles": thermo_meta["tiles"],
        "tile_index_ranges": thermo_meta["tile_index_ranges"],
        "levels": thermo_meta["levels"],
        "tile_size": thermo_meta["tile_size"],
        "input_range": thermo_meta["input_range"],
    }

    return full_vec, meta_ext


############################
# batch version
############################

def encode_batch_thermo_plus_sobel(
    imgs: ArrayLike,
    tiles: Tuple[int, int] = (4,4),
    levels: int = 8,
    input_range=(0,255),
    sobel_threshold_ratio: float = 0.2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    imgs: can be
        - np.ndarray of shape (N, H, W)
        - torch.Tensor of shape (N, H, W) or (N,1,H,W)

    Returns:
       X_bits: (N, total_bits) uint8 {0,1}
       meta_ref: meta dict from first sample (assumed consistent for all)
    """
    # If tensor, iterate sample by sample
    if isinstance(imgs, torch.Tensor):
        imgs_iter = imgs.detach().cpu()
        # normalize shapes like (N,1,H,W) -> (N,H,W)
        if imgs_iter.ndim == 4 and imgs_iter.shape[1] == 1:
            imgs_iter = imgs_iter[:,0,:,:]  # drop channel dim
        assert imgs_iter.ndim == 3, f"Expected (N,H,W), got {imgs_iter.shape}"
        imgs_list = [imgs_iter[i] for i in range(imgs_iter.shape[0])]
    else:
        imgs_np = np.asarray(imgs)
        # (N,1,H,W) -> (N,H,W)
        if imgs_np.ndim == 4 and imgs_np.shape[1] == 1:
            imgs_np = imgs_np[:,0,:,:]
        assert imgs_np.ndim == 3, f"Expected (N,H,W), got {imgs_np.shape}"
        imgs_list = [imgs_np[i] for i in range(imgs_np.shape[0])]

    X_list = []
    meta_ref = None
    for im in imgs_list:
        vec, meta = encode_image_thermo_plus_sobel(
            im,
            tiles=tiles,
            levels=levels,
            input_range=input_range,
            sobel_threshold_ratio=sobel_threshold_ratio
        )
        X_list.append(vec)
        if meta_ref is None:
            meta_ref = meta

    X_bits = np.stack(X_list, axis=0).astype(np.uint8)  # (N, total_bits)
    return X_bits, meta_ref


# ------------------ MNIST+thermometer bucket mapping（tile × level） ------------------
def bucket_mapper_mnist_thermo(global_bit_id: int,
                               W: int = 28, H: int = 28,
                               thermo_levels: int = 8,
                               tile_size: int = 7) -> str:
    """
    Using 28x28 + thermometer-8, assume bucket mapping:
      - pixel = bit // 8
      - level = bit % 8 → L(0-3)/H(4-7)
      - image divided into (H/tile_size) x (W/tile_size) tiles
    """
    pixel = global_bit_id // thermo_levels
    level = global_bit_id % thermo_levels
    x = pixel % W
    y = pixel // W
    tx = x // tile_size
    ty = y // tile_size
    lvl = "L" if level < (thermo_levels // 2) else "H"
    return f"T{tx}{ty}_{lvl}"  # 4x4x2=32 buckets


def save_meta(meta: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)




if __name__ == '__main__':
    input_path = 'D:/workspace/Adaptive_WNN/datasets'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    print('dataPath: ', training_labels_filepath)

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    for (img, label) in zip(x_train, y_train):
        print('img: ', img)
        print('lable: ', label)
        img = np.array(img)
        bit_vec, meta = encode_image_thermo_tiled(img, tiles=(4, 4), levels=8, input_range=(0.0, 255.0))

        # Summarize to a small dataframe and show to user
        df = pd.DataFrame({
            "key": ["tiles", "tile_size", "levels", "bits_per_tile", "total_bits", "first_32_bits"],
            "value": [
                str(meta["tiles"]),
                str(meta["tile_size"]),
                meta["levels"],
                meta["bits_per_tile"],
                meta["total_bits"],
                ''.join(map(str, bit_vec[:32].tolist()))
            ]
        })
        print('Thermometer Encoding Summary (Dummy 28x28)', df)





#======================
# CIRAR10-specific loading + encoding
#======================

import torch

def bitplane_encode_u8_rgb(x_u8: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    x_u8: uint8, shape [N, 3, 32, 32]
    return: float32 bits in {0,1}, shape [N, 3*32*32*bits]
    bit order: LSB->MSB
    """
    assert x_u8.dtype == torch.uint8
    assert x_u8.dim() == 4 and x_u8.size(1) == 3

    x = x_u8.to(torch.int32)  # [N,3,H,W]
    shifts = torch.arange(bits, device=x.device, dtype=torch.int32)  # [bits]
    # [N,3,H,W,1] >> [bits] -> broadcast -> [N,3,H,W,bits]
    b = (x.unsqueeze(-1) >> shifts) & 1
    b = b.to(torch.float32)
    return b.reshape(x.size(0), -1)


def bitplane_encode_u8_gray(x_u8: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    x_u8: uint8, shape [N, 32, 32] or [N,1,32,32]
    return: float32 bits in {0,1}, shape [N, 32*32*bits]
    """
    assert x_u8.dtype == torch.uint8
    if x_u8.dim() == 4:
        assert x_u8.size(1) == 1
        x_u8 = x_u8[:, 0]  # [N,32,32]
    assert x_u8.dim() == 3

    x = x_u8.to(torch.int32)  # [N,H,W]
    shifts = torch.arange(bits, device=x.device, dtype=torch.int32)  # [bits]
    b = (x.unsqueeze(-1) >> shifts) & 1  # [N,H,W,bits]
    b = b.to(torch.float32)
    return b.reshape(x.size(0), -1)

def bitplane_encode_u8_gray(x_u8: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    x_u8: uint8, shape [N, 32, 32] or [N,1,32,32]
    return: float32 bits in {0,1}, shape [N, 32*32*bits]
    """
    assert x_u8.dtype == torch.uint8
    if x_u8.dim() == 4:
        assert x_u8.size(1) == 1
        x_u8 = x_u8[:, 0]  # [N,32,32]
    assert x_u8.dim() == 3

    x = x_u8.to(torch.int32)  # [N,H,W]
    shifts = torch.arange(bits, device=x.device, dtype=torch.int32)  # [bits]
    b = (x.unsqueeze(-1) >> shifts) & 1  # [N,H,W,bits]
    b = b.to(torch.float32)
    return b.reshape(x.size(0), -1)











import torch
import torch.nn.functional as F

def sobel_edge_bits_batch(x_u8: torch.Tensor, ratio: float = 0.2) -> torch.Tensor:
    """
    x_u8: [B,H,W] uint8 (0..255)
    return: [B, H*W] float {0,1}
    """
    assert x_u8.ndim == 3
    x = x_u8.float().unsqueeze(1)  # [B,1,H,W]

    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy + 1e-8).squeeze(1)  # [B,H,W]

    # per-image threshold
    mmax = mag.flatten(1).amax(dim=1, keepdim=True)  # [B,1]
    thr  = ratio * mmax
    bits = (mag.flatten(1) > thr).float()            # [B,H*W]
    return bits


def encode_gray_thermo_plus_sobel_batch(
    x_u8: torch.Tensor,                 # [B,H,W] uint8
    thresholds: torch.Tensor,            # [L] float (CPU or device 都行)
    xmin: torch.Tensor,                  # [1,D] float
    xmax: torch.Tensor,                  # [1,D] float
    sobel_ratio: float = 0.2,
) -> torch.Tensor:
    """
    returns bits: [B, D_thermo + H*W] float32 {0,1}
    """
    device = x_u8.device

    # thermo: [B, H*W*L]
    thermo_bits = dt_thermometer_encode(
        x_u8, thresholds=thresholds, xmin=xmin, xmax=xmax
    ).to(device)

    # sobel: [B, H*W]
    sobel_bits = sobel_edge_bits_batch(x_u8, ratio=sobel_ratio).to(device)
    print("sobel_gray_bits shape:", sobel_bits.shape)

    return torch.cat([thermo_bits, sobel_bits], dim=1)




################################
# compute 2-level sobel thresholds from train set
###############################
import torch
import torch.nn.functional as F

@torch.no_grad()
def sobel_mag_gray_u8_batch(gray_u8: torch.Tensor, device=None) -> torch.Tensor:
    """
    gray_u8: [B,H,W] uint8 (0..255)
    return:  mag [B,H,W] float32
    """
    if device is None:
        device = gray_u8.device
    x = gray_u8.to(device).float() / 255.0  # [B,H,W]
    x = x.unsqueeze(1)  # [B,1,H,W]

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=torch.float32, device=device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],
                       [ 0, 0, 0],
                       [ 1, 2, 1]], dtype=torch.float32, device=device).view(1,1,3,3)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)  # [B,1,H,W]
    return mag[:, 0]  # [B,H,W]

@torch.no_grad()
def compute_sobel_two_thresholds(
    train_gray_u8: torch.Tensor,
    q1: float = 0.80,
    q2: float = 0.92,
    max_samples: int = 10000,
    device: str = "cpu",
) -> tuple[float, float]:
    """
    用 train 的 sobel magnitude 取兩個 quantile 當 threshold。
    回傳 (t1, t2)，且保證 t2 >= t1。
    """
    assert train_gray_u8.dim() == 3, f"expected [N,H,W], got {train_gray_u8.shape}"
    N = train_gray_u8.size(0)
    if N > max_samples:
        idx = torch.randperm(N)[:max_samples]
        gray = train_gray_u8[idx]
    else:
        gray = train_gray_u8

    mag = sobel_mag_gray_u8_batch(gray, device=device)  # [B,H,W]
    v = mag.flatten()  # [B*H*W]
    t1 = float(torch.quantile(v, q1).item())
    t2 = float(torch.quantile(v, q2).item())
    if t2 < t1:
        t1, t2 = t2, t1
    return t1, t2


##############################
# sobel 2-level encoding
##############################
@torch.no_grad()
def sobel_2level_bits_gray_u8(
    gray_u8: torch.Tensor,
    t1: float,
    t2: float,
    device=None,
) -> torch.Tensor:
    """
    gray_u8: [B,H,W] uint8
    return: bits [B, H*W*2] float32 {0,1}
    layout: pixel-major, then level(0=t1, 1=t2)
    """
    if device is None:
        device = gray_u8.device
    mag = sobel_mag_gray_u8_batch(gray_u8, device=device)  # [B,H,W]
    b1 = (mag > t1).to(torch.float32)  # [B,H,W]
    b2 = (mag > t2).to(torch.float32)  # [B,H,W]

    B, H, W = b1.shape
    # pixel-major then level: [B, H*W, 2] -> [B, H*W*2]
    bits = torch.stack([b1, b2], dim=-1).view(B, H * W * 2)
    return bits

@torch.no_grad()
def encode_gray_thermo_plus_sobel2_batch(
    gray_u8: torch.Tensor,             # [B,32,32] uint8
    thermo_bits: torch.Tensor,          # [B, H*W*thermo_levels] float {0,1}
    sobel_t1: float,
    sobel_t2: float,
    device=None,
) -> torch.Tensor:
    """
    concat: [THERMO (pixel-major, thermo_levels)] + [SOBEL2 (pixel-major, 2 levels)]
    return: [B, H*W*thermo_levels + H*W*2]
    """
    if device is None:
        device = gray_u8.device

    sobel2 = sobel_2level_bits_gray_u8(gray_u8.to(device), sobel_t1, sobel_t2, device=device)
    print("sobel2 bits shape:", sobel2.shape)

    out = torch.cat([thermo_bits.to(device), sobel2], dim=1)
    return out









def bucket_id_from_global_idx(
    idx: int,
    *,
    H: int = 32,
    W: int = 32,
    C: int = 3,
    rgb_bits_per_channel: int = 2,   # 你目前最好的設定
    thermo_levels: int = 32,
    tile: int = 4,                   # 你 patch=4x4，tile 也用 4 最直覺
    D_rgb: int = 6144,
    D_thermo: int = 32768,
    D_sobel: int = 1024,
) -> str:
    """
    concat layout:
      [0, D_rgb) -> rgb bitplane (pixel-major: ((c*H+y)*W+x)*B + b)
      [D_rgb, D_rgb+D_thermo) -> gray thermo (pixel-major: (y*W+x)*L + level)
      [D_rgb+D_thermo, end) -> gray sobel (pixel-major: y*W+x)

    returns a bucket string like:
      RGB_T(ty,tx)_C{c}_B{b}
      TH_T(ty,tx)_G{g}   (g = level group)
      SB_T(ty,tx)
    """
    assert 0 <= idx < (D_rgb + D_thermo + D_sobel)

    offset_rgb = 0
    offset_th  = D_rgb
    offset_sb  = D_rgb + D_thermo

    # --- RGB ---
    if idx < offset_th:
        j = idx - offset_rgb
        pix = j // rgb_bits_per_channel
        b = j % rgb_bits_per_channel
        x = pix % W
        y = (pix // W) % H
        c = pix // (H * W)
        tx = x // tile
        ty = y // tile
        return f"RGB_T{ty:02d}{tx:02d}_C{c}_B{b}"

    # --- THERMO ---
    if idx < offset_sb:
        j = idx - offset_th
        pix = j // thermo_levels
        level = j % thermo_levels
        x = pix % W
        y = pix // W
        tx = x // tile
        ty = y // tile
        # level-group: 32 -> 4 groups (0-7,8-15,16-23,24-31)
        g = level // (thermo_levels // 4)
        return f"TH_T{ty:02d}{tx:02d}_G{g}"

    # --- SOBEL ---
    j = idx - offset_sb
    x = j % W
    y = j // W
    tx = x // tile
    ty = y // tile
    return f"SB_T{ty:02d}{tx:02d}"





def bucket_id_from_global_idx_rgb_th_sobel2(idx: int):
    H=W=32
    C=3
    rgb_bits=2
    th_levels=32
    sb_levels=2

    D_rgb = C*H*W*rgb_bits       # 6144
    D_th  = H*W*th_levels        # 32768
    D_sb  = H*W*sb_levels        # 2048
    total_bits = D_rgb + D_th + D_sb  # 40960

    if idx < 0 or idx >= total_bits:
        raise ValueError(idx)

    # RGB
    if idx < D_rgb:
        local = idx
        pix = local // rgb_bits              # 0..(3*H*W-1)
        b = local % rgb_bits               # 0..1

        c = pix // (H * W)                   # 0..2
        pix_in_c = pix % (H * W)             # 0..1023

        x = pix_in_c % W
        y = pix_in_c // W

        tx = x // 4
        ty = y // 4
        return f"RGB_C{c}_T{tx}{ty}_b{b}"    # ✅ RGB_ 前綴 OK

    # TH
    if idx < D_rgb + D_th:
        local = idx - D_rgb
        pix = local // th_levels
        lvl = local % th_levels
        x = pix % W
        y = pix // W
        tx = x // 4
        ty = y // 4
        lh = "L" if lvl < (th_levels//2) else "H"
        return f"TH_T{tx}{ty}_{lh}_l{lvl}"

    # SB2
    local = idx - (D_rgb + D_th)
    pix = local // sb_levels
    lvl = local % sb_levels
    x = pix % W
    y = pix // W
    tx = x // 4
    ty = y // 4
    return f"SB2_T{tx}{ty}_l{lvl}"          # ✅ 建議用 SB2_，避免你 code 期待 SB2