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

