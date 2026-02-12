# src/data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets
import torchvision.transforms as T

# TODO: 改成你專案裡這兩個函式的實際 import
# from src.encode import compute_dt_thresholds, dt_thermometer_encode
from src.dataio.encode import dt_thermometer_encode, compute_dt_thresholds

@dataclass
class DatasetMeta:
    name: str
    z: int
    val_ratio: float


def _load_torchvision_grayscale_dataset(name: str, root: str):
    """
    Returns:
      x_train: uint8 tensor [N, 28, 28]
      y_train: long tensor [N]
      x_test:  uint8 tensor [N, 28, 28]
      y_test:  long tensor [N]
    """
    name_up = name.upper()
    if name_up == "MNIST":
        ds_tr = datasets.MNIST(root=root, train=True, download=True, transform=None)
        ds_te = datasets.MNIST(root=root, train=False, download=True, transform=None)
    elif name_up in ("FMNIST", "FASHIONMNIST", "FASHION-MNIST"):
        ds_tr = datasets.FashionMNIST(root=root, train=True, download=True, transform=None)
        ds_te = datasets.FashionMNIST(root=root, train=False, download=True, transform=None)
    elif name_up in ("KMNIST", "KUZUSHIJIMNIST", "KUZUSHIJI-MNIST"):
        ds_tr = datasets.KMNIST(root=root, train=True, download=True, transform=None)
        ds_te = datasets.KMNIST(root=root, train=False, download=True, transform=None)
    else:
        raise ValueError(f"Unsupported dataset: {name}. Supported: MNIST/FMNSIT/KMNIST")

    print(f'train set type: {type(ds_tr)}, test set type: {type(ds_te)}')
    print(f'train set xmin: {ds_tr.data.min()}, xmax: {ds_tr.data.max()}')
    print(f'test set xmin: {ds_te.data.min()}, xmax: {ds_te.data.max()}')
    # torchvision datasets store data/targets
    x_train = ds_tr.data  # uint8 [N,28,28]
    y_train = ds_tr.targets.long()
    x_test = ds_te.data
    y_test = ds_te.targets.long()
    return x_train, y_train, x_test, y_test


def build_loaders_bits(
    dataset: str,
    root: str,
    batch_size_train: int = 256,
    batch_size_eval: int = 512,
    val_ratio: float = 0.1,
    seed: int = 42,
    z: int = 32,
    device_for_encoding: Optional[torch.device] = None,
    shuffle_train: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, DatasetMeta]:
    """
    Build dataloaders of *bit-encoded* inputs.

    Returns:
      train_loader, val_loader, test_loader,
      in_bits, num_classes,
      meta (thresholds/xmin/xmax/z)
    """
    x_train_u8, y_train, x_test_u8, y_test = _load_torchvision_grayscale_dataset(dataset, root)

    total_size = len(x_train_u8)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size

    gen = torch.Generator().manual_seed(seed)
    full_ds = TensorDataset(x_train_u8, y_train)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=gen)

    '''full_x, full_y = full_ds.tensors
    x_train_u8 = full_x[train_ds.indices]
    y_train = full_y[train_ds.indices]
    x_val_u8   = full_x[val_ds.indices]
    y_val   = full_y[val_ds.indices]'''


    x_train_u8 = full_ds.tensors[0][train_ds.indices]
    y_train = full_ds.tensors[1][train_ds.indices]
    x_val_u8 = full_ds.tensors[0][val_ds.indices]
    y_val = full_ds.tensors[1][val_ds.indices]

    # ---------- DT thresholds computed on TRAIN (float) ----------
    # Keep on CPU for stats then encode on device_for_encoding if provided.
    # Your compute_dt_thresholds expects x_train like your previous code.
    thresholds, xmin, xmax = compute_dt_thresholds(x_train_u8, z=z)

    # ---------- Encode to bits ----------
    # IMPORTANT: dt_thermometer_encode in your code takes tensors on device.
    if device_for_encoding is None:
        device_for_encoding = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train_bits = dt_thermometer_encode(x_train_u8.to(device_for_encoding), thresholds, xmin, xmax)
    x_val_bits   = dt_thermometer_encode(x_val_u8.to(device_for_encoding),   thresholds, xmin, xmax)
    x_test_bits  = dt_thermometer_encode(x_test_u8.to(device_for_encoding),  thresholds, xmin, xmax)
    

    in_bits = x_train_bits.size(1)
    num_classes = 10

    train_loader = DataLoader(
        TensorDataset(x_train_bits, y_train),
        batch_size=batch_size_train,
        shuffle=shuffle_train,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_bits, y_val),
        batch_size=batch_size_eval,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(x_test_bits, y_test),
        batch_size=batch_size_eval,
        shuffle=False,
        drop_last=False,
    )

    meta = DatasetMeta(
        name=dataset,
        z=z,
        val_ratio=val_ratio,
    )
    return train_loader, val_loader, test_loader, in_bits, num_classes, meta