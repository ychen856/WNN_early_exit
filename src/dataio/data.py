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
from src.dataio.encode import bitplane_encode_u8_gray, bitplane_encode_u8_rgb, compute_sobel_two_thresholds, dt_thermometer_encode, compute_dt_thresholds, encode_gray_thermo_plus_sobel2_batch, encode_gray_thermo_plus_sobel_batch


import random, numpy as np, torch

from src.tools.utils import debug_xbits_layout

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 若你想更硬：某些算子會報錯就關掉
    # torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _filter_binary_cifar10(x_u8: torch.Tensor, y: torch.Tensor, cls_a: int, cls_b: int):
    mask = (y == cls_a) | (y == cls_b)
    x2 = x_u8[mask]
    y2 = y[mask].clone()
    # remap
    y2[y2 == cls_a] = 0
    y2[y2 == cls_b] = 1
    return x2, y2

def filter_binary_by_classes_aligned(
    x_rgb_u8: torch.Tensor,   # [N,3,32,32] or [N,32,32,3]
    x_gray_u8: torch.Tensor,  # [N,32,32]
    y: torch.Tensor,          # [N]
    cls_a: int,
    cls_b: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Keep only samples with y in {cls_a, cls_b}.
    Return y remapped: cls_a -> 0, cls_b -> 1
    """
    assert x_rgb_u8.size(0) == x_gray_u8.size(0) == y.size(0), \
        f"Mismatch N: rgb={x_rgb_u8.size(0)} gray={x_gray_u8.size(0)} y={y.size(0)}"

    y = y.to(torch.long)
    idx = torch.nonzero((y == cls_a) | (y == cls_b), as_tuple=False).view(-1)

    x_rgb_f = x_rgb_u8[idx].contiguous()
    x_gray_f = x_gray_u8[idx].contiguous()
    y_f = y[idx].contiguous()

    # remap to {0,1}
    y_bin = torch.empty_like(y_f)
    y_bin[y_f == cls_a] = 0
    y_bin[y_f == cls_b] = 1

    return x_rgb_f, x_gray_f, y_bin

@dataclass
class DatasetMeta:
    name: str
    z: int
    val_ratio: float
    channels: int
    height: int
    width: int


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
    elif name_up in ("CIFAR10_GRAY", "CIFAR10-GRAY", "CIFAR10GRAY"):
        ds_tr = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
        ds_te = datasets.CIFAR10(root=root, train=False, download=True, transform=None)

        # ds_tr.data: [N,H,W,3] uint8
        xtr = ds_tr.data.astype("float32")
        xte = ds_te.data.astype("float32")

        gray_tr = 0.299 * xtr[..., 0] + 0.587 * xtr[..., 1] + 0.114 * xtr[..., 2]
        gray_te = 0.299 * xte[..., 0] + 0.587 * xte[..., 1] + 0.114 * xte[..., 2]

        gray_tr = gray_tr.clip(0, 255).astype("uint8")  # [N,H,W]
        gray_te = gray_te.clip(0, 255).astype("uint8")

        x_train = torch.from_numpy(gray_tr)  # uint8 [N,H,W]
        x_test  = torch.from_numpy(gray_te)  # uint8 [N,H,W]
        y_train = torch.tensor(ds_tr.targets, dtype=torch.long)
        y_test  = torch.tensor(ds_te.targets, dtype=torch.long)

        return x_train, y_train, x_test, y_test  
    elif name_up in ('CIFAR10_BINARY_GRAY', 'CIFAR10-BINARY-GRAY', 'CIFAR10BINARYGRAY'):
        ds_tr = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
        ds_te = datasets.CIFAR10(root=root, train=False, download=True, transform=None)

        # ds_tr.data: [N,H,W,3] uint8
        xtr = ds_tr.data.astype("float32")
        xte = ds_te.data.astype("float32")

        gray_tr = 0.299 * xtr[..., 0] + 0.587 * xtr[..., 1] + 0.114 * xtr[..., 2]
        gray_te = 0.299 * xte[..., 0] + 0.587 * xte[..., 1] + 0.114 * xte[..., 2]

        gray_tr = gray_tr.clip(0, 255).astype("uint8")  # [N,H,W]
        gray_te = gray_te.clip(0, 255).astype("uint8")

        x_train = torch.from_numpy(gray_tr)  # uint8 [N,H,W]
        x_test  = torch.from_numpy(gray_te)  # uint8 [N,H,W]
        y_train = torch.tensor(ds_tr.targets, dtype=torch.long)
        y_test  = torch.tensor(ds_te.targets, dtype=torch.long)

        cls_a, cls_b = 3, 5  # you can also make it function args
        x_train_u8_grey, y_train = _filter_binary_cifar10(x_train, y_train, cls_a, cls_b)
        x_test_u8_grey,  y_test  = _filter_binary_cifar10(x_test,  y_test,  cls_a, cls_b)

        print(f"[binary CIFAR10] classes=({cls_a},{cls_b}) train={len(y_train)} test={len(y_test)}")
        # meta 你先可以 minimal 化；真的要塞 dataset_meta 之後再補
        #meta = {"dataset": "CIFAR10_BINARY", "z": z, "classes": [int(cls_a), int(cls_b)], "num_classes": 2, "shape": (3,32,32)}

        return ds_tr, x_train_u8_grey, y_train, ds_te,x_test_u8_grey, y_test




        '''# to grayscale Y = 0.299R + 0.587G + 0.114B
        ds_tr.data = (0.299 * ds_tr.data[:, :, :, 0] + 0.587 * ds_tr.data[:, :, :, 1] + 0.114 * ds_tr.data[:, :, :, 2])
        ds_te.data = (0.299 * ds_te.data[:, :, :, 0] + 0.587 * ds_te.data[:, :, :, 1] + 0.114 * ds_te.data[:, :, :, 2])
        print(f'ds_tr.data shape after grayscale: {ds_tr.data.shape}, dtype: {ds_tr.data.dtype}')
        x_train = torch.from_numpy(ds_tr.data, dtype=torch.uint8)  # uint8, [N,H,W,C]
        y_train = torch.tensor(ds_tr.targets, dtype=torch.long)
        x_test  = torch.from_numpy(ds_te.data, dtype=torch.uint8)
        y_test  = torch.tensor(ds_te.targets, dtype=torch.long)
        return x_train, y_train, x_test, y_test
'''
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


def _load_torchvision_color_dataset(
    dataset: str,
    root: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DatasetMeta]:
    """
    Return:
      x_train_u8: [N, 3, H, W] uint8
      y_train:    [N] int64
      x_test_u8:  [N, 3, H, W] uint8
      y_test:     [N] int64
      meta
    """
    name = dataset.upper()
    if name == "CIFAR10":
        # Define the data augmentation pipeline for the training set
        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
        test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=None)

        # torchvision CIFAR10 stores data as numpy: [N, H, W, C] uint8
        x_train = torch.from_numpy(train_ds.data)  # uint8, [N,H,W,C]
        y_train = torch.tensor(train_ds.targets, dtype=torch.long)
        x_test  = torch.from_numpy(test_ds.data)
        y_test  = torch.tensor(test_ds.targets, dtype=torch.long)

        # to [N, C, H, W]
        x_train_u8 = x_train.permute(0, 3, 1, 2).contiguous()
        x_test_u8  = x_test.permute(0, 3, 1, 2).contiguous()

        return x_train_u8, y_train, x_test_u8, y_test

    else:
        raise ValueError(f"Unsupported color dataset: {dataset}")



def load_binary_cifar10_u8(root: str, z: int = 32, cls_a: int = 0, cls_b: int = 1):
    x_train_u8, y_train, x_test_u8, y_test = _load_torchvision_color_dataset(dataset="CIFAR10", root=root)
    x_train_u8, y_train = _filter_binary_cifar10(x_train_u8, y_train, cls_a, cls_b)
    x_test_u8,  y_test  = _filter_binary_cifar10(x_test_u8,  y_test,  cls_a, cls_b)

    print(f"[binary CIFAR10] classes=({cls_a},{cls_b}) train={len(y_train)} test={len(y_test)}")
    # meta 你先可以 minimal 化；真的要塞 dataset_meta 之後再補
    meta = {"dataset": "CIFAR10_BINARY", "z": z, "classes": [int(cls_a), int(cls_b)], "num_classes": 2, "shape": (3,32,32)}
    return x_train_u8, y_train, x_test_u8, y_test, meta


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
    print('dataset:', dataset)
    num_classes = 10  # default, will be overridden for binary CIFAR10
    if dataset == "CIFAR10":
        x_train_u8, y_train, x_test_u8, y_test = _load_torchvision_color_dataset(dataset, root)
        meta = DatasetMeta(name="CIFAR10", z=z, val_ratio=0.1, channels=3, height=32, width=32)
        num_classes = 10
    elif dataset == "CIFAR10_2":
        x_train_rgb_u8, x_val_rgb_u8, x_test_rgb_u8, x_train_u8, x_val_u8, x_test_u8, y_train, y_val, y_test, gen = load_cifar10_rgb_and_gray_u8_aligned(
            root=root)
        meta = DatasetMeta(name="CIFAR10_2", z=z, val_ratio=0.1, channels=3, height=32, width=32)
        num_classes = 10
    elif dataset == "CIFAR10_GRAY":
        x_train_u8, y_train, x_test_u8, y_test = _load_torchvision_grayscale_dataset(dataset, root)
        meta = DatasetMeta(name="CIFAR10_GRAY", z=z, val_ratio=0.1, channels=1, height=32, width=32)
        num_classes = 10
    elif dataset == "CIFAR10_BINARY":
        x_train_u8, y_train, x_test_u8, y_test, meta = load_binary_cifar10_u8(root=root, z=z, cls_a=3, cls_b=5)
        num_classes = 2
    elif dataset in ('CIFAR10_BINARY_GRAY', 'CIFAR10-BINARY-GRAY', 'CIFAR10BINARYGRAY'):
        x_train_rgb_u8, x_val_rgb_u8, x_test_rgb_u8, x_train_gray_u8, x_val_gray_u8, x_test_gray_u8, y_train, y_val, y_test, gen = load_cifar10_rgb_and_gray_u8_aligned(
            root=root)
        cls_a, cls_b = 3, 5

        x_train_rgb_u8, x_train_u8, y_train = filter_binary_by_classes_aligned(x_train_rgb_u8, x_train_gray_u8, y_train, cls_a, cls_b)
        x_val_rgb_u8, x_val_u8, y_val = filter_binary_by_classes_aligned(x_val_rgb_u8, x_val_gray_u8, y_val, cls_a, cls_b)
        x_test_rgb_u8, x_test_u8, y_test = filter_binary_by_classes_aligned(x_test_rgb_u8, x_test_gray_u8, y_test, cls_a, cls_b)

       
        print(f'x train_rgb_u8 shape: {x_train_rgb_u8.shape}, x_val_rgb_u8 shape: {x_val_rgb_u8.shape}, x_test_rgb_u8 shape: {x_test_rgb_u8.shape}')
        print(f'x train_gray_u8 shape: {x_train_u8.shape}, x_val_gray_u8 shape: {x_val_u8.shape}, x_test_gray_u8 shape: {x_test_u8.shape}')

        meta = DatasetMeta(name="CIFAR10_BINARY_GRAY", z=z, val_ratio=0.1, channels=3, height=32, width=32)
        num_classes = 2
    else:
        x_train_u8, y_train, x_test_u8, y_test = _load_torchvision_grayscale_dataset(dataset, root)
        meta = DatasetMeta(
            name=dataset,
            z=z,
            val_ratio=val_ratio,
            channels=1,
            height=28,
            width=28,
        )
        num_classes = 10
    
    print(f'type: {x_train_u8.dtype}, min: {x_train_u8.min()}, max: {x_train_u8.max()}, shape: {x_train_u8.shape}')
    if dataset not in ('CIFAR10_BINARY_GRAY', 'CIFAR10-BINARY-GRAY', 'CIFAR10BINARYGRAY', 'CIFAR10_2'):
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



    if dataset in ('CIFAR10_GRAY', 'CIFAR10GRAY'):
        x_train_bits = encode_gray_thermo_plus_sobel_batch(x_train_u8.to(device_for_encoding), thresholds, xmin, xmax)
        x_val_bits   = encode_gray_thermo_plus_sobel_batch(x_val_u8.to(device_for_encoding),   thresholds, xmin, xmax)
        x_test_bits  = encode_gray_thermo_plus_sobel_batch(x_test_u8.to(device_for_encoding),  thresholds, xmin, xmax)
    elif dataset in ('CIFAR10_BINARY_GRAY', 'CIFAR10-BINARY-GRAY', 'CIFAR10BINARYGRAY'):
        x_train_bits_pre = bitplane_encode_u8_rgb(x_train_rgb_u8.to(device_for_encoding), bits=2)
        x_val_bits_pre   = bitplane_encode_u8_rgb(x_val_rgb_u8.to(device_for_encoding), bits=2)
        x_test_bits_pre  = bitplane_encode_u8_rgb(x_test_rgb_u8.to(device_for_encoding), bits=2)
        print(f'RGB bitplanes bits shape: {x_train_bits_pre.shape}')

        x_train_thermo_bits = dt_thermometer_encode(x_train_u8.to(device_for_encoding), thresholds, xmin, xmax)
        x_val_thermo_bits   = dt_thermometer_encode(x_val_u8.to(device_for_encoding),   thresholds, xmin, xmax)
        x_test_thermo_bits  = dt_thermometer_encode(x_test_u8.to(device_for_encoding),  thresholds, xmin, xmax)
        print(f'gray thermo bits shape: {x_train_thermo_bits.shape}')

        '''x_train_bits_post = encode_gray_thermo_plus_sobel_batch(x_train_u8.to(device_for_encoding), thresholds, xmin, xmax)
        x_val_bits_post   = encode_gray_thermo_plus_sobel_batch(x_val_u8.to(device_for_encoding),   thresholds, xmin, xmax)
        x_test_bits_post  = encode_gray_thermo_plus_sobel_batch(x_test_u8.to(device_for_encoding),  thresholds, xmin, xmax)'''
        sobel_t1, sobel_t2 = compute_sobel_two_thresholds(x_train_u8, q1=0.80, q2=0.92, device=device_for_encoding)
        print("sobel thresholds:", sobel_t1, sobel_t2)
        # thermo_bits: [B, 32768]  (你原本的 thermo output)
        x_train_bits_post = encode_gray_thermo_plus_sobel2_batch(x_train_u8, x_train_thermo_bits, sobel_t1, sobel_t2, device=device_for_encoding)
        x_val_bits_post   = encode_gray_thermo_plus_sobel2_batch(x_val_u8, x_val_thermo_bits, sobel_t1, sobel_t2, device=device_for_encoding)
        x_test_bits_post  = encode_gray_thermo_plus_sobel2_batch(x_test_u8, x_test_thermo_bits, sobel_t1, sobel_t2, device=device_for_encoding)

        print(f"thermo+sobel bits shape: {x_train_bits_post.shape}")
        # concat RGB bitplanes with gray thermo+sobel bits
        x_train_bits = torch.cat([x_train_bits_pre, x_train_bits_post], dim=1)
        x_val_bits   = torch.cat([x_val_bits_pre, x_val_bits_post], dim=1)
        x_test_bits  = torch.cat([x_test_bits_pre, x_test_bits_post], dim=1)
    elif dataset == 'CIFAR10_2':
        x_train_bits_pre = bitplane_encode_u8_rgb(x_train_rgb_u8.to(device_for_encoding), bits=2)
        x_val_bits_pre   = bitplane_encode_u8_rgb(x_val_rgb_u8.to(device_for_encoding), bits=2)
        x_test_bits_pre  = bitplane_encode_u8_rgb(x_test_rgb_u8.to(device_for_encoding), bits=2)
        print(f'RGB bitplanes bits shape: {x_train_bits_pre.shape}')

        x_train_thermo_bits = dt_thermometer_encode(x_train_u8.to(device_for_encoding), thresholds, xmin, xmax)
        x_val_thermo_bits   = dt_thermometer_encode(x_val_u8.to(device_for_encoding),   thresholds, xmin, xmax)
        x_test_thermo_bits  = dt_thermometer_encode(x_test_u8.to(device_for_encoding),  thresholds, xmin, xmax)
        print(f'gray thermo bits shape: {x_train_thermo_bits.shape}')

        sobel_t1, sobel_t2 = compute_sobel_two_thresholds(x_train_u8, q1=0.80, q2=0.92, device=device_for_encoding)
        print("sobel thresholds:", sobel_t1, sobel_t2)
        # thermo_bits: [B, 32768]  (你原本的 thermo output)
        x_train_bits_post = encode_gray_thermo_plus_sobel2_batch(x_train_u8, x_train_thermo_bits, sobel_t1, sobel_t2, device=device_for_encoding)
        x_val_bits_post   = encode_gray_thermo_plus_sobel2_batch(x_val_u8, x_val_thermo_bits, sobel_t1, sobel_t2, device=device_for_encoding)
        x_test_bits_post  = encode_gray_thermo_plus_sobel2_batch(x_test_u8, x_test_thermo_bits, sobel_t1, sobel_t2, device=device_for_encoding)

        print(f"thermo+sobel bits shape: {x_train_bits_post.shape}")
        # concat RGB bitplanes with gray thermo+sobel bits
        x_train_bits = torch.cat([x_train_bits_pre, x_train_bits_post], dim=1)
        x_val_bits   = torch.cat([x_val_bits_pre, x_val_bits_post], dim=1)
        x_test_bits  = torch.cat([x_test_bits_pre, x_test_bits_post], dim=1)

    else:
        x_train_bits = dt_thermometer_encode(x_train_u8.to(device_for_encoding), thresholds, xmin, xmax)
        x_val_bits   = dt_thermometer_encode(x_val_u8.to(device_for_encoding),   thresholds, xmin, xmax)
        x_test_bits  = dt_thermometer_encode(x_test_u8.to(device_for_encoding),  thresholds, xmin, xmax)



        

    '''x_train_bits = bitplane_encode_u8_rgb(x_train_u8.to(device_for_encoding))
    x_val_bits   = bitplane_encode_u8_rgb(x_val_u8.to(device_for_encoding))
    x_test_bits  = bitplane_encode_u8_rgb(x_test_u8.to(device_for_encoding))'''
    '''x_train_bits = bitplane_encode_u8_gray(x_train_u8.to(device_for_encoding))
    x_val_bits   = bitplane_encode_u8_gray(x_val_u8.to(device_for_encoding))
    x_test_bits  = bitplane_encode_u8_gray(x_test_u8.to(device_for_encoding))'''
    '''debug_xbits_layout(x_train_bits, C=3, H=32, W=32, Z_or_B=32, mode="thermo_feature_major")
    debug_xbits_layout(x_train_bits, C=3, H=32, W=32, Z_or_B=32, mode="thermo_threshold_major")
    debug_xbits_layout(x_train_bits, C=3, H=32, W=32, Z_or_B=32, mode="bitplane")'''

    in_bits = x_train_bits.size(1)
    

    train_loader = DataLoader(
        TensorDataset(x_train_bits, y_train),
        batch_size=batch_size_train,
        shuffle=shuffle_train,
        drop_last=False,
        generator=gen, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(x_val_bits, y_val),
        batch_size=batch_size_eval,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(x_test_bits, y_test),
        batch_size=batch_size_eval,
        shuffle=False,
        drop_last=False,
    )
    
    
    return train_loader, val_loader, test_loader, in_bits, num_classes, meta



def _rgb_to_gray_u8(x_u8_hwc: torch.Tensor) -> torch.Tensor:
    """
    x_u8_hwc: uint8 [N,H,W,3]  ->  gray uint8 [N,H,W]
    """
    x = x_u8_hwc.to(torch.float32)
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    return y.round().clamp(0, 255).to(torch.uint8)


def _make_train_val_indices(
    y: torch.Tensor,
    *,
    val_size: int,
    seed: int,
    stratified: bool,
    num_classes: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    returns idx_train, idx_val (both 1D LongTensor)
    """
    if val_size <= 0 or val_size >= y.numel():
        raise ValueError(f"val_size must be in (0,{y.numel()}), got {val_size}")

    g = torch.Generator().manual_seed(seed)

    if not stratified:
        perm = torch.randperm(y.numel(), generator=g)
        idx_val = perm[:val_size]
        idx_train = perm[val_size:]
        return idx_train, idx_val

    # stratified split
    idx_train_parts = []
    idx_val_parts = []

    base = val_size // num_classes
    rem = val_size % num_classes
    val_per_class = [base + (1 if c < rem else 0) for c in range(num_classes)]

    for c in range(num_classes):
        idx_c = torch.nonzero(y == c, as_tuple=False).view(-1)
        perm_c = idx_c[torch.randperm(idx_c.numel(), generator=g)]
        nv = val_per_class[c]
        idx_val_parts.append(perm_c[:nv])
        idx_train_parts.append(perm_c[nv:])

    idx_val = torch.cat(idx_val_parts, dim=0)
    idx_train = torch.cat(idx_train_parts, dim=0)

    # shuffle within split for randomness but still aligned across modalities
    idx_train = idx_train[torch.randperm(idx_train.numel(), generator=g)]
    idx_val = idx_val[torch.randperm(idx_val.numel(), generator=g)]
    return idx_train, idx_val, g


def load_cifar10_rgb_and_gray_u8_aligned(
    root: str,
    *,
    val_size: int = 5000,
    seed: int = 42,
    stratified: bool = True,
) -> Tuple[
    # RGB
    torch.Tensor, torch.Tensor, torch.Tensor,
    # Gray
    torch.Tensor, torch.Tensor, torch.Tensor,
    # labels
    torch.Tensor, torch.Tensor, torch.Tensor,
    DatasetMeta,
]:
    """
    Returns:
      x_train_rgb_u8: [Ntr,3,32,32] uint8
      x_val_rgb_u8:   [Nva,3,32,32] uint8
      x_test_rgb_u8:  [Nte,3,32,32] uint8

      x_train_gray_u8:[Ntr,32,32] uint8
      x_val_gray_u8:  [Nva,32,32] uint8
      x_test_gray_u8: [Nte,32,32] uint8

      y_train/y_val/y_test: [N] int64

    Important: train/val split indices are generated ONCE and used for both RGB+Gray.
    """
    ds_tr = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
    ds_te = datasets.CIFAR10(root=root, train=False, download=True, transform=None)

    # raw RGB uint8 in numpy [N,H,W,3]
    x_tr_hwc = torch.from_numpy(ds_tr.data)  # uint8 [50000,32,32,3]
    y_tr = torch.tensor(ds_tr.targets, dtype=torch.long)

    x_te_hwc = torch.from_numpy(ds_te.data)  # uint8 [10000,32,32,3]
    y_te = torch.tensor(ds_te.targets, dtype=torch.long)

    # aligned split indices (for train set only)
    idx_train, idx_val, g = _make_train_val_indices(
        y_tr, val_size=val_size, seed=seed, stratified=stratified, num_classes=10
    )

    # RGB -> [N,3,32,32]
    x_tr_rgb = x_tr_hwc.permute(0, 3, 1, 2).contiguous()
    x_te_rgb = x_te_hwc.permute(0, 3, 1, 2).contiguous()

    x_train_rgb_u8 = x_tr_rgb[idx_train].contiguous()
    x_val_rgb_u8 = x_tr_rgb[idx_val].contiguous()
    x_test_rgb_u8 = x_te_rgb.contiguous()

    # Gray -> [N,32,32], split with SAME idx
    x_tr_gray = _rgb_to_gray_u8(x_tr_hwc)  # [N,32,32]
    x_te_gray = _rgb_to_gray_u8(x_te_hwc)

    x_train_gray_u8 = x_tr_gray[idx_train].contiguous()
    x_val_gray_u8 = x_tr_gray[idx_val].contiguous()
    x_test_gray_u8 = x_te_gray.contiguous()

    # labels aligned
    y_train = y_tr[idx_train].contiguous()
    y_val = y_tr[idx_val].contiguous()
    y_test = y_te.contiguous()


    return (
        x_train_rgb_u8, x_val_rgb_u8, x_test_rgb_u8,
        x_train_gray_u8, x_val_gray_u8, x_test_gray_u8,
        y_train, y_val, y_test, g
    )