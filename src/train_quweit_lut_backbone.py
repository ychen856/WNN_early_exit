import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from timm.data import Mixup, create_transform
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
    from timm.optim import create_optimizer_v2
    from timm.scheduler import create_scheduler_v2
except ImportError as exc:
    raise ImportError("This script requires timm. Install with: pip install timm") from exc


# ============================================================
# Config
# ============================================================
@dataclass
class TrainConfig:
    dataset: str = "cifar10"
    data_dir: str = "./data"
    output_dir: str = "./outputs/quweit_early_exit_ready"

    image_size: int = 224
    num_classes: int = 10
    epochs: int = 300
    batch_size: int = 128
    num_workers: int = 8
    pin_memory: bool = True

    patch_size: int = 16
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    block_type: str = "dense"          # dense | weightless
    thermometer_bins: int = 8
    weightless_hidden_dim: int = 768    # 4 * 192, matches paper vision setup
    use_fake_lut: bool = True           # training-time differentiable approximation

    use_exit: bool = False
    exit_layers: str = "3,6,9"
    exit_loss_weight: float = 0.3
    exit_threshold: float = 0.8

    optimizer: str = "adamw"
    lr: float = 5e-4
    weight_decay: float = 5e-2
    warmup_epochs: int = 1
    min_lr: float = 1e-5
    clip_grad: float = 1.0
    smoothing: float = 0.1

    color_jitter: float = 0.3
    aa: str = "rand-m9-mstd0.5-inc1"
    train_interpolation: str = "bicubic"
    reprob: float = 0.25
    remode: str = "pixel"
    recount: int = 1

    mixup: float = 0.8
    cutmix: float = 1.0
    cutmix_minmax: Optional[Tuple[float, float]] = None
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"

    amp: bool = True
    seed: int = 42
    device: str = "cuda"
    resume: str = ""
    save_freq: int = 20
    eval_only: bool = False


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# ============================================================
# Data
# ============================================================
class ResizeWithCIFARStats:
    def __init__(self, image_size: int):
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

    def __call__(self, img):
        return self.transform(img)



def build_transforms(cfg: TrainConfig):
    train_transform = create_transform(
        input_size=cfg.image_size,
        is_training=True,
        color_jitter=cfg.color_jitter,
        auto_augment=cfg.aa,
        interpolation=cfg.train_interpolation,
        re_prob=cfg.reprob,
        re_mode=cfg.remode,
        re_count=cfg.recount,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )
    val_transform = ResizeWithCIFARStats(cfg.image_size)
    return train_transform, val_transform



def build_datasets(cfg: TrainConfig):
    train_transform, val_transform = build_transforms(cfg)
    if cfg.dataset.lower() == "cifar10":
        train_set = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=train_transform)
        val_set = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=val_transform)
        cfg.num_classes = 10
    elif cfg.dataset.lower() == "cifar100":
        train_set = datasets.CIFAR100(cfg.data_dir, train=True, download=True, transform=train_transform)
        val_set = datasets.CIFAR100(cfg.data_dir, train=False, download=True, transform=val_transform)
        cfg.num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")
    return train_set, val_set



def build_loaders(cfg: TrainConfig):
    train_set, val_set = build_datasets(cfg)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, cfg


# ============================================================
# Core modules: patch embedding / transformer pieces
# ============================================================
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DenseMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================
# Weightless / LUT-friendly approximation
# ============================================================
class ThermometerEncodingSTE(nn.Module):
    """
    Differentiable thermometer-style encoding.
    Output shape: [B, N, D, K]
    """

    def __init__(self, num_bins: int = 8, value_range: Tuple[float, float] = (-3.0, 3.0)):
        super().__init__()
        self.num_bins = num_bins
        low, high = value_range
        levels = torch.linspace(low, high, steps=num_bins)
        self.register_buffer("levels", levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        levels = self.levels.view(1, 1, 1, self.num_bins)
        x_exp = x.unsqueeze(-1)
        hard = (x_exp >= levels).float()
        soft = torch.sigmoid(8.0 * (x_exp - levels))
        return hard + (soft - soft.detach())


class FakeLUTLayer(nn.Module):
    """
    Training-time differentiable LUT approximation.
    Each output unit sees all encoded inputs and produces a scalar.
    This is still trainable in PyTorch, but the structure is organized so it can
    later be replaced by a real LUT export pipeline.
    """

    def __init__(self, in_dim: int, out_dim: int, bins: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bins = bins
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim, bins) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        # x_enc: [B, N, D, K]
        # output: [B, N, out_dim]
        out = torch.einsum("bndk,odk->bno", x_enc, self.weight)
        return out + self.bias


class ConditionalSummation(nn.Module):
    """
    Placeholder for the paper's conditional summation stage.
    Current version uses a simple learned gating over channels.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate.view(1, 1, -1)


class WeightlessBlock(nn.Module):
    """
    Early-exit-ready, differentiable QuWeiT-style block.
    Input/output shape is [B, N, D].

    Current implementation is intentionally split into:
      1) thermometer encoding
      2) LUT layer 1 (D -> 4D)
      3) LUT layer 2 (4D -> D)
      4) conditional summation

    For now, layer 2 re-encodes the hidden activations to keep the code simple.
    This is a practical training-time approximation, not yet the final hardware export path.
    """

    def __init__(self, dim: int, hidden_dim: int, bins: int = 8, drop: float = 0.0):
        super().__init__()
        self.encoder1 = ThermometerEncodingSTE(num_bins=bins)
        self.lut1 = FakeLUTLayer(in_dim=dim, out_dim=hidden_dim, bins=bins)
        self.encoder2 = ThermometerEncodingSTE(num_bins=bins)
        self.lut2 = FakeLUTLayer(in_dim=hidden_dim, out_dim=dim, bins=bins)
        self.cond_sum = ConditionalSummation(dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = self.encoder1(x)
        h = self.lut1(x_enc)
        h = torch.tanh(h)
        h_enc = self.encoder2(h)
        out = self.lut2(h_enc)
        out = self.cond_sum(out)
        out = self.dropout(out)
        return out


# ============================================================
# Transformer encoder block
# ============================================================
class EncoderBlock(nn.Module):
    def __init__(self, cfg: TrainConfig, drop_path: float):
        super().__init__()
        dim = cfg.embed_dim
        hidden_dim = int(cfg.embed_dim * cfg.mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=cfg.num_heads,
            qkv_bias=cfg.qkv_bias,
            attn_drop=cfg.attn_drop_rate,
            proj_drop=cfg.drop_rate,
        )
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        if cfg.block_type == "dense":
            self.mlp_or_weightless = DenseMLP(dim=dim, hidden_dim=hidden_dim, drop=cfg.drop_rate)
        elif cfg.block_type == "weightless":
            self.mlp_or_weightless = WeightlessBlock(
                dim=dim,
                hidden_dim=cfg.weightless_hidden_dim,
                bins=cfg.thermometer_bins,
                drop=cfg.drop_rate,
            )
        else:
            raise ValueError(f"Unsupported block_type: {cfg.block_type}")
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp_or_weightless(self.norm2(x)))
        return x


# ============================================================
# Exit heads and backbone
# ============================================================
class ExitHead(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = x[:, 0]
        cls = self.norm(cls)
        return self.fc(cls)


class QuWeiTViT(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(
            img_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_chans=3,
            embed_dim=cfg.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.drop_rate)

        dpr = torch.linspace(0, cfg.drop_path_rate, steps=cfg.depth).tolist()
        self.blocks = nn.ModuleList([EncoderBlock(cfg, drop_path=dpr[i]) for i in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        if cfg.use_exit:
            exit_layers = [int(x) for x in cfg.exit_layers.split(",") if x.strip()]
            self.exit_layers = exit_layers
            self.exit_heads = nn.ModuleDict({str(i): ExitHead(cfg.embed_dim, cfg.num_classes) for i in exit_layers})
        else:
            self.exit_layers = []
            self.exit_heads = nn.ModuleDict()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor, return_intermediate: bool = False):
        x = self.patch_embed(x)
        b = x.shape[0]
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        intermediates = []
        exit_logits = {}
        for idx, block in enumerate(self.blocks):
            x = block(x)
            layer_id = idx + 1
            if return_intermediate:
                intermediates.append(x)
            if layer_id in self.exit_layers:
                exit_logits[str(layer_id)] = self.exit_heads[str(layer_id)](x)

        x = self.norm(x)
        return x, intermediates, exit_logits

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        x, intermediates, exit_logits = self.forward_features(x, return_intermediate=return_intermediate)
        logits = self.head(x[:, 0])
        if return_intermediate or self.cfg.use_exit:
            return {
                "logits": logits,
                "intermediates": intermediates,
                "exit_logits": exit_logits,
            }
        return logits

    @torch.no_grad()
    def forward_early_exit(self, x: torch.Tensor, threshold: Optional[float] = None):
        threshold = self.cfg.exit_threshold if threshold is None else threshold
        x = self.patch_embed(x)
        b = x.shape[0]
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            layer_id = idx + 1
            if layer_id in self.exit_layers:
                logits = self.exit_heads[str(layer_id)](x)
                conf = logits.softmax(dim=-1).max(dim=-1).values
                if torch.all(conf >= threshold):
                    return {
                        "logits": logits,
                        "exited": True,
                        "exit_layer": layer_id,
                        "confidence": conf,
                    }

        x = self.norm(x)
        logits = self.head(x[:, 0])
        conf = logits.softmax(dim=-1).max(dim=-1).values
        return {
            "logits": logits,
            "exited": False,
            "exit_layer": self.cfg.depth,
            "confidence": conf,
        }


# ============================================================
# Loss / mixup / checkpoint
# ============================================================
def build_mixup_fn(cfg: TrainConfig):
    active = cfg.mixup > 0 or cfg.cutmix > 0.0 or cfg.cutmix_minmax is not None
    if not active:
        return None
    return Mixup(
        mixup_alpha=cfg.mixup,
        cutmix_alpha=cfg.cutmix,
        cutmix_minmax=cfg.cutmix_minmax,
        prob=cfg.mixup_prob,
        switch_prob=cfg.mixup_switch_prob,
        mode=cfg.mixup_mode,
        label_smoothing=cfg.smoothing,
        num_classes=cfg.num_classes,
    )



def save_checkpoint(state: Dict, is_best: bool, output_dir: Path, filename: str = "last.pth"):
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / filename
    torch.save(state, ckpt_path)
    if is_best:
        torch.save(state, output_dir / "best.pth")



def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_acc = checkpoint.get("best_acc", 0.0)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return start_epoch, best_acc


# ============================================================
# Train / eval
# ============================================================
def compute_losses(outputs, targets, train_criterion, ce_criterion, cfg: TrainConfig):
    main_logits = outputs["logits"] if isinstance(outputs, dict) else outputs
    main_loss = train_criterion(main_logits, targets)

    aux_loss = 0.0
    if cfg.use_exit and isinstance(outputs, dict):
        if isinstance(targets, torch.Tensor) and targets.ndim == 1:
            for _, exit_logits in outputs["exit_logits"].items():
                aux_loss = aux_loss + ce_criterion(exit_logits, targets)
        else:
            # mixup/cutmix soft labels: auxiliary loss uses the same training criterion
            for _, exit_logits in outputs["exit_logits"].items():
                aux_loss = aux_loss + train_criterion(exit_logits, targets)

    total_loss = main_loss + cfg.exit_loss_weight * aux_loss
    return total_loss, main_loss, aux_loss



def train_one_epoch(model, loader, train_criterion, ce_criterion, optimizer, device, epoch, scaler, mixup_fn, lr_scheduler, cfg: TrainConfig):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.time()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        hard_targets = targets.clone()

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
            outputs = model(images, return_intermediate=cfg.use_exit)
            loss, _, _ = compute_losses(outputs, targets, train_criterion, ce_criterion, cfg)

        scaler.scale(loss).backward()
        if cfg.clip_grad is not None and cfg.clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=epoch * len(loader) + step)

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        acc1 = accuracy(logits, hard_targets, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1.item(), images.size(0))

        if step % 50 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:3d}] Step [{step:4d}/{len(loader):4d}] "
                f"Loss {loss_meter.avg:.4f} Acc {acc_meter.avg:.2f}% LR {current_lr:.2e}"
            )

    elapsed = time.time() - start
    return {"loss": loss_meter.avg, "acc1": acc_meter.avg, "epoch_time_sec": elapsed}


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg: TrainConfig):
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    exit_stats = {int(k): 0 for k in cfg.exit_layers.split(",") if k.strip()} if cfg.use_exit else {}
    total_samples = 0

    model.eval()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images, return_intermediate=cfg.use_exit)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        loss = criterion(logits, targets)
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1.item(), images.size(0))
        acc5_meter.update(acc5.item(), images.size(0))

        if cfg.use_exit:
            ee = model.forward_early_exit(images)
            exit_stats[int(ee["exit_layer"])] = exit_stats.get(int(ee["exit_layer"]), 0) + images.size(0)
            total_samples += images.size(0)

    metrics = {
        "loss": loss_meter.avg,
        "acc1": acc1_meter.avg,
        "acc5": acc5_meter.avg,
    }
    if cfg.use_exit and total_samples > 0:
        metrics["exit_ratio"] = {k: v / total_samples for k, v in sorted(exit_stats.items())}
    return metrics


# ============================================================
# Build / parse
# ============================================================
def build_model(cfg: TrainConfig) -> nn.Module:
    return QuWeiTViT(cfg)



def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="QuWeiT-style backbone / early-exit-ready trainer")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/quweit_early_exit_ready")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-freq", type=int, default=20)
    parser.add_argument("--block-type", type=str, default="dense", choices=["dense", "weightless"])
    parser.add_argument("--thermometer-bins", type=int, default=8)
    parser.add_argument("--weightless-hidden-dim", type=int, default=768)
    parser.add_argument("--use-exit", action="store_true")
    parser.add_argument("--exit-layers", type=str, default="3,6,9")
    parser.add_argument("--exit-loss-weight", type=float, default=0.3)
    parser.add_argument("--exit-threshold", type=float, default=0.8)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        image_size=args.image_size,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        save_freq=args.save_freq,
        block_type=args.block_type,
        thermometer_bins=args.thermometer_bins,
        weightless_hidden_dim=args.weightless_hidden_dim,
        use_exit=args.use_exit,
        exit_layers=args.exit_layers,
        exit_loss_weight=args.exit_loss_weight,
        exit_threshold=args.exit_threshold,
        amp=not args.no_amp,
        eval_only=args.eval_only,
    )


# ============================================================
# Main
# ============================================================
def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, cfg = build_loaders(cfg)
    model = build_model(cfg).to(device)

    mixup_fn = build_mixup_fn(cfg)
    if mixup_fn is not None:
        train_criterion = SoftTargetCrossEntropy()
    elif cfg.smoothing > 0.0:
        train_criterion = LabelSmoothingCrossEntropy(smoothing=cfg.smoothing)
    else:
        train_criterion = nn.CrossEntropyLoss()
    ce_criterion = nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer_v2(
        model,
        opt=cfg.optimizer,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched="cosine",
        num_epochs=cfg.epochs,
        min_lr=cfg.min_lr,
        warmup_lr=1e-6,
        warmup_epochs=cfg.warmup_epochs,
        updates_per_epoch=len(train_loader),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    best_acc = 0.0
    start_epoch = 0

    if cfg.resume:
        start_epoch, best_acc = load_checkpoint(cfg.resume, model, optimizer, scheduler, scaler)
        print(f"Resumed from {cfg.resume}, start_epoch={start_epoch}, best_acc={best_acc:.2f}")

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    if cfg.eval_only:
        metrics = evaluate(model, val_loader, val_criterion, device, cfg)
        print(metrics)
        return

    print("Start training...")
    print(json.dumps(asdict(cfg), indent=2))

    for epoch in range(start_epoch, cfg.epochs):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            train_criterion=train_criterion,
            ce_criterion=ce_criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            mixup_fn=mixup_fn,
            lr_scheduler=scheduler,
            cfg=cfg,
        )

        val_stats = evaluate(model, val_loader, val_criterion, device, cfg)
        is_best = val_stats["acc1"] > best_acc
        best_acc = max(best_acc, val_stats["acc1"])

        print(
            f"Epoch {epoch:03d} | train_loss={train_stats['loss']:.4f} | "
            f"train_acc1={train_stats['acc1']:.2f} | val_loss={val_stats['loss']:.4f} | "
            f"val_acc1={val_stats['acc1']:.2f} | val_acc5={val_stats['acc5']:.2f} | "
            f"best_acc1={best_acc:.2f} | time={train_stats['epoch_time_sec']:.1f}s"
        )
        if "exit_ratio" in val_stats:
            print(f"Exit ratio: {val_stats['exit_ratio']}")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_acc": best_acc,
            "config": asdict(cfg),
        }
        save_checkpoint(state, is_best, output_dir, filename="last.pth")
        if (epoch + 1) % cfg.save_freq == 0:
            save_checkpoint(state, False, output_dir, filename=f"epoch_{epoch + 1:03d}.pth")

    print(f"Training finished. Best top-1 accuracy: {best_acc:.2f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
