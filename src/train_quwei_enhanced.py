# Quasi-Weightless Transformers (QuWeiT) Training - Enhanced Version
# Combines QuWeiT methodology with robust training infrastructure
# Paper: "Shrinking the Giant: Quasi-Weightless Transformers for Low Energy Inference"
# arXiv:2411.01818v1

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataio.data import build_loaders_bits, set_seed
from src.core.multiLayerWNN import MultiLayerWNN, save_ckpt
from src.core.IViTTBackbone import IViTTBackbone
from test.eval import eval_epoch
from torchvision import datasets, transforms

try:
    import timm
    from timm.data import Mixup, create_transform
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
    from timm.models.vision_transformer import VisionTransformer
    from timm.scheduler import create_scheduler_v2
    from timm.optim import create_optimizer_v2
except ImportError as exc:
    raise ImportError(
        "This script requires timm. Install with: pip install timm"
    ) from exc

@dataclass
class TrainConfig:
    dataset: str = "cifar10"  # cifar10 | cifar100
    data_dir: str = "./data"
    output_dir: str = "./outputs/ivitt_backbone"
    image_size: int = 224
    num_classes: int = 10
    epochs: int = 300
    batch_size: int = 128
    num_workers: int = 8
    pin_memory: bool = True

    # Model: chosen to match the paper's stated D=192 and hidden dim=768 (4xD)
    patch_size: int = 16
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    # Optimization: DeiT-style defaults / common ViT recipe.
    optimizer: str = "adamw"
    lr: float = 5e-4
    weight_decay: float = 5e-2
    warmup_epochs: int = 1
    min_lr: float = 1e-5
    clip_grad: float = 1.0
    smoothing: float = 0.1

    # Augmentation: DeiT-like recipe.
    color_jitter: float = 0.3
    aa: str = "rand-m9-mstd0.5-inc1"
    train_interpolation: str = "bicubic"
    reprob: float = 0.25
    remode: str = "pixel"
    recount: int = 1

    # Mixup/Cutmix
    mixup: float = 0.8
    cutmix: float = 1.0
    cutmix_minmax: Tuple[float, float] = None
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"

    amp: bool = True
    seed: int = 42
    device: str = "cuda"
    resume: str = ""
    save_freq: int = 20
    eval_only: bool = False


class AverageMeter:
    """Simple metric tracker."""
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


def build_optimizer_quwei(
    model: nn.Module,
    cfg: TrainConfig,
) -> torch.optim.Optimizer:
    """
    Build QuWeiT-specific optimizer with separate learning rates.
    
    Parameter groups:
    1. Layer 0 LUT table: standard learning rate
    2. Layer 0 encoded values: lower learning rate for stability
    3. Layer 1 LUT table: standard learning rate
    4. Layer 1 learnable connections: higher learning rate for exploration
    5. Layer 1 encoded values: lower learning rate for stability
    6. Classifier: standard learning rate
    """
    param_groups = []
    
    # Layer 0 (hard connections)
    if hasattr(model.layers[0], 'table'):
        param_groups.append({
            "params": [model.layers[0].table],
            "lr": cfg.lr_table0,
            "weight_decay": cfg.wd_table0,
            "name": "layer0_table"
        })
    
    if hasattr(model.layers[0], 'encoded_values') and model.layers[0].encoded_values is not None:
        param_groups.append({
            "params": [model.layers[0].encoded_values],
            "lr": cfg.lr_enc0,
            "weight_decay": cfg.wd_enc0,
            "name": "layer0_encoded"
        })
    
    # Layer 1 (learnable connections)
    if hasattr(model.layers[1], 'table'):
        param_groups.append({
            "params": [model.layers[1].table],
            "lr": cfg.lr_table1,
            "weight_decay": cfg.wd_table1,
            "name": "layer1_table"
        })
    
    if hasattr(model.layers[1], 'learnable_conn'):
        param_groups.append({
            "params": [model.layers[1].learnable_conn.logits],
            "lr": cfg.lr_conn1,
            "weight_decay": cfg.wd_conn1,
            "name": "layer1_connections"
        })
    
    if hasattr(model.layers[1], 'encoded_values') and model.layers[1].encoded_values is not None:
        param_groups.append({
            "params": [model.layers[1].encoded_values],
            "lr": cfg.lr_enc1,
            "weight_decay": cfg.wd_enc1,
            "name": "layer1_encoded"
        })
    
    # Classifier
    if hasattr(model, 'classifier'):
        param_groups.append({
            "params": [model.classifier.weight],
            "lr": cfg.lr_table0,
            "weight_decay": cfg.wd_table0,
            "name": "classifier"
        })
    
    if cfg.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    elif cfg.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    steps_per_epoch: int,
    cfg: TrainConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine annealing scheduler with warmup."""
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch
    min_lr_ratio = cfg.min_lr / cfg.lr
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup from min_lr to base lr
            return min_lr_ratio + (1.0 - min_lr_ratio) * float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing from base lr to min_lr
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def entropy_regularization(w: torch.Tensor, target: str = "low", lambda_ent: float = 1e-3) -> torch.Tensor:
    """Entropy regularization for learnable connections."""
    if target is None or lambda_ent == 0:
        return torch.tensor(0.0, device=w.device)
    
    eps = 1e-12
    entropy = -(w * (w.clamp_min(eps).log())).sum(dim=-1).mean()
    
    if target == "low":
        return lambda_ent * entropy
    elif target == "high":
        max_entropy = math.log(w.shape[-1])
        return lambda_ent * (max_entropy - entropy)
    else:
        return torch.tensor(0.0, device=w.device)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    model.eval()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1.item(), images.size(0))
        acc5_meter.update(acc5.item(), images.size(0))

    return {
        "loss": loss_meter.avg,
        "acc1": acc1_meter.avg,
        "acc5": acc5_meter.avg,
    }


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    scaler,
    mixup_fn,
    lr_scheduler,
    cfg: TrainConfig,
):
    model.train()
    loss_meter = AverageMeter()

    start = time.time()
    for step, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        if cfg.clip_grad is not None and cfg.clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        if lr_scheduler is not None:
            # LambdaLR.step() uses internal counter; compute current_step for lambda function
            current_step = epoch * len(train_loader) + step
            # For LambdaLR from PyTorch, we call step() and it internally uses the counter
            # We need to manually update the LR based on current_step
            lr_lambda_fn = lr_scheduler.lr_lambdas[0]  # Get the lambda function
            scale = lr_lambda_fn(current_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.lr * scale

        loss_meter.update(loss.item(), images.size(0))

        if step % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:03d}] Step [{step:04d}/{len(train_loader):04d}] "
                f"Loss {loss_meter.avg:.4f} LR {current_lr:.6e}"
            )

    elapsed = time.time() - start
    return {"loss": loss_meter.avg, "epoch_time_sec": elapsed}


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    best_acc: float,
    cfg: TrainConfig,
    is_best: bool = False,
):
    """Save checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_acc": best_acc,
        "config": asdict(cfg),
    }
    
    torch.save(ckpt, output_dir / "last.pth")
    if is_best:
        torch.save(ckpt, output_dir / "best.pth")
    
    # Periodic checkpoints
    if (epoch + 1) % cfg.save_freq == 0:
        torch.save(ckpt, output_dir / f"epoch_{epoch+1:03d}.pth")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[int, float]:
    """Load checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    
    start_epoch = ckpt.get("epoch", 0) + 1
    best_acc = ckpt.get("best_acc", 0.0)
    
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    
    return start_epoch, best_acc


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train I-ViT-T style backbone for QuWeiT baseline")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/ivitt_backbone")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-freq", type=int, default=20)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    
    cfg = TrainConfig(
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
        drop_path_rate=args.drop_path,
        mixup=args.mixup,
        cutmix=args.cutmix,
        smoothing=args.smoothing,
        image_size=args.image_size,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        save_freq=args.save_freq,
        amp=not args.no_amp,
        eval_only=args.eval_only,
    )
    return cfg

def build_model(cfg: TrainConfig) -> nn.Module:
    model = IViTTBackbone(
        num_classes=cfg.num_classes,
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
    )
    return model

class ResizeWithCIFARStats:
    """Validation preprocessing for CIFAR resized to ViT input size."""

    def __init__(self, image_size: int):
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
            ]
        )

    def __call__(self, img):
        return self.transform(img)

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
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

def save_best_checkpoint_atomic(
    path_out: str,
    model: torch.nn.Module,
    best_val_acc: float,
    epoch: int,
    optimizer=None,
    scheduler=None,
    extra: dict = None,
):
    """
    Save checkpoint to a temp file then atomically replace `path_out`.
    This avoids corrupting `path_out` if interrupted during write.
    """
    tmp_path = path_out + ".tmp"

    payload = {
        "epoch": epoch,
        "best_val_acc": float(best_val_acc),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if extra is not None:
        payload["extra"] = extra

    # write temp
    torch.save(payload, tmp_path)

    # atomic replace (best effort cross-platform)
    os.replace(tmp_path, path_out)

def save_best_fn(epoch, model, optimizer, scheduler, best_val_acc, output_dir):
    save_best_checkpoint_atomic(
        path_out=output_dir,
        model=model,
        best_val_acc=best_val_acc,
        epoch=epoch,
        optimizer=optimizer,
        scheduler=scheduler
    )


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    
    print(cfg)

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
    val_criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer_v2(
        model,
        opt=cfg.optimizer,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        num_epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        cfg=cfg,
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
        metrics = evaluate(model, val_loader, val_criterion, device)
        print(
            f"[Eval] loss={metrics['loss']:.4f} acc1={metrics['acc1']:.2f} acc5={metrics['acc5']:.2f}"
        )
        return

    print("Start training backbone...")
    print(json.dumps(asdict(cfg), indent=2))
    
    best_state = None
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(start_epoch, cfg.epochs):
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            mixup_fn=mixup_fn,
            lr_scheduler=scheduler,
            cfg=cfg,
        )

        val_stats = evaluate(model, val_loader, val_criterion, device)
        is_best = val_stats["acc1"] > best_acc
        best_acc = max(best_acc, val_stats["acc1"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"val_acc1={val_stats['acc1']:.2f} | "
            f"val_acc5={val_stats['acc5']:.2f} | "
            f"best_acc1={best_acc:.2f} | "
            f"time={train_stats['epoch_time_sec']:.1f}s"
        )


        if val_stats['acc1'] > best_val_acc:
            best_val_acc = val_stats['acc1']
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0

            save_best_fn(
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_acc,
                str(output_dir) + 'wnn_quwei_CIFAR.pth'
            )

            print(f"[BEST] epoch={epoch:03d} val_acc={best_val_acc:.2f}%")
        else:
            no_improve += 1


    print(f"Training finished. Best top-1 accuracy: {best_acc:.2f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
