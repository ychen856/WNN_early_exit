import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
    warmup_epochs: int = 5
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
    cutmix_minmax: Tuple[float, float] | None = None
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"

    amp: bool = True
    seed: int = 42
    device: str = "cuda"
    resume: str = ""
    save_freq: int = 20
    eval_only: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class IViTTBackbone(VisionTransformer):
    """
    A practical backbone that matches the paper's reported MLP dimensions:
    embed_dim = 192, hidden dim = 768 (= 4 * 192).

    This is effectively a ViT-Tiny/DeiT-Tiny style backbone and is a good
    baseline to pretrain before replacing MLP blocks with the weightless block.
    """

    def __init__(self, num_classes: int, image_size: int = 224, patch_size: int = 16):
        super().__init__(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
            representation_size=None,
            distilled=False,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
        )


def build_model(cfg: TrainConfig) -> nn.Module:
    model = IViTTBackbone(
        num_classes=cfg.num_classes,
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
    )
    return model


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


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


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
    loader,
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
    for step, (images, targets) in enumerate(loader):
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
            lr_scheduler.step_update(num_updates=epoch * len(loader) + step)

        loss_meter.update(loss.item(), images.size(0))

        if step % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:03d}] Step [{step:04d}/{len(loader):04d}] "
                f"Loss {loss_meter.avg:.4f} LR {current_lr:.6e}"
            )

    elapsed = time.time() - start
    return {"loss": loss_meter.avg, "epoch_time_sec": elapsed}


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
    val_criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer_v2(
        model,
        opt=cfg.optimizer,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    updates_per_epoch = len(train_loader)
    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched="cosine",
        num_epochs=cfg.epochs,
        min_lr=cfg.min_lr,
        warmup_lr=1e-6,
        warmup_epochs=cfg.warmup_epochs,
        updates_per_epoch=updates_per_epoch,
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

    for epoch in range(start_epoch, cfg.epochs):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
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

        save_checkpoint(
            state={
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_acc": best_acc,
                "config": asdict(cfg),
            },
            is_best=is_best,
            output_dir=output_dir,
            filename="last.pth",
        )

        if (epoch + 1) % cfg.save_freq == 0:
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "best_acc": best_acc,
                    "config": asdict(cfg),
                },
                is_best=False,
                output_dir=output_dir,
                filename=f"epoch_{epoch+1:03d}.pth",
            )

    print(f"Training finished. Best top-1 accuracy: {best_acc:.2f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
