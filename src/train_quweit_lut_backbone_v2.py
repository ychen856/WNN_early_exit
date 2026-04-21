from __future__ import annotations

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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.dataio.data import set_seed

try:
    from timm.data import Mixup
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
    from timm.optim import create_optimizer_v2
except ImportError as exc:
    raise ImportError(
        "This script requires timm. Install with: pip install timm"
    ) from exc


@dataclass
class TrainConfig:
    dataset: str = "cifar10"  # cifar10 | cifar100
    data_dir: str = "./data"
    output_dir: str = "./outputs/quweit_backbone"
    image_size: int = 224
    num_classes: int = 10
    epochs: int = 300
    batch_size: int = 128
    num_workers: int = 0  # Default to 0 for container safety; Kubernetes + SHM volume needed for >0
    pin_memory: bool = True

    # Model: chosen to match the paper's stated D=192 and hidden dim=768 (4xD)
    patch_size: int = 16
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    # dense | weightless
    block_type: str = "dense"
    thermometer_bins: int = 8
    weightless_hidden_dim: int = 768
    replace_layers: str = "all"   # e.g. all | 10,11,12 | 6,9,12

    # Optional early-exit-ready training hooks
    use_exit: bool = False
    exit_layers: str = "3,6,9"
    exit_loss_weight: float = 0.3
    exit_threshold: float = 0.8

    # Optimization
    optimizer: str = "adamw"
    lr: float = 5e-4
    weight_decay: float = 5e-2
    warmup_epochs: int = 1
    min_lr: float = 1e-5
    clip_grad: float = 1.0
    smoothing: float = 0.1

    # Augmentation
    color_jitter: float = 0.3
    aa: str = "rand-m9-mstd0.5-inc1"
    train_interpolation: str = "bicubic"
    reprob: float = 0.25
    remode: str = "pixel"
    recount: int = 1

    # Mixup/Cutmix
    mixup: float = 0.0
    cutmix: float = 1.0
    cutmix_minmax: Optional[Tuple[float, float]] = None
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"
    model_ema: bool = False
    model_ema_decay: float = 0.9999

    amp: bool = True
    seed: int = 42
    device: str = "cuda"
    resume: str = ""
    save_freq: int = 3
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


def _count_parameters(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def _linear_flops_and_macs(linear: Optional[nn.Linear], num_tokens: int = 1) -> Tuple[float, float]:
    if linear is None:
        return 0.0, 0.0
    macs = float(num_tokens * linear.in_features * linear.out_features)
    bias_flops = float(num_tokens * linear.out_features if linear.bias is not None else 0.0)
    flops = 2.0 * macs + bias_flops
    return flops, macs


def _conv2d_flops_and_macs(conv: Optional[nn.Conv2d], out_h: int, out_w: int) -> Tuple[float, float]:
    if conv is None:
        return 0.0, 0.0
    kernel_mul = conv.kernel_size[0] * conv.kernel_size[1] * (conv.in_channels // conv.groups)
    macs = float(out_h * out_w * conv.out_channels * kernel_mul)
    bias_flops = float(out_h * out_w * conv.out_channels if conv.bias is not None else 0.0)
    flops = 2.0 * macs + bias_flops
    return flops, macs


def _layernorm_flops(num_tokens: int, dim: int) -> float:
    return float(5 * num_tokens * dim)


def _attention_flops_and_macs(attn: Attention, num_tokens: int, dim: int) -> Tuple[float, float]:
    qkv_flops, qkv_macs = _linear_flops_and_macs(attn.qkv, num_tokens=num_tokens)
    proj_flops, proj_macs = _linear_flops_and_macs(attn.proj, num_tokens=num_tokens)
    qk_macs = float(num_tokens * num_tokens * dim)
    av_macs = float(num_tokens * num_tokens * dim)
    attn_macs = qk_macs + av_macs
    attn_flops = 2.0 * attn_macs
    return qkv_flops + proj_flops + attn_flops, qkv_macs + proj_macs + attn_macs


def _dense_mlp_flops_and_macs(mlp: DenseMLP, num_tokens: int) -> Tuple[float, float]:
    fc1_flops, fc1_macs = _linear_flops_and_macs(mlp.fc1, num_tokens=num_tokens)
    fc2_flops, fc2_macs = _linear_flops_and_macs(mlp.fc2, num_tokens=num_tokens)
    act_flops = float(4 * num_tokens * mlp.fc1.out_features)
    return fc1_flops + fc2_flops + act_flops, fc1_macs + fc2_macs


def _weightless_block_flops_and_macs(block: WeightlessBlock, num_tokens: int, dim: int) -> Tuple[float, float]:
    bins = int(block.encoder1.num_bins)
    hidden_dim = int(block.proj.in_features)
    encode_flops = float(num_tokens * dim * bins * 6)
    lut_macs = float(num_tokens * hidden_dim * dim * bins)
    lut_flops = 2.0 * lut_macs + float(num_tokens * hidden_dim)
    proj_flops, proj_macs = _linear_flops_and_macs(block.proj, num_tokens=num_tokens)
    cond_sum_flops = float(num_tokens * dim)
    act_flops = float(num_tokens * hidden_dim)
    return encode_flops + lut_flops + proj_flops + cond_sum_flops + act_flops, lut_macs + proj_macs


def _encoder_block_flops_and_macs(block: EncoderBlock, num_tokens: int, dim: int) -> Tuple[float, float]:
    flops = 2.0 * _layernorm_flops(num_tokens, dim)
    macs = 0.0
    attn_flops, attn_macs = _attention_flops_and_macs(block.attn, num_tokens, dim)
    flops += attn_flops
    macs += attn_macs
    if isinstance(block.mlp_or_weightless, DenseMLP):
        mlp_flops, mlp_macs = _dense_mlp_flops_and_macs(block.mlp_or_weightless, num_tokens)
    else:
        mlp_flops, mlp_macs = _weightless_block_flops_and_macs(block.mlp_or_weightless, num_tokens, dim)
    flops += mlp_flops
    macs += mlp_macs
    return flops, macs


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


class ModelEma:
    """Lightweight EMA tracker for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if not torch.is_floating_point(ema_value):
                ema_value.copy_(model_value)
                continue
            ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
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


class ThermometerEncodingSTE(nn.Module):
    """Differentiable thermometer-style encoding. Output: [B, N, D, K]"""

    def __init__(self, num_bins: int = 8, value_range: Tuple[float, float] = (-3.0, 3.0)):
        super().__init__()
        low, high = value_range
        self.num_bins = num_bins
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
    x_enc: [B, N, D, K] -> out: [B, N, O]
    """

    def __init__(self, in_dim: int, out_dim: int, bins: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim, bins) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bndk,odk->bno", x_enc, self.weight) + self.bias


class ConditionalSummation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate.view(1, 1, -1)


class WeightlessBlock(nn.Module):
    """
    More stable QuWeiT-style approximation (v2):
      norm -> thermometer -> LUT(D->H) -> tanh -> linear(H->D) -> conditional summation

    Compared with the previous version, this removes the second thermometer encoding,
    which was a major source of information loss and runtime overhead.
    Input/output shape stays [B, N, D].
    """

    def __init__(self, dim: int, hidden_dim: int, bins: int = 8, drop: float = 0.0):
        super().__init__()
        self.encoder1 = ThermometerEncodingSTE(num_bins=bins)
        self.lut1 = FakeLUTLayer(in_dim=dim, out_dim=hidden_dim, bins=bins)
        self.act = nn.Tanh()
        self.proj = nn.Linear(hidden_dim, dim)
        self.cond_sum = ConditionalSummation(dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = self.encoder1(x)
        h = self.lut1(x_enc)
        h = self.act(h)
        out = self.proj(h)
        out = self.cond_sum(out)
        out = self.dropout(out)
        return out


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


class ExitHead(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.norm(x[:, 0])
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

        replace_layers = getattr(cfg, "replace_layers", "")
        if replace_layers.strip().lower() in ["", "all"]:
            replace_layer_set = set(range(1, cfg.depth + 1)) if cfg.block_type == "weightless" else set()
        else:
            replace_layer_set = {int(x) for x in replace_layers.split(",") if x.strip()}

        dpr = torch.linspace(0, cfg.drop_path_rate, steps=cfg.depth).tolist()
        blocks = []
        for i in range(cfg.depth):
            layer_id = i + 1
            block_cfg = copy.deepcopy(cfg)
            if layer_id in replace_layer_set:
                block_cfg.block_type = "weightless"
            else:
                block_cfg.block_type = "dense"
            blocks.append(EncoderBlock(block_cfg, drop_path=dpr[i]))
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        if cfg.use_exit:
            self.exit_layers = [int(x) for x in cfg.exit_layers.split(",") if x.strip()]
            self.exit_heads = nn.ModuleDict({str(i): ExitHead(cfg.embed_dim, cfg.num_classes) for i in self.exit_layers})
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


def get_model_profile(model: QuWeiTViT) -> Dict[str, object]:
    num_tokens = model.patch_embed.num_patches + 1
    dim = model.cfg.embed_dim
    grid_size = model.patch_embed.grid_size

    patch_flops, patch_macs = _conv2d_flops_and_macs(model.patch_embed.proj, grid_size, grid_size)
    layer_profiles = []
    backbone_flops = patch_flops
    backbone_macs = patch_macs

    for block in model.blocks:
        layer_flops, layer_macs = _encoder_block_flops_and_macs(block, num_tokens, dim)
        layer_profiles.append({"flops": layer_flops, "macs": layer_macs})
        backbone_flops += layer_flops
        backbone_macs += layer_macs

    final_norm_flops = _layernorm_flops(1, dim)
    final_head_flops, final_head_macs = _linear_flops_and_macs(model.head, num_tokens=1)
    backbone_flops += final_norm_flops + final_head_flops
    backbone_macs += final_head_macs

    exit_profiles = []
    total_exit_head_params = 0
    total_exit_full_flops = 0.0
    total_exit_full_macs = 0.0
    for layer_id in model.exit_layers:
        head = model.exit_heads[str(layer_id)]
        head_flops = _layernorm_flops(1, dim)
        fc_flops, fc_macs = _linear_flops_and_macs(head.fc, num_tokens=1)
        head_flops += fc_flops
        head_macs = fc_macs
        head_params = _count_parameters(head)
        total_exit_head_params += head_params
        total_exit_full_flops += head_flops
        total_exit_full_macs += head_macs
        exit_profiles.append(
            {
                "layer_id": int(layer_id),
                "flops": head_flops,
                "macs": head_macs,
                "params": head_params,
            }
        )

    backbone_params = _count_parameters(model) - total_exit_head_params
    param_overhead_ratio = (
        total_exit_head_params / backbone_params if backbone_params > 0 else float("nan")
    )

    return {
        "num_backbone_layers": len(model.blocks),
        "patch_embed": {"flops": patch_flops, "macs": patch_macs},
        "layers": layer_profiles,
        "final_head": {"flops": final_norm_flops + final_head_flops, "macs": final_head_macs},
        "backbone_full_flops": backbone_flops,
        "backbone_full_macs": backbone_macs,
        "backbone_params": backbone_params,
        "exit_heads": exit_profiles,
        "total_exit_head_params": total_exit_head_params,
        "param_overhead_ratio": param_overhead_ratio,
        "full_model_flops_with_all_exits": backbone_flops + total_exit_full_flops,
        "full_model_macs_with_all_exits": backbone_macs + total_exit_full_macs,
    }


def print_eval_profile(tag: str, metrics: Dict[str, float]) -> None:
    if "avg_flops_per_sample" not in metrics:
        return
    print(
        f"[{tag} Profile] "
        f"avg_flops/sample={metrics['avg_flops_per_sample']:.2f} | "
        f"avg_macs/sample={metrics['avg_macs_per_sample']:.2f} | "
        f"avg_layers/sample={metrics['avg_layers_executed_per_sample']:.4f} | "
        f"backbone_params={int(metrics['backbone_params'])} | "
        f"exit_head_params={int(metrics['total_exit_head_params'])} | "
        f"calculated_overhead={metrics['compute_overhead_ratio']:.6f}"
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    steps_per_epoch: int,
    cfg: TrainConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing scheduler with warmup."""
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch
    min_lr_ratio = cfg.min_lr / cfg.lr

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    model_ema: Optional[ModelEma],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
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
        "scaler_state": scaler.state_dict() if scaler else None,
        "model_ema_state": model_ema.state_dict() if model_ema else None,
        "best_acc": best_acc,
        "config": asdict(cfg),
    }

    torch.save(ckpt, output_dir / "last.pth")
    if is_best:
        torch.save(ckpt, output_dir / "best.pth")

    print('save freq: ', cfg.save_freq)
    print('epoch: ', epoch)
    if (epoch + 1) % cfg.save_freq == 0:
        print('save..')
        torch.save(ckpt, output_dir / f"epoch_{epoch+1:03d}.pth")


def load_checkpoint(
    path: str,
    model: nn.Module,
    model_ema: Optional[ModelEma] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    """Load checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    start_epoch = ckpt.get("epoch", 0) + 1
    best_acc = ckpt.get("best_acc", 0.0)

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    if model_ema is not None and ckpt.get("model_ema_state") is not None:
        model_ema.load_state_dict(ckpt["model_ema_state"])

    return start_epoch, best_acc


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
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    resize = transforms.Resize(
        cfg.image_size,
        interpolation=transforms.InterpolationMode.BICUBIC,
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            resize,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
            ),
        ]
    )
    eval_transform = ResizeWithCIFARStats(cfg.image_size)
    clean_train_transform = ResizeWithCIFARStats(cfg.image_size)
    return train_transform, eval_transform, clean_train_transform


def build_datasets(cfg: TrainConfig):
    train_transform, val_transform, clean_train_transform = build_transforms(cfg)

    if cfg.dataset.lower() == "cifar10":
        train_set = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=train_transform)
        val_set = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=val_transform)
        clean_train_set = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=clean_train_transform)
        cfg.num_classes = 10
    elif cfg.dataset.lower() == "cifar100":
        train_set = datasets.CIFAR100(cfg.data_dir, train=True, download=True, transform=train_transform)
        val_set = datasets.CIFAR100(cfg.data_dir, train=False, download=True, transform=val_transform)
        clean_train_set = datasets.CIFAR100(cfg.data_dir, train=True, download=True, transform=clean_train_transform)
        cfg.num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    return train_set, val_set, clean_train_set


def build_loaders(cfg: TrainConfig):
    train_set, val_set, clean_train_set = build_datasets(cfg)

    def create_loader(dataset, is_train=True, workers=0, drop_last=None):
        loader_drop_last = is_train if drop_last is None else drop_last
        try:
            return DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=is_train,
                num_workers=workers,
                pin_memory=cfg.pin_memory,
                drop_last=loader_drop_last,
            )
        except RuntimeError as e:
            if workers > 0 and ("shared memory" in str(e).lower() or "shm" in str(e).lower()):
                print(f"[WARNING] SHM error with num_workers={workers}. Retrying with num_workers=0...")
                return DataLoader(
                    dataset,
                    batch_size=cfg.batch_size,
                    shuffle=is_train,
                    num_workers=0,
                    pin_memory=cfg.pin_memory,
                    drop_last=loader_drop_last,
                )
            raise

    train_loader = create_loader(train_set, is_train=True, workers=cfg.num_workers)
    val_loader = create_loader(val_set, is_train=False, workers=cfg.num_workers)
    clean_train_loader = create_loader(
        clean_train_set,
        is_train=False,
        workers=cfg.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, clean_train_loader, cfg


def compute_losses(outputs, targets, train_criterion, ce_criterion, cfg: TrainConfig):
    main_logits = outputs["logits"] if isinstance(outputs, dict) else outputs
    main_loss = train_criterion(main_logits, targets)

    aux_loss = 0.0
    if cfg.use_exit and isinstance(outputs, dict):
        if isinstance(targets, torch.Tensor) and targets.ndim == 1:
            for _, exit_logits in outputs["exit_logits"].items():
                aux_loss = aux_loss + ce_criterion(exit_logits, targets)
        else:
            for _, exit_logits in outputs["exit_logits"].items():
                aux_loss = aux_loss + train_criterion(exit_logits, targets)

    total_loss = main_loss + cfg.exit_loss_weight * aux_loss
    return total_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg: Optional[TrainConfig] = None):
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    profile = get_model_profile(model)
    exit_profile_by_layer = {item["layer_id"]: item for item in profile["exit_heads"]}
    exit_stats = {}
    total_samples = 0
    total_flops = 0.0
    total_macs = 0.0
    total_layers = 0.0

    model.eval()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images, return_intermediate=bool(cfg and cfg.use_exit))
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        loss = criterion(logits, targets)
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1.item(), images.size(0))
        acc5_meter.update(acc5.item(), images.size(0))
        batch_size = images.size(0)

        if cfg is not None and cfg.use_exit:
            ee = model.forward_early_exit(images)
            exit_layer = int(ee["exit_layer"])
            exit_stats[exit_layer] = exit_stats.get(exit_layer, 0) + batch_size
            total_samples += batch_size

            prefix_flops = profile["patch_embed"]["flops"] + sum(
                layer["flops"] for layer in profile["layers"][:exit_layer]
            )
            prefix_macs = profile["patch_embed"]["macs"] + sum(
                layer["macs"] for layer in profile["layers"][:exit_layer]
            )
            exit_head_flops = 0.0
            exit_head_macs = 0.0
            for layer_id in model.exit_layers:
                if layer_id > exit_layer:
                    break
                exit_head_info = exit_profile_by_layer[int(layer_id)]
                exit_head_flops += exit_head_info["flops"]
                exit_head_macs += exit_head_info["macs"]

            if exit_layer == model.cfg.depth:
                sample_flops = prefix_flops + profile["final_head"]["flops"] + exit_head_flops
                sample_macs = prefix_macs + profile["final_head"]["macs"] + exit_head_macs
            else:
                sample_flops = prefix_flops + exit_head_flops
                sample_macs = prefix_macs + exit_head_macs

            total_flops += batch_size * sample_flops
            total_macs += batch_size * sample_macs
            total_layers += batch_size * float(exit_layer)
        else:
            total_samples += batch_size
            total_flops += batch_size * float(profile["backbone_full_flops"])
            total_macs += batch_size * float(profile["backbone_full_macs"])
            total_layers += batch_size * float(profile["num_backbone_layers"])

    metrics = {
        "loss": loss_meter.avg,
        "acc1": acc1_meter.avg,
        "acc5": acc5_meter.avg,
        "avg_flops_per_sample": total_flops / max(total_samples, 1),
        "avg_macs_per_sample": total_macs / max(total_samples, 1),
        "avg_layers_executed_per_sample": total_layers / max(total_samples, 1),
        "backbone_params": float(profile["backbone_params"]),
        "total_exit_head_params": float(profile["total_exit_head_params"]),
        "param_overhead_ratio": float(profile["param_overhead_ratio"]),
        "compute_overhead_ratio": (
            total_flops / max(total_samples, 1) / float(profile["backbone_full_flops"])
            if profile["backbone_full_flops"] > 0 else float("nan")
        ),
    }
    metrics["compute_saving_ratio"] = (
        1.0 - metrics["compute_overhead_ratio"]
        if metrics["compute_overhead_ratio"] == metrics["compute_overhead_ratio"] else float("nan")
    )
    if total_samples > 0:
        metrics["exit_ratio"] = {k: v / total_samples for k, v in sorted(exit_stats.items())}
    return metrics


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    scaler,
    mixup_fn,
    model_ema,
    lr_scheduler,
    cfg: TrainConfig,
):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    ce_criterion = nn.CrossEntropyLoss()
    start = time.time()
    for step, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        hard_targets = targets.clone()

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
            outputs = model(images, return_intermediate=cfg.use_exit)
            loss = compute_losses(outputs, targets, criterion, ce_criterion, cfg)

        scaler.scale(loss).backward()
        if cfg.clip_grad is not None and cfg.clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        if model_ema is not None:
            model_ema.update(model)

        if lr_scheduler is not None:
            current_step = epoch * len(train_loader) + step
            lr_lambda_fn = lr_scheduler.lr_lambdas[0]
            scale = lr_lambda_fn(current_step)
            for param_group in optimizer.param_groups:
                base_lr = param_group.get("initial_lr", cfg.lr)
                param_group["lr"] = base_lr * scale

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        acc1 = accuracy(logits, hard_targets, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1.item(), images.size(0))

        if step % 100 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:03d}] Step [{step:04d}/{len(train_loader):04d}] "
                f"Loss {loss_meter.avg:.4f} Acc {acc_meter.avg:.2f}% LR {current_lr:.6e}"
            )

    elapsed = time.time() - start
    return {"loss": loss_meter.avg, "acc1": acc_meter.avg, "epoch_time_sec": elapsed}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train QuWeiT / I-ViT-T style backbone")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/quweit_backbone")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of data loading workers (default: 0). For Kubernetes, requires SHM volume mount.")

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--model-ema-decay", type=float, default=0.9999)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-freq", type=int, default=3)

    parser.add_argument("--block-type", type=str, default="dense", choices=["dense", "weightless"])
    parser.add_argument("--thermometer-bins", type=int, default=8)
    parser.add_argument("--weightless-hidden-dim", type=int, default=768)
    parser.add_argument("--replace-layers", type=str, default="all",
                        help="Which blocks to replace with weightless blocks. Examples: all, 10,11,12, 6,9,12")

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
        drop_path_rate=args.drop_path,
        mixup=args.mixup,
        cutmix=args.cutmix,
        smoothing=args.smoothing,
        model_ema=args.model_ema,
        model_ema_decay=args.model_ema_decay,
        image_size=args.image_size,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        save_freq=args.save_freq,
        block_type=args.block_type,
        thermometer_bins=args.thermometer_bins,
        weightless_hidden_dim=args.weightless_hidden_dim,
        replace_layers=args.replace_layers,
        use_exit=args.use_exit,
        exit_layers=args.exit_layers,
        exit_loss_weight=args.exit_loss_weight,
        exit_threshold=args.exit_threshold,
        amp=not args.no_amp,
        eval_only=args.eval_only,
    )


def build_model(cfg: TrainConfig) -> nn.Module:
    return QuWeiTViT(cfg)


def save_best_checkpoint_atomic(
    path_out: str,
    model: torch.nn.Module,
    best_val_acc: float,
    epoch: int,
    model_ema: Optional[ModelEma] = None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    extra: dict = None,
):
    path_out = Path(path_out)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_out.with_suffix(path_out.suffix + ".tmp")

    payload = {
        "epoch": epoch,
        "best_val_acc": float(best_val_acc),
        "model_state": model.state_dict(),
        "model_ema_state": model_ema.state_dict() if model_ema else None,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, tmp_path)
    os.replace(str(tmp_path), str(path_out))


def save_best_fn(epoch, model, model_ema, optimizer, scheduler, scaler, best_val_acc, output_path):
    save_best_checkpoint_atomic(
        path_out=Path(output_path),
        model=model,
        best_val_acc=best_val_acc,
        epoch=epoch,
        model_ema=model_ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    in_docker = os.path.exists('/.dockerenv')
    in_k8s = 'KUBERNETES_SERVICE_HOST' in os.environ
    shm_available = os.path.exists('/dev/shm') and os.access('/dev/shm', os.W_OK)
    shm_size = "unknown"
    if os.path.exists('/dev/shm'):
        try:
            stat = os.statvfs('/dev/shm')
            shm_size = f"{stat.f_bavail * stat.f_frsize / (1024**3):.1f}GB"
        except Exception:
            pass

    print("=" * 70)
    print("[ENVIRONMENT DETECTION]")
    print(f"  Docker detected: {in_docker}")
    print(f"  Kubernetes detected: {in_k8s}")
    print(f"  /dev/shm available: {shm_available}")
    print(f"  /dev/shm size: {shm_size}")
    print(f"  num_workers: {cfg.num_workers}")
    print("=" * 70)

    if (in_docker or in_k8s) and cfg.num_workers > 0 and not shm_available:
        print("[ERROR] Cannot use num_workers > 0 without /dev/shm!")
        print()
        print("[SOLUTION] Update your Kubernetes pod spec to include:")
        print()
        print("  containers:")
        print("  - name: vol-container")
        print("    volumeMounts:")
        print("    - mountPath: /dev/shm")
        print("      name: shm")
        print()
        print("  volumes:")
        print("  - name: shm")
        print("    emptyDir:")
        print("      medium: Memory")
        print("      sizeLimit: 8Gi")
        print()
        print("[TEMPORARY FIX] Run with --num-workers 0")
        print("=" * 70)
        raise RuntimeError("SHM not available for multi-worker DataLoader. See configuration above.")

    if (in_docker or in_k8s) and cfg.num_workers > 0:
        print(f"[INFO] Container mode with {cfg.num_workers} workers")
        print(f"[INFO] /dev/shm available: {shm_size}")
        print("=" * 70)

    print(cfg)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, clean_train_loader, cfg = build_loaders(cfg)
    model = build_model(cfg).to(device)
    model_ema = ModelEma(model, decay=cfg.model_ema_decay) if cfg.model_ema else None

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
    for param_group in optimizer.param_groups:
        param_group.setdefault("initial_lr", param_group["lr"])

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
        start_epoch, best_acc = load_checkpoint(
            cfg.resume,
            model,
            model_ema=model_ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        print(f"Resumed from {cfg.resume}, start_epoch={start_epoch}, best_acc={best_acc:.2f}")

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    if cfg.eval_only:
        metrics = evaluate(model, val_loader, val_criterion, device, cfg)
        print(
            f"[Eval] loss={metrics['loss']:.4f} acc1={metrics['acc1']:.2f} acc5={metrics['acc5']:.2f}"
        )
        print_eval_profile("Eval", metrics)
        if "exit_ratio" in metrics:
            print(f"[Eval] exit_ratio={metrics['exit_ratio']}")
        clean_metrics = evaluate(model, clean_train_loader, val_criterion, device, cfg)
        print(
            f"[Train-Clean] loss={clean_metrics['loss']:.4f} "
            f"acc1={clean_metrics['acc1']:.2f} acc5={clean_metrics['acc5']:.2f}"
        )
        print_eval_profile("Train-Clean", clean_metrics)
        if model_ema is not None:
            ema_metrics = evaluate(model_ema.module, val_loader, val_criterion, device, cfg)
            print(
                f"[EMA Eval] loss={ema_metrics['loss']:.4f} "
                f"acc1={ema_metrics['acc1']:.2f} acc5={ema_metrics['acc5']:.2f}"
            )
            print_eval_profile("EMA Eval", ema_metrics)
        return

    print("Start training backbone...")
    print(json.dumps(asdict(cfg), indent=2))

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0

    print('save freq: ', cfg.save_freq)
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            mixup_fn=mixup_fn,
            model_ema=model_ema,
            lr_scheduler=scheduler,
            cfg=cfg,
        )
        t1 = time.time()
        val_stats = evaluate(model, val_loader, val_criterion, device, cfg)
        train_clean_stats = evaluate(model, clean_train_loader, val_criterion, device, cfg)
        ema_val_stats = None
        if model_ema is not None:
            ema_val_stats = evaluate(model_ema.module, val_loader, val_criterion, device, cfg)
        t2 = time.time()
        tracked_val_acc = ema_val_stats["acc1"] if ema_val_stats is not None else val_stats["acc1"]
        is_best = tracked_val_acc > best_acc
        best_acc = max(best_acc, tracked_val_acc)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"train_acc1={train_stats['acc1']:.2f} | "
            f"train_clean_loss={train_clean_stats['loss']:.4f} | "
            f"train_clean_acc1={train_clean_stats['acc1']:.2f} | "
            f"train_clean_acc5={train_clean_stats['acc5']:.2f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"val_acc1={val_stats['acc1']:.2f} | "
            f"val_acc5={val_stats['acc5']:.2f} | "
            f"best_acc1={best_acc:.2f} | "
            f"time={train_stats['epoch_time_sec']:.1f}s"
        )
        if ema_val_stats is not None:
            print(
                f"EMA Epoch {epoch:03d} | "
                f"ema_val_loss={ema_val_stats['loss']:.4f} | "
                f"ema_val_acc1={ema_val_stats['acc1']:.2f} | "
                f"ema_val_acc5={ema_val_stats['acc5']:.2f}"
            )
            print_eval_profile(f"EMA Epoch {epoch:03d}", ema_val_stats)
        print_eval_profile(f"Val Epoch {epoch:03d}", val_stats)
        print_eval_profile(f"Train-Clean Epoch {epoch:03d}", train_clean_stats)
        if "exit_ratio" in val_stats:
            print(f"Exit ratio: {val_stats['exit_ratio']}")
        print('saving checkpoint...')
        save_checkpoint(
            output_dir=output_dir,
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_acc=best_acc,
            cfg=cfg,
            is_best=is_best,
        )
        t3 = time.time()
        print(f"[TIMING] train={t1-t0:.1f}s eval={t2-t1:.1f}s save={t3-t2:.1f}s")

        if tracked_val_acc > best_val_acc:
            best_val_acc = tracked_val_acc
            best_epoch = epoch
            best_source = model_ema.module if model_ema is not None else model
            best_state = copy.deepcopy(best_source.state_dict())
            no_improve = 0
            t4 = time.time()
            save_best_fn(
                epoch,
                model,
                model_ema,
                optimizer,
                scheduler,
                scaler,
                best_val_acc,
                str(output_dir / 'wnn_quwei_CIFAR.pth')
            )
            t5 = time.time()
            print(f"[TIMING] best_save={t5-t4:.1f}s")
            print(f"[BEST] epoch={epoch:03d} val_acc={best_val_acc:.2f}%")
        else:
            no_improve += 1

    print(f"Training finished. Best top-1 accuracy: {best_acc:.2f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
