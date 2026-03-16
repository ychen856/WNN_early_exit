# IViTTBackbone: Vision Transformer (ViT) for CIFAR-10
# Uses timm library for ViT implementation
# Designed for compatibility with train_backbone_trans_cifar_temp.py

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("timm library is required. Install with: pip install timm")

class IViTTBackbone(nn.Module):
    """
    Vision Transformer backbone for CIFAR-10/other datasets.
    Default: I-ViT-T (embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0)
    """
    def __init__(self, num_classes=10, img_size=224, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, drop_path_rate=0.1, patch_size: int = 16):
        super().__init__()
        self.vit = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        return self.vit(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

# Example usage:
# model = IViTTBackbone(num_classes=10)
# output = model(torch.randn(8, 3, 224, 224))
