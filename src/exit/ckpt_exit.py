import torch

def save_exit_head(path, exit_head, cfg: dict):
    payload = {
        "type": "exit_head",
        "cfg": cfg,
        "state_dict": exit_head.state_dict(),
    }
    torch.save(payload, path)

def load_exit_head(path, device="cpu"):
    payload = torch.load(path, map_location=device)
    return payload

from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class ExitConfig:
    layer_idx: int            # exit 掛在哪一層 (h_list index)
    k: int                    # keep 維度
    keep_mode: str            # "p*(1-p)*std" / "bias" / "bias*std"
    thr: float                # 用於 early-exit decision 的 margin thr (可先存預設)
    exit_tau: float           # head logits temperature
    keep_idx: torch.Tensor    # [k] long

    # normalization stats (for comparability)
    mu: torch.Tensor          # [k] float
    sigma: torch.Tensor       # [k] float

def serialize_exit_cfg_list(exit_cfg_list):
    out = []
    for ec in exit_cfg_list:
        out.append({
            "layer_idx": ec.layer_idx,
            "k": ec.k,
            "keep_mode": ec.keep_mode,
            "thr": float(ec.thr),
            "exit_tau": float(ec.exit_tau),
            "keep_idx": ec.keep_idx.cpu(),
            "mu": ec.mu.cpu(),
            "sigma": ec.sigma.cpu(),
        })
    return out