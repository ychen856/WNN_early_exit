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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch

@dataclass
@dataclass
class ExitConfig:
    layer_idx: int
    k: int
    keep_mode: str
    thr: float
    exit_tau: float
    exit_keep_idx: torch.Tensor  # [k] long cpu
    mu: torch.Tensor             # [k] float cpu
    sigma: torch.Tensor          # [k] float cpu
    use_norm: bool = True


    def to_payload(self) -> Dict[str, Any]:
        # 注意：payload 內可以直接放 CPU tensor，torch.save 能存
        return {
            "layer_idx": int(self.layer_idx),
            "k": int(self.k),
            "keep_mode": str(self.keep_mode),
            "thr": float(self.thr),
            "exit_tau": float(self.exit_tau),
            "exit_keep_idx": self.exit_keep_idx.cpu(),
            "mu": self.mu.cpu(),
            "sigma": self.sigma.cpu(),
            "use_norm": bool(self.use_norm),
        }

    @staticmethod
    def from_payload(d: Dict[str, Any]) -> "ExitConfig":
        # 保證回來的也是 CPU tensor（之後用時再 .to(device)）
        return ExitConfig(
            layer_idx=int(d["layer_idx"]),
            k=int(d["k"]),
            keep_mode=str(d["keep_mode"]),
            thr=float(d["thr"]),
            exit_tau=float(d["exit_tau"]),
            exit_keep_idx=d["exit_keep_idx"].cpu(),
            mu=d["mu"].cpu(),
            sigma=d["sigma"].cpu(),
            use_norm=bool(d.get("use_norm", True)),
        )


def serialize_exit_cfg_list(exit_cfg_list):
    out = []
    for ec in exit_cfg_list:
        out.append({
            "layer_idx": ec.layer_idx,
            "k": ec.k,
            "keep_mode": ec.keep_mode,
            "thr": float(ec.thr),
            "exit_tau": float(ec.exit_tau),
            "exit_keep_idx": ec.exit_keep_idx.cpu(),
            "mu": ec.mu.cpu(),
            "sigma": ec.sigma.cpu(),
        })
    return out


ExitCfgLike = Union[ExitConfig, Dict[str, Any]]

def normalize_exit_cfg_list(exit_cfg_list: Optional[List[ExitCfgLike]]) -> List[ExitConfig]:
    if not exit_cfg_list:
        return []
    out: List[ExitConfig] = []
    for item in exit_cfg_list:
        if isinstance(item, ExitConfig):
            out.append(item)
        elif isinstance(item, dict):
            out.append(ExitConfig.from_payload(item))
        else:
            raise TypeError(f"Unknown exit cfg type: {type(item)}")
    return out



