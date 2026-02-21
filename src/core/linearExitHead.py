from typing import List, Tuple
import torch
import torch.nn as nn

from src.exit.ckpt_exit import ExitConfig

class ExitHead(nn.Module):
    def __init__(self, k: int, num_classes: int, exit_tau: float,
                 exit_keep_idx: torch.Tensor,
                 mu: torch.Tensor, sigma: torch.Tensor,
                 use_norm: bool = False):
        super().__init__()
        self.k = k
        self.exit_tau = float(exit_tau)
        self.use_norm = bool(use_norm)

        self.register_buffer("exit_keep_idx", exit_keep_idx.long())
        self.register_buffer("mu", mu.float())
        self.register_buffer("sigma", sigma.float())

        self.classifier = nn.Linear(k, num_classes, bias=False)

    def forward(self, h_full: torch.Tensor) -> torch.Tensor:
        h = h_full[:, self.exit_keep_idx]
        if self.use_norm:
            h = (h - self.mu) / self.sigma
        return self.classifier(h) / self.exit_tau

def load_exit_pack(path: str, device):
    ckpt = torch.load(path, map_location=device)

    ex_cfg = ckpt.get("exit_config", None)
    if ex_cfg is None:
        raise ValueError("Exit pack missing exit_config.")

    # 新格式：exit_heads_state
    states = ckpt.get("exit_heads_state", None)

    # 舊格式：model_state 裡面可能有 exit head 的 weight
    model_state = ckpt.get("model_state", None)

    return ex_cfg, states, model_state, ckpt.get("extra", {})



def build_exit_heads_from_cfg(exit_cfg_list: List[ExitConfig], num_classes: int, device):
    heads = []
    for cfg in exit_cfg_list:
        head = ExitHead(
            k=int(cfg.k),
            num_classes=int(num_classes),
            exit_tau=float(cfg.exit_tau),
            exit_keep_idx=cfg.exit_keep_idx,
            mu=cfg.mu,
            sigma=cfg.sigma,
            use_norm=bool(cfg.use_norm),
        ).to(device)
        heads.append(head)
    return heads

def build_exits_from_ckpt(path: str, device, num_classes: int) -> Tuple[List[torch.nn.Module], List[ExitConfig]]:
    """
    Reads ckpt saved by save_ckpt_v2 and returns exits only.
    Return: exit_heads(list), exit_cfg_list(list[ExitConfig])
    """
    ckpt = torch.load(path, map_location=device)

    payload = ckpt.get("exit_cfg", None)
    exits_sd = ckpt.get("exits_state_dict", None)
    if payload is None or exits_sd is None:
        raise ValueError("ckpt missing exit_cfg or exits_state_dict")

    exit_cfg_list = [ExitConfig.from_payload(d) for d in payload]
    exit_heads = build_exit_heads_from_cfg(exit_cfg_list, num_classes=num_classes, device=device)

    assert len(exit_heads) == len(exits_sd), "exits_state_dict length mismatch"
    for h, sd in zip(exit_heads, exits_sd):
        h.load_state_dict(sd, strict=True)

    return exit_heads, exit_cfg_list