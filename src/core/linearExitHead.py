import torch
import torch.nn as nn

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


