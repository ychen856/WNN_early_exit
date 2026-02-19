import math
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch


@dataclass
class LutCoverageReport:
    k: int
    num_luts_total: int
    num_luts_sampled: int
    num_samples: int
    patterns_per_lut: int

    # aggregated (mean over sampled LUTs)
    coverage_mean: float
    coverage_p50: float
    coverage_p10: float
    coverage_p90: float

    entropy_mean: float          # normalized entropy in [0, 1]
    maxbin_ratio_mean: float     # mean(max_count / total_count_per_lut)
    gini_mean: float             # mean gini in [0, 1]

    # raw per-lut tensors (on CPU) for debugging / plotting
    coverage_per_lut: torch.Tensor
    entropy_per_lut: torch.Tensor
    maxbin_ratio_per_lut: torch.Tensor
    gini_per_lut: torch.Tensor


@torch.no_grad()
def lut_pattern_coverage(
    layer,
    loader,
    device: torch.device,
    *,
    num_luts_sample: int = 256,
    max_batches: Optional[int] = None,
    binarize_thr: float = 0.5,
) -> LutCoverageReport:
    """
    layer must have:
      - layer.conn_idx: [num_luts, k]  (LongTensor indices)
      - layer.num_luts
      - layer.lut_input_size (k)

    It computes, for a subset of LUTs, the distribution of idx in [0, 2^k-1].
    """

    assert hasattr(layer, "conn_idx")
    assert hasattr(layer, "num_luts")
    assert hasattr(layer, "lut_input_size")

    k = int(layer.lut_input_size)
    num_luts_total = int(layer.num_luts)
    P = 1 << k

    # sample LUTs
    M = min(int(num_luts_sample), num_luts_total)
    perm = torch.randperm(num_luts_total, device=device)
    lut_ids = perm[:M]

    # conn_idx for sampled LUTs: [M, k]
    conn = layer.conn_idx.to(device)[lut_ids]  # ensure on device
    conn_flat = conn.reshape(-1)               # [M*k]

    # counts: [M, P]
    counts = torch.zeros((M, P), dtype=torch.long, device=device)

    total_seen = 0
    for bi, (xb, yb) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        xb = xb.to(device)
        B = xb.size(0)

        # binarize input bits (match your forward)
        xb_bin = (xb > binarize_thr).to(torch.uint8)  # [B, D] or [B, ...]
        if xb_bin.dim() > 2:
            xb_bin = xb_bin.view(B, -1)

        # extract bits: [B, M, k]
        bits = xb_bin[:, conn_flat].view(B, M, k)  # uint8 0/1

        # compute idx: [B, M] in [0, P-1]
        # vectorized: idx = sum(bits[..., j] << (k-1-j))
        # (same as your loop idx=idx*2+bit)
        shifts = torch.arange(k - 1, -1, -1, device=device, dtype=torch.long)  # [k]
        idx = (bits.to(torch.long) << shifts.view(1, 1, k)).sum(dim=-1)        # [B, M]

        # update histogram per LUT using bincount per column
        # (loop over M is OK; M=256, B~256/512)
        for m in range(M):
            counts[m] += torch.bincount(idx[:, m], minlength=P)

        total_seen += B

    # ---- metrics per LUT ----
    counts_f = counts.to(torch.float32)
    tot = counts_f.sum(dim=1).clamp_min(1.0)  # [M]

    # coverage = nonzero bins / P
    nonzero = (counts > 0).sum(dim=1).to(torch.float32)  # [M]
    coverage = nonzero / float(P)

    # normalized entropy in [0,1]
    p = counts_f / tot.unsqueeze(1)  # [M,P]
    eps = 1e-12
    ent = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=1)  # [M]
    ent_norm = ent / math.log(P) if P > 1 else torch.zeros_like(ent)

    # max-bin ratio
    maxbin_ratio = (counts_f.max(dim=1).values / tot)

    # gini (distribution inequality): gini = 1 - sum(p^2) * P? (normalized variant)
    # We'll use: gini = 1 - sum(p^2)   (0=uniform-ish, 1=peaked)
    gini = 1.0 - (p * p).sum(dim=1)

    # ---- aggregate ----
    def q(t: torch.Tensor, qq: float) -> float:
        return float(torch.quantile(t.detach().cpu(), qq).item())

    report = LutCoverageReport(
        k=k,
        num_luts_total=num_luts_total,
        num_luts_sampled=M,
        num_samples=total_seen,
        patterns_per_lut=P,

        coverage_mean=float(coverage.mean().item()),
        coverage_p50=q(coverage, 0.50),
        coverage_p10=q(coverage, 0.10),
        coverage_p90=q(coverage, 0.90),

        entropy_mean=float(ent_norm.mean().item()),
        maxbin_ratio_mean=float(maxbin_ratio.mean().item()),
        gini_mean=float(gini.mean().item()),

        coverage_per_lut=coverage.detach().cpu(),
        entropy_per_lut=ent_norm.detach().cpu(),
        maxbin_ratio_per_lut=maxbin_ratio.detach().cpu(),
        gini_per_lut=gini.detach().cpu(),
    )
    return report
