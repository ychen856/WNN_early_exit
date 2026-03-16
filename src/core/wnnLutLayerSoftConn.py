import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Tuple, Optional

from src.core.wisard import LearnableConnSlots, bitprobs_to_addr_probs

class WNNLUTLayerSoftConn(nn.Module):
    """
    Soft-Connection WNN LUT Layer with Learnable Bit Selection and DWB Support.
    
    Replaces fixed bit-to-LUT mappings with learnable connections that adapt during training.
    This enables the second WNN layer to discover which input bits are most relevant for each LUT.
    
    Architecture:
    ==============
    Training Mode (soft expectation):
    1. Candidate bit selection: M candidate bits per LUT from in_bits
    2. Slot weights: Learn via Gumbel-softmax over candidates
    3. Soft address probabilities: Computed from bit probabilities
    4. Expected LUT output: Sum of (probability × LUT value) over all addresses
    5. Output: Real-valued with optional DWB encoded values
    
    Evaluation Mode (hard lookup):
    1. Deterministic connection selection (no Gumbel-softmax)
    2. Fast hard address computation
    3. Single LUT lookup per LUT (efficient inference)
    
    DWB Integration (Differentiable Weightless Block):
    ===================================================
    When encoded_values_out_dim > 0:
        binary_out = (lut_output > 0).float()  # [B, num_luts]
        output = (binary_out × encoded_values).sum(dim=1)  # [B, encoded_dim]
    
    This enables:
    - Dimensionality reduction: num_luts outputs → encoded_dim features
    - Learnable encoding: Encoded values optimized via backprop
    - Smooth optimization: No quantization in the summation
    - Hardware efficiency: Addition-only operations
    
    QuWeiT Integration:
    ===================
    This layer replaces the "MLP expansion phase" in transformer MLPs:
    - Input: [B, prev_dim] from previous layer
    - Purpose: Expand representation space, learn non-linear features
    - DWB: Projects back to [B, encoded_dim] for next layer or classifier
    - Phase schedule: Temperature transitions for connection learning
    
    Parameters:
    -----------
    in_bits: int
        Number of input dimension (e.g., 8 from layer0's DWB)
    num_luts: int
        Number of lookup tables (e.g., 1000)
    k: int
        Number of bits per LUT (k << M, typically 5 for layer 1)
    M: int
        Number of candidate bits per LUT (M >= k, typically 64-128)
    use_gumbel: bool
        Enable Gumbel-softmax for differentiable sampling
    gumbel_tau: float
        Temperature (higher=softer/uniform, lower=harder/deterministic)
    encoded_values_out_dim: Optional[int]
        **DWB Feature**: Output dimension when enabled
        - None: Standard mode [B, num_luts]
        - int: DWB mode [B, encoded_values_out_dim]
    
    Example:
    --------
    # Standard soft-connection layer
    layer1 = WNNLUTLayerSoftConn(in_bits=2048, num_luts=1000, k=5, M=128)
    
    # In training: soft expectation over candidate bits
    x = torch.randn(32, 2048)
    output_train = layer1(x)  # [32, 1000]
    
    # DWB-enabled for QuWeiT (feature expansion then reduction)
    layer1_dwb = WNNLUTLayerSoftConn(
        in_bits=8,  # From layer0 DWB output
        num_luts=1000,
        k=5,
        M=128,
        encoded_values_out_dim=8  # Project back to 8-D
    )
    output_dwb = layer1_dwb(x_prev)  # [32, 8]
    """
    def __init__(
        self,
        in_bits: int,
        num_luts: int,
        k: int,
        M: int = 64,
        init_std: float = 0.01,
        dropout_p: float = 0.1,
        seed: int = 42,
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
        # how to binarize input x_bits for eval (and for extracting candidate bits)
        # assume x_bits is real-valued (layer0 outputs), so we binarize with sign
        binarize_mode: str = "sign",  # "sign" or "threshold"
        encoded_values_out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_bits = in_bits
        self.num_luts = num_luts
        self.k = k
        self.M = M
        self.out_scale = nn.Parameter(torch.tensor(4.0))
        self.dropout = nn.Dropout(p=dropout_p)
        self.binarize_mode = binarize_mode

        if encoded_values_out_dim is not None:
            encoded_values = torch.randn(num_luts, encoded_values_out_dim) * init_std
            self.encoded_values = nn.Parameter(encoded_values)
        else:
            self.encoded_values = None

        self.learnable_conn = LearnableConnSlots(
            num_luts=num_luts, in_bits=in_bits, k=k, M=M, seed=seed,
            use_gumbel=use_gumbel, gumbel_tau=gumbel_tau,
        )

        table = torch.zeros(num_luts, 2 ** k).normal_(0.0, init_std)
        self.table = nn.Parameter(table)

    def _binarize(self, x_bits: torch.Tensor) -> torch.Tensor:
        # x_bits: [B, D] float
        if self.binarize_mode == "sign":
            return (x_bits > 0).to(torch.float32)   # 0/1 float
        elif self.binarize_mode == "threshold":
            thr = x_bits.median(dim=0, keepdim=True).values
            return (x_bits > thr).to(torch.float32)
        else:
            raise ValueError(self.binarize_mode)

    @torch.no_grad()
    def _hard_lookup(self, x_bits: torch.Tensor, conn_idx: torch.Tensor) -> torch.Tensor:
        """
        x_bits: [B, D] float
        conn_idx: [L, k] long, bit indices into D
        returns out: [B, L] float
        """
        B = x_bits.size(0)
        device = x_bits.device

        x01 = self._binarize(x_bits)  # [B,D] 0/1 float
        bits = x01[:, conn_idx.reshape(-1)].view(B, self.num_luts, self.k)  # [B,L,k]
        bits = (bits > 0.5).to(torch.long)

        idx = torch.zeros(B, self.num_luts, dtype=torch.long, device=device)
        for j in range(self.k):
            idx = idx * 2 + bits[:, :, j]

        out = torch.gather(self.table.unsqueeze(0).expand(B, -1, -1), 2, idx.unsqueeze(-1)).squeeze(-1)
        if self.encoded_values is not None:
            binary_out = (out > 0).float()
            return (binary_out.unsqueeze(-1) * self.encoded_values).sum(dim=1)
        else:
            return out

    def forward(self, x_bits: torch.Tensor) -> torch.Tensor:
        """
        x_bits: [B, in_bits] float (layer0 output)
        returns: [B, num_luts] float
        """
        '''if self.training:
            print("[softconn] TRAIN branch")
        else:
            print("[softconn] EVAL branch")'''


        B, D = x_bits.shape
        assert D == self.in_bits, (D, self.in_bits)

        if not self.training:
            conn_idx = self.learnable_conn()  # [L,k] long
            out = self._hard_lookup(x_bits, conn_idx)
            out = self.dropout(out)
            return out

        # -------- train: soft LUT --------
        # 1) binarize x_bits into 0/1 float for candidate extraction (still allows grad to table/conn)
        # NOTE: binarization itself is non-diff; but gradients we care about go through weights/table,
        # and x_bits comes from previous layer anyway.
        x01 = self._binarize(x_bits)  # [B,D] 0/1 float

        # 2) gather candidate bits for each LUT: cand_idx [L,M] -> [B,L,M]
        cand_idx = self.learnable_conn.cand_idx.to(x_bits.device)   # [L,M]
        cand_bits = x01[:, cand_idx.reshape(-1)].view(B, self.num_luts, self.M)  # [B,L,M]

        # 3) slot weights w: [L,k,M] softmax over M
        w = self.learnable_conn()  # training => [L,k,M]
        assert w.requires_grad, "w 不該是 no-grad tensor，表示 learnable_conn 的 logits 沒被用到"
        #print("w stats:", w.mean().item(), w.max().item(), w.min().item())
        w = w.to(x_bits.device)

        # 4) compute bit probabilities per slot: p_bits [B,L,k]
        # cand_bits: [B,L,M], w: [L,k,M]
        # -> expand to [B,L,k,M] then sum over M
        p_bits = (cand_bits.unsqueeze(2) * w.unsqueeze(0)).sum(dim=-1)  # [B,L,k]
        p_bits = p_bits.clamp(1e-6, 1-1e-6)

        # 5) address probs: [B,L,2^k]
        P = bitprobs_to_addr_probs(p_bits)  # [B,L,2^k]

        # 6) expected LUT output: out = sum_a P[a]*table[a]
        out = (P * self.table.unsqueeze(0)).sum(dim=-1)  # [B,L]
        out = out * self.out_scale 
        out = self.dropout(out)
        
        if self.encoded_values is not None:
            binary_out = (out > 0).float()
            return (binary_out.unsqueeze(-1) * self.encoded_values).sum(dim=1)
        else:
            return out