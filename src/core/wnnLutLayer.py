import torch
import torch.nn as nn
import torch.nn.functional as F

class WNNLUTLayer(nn.Module):
    """
    Unified WNN LUT layer that supports both:

    - mapping:  external bit-to-LUT wiring (list/np/tensor, shape [num_luts, lut_input_size])
    - conn_idx: explicit connection indices (tensor, shape [num_luts, lut_input_size])

    Internally, everything is stored as a registered buffer `conn_idx`
    so that pruning / export code can always rely on `layer.conn_idx`.
    """

    def __init__(
        self,
        in_bits,
        num_luts,
        lut_input_size: int = 6,
        mapping=None,
        conn_idx=None,
        init_std: float = 0.01,
        dropout_p = 0.2,
        binarize_input = True,

        use_running_thr: bool = True,
        thr_momentum: float = 0.01,   # EMA step
        thr_init: float = 0.5,        # init threshold
        thr_mode: str = "mean",       # "mean" or "median"
        thr_warmup_steps: int = 0,    # 先 0，之後需要可加 warmup
    ):
        super().__init__()

        self.in_bits = in_bits
        self.num_luts = num_luts
        self.lut_input_size = lut_input_size
        self.dropout = nn.Dropout(p=dropout_p)
        self.binarize_input = binarize_input

        # additional args for adaptive thresholding (if enabled)
        self.use_running_thr = use_running_thr
        self.thr_momentum = float(thr_momentum)
        self.thr_mode = str(thr_mode)
        self.thr_warmup_steps = int(thr_warmup_steps)

        # [1, in_bits] so broadcast works
        self.register_buffer("running_thr", torch.full((1, in_bits), float(thr_init)))
        self.register_buffer("thr_step", torch.zeros((), dtype=torch.long))

        # -----------------------------
        # 1) Decide connection indices
        # -----------------------------
        if conn_idx is not None and mapping is not None:
            if not isinstance(conn_idx, torch.Tensor):
                conn = torch.tensor(conn_idx, dtype=torch.long)
            else:
                conn = conn_idx.clone().long()
        elif conn_idx is not None:
            if not isinstance(conn_idx, torch.Tensor):
                conn = torch.tensor(conn_idx, dtype=torch.long)
            else:
                conn = conn_idx.clone().long()
        elif mapping is not None:
            if not isinstance(mapping, torch.Tensor):
                conn = torch.tensor(mapping, dtype=torch.long)
            else:
                conn = mapping.clone().long()
        else:
            conn = torch.randint(
                low=0,
                high=in_bits,
                size=(num_luts, lut_input_size),
                dtype=torch.long,
            )

        # Validate shape
        if conn.dim() != 2 or conn.shape[0] != num_luts or conn.shape[1] != lut_input_size:
            raise ValueError(
                f"conn_idx/mapping shape mismatch: got {tuple(conn.shape)}, "
                f"expected ({num_luts}, {lut_input_size})"
            )

        # let prune / export / extract_real_mapping can have `layer.conn_idx`
        self.register_buffer("conn_idx", conn)
        print("conn_idx min/max:", conn.min().item(), conn.max().item())
        print("expected max ~", in_bits-1)

        # -----------------------------
        # 2) Initialize LUT table
        # -----------------------------
        table = torch.zeros(num_luts, 2 ** lut_input_size)
        table = table.normal_(mean=0.0, std=init_std)
        self.table = nn.Parameter(table)

    def forward(self, x_bits):
        B = x_bits.size(0)
        device = x_bits.device

        # Extract k bits for each LUT
        bits = x_bits[:, self.conn_idx.view(-1)].view(B, self.num_luts, self.lut_input_size)

        # Address bits must be 0/1
        '''if self.binarize_input:
            # for LUT outputs (real, around 0)
            bits = (bits > 0.0).to(torch.long)
        else:
            # for bitplane input (0/1)
            bits = (bits > 0.5).to(torch.long)'''
        if self.binarize_input:
            # layer0：bitplane already 0/1
            bits = (bits > 0.5).long()
        else:
            # layer>=1：dynamic threshold，避免全 0 / 全 1
            thr = bits.median(dim=0, keepdim=True).values   # [1, num_luts, k]
            bits = (bits > thr).long()


        bits = x_bits[:, self.conn_idx.view(-1)].view(B, self.num_luts, self.lut_input_size)

        if self.binarize_input:
            # layer>=1: 用動態 threshold，避免全 0 / 全 1
            # 這個 thr 是針對每個 LUT-input 位置算的（k 維），很便宜也很穩
            thr = bits.median(dim=0, keepdim=True).values   # [1, num_luts, k]
            bits = (bits > thr).to(torch.long)
        else:
            # layer0: bitplane 已經是 0/1
            bits = (bits > 0.5).to(torch.long)
        
        #print(f'bits one rate: {(bits > 0).float().mean().item():.4f}, std: {bits.float().std().item():.4f}, min: {bits.min().item()}, max: {bits.max().item()}')
        
        # idx
        idx = torch.zeros(B, self.num_luts, dtype=torch.long, device=device)
        for j in range(self.lut_input_size):
            idx = idx * 2 + bits[:, :, j]

        #print("layer1 bits ones rate", bits.float().mean().item(),
        #    "unique idx", torch.unique(idx).numel())


        table_expanded = self.table.unsqueeze(0).expand(B, -1, -1)
        out = torch.gather(table_expanded, 2, idx.unsqueeze(-1)).squeeze(-1)

        out = self.dropout(out)
        #print(f'out mean: {out.mean().item():.4f}, std: {out.std().item():.4f}, min: {out.min().item()}, max: {out.max().item()}')
        return out
    

    def forward_temp(self, x_bits):
        B = x_bits.size(0)
        device = x_bits.device

        # Binarize input bits to 0/1
        '''if self.binarize_input:
            #x_bits = (x_bits > 0.5).float()
            thr = x_bits.mean(dim=0, keepdim=True)
            x_bits = (x_bits > thr).float()
        else:
            thr = x_bits.median(dim=0, keepdim=True).values
            x_bits = (x_bits > thr).float()'''
        if self.binarize_input:
            #x_bits = (x_bits > 0.5).float()
            x_bits = (x_bits > 0.0).float()
        else:
            #thr = x_bits.median(dim=0, keepdim=True).values
            #x_bits = (x_bits > thr).float()
            x_bits = (x_bits > 0.0).float()

        #print(f'x_bits mean: {x_bits.mean().item():.4f}, std: {x_bits.std().item():.4f}, min: {x_bits.min().item()}, max: {x_bits.max().item()}')

        # Extract k bits for each LUT
        # conn_idx: [num_luts, k]
        # -> [B, num_luts, k]
        bits = x_bits[:, self.conn_idx.view(-1)].view(
            B, self.num_luts, self.lut_input_size
        )

        #print(f"bits raw: mean={bits.float().mean():.4f} "
        #      f"std={bits.float().std():.4f} min={bits.float().min():.4f} max={bits.float().max():.4f}")

        # ✅ 不管第幾層，用來當 address 的 bits 都要是 0/1
        bits = (bits > 0.5).to(torch.long)   # 或 .float() 再 long 都行

        #print(f"bits>0.5 ones_rate={(bits > 0).float().mean().item():.4f}")

        # idx = (((b0)*2 + b1)*2 + b2)*2 + ...
        idx = torch.zeros(B, self.num_luts, dtype=torch.long, device=device)
        for j in range(self.lut_input_size):
            idx = idx * 2 + bits[:, :, j].long()

        # LUT table: table: [num_luts, 2^k]
        table_expanded = self.table.unsqueeze(0).expand(B, -1, -1)  # [B, num_luts, 2^k]
        out = torch.gather(table_expanded, 2, idx.unsqueeze(-1)).squeeze(-1)

        # sigmoid activation (same as PyTorch version   )
        out = torch.sigmoid(out)
        #out = out.clamp(-5, 5)
        out = self.dropout(out)

        return out
    
    def forward_temp(self, x_bits):
        B = x_bits.size(0)
        device = x_bits.device
        x_bits = x_bits.float()

        # ---- update running_thr on TRAIN ----
        if self.use_running_thr and self.training:
            with torch.no_grad():
                thr_batch = x_bits.mean(dim=0, keepdim=True)  # 建議先 mean
                if self.thr_step < self.thr_warmup_steps:
                    self.running_thr.copy_(thr_batch)
                else:
                    m = self.thr_momentum
                    self.running_thr.mul_(1.0 - m).add_(thr_batch, alpha=m)
                self.thr_step.add_(1)

        # ---- extract k features ----
        flat_idx = self.conn_idx.view(-1)  # [num_luts*k]
        x_sel = x_bits[:, flat_idx]        # [B, num_luts*k]

        # ---- per-feature threshold for those selected dims ----
        if self.use_running_thr:
            thr_sel = self.running_thr[:, flat_idx].to(device)  # [1, num_luts*k]
            bits = (x_sel > thr_sel).to(torch.long)
        else:
            bits = (x_sel > 0.5).to(torch.long)

        bits = bits.view(B, self.num_luts, self.lut_input_size)

        # idx
        idx = torch.zeros(B, self.num_luts, dtype=torch.long, device=device)
        for j in range(self.lut_input_size):
            idx = idx * 2 + bits[:, :, j]

        table_expanded = self.table.unsqueeze(0).expand(B, -1, -1)
        out = torch.gather(table_expanded, 2, idx.unsqueeze(-1)).squeeze(-1)

        # 你 CIFAR10 目前看起來 no-sigmoid 路線比較接近你之前的最好結果，就先不加 sigmoid
        out = self.dropout(out)
        return out