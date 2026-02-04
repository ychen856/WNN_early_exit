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
    ):
        super().__init__()

        self.in_bits = in_bits
        self.num_luts = num_luts
        self.lut_input_size = lut_input_size

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

        # -----------------------------
        # 2) Initialize LUT table
        # -----------------------------
        table = torch.zeros(num_luts, 2 ** lut_input_size)
        table = table.normal_(mean=0.0, std=init_std)
        self.table = nn.Parameter(table)

    def forward(self, x_bits):
        B = x_bits.size(0)
        device = x_bits.device

        # Binarize input bits to 0/1
        x_bits = (x_bits > 0.5).float()

        # Extract k bits for each LUT
        # conn_idx: [num_luts, k]
        # -> [B, num_luts, k]
        bits = x_bits[:, self.conn_idx.view(-1)].view(
            B, self.num_luts, self.lut_input_size
        )

        # idx = (((b0)*2 + b1)*2 + b2)*2 + ...
        idx = torch.zeros(B, self.num_luts, dtype=torch.long, device=device)
        for j in range(self.lut_input_size):
            idx = idx * 2 + bits[:, :, j].long()

        # LUT table: table: [num_luts, 2^k]
        table_expanded = self.table.unsqueeze(0).expand(B, -1, -1)  # [B, num_luts, 2^k]
        out = torch.gather(table_expanded, 2, idx.unsqueeze(-1)).squeeze(-1)

        # sigmoid activation (same as PyTorch version   )
        out = torch.sigmoid(out)
        return out