import argparse
from functools import partial
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

def cifar10_u8_collate_binary(batch, pos_label: int):
    xs = torch.stack(
        [torch.from_numpy(item[0]).permute(2, 0, 1).contiguous() for item in batch],
        dim=0,
    )  # uint8 [B,3,32,32]
    ys_raw = torch.tensor([item[1] for item in batch], dtype=torch.long)
    ys = (ys_raw == pos_label).long()
    return xs, ys

# ----------------------------
# 0) Utils
# ----------------------------
def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(total, 1)

# ----------------------------
# 1) Bitplane encoder (RGB -> bits)
# ----------------------------
@torch.no_grad()
def rgb_u8_to_bitplane_bits(x_u8: torch.Tensor, bits_per_channel: int = 8) -> torch.Tensor:
    """
    x_u8: [B,3,32,32] uint8
    return: [B, 3*32*32*bits] float32 {0,1} (threshold-major in each pixel is fine)
    """
    assert x_u8.dtype == torch.uint8
    assert x_u8.ndim == 4 and x_u8.size(1) == 3
    B, C, H, W = x_u8.shape
    x = x_u8.to(torch.int32)

    # bits: MSB..LSB or LSB..MSB doesn't matter much; keep consistent
    # Here: LSB-first
    planes = []
    for b in range(bits_per_channel):
        planes.append(((x >> b) & 1).to(torch.float32))  # [B,3,32,32]
    # stack -> [B, bits, 3, 32, 32] then flatten
    out = torch.stack(planes, dim=1).reshape(B, bits_per_channel * C * H * W)
    return out

# ----------------------------
# 2) WNN LUT layer with "layer1 uses x_bits > 0"
# ----------------------------
class WNNLUTLayer(nn.Module):
    def __init__(self, in_bits, num_luts, lut_input_size=6, dropout_p=0.0, init_std=0.01,
                 is_first_layer: bool = False):
        super().__init__()
        self.in_bits = int(in_bits)
        self.num_luts = int(num_luts)
        self.lut_input_size = int(lut_input_size)
        self.is_first_layer = bool(is_first_layer)

        conn = torch.randint(low=0, high=self.in_bits, size=(self.num_luts, self.lut_input_size), dtype=torch.long)
        self.register_buffer("conn_idx", conn)

        table = torch.zeros(self.num_luts, 2 ** self.lut_input_size).normal_(mean=0.0, std=init_std)
        self.table = nn.Parameter(table)

        self.dropout = nn.Dropout(p=float(dropout_p))

    def forward(self, x_bits: torch.Tensor) -> torch.Tensor:
        """
        x_bits:
          - layer0 input: {0,1} float (bitplane) -> binarize with >0.5
          - layer>=1 input: real -> binarize with >0   (EXACTLY your exp1)
        """
        B = x_bits.size(0)
        device = x_bits.device

        if self.is_first_layer:
            xb = (x_bits > 0.5).to(torch.long)    # 0/1
        else:
            xb = (x_bits > 0.0).to(torch.long)    # ✅ exp1: sign-binarize

        bits = xb[:, self.conn_idx.view(-1)].view(B, self.num_luts, self.lut_input_size)  # [B,L,k], 0/1 long

        idx = torch.zeros(B, self.num_luts, dtype=torch.long, device=device)
        for j in range(self.lut_input_size):
            idx = idx * 2 + bits[:, :, j]

        table_exp = self.table.unsqueeze(0).expand(B, -1, -1)  # [B,L,2^k]
        out = torch.gather(table_exp, 2, idx.unsqueeze(-1)).squeeze(-1)  # [B,L]

        out = torch.sigmoid(out)  # keep your original choice
        out = self.dropout(out)
        return out

# ----------------------------
# 3) Simple MultiLayer WNN backbone for binary CIFAR10
# ----------------------------
class MultiLayerWNN(nn.Module):
    def __init__(self, in_bits, hidden_luts, lut_input_size, num_classes,
                 drop_ps=None, init_std=0.01):
        super().__init__()
        if drop_ps is None:
            drop_ps = [0.0] * len(hidden_luts)
        assert len(drop_ps) == len(hidden_luts)

        layers = []
        prev_bits = in_bits
        for i, n_lut in enumerate(hidden_luts):
            layers.append(
                WNNLUTLayer(
                    in_bits=prev_bits,
                    num_luts=n_lut,
                    lut_input_size=lut_input_size,
                    dropout_p=drop_ps[i],
                    init_std=init_std,
                    is_first_layer=(i == 0),
                )
            )
            prev_bits = n_lut

        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(prev_bits, num_classes)

    def forward(self, x_bits: torch.Tensor) -> torch.Tensor:
        h = x_bits
        for layer in self.layers:
            h = layer(h)
        return self.classifier(h)

# ----------------------------
# 4) CIFAR10 binary dataloaders
# ----------------------------
def make_binary_cifar10_loaders(root: str, a: int, b: int, batch_size: int, num_workers: int = 0):
    tr = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
    te = datasets.CIFAR10(root=root, train=False, download=True, transform=None)

    def filter_indices(ds):
        idx = []
        for i, y in enumerate(ds.targets):
            if y == a or y == b:
                idx.append(i)
        return idx

    tr_idx = filter_indices(tr)
    te_idx = filter_indices(te)

    tr_sub = Subset(tr, tr_idx)
    te_sub = Subset(te, te_idx)

    # map labels to {0,1}
    def collate_fn(batch):
        xs = torch.stack([torch.from_numpy(item[0]).permute(2,0,1).contiguous() for item in batch], dim=0)  # uint8 [B,3,32,32]
        ys_raw = torch.tensor([item[1] for item in batch], dtype=torch.long)
        ys = (ys_raw == b).long()  # class a -> 0, class b -> 1
        return xs, ys

    collate = partial(cifar10_u8_collate_binary, pos_label=b)
    tr_loader = DataLoader(tr_sub, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    te_loader = DataLoader(te_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    return tr_loader, te_loader

# ----------------------------
# 5) Train loop
# ----------------------------
def train(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int = 5,
    save_path: str = None,
    bits_per_channel: int = 8,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = {"val_acc": -1.0, "state": None, "epoch": -1}

    bad = 0
    for ep in range(epochs):
        model.train()
        tot = 0
        cor = 0
        loss_sum = 0.0

        for x_u8, y in train_loader:
            x_u8 = x_u8.to(device, non_blocking=True)  # uint8
            y = y.to(device, non_blocking=True)

            x_bits = rgb_u8_to_bitplane_bits(x_u8, bits_per_channel=bits_per_channel)  # float {0,1}
            opt.zero_grad(set_to_none=True)
            logits = model(x_bits)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(dim=-1)
            cor += (pred == y).sum().item()
            tot += y.size(0)

        tr_acc = cor / max(tot, 1)
        tr_loss = loss_sum / max(tot, 1)
        va_acc = eval_binary(model, val_loader, device, bits_per_channel)

        if va_acc > best["val_acc"]:
            best["val_acc"] = va_acc
            best["epoch"] = ep
            best["state"] = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({"model": best["state"], "best": best}, save_path)
                print(f"[BEST] epoch={ep:03d} val_acc={va_acc*100:.2f}% -> saved")
        else:
            bad += 1

        print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | train_acc={tr_acc*100:.2f}% | val_acc={va_acc*100:.2f}% | lr={lr:.2e}")

        if bad >= patience:
            print(f"Early stop: patience={patience}")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=True)
    return model, best

@torch.no_grad()
def eval_binary(model, loader, device, bits_per_channel: int = 8):
    model.eval()
    correct = 0
    total = 0
    for x_u8, y in loader:
        x_u8 = x_u8.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_bits = rgb_u8_to_bitplane_bits(x_u8, bits_per_channel=bits_per_channel)
        logits = model(x_bits)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

# ----------------------------
# 6) Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./data")
    ap.add_argument("--pair", type=str, default="3,5", help="e.g. '0,1' or '3,5'")
    ap.add_argument("--hidden", type=str, default="4500,2000", help="e.g. '4500,2000'")
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--drop", type=str, default="0.2,0.1", help="dropout per layer, e.g. '0.2,0.1'")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--bits_per_channel", type=int, default=8)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    a, b = [int(x) for x in args.pair.split(",")]
    hidden_luts = [int(x) for x in args.hidden.split(",")]
    drop_ps = [float(x) for x in args.drop.split(",")]
    assert len(drop_ps) == len(hidden_luts), "len(drop) must match len(hidden)"

    train_loader, val_loader = make_binary_cifar10_loaders(args.root, a, b, args.batch_size)

    in_bits = 3 * 32 * 32 * args.bits_per_channel
    model = MultiLayerWNN(
        in_bits=in_bits,
        hidden_luts=hidden_luts,
        lut_input_size=args.k,
        num_classes=2,
        drop_ps=drop_ps,
        init_std=0.01,
    ).to(device)

    save_path = args.save if args.save else None
    model, best = train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_path=save_path,
        bits_per_channel=args.bits_per_channel,
    )
    print(f"Best val acc: {best['val_acc']*100:.2f}% at epoch {best['epoch']:03d}")

if __name__ == "__main__":
    main()