# src/train/train_wnn.py
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import json
from networkx import sigma
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn.functional as F
from src.core.linearExitHead import ExitHead
from src.dataio.data import build_loaders_bits
from src.dataio.mapping import make_tuple_mapping, audit_mapping
from src.exit.analyze_hidden import analyze_hidden_for_exit, compute_mu_sigma, select_exit_keep_idx
from src.exit.ckpt_exit import ExitConfig
from src.prune import *
from src.early_exit import *
from src.tools.utils import print_sweep_table
from test import *
from src.core.infer import *
from src.core.multiLayerWNN import MultiLayerWNN, load_ckpt, save_ckpt, save_ckpt_v2
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

# from core.decision import tune_decision  #  Step 2

CANONICAL_MAPPING = Path("/Users/yi-chunchen/workspace/WNN_early_exit/models/meta/tuple_mapping.json")

def load_or_create_mapping(bit_len, tiles, num_luts, addr_bits, seed=42, save_path=CANONICAL_MAPPING):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        mapping = json.loads(save_path.read_text())
        # alignment check
        assert len(mapping) == num_luts, "num_luts mismatch with saved mapping"
        return mapping

    mapping = make_tuple_mapping(
        num_luts=num_luts,
        addr_bits=addr_bits,
        bit_len=bit_len,
        tiles=tiles,          #  None or meta["tile_index_ranges"]
        seed=seed
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    return mapping


def get_lr(epoch):
    if epoch < 25:
        return 1e-3
    elif epoch < 55:
        return 3e-4
    else:
        return 1e-4

def compute_accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

# -----------------------------
# 2) Train exit head on cached features
# -----------------------------
def train_exit_head_on_cached(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes=10,
    num_epochs=50,
    lr=3e-3,
    weight_decay=1e-4,
    batch_size=1024,
    device="cpu",
):
    """
    Trains a simple Linear classifier on cached features.
    Returns:
      clf: nn.Linear(K -> num_classes)
      best_state: best weights (loaded into clf)
    """
    K = X_train.size(1)
    clf = nn.Linear(K, num_classes, bias=True).to(device)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # --- train ---
        clf.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = clf(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # --- val ---
        clf.eval()
        v_total_loss = 0.0
        v_total = 0
        v_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = clf(xb)
                loss = F.cross_entropy(logits, yb)
                v_total_loss += loss.item() * xb.size(0)
                pred = logits.argmax(dim=-1)
                v_correct += (pred == yb).sum().item()
                v_total += xb.size(0)

        val_loss = v_total_loss / v_total
        val_acc = v_correct / v_total

        print(
            f"[cached-exit] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}

    if best_state is not None:
        clf.load_state_dict(best_state)

    return clf

def train_exit_head(model, train_loader, val_loader, device,
                    num_epochs=50, base_lr=1e-3, weight_decay=1e-4):
    model.to(device)

    # freeze backbone + final classifier
    for p in model.layers.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False

    # train only exit head
    for p in model.exit1_classifier.parameters():
        p.requires_grad = True

    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable:", trainable)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

    best_state = None
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, exit1_logits, _ = model.forward_with_all_hidden_and_exits(xb)
            loss = F.cross_entropy(exit1_logits, yb)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = eval_exit1_epoch(model, train_loader, device)
        val_loss, val_acc = eval_exit1_epoch(model, val_loader, device)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
              f"train_acc={train_acc*100:.2f}% | val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def collect_hidden_activations(model, data_loader, device):
    model.eval()
    all_h = []
    all_y = []

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits, h_last = model(xb, return_hidden=True)  # the forwarding need to be able to support return_hidden
        all_h.append(h_last.cpu())
        all_y.append(yb.cpu())

    H = torch.cat(all_h, dim=0)
    Y = torch.cat(all_y, dim=0)
    return H, Y





def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_exits(s: str) -> List[Tuple[int, int]]:
    """
    Parse: "0:256,1:512,2:256" -> [(0,256),(1,512),(2,256)]
    """
    out = []
    for item in parse_csv_list(s):
        if ":" not in item:
            raise ValueError(f"Bad --exits item: {item}, expected layer:K")
        li, k = item.split(":")
        out.append((int(li), int(k)))
    return out


def broadcast_or_match(values: List[str], n: int, name: str) -> List[str]:
    """
    If len(values)==1 -> broadcast to n
    If len(values)==n -> keep
    else -> error
    """
    if len(values) == 1:
        return values * n
    if len(values) == n:
        return values
    raise ValueError(f"--{name} expects 1 value or {n} values, got {len(values)}")


@dataclass
class ExitSpec:
    layer_idx: int
    K: int
    keep_mode: str
    exit_tau: float


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)




############################################
# V2
############################################
def _parse_list(s, cast=int):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]

def _broadcast(xs, n):
    if len(xs) == 1:
        return xs * n
    if len(xs) == n:
        return xs
    raise ValueError(f"Need 1 or {n} values, got {len(xs)}")

@torch.no_grad()
def cache_exit_features(model, loader, device, layer_idx, keep_idx, mu, sigma, use_norm: bool):
    model.eval()
    Xs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        _, h_list = model.forward_with_all_hidden(xb)
        h = h_list[layer_idx][:, keep_idx]
        if use_norm:
            h = (h - mu.to(h.device)) / sigma.to(h.device)
        Xs.append(h.detach().cpu())
        ys.append(yb.detach().cpu())
    return torch.cat(Xs, 0), torch.cat(ys, 0)

def train_one_exit_cached(head, X_train, y_train, X_val, y_val, device, epochs=50, lr=3e-3, wd=1e-3, bs=512):
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)

    best = None
    best_val = 0.0
    N = X_train.size(0)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(N)
        tot_loss = 0.0
        corr = 0
        tot = 0

        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)

            opt.zero_grad()
            logits = head.classifier(xb) / head.exit_tau  # cached 已經是 [N,k]
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            tot_loss += loss.item() * yb.size(0)
            corr += (logits.argmax(-1) == yb).sum().item()
            tot += yb.size(0)

        head.eval()
        with torch.no_grad():
            v_logits = head.classifier(X_val.to(device)) / head.exit_tau
            v_acc = (v_logits.argmax(-1).cpu() == y_val).float().mean().item()
            v_loss = F.cross_entropy(v_logits, y_val.to(device)).item()

        print(f"[exit layer] ep{ep:03d} train_loss={tot_loss/tot:.4f} train_acc={corr/tot*100:.2f}% "
              f"| val_loss={v_loss:.4f} val_acc={v_acc*100:.2f}%")

        if v_acc > best_val:
            best_val = v_acc
            best = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best is not None:
        head.load_state_dict(best)
    return head, best_val













if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--backbone_ckpt", type=str, required=True)
    parser.add_argument("--path_out", type=str, required=True, help="Save ckpt with exit_config list")

    parser.add_argument("--exit_layers", type=str, default="0", help='e.g. "0" or "0,1"')
    parser.add_argument("--k", type=str, default="256", help='e.g. "256" or "256,512" (broadcast ok)')
    parser.add_argument("--keep_mode", type=str, default="p*(1-p)*std", help='broadcast ok')
    parser.add_argument("--exit_tau", type=str, default="1.0", help='broadcast ok')
    parser.add_argument("--thr", type=str, default="0.5", help='broadcast ok (for future online routing)')

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size_cached", type=int, default=512)
    parser.add_argument("--use_norm", action="store_true", default=True)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders
    train_loader, val_loader, test_loader, in_bits, C, ds_meta = build_loaders_bits(
        dataset=args.dataset,
        root="/Users/yi-chunchen/workspace/WNN_early_exit/datasets/",
        batch_size_train=256,
        batch_size_eval=512,
        val_ratio=0.1,
        seed=42,
        z=32,
        device_for_encoding=device,
        shuffle_train=False,
    )

    # backbone cfg 不動：從 ckpt 讀
    model, bb_cfg, ex_cfg, extra = load_ckpt(args.backbone_ckpt, device)

    # 這支 script 是「從 backbone 建 exit heads」，不應該吃到既有 exit cfg
    if ex_cfg is not None:
        print("[warn] backbone_ckpt already contains exit_config; will ignore and rebuild exits from scratch.")

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 之後就用 bb_cfg 當 backbone_cfg（保存時原樣寫回）
    backbone_cfg = bb_cfg


    exit_layers = _parse_list(args.exit_layers, int)
    ks = _broadcast(_parse_list(args.k, int), len(exit_layers))
    keep_modes = _broadcast([x.strip() for x in args.keep_mode.split(",")], len(exit_layers))
    exit_taus = _broadcast([float(x.strip()) for x in args.exit_tau.split(",")], len(exit_layers))
    thrs = _broadcast([float(x.strip()) for x in args.thr.split(",")], len(exit_layers))

    exit_heads = []
    exit_cfg_list = []

    for layer_idx, k, kmode, exit_tau, thr in zip(exit_layers, ks, keep_modes, exit_taus, thrs):
        print("\n" + "="*80)
        print(f"Build/Train exit @ layer {layer_idx} | k={k} mode={kmode} exit_tau={exit_tau} thr={thr}")
        print("="*80)

        mean_d, std_d, p1_d, bias = analyze_hidden_for_exit(model, train_loader, device, layer_idx=layer_idx)
        exit_keep_idx = select_exit_keep_idx(mean_d, std_d, p1_d, bias, k=k, keep_mode=kmode)

        mu, sigma = compute_mu_sigma(model, train_loader, device, layer_idx=layer_idx, exit_keep_idx=exit_keep_idx)

        # cache (optional normalization)
        X_train, y_train = cache_exit_features(model, train_loader, device, layer_idx, exit_keep_idx, mu, sigma, args.use_norm)
        X_val, y_val     = cache_exit_features(model, val_loader,   device, layer_idx, exit_keep_idx, mu, sigma, args.use_norm)
        X_test, y_test   = cache_exit_features(model, test_loader,  device, layer_idx, exit_keep_idx, mu, sigma, args.use_norm)
        print(f"[cache] train {tuple(X_train.shape)} val {tuple(X_val.shape)} test {tuple(X_test.shape)}")

        # head from scratch (but classifier trained on cached X)
        head = ExitHead(k=k, num_classes=C, exit_tau=exit_tau,
                        exit_keep_idx=exit_keep_idx, mu=mu, sigma=sigma,
                        use_norm=args.use_norm)

        # 只訓練 classifier.weight（因為 keep_idx/mu/sigma 是 buffer）
        head, best_val = train_one_exit_cached(
            head, X_train, y_train, X_val, y_val,
            device, epochs=args.epochs, lr=args.lr, wd=args.weight_decay, bs=args.batch_size_cached
        )

        # 存 cfg（list item）
        exit_cfg_list.append(ExitConfig(
            layer_idx=layer_idx,
            k=k,
            keep_mode=kmode,
            thr=thr,
            exit_tau=exit_tau,
            exit_keep_idx=exit_keep_idx.cpu(),
            mu=mu.cpu(),
            sigma=sigma.cpu(),
        ))

        exit_heads.append(head.cpu())

        # quick test acc of this exit alone
        with torch.no_grad():
            logits = (head.classifier(X_test.to(device)) / head.exit_tau).cpu()
            acc = (logits.argmax(-1) == y_test).float().mean().item()
        print(f"[exit@layer{layer_idx}] test_exit_acc={acc*100:.2f}% | best_val={best_val*100:.2f}%")

    

    # 最後存成一個 ckpt：backbone_cfg 不動 + backbone weights + exit_cfg_list
    payload_exit_cfg = [ec.to_payload() for ec in exit_cfg_list]

    '''payload_exit_cfg = []
    for ec in exit_cfg_list:
        payload_exit_cfg.append({
            "layer_idx": ec.layer_idx,
            "k": ec.k,
            "keep_mode": ec.keep_mode,
            "thr": float(ec.thr),
            "exit_tau": float(ec.exit_tau),
            "exit_keep_idx": ec.exit_keep_idx,
            "mu": ec.mu,
            "sigma": ec.sigma,
        })'''

    save_ckpt_v2(
        args.path_out,
        model,                 # backbone model
        backbone_cfg,          # backbone cfg 不動
        exit_cfg_list=payload_exit_cfg,  # <-- exit cfg list
        extra={"dataset": args.dataset}
    )

    print("\nSaved:", args.path_out)
    print("Exit cfg list length:", len(payload_exit_cfg))

    thrs = [0.0, 0.5, 1.0, 2.0, 4.0]
    for thr in thrs:
        out = eval_overall_at_thr_multi_exit(
            model, test_loader, device,
            thr=thr,
            exit_id=0,
            exit_cfg_list=payload_exit_cfg,   # <-- 用 ExitConfig list
            exit_heads=exit_heads,
            use_prob_margin=False,
        )
        print(thr, out["exit_rate"], out["overall_acc"], out["exited_acc"], out["non_exited_acc"],
              out["margin_mean"], out["margin_p95"])
    
    print('=======================================')
    thrs0 = [0.5, 1.0, 1.5]
    thrs1 = [1.5, 2.0, 2.5]

    for thr0 in thrs0:
        for thr1 in thrs1:
            out = eval_cascade_multi_exit(
                    model, test_loader, device,
                    exit_heads=exit_heads,
                    exit_cfg_list=payload_exit_cfg,
                    thrs=[thr0, thr1],
                    use_prob_margin=False,
                )
            s = sum(out["exit_rates"]) + out["final_rate"]
            assert abs(s - 1.0) < 1e-6, s
            '''print(thr0, thr1,
                out["overall_acc"], out["exit_rates"], out["final_rate"],
                out["exit_accs"],
                out["final_acc"])'''

            r0, r1 = out["exit_rates"]
            rF = out["final_rate"]

            exp_layers = 1*r0 + 2*r1 + 3*rF
            compute_ratio = exp_layers / 3.0

            print(thr0, thr1,
                out["overall_acc"],
                out["exit_rates"], out["final_rate"],
                "E_layers", round(exp_layers, 4),
                "compute", round(compute_ratio, 4))

