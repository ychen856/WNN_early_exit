# src/train/train_wnn.py
import argparse
import copy
from pathlib import Path
import json
from networkx import sigma
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from src.core.linearExitHead import build_exits_from_ckpt
from src.dataio.data import build_loaders_bits
from src.dataio.mapping import make_tuple_mapping, audit_mapping
from src.early_exit import _head_logits_from_hidden_trainable, _margin_from_logits
from src.exit.analyze_hidden import compute_mu_sigma
from src.prune import *
from src.early_exit import *
from src.tools.utils import print_sweep_table  
from test import *
from src.core.infer import *
from src.core.multiLayerWNN import MultiLayerWNN, build_backbone_from_ckpt, load_ckpt, save_ckpt, save_ckpt_v2
from src.dataio.encode import minmax_normalize, thermometer_encode, dt_thermometer_encode, compute_dt_thresholds
from src.tools.fpga_tools.export_fpga_bundle import export_multilayer_2layer_for_fpga, verify_multilayer_export
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


import torch
import torch.nn.functional as F

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

@torch.no_grad()
def eval_final_only(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)  # final only
        loss = F.cross_entropy(logits, yb)
        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        loss_sum += loss.item() * yb.numel()
    return loss_sum / total, correct / total

@torch.no_grad()
def eval_exit1_only(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        _, exit1_logits, _ = model.forward_with_all_hidden_and_exits(xb)
        loss = F.cross_entropy(exit1_logits, yb)
        pred = exit1_logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        loss_sum += loss.item() * yb.numel()
    return loss_sum / total, correct / total

@torch.no_grad()
def eval_with_gate(model, loader, device, thr=2.0):
    """
    Use your gating rule (logit margin) on exit1 logits.
    Returns: overall_acc, exit_rate
    """
    model.eval()
    total, correct, exited = 0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        final_logits, exit1_logits, _ = model.forward_with_all_hidden_and_exits(xb)

        # margin on logits (NOT softmax prob) — matches your earlier approach
        top2 = torch.topk(exit1_logits, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        exit_mask = margin > thr

        logits = final_logits.clone()
        logits[exit_mask] = exit1_logits[exit_mask]

        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        exited += exit_mask.sum().item()

    overall_acc = correct / total
    exit_rate = exited / total
    return overall_acc, exit_rate

# -------------------------
# G2 training
# -------------------------

def dump_trainable(model, tag=""):
    trainable = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable.append((n, tuple(p.shape)))
    print(f"[{tag}] trainable params ({len(trainable)}):")
    for n, shp in trainable[:40]:
        print("  ", n, shp)
    if len(trainable) > 40:
        print("  ...")

def freeze_g2(model, freeze_exit=True):
    # freeze layer1
    for p in model.layers[0].parameters():
        p.requires_grad = False

    # train layer2 + final
    for p in model.layers[1].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    # exit head normally freeze in G2
    for p in model.exit1_classifier.parameters():
        p.requires_grad = (not freeze_exit)

@dataclass
class G2Config:
    num_epochs: int = 20
    lr_layer2: float = 3e-4
    lr_final: float = 3e-4
    lr_exit: float = 0.0           # 0 = freeze exit head
    lambda_exit: float = 0.0       # start with 0; can set 0.05~0.1
    weight_decay: float = 1e-3
    grad_clip: Optional[float] = 1.0
    thr_eval: float = 2.0


def cotrain_g2_multi_exit(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs: int = 30,

    # 哪些 layer 要 train（通常 layer1, layer2）
    train_layer_indices: Sequence[int] = (1, 2),

    # freeze 哪些 layer（g2 常見 freeze layer0）
    freeze_layer_indices: Sequence[int] = (0,),


    # 你已 build 好的 exits
    exit_heads: List[nn.Module],
    payload_exit_cfg: List[dict],   # list[dict]，包含 layer_idx / exit_tau...（給 eval 用）

    # cascade gate thresholds（長度 = num_exits）
    thrs: Sequence[float] = (1.0, 1.5),
    use_prob_margin: bool = False,

    # loss weights
    lambda_final: float = 1.0,
    lambda_exits: Optional[Sequence[float]] = None,  # e.g. (0.3, 0.3)

    # lr
    lr_backbone: float = 3e-4,
    lr_classifier: float = 3e-4,
    lr_exits: float = 3e-3,

    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,

    # --- new (gate-aware) ---
    gate_temp: float = 1.0,          # sigmoid 溫度，0.5~1.0 都可
    use_gate_weighting: bool = True,  # 開關

    # best selection metric
    best_metric: str = "val_overall_acc",  # 目前只用 cascade overall
):
    """
    g2: update layer1+layer2 + classifier + all exit heads
        (freeze layer0)

    Best model selected by eval_cascade_multi_exit(val, thrs).
    """
    model = model.to(device)

    assert len(exit_heads) == len(payload_exit_cfg), "exit_heads and payload_exit_cfg must align"
    num_exits = len(exit_heads)
    assert len(thrs) == num_exits, "len(thrs) must equal num_exits"

    if lambda_exits is None:
        lambda_exits = [0.3] * num_exits
    else:
        assert len(lambda_exits) == num_exits

    # -----------------------
    # 0) Freeze all first
    # -----------------------
    set_requires_grad(model, False)

    # freeze specified layers (redundant but explicit)
    for li in freeze_layer_indices:
        if 0 <= li < len(model.layers):
            set_requires_grad(model.layers[li], False)

    # unfreeze specified train layers
    for li in train_layer_indices:
        if li < 0 or li >= len(model.layers):
            raise ValueError(f"train_layer_indices contains out-of-range layer {li}")
        set_requires_grad(model.layers[li], True)

    # unfreeze final classifier (g2 必須動 classifier)
    if not hasattr(model, "classifier"):
        raise ValueError("model has no classifier; g2 expects final classifier exists.")
    set_requires_grad(model.classifier, True)

    # exit heads trainable
    exit_heads = [h.to(device) for h in exit_heads]
    for h in exit_heads:
        set_requires_grad(h, True)
    # freeze ex0
    #set_requires_grad(exit_heads[0], False)

    # -----------------------
    # 1) Build optimizer param groups
    # -----------------------
    params_backbone = []
    for li in train_layer_indices:
        params_backbone += [p for p in model.layers[li].parameters() if p.requires_grad]

    params_classifier = [p for p in model.classifier.parameters() if p.requires_grad]

    params_exits = []
    for h in exit_heads:
        params_exits += [p for p in h.parameters() if p.requires_grad]
    # only accept ex1
    #params_exits += [p for p in exit_heads[1].parameters() if p.requires_grad]

    print(f"[g2] trainable params: backbone={sum(p.numel() for p in params_backbone)} "
          f"classifier={sum(p.numel() for p in params_classifier)} "
          f"exits={sum(p.numel() for p in params_exits)}")

    optimizer = torch.optim.AdamW(
        [
            {"params": params_backbone,   "lr": lr_backbone,   "weight_decay": weight_decay},
            {"params": params_classifier, "lr": lr_classifier, "weight_decay": weight_decay},
            {"params": params_exits,      "lr": lr_exits,      "weight_decay": weight_decay},
        ]
    )

    best = {"val_overall_acc": -1.0, "state": None}

    # -----------------------
    # 2) Train loop
    # -----------------------
    epoch_idx = 0
    for epoch in range(num_epochs):
        model.train()
        for h in exit_heads:
            h.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            
            final_logits, h_list = model.forward_with_all_hidden(xb)

            # per-sample CE
            ce_final = F.cross_entropy(final_logits, yb, reduction="none")  # [B]

            # -------- gate-aware weighting --------
            if use_gate_weighting:
                # soft gate weights w_i = sigmoid((margin_i - thr_i)/T)
                w_list = []
                ce_exit_list = []
                m_list = []
                for i in range(num_exits):
                    cfg = payload_exit_cfg[i]
                    layer_idx = int(cfg["layer_idx"])
                    thr_i = float(thrs[i])

                    h_i = h_list[layer_idx]
                    logits_i = _head_logits_from_hidden_trainable(exit_heads[i], h_i, device)  # [B,C]

                    # per-sample exit CE
                    ce_i = F.cross_entropy(logits_i, yb, reduction="none")  # [B]
                    ce_exit_list.append(ce_i)

                    # margin (logits margin) -> [B]
                    m_i = _margin_from_logits(logits_i, use_prob=use_prob_margin)
                    m_list.append(m_i)

                    # soft gate weight
                    w_i = torch.sigmoid((m_i - thr_i) / gate_temp)  # [B] in (0,1)
                    w_list.append(w_i)

                '''# final weight = prob of "still undecided"
                # 用連乘近似：w_final = Π (1 - w_i)
                w_final = torch.ones_like(ce_final)
                for w_i in w_list:
                    w_final = w_final * (1.0 - w_i)

                # weighted final loss
                loss_final = (w_final * ce_final).mean()'''

                '''# weighted exit losses
                loss_exit_sum = 0.0
                for i in range(num_exits):
                    w_i = w_list[i].detach()  # ✅ 這裡建議 detach，避免 gate 自己被 CE 拉到極端
                    loss_exit_i = (w_i * ce_exit_list[i]).mean()
                    loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * loss_exit_i

                loss = float(lambda_final) * loss_final + loss_exit_sum'''
                # ce_final: [B]
                # ce_i: [B] for each exit

                eps = 1e-8
                u = torch.ones_like(ce_final)  # [B], undecided prob
                loss_exit_sum = 0.0
                #T0 ∈ {2.5, 3.0, 3.5}
	            #T1 ∈ {1.0, 1.25, 1.5}
                # (T0,T1) = (3.0,1.0), (3.0,1.25), (3.0,1.5), (3.5,1.25) 
                # T1 ∈ {1.10, 1.25, 1.40, 1.50, 1.60, 1.75}
                T = [3.5, 1.25]
                for i in range(num_exits):
                    m_i = m_list[i]                  # [B]
                    ce_i = ce_exit_list[i]           # [B]
                    thr_i = float(thrs[i])
                    Ti = float(T[i])                 # or shared T

                    w_i = torch.sigmoid((m_i - thr_i) / Ti)   # [B] in (0,1)
                    take_i = u * w_i                           # [B]

                    # gate 不想被 CE 拉爆：detach weights
                    take_i_det = take_i.detach()

                    # ✅ weighted average (NOT mean)
                    loss_i = (take_i_det * ce_i).sum() / (take_i_det.sum() + eps)
                    loss_exit_sum = loss_exit_sum + lambda_exits[i] * loss_i

                    # 更新 undecided（這裡建議也用 detach 避免數值怪）
                    #u = u * (1.0 - w_i.detach())
                    u = u * (1.0 - w_i)

                # final weighted average
                u_det = u.detach()
                loss_final = (u_det * ce_final).sum() / (u_det.sum() + eps)

                loss = lambda_final * loss_final + loss_exit_sum

            else:
                # fallback: 原本的 naive loss
                loss_final = ce_final.mean()
                loss_exit_sum = 0.0
                for i in range(num_exits):
                    cfg = payload_exit_cfg[i]
                    layer_idx = int(cfg["layer_idx"])
                    h_i = h_list[layer_idx]
                    logits_i = _head_logits_from_hidden_trainable(exit_heads[i], h_i, device)
                    loss_exit_i = F.cross_entropy(logits_i, yb)
                    loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * loss_exit_i
                loss = float(lambda_final) * loss_final + loss_exit_sum






            loss.backward()
            if grad_clip is not None:
                if params_backbone:
                    torch.nn.utils.clip_grad_norm_(params_backbone, grad_clip)
                if params_classifier:
                    torch.nn.utils.clip_grad_norm_(params_classifier, grad_clip)
                if params_exits:
                    torch.nn.utils.clip_grad_norm_(params_exits, grad_clip)

            optimizer.step()

        # -----------------------
        # 3) Eval (use your existing cascade eval)
        # -----------------------
        # 你已有 eval_cascade_multi_exit
        out_val = eval_cascade_multi_exit(
            model, val_loader, device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=thrs,
            use_prob_margin=use_prob_margin,
            log_margins=False,
        )
        va_overall = out_val["overall_acc"]

        print(
            f"[G2] Ep{epoch:03d} | overall@{tuple(thrs)} va={va_overall*100:.2f} "
            f"| exit_rates={out_val['exit_rates']} final_rate={out_val['final_rate']:.4f}"
        )

        if va_overall > best["val_overall_acc"]:
            best["val_overall_acc"] = va_overall
            # 存 backbone + exits
            best["state"] = {
                "model": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                "exits": [copy.deepcopy({k: v.detach().cpu() for k, v in h.state_dict().items()}) for h in exit_heads],
            }

    # -----------------------
    # 4) Restore best
    # -----------------------
    if best["state"] is not None:
        model.load_state_dict(best["state"]["model"], strict=True)
        for i, h in enumerate(exit_heads):
            h.load_state_dict(best["state"]["exits"][i], strict=True)

    return model, exit_heads, best




# -----------------------------
# 4) End-to-end driver function
# -----------------------------
def run_cached_exit_pipeline(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    exit1_keep_idx,
    num_classes=10,
    cache_batches_train=None,
    cache_batches_val=None,
    cache_batches_test=None,
    normalize=True,
    num_epochs=50,
    lr=3e-3,
    weight_decay=1e-4,
    thr_list=(0.0, 0.5, 1.0, 2.0, 4.0),
):
    # 1) Cache features
    X_tr, y_tr, mu, sigma = cache_exit1_features(
        model, train_loader, device, exit1_keep_idx,
        max_batches=cache_batches_train, normalize=normalize
    )
    X_va, y_va, _, _ = cache_exit1_features(
        model, val_loader, device, exit1_keep_idx,
        max_batches=cache_batches_val, normalize=normalize
    )
    # For test, we only need mu/sigma from train if normalize=True
    X_te, y_te, _, _ = cache_exit1_features(
        model, test_loader, device, exit1_keep_idx,
        max_batches=cache_batches_test, normalize=normalize
    )

    print(f"[cache] train X={tuple(X_tr.shape)}, val X={tuple(X_va.shape)}, test X={tuple(X_te.shape)}")

    # 2) Train classifier on cached
    clf = train_exit_head_on_cached(
        X_tr, y_tr, X_va, y_va,
        num_classes=num_classes,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=1024,
        device=device,
    )

    # 3) Evaluate metrics on test
    res = eval_cached_exit_metrics(
        model, clf, test_loader, device,
        exit1_keep_idx=exit1_keep_idx,
        mu=mu if normalize else None,
        sigma=sigma if normalize else None,
        thr_list=thr_list
    )
    print("[cached-exit metrics]", res)
    return clf, res

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--backbone_ckpt", type=str, required=True)
    parser.add_argument("--path_out", type=str, required=True, help="Save ckpt with exit_config list")

    parser.add_argument("--exit_layers", type=str, default="0", help='e.g. "0" or "0,1"')
    parser.add_argument("--k", type=str, default="256", help='e.g. "256" or "256,512" (broadcast ok)')
    parser.add_argument("--keep_mode", type=str, default="p*(1-p)*std", help='broadcast ok')
    parser.add_argument("--exit_tau", type=str, default="1.0", help='broadcast ok')

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size_cached", type=int, default=512)
    parser.add_argument("--use_norm", action="store_true", default=True)
    parser.add_argument("--thr", type=str, default="1.0,1.5",
                    help="comma-separated thresholds per exit, e.g. 1.0,1.5")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders
    train_loader, val_loader, test_loader, in_bits, C, ds_meta = build_loaders_bits(
        dataset=args.dataset,
        root="/Users/yi-chunchen/workspace/WNN_early_exit/datasets/",
        batch_size_train=256,
        batch_size_eval=512,
        val_ratio=0.1,
        seed=3,
        z=32,
        device_for_encoding=device,
        shuffle_train=False,
    )

    backbone, bb_cfg, extra = build_backbone_from_ckpt(args.backbone_ckpt, device)
    backbone.eval()
    C = int(bb_cfg["num_classes"])

    exit_heads, exit_cfg_list = build_exits_from_ckpt(args.backbone_ckpt, device, num_classes=C)

    test_loss, test_acc = eval_epoch(backbone, test_loader, device)
    print("[final-only] test_acc", test_acc)
    # 之後直接用 backbone + exit_heads + exit_cfg_list 做 cascade eval / g1 training
    out = eval_cascade_multi_exit(
        backbone, test_loader, device,
        exit_heads=exit_heads,
        exit_cfg_list=[ec.to_payload() for ec in exit_cfg_list],  # 或你也可以把 eval 改成吃 ExitConfig 物件
        thrs=[1.0, 1.5],
        use_prob_margin=False,
    )
    print(out)


    # group 1 co-train
    # thrs 由系統輸入 "1.0,1.5"
    thr_list = [float(x) for x in args.thr.split(",")]
    assert len(thr_list) == 2

    payload_exit_cfg = [ec.to_payload() for ec in exit_cfg_list]

    # g2: train layer1+layer2 + classifier + exits
    backbone, exit_heads, best = cotrain_g2_multi_exit(
        model=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=30,

        train_layer_indices=(1, 2),
        freeze_layer_indices=(0,),

        exit_heads=exit_heads,
        payload_exit_cfg=payload_exit_cfg,
        thrs=thr_list,

        lambda_final=1.0,
        lambda_exits=(0.05, 0.10),

        lr_backbone=1e-4,
        lr_classifier=3e-4,
        lr_exits=5e-4,
        weight_decay=1e-3,
    )

    print("Best val overall acc:", best["val_overall_acc"])

    # 最後存成一個 ckpt：backbone_cfg 不動 + backbone weights + exit_cfg_list
    payload_exit_cfg = [ec.to_payload() for ec in exit_cfg_list]


    save_ckpt_v2(
        args.path_out,
        backbone,                 # backbone model
        exit_heads,
        bb_cfg,          # backbone cfg 不動
        exit_cfg_list=payload_exit_cfg,  # <-- exit cfg list
        extra={"dataset": args.dataset}
    )

    print("\nSaved:", args.path_out)
    print("Exit cfg list length:", len(payload_exit_cfg))
    
    test_loss, test_acc = eval_epoch(backbone, test_loader, device)
    print("[final-only] test_acc", test_acc)

    thrs = [0.0, 0.5, 1.0, 2.0, 4.0]
    for thr in thrs:
        out = eval_overall_at_thr_multi_exit(
            backbone, test_loader, device,
            thr=thr,
            exit_id=0,
            exit_cfg_list=payload_exit_cfg,   # <-- 用 ExitConfig list
            exit_heads=exit_heads,
            use_prob_margin=False,
        )
        print(thr, out["exit_rate"], out["overall_acc"], out["exited_acc"], out["non_exited_acc"],
              out["margin_mean"], out["margin_p95"])
        print_eval_profile(f"G2-v2 exit0@thr={thr}", out)
    
    print('=======================================')
    thrs0 = [0.0, 0.25, 0.5, 0.75, 1.0]
    thrs1 = [1.2, 1.5, 1.8, 2.0]

    for thr0 in thrs0:
        for thr1 in thrs1:
            out = eval_cascade_multi_exit(
                    backbone, test_loader, device,
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

            print(
                f"{thr0:>4} {thr1:>4} | "
                f"overall={out['overall_acc']:.4f} | "
                f"r0={out['exit_rates'][0]:.4f} a0={out['exit_accs'][0]:.4f} | "
                f"r1={out['exit_rates'][1]:.4f} a1={out['exit_accs'][1]:.4f} | "
                f"rf={out['final_rate']:.4f} af={out['final_acc']:.4f}"
            )
            m0 = out["margin_stats"][0]
            m1 = out["margin_stats"][1]
            print(f" | m0f={m0['mean']:.2f}/{m0['p95']:.2f} m1={m1['mean']:.2f}/{m1['p95']:.2f}")
            m0_detail = out['margin_stats'][2]
            m1_detail = out['margin_stats'][3]
            print(f" | m0_undecided={m0_detail['undecided_mean']:.2f} m0_undecided_p95={m0_detail['undecided_p95']:.2f} m0_taken_mean={m0_detail['taken_mean']:.2f} m0_taken_p95={m0_detail['taken_p95']:.2f}")
            print(f" | m1_undecided={m1_detail['undecided_mean']:.2f} m1_undecided_p95={m1_detail['undecided_p95']:.2f} m1_taken_mean={m1_detail['taken_mean']:.2f} m1_taken_p95={m1_detail['taken_p95']:.2f}")
            print_eval_profile(f"G2-v2 cascade@({thr0},{thr1})", out)




    
