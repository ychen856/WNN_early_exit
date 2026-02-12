# src/train/train_wnn.py
from pathlib import Path
import json
from networkx import sigma
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from src.dataio.mapping import make_tuple_mapping, audit_mapping
from src.prune import *
from src.early_exit import *
from src.tools.utils import print_sweep_table  
from test import *
from src.core.infer import *
from src.core.multiLayerWNN import MultiLayerWNN, load_ckpt, save_ckpt
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


def train_g2(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    cfg: G2Config,
) -> nn.Module:
    model.to(device)

    # Freeze layer1
    for p in model.layers[0].parameters():
        p.requires_grad = False

    # Optionally freeze exit head
    if cfg.lr_exit <= 0:
        for p in model.exit1_classifier.parameters():
            p.requires_grad = False

    # Make sure layer2 + final are trainable
    for p in model.layers[1].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    params = []
    params_layer2 = [p for p in model.layers[1].parameters() if p.requires_grad]
    params_final = [p for p in model.classifier.parameters() if p.requires_grad]
    params.append({"params": params_layer2, "lr": cfg.lr_layer2, "weight_decay": cfg.weight_decay})
    params.append({"params": params_final,  "lr": cfg.lr_final,  "weight_decay": cfg.weight_decay})

    if cfg.lr_exit > 0:
        params_exit = [p for p in model.exit1_classifier.parameters() if p.requires_grad]
        params.append({"params": params_exit, "lr": cfg.lr_exit, "weight_decay": cfg.weight_decay})

    # ---- usage ----
    dump_trainable(model, "before-freeze")
    freeze_g2(model, freeze_exit=True)
    dump_trainable(model, "after-freeze")
    optimizer = torch.optim.AdamW(params)

    best_state = None
    best_overall = -1.0

    for ep in range(cfg.num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            final_logits, exit1_logits, exit_mask = model.forward_g2_with_mask(xb, thr=cfg.thr_eval)

            if (~exit_mask).any():
                loss_final = F.cross_entropy(final_logits[~exit_mask], yb[~exit_mask])
            else:
                loss_final = torch.zeros([], device=xb.device)
                continue

            loss = loss_final  # G2 不含 exit loss
            loss.backward()
            optimizer.step()

        # ---- eval ----
        tr_final = eval_final_acc(model, train_loader, device)
        va_final = eval_final_acc(model, val_loader, device)
        tr_exit = eval_exit1_acc(model, train_loader, device)
        va_exit = eval_exit1_acc(model, val_loader, device)
        metrics = eval_overall_at_thr(model, val_loader, device, thr=cfg.thr_eval)

        print(
            f"[G2] Ep{ep:03d} final_acc tr/va={tr_final*100:.2f}/{va_final*100:.2f} | "
            f"exit1_acc tr/va={tr_exit*100:.2f}/{va_exit*100:.2f} | "
            f"overall@thr={cfg.thr_eval} va={metrics['overall_acc']*100:.2f} "
            f"exit_rate={metrics['exit_rate']*100:.2f}"
        )

        if metrics["overall_acc"] > best_overall:
            best_overall = metrics["overall_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    print(f"[G2] Best val overall@thr={cfg.thr_eval}: {best_overall:.4f}")
    return model





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
    # load dataset
    print('data/model initialization...')
    input_path = '/workspace/WNN_early_exit/datasets'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


    total_size = len(x_train)
    val_size = int(0.1 * total_size)
    train_size = total_size - val_size
    seed = torch.Generator().manual_seed(42)

    train_ds, val_ds = random_split(
        TensorDataset(x_train, y_train),
        [train_size, val_size], 
        generator=seed
    )
    (x_train, y_train) = train_ds.dataset[train_ds.indices]
    (x_val, y_val) = val_ds.dataset[val_ds.indices]

    print('train_ds', train_ds)
    # CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = 32  # 16 / 32 / 64
    # oneshot: use the training dataset, calculate DT thresholds + normalization
    thresholds, xmin, xmax = compute_dt_thresholds(x_train, z=z)

    # Encode train / test
    x_train_bits = dt_thermometer_encode(x_train.to(device), thresholds, xmin, xmax)
    x_val_bits   = dt_thermometer_encode(x_val.to(device),   thresholds, xmin, xmax)
    x_test_bits   = dt_thermometer_encode(x_test.to(device),   thresholds, xmin, xmax)

    in_bits = x_train_bits.size(1)

    
    train_ds = TensorDataset(x_train_bits, y_train)
    val_ds = TensorDataset(x_val_bits, y_val)
    test_ds   = TensorDataset(x_test_bits, y_test)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False)
    val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader   = DataLoader(test_ds, batch_size=512, shuffle=False)


    '''# CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    z = 32  # 16 / 32 / 64
    # oneshot: use the training dataset, calculate DT thresholds + normalization
    thresholds, xmin, xmax = compute_dt_thresholds(x_train, z=z)

    # Encode train / test
    x_train_bits = dt_thermometer_encode(x_train.to(device), thresholds, xmin, xmax)
    x_test_bits   = dt_thermometer_encode(x_test.to(device),   thresholds, xmin, xmax)

    in_bits = x_train_bits.size(1)

    
    train_ds = TensorDataset(x_train_bits, y_train)
    total_size = len(train_ds)
    val_size = int(0.1 * total_size)
    train_size = total_size - val_size
    seed = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(
        train_ds, 
        [train_size, val_size], 
        generator=seed
    )
    
    test_ds   = TensorDataset(x_test_bits, y_test)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False)
    val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader   = DataLoader(test_ds, batch_size=512, shuffle=False)'''

    model, bb_cfg, ex_cfg, _ = load_ckpt("/Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_w_exit_g1_v1.pth", device)
    
    for p in model.parameters():
        print(p.shape, p.requires_grad)

    # baseline
    exit_loss, exit_acc = eval_exit1_epoch(model, test_loader, device)
    print(f"[G1 Exit head only] test_acc={exit_acc*100:.2f}%")
    final_acc = eval_final_acc(model, test_loader, device)
    print(f"[G1 Final head only] test_acc={final_acc*100:.2f}%")
    m = eval_overall_at_thr(model, test_loader, device, thr=2.0)
    print(f"[G1 Overall@thr=2.0] test_acc={m['overall_acc']*100:.2f}%, exit_rate={m['exit_rate']*100:.2f}%")

    
    g2_cfg = G2Config(
        num_epochs=15,
        lr_layer2=3e-4,
        lr_final=3e-4,
        lr_exit=0.0,
        lambda_exit=0.0,
        thr_eval=2.0,
        weight_decay=1e-3,
    )
    model = train_g2(model, train_loader, val_loader, device, g2_cfg)

    #print("Best val overall acc:", best["val_overall_acc"])
    save_ckpt("/Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_w_exit_g2_v1.pth", model, bb_cfg, ex_cfg)


    '''for thr in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5, 6.0]:
         print(f"--- Evaluate early exit with thr={thr} ---")
         avg_loss, acc, exit_rate = eval_epoch_w_exit(model, test_loader, device, thr=thr)
         print(f"thr={thr}, test_acc={acc*100:.2f}%, exit_rate={exit_rate*100:.2f}%")

    exit_loss, exit_acc = eval_exit1_epoch(model, test_loader, device)
    print(f"[Exit head only] test_acc={exit_acc*100:.2f}%")

    for thr in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5, 6.0]:
         print(f"--- Evaluate early exit with thr={thr} ---")
         avg_loss, acc, exit_rate, exited_acc, exited_class_histogram = eval_epoch_w_exit2(model, test_loader, device, thr=thr)
         print(f"thr={thr}, test_acc={acc*100:.2f}%, exit_rate={exit_rate*100:.2f}%, exited_acc={exited_acc*100:.2f}%")
         print(f"exited class histogram: {exited_class_histogram}")

    thr_list = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    all_metrics = []
    for thr in thr_list:
        m = eval_epoch_w_exit_metrics(model, test_loader, device, thr=thr, num_classes=10)
        all_metrics.append(m)

    print_sweep_table(all_metrics)

    # 如果你想看某個 thr 的 exited pred/true 分佈（例如最佳點 thr=1.5）
    best = all_metrics[3]
    print("pred_hist:", best["pred_hist"])
    print("true_hist:", best["true_hist"])'''

    thr_list = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    out = stage2_sweep_val_test(model, val_loader, test_loader, device, thr_list)

    exit_loss, exit_acc = eval_exit1_epoch(model, test_loader, device)
    print(f"[G2 Exit head only] test_acc={exit_acc*100:.2f}%")
    final_acc = eval_final_acc(model, test_loader, device)
    print(f"[G2 Final head only] test_acc={final_acc*100:.2f}%")
    m = eval_overall_at_thr(model, test_loader, device, thr=2.0)
    print(f"[G2 Overall@thr=2.0] test_acc={m['overall_acc']*100:.2f}%, exit_rate={m['exit_rate']*100:.2f}%")

    