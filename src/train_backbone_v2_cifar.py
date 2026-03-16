# src/train/train_wnn.py
import copy
from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
import torch.utils.data as d
from src.core.wisard import build_cifar10_layer0_mapping, hard_unique_stats
from src.dataio.mapping import make_tuple_mapping, audit_mapping
from src.dataio.data import build_loaders_bits
from src.early_exit import eval_exit1_epoch, eval_final_acc, eval_overall_at_thr
from src.prune import *
from src.tools.fpga_tools.fpga_export_utils import export_lut_init_files
from src.tools.lut_converage import lut_pattern_coverage
from src.tools.utils import debug_conn_idx
from test import *
from src.core.infer import *
from src.core.multiLayerWNN import MultiLayerWNN, save_best_checkpoint_atomic, save_ckpt
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

def set_model_dropout_p(model, p: float):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p


import torch

def build_optimizer(
    model, *,
    lr_table0=3e-4, lr_table1=3e-4, lr_conn1=3e-5, lr_cls=3e-4,
    wd_table0=1e-2, wd_table1=3e-2, wd_conn1=0.0, wd_cls=1e-3,
):
    # ---- sanity: required attrs ----
    assert hasattr(model, "layers") and len(model.layers) >= 2
    assert hasattr(model.layers[0], "table")
    assert hasattr(model.layers[1], "table")
    assert hasattr(model.layers[1], "learnable_conn")
    assert hasattr(model.layers[1].learnable_conn, "logits")
    assert hasattr(model, "classifier") and hasattr(model.classifier, "weight")

    p_table0 = model.layers[0].table
    p_table1 = model.layers[1].table
    p_conn1  = model.layers[1].learnable_conn.logits
    p_cls    = model.classifier.weight

    # ---- de-dup check ----
    ids = [id(p_table0), id(p_table1), id(p_conn1), id(p_cls)]
    assert len(set(ids)) == len(ids), "Duplicate parameter found across optimizer groups."

    param_groups = [
        {"params": [p_table0], "lr": lr_table0, "weight_decay": wd_table0},
        {"params": [p_table1], "lr": lr_table1, "weight_decay": wd_table1},
        {"params": [p_conn1],  "lr": lr_conn1,  "weight_decay": wd_conn1},
        {"params": [p_cls],    "lr": lr_cls,    "weight_decay": wd_cls},
    ]
    return torch.optim.AdamW(param_groups)

def train_model2(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=50,
    base_lr=3e-3,
    weight_decay=1e-3,
    eta_min=1e-4,          # cosine 最低 lr
    grad_clip=1.0,
    early_stop_patience=0, # 0 = 不 early stop；建議先設 8
):
    '''optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay
    )'''
    # assume model has model.layers = nn.ModuleList([...])
    table0 = model.layers[0].table
    table1 = model.layers[1].table

    # collect "rest" params by id exclusion (most robust)
    optimizer = build_optimizer(
        model,
        lr_table0=3e-4,
        lr_table1=3e-4,
        #lr_conn1=1e-3,      # 先保守
        lr_conn1=5e-4,
        lr_cls=3e-4,
        wd_table0=1e-2,
        wd_table1=1e-2,     # 你想衝也可以試 1e-1
        wd_conn1=0.0,
        wd_cls=1e-3,
    )


    '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, threshold=1e-3, verbose=True
    )'''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, threshold=5e-4, verbose=True
    )

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(num_epochs):
        lc._cached_w = None
        
        '''# example schedule
        if epoch < 5:
            p = 0.05
        elif epoch < 15:
            p = 0.10
        else:
            p = 0.15
        set_model_dropout_p(model, p)'''
        # ---- train one epoch ----
        model.train()
        lc = model.layers[1].learnable_conn
        
        lc.use_gumbel = True  # 確保 tau schedule 生效
        if epoch < 6:
            lc.gumbel_tau = 2.0
            ent_target = "high"   # maximize entropy
            lambda_ent = 0
            lc.conn_temp = 2.0
        elif epoch < 12:
            lc.gumbel_tau = 0.5
            ent_target = None     # no entropy reg
            lambda_ent = 0.0
            lc.conn_temp = 1.0
        '''else:
            lc.gumbel_tau = 0.7
            ent_target = "low"    # minimize entropy
            lambda_ent = 1e-3
            lc.conn_temp = 1.0'''

        with torch.no_grad():
            w = lc._weights()
            ent_dbg = -(w * (w.clamp_min(1e-12).log())).sum(dim=-1).mean().item()
        print(f'conn_temp: {lc.conn_temp}')
        print(f'logit std: {lc.logits.detach().std().item()}')
        print(f'effective_std: {lc.logits.detach().std().item()/lc.conn_temp}')
        print("ent_dbg(from _weights) =", ent_dbg, "entropy_loss(low) =", float(lc.entropy_loss("low")))

        '''if epoch < 6:
            lc.gumbel_tau = 2.0      # exploration / almost-uniform
            lambda_ent = 0.0            # 其實可以不加 high entropy
            ent_target = "high"
        elif epoch < 12:
            lc.gumbel_tau = 0.5      # start committing
            lambda_ent = 0.2            # push low entropy
            ent_target = None
        else:
            lc.gumbel_tau = 0.3      # commit harder
            lambda_ent = 0.2
            ent_target = "low"'''

        lambda_div = 0.0  # 先關掉，等 entropy 跑順再加回來（建議 1e-4 ~ 5e-4）
        print(f"[epoch {epoch}] tau={lc.gumbel_tau} ent_target={ent_target} lam_ent={lambda_ent} lam_div={lambda_div}")

        # ---- optional: print entropy stats before epoch ----
        with torch.no_grad():
            #w0 = torch.softmax(lc.logits, dim=-1)  # [L,k,M]
            w0 = lc._weights()
            ent0 = -(w0 * (w0.clamp_min(1e-12).log())).sum(dim=-1).mean()
            wmax0 = w0.max(dim=-1).values.mean()
            print(f"[pre] ent={ent0.item():.4f} w_max={wmax0.item():.4f} logits_std={lc.logits.std().item():.6f}")

        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)

            loss_ce = F.cross_entropy(logits, yb, label_smoothing=0.1)

            # entropy/div loss computed per-batch (but schedule fixed by ent_target/lambda_ent)
            '''if lambda_ent > 0:
                loss_ent = lc.entropy_loss(ent_target)
            else:
                loss_ent = logits.new_tensor(0.0)

            if lambda_div > 0:
                loss_div = lc.diversity_loss()
            else:
                loss_div = logits.new_tensor(0.0)'''
            if ent_target is not None and lambda_ent > 0:
                loss_ent = lc.entropy_loss(ent_target)  # ✅ uses cached_w from this forward
                loss = loss + lambda_ent * loss_ent



            '''if ent_target is None or lambda_ent == 0.0:
                loss_ent = torch.zeros((), device=device)
            else:
                loss_ent = lc.entropy_loss(target=ent_target)'''

            if lambda_div == 0.0:
                loss_div = torch.zeros((), device=device)
            else:
                loss_div = lc.diversity_loss()

            loss = loss_ce + lambda_ent * loss_ent + lambda_div * loss_div
            loss.backward()

            # ---- debug gradients occasionally ----
            if step % 50 == 0:
                g = lc.logits.grad
                if g is None:
                    print("[grad] logits.grad=None (check detach somewhere)")
                else:
                    print(f"[grad] mean={g.abs().mean().item():.2e} max={g.abs().max().item():.2e} "
                        f"loss_ce={loss_ce.item():.4f} loss_ent={loss_ent.item():.4f}")

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

        # ---- optional: print entropy stats after epoch ----
        with torch.no_grad():
            #w1 = torch.softmax(lc.logits, dim=-1)
            w1 = lc._weights()
            ent1 = -(w1 * (w1.clamp_min(1e-12).log())).sum(dim=-1).mean()
            wmax1 = w1.max(dim=-1).values.mean()
            # hard selection diversity
            u_slot, u_bit = hard_unique_stats(model.layers[1].learnable_conn)
            print("hard_unique slot/bit:", u_slot, u_bit)
            hard_sel = torch.argmax(lc.logits, dim=-1)  # [L,k]
            hard_unique = hard_sel.unique().numel()
            print(f"[post] ent={ent1.item():.4f} w_max={wmax1.item():.4f} "
                f"logits_std={lc.logits.std().item():.6f} hard_unique={hard_unique}")

        # ---- eval ----
        train_loss, train_acc = eval_epoch(model, train_loader, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        # ---- track best ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0

            best_val_acc = val_acc
            best_epoch = epoch

            save_best_checkpoint_atomic(
                path_out=args.path_out,   # 最終檔名就會永遠是 best 的
                model=model,
                best_val_acc=best_val_acc,
                epoch=epoch,
                optimizer=optimizer,      # 想要可 resume 才存
                scheduler=scheduler,
                extra={"dataset": args.dataset},
            )
            print(f"[BEST] epoch={epoch:03d} val_acc={val_acc*100:.2f}% -> saved")
        else:
            no_improve += 1



        # ---- step scheduler (cosine: 每 epoch step) ----
        #scheduler.step()
        scheduler.step(val_acc)

        # debug
        '''print("layer1 table std:", model.layers[1].table.data.std().item())
        print(f'layer1 topk hard: {model.layers[1].learnable_conn._topk_hard()[:2]}')'''
        

        # optional: print lr
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | "
              f"val_acc={val_acc*100:.2f}% | lr={cur_lr:.2e}")

        # ---- early stopping ----
        if early_stop_patience and no_improve >= early_stop_patience:
            print(f"[EarlyStop] no improvement for {early_stop_patience} epochs. "
                  f"best_val={best_val_acc*100:.2f}% @ epoch {best_epoch}")
            break

    # ---- load best ----
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


import copy
import torch
import torch.nn.functional as F

@torch.no_grad()
def _entropy_from_w(w: torch.Tensor) -> float:
    # w: [..., M]
    ent = -(w * w.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return float(ent.item())

@torch.no_grad()
def _w_max_from_w(w: torch.Tensor) -> float:
    # w: [..., M]
    return float(w.max(dim=-1).values.mean().item())

def save_best_fn(epoch, model, optimizer, scheduler, best_val_acc):
    save_best_checkpoint_atomic(
        path_out=args.path_out,
        model=model,
        best_val_acc=best_val_acc,
        epoch=epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        extra={"dataset": args.dataset},
    )

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=50,
    grad_clip=1.0,
    early_stop_patience=0,  # 0 = no early stop
    # --- optimizer hyperparams (keep your current defaults) ---
    lr_table0=3e-4,
    lr_table1=3e-4,
    lr_conn1=5e-4,
    lr_cls=3e-4,
    wd_table0=1e-2,
    wd_table1=1e-2,
    wd_conn1=0.0,
    wd_cls=1e-3,
    # --- loss/reg ---
    label_smoothing=0.1,
    lambda_div=0.0,
    # --- scheduler ---
    plateau_factor=0.5,
    plateau_patience=10,
    plateau_threshold=5e-4,
    # --- checkpoint ---
    save_best_fn=None,  # function(epoch, model, optimizer, scheduler, best_val_acc)
    # --- debug ---
    debug_every_steps=50,
    debug_first_batch_only=True,
):
    """
    Assumptions:
      - model.layers[1] is WNNLUTLayerSoftConn
      - model.layers[1].learnable_conn is LearnableConnSlots
      - learnable_conn caches w in forward() when training, and entropy_loss()/diversity_loss() use cached_w
      - eval_epoch(model, loader, device) exists and calls model.eval()
    """

    model = model.to(device)

    optimizer = build_optimizer(
        model,
        lr_table0=lr_table0,
        lr_table1=lr_table1,
        lr_conn1=lr_conn1,
        lr_cls=lr_cls,
        wd_table0=wd_table0,
        wd_table1=wd_table1,
        wd_conn1=wd_conn1,
        wd_cls=wd_cls,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=plateau_factor,
        patience=plateau_patience,
        threshold=plateau_threshold,
        verbose=True,
    )

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0

    # handy refs
    lc = model.layers[1].learnable_conn

    for epoch in range(num_epochs):
        # -------------------------
        # 0) epoch-level schedule
        # -------------------------
        model.train()
        lc.use_gumbel = True  # ensure schedule takes effect

        # schedule you are using now (you can tweak later)
        if epoch < 6:
            lc.gumbel_tau = 2.0
            lc.conn_temp = 2.0
            ent_target = "high"
            lambda_ent = 0.0
        elif epoch < 12:
            lc.gumbel_tau = 0.5
            lc.conn_temp = 1.0
            ent_target = None
            lambda_ent = 0.0
        '''else:
            lc.gumbel_tau = 0.3
            lc.conn_temp = 0.5
            ent_target = "low"
            lambda_ent = 1e-2'''

        print(f"[epoch {epoch}] gumbel_tau={lc.gumbel_tau} conn_temp={lc.conn_temp} "
              f"ent_target={ent_target} lam_ent={lambda_ent} lam_div={lambda_div}")

        # -------------------------
        # 1) pre-epoch debug (use SAME mechanism as training)
        # -------------------------
        with torch.no_grad():
            # IMPORTANT: don't call lc._weights() here (it recomputes).
            # We just show logits dispersion and the theoretical entropy if softmax(logits/conn_temp).
            logits_eff = lc.logits / max(lc.conn_temp, 1e-6)
            w_pre = torch.softmax(logits_eff, dim=-1)
            ent_pre = _entropy_from_w(w_pre)
            wmax_pre = _w_max_from_w(w_pre)
            print(f"[pre] ent={ent_pre:.4f} w_max={wmax_pre:.4f} "
                  f"logits_std={float(lc.logits.std().item()):.6f} eff_std={float(logits_eff.std().item()):.6f}")

        # -------------------------
        # 2) train one epoch
        # -------------------------
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # reset cache so entropy_loss uses THIS forward's w
            lc._cached_w = None

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss_ce = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)

            loss = loss_ce
            loss_ent = logits.new_tensor(0.0)
            loss_div = logits.new_tensor(0.0)

            if ent_target is not None and lambda_ent > 0:
                loss_ent = lc.entropy_loss(ent_target)
                loss = loss + lambda_ent * loss_ent

            if lambda_div > 0:
                loss_div = lc.diversity_loss()
                loss = loss + lambda_div * loss_div

            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            # stats
            running_loss += float(loss_ce.item()) * yb.numel()
            preds = logits.argmax(dim=1)
            running_correct += int((preds == yb).sum().item())
            running_total += int(yb.numel())

            # debug (optional)
            if (step % debug_every_steps == 0) and (not debug_first_batch_only or step == 0):
                g = lc.logits.grad
                g_mean = float(g.abs().mean().item()) if g is not None else float("nan")
                g_max = float(g.abs().max().item()) if g is not None else float("nan")

                with torch.no_grad():
                    # cached_w should exist because forward was called and lc.training==True
                    w = lc.get_cached_w()
                    ent_dbg = _entropy_from_w(w)
                    wmax_dbg = _w_max_from_w(w)

                print(f"[step {step:04d}] loss_ce={loss_ce.item():.4f} "
                      f"loss_ent={float(loss_ent.item()):.4f} loss_div={float(loss_div.item()):.4f} "
                      f"grad_mean={g_mean:.2e} grad_max={g_max:.2e} ent={ent_dbg:.4f} w_max={wmax_dbg:.4f}")

        train_acc_fast = running_correct / max(running_total, 1)
        train_loss_fast = running_loss / max(running_total, 1)

        # -------------------------
        # 3) eval (uses your eval_epoch)
        # -------------------------
        train_loss, train_acc = eval_epoch(model, train_loader, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        # keep both: fast train stats (from CE only) and eval stats
        print(f"[epoch {epoch:03d}] train_fast loss={train_loss_fast:.4f} acc={train_acc_fast*100:.2f}% | "
              f"eval train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}%")

        # -------------------------
        # 4) best checkpoint
        # -------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0

            if save_best_fn is not None:
                save_best_fn(epoch, model, optimizer, scheduler, best_val_acc)

            print(f"[BEST] epoch={epoch:03d} val_acc={best_val_acc*100:.2f}%")
        else:
            no_improve += 1

        # -------------------------
        # 5) scheduler + early stop
        # -------------------------
        scheduler.step(val_acc)

        # print current lr (group 0)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"lr(group0)={cur_lr:.2e}")

        if early_stop_patience and no_improve >= early_stop_patience:
            print(f"[EarlyStop] no improvement for {early_stop_patience} epochs. "
                  f"best_val={best_val_acc*100:.2f}% @ epoch {best_epoch}")
            break

    # -------------------------
    # 6) load best
    # -------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    return model #, {"best_val_acc": best_val_acc, "best_epoch": best_epoch}



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


def export_wnn_for_fpga(model, path: str, quant_bits: int = None):
    """
    pack the MultiLayerWNNLUT and connection structure into a .npz file,
    in order to facilitate parsing and initialization on FPGA
    If quant_bits is not None (e.g., 8, 16), simple symmetric quantization is performed:
      table_q = round(table * scale), and scale is saved together.
    """
    model_cpu = model.cpu()
    layers = list(model_cpu.layers)
    num_layers = len(layers)

    export_data = {}
    export_data["num_layers"] = num_layers
    export_data["input_bits"] = layers[0].in_bits
    export_data["num_classes"] = model_cpu.classifier.out_features

    # classifier
    W_cls = model_cpu.classifier.weight.detach().numpy().astype(np.float32)
    export_data["classifier_weight"] = W_cls  # shape [C, H_last]

    # per layer
    for l, layer in enumerate(layers):
        prefix = f"layer{l}_"
        export_data[prefix + "in_bits"] = int(layer.in_bits)
        export_data[prefix + "num_luts"] = int(layer.num_luts)
        export_data[prefix + "lut_input_size"] = int(layer.lut_input_size)

        conn = layer.conn_idx.detach().cpu().numpy().astype(np.int32)  # [num_luts, k]
        export_data[prefix + "conn_idx"] = conn

        table = layer.table.detach().cpu().numpy().astype(np.float32)  # [num_luts, 2^k]

        if quant_bits is not None:
            qmax = 2 ** (quant_bits - 1) - 1
            max_abs = np.max(np.abs(table)) + 1e-8
            scale = qmax / max_abs
            table_q = np.round(table * scale).astype(np.int16)
            export_data[prefix + "table_q"] = table_q
            export_data[prefix + "table_scale"] = np.float32(1.0 / scale)
        else:
            export_data[prefix + "table"] = table

    np.savez_compressed(path, **export_data)
    print(f"[export_wnn_for_fpga] Saved WNN config to {path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST",
                        help="Dataset name (currently only MNIST is wired)")
    parser.add_argument("--path_out", type=str, required=True,
                        help="Path to save the backbone checkpoint")
    parser.add_argument("--base_lr", type=float, default=1e-3, help="Base learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for training")
    parser.add_argument("--dropout_p",type=float, default="0")
    # NOTE: backbone training does not need --k; keep for now if you want
    # parser.add_argument("--k", type=int, default=1024)

    args = parser.parse_args()

    # CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #mnist: z=32, lut_input_size=3, base_lr=3e-3, weight_decay=1e-3
    #kmnist: z=64, lut_input_size=2, base_lr=1e-2, weight_decay=1e-3

    train_loader, val_loader, test_loader, in_bits, C, ds_meta = build_loaders_bits(
        dataset=args.dataset,
        root="/Users/yi-chunchen/workspace/WNN_early_exit/datasets/",   # 你現在 datasets 根目錄
        batch_size_train=256,
        batch_size_eval=512,
        val_ratio=0.1,
        seed=42,
        z=32,
        device_for_encoding=device,
        shuffle_train=True,
    )


    #把 conn0 做成「分 bucket 抽樣」+「去重」
    #sobel bits 不要用 “global threshold ratio”，改成 “per-image adaptive”
    

    lut_input_size = 9  # 3 for MNIST, 2 for KMNIST; you can also treat it as a tunable hyperparameter and try different values
    hidden_luts = [6000, 4000]  # 可以先不動，等後續 pruned backbone 再調整

    print(ds_meta)
    '''# CIFAR10: C=3, H=W=32
    first_layer_mapping = build_cifar10_layer0_mapping(
        num_luts=hidden_luts[0],
        k=lut_input_size,
        z=ds_meta.z,
        C=ds_meta.channels,
        seed=42,
        device="cpu",
    )'''

    backbone_cfg = dict(
        arch="MultiLayerWNN",
        in_bits=in_bits,
        num_classes=C,
        lut_input_size=lut_input_size,
        lut_input_size_list=[9, 5],
        hidden_luts=hidden_luts,
        tau=0.165,
        #tau=0.5,
        mapping=None,
        #mapping=first_layer_mapping,
        dropout_p=args.dropout_p,  # ✅ 新增：給 MultiLayerWNN 再往下傳
        #dataset_meta=dict(name=ds_meta.name, z=ds_meta.z)
    )


    
    model = MultiLayerWNN(
        in_bits=backbone_cfg['in_bits'],
        num_classes=backbone_cfg['num_classes'],
        lut_input_size=backbone_cfg['lut_input_size'],
        lut_input_size_list=backbone_cfg['lut_input_size_list'],
        hidden_luts=backbone_cfg['hidden_luts'],  # (2000, 1000)
        tau=backbone_cfg['tau'],               # Table 15 x 1/0.165 (~= 0.165)
        mapping=backbone_cfg['mapping'],       # [first_layer_mapping, None]
    ).to(device)

    debug_conn_idx(model.layers[0].conn_idx, model.layers[0].in_bits, name="layer0.conn_idx")
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("bb_cfg in_bits:", backbone_cfg["in_bits"])
    print("layer0 in_bits:", model.layers[0].in_bits)
    print("layer0 conn min/max:", model.layers[0].conn_idx.min().item(), model.layers[0].conn_idx.max().item())
    print("layer0 conn negative count:", (model.layers[0].conn_idx < 0).sum().item())
    uniq = torch.unique(model.layers[0].conn_idx).numel()
    print("coverage:", uniq / float(in_bits))

    for name, param in model.named_parameters():
        print(name)
        

    print(type(model.layers[1].learnable_conn))
    #model = train_model(model, train_loader, val_loader, device, num_epochs=50, base_lr=args.base_lr, weight_decay=args.weight_decay)
    model = train_model(model, train_loader, val_loader, device, num_epochs=50, save_best_fn=save_best_fn)
    
    '''rep = lut_pattern_coverage(
        model.layers[0],
        train_loader,
        device,
        num_luts_sample=256,
        max_batches=200,   # 先快速跑，之後拿掉看全量
    )
    print(rep)
    print("coverage mean:", rep.coverage_mean, "p10/p50/p90:", rep.coverage_p10, rep.coverage_p50, rep.coverage_p90)
    print("entropy mean:", rep.entropy_mean, "maxbin_ratio mean:", rep.maxbin_ratio_mean, "gini mean:", rep.gini_mean)'''

    
    #save_ckpt("/Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_unpruned_v1.pth", model, backbone_cfg, exit_config=None, extra={"dataset": "MNIST"})
    save_ckpt(
        args.path_out,
        model,
        backbone_cfg,
        
        exit_config=None,
        extra = {
            "dataset": args.dataset,
            "seed": 42,
            "val_ratio": 0.1,
        }
    )

    # evaluation
    train_loss_before, train_acc_before = eval_epoch(model, train_loader, device)
    val_loss_before, val_acc_before = eval_epoch(model, val_loader, device)
    test_loss_before,  test_acc_before  = eval_epoch(model, test_loader,  device)

    print(f"[Backbone] train_acc={train_acc_before*100:.2f}% |"
          f"val_acc={val_acc_before*100:.2f}% | "
        f"test_acc={test_acc_before*100:.2f}%")
    
    