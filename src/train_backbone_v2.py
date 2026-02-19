# src/train/train_wnn.py
import copy
from pathlib import Path
import argparse
import json
import torch
import torch.nn.functional as F
import torch.utils.data as d
from torch.utils.data import DataLoader, random_split
from src.dataio.mapping import make_tuple_mapping, audit_mapping
from src.dataio.data import build_loaders_bits
from src.early_exit import eval_exit1_epoch, eval_final_acc, eval_overall_at_thr
from src.prune import *
from src.tools.fpga_tools.fpga_export_utils import export_lut_init_files
from src.tools.lut_converage import lut_pattern_coverage
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

def train_model(
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay
    )

    '''# Cosine decay: lr 從 base_lr -> eta_min
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=eta_min
    )'''
    '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, threshold=1e-3, verbose=True
    )'''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, threshold=5e-4, verbose=True
    )

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(num_epochs):
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
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            #loss = F.cross_entropy(logits, yb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

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
        seed=3,
        z=32,
        device_for_encoding=device,
        shuffle_train=True,
    )

    backbone_cfg = dict(
        arch="MultiLayerWNN",
        in_bits=in_bits,
        num_classes=C,
        lut_input_size=3,
        hidden_luts=(3000, 1500, 800),
        tau=0.165,
        mapping=None,
        dropout_p=args.dropout_p,  # ✅ 新增：給 MultiLayerWNN 再往下傳
        dataset_meta=dict(name=ds_meta.name, z=ds_meta.z)
    )

    
    model = MultiLayerWNN(
        in_bits=backbone_cfg['in_bits'],
        num_classes=backbone_cfg['num_classes'],
        lut_input_size=backbone_cfg['lut_input_size'],
        hidden_luts=backbone_cfg['hidden_luts'],  # (2000, 1000)
        tau=backbone_cfg['tau'],               # Table 15 x 1/0.165 (~= 0.165)
    ).to(device)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model = train_model(model, train_loader, val_loader, device, num_epochs=60, base_lr=args.base_lr, weight_decay=args.weight_decay)
    
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
    
    