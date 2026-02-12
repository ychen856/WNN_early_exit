# src/train/train_wnn.py
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import torch.utils.data as d
from torch.utils.data import DataLoader, random_split
from src.dataio.mapping import make_tuple_mapping, audit_mapping
from src.early_exit import eval_exit1_epoch, eval_final_acc, eval_overall_at_thr
from src.prune import *
from src.tools.fpga_tools.fpga_export_utils import export_lut_init_files
from test import *
from src.core.infer import *
from src.core.multiLayerWNN import MultiLayerWNN, save_ckpt
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


def train_model(model, train_loader, val_loader, device,
                num_epochs=50, base_lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    best_state = None
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ---- train one epoch ----
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ---- eval on train / val using *same* function ----
        train_loss, train_acc = eval_epoch(model, train_loader, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"train_acc={train_acc*100:.2f}% | val_acc={val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

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
    

'''def save_backbone(path, model, backbone_cfg: dict, extra: dict = None):
    ckpt = {
        "format_version": 1,
        "backbone": backbone_cfg,
        "model_state": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(ckpt, path)'''


'''def load_backbone(path, device):
    ckpt = torch.load(path, map_location=device)

    # backward-compat: if it's a pure state_dict, throw a helpful error
    if "config" not in ckpt or "model_state" not in ckpt:
        raise ValueError(
            "This checkpoint has no config. Please re-save backbone with save_backbone(...)."
        )

    cfg = ckpt["config"]
    model = MultiLayerWNN(
        in_bits=cfg["in_bits"],
        num_classes=cfg["num_classes"],
        lut_input_size=cfg["lut_input_size"],
        hidden_luts=tuple(cfg["hidden_luts"]),
        mapping=cfg.get("mapping", None),
        tau=float(cfg.get("tau", 1.0)),
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=True)
    if len(missing) or len(unexpected):
        # strict=True normally makes these empty; this is extra paranoia
        print("[load_backbone] missing:", missing)
        print("[load_backbone] unexpected:", unexpected)

    return model, cfg, ckpt.get("extra", {})'''

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
    val_ds   = TensorDataset(x_test_bits, y_test)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader   = DataLoader(val_ds, batch_size=512, shuffle=False)'''


    C = 10

    backbone_cfg = dict(
        arch="MultiLayerWNN",
        in_bits=in_bits,
        num_classes=10,
        lut_input_size=6,
        hidden_luts=(2000, 1000),
        tau=0.165,
        mapping=None,          # 你如果有第一層 mapping 就放這裡（或放 conn_idx 也行）
    )

    model = MultiLayerWNN(
        in_bits=backbone_cfg['in_bits'],
        num_classes=backbone_cfg['num_classes'],
        lut_input_size=backbone_cfg['lut_input_size'],
        hidden_luts=backbone_cfg['hidden_luts'],  # (2000, 1000)
        tau=backbone_cfg['tau'],               # Table 15 x 1/0.165 (~= 0.165)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model = train_model(model, train_loader, val_loader, device, num_epochs=30, base_lr=1e-3)
    save_ckpt("/Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_unpruned_v1.pth", model, backbone_cfg, exit_config=None, extra={"dataset": "MNIST"})
    

    # evaluation
    train_loss_before, train_acc_before = eval_epoch(model, train_loader, device)
    val_loss_before, val_acc_before = eval_epoch(model, val_loader, device)
    test_loss_before,  test_acc_before  = eval_epoch(model, test_loader,  device)

    print(f"[Backbone] train_acc={train_acc_before*100:.2f}% |"
          f"val_acc={val_acc_before*100:.2f}% | "
        f"test_acc={test_acc_before*100:.2f}%")
    