import json
import random
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

# --- PROJECT IMPORTS ---
from src.dataio.mapping import make_tuple_mapping
from src.core.multiLayerWNN import MultiLayerWNN
from src.dataio.encode import dt_thermometer_encode, compute_dt_thresholds
from src.prune import collect_hidden_activations, compute_importance_weighted, build_pruned_classifier
from test import eval_epoch

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "SEED": 42,
    "NUM_LUTS_FULL": 2500,    # Starting number of LUTs
    "PRUNE_RATIO": 0.1,       # Target ratio 
    "MIN_KEEP": 10,           # Minimum LUTs to keep 
    "ADDR_BITS": 6,           # Address bits per LUT
    "EXPORT_DIR": "src/exports/fpga_bundle",
    "EPOCHS_FULL": 15,
    "EPOCHS_FINE": 5,
    "BATCH_SIZE": 256
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def load_or_create_mapping(bit_len, num_luts, addr_bits, seed=42, save_path=None):
    if save_path: save_path = Path(save_path)
    
    if save_path and save_path.exists():
        try:
            mapping = json.loads(save_path.read_text())
            if len(mapping) == num_luts:
                print(f"Loaded existing mapping: {num_luts} LUTs")
                return mapping
        except: pass

    print(f"Generating mapping: {num_luts} LUTs, Seed {seed}")
    random.seed(seed); np.random.seed(seed)
    mapping = make_tuple_mapping(num_luts=num_luts, addr_bits=addr_bits, bit_len=bit_len, tiles=None, seed=seed)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f: json.dump(mapping, f)
    return mapping

def train_model(model, train_loader, val_loader, device, num_epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        _, val_acc = eval_epoch(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:02d}: Val Acc {val_acc*100:.2f}%")

    if best_state:
        model.load_state_dict(best_state)
    return model

# -----------------------------------------------------------------------------
# EXPORT LOGIC
# -----------------------------------------------------------------------------

def export_pruned_hardware(model, full_mapping, keep_idx, save_dir, x_test_bits, y_test, addr_bits=6):
    save_dir = Path(save_dir)
    luts_dir = save_dir / "luts"

    # Clean existing directory to prevent stale files from previous runs
    if luts_dir.exists():
        shutil.rmtree(luts_dir)
    luts_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Slice Model
    w_cls = model.classifier.weight.detach().cpu() 
    w_lut_full = model.layers[0].table.detach().cpu()
    
    keep_indices_list = keep_idx.cpu().long().numpy().tolist()
    
    # Slice Tables and Mapping
    w_lut_sliced = w_lut_full[keep_indices_list]
    mapping_sliced = [full_mapping[i] for i in keep_indices_list]
    
    # 2. Fuse (Weight * Sigmoid(Table))
    lut_act = torch.sigmoid(w_lut_sliced) 
    W_C = w_cls.t().unsqueeze(2) 
    W_L = lut_act.unsqueeze(1)   
    fused_float = (W_C * W_L).numpy() 
    
    # 3. Quantize
    w_min, w_max = fused_float.min(), fused_float.max()
    scale = 255.0 / (w_max - w_min + 1e-9)
    fused_int = ((fused_float - w_min) * scale).astype(np.int32)
    
    # 4. Verify Logic Consistency
    x_np = x_test_bits.cpu().numpy()
    y_np = y_test.cpu().numpy()
    num_samples = len(x_np)
    num_kept = len(mapping_sliced)
    
    powers = 2 ** np.arange(addr_bits - 1, -1, -1) 
    addresses = np.zeros((num_samples, num_kept), dtype=int)
    
    for i, wiring in enumerate(mapping_sliced):
        bits = x_np[:, wiring]
        addresses[:, i] = bits.dot(powers)
        
    scores = np.zeros((num_samples, 10), dtype=int)
    for i in range(num_kept):
        lut_T = fused_int[i].T 
        scores += lut_T[addresses[:, i]]
        
    acc = (scores.argmax(axis=1) == y_np).mean()
    print(f"Export Verification Accuracy: {acc*100:.2f}%")
    
    if acc < 0.80:
        raise ValueError("Export validation failed. Accuracy is too low.")
    
    # 5. Write Files
    with open(save_dir / "addr_bits_per_lut.mem", "w") as f:
        f.write("\n".join([f"{addr_bits:x}"] * num_kept))
        
    with open(save_dir / "kept_bits.mem", "w") as f:
        for wires in mapping_sliced:
            f.write(" ".join(f"{b:x}" for b in wires) + "\n")
            
    for i in range(num_kept):
        lut_data = fused_int[i].T 
        filename = luts_dir / f"lut_{i:03d}.mem"
        with open(filename, "w") as f:
            for row in lut_data:
                f.write(" ".join(f"{val:x}" for val in row) + "\n")

    print(f"Exported {num_kept} LUTs to {save_dir}")

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Initializing Data...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)])
    
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    x_tr = train_data.data.float().view(-1, 784)
    y_tr = train_data.targets
    x_te = test_data.data.float().view(-1, 784)
    y_te = test_data.targets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    thresh, xmin, xmax = compute_dt_thresholds(x_tr, z=32)
    x_tr_bits = dt_thermometer_encode(x_tr.to(device), thresh, xmin, xmax)
    x_te_bits = dt_thermometer_encode(x_te.to(device), thresh, xmin, xmax)
    
    tr_loader = DataLoader(TensorDataset(x_tr_bits, y_tr), batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    te_loader = DataLoader(TensorDataset(x_te_bits, y_te), batch_size=CONFIG["BATCH_SIZE"]*2, shuffle=False)

    random.seed(CONFIG["SEED"]); np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    
    # Create Full Mapping
    mapping_full = load_or_create_mapping(x_tr_bits.shape[1], CONFIG["NUM_LUTS_FULL"], CONFIG["ADDR_BITS"], seed=CONFIG["SEED"])
    
    # Init Model
    random.seed(CONFIG["SEED"]); np.random.seed(CONFIG["SEED"]); torch.manual_seed(CONFIG["SEED"])
    model = MultiLayerWNN(
        in_bits=x_tr_bits.shape[1], num_classes=10, lut_input_size=CONFIG["ADDR_BITS"],
        hidden_luts=(CONFIG["NUM_LUTS_FULL"],), mapping=mapping_full, tau=0.165
    ).to(device)

    print("Training Full Model...")
    model = train_model(model, tr_loader, te_loader, device, num_epochs=CONFIG["EPOCHS_FULL"])
    
    print(f"Pruning with ratio: {CONFIG['PRUNE_RATIO']}")
    H, Y = collect_hidden_activations(model, tr_loader, device)
    importance = compute_importance_weighted(H.to(device), model)
    
    keep_idx = build_pruned_classifier(model, importance, keep_ratio=CONFIG['PRUNE_RATIO'], min_keep=CONFIG["MIN_KEEP"])
    print(f"LUTs remaining: {keep_idx.numel()}")

    print("Finetuning...")
    for name, p in model.named_parameters():
        p.requires_grad = ("table" not in name)
        
    model = train_model(model, tr_loader, te_loader, device, num_epochs=CONFIG["EPOCHS_FINE"])
    
    export_pruned_hardware(
        model=model,
        full_mapping=mapping_full,
        keep_idx=keep_idx,
        save_dir=CONFIG["EXPORT_DIR"],
        x_test_bits=x_te_bits,
        y_test=y_te,
        addr_bits=CONFIG["ADDR_BITS"]
    )