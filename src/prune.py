from collections import defaultdict
from typing import List, Optional, Sequence, Tuple, Dict
from src.tools.utils import _assert_power_of_two
from src.dataio.encode import bucket_mapper_mnist_thermo
from src.core.multiLayerWNN import MultiLayerWNN
from src.core.wnnLutLayer import WNNLUTLayer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from test.eval import eval_epoch


def make_keep_mask(w_lut, keep_ratio):
    L = w_lut.shape[0]
    order = np.argsort(-w_lut)  # desc
    k = max(1, int(round(L * keep_ratio)))
    keep_ids = order[:k]
    mask = np.zeros(L, dtype=np.float32)
    mask[keep_ids] = 1.0
    return mask, keep_ids.tolist()  # shape (L,)

def eval_masked(model, X_bits, y, lut_mask, alpha: float = 1.0):
    correct = 0
    for i in range(X_bits.shape[0]):
        pred = predict_masked(model, X_bits[i], lut_mask, alpha=alpha)
        if pred == int(y[i]):
            correct += 1
    return correct / X_bits.shape[0]


def predict_masked(model, bit_vec, lut_mask, alpha: float = 1.0):
    """
    baseline log-ratio scoring，only use the LUT's lut_mask==1。
    lut_mask: shape (L,), values in {0.0, 1.0}
    """
    C = model.num_classes
    L = model.num_luts_per_class
    assert lut_mask.shape[0] == L

    addr = model._addresses_for_sample(bit_vec)  # (L,)

    votes = model.table[
        np.arange(C)[:, None],
        np.arange(L)[None, :],
        addr[None, :]
    ].astype(np.float32)  # (C,L)

    denom = votes.sum(axis=0, keepdims=True) + C * alpha  # (1,L)
    post = (votes + alpha) / denom  # (C,L)
    log_post = np.log(post + 1e-9)  # (C,L)

    # set contribution of the pruned LUT to 0
    masked_log = log_post * lut_mask[None, :]  # (C,L)

    scores = masked_log.sum(axis=1)  # (C,)
    return int(np.argmax(scores))

def drop_luts_by_priority(profile, lut_priority, keep_ratio):
    keep = max(1, int(round(len(profile["lut_tables"]) * keep_ratio)))
    order = np.argsort(-lut_priority)[:keep]  # keep first k

    lut_tables = [profile["lut_tables"][i] for i in order]
    kept_bits  = [profile["kept_global_bits_per_lut"][i] for i in order]
    m_list     = [profile["addr_bits_per_lut"][i] for i in order]
    return dict(
        lut_tables=lut_tables,
        kept_global_bits_per_lut=kept_bits,
        addr_bits_per_lut=m_list,
        num_classes=profile["num_classes"],
        alpha=profile["alpha"],
        bit_order="lsb",
    )


def select_top_luts_by_priority(lut_priority: np.ndarray, keep_ratio: float) -> np.ndarray:
    """based on LUT's priority, keep top-K%"""
    L = len(lut_priority)
    k = max(1, int(round(L * keep_ratio)))
    order = np.argsort(-lut_priority)[:k]
    return np.sort(order)

def prune_model_by_lut_indices(model, tuple_mapping: List[List[int]], keep_ids: np.ndarray):
    """prune original model and mapping table"""
    keep_ids = np.asarray(keep_ids, dtype=int)
    new_table = model.table[:, keep_ids, :].copy()        # (C, L_kept, 2^n)
    new_mapping = [tuple_mapping[i] for i in keep_ids]    # length L_kept
    return new_table, new_mapping

def drop_profile_to_luts(profile: Dict, keep_ids: np.ndarray) -> Dict:
    """from profile（bit-pruned） do the furthur LUT pruning。"""
    keep_ids = np.asarray(keep_ids, dtype=int)
    lut_tables = [profile["lut_tables"][i] for i in keep_ids]
    kept_bits  = [profile["kept_global_bits_per_lut"][i] for i in keep_ids]
    m_list     = [profile["addr_bits_per_lut"][i] for i in keep_ids]
    return dict(
        lut_tables=lut_tables,
        kept_global_bits_per_lut=kept_bits,
        addr_bits_per_lut=m_list,
        num_classes=profile["num_classes"],
        alpha=profile["alpha"],
        bit_order=profile.get("bit_order","lsb"),
        meta=[profile.get("meta", [{}]*len(profile["lut_tables"]))[i] for i in keep_ids]
    )

def _pick_local_bits(
    mode: str,
    lut_bits: List[int],
    bit_priority: np.ndarray,
    k: int,
    coverage_r: int,
    bucket_mapper,
    coverage_k_threshold: int,
) -> List[int]:
    if mode == "hard":
        return select_local_bits_with_coverage(
            lut_bits, bit_priority, k,
            r_per_bucket=coverage_r, bucket_mapper=bucket_mapper
        )
    else:
        return select_local_bits_soft_coverage(
            lut_bits, bit_priority, k,
            r_per_bucket_when_small=coverage_r,
            bucket_mapper=bucket_mapper,
            coverage_k_threshold=coverage_k_threshold
        )



def compute_bit_priority_entropy(X_bits: np.ndarray, y: np.ndarray, num_classes: int, eps: float = 1e-9):
    """
    X_bits: (N, B) 0/1 bit matrix
    y:      (N,)   label 0..C-1
    num_classes: C
    return:
      bit_priority: (B,) larger -> more important (priority = -entropy)
    """
    N, B = X_bits.shape
    C = num_classes

    y = np.array(y)
    X_bits = np.array(X_bits)

    # class × bit
    ones_cb = np.zeros((C, B), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        if np.any(idx):
            ones_cb[c] = X_bits[idx].sum(axis=0)

    # #samples
    class_cnt = np.array([np.sum(y == c) for c in range(C)], dtype=np.float64) + eps  # (C,)
    # p(bit=1 | class=c)
    p1_cb = ones_cb / class_cnt[:, None]  # (C,B)

    # w_c ∝ p1_cb[c,b] * P(class=c)
    p_class = class_cnt / class_cnt.sum()
    w_cb = p1_cb * p_class[:, None]           # (C,B)
    w_sum = w_cb.sum(axis=0, keepdims=True) + eps
    p_class_given_bit = w_cb / w_sum          # (C,B)

    # entropy over classes for each bit
    entropy_b = -np.sum(p_class_given_bit * np.log(p_class_given_bit + eps), axis=0)  # (B,)
    priority_b = -entropy_b.astype(np.float32)
    return priority_b


# ---------------------------
# Core: reduce one LUT by kept local positions
# ---------------------------
def _reduce_counts_for_lut(counts_c_a: np.ndarray,
                           keep_positions_in_lut: List[int],
                           n_addr_bits: int) -> Tuple[np.ndarray, List[int]]:
    """
    counts_c_a: (C, 2^n)
    keep_positions_in_lut
    return:
      reduced: (C, 2^m)
      kept_local_pos_sorted: List[int], asc
    """
    counts_c_a = np.asarray(counts_c_a)
    C, A = counts_c_a.shape
    _assert_power_of_two(A)
    n_from_A = A.bit_length() - 1
    if n_addr_bits != n_from_A:
        n_addr_bits = n_from_A

    keep_positions = sorted(set(int(p) for p in keep_positions_in_lut))
    if any(p < 0 or p >= n_addr_bits for p in keep_positions):
        raise ValueError(f"keep_positions {keep_positions} out of range [0,{n_addr_bits-1}]")

    drop_positions = [p for p in range(n_addr_bits) if p not in keep_positions]

    # (C, 2^n) -> (C, [2]*n)
    arr = counts_c_a.reshape(C, *([2] * n_addr_bits))
    if drop_positions:
        axes_to_sum = tuple(1 + p for p in sorted(drop_positions))
        arr = arr.sum(axis=axes_to_sum, keepdims=False)

    reduced = arr.reshape(C, -1)  # (C, 2^m)
    return reduced, keep_positions


# ------------------  per-LUT select top-k by coverage per LUT ------------------
def select_local_bits_with_coverage(
    lut_bits: List[int],
    bit_priority: np.ndarray,
    k: int,
    r_per_bucket: int = 1,
    bucket_mapper=bucket_mapper_mnist_thermo
) -> List[int]:
    """
    Always return k local positions.
    """
    n = len(lut_bits)
    k = max(1, min(k, n))  # k <= n

    # order by bit_priority
    local_scored = [(p, float(bit_priority[lut_bits[p]])) for p in range(n)]
    local_scored.sort(key=lambda x: x[1], reverse=True)

    # 分 bucket
    buckets = defaultdict(list)
    for p, sc in local_scored:
        name = bucket_mapper(lut_bits[p])
        buckets[name].append((p, sc))

    keep = []

    # 先滿足 coverage bucket
    if r_per_bucket > 0:
        for name, items in buckets.items():
            take = min(r_per_bucket, len(items))
            keep.extend([pos for pos, _ in items[:take]])

    # if less than k, use the remaining highest bit_priority to fill
    if len(keep) < k:
        taken = set(keep)
        remaining = [it for sub in buckets.values() for it in sub]
        remaining.sort(key=lambda x: x[1], reverse=True)
        for pos, _ in remaining:
            if pos not in taken:
                keep.append(pos)
                taken.add(pos)
                if len(keep) == k:
                    break

    # if still less than k (rare), fill with any index
    if len(keep) < k:
        all_pos = list(range(n))
        for pos in all_pos:
            if pos not in keep:
                keep.append(pos)
                if len(keep) == k:
                    break

    # finally sort
    keep = sorted(keep[:k])
    return keep


def select_local_bits_with_coverage_temp(lut_bits: List[int],
                                    bit_priority: np.ndarray,
                                    k: int,
                                    r_per_bucket: int = 1,
                                    bucket_mapper=bucket_mapper_mnist_thermo) -> List[int]:
    """
    return：local addr index
    """
    n = len(lut_bits)
    k = max(1, min(k, n))

    # order by bit_priority
    local_scored = [(p, float(bit_priority[lut_bits[p]])) for p in range(n)]
    local_scored.sort(key=lambda x: x[1], reverse=True)

    # distribution
    buckets = defaultdict(list)  # name -> list[(pos, score)]
    for p, sc in local_scored:
        name = bucket_mapper(lut_bits[p])
        buckets[name].append((p, sc))

    # satisfy the coverage
    keep = []
    if r_per_bucket > 0:
        for name, items in buckets.items():
            take = min(r_per_bucket, len(items))
            keep.extend([pos for pos, _ in items[:take]])

    # make sure the #keeps reach k
    if len(keep) < k:
        taken = set(keep)
        remaining = [it for sub in buckets.values() for it in sub]
        remaining.sort(key=lambda x: x[1], reverse=True)
        for pos, _ in remaining:
            if pos not in taken:
                keep.append(pos); taken.add(pos)
                if len(keep) == k: break

    keep = sorted(set(keep))
    if len(keep) == 0:
        keep = [local_scored[0][0]]
    return keep



def profile_stats(profile, n_full):
    # n_full = the bitwidth during training
    m_list = profile["addr_bits_per_lut"]          # each LUT's m_l
    L_kept = len(m_list)
    avg_m = sum(m_list) / L_kept
    entries = sum(1<<m for m in m_list)
    entries_full = L_kept * (1<<n_full)
    compression = entries / entries_full           # < 1.0 small is the better
    return dict(L=L_kept, avg_m=avg_m, entries=entries,
                compression=compression)

def select_local_bits_soft_coverage(
    lut_bits: List[int],
    bit_priority: np.ndarray,
    k: int,
    *,
    r_per_bucket_when_small: int = 1,
    bucket_mapper=bucket_mapper_mnist_thermo,
    coverage_k_threshold: int = 4
) -> List[int]:
    """
    Soft coverage:
      - If k <= coverage_k_threshold: use coverage (per bucket keep r).
      - Else: pure per-LUT top-k by bit_priority (no coverage).
    """
    n = len(lut_bits)
    k = max(1, min(k, n))
    # rank locally by bit_priority
    local_scored = [(p, float(bit_priority[lut_bits[p]])) for p in range(n)]
    local_scored.sort(key=lambda x: x[1], reverse=True)

    if k <= coverage_k_threshold and r_per_bucket_when_small > 0:
        # coverage phase (same as your hard version)
        from collections import defaultdict
        buckets = defaultdict(list)
        for p, sc in local_scored:
            buckets[bucket_mapper(lut_bits[p])].append((p, sc))

        keep = []
        for _, items in buckets.items():
            items.sort(key=lambda x: x[1], reverse=True)
            take = min(r_per_bucket_when_small, len(items))
            keep.extend([pos for pos, _ in items[:take]])

        if len(keep) < k:
            taken = set(keep)
            remaining = [it for sub in buckets.values() for it in sub]
            remaining.sort(key=lambda x: x[1], reverse=True)
            for pos, _ in remaining:
                if pos not in taken:
                    keep.append(pos); taken.add(pos)
                    if len(keep) == k: break
        keep = sorted(set(keep))
        if not keep:
            keep = [local_scored[0][0]]
        return keep
    else:
        # non-coverage phase (free to compress)
        return sorted([p for p, _ in local_scored[:k]])
    


##############################
#
#####################3########
@torch.no_grad()
def collect_hidden_activations(model, data_loader, device):
    """
    collect the last hidden h_last
    return:
      H: [N, H]  (put all the sample into a table)
      Y: [N]     (corresponding label)
    """
    model.eval()
    all_h = []
    all_y = []

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits, h_last = model(xb, return_hidden=True)  # h_last: [B, H]
        all_h.append(h_last.cpu())
        all_y.append(yb.cpu())

    H = torch.cat(all_h, dim=0)  # [N, H]
    Y = torch.cat(all_y, dim=0)  # [N]
    return H, Y


def compute_importance_weighted(H: torch.Tensor, model: nn.Module):
    """
    H: [N, H] last hidden's activation
    model: pretrained WNN (includes classifier)

    importance_j = std(h_j) * sum_c |W[c,j]|
    """
    with torch.no_grad():
        std = H.std(dim=0)                    # [H]
        W = model.classifier.weight.data      # [C, H]
        w_abs = W.abs().sum(dim=0)            # [H]
        importance = std * w_abs
    return importance

def build_pruned_classifier(model, importance, keep_ratio=0.5, min_keep=64):
    """
    according to the importance, select the keeped hidden dimension
    create classifier, and put the keep_idx back model
    """
    device = next(model.parameters()).device
    H = importance.numel()

    keep_dim = max(min_keep, int(H * keep_ratio))
    keep_dim = min(keep_dim, H)

    _, idx = torch.topk(importance, k=keep_dim, largest=True, sorted=True)
    keep_idx = idx.to(device)

    old_W = model.classifier.weight.data  # [C, H]
    W_pruned = old_W[:, keep_idx]        # [C, keep_dim]

    num_classes = old_W.size(0)
    new_classifier = nn.Linear(keep_dim, num_classes, bias=False).to(device)
    new_classifier.weight.data.copy_(W_pruned)

    model.classifier = new_classifier
    model.keep_idx = keep_idx    # forward check

    return keep_idx

@torch.no_grad()
def collect_layer_activations(
    model: MultiLayerWNN,
    data_loader,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    collect each layer's hidden activation.
    return:
      layer_H: list of length L
        - layer_H[l]: [N, num_luts_l]
    """
    model.eval()
    num_layers = len(model.layers)
    layer_buffers = [[] for _ in range(num_layers)]

    for xb, yb in data_loader:
        xb = xb.to(device)
        logits, h_list = model.forward_with_all_hidden(xb)
        for l, h in enumerate(h_list):
            layer_buffers[l].append(h.detach().cpu())

    layer_H = [torch.cat(buf, dim=0) for buf in layer_buffers]
    return layer_H


def lut_importance_from_activations(H_l: torch.Tensor) -> torch.Tensor:
    """
    H_l: [N, H_l]
    importance_j = std(h_j)
    """
    return H_l.std(dim=0)  # [H_l]


def plan_lut_pruning_for_layer(
    H_l: torch.Tensor,
    keep_ratio: float = 0.9,
    min_keep: int = 32,
) -> torch.Tensor:
    """
    Decide which LUTs to keep based on activation importance.
    Return:
      keep_idx: [keep_dim] long tensor
    """
    importance = lut_importance_from_activations(H_l)  # [H_l]
    H = importance.numel()
    keep_dim = max(min_keep, int(H * keep_ratio))
    keep_dim = min(keep_dim, H)

    _, idx = torch.topk(importance, k=keep_dim, largest=True, sorted=True)
    return idx  # [keep_dim]


def apply_lut_pruning_for_layer(
    model: MultiLayerWNN,
    layer_id: int,
    keep_idx: torch.Tensor,
):
    """
    Actually perform LUT pruning on a specific layer and update the input of the next layer / classifier.
    """
    device = next(model.parameters()).device
    L = len(model.layers)
    layer = model.layers[layer_id]

    old_num_luts = layer.num_luts
    keep_idx = keep_idx.to(device)
    assert keep_idx.dim() == 1
    assert keep_idx.max().item() < old_num_luts

    new_num_luts = keep_idx.numel()

    # 1) Build the new layer
    with torch.no_grad():
        old_table = layer.table.data.clone()        # [old_num_luts, 2^k]
        old_conn = layer.conn_idx.data.clone()      # [old_num_luts, k]
        lut_input_size = layer.lut_input_size
        in_bits = layer.in_bits

        new_table = old_table[keep_idx, :]          # [new_num_luts, 2^k]
        new_conn  = old_conn[keep_idx, :]           # [new_num_luts, k]

        new_layer = WNNLUTLayer(
            in_bits=in_bits,
            num_luts=new_num_luts,
            lut_input_size=lut_input_size,
        ).to(device)

        new_layer.table.data.copy_(new_table)
        new_layer.conn_idx.data.copy_(new_conn)

    model.layers[layer_id] = new_layer
    model.layer_out_luts[layer_id] = new_num_luts

    # 2) Build old→new index mapping
    mapping = -torch.ones(old_num_luts, dtype=torch.long, device=device)
    mapping[keep_idx] = torch.arange(new_num_luts, dtype=torch.long, device=device)

    # 3) Update next layer or classifier
    with torch.no_grad():
        if layer_id < L - 1:
            # Update next layer's conn_idx / in_bits
            next_layer = model.layers[layer_id + 1]
            conn_next = next_layer.conn_idx.data.clone()  # [num_luts_{l+1}, k]
            mapped = mapping[conn_next]                   # may have -1

            # Replace -1 positions with random valid indices (simple version, can refine with bit-pruning later)
            mask_invalid = (mapped < 0)
            if mask_invalid.any():
                rand_idx = torch.randint(
                    low=0,
                    high=new_num_luts,
                    size=mask_invalid.sum().shape,
                    device=device,
                )
                mapped[mask_invalid] = rand_idx

            next_layer.conn_idx.data.copy_(mapped)
            next_layer.in_bits = new_num_luts
            model.layer_in_bits[layer_id + 1] = new_num_luts
        else:
            # Last layer, update classifier
            old_W = model.classifier.weight.data.clone()  # [C, old_num_luts]
            W_pruned = old_W[:, keep_idx]                 # [C, new_num_luts]
            num_classes = old_W.size(0)
            new_classifier = nn.Linear(new_num_luts, num_classes, bias=False).to(device)
            new_classifier.weight.data.copy_(W_pruned)
            model.classifier = new_classifier


def plan_bit_rewire_for_layer(
    model: MultiLayerWNN,
    layer_id: int,
    candidate_pool: int = 64,
) -> torch.Tensor:
    """
    Simple coverage-based bit reallocation:
    - The in_bits of each layer remains unchanged
    - Each LUT sequentially selects k bits with the lowest current usage
    Return:
      new_conn_idx: [num_luts_l, lut_input_size]
    """

    layer = model.layers[layer_id]
    in_bits = layer.in_bits
    num_luts = layer.num_luts
    k = layer.lut_input_size
    device = next(model.parameters()).device

    usage = torch.zeros(in_bits, dtype=torch.long, device=device)
    '''new_conn = torch.empty(num_luts, k, dtype=torch.long, device=device)

    for j in range(num_luts):
        for t in range(k):
            # Select the bit with the lowest usage from a random subset
            if in_bits <= candidate_pool:
                candidates = torch.arange(in_bits, device=device)
            else:
                perm = torch.randperm(in_bits, device=device)
                candidates = perm[:candidate_pool]
            cand_usage = usage[candidates]
            best_idx = candidates[cand_usage.argmin()]
            new_conn[j, t] = best_idx
            usage[best_idx] += 1

    return new_conn  # [num_luts_l, k]'''
    old_conn = layer.conn_idx.data.cpu().numpy()
    bit_usage = usage  
    new_conn = select_local_bits_soft_coverage(
        old_conn,
        bit_usage,
        in_bits,
        lut_input_size=k,
        # Can add threshold/temperature/smoothing parameters
    )
    return torch.from_numpy(new_conn).long()


def finetune_after_prune(
    model: MultiLayerWNN,
    train_loader,
    val_loader,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 3,
    freeze_lut: bool = True,
):
    """
    LUT pruning / bit rewire followed by fine-tuning for a few epochs.
    By default, freeze LUTs and only train the classifier (consistent with your hidden pruning logic).
    """
    model.to(device)

    if freeze_lut:
        for name, p in model.named_parameters():
            if "table" in name:
                p.requires_grad = False
            else:
                p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    for epoch in range(epochs):
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

        train_loss, train_acc = eval_epoch(model, train_loader, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        print(
            f"[Finetune {epoch}] train_acc={train_acc*100:.2f}%, "
            f"val_acc={val_acc*100:.2f}%"
        )


def prune_wnn_multilayer(
    model,
    train_loader,
    val_loader,
    device,
    lut_keep_ratios,
    do_bit_pruning=False,
    bit_k_new_per_layer=None,
    finetune_epochs=3,
):
    """
    High-level API:
      1) collect per-layer activations
      2) per-layer bit rewire (optional)
      3) per-layer LUT pruning (according to lut_keep_ratios)
      4) finetune for a few epochs
    lut_keep_ratios length must equal the number of layers, e.g., (0.9, 0.9).
    """
    model.to(device)
    num_layers = len(model.layers)
    assert len(lut_keep_ratios) == num_layers

    # 0) baseline
    train_loss, train_acc = eval_epoch(model, train_loader, device)
    val_loss, val_acc = eval_epoch(model, val_loader, device)
    print(
        f"[Before prune] train_acc={train_acc*100:.2f}%, "
        f"val_acc={val_acc*100:.2f}%"
    )

    # 1) collect per-layer activations
    print("Collecting layer activations...")
    layer_H = collect_layer_activations(model, train_loader, device)
    for l, H_l in enumerate(layer_H):
        print(f"  Layer {l}: H_l shape = {H_l.shape}")

    # 2) first do bit rewire (without changing dim)
    '''if do_bit_rewire:
        print("Applying bit rewire for each layer...")
        for l in range(num_layers):
            new_conn = plan_bit_rewire_for_layer(model, l)
            apply_bit_rewire_for_layer(model, l, new_conn)'''

    if do_bit_pruning:
        print("Applying bit pruning (coverage-based) for each layer...")
        # If you didn't provide bit_k_new_per_layer, we enable adaptive k
        adaptive_cfg_default = {"coverage_ratio": 0.9, "k_min": 3}

        if bit_k_new_per_layer is None:
            bit_k_new_per_layer = [None] * num_layers  # Use adaptive_cfg for each layer

        assert len(bit_k_new_per_layer) == num_layers

        for l in range(num_layers):
            k_new = bit_k_new_per_layer[l]

            # You can use a per-layer adaptive config list, but for now just use the same params
            adaptive_cfg = adaptive_cfg_default if k_new is None else None

            new_conn, k_effective = plan_bit_pruning_for_layer(
                model,
                layer_id=l,
                k_new=k_new,
                r_per_bucket=1,
                adaptive_cfg=adaptive_cfg,
            )
            print(f"  Layer {l}: k_old={model.layers[l].lut_input_size}, k_eff={k_effective}")
            apply_bit_pruning_for_layer(model, l, new_conn)


    # 3) then do LUT pruning (will change dim, need to handle sequentially & update layer_H mapping)
    print("Applying LUT pruning for each layer...")
    for l in range(num_layers):
        keep_r = lut_keep_ratios[l]
        H_l = layer_H[l]  # (simplified) reuse activations collected before pruning for importance

        keep_idx = plan_lut_pruning_for_layer(H_l, keep_ratio=keep_r)
        print(
            f"  Layer {l}: num_luts={H_l.shape[1]} -> "
            f"{keep_idx.numel()} (keep_ratio={keep_r})"
        )
        apply_lut_pruning_for_layer(model, l, keep_idx)

    # 4) finetune
    if finetune_epochs > 0:
        print("Finetuning after pruning...")
        finetune_after_prune(
            model,
            train_loader,
            val_loader,
            device,
            lr=1e-3,
            epochs=finetune_epochs,
            freeze_lut=False,
        )

    # 5) final results
    train_loss, train_acc = eval_epoch(model, train_loader, device)
    val_loss, val_acc = eval_epoch(model, val_loader, device)
    print(
        f"[After prune] train_acc={train_acc*100:.2f}%, "
        f"val_acc={val_acc*100:.2f}%"
    )

    return model

def prune_wnn_multilayer_C(
    model,
    train_loader,
    val_loader,
    device,
    k_new_per_layer,       # e.g. (6, 5) or (5, 4)
    lut_keep_ratios,       # e.g. (0.9, 0.9)
    lut_keep_global,
    finetune_epochs=5,
    use_global_lut_prune = True
):
    model.to(device)
    num_layers = len(model.layers)
    assert len(k_new_per_layer) == num_layers
    assert len(lut_keep_ratios) == num_layers

    # 0) before
    train_loss, train_acc = eval_epoch(model, train_loader, device)
    val_loss, val_acc = eval_epoch(model, val_loader, device)
    print(f"[Before pruneC] train_acc={train_acc*100:.2f}%, val_acc={val_acc*100:.2f}%")

    # 1) shrink k for each layer
    for l in range(num_layers):
        k_old = model.layers[l].lut_input_size
        k_new = k_new_per_layer[l]
        if k_new < k_old:
            shrink_lut_input_size_for_layer(model, l, k_new=k_new, r_per_bucket=1)
        else:
            print(f"Layer {l}: skip k shrink (k_new={k_new} >= k_old={k_old})")

    # 2) collect activations (after k shrink)
    print("Collecting layer activations after k-shrink...")
    layer_H = collect_layer_activations(model, train_loader, device)
    
    # collect activations after k-shrink
    layer_H = collect_layer_activations(model, train_loader, device)

    if use_global_lut_prune:
        global_lut_pruning(
            model,
            layer_H,
            lut_keep_global=lut_keep_global,  # a global ratio
            device=device,
        )
    else:
        # Original per-layer implementation
        for l in range(num_layers):
            keep_r = lut_keep_ratios[l]
            H_l = layer_H[l]
            keep_idx = plan_lut_pruning_for_layer(H_l, keep_ratio=keep_r)
            apply_lut_pruning_for_layer(model, l, keep_idx)


    # 4) finetune
    if finetune_epochs > 0:
        print("Finetuning after C-pruning...")
        finetune_after_prune(
            model,
            train_loader,
            val_loader,
            device,
            lr=1e-3,
            epochs=finetune_epochs,
            freeze_lut=False,   # Recommend allowing LUT to adjust after k shrink
        )

    train_loss, train_acc = eval_epoch(model, train_loader, device)
    val_loss, val_acc = eval_epoch(model, val_loader, device)
    print(f"[After pruneC] train_acc={train_acc*100:.2f}%, val_acc={val_acc*100:.2f}%")

    return model

############## ADD ###############
def compute_bit_priority_for_layer(model, layer_id):
    """
    Calculate the usage count of each input bit in this layer based on conn_idx,
    which serves as the bit_priority.
    """
    layer = model.layers[layer_id]
    in_bits = layer.in_bits
    conn = layer.conn_idx.detach().cpu().numpy()  # [num_luts, k]

    bit_priority = np.zeros(in_bits, dtype=np.float32)
    for lut_bits in conn:
        for g in lut_bits:
            bit_priority[int(g)] += 1.0

    if bit_priority.max() > 0:
        bit_priority /= bit_priority.max()

    return bit_priority  # [in_bits]


def plan_bit_pruning_for_layer(
    model,
    layer_id: int,
    k_new: int,
    r_per_bucket: int = 1,
):
    """
    Perform coverage-based bit selection for a specific layer:
      - Fixed k_new (specified by you, e.g., 5)
      - Returns:
          new_conn_idx: [num_luts, k_new]  (global bit index)
          keep_local_pos_list: list of length num_luts, each element is a list of local positions
    """
    layer = model.layers[layer_id]
    conn = layer.conn_idx.detach().cpu().numpy()  # [num_luts, k_old]
    num_luts, k_old = conn.shape

    # all layers bit priority 
    bit_priority = compute_bit_priority_for_layer(model, layer_id)  # [in_bits]

    new_conn_list = []
    keep_local_pos_list = []

    for lut_bits in conn:              # lut_bits: array of length k_old (global index)
        lut_bits_list = [int(x) for x in lut_bits]

        keep_local_pos = select_local_bits_with_coverage(
            lut_bits=lut_bits_list,
            bit_priority=bit_priority,
            k=k_new,
            r_per_bucket=r_per_bucket,
        )
        # local pos → global bits
        new_bits = [lut_bits_list[p] for p in keep_local_pos]

        new_conn_list.append(new_bits)
        keep_local_pos_list.append(sorted(keep_local_pos))

    new_conn_idx = np.array(new_conn_list, dtype=np.int64)  # [num_luts, k_new]
    return new_conn_idx, keep_local_pos_list


def apply_bit_pruning_for_layer(model, layer_id, new_conn_idx):
    layer = model.layers[layer_id]
    num_luts = layer.num_luts
    k_old = layer.lut_input_size

    assert new_conn_idx.shape[0] == num_luts
    k_new = new_conn_idx.shape[1]

    # For now, require k_new == k_old for safety
    assert k_new == k_old, (
        f"k_new ({k_new}) != k_old ({k_old}); "
        "If you want to actually shrink k, WNNLUTLayer.table also needs to be rebuilt, "
        "recommend confirming adaptive behavior before doing table shrink."
    )

    layer.conn_idx.data = torch.from_numpy(new_conn_idx).long().to(layer.conn_idx.device)



def choose_adaptive_k_from_bit_priority(
    bit_priority: np.ndarray,
    base_k: int,
    k_min: int = 3,
    coverage_ratio: float = 0.9,
) -> int:
    """
    Automatically determine k_new based on the distribution of bit_priority:
      - Sort and find the smallest m such that the sum of the top m bit importances >= coverage_ratio * total
      - Finally clamp to [k_min, base_k]

    bit_priority: shape [in_bits]
    """
    pr = np.asarray(bit_priority, dtype=np.float64)
    total = pr.sum()
    if total <= 0:
        # if all zeros, base_k
        return base_k

    sorted_vals = np.sort(pr)[::-1]  # desc
    cum = np.cumsum(sorted_vals)
    thresh = coverage_ratio * total
    
    m = np.searchsorted(cum, thresh) + 1  # smallest m such that cum[m-1] >= thresh

    k_new = max(k_min, min(int(m), int(base_k)))
    return k_new


def shrink_lut_input_size_for_layer(
    model,
    layer_id: int,
    k_new: int,
    r_per_bucket: int = 1,
    reduce_mode: str = "mean",   # "sum" or "mean"
):
    """
    For a specific layer:
      1) Use coverage-based per-LUT bit selection to determine which local positions to keep (length k_new)
      2) Rebuild conn_idx
      3) Rebuild table: [num_luts, 2^k_old] → [num_luts, 2^k_new]
    """
    layer = model.layers[layer_id]
    device = next(model.parameters()).device

    k_old = layer.lut_input_size
    assert k_new <= k_old, "k_new must be <= k_old"

    num_luts = layer.num_luts
    old_table = layer.table.data.detach().clone().to(device)  # [num_luts, 2^k_old]

    # 1) coverage-based bit selection
    new_conn_idx_np, keep_local_pos_list = plan_bit_pruning_for_layer(
        model, layer_id, k_new=k_new, r_per_bucket=r_per_bucket
    )  # new_conn_idx_np: [num_luts, k_new]

    # 2) rebuild table per LUT
    new_table = torch.empty(num_luts, 2 ** k_new, device=device)

    for j in range(num_luts):
        row = old_table[j]  # [2^k_old]
        keep_local = keep_local_pos_list[j]  # e.g. [0,2,4]
        drop_local = [p for p in range(k_old) if p not in keep_local]

        # (2^k_old) → [2]*k_old
        row_nd = row.view(*([2] * k_old))  # dims: d0, d1, ..., d_{k_old-1}

        # sum (or mean) over drop dims
        if drop_local:
            # Note: summing over dimensions shrinks them, so summing from the largest index is safer
            for axis in sorted(drop_local, reverse=True):
                row_nd = row_nd.sum(dim=axis)
            if reduce_mode == "mean":
                row_nd = row_nd / (2 ** len(drop_local))

        # Now row_nd shape = [2]*k_new, flatten to [2^k_new]
        new_row = row_nd.reshape(-1)
        new_table[j] = new_row

    # 3) Update layer's table, conn_idx, lut_input_size, powers
    new_conn_idx = torch.from_numpy(new_conn_idx_np).long().to(device)

    layer.table = nn.Parameter(new_table)                  # [num_luts, 2^k_new]
    layer.conn_idx.data = new_conn_idx                     # [num_luts, k_new]
    layer.lut_input_size = k_new

    # Rebuild powers: [1,1,k_new]
    powers = (2 ** torch.arange(k_new, device=device)).float()
    layer.powers = powers.view(1, 1, -1)

    print(
        f"Layer {layer_id}: k_old={k_old} -> k_new={k_new}, "
        f"table shape: (num_luts={num_luts}, 2^{k_old}) -> 2^{k_new}"
    )


def compute_layer_budgets_from_global(
    model,
    bit_keep_global: float,
    lut_keep_global: float,
    k_min: int = 3,
    bit_sensitivity_per_layer=None,
    lut_sensitivity_per_layer=None,
):
    """
    Based on the global bit_keep_global / lut_keep_global provided by the user,
    automatically determine for each layer:
      - k_new_per_layer: tuple[int]
      - lut_keep_ratios: tuple[float]
    """
    num_layers = len(model.layers)
    base_k = [layer.lut_input_size for layer in model.layers]
    base_luts = [layer.num_luts for layer in model.layers]

    # Default sensitivity: earlier layers are more sensitive (higher weight), later layers less sensitive
    if bit_sensitivity_per_layer is None:
        # For example layer0:2, layer1:1, layer2:1 ...
        bit_sensitivity_per_layer = [max(1.0, float(num_layers - l)) for l in range(num_layers)]
    if lut_sensitivity_per_layer is None:
        lut_sensitivity_per_layer = [max(1.0, float(num_layers - l)) for l in range(num_layers)]

    # Normalize
    bit_S = sum(bit_sensitivity_per_layer)
    lut_S = sum(lut_sensitivity_per_layer)

    # 1) k_new_per_layer
    k_new_per_layer = []
    for l in range(num_layers):
        k0 = base_k[l]
        # Target global pruning is (1 - bit_keep_global),
        # actual pruning ratio per layer = global * (sensitivity_l / S)
        layer_prune = (1.0 - bit_keep_global) * (bit_sensitivity_per_layer[l] / bit_S)
        keep_ratio_l = 1.0 - layer_prune
        k_new = max(k_min, int(round(k0 * keep_ratio_l)))
        k_new = min(k_new, k0)
        k_new_per_layer.append(k_new)

    # 2) lut_keep_ratios per layer
    lut_keep_ratios = []
    for l in range(num_layers):
        # Similar approach, but you can also directly apply lut_keep_global to all layers
        layer_prune = (1.0 - lut_keep_global) * (lut_sensitivity_per_layer[l] / lut_S)
        keep_l = 1.0 - layer_prune
        # Safety clamp [0.1, 1.0]
        keep_l = float(max(0.1, min(keep_l, 1.0)))
        lut_keep_ratios.append(keep_l)

    return tuple(k_new_per_layer), tuple(lut_keep_ratios)


def prune_wnn_with_budget(
    model,
    train_loader,
    val_loader,
    device,
    bit_keep_global: float,
    lut_keep_global: float,
    k_min: int = 3,
    finetune_epochs: int = 5,
    use_global_lut_prune = True
):
    """
    Users provide:
      - bit_keep_global: e.g., 0.8 (keep 80% of total bits)
      - lut_keep_global: e.g., 0.9 (keep 90% of total LUTs)

    Internally:
      1) Allocate k_new / lut_keep_ratios per layer based on sensitivity
      2) Call prune_wnn_multilayer_C
      3) Print compression rate + accuracy
    """
    # 0) before metrics (including LUT entries)
    per_layer_before, total_entries_before = compute_lut_stats(model)
    train_loss, train_acc = eval_epoch(model, train_loader, device)
    val_loss, val_acc = eval_epoch(model, val_loader, device)
    print(
        f"[Before pruneC] train_acc={train_acc*100:.2f}%, "
        f"val_acc={val_acc*100:.2f}%, "
        f"total_LUT_entries={total_entries_before}"
    )

    # Calculate per-layer budget
    k_new_per_layer, lut_keep_ratios = compute_layer_budgets_from_global(
        model,
        bit_keep_global=bit_keep_global,
        lut_keep_global=lut_keep_global,
        k_min=k_min,
    )

    print("Global budgets:")
    print(f"  bit_keep_global = {bit_keep_global}")
    print(f"  lut_keep_global = {lut_keep_global}")
    print("Per-layer budgets:")
    for l, (k_new, keep_r) in enumerate(zip(k_new_per_layer, lut_keep_ratios)):
        k_old = model.layers[l].lut_input_size
        print(f"  Layer {l}: k {k_old} -> {k_new}, lut_keep_ratio = {keep_r:.3f}")

    # Call the existing C-pruning pipeline
    model_pruned = prune_wnn_multilayer_C(
        model,
        train_loader,
        val_loader,
        device,
        k_new_per_layer=k_new_per_layer,
        lut_keep_ratios=lut_keep_ratios,
        lut_keep_global=lut_keep_global,
        finetune_epochs=finetune_epochs,
        use_global_lut_prune = use_global_lut_prune
    )

    
    per_layer_after, total_entries_after = compute_lut_stats(model)
    compression = total_entries_after / total_entries_before

    train_loss, train_acc = eval_epoch(model, train_loader, device)
    val_loss, val_acc = eval_epoch(model, val_loader, device)
    print(
        f"[After pruneC] train_acc={train_acc*100:.2f}%, "
        f"val_acc={val_acc*100:.2f}%, "
        f"total_LUT_entries={total_entries_after}, "
        f"compression={compression*100:.2f}% of original"
    )

    print("Per-layer LUT stats before/after:")
    for bef, aft in zip(per_layer_before, per_layer_after):
        print(
            f"  Layer {bef['layer']}: "
            f"LUTs {bef['num_luts']}->{aft['num_luts']}, "
            f"k {bef['k']}->{aft['k']}, "
            f"entries {bef['entries']}->{aft['entries']}"
        )

    return model_pruned


def compute_lut_stats_temp(model):
    """
    Returns:
      per_layer: list of dict (layer, num_luts, k, entries)
      total_entries: sum of entries across all layers
    """
    per_layer = []
    total_entries = 0
    for l, layer in enumerate(model.layers):
        n = layer.num_luts
        k = layer.lut_input_size
        entries = n * (2 ** k)
        per_layer.append(
            dict(
                layer=l,
                num_luts=int(n),
                k=int(k),
                entries=int(entries),
            )
        )
        total_entries += entries

    return per_layer, int(total_entries)

def compute_lut_stats(model):
    """
    Returns:
      per_layer: list of dict (layer, num_luts, k, entries)
      total_entries: sum of LUT entries across all layers
    """
    per_layer = []
    total_entries = 0
    for l, layer in enumerate(model.layers):
        n = int(layer.num_luts)
        k = int(layer.lut_input_size)
        entries = n * (2 ** k)
        per_layer.append(
            dict(
                layer=l,
                num_luts=n,
                k=k,
                entries=entries,
            )
        )
        total_entries += entries

    return per_layer, int(total_entries)


def prune_wnn_with_budget_global_temp(
    model,
    train_loader,
    val_loader,
    device,
    bit_keep_global: float,
    lut_keep_global: float,
    k_min: int = 3,
    finetune_epochs: int = 5,
):
    """
    Complete pruning pipeline:  

      1) According to bit_keep_global, determine k_new for each layer (bit pruning)
      2) Perform k shrink for each layer (including table reconstruction)
      3) Collect activations
      4) Perform global LUT pruning (flatten all layers, prune by importance top-K)
      5) Finetune
      6) Print before/after accuracy + total compression rate
    """
    model.to(device)

    # ---- 0) before stats ----
    per_layer_before, total_entries_before = compute_lut_stats(model)
    train_loss, train_acc = eval_epoch(model, train_loader, device)
    val_loss, val_acc = eval_epoch(model, val_loader, device)
    print(
        f"[Before pruneC] train_acc={train_acc*100:.2f}%, "
        f"val_acc={val_acc*100:.2f}%, "
        f"total_LUT_entries={total_entries_before}"
    )

    # ---- 1) bit shrink: k_old -> k_new per layer ----
    k_new_per_layer = compute_k_new_per_layer_from_bit_keep(
        model,
        bit_keep_global=bit_keep_global,
        k_min=k_min,
    )
    print("Bit budgets:")
    for l, k_new in enumerate(k_new_per_layer):
        k_old = model.layers[l].lut_input_size
        print(f"  Layer {l}: k {k_old} -> {k_new}")

    for l, k_new in enumerate(k_new_per_layer):
        k_old = model.layers[l].lut_input_size
        if k_new < k_old:
            shrink_lut_input_size_for_layer(model, l, k_new=k_new, r_per_bucket=1)
        else:
            print(f"  Layer {l}: skip k shrink (k_new={k_new} >= k_old={k_old})")

    # ---- 2) collect activations after k-shrink ----
    print("Collecting layer activations after k-shrink...")
    layer_H = collect_layer_activations(model, train_loader, device)

    # ---- 3) global LUT pruning ----
    print(f"Applying GLOBAL LUT pruning with lut_keep_global={lut_keep_global}...")
    total_luts_before = sum(h.shape[1] for h in layer_H)
    global_lut_pruning_from_activations(
        model,
        layer_H,
        lut_keep_global=lut_keep_global,
        device=device,
    )
    total_luts_after = sum(layer.num_luts for layer in model.layers)
    print(
        f"  Total LUTs: {total_luts_before} -> {total_luts_after} "
        f"({total_luts_after/total_luts_before*100:.2f}% of original LUT count)"
    )

    # ---- 4) finetune ----
    if finetune_epochs > 0:
        print("Finetuning after C-pruning + global LUT pruning...")
        finetune_after_prune(
            model,
            train_loader,
            val_loader,
            device,
            lr=1e-3,
            epochs=finetune_epochs,
            freeze_lut=False,
        )

    # ---- 5) after stats ----
    per_layer_after, total_entries_after = compute_lut_stats(model)
    compression = total_entries_after / total_entries_before

    train_loss, train_acc = eval_epoch(model, train_loader, device)
    val_loss, val_acc = eval_epoch(model, val_loader, device)
    print(
        f"[After pruneC] train_acc={train_acc*100:.2f}%, "
        f"val_acc={val_acc*100:.2f}%, "
        f"total_LUT_entries={total_entries_after}, "
        f"compression={compression*100:.2f}% of original"
    )

    print("Per-layer LUT stats before/after:")
    for bef, aft in zip(per_layer_before, per_layer_after):
        print(
            f"  Layer {bef['layer']}: "
            f"LUTs {bef['num_luts']}->{aft['num_luts']}, "
            f"k {bef['k']}->{aft['k']}, "
            f"entries {bef['entries']}->{aft['entries']}"
        )

    return model


def prune_wnn_with_budget_global(
    model,
    train_loader,
    val_loader,
    device,
    bit_keep_global: float,
    lut_keep_global: float,
    k_min: int = 3,
    finetune_epochs: int = 5,
):
    """
    Complete pruning pipeline:  

      1) According to bit_keep_global, determine k_new for each layer (bit pruning)
      2) Perform k shrink for each layer (including table reconstruction)
      3) Collect activations
      4) Perform global LUT pruning (flatten all layers, prune by importance top-K)
      5) Finetune
      6) Print before/after accuracy + total compression rate
    """
    model.to(device)

    # ---- 0) before stats ----
    per_layer_before, total_entries_before = compute_lut_stats(model)
    train_loss, train_acc = eval_epoch(model, train_loader, device)
    val_loss, val_acc = eval_epoch(model, val_loader, device)
    print(
        f"[Before pruneC] train_acc={train_acc*100:.2f}%, "
        f"val_acc={val_acc*100:.2f}%, "
        f"total_LUT_entries={total_entries_before}"
    )

    # ---- 1) bit shrink: k_old -> k_new per layer ----
    k_new_per_layer = compute_k_new_per_layer_from_bit_keep(
        model,
        bit_keep_global=bit_keep_global,
        k_min=k_min,
    )
    print("Bit budgets:")
    for l, k_new in enumerate(k_new_per_layer):
        k_old = model.layers[l].lut_input_size
        print(f"  Layer {l}: k {k_old} -> {k_new}")

    for l, k_new in enumerate(k_new_per_layer):
        k_old = model.layers[l].lut_input_size
        if k_new < k_old:
            shrink_lut_input_size_for_layer(model, l, k_new=k_new, r_per_bucket=1)
        else:
            print(f"  Layer {l}: skip k shrink (k_new={k_new} >= k_old={k_old})")

    # ---- 2) collect activations after k-shrink ----
    print("Collecting layer activations after k-shrink...")
    layer_H = collect_layer_activations(model, train_loader, device)

    # ---- 3) global LUT pruning ----
    '''print(f"Applying GLOBAL LUT pruning with lut_keep_global={lut_keep_global}...")
    total_luts_before = sum(h.shape[1] for h in layer_H)
    global_lut_pruning_from_activations(
        model,
        layer_H,
        lut_keep_global=lut_keep_global,
        device=device,
    )
    total_luts_after = sum(layer.num_luts for layer in model.layers)
    print(
        f"  Total LUTs: {total_luts_before} -> {total_luts_after} "
        f"({total_luts_after/total_luts_before*100:.2f}% of original LUT count)"
    )'''
    print(f"Applying GLOBAL LUT pruning with lut_keep_global={lut_keep_global}...")
    total_luts_before = sum(h.shape[1] for h in layer_H)

    layer_lut_alloc = global_lut_pruning_from_activations(
        model,
        layer_H,
        lut_keep_global=lut_keep_global,
        device=device,
    )

    total_luts_after = sum(layer.num_luts for layer in model.layers)
    print(
        f"  Total LUTs: {total_luts_before} -> {total_luts_after} "
        f"({total_luts_after/total_luts_before*100:.2f}% of original LUT count)"
    )

    # Print "per-layer pruning rate allocation"
    print("Per-layer LUT pruning allocation (global view):")
    for l, (n_before, n_after) in layer_lut_alloc.items():
        keep_ratio  = n_after / n_before
        prune_ratio = 1.0 - keep_ratio
        share_after = n_after / total_luts_after  # proportion of remaining LUTs

        print(
            f"  Layer {l}: "
            f"LUTs {n_before}->{n_after}, "
            f"keep={keep_ratio*100:.2f}%, "
            f"prune={prune_ratio*100:.2f}%, "
            f"share_after={share_after*100:.2f}% of remaining LUTs"
        )

    train_loss, train_acc = eval_epoch(model, train_loader, device)
    val_loss, val_acc = eval_epoch(model, val_loader, device)
    print(
        f"[Before finetuning] train_acc={train_acc*100:.2f}%, "
        f"val_acc={val_acc*100:.2f}% "
    )

    # ---- 4) finetune ----
    if finetune_epochs > 0:
        print("Finetuning after C-pruning + global LUT pruning...")
        finetune_after_prune(
            model,
            train_loader,
            val_loader,
            device,
            lr=1e-3,
            epochs=finetune_epochs,
            freeze_lut=False,
        )

    # ---- 5) after stats ----
    per_layer_after, total_entries_after = compute_lut_stats(model)
    compression = total_entries_after / total_entries_before

    train_loss, train_acc = eval_epoch(model, train_loader, device)
    val_loss, val_acc = eval_epoch(model, val_loader, device)
    print(
        f"[After finetuning] train_acc={train_acc*100:.2f}%, "
        f"val_acc={val_acc*100:.2f}%, "
        f"total_LUT_entries={total_entries_after}, "
        f"compression={compression*100:.2f}% of original"
    )

    print("Per-layer LUT stats before/after:")
    for bef, aft in zip(per_layer_before, per_layer_after):
        print(
            f"  Layer {bef['layer']}: "
            f"LUTs {bef['num_luts']}->{aft['num_luts']}, "
            f"k {bef['k']}->{aft['k']}, "
            f"entries {bef['entries']}->{aft['entries']}"
        )

    return model
        
        
def allocate_lut_ratios_global(num_luts, lut_keep_global, sensitivity):
    """
    num_luts: list, e.g. [2000, 1000]
    lut_keep_global: e.g. 0.6
    sensitivity: list, e.g. [2.0, 1.0]
    
    Returns per-layer lut_keep_ratios such that:
        sum(num_luts[l] * keep_l[l]) = lut_keep_global * total_luts
    """
    L = len(num_luts)
    total = sum(num_luts)
    target = lut_keep_global * total  # target total remaining LUTs
    
    S = sum(sensitivity)

    # General form:
    # keep_l = (sensitivity[l] / S) * t
    # Solve for t such that sum(num_luts * keep_l) = target
    # => t = target / sum(num_luts[l] * (sens[l]/S))
    
    denom = 0
    for nl, w in zip(num_luts, sensitivity):
        denom += nl * (w / S)
    
    t = target / denom
    
    keep = []
    for w in sensitivity:
        k = (w / S) * t
        k = float(max(0.1, min(k, 1.0)))  # clamp to safe range
        keep.append(k)

    return keep


def global_lut_pruning(
    model,
    layer_H,
    lut_keep_global: float,
    device,
):
    num_layers = len(layer_H)

    # 1) flatten
    all_scores = []
    all_layer_ids = []
    all_lut_ids = []

    for l in range(num_layers):
        H_l = layer_H[l]                   # [N, num_luts_l]
        imp_l = H_l.abs().mean(dim=0)      # [num_luts_l]
        num_luts_l = imp_l.shape[0]

        all_scores.append(imp_l)
        all_layer_ids.append(
            torch.full((num_luts_l,), l, dtype=torch.long, device=imp_l.device)
        )
        all_lut_ids.append(torch.arange(num_luts_l, device=imp_l.device))

    all_scores    = torch.cat(all_scores)
    all_layer_ids = torch.cat(all_layer_ids)
    all_lut_ids   = torch.cat(all_lut_ids)

    total_luts = all_scores.numel()
    global_keep = int(round(total_luts * lut_keep_global))
    global_keep = max(1, min(global_keep, total_luts))

    scores_sorted, idx_sorted = torch.sort(all_scores, descending=True)
    keep_global_idx = idx_sorted[:global_keep]

    keep_layer_ids = all_layer_ids[keep_global_idx]
    keep_lut_ids   = all_lut_ids[keep_global_idx]

    # 2) per-layer collect
    keep_idx_per_layer = {}
    for l in range(num_layers):
        mask = (keep_layer_ids == l)
        keep_l = keep_lut_ids[mask]
        if keep_l.numel() == 0:
            # At least keep one LUT
            keep_l = torch.tensor([0], device=device)
        keep_idx_per_layer[l] = keep_l.sort()[0]

    # 3) apply
    for l in range(num_layers):
        apply_lut_pruning_for_layer(model, l, keep_idx_per_layer[l])


def compute_k_new_per_layer_from_bit_keep(
    model,
    bit_keep_global: float,
    k_min: int = 3,
):
    """
    According to bit_keep_global, determine the new k (lut_input_size) for each layer.
    Currently using a simple version: k_new for each layer = round(k_old * bit_keep_global), then clamp to [k_min, k_old].
    """
    num_layers = len(model.layers)
    base_k = [layer.lut_input_size for layer in model.layers]

    k_new_per_layer = []
    for l in range(num_layers):
        k0 = base_k[l]
        if bit_keep_global >= 1.0:
            k_new = k0
        else:
            k_new = max(k_min, int(round(k0 * bit_keep_global)))
            k_new = min(k_new, k0)
        k_new_per_layer.append(k_new)

    return tuple(k_new_per_layer)


def global_lut_pruning_from_activations(
    model,
    layer_H,
    lut_keep_global: float,
    device,
):
    """
    Global LUT pruning:
      - Flatten all layers' LUTs into a single pool
      - Perform global sorting based on activation importance
      - Keep the top global_keep LUTs, prune the rest
      - Return per-layer before/after LUT counts for displaying actual pruning rates per layer
    """
    num_layers = len(layer_H)

    # Record before LUT counts for each layer
    luts_before = [h.shape[1] for h in layer_H]

    # 1) Flatten all layers' LUT scores
    all_scores = []
    all_layer_ids = []
    all_lut_ids = []

    for l in range(num_layers):
        H_l = layer_H[l].to(device)          # [N, num_luts_l]
        imp_l = H_l.abs().mean(dim=0)        # [num_luts_l]
        num_luts_l = imp_l.shape[0]

        all_scores.append(imp_l)
        all_layer_ids.append(
            torch.full((num_luts_l,), l, dtype=torch.long, device=device)
        )
        all_lut_ids.append(torch.arange(num_luts_l, device=device))

    all_scores    = torch.cat(all_scores, dim=0)     # [total_luts]
    all_layer_ids = torch.cat(all_layer_ids, dim=0)  # [total_luts]
    all_lut_ids   = torch.cat(all_lut_ids, dim=0)    # [total_luts]

    total_luts = all_scores.numel()
    global_keep = int(round(total_luts * lut_keep_global))
    # At least keep one LUT per layer (conservative)
    global_keep = max(num_layers, min(global_keep, total_luts))

    # 2) Global sorting
    scores_sorted, idx_sorted = torch.sort(all_scores, descending=True)
    keep_global_idx = idx_sorted[:global_keep]

    keep_layer_ids = all_layer_ids[keep_global_idx]
    keep_lut_ids   = all_lut_ids[keep_global_idx]

    # 3) Restore per-layer keep_idx
    keep_idx_per_layer = {}
    for l in range(num_layers):
        mask = (keep_layer_ids == l)
        keep_l = keep_lut_ids[mask]
        if keep_l.numel() == 0:
            keep_l = torch.tensor([0], device=device)  # At least keep one LUT
        keep_idx_per_layer[l] = keep_l.sort()[0]

    # 4) Apply to each layer
    for l in range(num_layers):
        keep_idx = keep_idx_per_layer[l]
        apply_lut_pruning_for_layer(model, l, keep_idx)

    # 5) Record after LUT counts for each layer
    luts_after = [int(layer.num_luts) for layer in model.layers]

    # Return per-layer before/after for external pruning rate calculation
    layer_lut_alloc = {
        l: (int(luts_before[l]), int(luts_after[l]))
        for l in range(num_layers)
    }
    return layer_lut_alloc
