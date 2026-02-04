from collections import defaultdict

import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union, Sequence
from src.prune import _reduce_counts_for_lut, select_local_bits_soft_coverage, select_top_luts_by_priority
from src.tools.utils import _assert_power_of_two, _lut_addr_entropy_unique
from src.tools.utils import _addr_from_bits
from src.dataio.encode import bucket_mapper_mnist_thermo
import heapq

# ------------------ create "adaptive k" profil (unaligned m) w a given rate ------------------
def build_runtime_profile_per_lut_adaptive(model,
                                           tuple_mapping: List[List[int]],
                                           bit_priority: np.ndarray,
                                           bits_keep_ratio: float,
                                           X_bits_val: np.ndarray,
                                           *,
                                           coverage_r: int = 1,
                                           H_min: float = 2.0,
                                           U_min: int = 64,
                                           k_cap: Optional[int] = None,
                                           bucket_mapper=bucket_mapper_mnist_thermo) -> Dict:
    """
    for all LUT：
      1) select keeped local address (include converage) by top k
      2) get (C, 2^m_l)
      3) verify H/unique: if lower than (H_min, U_min) -> k += 1, until converge or k reach the upper bound (n or k_cap)

    Return:
      {
        "lut_tables": [ (C, 2^m_l) ... ],
        "kept_global_bits_per_lut": [ [gbits...] ... ]  # LUT global bits
        "addr_bits_per_lut": [m_l ...],
        "num_classes": C,
        "alpha": float(model.alpha or 1.0),
        "bit_order": "lsb"
      }
    """
    table = np.asarray(model.table)  # (C, L, 2^n)
    C, L, A = table.shape
    _assert_power_of_two(A)
    n_addr_bits = A.bit_length() - 1

    k_base = max(1, int(round(n_addr_bits * bits_keep_ratio)))
    if k_cap is None:
        k_cap = n_addr_bits

    lut_tables, kept_global_bits_all, m_list = [], [], []

    for lut_id in range(L):
        lut_bits = tuple_mapping[lut_id]  # length n；same order as the trained n
        n = len(lut_bits)
        assert n == n_addr_bits

        # adaptive k loop
        k = min(max(1, k_base), min(k_cap, n))
        while True:
            keep_local = select_local_bits_with_coverage(
                lut_bits=lut_bits,
                bit_priority=bit_priority,
                k=k,
                r_per_bucket=coverage_r,
                bucket_mapper=bucket_mapper
            )
            reduced, kept_local_pos = _reduce_counts_for_lut(table[:, lut_id, :], keep_local, n_addr_bits)
            kept_global_bits = [lut_bits[p] for p in kept_local_pos]
            H, U = _lut_addr_entropy_unique(kept_global_bits, X_bits_val)

            # reach the upper bound -> converge
            if (H >= H_min and U >= U_min) or (k >= n) or (k >= k_cap):
                break
            # otherwise k = k + 1
            k += 1

        lut_tables.append(reduced.astype(np.float32))
        kept_global_bits_all.append(kept_global_bits)
        m_list.append(int(np.log2(reduced.shape[1])))

    profile = dict(
        lut_tables=lut_tables,
        kept_global_bits_per_lut=kept_global_bits_all,
        addr_bits_per_lut=m_list,
        num_classes=C,
        alpha=float(getattr(model, "alpha", 1.0)),
        bit_order="lsb",
    )
    return profile



def build_runtime_profile_per_lut_adaptive3(
    model,
    tuple_mapping,
    bit_priority,
    bits_keep_ratio,
    X_bits_val,
    *,
    # coverage
    coverage_mode="soft",            # "soft" or "hard"
    coverage_r=1,
    coverage_k_threshold=4,

    # safety
    H_min=1.8,
    U_min=48,
    H_target=2.6,
    dH_min=0.02,

    # per-LUT cap (int or array)
    k_cap=None,

    # bucket mapper
    bucket_mapper=bucket_mapper_mnist_thermo
):
    """
    Adaptive bit-folding with entropy guard + soft/hard coverage + per-LUT caps.
    Compatible with joint LUT pruning.
    """
    table = np.asarray(model.table)
    C, L, A = table.shape
    n_addr_bits = A.bit_length() - 1
    _assert_power_of_two(A)

    # global base k
    k_base_global = max(1, int(round(n_addr_bits * bits_keep_ratio)))

    # unify k_cap -> array
    if k_cap is None:
        kcap_arr = np.full(L, n_addr_bits, dtype=np.int32)
    elif isinstance(k_cap, int):
        kcap_arr = np.full(L, int(k_cap), dtype=np.int32)
    else:
        kcap_arr = np.asarray(k_cap, dtype=np.int32)
        assert kcap_arr.shape[0] == L

    lut_tables = []
    kept_bits_all = []
    m_list = []
    meta = []

    for lut_id in range(L):
        lut_bits = tuple_mapping[lut_id]
        n = len(lut_bits)

        this_cap = kcap_arr[lut_id]
        k = max(1, min(k_base_global, this_cap, n))
        prev_H = None

        while True:
            # ----- select local bits -----
            if coverage_mode == "hard":
                keep_local = select_local_bits_with_coverage(
                    lut_bits, bit_priority, k,
                    r_per_bucket=coverage_r,
                    bucket_mapper=bucket_mapper
                )
            else:  # "soft"
                keep_local = select_local_bits_soft_coverage(
                    lut_bits, bit_priority, k,
                    r_per_bucket_when_small=coverage_r,
                    bucket_mapper=bucket_mapper,
                    coverage_k_threshold=coverage_k_threshold
                )

            # ----- reduce LUT -----
            reduced_tab, kept_local_pos = _reduce_counts_for_lut(
                table[:, lut_id, :], keep_local, n_addr_bits
            )
            kept_global_bits = [lut_bits[p] for p in kept_local_pos]

            # ----- compute entropy H, unique U -----
            H, U = _lut_addr_entropy_unique(kept_global_bits, X_bits_val)

            # ----- stopping condition -----
            stop = False
            if H >= H_min and U >= U_min:
                if H > H_target:
                    stop = True
                elif prev_H is not None and (H - prev_H) < dH_min:
                    stop = True
            if stop or (k >= n) or (k >= this_cap):
                lut_tables.append(reduced_tab.astype(np.float32))
                kept_bits_all.append(kept_global_bits)
                m_list.append(int(np.log2(reduced_tab.shape[1])))
                meta.append({"lut": lut_id, "k_final": k, "H": float(H), "U": int(U)})
                break

            prev_H = H
            k += 1

    return dict(
        lut_tables=lut_tables,
        kept_global_bits_per_lut=kept_bits_all,
        addr_bits_per_lut=m_list,
        num_classes=C,
        alpha=float(getattr(model, "alpha", 1.0)),
        bit_order="lsb",
        meta=meta
    )


def build_profile_from_fixed_k(
    model,
    tuple_mapping: List[List[int]],
    bit_priority: np.ndarray,
    k_or_array: Union[int, Sequence[int]],
    X_bits_val: np.ndarray,
    *,
    coverage_k_threshold: int = 4,
    coverage_r: int = 1,
    bucket_mapper=None,           # e.g. bucket_mapper_mnist_thermo
    alpha: float = None
) -> Dict:
    """
    given k（global k or per-LUT array）generate compressed profile（LSB-first）。
    """
    table = np.asarray(model.table)     # (C, L, 2^n)
    C, L, A = table.shape
    n_addr_bits = A.bit_length() - 1

    if isinstance(k_or_array, int):
        k_arr = np.full(L, int(k_or_array), dtype=np.int32)
    else:
        k_arr = np.asarray(k_or_array, dtype=np.int32)
        assert k_arr.shape[0] == L

    lut_tables, kept_global_bits_all, m_list, meta = [], [], [], []

    for lut_id in range(L):
        lut_bits = tuple_mapping[lut_id]
        n = len(lut_bits)
        k = int(np.clip(k_arr[lut_id], 1, n))

        # soft-coverage：small k -> coverage, otherwise -> priority
        keep_local = select_local_bits_soft_coverage(
            lut_bits=lut_bits, bit_priority=bit_priority, k=k,
            r_per_bucket_when_small=coverage_r,
            bucket_mapper=bucket_mapper,
            coverage_k_threshold=coverage_k_threshold
        )

        reduced, kept_local_pos = _reduce_counts_for_lut(
            table[:, lut_id, :], keep_local, n_addr_bits
        )
        kept_global_bits = [lut_bits[p] for p in kept_local_pos]

        # record meta
        H, U = _lut_addr_entropy_unique(kept_global_bits, X_bits_val)
        lut_tables.append(reduced.astype(np.float32))
        kept_global_bits_all.append(kept_global_bits)
        m_list.append(int(np.log2(reduced.shape[1])))
        meta.append({"lut": lut_id, "k_final": k, "H": float(H), "U": int(U)})

    return dict(
        lut_tables=lut_tables,
        kept_global_bits_per_lut=kept_global_bits_all,
        addr_bits_per_lut=m_list,
        num_classes=C,
        alpha=float(alpha if alpha is not None else getattr(model, "alpha", 1.0)),
        bit_order="lsb",
        meta=meta
    )


def make_bit_efficiency_curves(
    model,
    tuple_mapping: List[List[int]],
    bit_priority: np.ndarray,
    X_bits_val: np.ndarray,
    *,
    H_target: float = 2.6,
    coverage_k_threshold: int = 4,
    coverage_r: int = 1,
    bucket_mapper=None
) -> List[Dict]:
    """
    return per-LUT curve：
      curves[l] = {
        "k": [0..n],
        "H": [...],
        "U": [...],
        "benefit": [ΔH_trunc at step k->k+1],
        "cost":    [Δentries at step k->k+1]
      }
    """
    table = np.asarray(model.table)     # (C, L, 2^n)
    C, L, A = table.shape
    n_addr_bits = A.bit_length() - 1

    curves = []
    for lut_id in range(L):
        lut_bits = tuple_mapping[lut_id]
        n = len(lut_bits)

        ks, Hs, Us = [], [], []
        kept_so_far = []
        # order by priority
        local_scored = [(p, float(bit_priority[lut_bits[p]])) for p in range(n)]
        local_scored.sort(key=lambda x: x[1], reverse=True)
        ordered_pos = [p for p, _ in local_scored]

        for k in range(0, n + 1):
            if k == 0:
                kept_local = []
            else:
                # soft-coverage
                if k <= coverage_k_threshold:
                    kept_local = select_local_bits_soft_coverage(
                        lut_bits, bit_priority, k,
                        r_per_bucket_when_small=coverage_r,
                        bucket_mapper=bucket_mapper,
                        coverage_k_threshold=coverage_k_threshold
                    )
                else:
                    kept_local = sorted(ordered_pos[:k])

            reduced, kept_local_pos = _reduce_counts_for_lut(
                table[:, lut_id, :], kept_local, n_addr_bits
            )
            kept_global_bits = [lut_bits[p] for p in kept_local_pos]
            H, U = _lut_addr_entropy_unique(kept_global_bits, X_bits_val)

            ks.append(k); Hs.append(float(H)); Us.append(int(U))

        # boundary benefit/cost（k->k+1）
        benefits, costs = [], []
        for k in range(0, n):
            h0 = min(Hs[k], H_target); h1 = min(Hs[k+1], H_target)
            benefits.append(max(0.0, h1 - h0))
            costs.append((1 << (k+1)) - (1 << k))

        curves.append(dict(k=ks, H=Hs, U=Us, benefit=benefits, cost=costs))
    return curves


def allocate_bits_greedy(
    curves: List[Dict],
    keep_ids: np.ndarray,
    *,
    n_full: int,
    addr_budget_ratio: float
) -> np.ndarray:
    """
    under address budget（entries_budget）distribute m（k_final） in greedy。
    keep_ids: keeped LUT index（after LUT pruning）
    return per-LUT's k_final（length = len(keep_ids)；index by keep_ids' order）
    """
    keep_ids = np.asarray(keep_ids, dtype=int)
    Lk = keep_ids.shape[0]
    # total budget
    entries_budget = addr_budget_ratio * (Lk * (1 << n_full))

    # initial k=0, each LUT's entries = 1
    current_entries = Lk * 1
    k_final = np.zeros(Lk, dtype=np.int32)

    # use -ratio for max-heap
    heap = []
    for j, lut in enumerate(keep_ids):
        benefit = curves[lut]["benefit"]
        cost    = curves[lut]["cost"]
        if len(benefit) > 0:
            ratio = (benefit[0] / max(1, cost[0])) if cost[0] > 0 else 0.0
            heapq.heappush(heap, (-ratio, j, 0))  # (neg_ratio, local_idx, step_k)

    while heap and current_entries <= entries_budget:
        neg_ratio, j, step = heapq.heappop(heap)
        lut = keep_ids[j]
        # cost
        c = curves[lut]["cost"][step]
        if current_entries + c > entries_budget:
            # if overflow
            continue
        # accept
        current_entries += c
        k_final[j] += 1
        # if next step
        if step + 1 < len(curves[lut]["benefit"]):
            nb = curves[lut]["benefit"][step+1]
            nc = curves[lut]["cost"][step+1]
            ratio = (nb / max(1, nc)) if nc > 0 else 0.0
            heapq.heappush(heap, (-ratio, j, step+1))

    return k_final


def build_joint_budget_profile(
    model,
    tuple_mapping: List[List[int]],
    bit_priority: np.ndarray,
    lut_priority: np.ndarray,
    X_bits_val: np.ndarray,
    *,
    luts_keep_ratio: float,
    addr_budget_ratio: float,
    n_full: int,
    H_target: float = 2.6,
    coverage_k_threshold: int = 4,
    coverage_r: int = 1,
    bucket_mapper=None
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    choose keep_ids（LUT pruning）, use efficiency curve + greedy for address distribution
    use build_profile_from_fixed_k generate profile。
    return：profile, keep_ids, k_final_global（length = L；pruned = 0）
    """
    # 1) LUT pruning
    keep_ids = select_top_luts_by_priority(lut_priority, luts_keep_ratio)

    # 2) efficiency curve
    curves = make_bit_efficiency_curves(
        model, tuple_mapping, bit_priority, X_bits_val,
        H_target=H_target,
        coverage_k_threshold=coverage_k_threshold,
        coverage_r=coverage_r,
        bucket_mapper=bucket_mapper
    )

    # 3) keep_ids do greedy dist
    k_final_kept = allocate_bits_greedy(
        curves, keep_ids, n_full=n_full, addr_budget_ratio=addr_budget_ratio
    )

    # 4) fixed k, generate profile（for kept LUTs）
    pruned_table = model.table[:, keep_ids, :]
    pruned_mapping = [tuple_mapping[i] for i in keep_ids]

    class _M: pass
    m = _M(); m.table = pruned_table; m.alpha = getattr(model, "alpha", 1.0)

    profile = build_profile_from_fixed_k(
        model=m,
        tuple_mapping=pruned_mapping,
        bit_priority=bit_priority,
        k_or_array=k_final_kept,
        X_bits_val=X_bits_val,
        coverage_k_threshold=coverage_k_threshold,
        coverage_r=coverage_r,
        bucket_mapper=bucket_mapper,
        alpha=getattr(model, "alpha", 1.0)
    )

    # 5) generate global k_final
    k_final_global = np.zeros(len(tuple_mapping), dtype=np.int32)
    k_final_global[keep_ids] = k_final_kept

    return profile, keep_ids, k_final_global




def predict_with_profile_varm(profile: Dict, bit_vec: np.ndarray, mode: str = "log_posterior") -> int:
    alpha = profile["alpha"]
    C = profile["num_classes"]
    scores = np.zeros(C, dtype=np.float32)

    for lut_tab, gbits in zip(profile["lut_tables"], profile["kept_global_bits_per_lut"]):
        addr = _addr_from_bits(bit_vec, gbits)
        counts = lut_tab[:, addr]  # (C,)
        if mode == "log_count":
            scores += np.log(counts + alpha + 1e-9)
        else:
            denom = counts.sum() + C * alpha
            post = (counts + alpha) / denom
            log_post = np.log(post + 1e-9)
            if mode == "zero_mean_log_posterior":
                log_post = log_post - log_post.mean()
            scores += log_post

    return int(np.argmax(scores))


def eval_with_profile_varm(profile: Dict, X_bits: np.ndarray, y: np.ndarray, mode: str = "log_posterior") -> float:
    X_bits = np.asarray(X_bits); y = np.asarray(y)
    correct = 0
    for i in range(X_bits.shape[0]):
        if predict_with_profile_varm(profile, X_bits[i], mode=mode) == int(y[i]):
            correct += 1
    return correct / X_bits.shape[0]

def profile_stats_total(profile: Dict, n_full: int, L_full: int) -> Dict:
    #calculate Addr_comp、L_comp、Total_comp
    m_list = profile["addr_bits_per_lut"]
    L_kept  = len(m_list)
    addr_entries = sum(1 << m for m in m_list)
    addr_full_per_lut = 1 << n_full
    addr_comp = addr_entries / (L_kept * addr_full_per_lut) if L_kept > 0 else 0.0
    L_comp = L_kept / float(L_full)
    total_comp = addr_entries / float(L_full * addr_full_per_lut)
    return dict(L_kept=L_kept, addr_comp=addr_comp, L_comp=L_comp,
                total_comp=total_comp, avg_m=(sum(m_list)/L_kept if L_kept else 0.0))



# ---------------------------
# Build runtime profile (bits_keep × luts_keep), with alignment
# ---------------------------
def build_runtime_profile(model,
                          tuple_mapping: List[List[int]],
                          keep_bits_set: Set[int],
                          lut_priority: np.ndarray,
                          luts_keep_ratio: float,
                          *,
                          bit_priority: Optional[np.ndarray] = None,
                          align_strategy: str = "min") -> Dict:
    """
    estimates profile（under both bit pruning and LUT pruning）：
      - for each LUT, marginalization by keep_bits_set（sum the pruned bits）
      - reselect top-K% LUT
      - align LUT addr_bits（target_m）
        - 'min': align with min m
        - 'topk_by_bit_priority': choose the target_m with the highest score, use bit_priority in LUT kept bits

    return profile dict：
      {
        "lut_table": (C, L_active_aligned, 2^target_m) float32，
        "kept_global_bits_per_lut": List[List[int]]  # LUT global bits
        "addr_bits": target_m,
        "num_classes": C,
        "alpha": model.alpha (default 1.0)
      }
    """
    table = np.asarray(model.table)  # (C,L,A)
    if table.ndim != 3:
        raise ValueError(f"model.table must be (C,L,A), got shape={table.shape}")
    C, L, A = table.shape
    _assert_power_of_two(A)
    n_addr_bits = A.bit_length() - 1

    if len(tuple_mapping) != L:
        raise ValueError(f"tuple_mapping length {len(tuple_mapping)} != L {L}")

    order_lut = np.argsort(-np.asarray(lut_priority))
    kL = max(1, int(round(L * luts_keep_ratio)))
    candidate_ids = order_lut[:kL].tolist()

    reduced_list = []
    kept_local_pos_list = []
    kept_global_bits_list = []
    m_list = []
    active_ids = []

    # marginalization each selected LUT（based on keep_bits_set）
    for lut_id in candidate_ids:
        lut_bits = tuple_mapping[lut_id]  # global bit index（length = n_addr_bits；order=training address order）
        # convert into local address index
        keep_pos = [p for p, gbit in enumerate(lut_bits) if gbit in keep_bits_set]
        if len(keep_pos) == 0:
            continue
        counts = table[:, lut_id, :]
        reduced, kept_local_pos = _reduce_counts_for_lut(counts, keep_pos, n_addr_bits)
        m_i = int(np.log2(reduced.shape[1]))
        if m_i == 0:
            continue

        # record the corresponding global bits（order= local asc）
        kept_global_bits = [lut_bits[p] for p in kept_local_pos]

        reduced_list.append(reduced)
        kept_local_pos_list.append(kept_local_pos)
        kept_global_bits_list.append(kept_global_bits)
        m_list.append(m_i)
        active_ids.append(lut_id)

    if len(reduced_list) == 0:
        raise ValueError("No active LUTs after bit reduction. Check keep_bits_set / mapping consistency.")

    # decide target_m
    if align_strategy == "min":
        target_m = int(min(m_list))
    elif align_strategy == "topk_by_bit_priority":
        if bit_priority is None:
            raise ValueError("align_strategy='topk_by_bit_priority' requires bit_priority.")

        target_m = int(min(m_list))
    else:
        raise ValueError("align_strategy must be 'min' or 'topk_by_bit_priority'.")

    aligned_tables = []
    aligned_bits_per_lut = []

    for reduced, kept_local_pos, kept_global_bits in zip(reduced_list,
                                                         kept_local_pos_list,
                                                         kept_global_bits_list):
        mi = int(np.log2(reduced.shape[1]))
        arr = reduced
        gbits = kept_global_bits

        if mi > target_m:
            #  mi -> target_m
            if align_strategy == "min":
                arr = arr.reshape(C, *([2] * mi))
                drop_cnt = mi - target_m
                arr = arr.sum(axis=tuple(range(-1, -1 - drop_cnt, -1)))
                arr = arr.reshape(C, -1)
                gbits = gbits[:target_m]  # keep the first target_m

            else:  # "topk_by_bit_priority"
                # for each LUT's kept_global_bits, order by bit_priority, keep the target_m with the highest score
                ranked = sorted(gbits, key=lambda g: bit_priority[g], reverse=True)
                keep_gbits = set(ranked[:target_m])

                # delete local position（by gbits order）
                drop_local_idxs = [i for i, g in enumerate(gbits) if g not in keep_gbits]

                arr = arr.reshape(C, *([2] * mi))
                for dp in sorted(drop_local_idxs, reverse=True):
                    arr = arr.sum(axis=1 + dp, keepdims=False)
                arr = arr.reshape(C, -1)
                # order the keeped global bits：filtered bits with local order
                gbits = [g for g in gbits if g in keep_gbits]

        elif mi < target_m:
            # dummy
            continue

        # align
        aligned_tables.append(arr)
        aligned_bits_per_lut.append(gbits)

    if len(aligned_tables) == 0:
        raise ValueError("No aligned LUTs remain after address width alignment.")

    lut_table = np.stack(aligned_tables, axis=1).astype(np.float32)  # (C, L_active_aligned, 2^target_m)

    profile = dict(
        lut_table=lut_table,
        kept_global_bits_per_lut=aligned_bits_per_lut,  # each addr=target_m
        addr_bits=target_m,
        num_classes=C,
        alpha=float(getattr(model, "alpha", 1.0)),
        active_lut_ids=active_ids
    )
    return profile


# ------------------ find local(per-LUT) top-k based on coverage ------------------
def select_local_bits_with_coverage(lut_bits: List[int],
                                    bit_priority: np.ndarray,
                                    k: int,
                                    r_per_bucket: int = 1,
                                    bucket_mapper=bucket_mapper_mnist_thermo) -> List[int]:
    """
    lut_bits: the LUT global bits tabel
    k: # final keep bits
    r_per_bucket: each bucket picks r
    return：local address index
    """
    n = len(lut_bits)
    k = max(1, min(k, n))

    # order by bit_priority
    local_scored = [(p, float(bit_priority[lut_bits[p]])) for p in range(n)]
    local_scored.sort(key=lambda x: x[1], reverse=True)

    # split into buckets
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

    # make the #keeps to k
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

