import numpy as np

def fold_addresses_np(tables, N, M, endianness="little"):
    """
    tables: np.ndarray, shape (C, L, 2**N), dtype=np.uint16/32
    return: np.ndarray, shape (C, L, 2**M)
    """
    C, L, S = tables.shape
    assert S == (1 << N)
    newS = 1 << M
    out = np.zeros((C, L, newS), dtype=tables.dtype)

    # 建立查表：每個 child addr 對應 parent addr
    idx = np.arange(S, dtype=np.uint32)
    if endianness == "little":
        parent = idx & ((1 << M) - 1)
    else:
        parent = idx >> (N - M)

    # each LUT group-by sum
    for c in range(C):
        for l in range(L):
            np.add.at(out[c, l], parent, tables[c, l])
    return out


def build_denylist_per_lut(stats, freq_threshold=1e-4, topk_guard=128): ...
