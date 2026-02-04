import numpy as np


def make_tuple_mapping(num_luts, addr_bits, bit_len, tiles=None, seed=42):
    """
    generate tile-aware or global tuple mapping。
    tiles: [(start, end), ...]
    """
    rng = np.random.default_rng(seed)
    mapping = []
    for l in range(num_luts):
        if tiles:
            t = tiles[l % len(tiles)]
            idx_pool = np.arange(t[0], t[1])
        else:
            idx_pool = np.arange(bit_len)
        sel = rng.choice(idx_pool, size=addr_bits, replace=False)
        mapping.append(sel.tolist())
    return mapping


def audit_mapping(mapping, bit_len):
    """
    check the mapping covering rate and duplicaation, for sanity check
    """
    hits = np.zeros(bit_len, dtype=int)
    for sel in mapping:
        hits[sel] += 1
    return {
        "min_hits": int(hits.min()),
        "max_hits": int(hits.max()),
        "mean_hits": float(hits.mean()),
        "total_bits": int(bit_len)
    }


