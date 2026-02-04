import numpy as np


def compute_lut_priority_entropy(model, eps: float = 1e-9):
    """
    Compute a per-LUT priority score using only the trained table,
    no validation pass, no full inference.

    Intuition:
    - For each LUT ℓ, we ask:
        "Does this LUT clearly fire for one class more than others?"
      If yes, it's discriminative -> keep.
      If it's equally likely to vote for any class, it's noisy -> drop.

    Steps:
    For LUT ℓ:
      class_hist[c] = sum_over_addr table[c, ℓ, :]
      p[c] = class_hist[c] / sum_c class_hist[c]
      entropy = -Σ_c p[c] * log(p[c])
      priority[ℓ] = -entropy  (lower entropy -> higher priority)

    Returns:
      priority: np.ndarray of shape (L,), float32, higher is better
    """

    table = model.table  # shape (C, L, A)
    C, L, A = table.shape
    # sum over address dimension to get per-class counts for each LUT:
    # class_hist[c, ℓ] = Σ_addr table[c, ℓ, addr]
    class_hist = np.sum(table, axis=2, dtype=np.float64)  # shape (C, L)

    # avoid div0
    lut_totals = np.sum(class_hist, axis=0) + eps          # shape (L,)
    p = class_hist / lut_totals[None, :]                   # shape (C, L)

    # entropy[ℓ] = - Σ_c p[c,ℓ] log(p[c,ℓ])
    entropy = -np.sum(p * np.log(p + eps), axis=0)         # shape (L,)

    priority = -entropy.astype(np.float32)                 # higher = more class-specific
    return priority
