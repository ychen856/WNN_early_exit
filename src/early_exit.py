import torch
import torch.nn.functional as F

@torch.no_grad()
def eval_exit1_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        _, exit1_logits, _ = model.forward_with_all_hidden_and_exits(xb)
        loss = F.cross_entropy(exit1_logits, yb)
        total_loss += loss.item() * xb.size(0)
        pred = exit1_logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def infer_with_early_exit(model, x_bits, thr=0.3):
    model.eval()
    # get h1
    h = model.layers[0](x_bits)
    h_used = h[:, model.exit1_keep_idx]
    h_exit = (h_used - model.exit1_mu) / model.exit1_sigma
    logits1 = model.exit1_classifier(h_exit) / model.exit_tau

    top2 = torch.topk(logits1, k=2, dim=-1).values

    margin = top2[:, 0] - top2[:, 1]
    exit_mask = margin > thr
    if exit_mask.all():
        return logits1, exit_mask

    # for the rest, run full model
    # (最簡單寫法：直接算 full，再混合。要更快就只對沒 exit 的子 batch 繼續跑第二層。)
    logits_full = model(x_bits)
    logits = logits_full.clone()
    logits[exit_mask] = logits1[exit_mask]
    return logits, exit_mask

@torch.no_grad()
def eval_epoch_w_exit(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    exit_count = 0
    sample_count = 0

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits, exit_mask = infer_with_early_exit(model, xb, thr=0.3)
        loss = F.cross_entropy(logits, yb)

        preds = logits.argmax(dim=1)
        exit_count += exit_mask.sum().item()
        sample_count += exit_mask.numel()
        total_correct += (preds == yb).sum().item()
        total_samples += yb.numel()
        total_loss += loss.item() * yb.numel()

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    exit_rate = exit_count / sample_count
    return avg_loss, acc, exit_rate


@torch.no_grad()
def analyze_h1_stats(model, loader, device, num_batches=5, thr=0.5):
    model.eval()
    all_h = []
    for i, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        h1 = model.layers[0](xb).float()   # [B, 2000]
        all_h.append(h1.cpu())
        if i + 1 >= num_batches:
            break

    H = torch.cat(all_h, dim=0)  # [N, 2000]
    # per-dim stats
    mean_per_dim = H.mean(dim=0)                 # [2000]
    std_per_dim  = H.std(dim=0, unbiased=False)  # [2000]
    p1_per_dim   = (H > thr).float().mean(dim=0) # fraction of ones per dim

    # global
    print(f"H shape: {tuple(H.shape)}")
    print(f"Global mean: {H.mean().item():.6f}, global std: {H.std(unbiased=False).item():.6f}")
    print(f"Global ones rate (thr={thr}): {(H>thr).float().mean().item():.6f}")

    # find most informative dims (most biased away from 0.5)
    bias = (p1_per_dim - 0.5).abs()
    topk = torch.topk(bias, k=20)

    print("\nTop-20 dims by |P(h>thr)-0.5| (more biased => potentially more signal):")
    for rank in range(20):
        j = topk.indices[rank].item()
        print(f"  dim {j:4d}: p1={p1_per_dim[j].item():.4f}, mean={mean_per_dim[j].item():.4f}, std={std_per_dim[j].item():.6f}, bias={bias[j].item():.4f}")

    # also show the flattest dims (closest to coin flip)
    lowk = torch.topk(bias, k=10, largest=False)
    print("\nBottom-10 dims by |P(h>thr)-0.5| (closest to coin flip):")
    for rank in range(10):
        j = lowk.indices[rank].item()
        print(f"  dim {j:4d}: p1={p1_per_dim[j].item():.4f}, mean={mean_per_dim[j].item():.4f}, std={std_per_dim[j].item():.6f}, bias={bias[j].item():.4f}")

    return {
        "H": H,
        "mean_per_dim": mean_per_dim,
        "std_per_dim": std_per_dim,
        "p1_per_dim": p1_per_dim,
        "bias": bias,
    }


@torch.no_grad()
def eval_exit_metrics(model, loader, device, thr_list=(0.0, 0.5, 1.0, 2.0, 4.0)):
    model.eval()

    total = 0
    correct_exit = 0

    # collect margins to compute stats
    all_margins = []

    # per-threshold counters
    exit_cnt = {thr: 0 for thr in thr_list}
    correct_overall = {thr: 0 for thr in thr_list}

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        B = xb.size(0)
        total += B

        # forward: get exit logits (and also final logits)
        final_logits, exit1_logits, _ = model.forward_with_all_hidden_and_exits(xb)

        # exit-only accuracy
        pred_exit = exit1_logits.argmax(dim=-1)
        correct_exit += (pred_exit == yb).sum().item()

        # logit margin
        top2 = torch.topk(exit1_logits, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]   # [B]
        all_margins.append(margin.detach().cpu())

        # for each threshold, mix exit vs final
        for thr in thr_list:
            mask = margin > thr
            exit_cnt[thr] += mask.sum().item()

            mixed = final_logits.clone()
            mixed[mask] = exit1_logits[mask]

            pred = mixed.argmax(dim=-1)
            correct_overall[thr] += (pred == yb).sum().item()

    all_margins = torch.cat(all_margins, dim=0)
    exit1_acc = correct_exit / total
    margin_mean = all_margins.mean().item()
    margin_p95 = torch.quantile(all_margins, 0.95).item()

    results = {
        "exit1_acc": exit1_acc,
        "margin_mean": margin_mean,
        "margin_p95": margin_p95,
        "by_thr": {}
    }
    for thr in thr_list:
        results["by_thr"][thr] = {
            "exit_rate": exit_cnt[thr] / total,
            "overall_acc": correct_overall[thr] / total
        }
    return results




# -----------------------------
# 1) Extract + cache H_used
# -----------------------------
@torch.no_grad()
def cache_exit1_features(
    model,
    loader,
    device,
    exit1_keep_idx: torch.Tensor,
    max_batches=None,
    normalize=True,
):
    """
    Returns:
      X: [N, K] cached features
      Y: [N] labels
      mu, sigma: [K] (if normalize=True) else (None, None)
    """
    model.eval()
    K = exit1_keep_idx.numel()
    feats = []
    labels = []

    # Ensure keep_idx on device for slicing
    keep_idx_dev = exit1_keep_idx.to(device)

    for bi, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        yb = yb.to(device)

        # h1
        h1 = model.layers[0](xb).float()          # [B, 2000]
        h_used = h1[:, keep_idx_dev]             # [B, K]

        feats.append(h_used.detach().cpu())
        labels.append(yb.detach().cpu())

        if max_batches is not None and (bi + 1) >= max_batches:
            break

    X = torch.cat(feats, dim=0)   # [N, K]
    Y = torch.cat(labels, dim=0)  # [N]

    if normalize:
        mu = X.mean(dim=0)  # [K]
        sigma = X.std(dim=0, unbiased=False).clamp_min(1e-6)
        X = (X - mu) / sigma
        return X, Y, mu, sigma
    else:
        return X, Y, None, None



# -----------------------------
# 3) Evaluate exit metrics using cached classifier + full model
# -----------------------------
@torch.no_grad()
def eval_cached_exit_metrics(
    model,
    clf,                  # trained nn.Linear on cached features
    loader,
    device,
    exit1_keep_idx,
    mu=None,
    sigma=None,
    thr_list=(0.0, 0.5, 1.0, 2.0, 4.0),
):
    """
    Computes:
      - exit1_acc (exit head only)
      - margin stats
      - exit_rate & overall_acc for different thresholds
    Uses logit margin (top1-top2) for gating.
    """
    model.eval()
    clf.eval()

    keep_idx_dev = exit1_keep_idx.to(device)

    total = 0
    correct_exit = 0
    all_margins = []

    exit_cnt = {thr: 0 for thr in thr_list}
    correct_overall = {thr: 0 for thr in thr_list}

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        B = xb.size(0)
        total += B

        # full model final logits
        final_logits = model(xb)

        # h1 -> h_used -> (optional normalize) -> exit logits
        h1 = model.layers[0](xb).float()
        h_used = h1[:, keep_idx_dev]  # [B, K]
        if (mu is not None) and (sigma is not None):
            mu_dev = mu.to(device)
            sigma_dev = sigma.to(device)
            h_used = (h_used - mu_dev) / sigma_dev

        exit1_logits = clf(h_used)

        # exit-only acc
        pred_exit = exit1_logits.argmax(dim=-1)
        correct_exit += (pred_exit == yb).sum().item()

        # margins
        top2 = torch.topk(exit1_logits, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        all_margins.append(margin.detach().cpu())

        # mix for thresholds
        for thr in thr_list:
            mask = margin > thr
            exit_cnt[thr] += mask.sum().item()

            mixed = final_logits.clone()
            mixed[mask] = exit1_logits[mask]
            pred = mixed.argmax(dim=-1)
            correct_overall[thr] += (pred == yb).sum().item()

    all_margins = torch.cat(all_margins, dim=0)
    exit1_acc = correct_exit / total

    results = {
        "exit1_acc": exit1_acc,
        "margin_mean": all_margins.mean().item(),
        "margin_p95": torch.quantile(all_margins, 0.95).item(),
        "by_thr": {}
    }
    for thr in thr_list:
        results["by_thr"][thr] = {
            "exit_rate": exit_cnt[thr] / total,
            "overall_acc": correct_overall[thr] / total
        }
    return results


