import torch

'''@torch.no_grad()
def collect_hidden(model, loader, device, layer_idx: int, num_batches: int = 20):
    model.eval()
    Hs = []
    for bi, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        _, h_list = model.forward_with_all_hidden(xb)  # 你已有
        Hs.append(h_list[layer_idx].detach().cpu())
        if num_batches is not None and (bi + 1) >= num_batches:
            break
    return torch.cat(Hs, dim=0)  # [N, D]

@torch.no_grad()
def analyze_hidden_stats(model, loader, device, layer_idx: int, thr: float = 0.5, num_batches: int = 5):
    H = collect_hidden(model, loader, device, layer_idx=layer_idx, num_batches=num_batches)  # [N,D]
    mean_per_dim = H.mean(dim=0)
    std_per_dim = H.std(dim=0)
    p1_per_dim = (H > thr).float().mean(dim=0)
    bias = (p1_per_dim - 0.5).abs()

    return {
        "H": H,  # 可能很大：如果你不想回傳就拿掉
        "mean_per_dim": mean_per_dim,
        "std_per_dim": std_per_dim,
        "p1_per_dim": p1_per_dim,
        "bias": bias,
    }

def pick_keep_idx(stats: dict, K: int, mode: str = "p*(1-p)*std"):
    """
    mode options:
      - "bias"
      - "bias*std"
      - "p*(1-p)*std"  (你目前用的)
    """
    p = stats["p1_per_dim"]
    s = stats["std_per_dim"]
    b = stats["bias"]

    if mode == "bias":
        score = b
    elif mode == "bias*std":
        score = b * s
    elif mode == "p*(1-p)*std":
        score = (p * (1 - p)) * s
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return torch.topk(score, k=K).indices'''



@torch.no_grad()
def analyze_hidden_for_exit(model, loader, device, layer_idx: int, thr_bin: float = 0.5, num_batches: int = 5):
    """
    回傳:
      mean_per_dim, std_per_dim, p1_per_dim, bias  (all [D] on CPU)
    """
    model.eval()
    Hs = []
    for bi, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        _, h_list = model.forward_with_all_hidden(xb)
        Hs.append(h_list[layer_idx].detach().cpu())
        if (bi + 1) >= num_batches:
            break
    H = torch.cat(Hs, 0)  # [N, D]

    mean_per_dim = H.mean(0)
    std_per_dim = H.std(0)
    p1_per_dim = (H > thr_bin).float().mean(0)
    bias = (p1_per_dim - 0.5).abs()
    return mean_per_dim, std_per_dim, p1_per_dim, bias

def select_keep_idx(mean_per_dim, std_per_dim, p1_per_dim, bias, k: int, keep_mode: str):
    if keep_mode == "bias":
        score = bias
    elif keep_mode == "bias*std":
        score = bias * std_per_dim
    elif keep_mode == "p*(1-p)*std":
        score = (p1_per_dim * (1 - p1_per_dim)) * std_per_dim
    else:
        raise ValueError(f"Unknown keep_mode={keep_mode}")
    return torch.topk(score, k=k).indices  # [k], CPU long

@torch.no_grad()
def compute_mu_sigma(model, loader, device, layer_idx: int, keep_idx: torch.Tensor, num_batches: int = None):
    model.eval()
    Hs = []
    for bi, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        _, h_list = model.forward_with_all_hidden(xb)
        h = h_list[layer_idx].detach().cpu()[:, keep_idx]
        Hs.append(h)
        if num_batches is not None and (bi + 1) >= num_batches:
            break
    H = torch.cat(Hs, 0)  # [N,k]
    mu = H.mean(0)
    sigma = H.std(0).clamp_min(1e-6)
    return mu, sigma