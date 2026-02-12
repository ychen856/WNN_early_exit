import torch
import torch.nn.functional as F

@torch.no_grad()
def cache_features(model, loader, device, layer_idx: int, keep_idx=None):
    model.eval()
    Xs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        _, h_list = model.forward_with_all_hidden(xb)
        h = h_list[layer_idx]
        if keep_idx is not None:
            h = h[:, keep_idx.to(h.device)]
        Xs.append(h.detach().cpu())
        ys.append(yb.detach().cpu())
    return torch.cat(Xs, 0), torch.cat(ys, 0)

def train_exit_head_cached(exit_head, X_train, y_train, X_val, y_val,
                           device, num_epochs=50, lr=3e-3, weight_decay=1e-3, batch_size=512):
    exit_head.to(device)
    opt = torch.optim.AdamW(exit_head.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = 0.0

    N = X_train.size(0)

    for ep in range(num_epochs):
        exit_head.train()
        perm = torch.randperm(N)
        total_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)

            opt.zero_grad()
            logits = exit_head(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(dim=-1) == yb).sum().item()
            total += yb.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        exit_head.eval()
        with torch.no_grad():
            logits = exit_head(X_val.to(device))
            val_loss = F.cross_entropy(logits, y_val.to(device)).item()
            val_acc = (logits.argmax(dim=-1).cpu() == y_val).float().mean().item()

        print(f"[cached-exit] Epoch {ep:03d} | train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
              f"| val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in exit_head.state_dict().items()}

    if best_state is not None:
        exit_head.load_state_dict(best_state)
    return exit_head, best_val