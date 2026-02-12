import torch
import torch.nn.functional as F

@torch.no_grad()
def exit_metrics(exit_head, X_test, y_test, device, thr_list=(0.0, 0.5, 1.0, 2.0, 4.0)):
    exit_head.eval()
    logits = exit_head(X_test.to(device)).cpu()
    y = y_test.cpu()

    # margin on logits (not softmax): consistent with你之前做法
    top2 = torch.topk(logits, k=2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]

    base_acc = (logits.argmax(dim=-1) == y).float().mean().item()

    out = {
        "exit1_acc": base_acc,
        "margin_mean": margin.mean().item(),
        "margin_p95": margin.kthvalue(int(0.95 * (margin.numel()-1)) + 1).values.item(),
        "by_thr": {}
    }

    for thr in thr_list:
        mask = margin > thr
        exit_rate = mask.float().mean().item()
        # 注意：這裡是 “如果用這個 thr 就 exit”，但 overall_acc 需要搭配 final logits 才有意義
        # 目前先給你 exit head 自己的準確率（只在 exit subset 上）
        if mask.any():
            acc_on_exit = (logits[mask].argmax(dim=-1) == y[mask]).float().mean().item()
        else:
            acc_on_exit = float("nan")
        out["by_thr"][float(thr)] = {"exit_rate": exit_rate, "acc_on_exit": acc_on_exit}

    return out