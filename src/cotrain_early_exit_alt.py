import argparse
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.linearExitHead import build_exits_from_ckpt
from src.core.multiLayerWNN import build_backbone_from_ckpt, save_ckpt_v2
from src.dataio.data import build_loaders_bits
from src.early_exit import (
    _head_logits_from_hidden_trainable,
    _margin_from_logits,
    eval_cascade_multi_exit,
)
from src.tools.utils import _head_logits_from_hidden
from test.eval import eval_epoch


@dataclass
class MultiExitModelConfig:
    payload_exit_cfg: List[dict]
    train_layer_indices: Optional[Sequence[int]] = None
    freeze_layer_indices: Sequence[int] = ()
    train_exit_ids: Sequence[int] = ()


@dataclass
class MultiExitLossConfig:
    lambda_final: float = 1.0
    lambda_exits: Optional[Sequence[float]] = None
    use_gate_weighting: bool = True
    gate_T: Optional[Sequence[float]] = None
    use_prob_margin: bool = False
    beta_tail: float = 0.0
    quota_cfg: Optional[dict] = None


@dataclass
class AlternatingPhaseConfig:
    cycles: int = 3
    epochs_F: int = 1
    epochs_H: int = 1
    final_train_layers: Sequence[int] = ()
    final_train_classifier: bool = True
    train_exit_ids: Sequence[int] = ()
    lr_backbone_F: float = 1e-5
    lr_classifier_F: float = 1e-5
    lr_exits_H: float = 5e-5


@dataclass
class MultiExitOptimConfig:
    lr_backbone: float = 1e-4
    lr_classifier: float = 3e-4
    lr_exits: float = 5e-4
    weight_decay: float = 1e-3
    grad_clip: Optional[float] = 1.0


@dataclass
class StageConfig:
    name: str
    epochs: int
    train_layers: Optional[Sequence[int]] = None
    freeze_layers: Optional[Sequence[int]] = None
    train_exit_ids: Optional[Sequence[int]] = None
    train_classifier: bool = True
    lr_backbone: Optional[float] = None
    lr_classifier: Optional[float] = None
    lr_exits: Optional[float] = None


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def _clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _resolve_train_layers(model, train_layers: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if train_layers is None:
        return tuple(range(len(model.layers)))
    return tuple(int(x) for x in train_layers)


def _resolve_lambda_exits(lambda_exits: Optional[Sequence[float]], num_exits: int) -> List[float]:
    if lambda_exits is None:
        return [0.05] * num_exits
    if len(lambda_exits) != num_exits:
        raise ValueError(f"lambda_exits must match num_exits, got {len(lambda_exits)} vs {num_exits}")
    return [float(x) for x in lambda_exits]


def _resolve_gate_T(gate_T: Optional[Sequence[float]], num_exits: int) -> List[float]:
    if gate_T is None:
        return [1.0] * num_exits
    if len(gate_T) != num_exits:
        raise ValueError(f"gate_T must match num_exits, got {len(gate_T)} vs {num_exits}")
    return [float(x) for x in gate_T]


def _safe_float(x: float, fallback: float = 0.0) -> float:
    x = float(x)
    return fallback if x != x else x


def compute_exit_loss_by_layer(
    *,
    layer_idx: int,
    exit_logits: torch.Tensor,
    final_logits: torch.Tensor,
    yb: torch.Tensor,
    cfg: Dict[str, Any],
):
    mode = cfg.get("mode", "baseline")
    override = dict(cfg.get("override", {}))

    kd_T = float(override.get("kd_T", 2.0))
    lambda_kd = float(override.get("lambda_kd", 0.7))

    loss_ce = F.cross_entropy(exit_logits, yb)
    loss = loss_ce

    if mode in {"kd", "kd_final_correct"}:
        with torch.no_grad():
            teacher_prob = F.softmax(final_logits / kd_T, dim=-1)
            final_pred = final_logits.argmax(dim=-1)
            final_correct = final_pred.eq(yb)

        student_log_prob = F.log_softmax(exit_logits / kd_T, dim=-1)
        if mode == "kd_final_correct":
            if final_correct.any():
                loss_kd = F.kl_div(
                    student_log_prob[final_correct],
                    teacher_prob[final_correct],
                    reduction="batchmean",
                ) * (kd_T * kd_T)
            else:
                loss_kd = exit_logits.new_zeros(())
        else:
            loss_kd = F.kl_div(
                student_log_prob,
                teacher_prob,
                reduction="batchmean",
            ) * (kd_T * kd_T)
        loss = loss + lambda_kd * loss_kd
    elif mode != "baseline":
        raise ValueError(f"Unsupported exit loss mode at layer {layer_idx}: {mode}")

    return loss


def configure_trainable_params(
    model,
    exit_heads,
    train_layers,
    train_exit_ids,
    train_classifier=True,
):
    set_requires_grad(model, False)
    for h in exit_heads:
        set_requires_grad(h, False)

    train_layers = tuple(int(li) for li in train_layers)
    train_exit_ids = tuple(int(i) for i in train_exit_ids)

    for li in train_layers:
        if li < 0 or li >= len(model.layers):
            raise ValueError(f"train_layers contains out-of-range layer {li}")
        set_requires_grad(model.layers[li], True)

    if train_classifier:
        set_requires_grad(model.classifier, True)

    for i in train_exit_ids:
        if i < 0 or i >= len(exit_heads):
            raise ValueError(f"train_exit_ids contains out-of-range exit {i}")
        set_requires_grad(exit_heads[i], True)

    params_backbone = [p for li in train_layers for p in model.layers[li].parameters() if p.requires_grad]
    params_classifier = [p for p in model.classifier.parameters() if p.requires_grad]
    params_exits = [p for i in train_exit_ids for p in exit_heads[i].parameters() if p.requires_grad]
    total_trainable = (
        sum(p.numel() for p in params_backbone)
        + sum(p.numel() for p in params_classifier)
        + sum(p.numel() for p in params_exits)
    )

    print(
        f"[alt] trainable params | layers={train_layers} exits={train_exit_ids} "
        f"| backbone={sum(p.numel() for p in params_backbone)} "
        f"| classifier={sum(p.numel() for p in params_classifier)} "
        f"| exits={sum(p.numel() for p in params_exits)} "
        f"| total={total_trainable}"
    )


def build_optimizer_for_stage(
    model,
    exit_heads,
    train_layers,
    train_exit_ids,
    lr_backbone,
    lr_classifier,
    lr_exits,
    weight_decay,
):
    train_layers = tuple(int(li) for li in train_layers)
    train_exit_ids = tuple(int(i) for i in train_exit_ids)

    params_backbone = [p for li in train_layers for p in model.layers[li].parameters() if p.requires_grad]
    params_classifier = [p for p in model.classifier.parameters() if p.requires_grad]
    params_exits = [p for i in train_exit_ids for p in exit_heads[i].parameters() if p.requires_grad]

    groups = []
    if params_backbone:
        groups.append({"params": params_backbone, "lr": lr_backbone, "weight_decay": weight_decay})
    if params_classifier:
        groups.append({"params": params_classifier, "lr": lr_classifier, "weight_decay": weight_decay})
    if params_exits:
        groups.append({"params": params_exits, "lr": lr_exits, "weight_decay": weight_decay})
    if not groups:
        raise ValueError("No trainable parameters found for this stage.")

    optimizer = torch.optim.AdamW(groups)
    return optimizer, params_backbone, params_classifier, params_exits


def _compute_tail_mask(
    h_list,
    yb,
    *,
    exit_heads,
    payload_exit_cfg,
    thrs_train,
    device,
    use_prob_margin,
):
    tail_mask = torch.ones_like(yb, dtype=torch.bool)
    for i, cfg in enumerate(payload_exit_cfg):
        li = int(cfg["layer_idx"])
        logits_i = _head_logits_from_hidden(exit_heads[i], h_list[li], device)
        margin_i = _margin_from_logits(logits_i, use_prob=use_prob_margin)
        take_i = tail_mask & (margin_i > float(thrs_train[i]))
        tail_mask = tail_mask & (~take_i)
    return tail_mask


def train_one_epoch_multi_exit(
    model,
    train_loader,
    device,
    *,
    optimizer,
    exit_heads,
    payload_exit_cfg,
    thrs_train,
    lambda_final,
    lambda_exits,
    use_gate_weighting,
    gate_T,
    use_prob_margin,
    beta_tail,
    grad_clip,
):
    model.train()
    for h in exit_heads:
        h.train()

    eps = 1e-8
    num_exits = len(exit_heads)
    total_loss = 0.0
    total_final = 0.0
    total_exit = 0.0
    num_batches = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)

        final_logits, h_list = model.forward_with_all_hidden(xb)
        ce_final = F.cross_entropy(final_logits, yb, reduction="none")
        ce_all = ce_final.mean()
        loss_final = ce_all

        loss_exit_sum = 0.0
        if use_gate_weighting:
            ce_exit_list = []
            margin_list = []
            for i in range(num_exits):
                li = int(payload_exit_cfg[i]["layer_idx"])
                logits_i = _head_logits_from_hidden_trainable(exit_heads[i], h_list[li], device)
                ce_exit_list.append(F.cross_entropy(logits_i, yb, reduction="none"))
                margin_list.append(_margin_from_logits(logits_i, use_prob=use_prob_margin))

            undecided = torch.ones_like(ce_final)
            for i in range(num_exits):
                if not any(p.requires_grad for p in exit_heads[i].parameters()):
                    weight_i = torch.sigmoid((margin_list[i] - float(thrs_train[i])) / float(gate_T[i]))
                    undecided = undecided * (1.0 - weight_i)
                    continue

                weight_i = torch.sigmoid((margin_list[i] - float(thrs_train[i])) / float(gate_T[i]))
                take_i = undecided * weight_i
                take_i_det = take_i.detach()
                loss_i = (take_i_det * ce_exit_list[i]).sum() / (take_i_det.sum() + eps)
                loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * loss_i
                undecided = undecided * (1.0 - weight_i)

            undecided_det = undecided.detach()
            loss_final = (undecided_det * ce_final).sum() / (undecided_det.sum() + eps)
        else:
            for i in range(num_exits):
                if not any(p.requires_grad for p in exit_heads[i].parameters()):
                    continue
                li = int(payload_exit_cfg[i]["layer_idx"])
                logits_i = _head_logits_from_hidden_trainable(exit_heads[i], h_list[li], device)
                loss_i = F.cross_entropy(logits_i, yb)
                loss_exit_sum = loss_exit_sum + float(lambda_exits[i]) * loss_i

        if beta_tail > 0.0:
            with torch.no_grad():
                tail_mask = _compute_tail_mask(
                    h_list,
                    yb,
                    exit_heads=exit_heads,
                    payload_exit_cfg=payload_exit_cfg,
                    thrs_train=thrs_train,
                    device=device,
                    use_prob_margin=use_prob_margin,
                )
            if tail_mask.any():
                ce_tail = F.cross_entropy(final_logits[tail_mask], yb[tail_mask])
            else:
                ce_tail = 0.0 * ce_all
            loss_final = loss_final + float(beta_tail) * ce_tail

        loss = float(lambda_final) * loss_final + loss_exit_sum
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                grad_clip,
            )

        optimizer.step()

        total_loss += float(loss.detach().item())
        total_final += float(loss_final.detach().item())
        total_exit += float(loss_exit_sum.detach().item()) if torch.is_tensor(loss_exit_sum) else float(loss_exit_sum)
        num_batches += 1

    denom = max(num_batches, 1)
    return {
        "loss": total_loss / denom,
        "loss_final": total_final / denom,
        "loss_exit": total_exit / denom,
    }


@torch.no_grad()
def _eval_final_only_on_tail(
    model,
    val_loader,
    device,
    *,
    exit_heads,
    payload_exit_cfg,
    thrs_tail_anchor,
    use_prob_margin,
):
    model.eval()
    for h in exit_heads:
        h.eval()

    correct = 0
    total = 0
    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        final_logits, h_list = model.forward_with_all_hidden(xb)
        tail_mask = _compute_tail_mask(
            h_list,
            yb,
            exit_heads=exit_heads,
            payload_exit_cfg=payload_exit_cfg,
            thrs_train=thrs_tail_anchor,
            device=device,
            use_prob_margin=use_prob_margin,
        )
        if tail_mask.any():
            pred = final_logits[tail_mask].argmax(dim=1)
            correct += int((pred == yb[tail_mask]).sum().item())
            total += int(tail_mask.sum().item())

    acc = float(correct / total) if total > 0 else float("nan")
    return {"final_tail_acc": acc, "tail_count": int(total)}


def evaluate_multi_exit_bundle(
    model,
    val_loader,
    device,
    exit_heads,
    payload_exit_cfg,
    thrs_eval_list,
    thrs_tail_anchor,
    best_eval_idx: int = 0,
    use_prob_margin: bool = False,
):
    eval_records = []
    thrs_eval_list = [tuple(x) for x in thrs_eval_list]
    if not (0 <= best_eval_idx < len(thrs_eval_list)):
        raise ValueError(f"best_eval_idx out of range: {best_eval_idx}")
    if len(thrs_tail_anchor) != len(exit_heads):
        raise ValueError("thrs_tail_anchor must match num_exits")

    for k, thrs_eval in enumerate(thrs_eval_list):
        if len(thrs_eval) != len(exit_heads):
            raise ValueError("Each thrs_eval must match num_exits")
        out_val = eval_cascade_multi_exit(
            model,
            val_loader,
            device,
            exit_heads=exit_heads,
            exit_cfg_list=payload_exit_cfg,
            thrs=thrs_eval,
            use_prob_margin=use_prob_margin,
            log_margins=False,
        )
        tail_stats = _eval_final_only_on_tail(
            model,
            val_loader,
            device,
            exit_heads=exit_heads,
            payload_exit_cfg=payload_exit_cfg,
            thrs_tail_anchor=tuple(thrs_tail_anchor),
            use_prob_margin=use_prob_margin,
        )

        record = {
            "eval_idx": k,
            "thrs": tuple(thrs_eval),
            "overall_acc": float(out_val["overall_acc"]),
            "final_acc": float(out_val.get("final_acc", 0.0)),
            "final_tail_acc": float(tail_stats["final_tail_acc"]),
            "tail_count": int(tail_stats["tail_count"]),
            "final_rate": float(out_val["final_rate"]),
            "exit_rates": list(out_val["exit_rates"]),
            "exit_accs": list(out_val.get("exit_accs", [])) if "exit_accs" in out_val else None,
            "raw": out_val,
        }
        if "non_exit_accs" in out_val:
            record["non_exit_accs"] = list(out_val["non_exit_accs"])
        eval_records.append(record)

    selected = eval_records[best_eval_idx]
    return {"records": eval_records, "selected": selected}


def compute_selection_metric(
    record: Dict[str, Any],
    best_metric: str,
    combo_metric_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    min_exit_accs: Optional[Sequence[Optional[float]]] = None,
    min_tail_count: int = 0,
    baseline_overall: Optional[float] = None,
    baseline_tail_acc: Optional[float] = None,
    max_overall_drop: float = 0.005,
    max_tail_drop: float = 0.01,
) -> float:
    overall = float(record["overall_acc"])
    final_only = float(record["final_acc"])
    final_tail = _safe_float(record["final_tail_acc"], fallback=-1.0)
    normalized_exit_rate = float(sum(record.get("exit_rates", [])))
    if best_metric == "val_overall_acc":
        metric = overall
    elif best_metric == "val_final_only":
        metric = final_only
    elif best_metric == "val_final_tail_only":
        metric = final_tail
    elif best_metric == "val_combo":
        w_overall, w_tail, w_exit = combo_metric_weights
        metric = w_overall * overall + w_tail * final_tail + w_exit * normalized_exit_rate
    else:
        raise ValueError(
            "best_metric must be one of "
            "['val_overall_acc', 'val_final_only', 'val_final_tail_only', 'val_combo']"
        )

    if best_metric == "val_final_tail_only" and int(record["tail_count"]) < int(min_tail_count):
        return -1.0

    if min_exit_accs is not None and record.get("exit_accs") is not None:
        exit_accs = record["exit_accs"]
        for i, min_acc in enumerate(min_exit_accs):
            if min_acc is None:
                continue
            if i >= len(exit_accs):
                continue
            if exit_accs[i] < min_acc:
                return -1.0
    if baseline_overall is not None and overall < float(baseline_overall) - float(max_overall_drop):
        return -1.0
    if baseline_tail_acc is not None and final_tail < float(baseline_tail_acc) - float(max_tail_drop):
        return -1.0
    return metric


def cotrain_g3_multi_exit_staged(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    *,
    exit_heads: List[nn.Module],
    model_cfg: MultiExitModelConfig,
    loss_cfg: MultiExitLossConfig,
    optim_cfg: MultiExitOptimConfig,
    stage_cfgs: Sequence[StageConfig],
    thrs_train: Sequence[float],
    thrs_eval_list: Optional[Sequence[Sequence[float]]] = None,
    thrs_tail_anchor: Optional[Sequence[float]] = None,
    best_eval_idx: int = 0,
    best_metric: str = "val_combo",
    combo_metric_weights: Tuple[float, float] = (0.7, 0.3),
    min_exit_accs: Optional[Sequence[Optional[float]]] = None,
    min_tail_count: int = 0,
):
    model = model.to(device)
    exit_heads = [h.to(device) for h in exit_heads]

    num_exits = len(exit_heads)
    if len(model_cfg.payload_exit_cfg) != num_exits:
        raise ValueError("exit_heads and payload_exit_cfg must align")
    if len(thrs_train) != num_exits:
        raise ValueError("thrs_train must match num_exits")

    lambda_exits = _resolve_lambda_exits(loss_cfg.lambda_exits, num_exits)
    gate_T = _resolve_gate_T(loss_cfg.gate_T, num_exits)

    default_train_layers = _resolve_train_layers(model, model_cfg.train_layer_indices)
    default_train_exit_ids = tuple(int(x) for x in model_cfg.train_exit_ids)
    default_freeze_layers = tuple(int(x) for x in model_cfg.freeze_layer_indices)

    if thrs_eval_list is None:
        thrs_eval_list = [tuple(thrs_train)]
    else:
        thrs_eval_list = [tuple(x) for x in thrs_eval_list]
    if thrs_tail_anchor is None:
        thrs_tail_anchor = tuple(thrs_train)
    else:
        thrs_tail_anchor = tuple(thrs_tail_anchor)

    best = {
        "metric": -1.0,
        "stage": None,
        "epoch": None,
        "eval_idx": None,
        "selected": None,
        "state_model": None,
        "state_exits": None,
    }

    global_epoch = 0

    for stage in stage_cfgs:
        stage_name = stage.name if isinstance(stage, StageConfig) else stage.get("name", f"stage_{global_epoch}")
        stage_epochs = stage.epochs if isinstance(stage, StageConfig) else int(stage["epochs"])
        train_layers = tuple(
            _resolve_train_layers(
                model,
                stage.train_layers if isinstance(stage, StageConfig) else stage.get("train_layers", default_train_layers),
            )
        )
        stage_freeze_layers = tuple(
            int(x)
            for x in (
                stage.freeze_layers
                if isinstance(stage, StageConfig)
                else stage.get("freeze_layers", default_freeze_layers)
            )
            or ()
        )
        train_exit_ids = tuple(
            int(x)
            for x in (
                stage.train_exit_ids
                if isinstance(stage, StageConfig)
                else stage.get("train_exit_ids", default_train_exit_ids)
            )
        )
        train_classifier = stage.train_classifier if isinstance(stage, StageConfig) else stage.get("train_classifier", True)

        train_layers = tuple(li for li in train_layers if li not in stage_freeze_layers)

        configure_trainable_params(
            model,
            exit_heads,
            train_layers=train_layers,
            train_exit_ids=train_exit_ids,
            train_classifier=train_classifier,
        )

        lr_backbone = stage.lr_backbone if isinstance(stage, StageConfig) else stage.get("lr_backbone")
        lr_classifier = stage.lr_classifier if isinstance(stage, StageConfig) else stage.get("lr_classifier")
        lr_exits = stage.lr_exits if isinstance(stage, StageConfig) else stage.get("lr_exits")

        optimizer, _, _, _ = build_optimizer_for_stage(
            model,
            exit_heads,
            train_layers=train_layers,
            train_exit_ids=train_exit_ids,
            lr_backbone=optim_cfg.lr_backbone if lr_backbone is None else float(lr_backbone),
            lr_classifier=optim_cfg.lr_classifier if lr_classifier is None else float(lr_classifier),
            lr_exits=optim_cfg.lr_exits if lr_exits is None else float(lr_exits),
            weight_decay=optim_cfg.weight_decay,
        )

        for local_epoch in range(stage_epochs):
            train_stats = train_one_epoch_multi_exit(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                exit_heads=exit_heads,
                payload_exit_cfg=model_cfg.payload_exit_cfg,
                thrs_train=tuple(thrs_train),
                lambda_final=loss_cfg.lambda_final,
                lambda_exits=lambda_exits,
                use_gate_weighting=loss_cfg.use_gate_weighting,
                gate_T=gate_T,
                use_prob_margin=loss_cfg.use_prob_margin,
                beta_tail=loss_cfg.beta_tail,
                grad_clip=optim_cfg.grad_clip,
            )
            eval_stats = evaluate_multi_exit_bundle(
                model,
                val_loader,
                device,
                exit_heads,
                model_cfg.payload_exit_cfg,
                thrs_eval_list,
                thrs_tail_anchor,
                best_eval_idx=best_eval_idx,
                use_prob_margin=loss_cfg.use_prob_margin,
            )

            selected = eval_stats["selected"]
            metric = compute_selection_metric(
                selected,
                best_metric=best_metric,
                combo_metric_weights=combo_metric_weights,
                min_exit_accs=min_exit_accs,
                min_tail_count=min_tail_count,
            )
            exit_accs = selected.get("exit_accs", None)

            print(
                f"[g3-staged:{stage_name}] ep={global_epoch:03d} local={local_epoch:03d} "
                f"| train_loss={train_stats['loss']:.4f} "
                f"| overall={selected['overall_acc']*100:.2f} "
                f"| final={selected['final_acc']*100:.2f} "
                f"| final_tail={selected['final_tail_acc']*100:.2f} "
                f"| final_rate={selected['final_rate']:.4f} "
                f"| exit_rates={selected['exit_rates']} "
                f"| exit_accs={exit_accs} "
                f"| metric={metric:.4f}"
            )

            if metric > best["metric"]:
                best["metric"] = metric
                best["stage"] = stage_name
                best["epoch"] = global_epoch
                best["eval_idx"] = selected["eval_idx"]
                best["selected"] = copy.deepcopy(selected)
                best["state_model"] = _clone_state_dict(model)
                best["state_exits"] = [_clone_state_dict(h) for h in exit_heads]

            global_epoch += 1

    if best["state_model"] is not None:
        model.load_state_dict(best["state_model"], strict=True)
        for i, h in enumerate(exit_heads):
            h.load_state_dict(best["state_exits"][i], strict=True)

    return model, exit_heads, best


def cotrain_multi_exit_alternating(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    *,
    exit_heads: List[nn.Module],
    model_cfg: MultiExitModelConfig,
    optim_cfg: MultiExitOptimConfig,
    alt_cfg: AlternatingPhaseConfig,
    thrs_eval_list: Sequence[Sequence[float]],
    thrs_tail_anchor: Sequence[float],
    exit_loss_by_layer: Dict[int, Dict[str, Any]],
    baseline_overall: float,
    baseline_tail_acc: float,
    best_metric: str = "val_combo",
    best_eval_idx: int = 0,
    combo_metric_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    min_exit_accs: Optional[Sequence[Optional[float]]] = None,
    max_overall_drop: float = 0.005,
    max_tail_drop: float = 0.01,
    use_prob_margin: bool = False,
):
    model = model.to(device)
    exit_heads = [h.to(device) for h in exit_heads]

    initial_eval = evaluate_multi_exit_bundle(
        model,
        val_loader,
        device,
        exit_heads,
        model_cfg.payload_exit_cfg,
        thrs_eval_list,
        thrs_tail_anchor,
        best_eval_idx=best_eval_idx,
        use_prob_margin=use_prob_margin,
    )
    initial_selected = copy.deepcopy(initial_eval["selected"])
    initial_metric = compute_selection_metric(
        initial_selected,
        best_metric=best_metric,
        combo_metric_weights=combo_metric_weights,
        min_exit_accs=min_exit_accs,
        baseline_overall=baseline_overall,
        baseline_tail_acc=baseline_tail_acc,
        max_overall_drop=max_overall_drop,
        max_tail_drop=max_tail_drop,
    )

    best = {
        "metric": initial_metric,
        "cycle": -1,
        "selected": initial_selected,
        "state_model": _clone_state_dict(model),
        "state_exits": [_clone_state_dict(h) for h in exit_heads],
    }

    final_train_layers = tuple(int(x) for x in alt_cfg.final_train_layers)
    train_exit_ids = tuple(int(x) for x in alt_cfg.train_exit_ids)

    for cycle in range(int(alt_cfg.cycles)):
        configure_trainable_params(
            model,
            exit_heads,
            train_layers=final_train_layers,
            train_exit_ids=(),
            train_classifier=bool(alt_cfg.final_train_classifier),
        )
        optimizer_F, _, _, _ = build_optimizer_for_stage(
            model,
            exit_heads,
            train_layers=final_train_layers,
            train_exit_ids=(),
            lr_backbone=float(alt_cfg.lr_backbone_F),
            lr_classifier=float(alt_cfg.lr_classifier_F),
            lr_exits=0.0,
            weight_decay=optim_cfg.weight_decay,
        )

        for epoch_F in range(int(alt_cfg.epochs_F)):
            model.train()
            for h in exit_heads:
                h.eval()

            running_loss = 0.0
            batch_count = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                final_logits, _, _ = model.forward_with_all_hidden_and_exits(xb)
                loss = F.cross_entropy(final_logits, yb)

                optimizer_F.zero_grad(set_to_none=True)
                loss.backward()
                if optim_cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer_F.param_groups for p in group["params"]],
                        optim_cfg.grad_clip,
                    )
                optimizer_F.step()

                running_loss += float(loss.detach().item())
                batch_count += 1

            print(
                f"[alt:F] cycle={cycle:02d} epoch={epoch_F:02d} "
                f"| loss={running_loss / max(batch_count, 1):.4f}"
            )

        configure_trainable_params(
            model,
            exit_heads,
            train_layers=(),
            train_exit_ids=train_exit_ids,
            train_classifier=False,
        )
        optimizer_H, _, _, _ = build_optimizer_for_stage(
            model,
            exit_heads,
            train_layers=(),
            train_exit_ids=train_exit_ids,
            lr_backbone=0.0,
            lr_classifier=0.0,
            lr_exits=float(alt_cfg.lr_exits_H),
            weight_decay=optim_cfg.weight_decay,
        )

        for epoch_H in range(int(alt_cfg.epochs_H)):
            model.eval()
            for h in exit_heads:
                h.train()

            running_loss = 0.0
            batch_count = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                with torch.no_grad():
                    final_logits, h_list = model.forward_with_all_hidden(xb)

                loss_sum = None
                trained_exit_count = 0
                for exit_id, cfg in enumerate(model_cfg.payload_exit_cfg):
                    if exit_id not in train_exit_ids:
                        continue
                    layer_idx = int(cfg["layer_idx"])
                    h_i = h_list[layer_idx].detach()
                    exit_logits = _head_logits_from_hidden_trainable(exit_heads[exit_id], h_i, device)
                    loss_i = compute_exit_loss_by_layer(
                        layer_idx=layer_idx,
                        exit_logits=exit_logits,
                        final_logits=final_logits.detach(),
                        yb=yb,
                        cfg=exit_loss_by_layer[layer_idx],
                    )
                    loss_sum = loss_i if loss_sum is None else loss_sum + loss_i
                    trained_exit_count += 1

                if loss_sum is None or trained_exit_count == 0:
                    raise ValueError("No exit loss was accumulated in H phase.")
                loss = loss_sum / float(trained_exit_count)

                optimizer_H.zero_grad(set_to_none=True)
                loss.backward()
                if optim_cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer_H.param_groups for p in group["params"]],
                        optim_cfg.grad_clip,
                    )
                optimizer_H.step()

                running_loss += float(loss.detach().item())
                batch_count += 1

            print(
                f"[alt:H] cycle={cycle:02d} epoch={epoch_H:02d} "
                f"| loss={running_loss / max(batch_count, 1):.4f}"
            )

        eval_stats = evaluate_multi_exit_bundle(
            model,
            val_loader,
            device,
            exit_heads,
            model_cfg.payload_exit_cfg,
            thrs_eval_list,
            thrs_tail_anchor,
            best_eval_idx=best_eval_idx,
            use_prob_margin=use_prob_margin,
        )
        selected = eval_stats["selected"]
        metric = compute_selection_metric(
            selected,
            best_metric=best_metric,
            combo_metric_weights=combo_metric_weights,
            min_exit_accs=min_exit_accs,
            baseline_overall=baseline_overall,
            baseline_tail_acc=baseline_tail_acc,
            max_overall_drop=max_overall_drop,
            max_tail_drop=max_tail_drop,
        )

        print(
            f"[alt:eval] cycle={cycle:02d} "
            f"| overall={selected['overall_acc']*100:.2f} "
            f"| final={selected['final_acc']*100:.2f} "
            f"| final_tail={_safe_float(selected['final_tail_acc'])*100:.2f} "
            f"| final_rate={selected['final_rate']:.4f} "
            f"| exit_rates={selected['exit_rates']} "
            f"| exit_accs={selected.get('exit_accs', None)} "
            f"| metric={metric:.4f}"
        )

        if metric > best["metric"]:
            best["metric"] = metric
            best["cycle"] = cycle
            best["selected"] = copy.deepcopy(selected)
            best["state_model"] = _clone_state_dict(model)
            best["state_exits"] = [_clone_state_dict(h) for h in exit_heads]

    if best["state_model"] is not None:
        model.load_state_dict(best["state_model"], strict=True)
        for i, h in enumerate(exit_heads):
            h.load_state_dict(best["state_exits"][i], strict=True)

    return model, exit_heads, best



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--backbone_ckpt", type=str, required=True)
    parser.add_argument("--path_out", type=str, required=True, help="Save ckpt with exit_config list")

    parser.add_argument("--exit_layers", type=str, default="0", help='e.g. "0" or "0,1"')
    parser.add_argument("--k", type=str, default="256", help='e.g. "256" or "256,512" (broadcast ok)')
    parser.add_argument("--keep_mode", type=str, default="p*(1-p)*std", help='broadcast ok')
    parser.add_argument("--exit_tau", type=str, default="1.0", help='broadcast ok')

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size_cached", type=int, default=512)
    parser.add_argument("--use_norm", action="store_true", default=True)
    parser.add_argument("--thr", type=str, default="1.0,1.5",
                    help="comma-separated thresholds per exit, e.g. 1.0,1.5")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders
    train_loader, val_loader, test_loader, in_bits, C, ds_meta = build_loaders_bits(
        dataset=args.dataset,
        root="/Users/yi-chunchen/workspace/WNN_early_exit/datasets/",
        batch_size_train=256,
        batch_size_eval=512,
        val_ratio=0.1,
        seed=3,
        z=32,
        device_for_encoding=device,
        shuffle_train=True,
    )

    backbone, bb_cfg, extra = build_backbone_from_ckpt(args.backbone_ckpt, device)
    
    C = int(bb_cfg["num_classes"])

    exit_heads, exit_cfg_list = build_exits_from_ckpt(args.backbone_ckpt, device, num_classes=C)

    test_loss, test_acc = eval_epoch(backbone, test_loader, device)
    print("[final-only] test_acc", test_acc)

    model_cfg = MultiExitModelConfig(
        payload_exit_cfg=[ec.to_payload() for ec in exit_cfg_list],
        train_layer_indices=None,
        freeze_layer_indices=(),
        train_exit_ids=tuple(range(len(exit_heads))),
    )

    '''loss_cfg = MultiExitLossConfig(
        lambda_final=1.0,
        lambda_exits=[0.05] * len(exit_heads),
        use_gate_weighting=True,
        gate_T=[1.0] * len(exit_heads),
        use_prob_margin=False,
        beta_tail=0.0,
        quota_cfg=None,
    ) '''  

    optim_cfg = MultiExitOptimConfig(
        lr_backbone=1e-5,
        lr_classifier=1e-5,
        lr_exits=5e-5,
        weight_decay=args.weight_decay,
        grad_clip=1.0,
    )
    '''alt_cfg = AlternatingPhaseConfig(
        cycles=3,
        epochs_F=1,
        epochs_H=1,
        final_train_layers=[1],
        final_train_classifier=False,
        train_exit_ids=(1, ),
        lr_backbone_F=1e-6,
        lr_classifier_F=0.0, #1e-5,
        lr_exits_H=3e-5,
    )'''

    '''alt_cfg = AlternatingPhaseConfig(
        cycles=3,
        epochs_F=0,
        epochs_H=1,
        final_train_layers=[1],
        final_train_classifier=False,
        train_exit_ids=(1, ),
        lr_backbone_F=1e-6,
        lr_classifier_F=0.0, #1e-5,
        lr_exits_H=3e-5,
    )'''

    alt_cfg = AlternatingPhaseConfig(
        cycles=3,
        epochs_F=0,
        epochs_H=1,
        final_train_layers=[1],
        final_train_classifier=False,
        train_exit_ids=(1, ),
        lr_backbone_F=1e-7,
        lr_classifier_F=0.0, #1e-5,
        lr_exits_H=3e-5,
    )

    use_prob_margin = False
    exit_loss_by_layer = {
        0: {"mode": "kd_final_correct", "override": {"kd_T": 2.0, "lambda_kd": 0.7}},
        1: {"mode": "kd_final_correct", "override": {"kd_T": 2.0, "lambda_kd": 0.7}},
    }
    min_exit_accs = (0.98, 0.98)

    thrs_train = [float(x) for x in args.thr.split(",")]
    thrs_eval_list = [tuple(thrs_train)]
    thrs_tail_anchor = tuple(thrs_train)



    ######################

    print(f'thr eval list: {thrs_eval_list}')
    eval_stats = evaluate_multi_exit_bundle(
                backbone,
                val_loader,
                device,
                exit_heads,
                model_cfg.payload_exit_cfg,
                thrs_eval_list,
                thrs_tail_anchor,
                best_eval_idx=0,
                use_prob_margin=use_prob_margin,
            )

    selected = eval_stats["selected"]
    baseline_overall = float(selected["overall_acc"])
    baseline_tail_acc = _safe_float(selected["final_tail_acc"], fallback=0.0)
    print(
        f"[before training] "
        f"| overall={selected['overall_acc']*100:.2f} "
        f"| final={selected['final_acc']*100:.2f} "
        f"| final_tail={_safe_float(selected['final_tail_acc'])*100:.2f} "
        f"| final_rate={selected['final_rate']:.4f} "
        f"| exit_rates={selected['exit_rates']} "
        f"| exit_accs={selected.get('exit_accs', None)} "
    )

    ######################

    print("Starting co-training with alternating F/H tuning...")

    model, exit_heads, best = cotrain_multi_exit_alternating(
        model=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        exit_heads=exit_heads,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        alt_cfg=alt_cfg,
        thrs_eval_list=thrs_eval_list,
        thrs_tail_anchor=thrs_tail_anchor,
        exit_loss_by_layer=exit_loss_by_layer,
        baseline_overall=baseline_overall,
        baseline_tail_acc=baseline_tail_acc,
        best_metric="val_combo",
        best_eval_idx=0,
        combo_metric_weights=(0.5, 0.3, 0.2),
        min_exit_accs=min_exit_accs,
        max_overall_drop=0.005,
        max_tail_drop=0.01,
        use_prob_margin=use_prob_margin,
    )

    print("Best eval record:", best["selected"])

    save_ckpt_v2(
        args.path_out,
        backbone,
        exit_heads,
        bb_cfg,
        exit_cfg_list=[ec.to_payload() for ec in exit_cfg_list],
        extra={"dataset": args.dataset, "best_eval_record": best["selected"], "best_metric": best["metric"]},
    )
    print("\nSaved:", args.path_out)


    
