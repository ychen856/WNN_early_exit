import pandas as pd
import numpy as np
import os

from typing import Optional
import pandas as pd

# =========================
# User settings
# =========================

FMNIST_CSV = "/Users/yi-chunchen/workspace/WNN_early_exit/plot/Untitled spreadsheet - fmnist - constrain - v2.csv"
CIFAR10_CSV = "/Users/yi-chunchen/workspace/WNN_early_exit/plot/Untitled spreadsheet - cifar - constrain- v2.csv"
EXPORT_DIR = "/Users/yi-chunchen/workspace/WNN_early_exit/plot/optimal_selection"

EPSILON = 1.0  # percentage point

# Replace with your real baseline VAL accuracy
BASELINE_VAL_ACC = {
    "fmnist": 92.08,
    "cifar10": 99.16,
}
PROFILE_MODES = ["all", "constrain"]

TOP_K = 5

# Candidate mixed-pruning patterns for prediction.
# These assume keep ratio / remaining LUT entry percentage.
FMNIST_RHO_PATTERNS = [
    (100, 100),
    (100, 75),
    (100, 50),
    (100, 25),
    (75, 75),
    (75, 50),
    (75, 25),
    (50, 50),
    (50, 25),
    (25, 25),
]

CIFAR10_RHO_PATTERNS = [
    (100, 100, 100, 100),
    (100, 100, 100, 75),
    (100, 100, 75, 75),
    (100, 75, 75, 50),
    (100, 75, 50, 50),
    (75, 75, 50, 50),
    (100, 100, 50, 50),
]


# ============================================================
# Column normalization
# ============================================================

def normalize_columns(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Normalize column names for each dataset.

    FMNIST format:
        Thr 0, Thr 1
        exit0_rate%, exit0_acc%
        exit1_rate%, exit1_acc%

    CIFAR-10 format:
        Thr 2, Thr 4, Thr 6, Thr 8
        exit2_rate%, exit2_acc%
        exit4_rate%, exit4_acc%
        exit6_rate%, exit6_acc%
        exit8_rate%, exit8_acc%
    """
    common_map = {
        "method": "method",
        "dataset": "dataset",
        "pruning rate (%)": "pruning_rate",
        "Overall acc %": "overall_acc",
        "final_rate%": "final_rate",
        "final_acc%": "final_acc",
        "expLayers": "expLayers",
        "avgFLOPs": "avgFLOPs",
        "avgMACs": "avgMACs",

        # Optional future per-exit pruning columns
        "rho0": "rho0",
        "rho1": "rho1",
        "rho2": "rho2",
        "rho4": "rho4",
        "rho6": "rho6",
        "rho8": "rho8",

        # Optional future per-exit mask IDs
        "mask0": "mask0",
        "mask1": "mask1",
        "mask2": "mask2",
        "mask4": "mask4",
        "mask6": "mask6",
        "mask8": "mask8",
        "mask_id": "mask_id",
    }

    fmnist_map = {
        "Thr 0": "thr0",
        "Thr 1": "thr1",
        "exit0_rate%": "exit0_rate",
        "exit0_acc%": "exit0_acc",
        "exit1_rate%": "exit1_rate",
        "exit1_acc%": "exit1_acc",
    }

    cifar10_map = {
        "Thr 2": "thr2",
        "Thr 4": "thr4",
        "Thr 6": "thr6",
        "Thr 8": "thr8",
        "exit2_rate%": "exit2_rate",
        "exit2_acc%": "exit2_acc",
        "exit4_rate%": "exit4_rate",
        "exit4_acc%": "exit4_acc",
        "exit6_rate%": "exit6_rate",
        "exit6_acc%": "exit6_acc",
        "exit8_rate%": "exit8_rate",
        "exit8_acc%": "exit8_acc",
    }

    rename_map = dict(common_map)

    if task == "fmnist":
        rename_map.update(fmnist_map)
    elif task == "cifar10":
        rename_map.update(cifar10_map)
    else:
        raise ValueError(f"Unknown task: {task}")

    df = df.rename(columns={c: rename_map.get(c.strip(), c.strip()) for c in df.columns})
    return df


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert known numeric columns to numeric.
    Safe for both FMNIST and CIFAR-10 because missing columns are skipped.
    """
    numeric_cols = [
        "pruning_rate",
        "overall_acc",
        "final_rate",
        "final_acc",
        "expLayers",
        "avgFLOPs",
        "avgMACs",

        # FMNIST thresholds / exits
        "thr0",
        "thr1",
        "exit0_rate",
        "exit0_acc",
        "exit1_rate",
        "exit1_acc",

        # CIFAR-10 thresholds / exits
        "thr2",
        "thr4",
        "thr6",
        "thr8",
        "exit2_rate",
        "exit2_acc",
        "exit4_rate",
        "exit4_acc",
        "exit6_rate",
        "exit6_acc",
        "exit8_rate",
        "exit8_acc",

        # Per-exit pruning rates
        "rho0",
        "rho1",
        "rho2",
        "rho4",
        "rho6",
        "rho8",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def normalize_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the dataset column into a split column.
    Expected dataset values may be:
        Val, Test, validation, test, FMNIST Val, CIFAR10 Test, etc.
    """
    if "dataset" not in df.columns:
        raise ValueError("CSV must contain a `dataset` column for Val/Test split.")

    df = df.copy()

    def infer_split(x):
        s = str(x).lower()
        if "val" in s:
            return "val"
        if "test" in s:
            return "test"
        return "unknown"

    df["split"] = df["dataset"].apply(infer_split)
    return df


def normalize_pruning_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize pruning rate into percentage scale.
    If values are in [0, 1], convert them to [0, 100].
    """
    df = df.copy()

    if "pruning_rate" not in df.columns:
        return df

    max_val = df["pruning_rate"].dropna().max()

    if max_val <= 1.0:
        df["pruning_rate"] = df["pruning_rate"] * 100.0

    return df


def expand_shared_pruning(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Backward compatibility:
    If the CSV only has shared `pruning_rate`, expand it into per-exit rho columns.

    FMNIST:
        rho0 = pruning_rate
        rho1 = pruning_rate

    CIFAR-10:
        rho2 = pruning_rate
        rho4 = pruning_rate
        rho6 = pruning_rate
        rho8 = pruning_rate
    """
    df = df.copy()

    if "pruning_rate" not in df.columns:
        return df

    if task == "fmnist":
        for col in ["rho0", "rho1"]:
            if col not in df.columns:
                df[col] = df["pruning_rate"]

    elif task == "cifar10":
        for col in ["rho2", "rho4", "rho6", "rho8"]:
            if col not in df.columns:
                df[col] = df["pruning_rate"]

    else:
        raise ValueError(f"Unknown task: {task}")

    return df


# ============================================================
# Task specs
# ============================================================

def get_exit_specs(task: str) -> list[dict]:
    if task == "fmnist":
        return [
            {
                "name": "0",
                "rho_col": "rho0",
                "thr_col": "thr0",
                "rate_col": "exit0_rate",
                "acc_col": "exit0_acc",
                "cond_col": "cond_exit0",
                "mask_col": "mask0",
            },
            {
                "name": "1",
                "rho_col": "rho1",
                "thr_col": "thr1",
                "rate_col": "exit1_rate",
                "acc_col": "exit1_acc",
                "cond_col": "cond_exit1",
                "mask_col": "mask1",
            },
        ]

    if task == "cifar10":
        return [
            {
                "name": "2",
                "rho_col": "rho2",
                "thr_col": "thr2",
                "rate_col": "exit2_rate",
                "acc_col": "exit2_acc",
                "cond_col": "cond_exit2",
                "mask_col": "mask2",
            },
            {
                "name": "4",
                "rho_col": "rho4",
                "thr_col": "thr4",
                "rate_col": "exit4_rate",
                "acc_col": "exit4_acc",
                "cond_col": "cond_exit4",
                "mask_col": "mask4",
            },
            {
                "name": "6",
                "rho_col": "rho6",
                "thr_col": "thr6",
                "rate_col": "exit6_rate",
                "acc_col": "exit6_acc",
                "cond_col": "cond_exit6",
                "mask_col": "mask6",
            },
            {
                "name": "8",
                "rho_col": "rho8",
                "thr_col": "thr8",
                "rate_col": "exit8_rate",
                "acc_col": "exit8_acc",
                "cond_col": "cond_exit8",
                "mask_col": "mask8",
            },
        ]

    raise ValueError(f"Unknown task: {task}")


def get_config_cols(df: pd.DataFrame, task: str) -> list[str]:
    """
    Use task-specific config columns for matching validation-selected
    configuration to the corresponding test row.
    """
    exit_specs = get_exit_specs(task)

    possible_cols = ["method"]

    for spec in exit_specs:
        possible_cols.append(spec["rho_col"])

    for spec in exit_specs:
        possible_cols.append(spec["thr_col"])

    # Optional mask columns if present
    for spec in exit_specs:
        possible_cols.append(spec["mask_col"])

    possible_cols.append("mask_id")

    return [c for c in possible_cols if c in df.columns]


# ============================================================
# Utility
# ============================================================

def safe_div(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den):
        return 0.0
    if den <= 1e-12:
        return 0.0
    return float(num) / float(den)


def match_test_row(
    df_test: pd.DataFrame,
    best_val: pd.Series,
    config_cols: list[str],
) -> pd.DataFrame:
    """
    Match test rows using method, per-exit rho, thresholds, and optional masks.
    """
    matched = df_test.copy()

    for col in config_cols:
        if col not in matched.columns or col not in best_val.index:
            continue

        val = best_val[col]

        if pd.isna(val):
            matched = matched[matched[col].isna()]
        elif pd.api.types.is_numeric_dtype(matched[col]):
            matched = matched[np.isclose(matched[col], val, equal_nan=True)]
        else:
            matched = matched[matched[col].astype(str) == str(val)]

    return matched



def filter_profile_mode(df: pd.DataFrame, profile_mode: str) -> pd.DataFrame:
    """
    profile_mode:
        "all"       -> use all methods
        "constrain" -> use only rows whose method contains constrain/constraint
    """
    if profile_mode == "all":
        return df.copy()

    if profile_mode == "constrain":
        if "method" not in df.columns:
            raise ValueError("`method` column is required for constrain mode.")

        method_str = df["method"].astype(str).str.lower()

        mask = (
            method_str.str.contains("constrain", na=False)
            | method_str.str.contains("constraint", na=False)
        )

        filtered = df[mask].copy()

        if filtered.empty:
            raise ValueError("No rows left after constrain filtering.")

        return filtered

    raise ValueError(f"Unknown profile_mode: {profile_mode}")


# ============================================================
# Shared-pruning measured selection
# ============================================================

def select_best_point(
    df: pd.DataFrame,
    task: str,
    baseline_val_acc: float,
    epsilon: float,
    profile_mode: str,
) -> dict:
    target_acc = baseline_val_acc - epsilon

    df_mode = filter_profile_mode(df, profile_mode)

    df_val = df_mode[df_mode["split"] == "val"].copy()
    df_test = df_mode[df_mode["split"] == "test"].copy()

    feasible = df_val[df_val["overall_acc"] >= target_acc].copy()

    if feasible.empty:
        raise ValueError(
            f"No feasible candidates for {task} | {profile_mode}. "
            f"Target acc={target_acc:.2f}, "
            f"max val acc={df_val['overall_acc'].max():.2f}"
        )

    sort_cols = ["final_rate"]
    ascending = [True]

    if "avgFLOPs" in feasible.columns:
        sort_cols.append("avgFLOPs")
        ascending.append(True)

    sort_cols.append("overall_acc")
    ascending.append(False)

    best_val = feasible.sort_values(sort_cols, ascending=ascending).iloc[0]

    config_cols = get_config_cols(df_mode, task)
    matched_test = match_test_row(df_test, best_val, config_cols)

    best_test = None if matched_test.empty else matched_test.iloc[0]

    return {
        "task": task,
        "profile_mode": profile_mode,
        "baseline_val_acc": baseline_val_acc,
        "target_acc": target_acc,
        "num_val_candidates": len(df_val),
        "num_feasible_candidates": len(feasible),
        "best_val": best_val,
        "best_test": best_test,
        "feasible": feasible,
        "processed": df_mode,
    }


# ============================================================
# Prediction: conditional exit-rate reconstruction
# ============================================================

def add_conditional_exit_probs(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Convert global exit rates to conditional exit probabilities.

    Input rates are percentage values:
        32.5 means 32.5%.
    """
    df = df.copy()

    if task == "fmnist":
        r0 = df["exit0_rate"] / 100.0
        r1 = df["exit1_rate"] / 100.0

        df["cond_exit0"] = r0
        df["cond_exit1"] = [
            safe_div(r1_i, 1.0 - r0_i)
            for r1_i, r0_i in zip(r1, r0)
        ]

    elif task == "cifar10":
        r2 = df["exit2_rate"] / 100.0
        r4 = df["exit4_rate"] / 100.0
        r6 = df["exit6_rate"] / 100.0
        r8 = df["exit8_rate"] / 100.0

        df["cond_exit2"] = r2

        df["cond_exit4"] = [
            safe_div(r4_i, 1.0 - r2_i)
            for r4_i, r2_i in zip(r4, r2)
        ]

        df["cond_exit6"] = [
            safe_div(r6_i, 1.0 - r2_i - r4_i)
            for r6_i, r2_i, r4_i in zip(r6, r2, r4)
        ]

        df["cond_exit8"] = [
            safe_div(r8_i, 1.0 - r2_i - r4_i - r6_i)
            for r8_i, r2_i, r4_i, r6_i in zip(r8, r2, r4, r6)
        ]

    else:
        raise ValueError(f"Unknown task: {task}")

    return df


'''def find_profile_row(
    df_val: pd.DataFrame,
    method: Optional[str],
    rho: float,
    thr_vector: dict,
    exit_spec: dict,
) -> Optional[pd.Series]:
    """
    Find a shared-pruning profiling row for one exit.

    Priority:
        1. Match method + pruning_rate + full threshold vector.
        2. If failed, match method + pruning_rate + local threshold only.
        3. If multiple local matches, choose the row with highest local exit accuracy.

    This is used only for prediction, not measured result.
    """
    local_thr_col = exit_spec["thr_col"]

    matched = df_val.copy()

    if method is not None and "method" in matched.columns:
        matched = matched[matched["method"].astype(str) == str(method)]

    if "pruning_rate" not in matched.columns:
        raise ValueError("Prediction currently requires `pruning_rate` from shared-pruning profiling CSV.")

    matched = matched[np.isclose(matched["pruning_rate"], rho, equal_nan=True)]

    if matched.empty:
        return None

    # Full threshold-vector matching
    full = matched.copy()
    for thr_col, thr_val in thr_vector.items():
        if thr_col in full.columns:
            full = full[np.isclose(full[thr_col], thr_val, equal_nan=True)]

    if not full.empty:
        return full.iloc[0]

    # Local threshold fallback
    if local_thr_col in matched.columns and local_thr_col in thr_vector:
        local = matched[np.isclose(matched[local_thr_col], thr_vector[local_thr_col], equal_nan=True)]

        if not local.empty:
            acc_col = exit_spec["acc_col"]
            if acc_col in local.columns:
                return local.sort_values(acc_col, ascending=False).iloc[0]
            return local.iloc[0]

    return None'''


def reconstruct_global_rates(cond_probs: list[float]) -> tuple[list[float], float]:
    """
    Given conditional exit probabilities, reconstruct global exit rates.

    Example:
        p0 = P(exit at 0)
        p1 = P(exit at 1 | not exit at 0)

    Return:
        global exit rates and final rate.
    """
    remaining = 1.0
    global_rates = []

    for p in cond_probs:
        p = min(max(float(p), 0.0), 1.0)
        r = remaining * p
        global_rates.append(r)
        remaining -= r

    final_rate = max(0.0, remaining)
    return global_rates, final_rate


'''def predict_mixed_candidate(
    df_val: pd.DataFrame,
    task: str,
    base_row: pd.Series,
    rho_pattern: tuple,
    final_acc_mode: str = "base",
) -> Optional[dict]:
    """
    Predict one mixed-pruning candidate.

    base_row provides:
        method and threshold vector.

    rho_pattern provides:
        per-exit pruning rates.

    final_acc_mode:
        "base": use base_row final_acc.
        "worst": use the minimum final_acc among involved source rows.
    """
    exit_specs = get_exit_specs(task)
    method = base_row.get("method", None)

    thr_vector = {}
    for spec in exit_specs:
        thr_col = spec["thr_col"]
        if thr_col not in base_row.index:
            return None
        thr_vector[thr_col] = base_row[thr_col]

    cond_probs = []
    exit_accs = []
    used_rows = []

    result = {
        "task": task,
        "method": method,
        "base_pruning_rate": base_row.get("pruning_rate", np.nan),
    }

    for spec, rho in zip(exit_specs, rho_pattern):
        profile_row = find_profile_row(
            df_val=df_val,
            method=method,
            rho=rho,
            thr_vector=thr_vector,
            exit_spec=spec,
        )

        if profile_row is None:
            return None

        cond_col = spec["cond_col"]
        acc_col = spec["acc_col"]

        if cond_col not in profile_row.index or acc_col not in profile_row.index:
            return None

        cond_probs.append(profile_row[cond_col])
        exit_accs.append(profile_row[acc_col])
        used_rows.append(profile_row)

        result[spec["rho_col"]] = rho
        result[spec["thr_col"]] = thr_vector[spec["thr_col"]]
        result[f"source_exit{spec['name']}_overall_acc"] = profile_row.get("overall_acc", np.nan)
        result[f"source_exit{spec['name']}_final_rate"] = profile_row.get("final_rate", np.nan)
        result[f"source_exit{spec['name']}_pruning_rate"] = profile_row.get("pruning_rate", np.nan)

    global_rates, final_rate = reconstruct_global_rates(cond_probs)

    if final_acc_mode == "worst":
        final_acc_values = [row.get("final_acc", np.nan) for row in used_rows]
        final_acc_values = [v for v in final_acc_values if not pd.isna(v)]
        final_acc = min(final_acc_values) if final_acc_values else np.nan
    else:
        final_acc = base_row.get("final_acc", np.nan)

    if pd.isna(final_acc):
        return None

    pred_acc = 0.0

    for r, a in zip(global_rates, exit_accs):
        if pd.isna(a):
            return None
        pred_acc += r * (a / 100.0)

    pred_acc += final_rate * (final_acc / 100.0)
    pred_acc *= 100.0

    result["pred_acc"] = pred_acc
    result["pred_final_rate"] = final_rate * 100.0
    result["pred_final_acc"] = final_acc

    for spec, r, a, p in zip(exit_specs, global_rates, exit_accs, cond_probs):
        result[f"pred_exit{spec['name']}_rate"] = r * 100.0
        result[f"pred_exit{spec['name']}_acc"] = a
        result[f"cond_exit{spec['name']}"] = p

    # Optional approximate cost.
    # We do not fully estimate FLOPs here because mixed rho may change per-exit overhead.
    # The main cost surrogate remains final_rate.
    result["prediction_type"] = "conditional_rate_reconstruction"
    result["is_measured"] = False

    return result'''


'''def generate_predicted_mixed_candidates(
    df: pd.DataFrame,
    task: str,
    baseline_val_acc: float,
    epsilon: float,
    top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate predicted mixed-pruning candidates from measured shared-pruning profiling.

    The predictor uses each measured validation row as a threshold-vector template.
    For each mixed pruning pattern, it reconstructs expected global exit rates
    using conditional exit probabilities.
    """
    target_acc = baseline_val_acc - epsilon

    df_val = df[df["split"] == "val"].copy()
    df_val = expand_shared_pruning(df_val, task)
    df_val = add_conditional_exit_probs(df_val, task)


    if task == "fmnist":
        rho_patterns = FMNIST_RHO_PATTERNS
    elif task == "cifar10":
        rho_patterns = CIFAR10_RHO_PATTERNS
    else:
        raise ValueError(f"Unknown task: {task}")

    exit_specs = get_exit_specs(task)

    candidates = []
    seen = set()

    for _, base_row in df_val.iterrows():
        for rho_pattern in rho_patterns:
            pred = predict_mixed_candidate(
                df_val=df_val,
                task=task,
                base_row=base_row,
                rho_pattern=rho_pattern,
                final_acc_mode="base",
            )

            if pred is None:
                continue

            key = (
                pred.get("method"),
                tuple(rho_pattern),
                tuple(pred.get(spec["thr_col"]) for spec in exit_specs),
            )

            if key in seen:
                continue

            seen.add(key)

            pred["target_acc"] = target_acc
            pred["is_pred_feasible"] = pred["pred_acc"] >= target_acc

            candidates.append(pred)

    pred_df = pd.DataFrame(candidates)

    if pred_df.empty:
        return pred_df, pred_df

    feasible = pred_df[pred_df["is_pred_feasible"]].copy()

    if feasible.empty:
        # If none satisfy target, still output top-k by highest predicted acc,
        # then lower final rate.
        topk = pred_df.sort_values(
            ["pred_acc", "pred_final_rate"],
            ascending=[False, True],
        ).head(top_k)
    else:
        topk = feasible.sort_values(
            ["pred_final_rate", "pred_acc"],
            ascending=[True, False],
        ).head(top_k)

    return pred_df, topk'''


# ============================================================
# Summary construction
# ============================================================

def summarize_task_exits(
    task: str,
    prefix: str,
    row_data: pd.Series,
) -> dict:
    """
    Summarize task-specific exit rates/accuracies for either val or test.
    """
    out = {}

    for spec in get_exit_specs(task):
        name = spec["name"]
        out[f"{prefix}_exit{name}_rate"] = row_data.get(spec["rate_col"], np.nan)
        out[f"{prefix}_exit{name}_acc"] = row_data.get(spec["acc_col"], np.nan)

    return out


def generate_predicted_mixed_candidates(
    df: pd.DataFrame,
    task: str,
    baseline_val_acc: float,
    epsilon: float,
    top_k: int,
    profile_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_acc = baseline_val_acc - epsilon

    df_val = df[df["split"] == "val"].copy()
    df_val = filter_profile_mode(df_val, profile_mode)

    df_val = expand_shared_pruning(df_val, task)
    df_val = add_conditional_exit_probs(df_val, task)

    profiles = build_local_exit_profile(df_val, task)
    exit_specs = get_exit_specs(task)

    if task == "fmnist":
        rho_patterns = FMNIST_RHO_PATTERNS
    elif task == "cifar10":
        rho_patterns = CIFAR10_RHO_PATTERNS
    else:
        raise ValueError(f"Unknown task: {task}")

    candidates = []
    seen = set()

    for _, row in df_val.iterrows():
        method = row.get("method", None)

        thr_vector = {}
        ok = True

        for spec in exit_specs:
            thr_col = spec["thr_col"]
            if thr_col not in row.index or pd.isna(row[thr_col]):
                ok = False
                break
            thr_vector[thr_col] = row[thr_col]

        if not ok:
            continue

        for rho_pattern in rho_patterns:
            pred = predict_mixed_candidate_local(
                profiles=profiles,
                task=task,
                method=method,
                thr_vector=thr_vector,
                rho_pattern=rho_pattern,
                final_acc_mode="mean",
            )

            if pred is None:
                continue

            key = (
                pred.get("method"),
                tuple(rho_pattern),
                tuple(pred.get(spec["thr_col"]) for spec in exit_specs),
            )

            if key in seen:
                continue

            seen.add(key)

            pred["profile_mode"] = profile_mode
            pred["target_acc"] = target_acc
            pred["is_pred_feasible"] = pred["pred_acc"] >= target_acc
            candidates.append(pred)

    pred_df = pd.DataFrame(candidates)

    if pred_df.empty:
        return pred_df, pred_df

    feasible = pred_df[pred_df["is_pred_feasible"]].copy()

    if feasible.empty:
        topk = pred_df.sort_values(
            ["pred_acc", "pred_final_rate"],
            ascending=[False, True],
        ).head(top_k)
    else:
        topk = feasible.sort_values(
            ["pred_final_rate", "pred_acc"],
            ascending=[True, False],
        ).head(top_k)

    return pred_df, topk


def build_local_exit_profile(df_val: pd.DataFrame, task: str) -> dict:
    """
    Build local per-exit profile tables.

    Each exit profile is indexed by:
        method, pruning_rate, local threshold

    It stores:
        conditional exit probability
        exit accuracy
        final accuracy
        source overall accuracy
        source final rate

    This avoids requiring full threshold-vector matching.
    """
    profiles = {}

    for spec in get_exit_specs(task):
        thr_col = spec["thr_col"]
        cond_col = spec["cond_col"]
        acc_col = spec["acc_col"]

        required = ["method", "pruning_rate", thr_col, cond_col, acc_col, "final_acc"]
        missing = [c for c in required if c not in df_val.columns]

        if missing:
            raise ValueError(f"{task} exit{spec['name']} missing columns: {missing}")

        group_cols = ["method", "pruning_rate", thr_col]

        # Aggregate in case duplicate rows exist.
        # For prediction, mean is safer than choosing one arbitrary row.
        agg = (
            df_val
            .groupby(group_cols, dropna=False)
            .agg(
                cond_prob=(cond_col, "mean"),
                exit_acc=(acc_col, "mean"),
                final_acc=("final_acc", "mean"),
                source_overall_acc=("overall_acc", "mean"),
                source_final_rate=("final_rate", "mean"),
                count=(acc_col, "count"),
            )
            .reset_index()
        )

        profiles[spec["name"]] = {
            "spec": spec,
            "table": agg,
        }

    return profiles


def lookup_local_profile(
    profiles: dict,
    exit_name: str,
    method: str,
    rho: float,
    thr_value: float,
) -> Optional[pd.Series]:
    """
    Lookup one exit's local profile using:
        method, pruning_rate, local threshold

    If exact threshold match fails, use nearest threshold under same method/rho.
    """
    obj = profiles[exit_name]
    spec = obj["spec"]
    table = obj["table"]
    thr_col = spec["thr_col"]

    matched = table.copy()

    matched = matched[matched["method"].astype(str) == str(method)]
    matched = matched[np.isclose(matched["pruning_rate"], rho, equal_nan=True)]

    if matched.empty:
        return None

    exact = matched[np.isclose(matched[thr_col], thr_value, equal_nan=True)]

    if not exact.empty:
        return exact.iloc[0]

    # fallback: nearest local threshold
    matched = matched.copy()
    matched["thr_distance"] = (matched[thr_col] - thr_value).abs()
    return matched.sort_values(["thr_distance", "exit_acc"], ascending=[True, False]).iloc[0]


def predict_mixed_candidate_local(
    profiles: dict,
    task: str,
    method: str,
    thr_vector: dict,
    rho_pattern: tuple,
    final_acc_mode: str = "mean",
) -> Optional[dict]:
    """
    Predict one mixed-pruning candidate from local per-exit profiles.
    """
    exit_specs = get_exit_specs(task)

    cond_probs = []
    exit_accs = []
    final_accs = []
    source_rows = []

    result = {
        "task": task,
        "method": method,
    }

    for spec, rho in zip(exit_specs, rho_pattern):
        exit_name = spec["name"]
        thr_col = spec["thr_col"]
        rho_col = spec["rho_col"]

        if thr_col not in thr_vector:
            return None

        thr_value = thr_vector[thr_col]

        row = lookup_local_profile(
            profiles=profiles,
            exit_name=exit_name,
            method=method,
            rho=rho,
            thr_value=thr_value,
        )

        if row is None:
            return None

        cond_probs.append(row["cond_prob"])
        exit_accs.append(row["exit_acc"])
        final_accs.append(row["final_acc"])
        source_rows.append(row)

        result[rho_col] = rho
        result[thr_col] = thr_value
        result[f"source_exit{exit_name}_local_thr_used"] = row[thr_col]
        result[f"source_exit{exit_name}_count"] = row["count"]
        result[f"source_exit{exit_name}_overall_acc"] = row["source_overall_acc"]
        result[f"source_exit{exit_name}_final_rate"] = row["source_final_rate"]

    global_rates, final_rate = reconstruct_global_rates(cond_probs)

    if final_acc_mode == "worst":
        final_acc = min(final_accs)
    else:
        final_acc = float(np.mean(final_accs))

    pred_acc = 0.0

    for r, a in zip(global_rates, exit_accs):
        pred_acc += r * (a / 100.0)

    pred_acc += final_rate * (final_acc / 100.0)
    pred_acc *= 100.0

    result["pred_acc"] = pred_acc
    result["pred_final_rate"] = final_rate * 100.0
    result["pred_final_acc"] = final_acc
    result["prediction_type"] = "local_conditional_rate_reconstruction"
    result["is_measured"] = False

    for spec, r, a, p in zip(exit_specs, global_rates, exit_accs, cond_probs):
        name = spec["name"]
        result[f"pred_exit{name}_rate"] = r * 100.0
        result[f"pred_exit{name}_acc"] = a
        result[f"cond_exit{name}"] = p

    return result



def summarize_selection(result: dict) -> dict:
    task = result["task"]
    best_val = result["best_val"]
    best_test = result["best_test"]

    row = {
        "task": task,
        "baseline_val_acc": result["baseline_val_acc"],
        "target_acc": result["target_acc"],
        "num_val_candidates": result["num_val_candidates"],
        "num_feasible_candidates": result["num_feasible_candidates"],

        "method": best_val.get("method", np.nan),
        "pruning_rate": best_val.get("pruning_rate", np.nan),

        "val_acc": best_val.get("overall_acc", np.nan),
        "val_final_rate": best_val.get("final_rate", np.nan),
        "val_final_acc": best_val.get("final_acc", np.nan),
        "val_expLayers": best_val.get("expLayers", np.nan),
        "val_avgFLOPs": best_val.get("avgFLOPs", np.nan),
        "val_avgMACs": best_val.get("avgMACs", np.nan),
    }

    # Add selected thresholds and rho values
    for spec in get_exit_specs(task):
        row[spec["rho_col"]] = best_val.get(spec["rho_col"], np.nan)
        row[spec["thr_col"]] = best_val.get(spec["thr_col"], np.nan)

    row.update(summarize_task_exits(task, "val", best_val))

    if best_test is not None:
        row.update({
            "test_acc": best_test.get("overall_acc", np.nan),
            "test_final_rate": best_test.get("final_rate", np.nan),
            "test_final_acc": best_test.get("final_acc", np.nan),
            "test_expLayers": best_test.get("expLayers", np.nan),
            "test_avgFLOPs": best_test.get("avgFLOPs", np.nan),
            "test_avgMACs": best_test.get("avgMACs", np.nan),
            "val_test_gap": abs(
                best_val.get("overall_acc", np.nan)
                - best_test.get("overall_acc", np.nan)
            ),
        })
        row.update(summarize_task_exits(task, "test", best_test))
    else:
        row.update({
            "test_acc": np.nan,
            "test_final_rate": np.nan,
            "test_final_acc": np.nan,
            "test_expLayers": np.nan,
            "test_avgFLOPs": np.nan,
            "test_avgMACs": np.nan,
            "val_test_gap": np.nan,
        })

        for spec in get_exit_specs(task):
            name = spec["name"]
            row[f"test_exit{name}_rate"] = np.nan
            row[f"test_exit{name}_acc"] = np.nan

    return row


# ============================================================
# Export
# ============================================================

def export_single_task(
    csv_path: str,
    task: str,
    baseline_val_acc: float,
    epsilon: float,
    export_dir: str,
    top_k: int,
) -> list[dict]:
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]

    df = normalize_columns(df, task)
    df = ensure_numeric(df)
    df = normalize_pruning_scale(df)
    df = normalize_split(df)
    df = expand_shared_pruning(df, task)

    outputs = []

    for profile_mode in PROFILE_MODES:
        mode_dir = os.path.join(export_dir, task, profile_mode)
        os.makedirs(mode_dir, exist_ok=True)
        result = select_best_point(
            df=df,
            task=task,
            baseline_val_acc=baseline_val_acc,
            epsilon=epsilon,
            profile_mode=profile_mode,
        )

        summary = summarize_selection(result)
        summary["profile_mode"] = profile_mode
        summary_df = pd.DataFrame([summary])
        pred_df, topk_df = generate_predicted_mixed_candidates(
            df=df,
            task=task,
            baseline_val_acc=baseline_val_acc,
            epsilon=epsilon,
            top_k=top_k,
            profile_mode=profile_mode,
        )

        summary_path = os.path.join(
            mode_dir,
            f"{task}_shared_measured_optimal_summary_{profile_mode}.csv",
        )

        pred_path = os.path.join(
            mode_dir,
            f"{task}_predicted_mixed_pruning_candidates_{profile_mode}.csv",
        )

        topk_path = os.path.join(
            mode_dir,
            f"{task}_predicted_mixed_pruning_top{top_k}_{profile_mode}.csv",
        )

        summary_df.to_csv(summary_path, index=False)
        pred_df.to_csv(pred_path, index=False)
        topk_df.to_csv(topk_path, index=False)

        print(f"\n=== {task.upper()} | {profile_mode} ===")
        print(summary_df.to_string(index=False))

        print(f"\nPredicted mixed-pruning top-{top_k}:")
        if topk_df.empty:
            print("[Warning] empty")
        else:
            print(topk_df.to_string(index=False))

        print("\nSaved:")
        print(f"  {summary_path}")
        print(f"  {pred_path}")
        print(f"  {topk_path}")

        outputs.append({
            "task": task,
            "profile_mode": profile_mode,
            "summary": summary,
            "summary_path": summary_path,
            "pred_path": pred_path,
            "topk_path": topk_path,
        })

    return outputs


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)

    all_summaries = []

    fmnist_outputs = export_single_task(
        csv_path=FMNIST_CSV,
        task="fmnist",
        baseline_val_acc=BASELINE_VAL_ACC["fmnist"],
        epsilon=EPSILON,
        export_dir=EXPORT_DIR,
        top_k=TOP_K,
    )

    cifar10_outputs = export_single_task(
        csv_path=CIFAR10_CSV,
        task="cifar10",
        baseline_val_acc=BASELINE_VAL_ACC["cifar10"],
        epsilon=EPSILON,
        export_dir=EXPORT_DIR,
        top_k=TOP_K,
    )

    for item in fmnist_outputs + cifar10_outputs:
        all_summaries.append(item["summary"])

    all_summary_df = pd.DataFrame(all_summaries)
    all_summary_path = os.path.join(EXPORT_DIR, "all_profile_modes_summary.csv")
    all_summary_df.to_csv(all_summary_path, index=False)

    print("\n=== All profile modes summary ===")
    print(all_summary_df.to_string(index=False))
    print(f"\nSaved combined summary:\n  {all_summary_path}")

if __name__ == "__main__":
    main()