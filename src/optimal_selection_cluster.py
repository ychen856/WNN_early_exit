import os
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# User settings
# ============================================================

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
PRED_ACC_MARGIN = 0.2
TOP_K = 5

THRESHOLD_CLUSTER_K = {
    "fmnist": 8,
    "cifar10": 8,
}

FMNIST_RHO_PATTERNS = [
    (100, 100),
    (100, 75),
    (100, 50),
    (75, 75),
    (75, 50),
    (50, 50),
]

CIFAR10_RHO_PATTERNS = [
    (100, 100, 100, 100),
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

        "rho0": "rho0",
        "rho1": "rho1",
        "rho2": "rho2",
        "rho4": "rho4",
        "rho6": "rho6",
        "rho8": "rho8",

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
    numeric_cols = [
        "pruning_rate",
        "overall_acc",
        "final_rate",
        "final_acc",
        "expLayers",
        "avgFLOPs",
        "avgMACs",

        "thr0",
        "thr1",
        "exit0_rate",
        "exit0_acc",
        "exit1_rate",
        "exit1_acc",

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


def normalize_pruning_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    If pruning_rate is in [0, 1], convert to [0, 100].
    """
    df = df.copy()

    if "pruning_rate" not in df.columns:
        return df

    max_val = df["pruning_rate"].dropna().max()

    if pd.notna(max_val) and max_val <= 1.0:
        df["pruning_rate"] = df["pruning_rate"] * 100.0

    return df


def normalize_split(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset" not in df.columns:
        raise ValueError("CSV must contain a `dataset` column.")

    df = df.copy()

    def infer_split(x):
        s = str(x).strip().lower()

        if s in ["val", "valid", "validation"]:
            return "val"
        if s in ["test", "testing"]:
            return "test"
        if "val" in s:
            return "val"
        if "test" in s:
            return "test"
        return "unknown"

    df["split"] = df["dataset"].apply(infer_split)
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


def get_rho_patterns(task: str) -> list[tuple]:
    if task == "fmnist":
        return FMNIST_RHO_PATTERNS
    if task == "cifar10":
        return CIFAR10_RHO_PATTERNS
    raise ValueError(f"Unknown task: {task}")


def expand_shared_pruning(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    If only shared pruning_rate exists, expand it into per-exit rho columns.
    """
    df = df.copy()

    if "pruning_rate" not in df.columns:
        return df

    if task == "fmnist":
        rho_cols = ["rho0", "rho1"]
    elif task == "cifar10":
        rho_cols = ["rho2", "rho4", "rho6", "rho8"]
    else:
        raise ValueError(f"Unknown task: {task}")

    for col in rho_cols:
        if col not in df.columns:
            df[col] = df["pruning_rate"]

    return df


# ============================================================
# Filtering
# ============================================================

def filter_profile_mode(df: pd.DataFrame, profile_mode: str) -> pd.DataFrame:
    """
    profile_mode:
        all       -> use all methods
        constrain -> only use rows whose method contains constrain/constraint
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
# Conditional exit probabilities
# ============================================================

def safe_div(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den):
        return 0.0
    if den <= 1e-12:
        return 0.0
    return float(num) / float(den)


def add_conditional_exit_probs(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Convert global exit rates into conditional exit probabilities.

    Example:
        global r0 = P(exit0)
        global r1 = P(exit1)
        cond p1 = P(exit1 | not exit0)
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


def reconstruct_global_rates(cond_probs: list[float]) -> tuple[list[float], float]:
    remaining = 1.0
    global_rates = []

    for p in cond_probs:
        p = min(max(float(p), 0.0), 1.0)
        r = remaining * p
        global_rates.append(r)
        remaining -= r

    return global_rates, max(0.0, remaining)


# ============================================================
# Threshold clustering
# ============================================================

def assign_ordered_clusters(values: np.ndarray, k: int) -> dict[float, int]:
    """
    Pure numpy ordered clustering for 1-D threshold values.

    This does not require sklearn.
    It sorts unique values and splits them into k ordered groups.
    """
    unique_vals = np.array(sorted(np.unique(values)), dtype=float)
    k_eff = min(k, len(unique_vals))

    if len(unique_vals) == 0:
        return {}

    if k_eff <= 1:
        return {float(v): 0 for v in unique_vals}

    chunks = np.array_split(unique_vals, k_eff)

    value_to_cluster = {}

    for cluster_id, chunk in enumerate(chunks):
        for v in chunk:
            value_to_cluster[float(v)] = int(cluster_id)

    return value_to_cluster


def add_threshold_clusters(
    df: pd.DataFrame,
    task: str,
    k: int,
) -> pd.DataFrame:
    """
    Cluster thresholds independently for each:
        method + pruning_rate + exit head

    Cluster id is ordered by threshold magnitude:
        0 = lowest threshold group
        k-1 = highest threshold group

    The cluster is only used for candidate aggregation/dedup.
    Final test should use selected raw threshold representative.
    """
    df = df.copy()
    group_base = ["method", "pruning_rate"]

    for spec in get_exit_specs(task):
        thr_col = spec["thr_col"]
        cluster_col = f"{thr_col}_cluster"

        if thr_col not in df.columns:
            continue

        df[cluster_col] = pd.Series(pd.NA, index=df.index, dtype="Int64")

        for _, idx in df.groupby(group_base, dropna=False).groups.items():
            sub = df.loc[idx, thr_col].dropna()

            if sub.empty:
                continue

            value_to_cluster = assign_ordered_clusters(sub.to_numpy(dtype=float), k=k)

            df.loc[idx, cluster_col] = df.loc[idx, thr_col].map(
                lambda x: value_to_cluster.get(float(x), pd.NA)
                if pd.notna(x)
                else pd.NA
            )

        df[cluster_col] = df[cluster_col].astype("Int64")

    return df


def medoid_value(values: pd.Series) -> float:
    """
    Return the observed value closest to the median.
    This is better than mean for final selected threshold because
    it is a real threshold from the validation profile.
    """
    clean = values.dropna().astype(float)

    if clean.empty:
        return np.nan

    median = clean.median()
    idx = (clean - median).abs().idxmin()
    return float(clean.loc[idx])


# ============================================================
# Local profile tables
# ============================================================

def build_local_exit_profile(df_val: pd.DataFrame, task: str) -> dict:
    """
    Build per-exit local profile tables indexed by:
        method + pruning_rate + threshold_cluster

    Each row stores:
        conditional exit probability
        exit accuracy
        final accuracy
        selected raw threshold representative
    """
    profiles = {}

    for spec in get_exit_specs(task):
        name = spec["name"]
        thr_col = spec["thr_col"]
        cluster_col = f"{thr_col}_cluster"
        cond_col = spec["cond_col"]
        acc_col = spec["acc_col"]

        required = [
            "method",
            "pruning_rate",
            cluster_col,
            cond_col,
            acc_col,
            "final_acc",
            "overall_acc",
            "final_rate",
        ]

        missing = [c for c in required if c not in df_val.columns]

        if missing:
            raise ValueError(f"{task} exit{name} missing columns: {missing}")

        group_cols = ["method", "pruning_rate", cluster_col]

        agg = (
            df_val
            .groupby(group_cols, dropna=False)
            .agg(
                cond_prob=(cond_col, "mean"),
                exit_acc=(acc_col, "mean"),
                final_acc=("final_acc", "mean"),
                source_overall_acc=("overall_acc", "mean"),
                source_final_rate=("final_rate", "mean"),
                raw_thr_mean=(thr_col, "mean"),
                raw_thr_median=(thr_col, "median"),
                raw_thr_selected=(thr_col, medoid_value),
                raw_thr_min=(thr_col, "min"),
                raw_thr_max=(thr_col, "max"),
                count=(acc_col, "count"),
            )
            .reset_index()
        )

        profiles[name] = {
            "spec": spec,
            "table": agg,
            "cluster_col": cluster_col,
        }

    return profiles


def lookup_local_profile_by_cluster(
    profiles: dict,
    exit_name: str,
    method: str,
    rho: float,
    thr_cluster: int,
) -> Optional[pd.Series]:
    """
    Lookup one exit's profile with:
        method + pruning_rate + threshold_cluster

    If exact cluster is unavailable for the target rho, use nearest cluster.
    """
    obj = profiles[exit_name]
    table = obj["table"]
    cluster_col = obj["cluster_col"]

    matched = table.copy()

    matched = matched[matched["method"].astype(str) == str(method)]
    matched = matched[np.isclose(matched["pruning_rate"], rho, equal_nan=True)]

    if matched.empty:
        return None

    exact = matched[matched[cluster_col].astype("Int64") == int(thr_cluster)]

    if not exact.empty:
        return exact.iloc[0]

    matched = matched.copy()
    matched["cluster_distance"] = (
        matched[cluster_col].astype(float) - float(thr_cluster)
    ).abs()

    return matched.sort_values(
        ["cluster_distance", "exit_acc"],
        ascending=[True, False],
    ).iloc[0]


# ============================================================
# Mixed-pruning prediction
# ============================================================

def predict_mixed_candidate(
    profiles: dict,
    task: str,
    method: str,
    thr_cluster_vector: dict,
    rho_pattern: tuple,
    profile_mode: str,
    final_acc_mode: str = "mean",
) -> Optional[dict]:
    """
    Predict one mixed-pruning candidate.

    Cluster ids are used to identify the operating point.
    raw_thr_selected is the actual threshold value to apply in test rerun.
    """
    exit_specs = get_exit_specs(task)

    cond_probs = []
    exit_accs = []
    final_accs = []

    result = {
        "task": task,
        "profile_mode": profile_mode,
        "method": method,
    }

    for spec, rho in zip(exit_specs, rho_pattern):
        exit_name = spec["name"]
        rho_col = spec["rho_col"]
        thr_col = spec["thr_col"]
        cluster_col = f"{thr_col}_cluster"

        if cluster_col not in thr_cluster_vector:
            return None

        thr_cluster = int(thr_cluster_vector[cluster_col])

        row = lookup_local_profile_by_cluster(
            profiles=profiles,
            exit_name=exit_name,
            method=method,
            rho=rho,
            thr_cluster=thr_cluster,
        )

        if row is None:
            return None

        cond_probs.append(row["cond_prob"])
        exit_accs.append(row["exit_acc"])
        final_accs.append(row["final_acc"])

        result[rho_col] = rho
        result[cluster_col] = thr_cluster

        result[f"{thr_col}_selected"] = row["raw_thr_selected"]
        result[f"{thr_col}_cluster_mean"] = row["raw_thr_mean"]
        result[f"{thr_col}_cluster_median"] = row["raw_thr_median"]
        result[f"{thr_col}_cluster_min"] = row["raw_thr_min"]
        result[f"{thr_col}_cluster_max"] = row["raw_thr_max"]

        result[f"source_exit{exit_name}_count"] = row["count"]
        result[f"source_exit{exit_name}_overall_acc"] = row["source_overall_acc"]
        result[f"source_exit{exit_name}_final_rate"] = row["source_final_rate"]

    global_rates, final_rate = reconstruct_global_rates(cond_probs)

    if final_acc_mode == "worst":
        final_acc = float(np.nanmin(final_accs))
    else:
        final_acc = float(np.nanmean(final_accs))

    pred_acc = 0.0

    for r, a in zip(global_rates, exit_accs):
        if pd.isna(a):
            return None
        pred_acc += r * (float(a) / 100.0)

    pred_acc += final_rate * (final_acc / 100.0)
    pred_acc *= 100.0

    result["pred_acc"] = pred_acc
    result["pred_final_rate"] = final_rate * 100.0
    result["pred_final_acc"] = final_acc
    result["prediction_type"] = "clustered_conditional_rate_reconstruction"
    result["is_measured"] = False

    for spec, r, a, p in zip(exit_specs, global_rates, exit_accs, cond_probs):
        name = spec["name"]
        result[f"pred_exit{name}_rate"] = r * 100.0
        result[f"pred_exit{name}_acc"] = a
        result[f"cond_exit{name}"] = p

    return result


def deduplicate_pred_candidates(pred_df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Deduplicate by:
        method + rho pattern + threshold clusters

    Keep:
        highest pred_acc, then lowest pred_final_rate.
    """
    if pred_df.empty:
        return pred_df

    exit_specs = get_exit_specs(task)

    key_cols = ["method"]
    key_cols += [spec["rho_col"] for spec in exit_specs]
    key_cols += [f"{spec['thr_col']}_cluster" for spec in exit_specs]
    key_cols = [c for c in key_cols if c in pred_df.columns]

    pred_df = pred_df.sort_values(
        ["pred_acc", "pred_final_rate"],
        ascending=[False, True],
    )

    return pred_df.drop_duplicates(subset=key_cols, keep="first").copy()


def generate_predicted_mixed_candidates(
    df: pd.DataFrame,
    task: str,
    baseline_val_acc: float,
    epsilon: float,
    pred_acc_margin: float,
    top_k: int,
    profile_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate mixed-pruning predicted candidates.

    Important:
        - Uses validation rows only.
        - Splits all/constrain by profile_mode.
        - Uses threshold clusters for candidate identity.
        - Outputs selected raw thresholds for direct test rerun.
    """
    target_acc = baseline_val_acc - epsilon
    pred_target_acc = target_acc + pred_acc_margin

    df_val = df[df["split"] == "val"].copy()

    if df_val.empty:
        raise ValueError(f"{task}: no validation rows found.")

    df_val = filter_profile_mode(df_val, profile_mode)
    df_val = expand_shared_pruning(df_val, task)
    df_val = add_conditional_exit_probs(df_val, task)
    df_val = add_threshold_clusters(
        df_val,
        task=task,
        k=THRESHOLD_CLUSTER_K[task],
    )

    profiles = build_local_exit_profile(df_val, task)
    exit_specs = get_exit_specs(task)
    rho_patterns = get_rho_patterns(task)

    candidates = []
    seen = set()

    for _, row in df_val.iterrows():
        method = row.get("method", None)

        if method is None or pd.isna(method):
            continue

        thr_cluster_vector = {}
        ok = True

        for spec in exit_specs:
            cluster_col = f"{spec['thr_col']}_cluster"

            if cluster_col not in row.index or pd.isna(row[cluster_col]):
                ok = False
                break

            thr_cluster_vector[cluster_col] = int(row[cluster_col])

        if not ok:
            continue

        for rho_pattern in rho_patterns:
            pred = predict_mixed_candidate(
                profiles=profiles,
                task=task,
                method=str(method),
                thr_cluster_vector=thr_cluster_vector,
                rho_pattern=rho_pattern,
                profile_mode=profile_mode,
                final_acc_mode="mean",
            )

            if pred is None:
                continue

            key = (
                pred.get("method"),
                tuple(rho_pattern),
                tuple(pred.get(f"{spec['thr_col']}_cluster") for spec in exit_specs),
            )

            if key in seen:
                continue

            seen.add(key)

            pred["target_acc"] = target_acc
            pred["pred_target_acc"] = pred_target_acc
            pred["is_pred_feasible"] = pred["pred_acc"] >= pred_target_acc

            candidates.append(pred)

    pred_df = pd.DataFrame(candidates)

    if pred_df.empty:
        return pred_df, pred_df

    pred_df = deduplicate_pred_candidates(pred_df, task)

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


# ============================================================
# Shared measured optimal selection
# ============================================================

def get_config_cols(df: pd.DataFrame, task: str) -> list[str]:
    exit_specs = get_exit_specs(task)

    possible_cols = ["method", "pruning_rate"]

    for spec in exit_specs:
        possible_cols.append(spec["rho_col"])

    for spec in exit_specs:
        possible_cols.append(spec["thr_col"])

    for spec in exit_specs:
        possible_cols.append(spec["mask_col"])

    possible_cols.append("mask_id")

    return [c for c in possible_cols if c in df.columns]


def match_test_row(
    df_test: pd.DataFrame,
    best_val: pd.Series,
    config_cols: list[str],
) -> pd.DataFrame:
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


def select_best_shared_measured_point(
    df: pd.DataFrame,
    task: str,
    baseline_val_acc: float,
    epsilon: float,
    profile_mode: str,
) -> dict:
    """
    Shared-pruning measured selection.

    This is the measured baseline selection, not mixed prediction.
    """
    target_acc = baseline_val_acc - epsilon

    df_mode = filter_profile_mode(df, profile_mode)
    df_val = df_mode[df_mode["split"] == "val"].copy()
    df_test = df_mode[df_mode["split"] == "test"].copy()

    if df_val.empty:
        raise ValueError(f"{task} | {profile_mode}: no validation rows.")

    feasible = df_val[df_val["overall_acc"] >= target_acc].copy()

    if feasible.empty:
        raise ValueError(
            f"{task} | {profile_mode}: no feasible shared measured candidates. "
            f"target={target_acc:.2f}, max_val={df_val['overall_acc'].max():.2f}"
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

    if matched_test.empty:
        best_test = None
    else:
        best_test = matched_test.iloc[0]

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
    }


def summarize_task_exits(task: str, prefix: str, row_data: pd.Series) -> dict:
    out = {}

    for spec in get_exit_specs(task):
        name = spec["name"]
        out[f"{prefix}_exit{name}_rate"] = row_data.get(spec["rate_col"], np.nan)
        out[f"{prefix}_exit{name}_acc"] = row_data.get(spec["acc_col"], np.nan)

    return out


def summarize_shared_selection(result: dict) -> dict:
    task = result["task"]
    best_val = result["best_val"]
    best_test = result["best_test"]

    row = {
        "task": task,
        "profile_mode": result["profile_mode"],
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

    return row


# ============================================================
# Export
# ============================================================

def load_and_prepare_csv(csv_path: str, task: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]

    df = normalize_columns(df, task)
    df = ensure_numeric(df)
    df = normalize_pruning_scale(df)
    df = normalize_split(df)
    df = expand_shared_pruning(df, task)

    return df


def export_single_task(
    csv_path: str,
    task: str,
    baseline_val_acc: float,
    epsilon: float,
    pred_acc_margin: float,
    export_dir: str,
    top_k: int,
) -> list[dict]:
    df = load_and_prepare_csv(csv_path, task)

    print(f"\n[{task}] split counts:")
    print(df["split"].value_counts(dropna=False).to_string())

    if "pruning_rate" in df.columns:
        print(f"\n[{task}] unique pruning_rate:")
        print(sorted(df["pruning_rate"].dropna().unique()))

    outputs = []

    for profile_mode in PROFILE_MODES:
        mode_dir = os.path.join(export_dir, task, profile_mode)
        os.makedirs(mode_dir, exist_ok=True)

        print(f"\n=== Processing {task.upper()} | profile_mode={profile_mode} ===")

        # Shared measured selection
        shared_result = select_best_shared_measured_point(
            df=df,
            task=task,
            baseline_val_acc=baseline_val_acc,
            epsilon=epsilon,
            profile_mode=profile_mode,
        )

        shared_summary = summarize_shared_selection(shared_result)
        shared_summary_df = pd.DataFrame([shared_summary])

        feasible_df = shared_result["feasible"].copy()
        feasible_df["task"] = task
        feasible_df["profile_mode"] = profile_mode
        feasible_df["target_acc"] = shared_result["target_acc"]
        feasible_df["is_selected"] = False

        best_index = shared_result["best_val"].name

        if best_index in feasible_df.index:
            feasible_df.loc[best_index, "is_selected"] = True

        # Predicted mixed-pruning selection
        pred_df, topk_df = generate_predicted_mixed_candidates(
            df=df,
            task=task,
            baseline_val_acc=baseline_val_acc,
            epsilon=epsilon,
            pred_acc_margin=pred_acc_margin,
            top_k=top_k,
            profile_mode=profile_mode,
        )

        # Export paths
        shared_summary_path = os.path.join(
            mode_dir,
            f"{task}_shared_measured_optimal_summary_{profile_mode}.csv",
        )

        feasible_path = os.path.join(
            mode_dir,
            f"{task}_shared_measured_feasible_candidates_{profile_mode}.csv",
        )

        pred_path = os.path.join(
            mode_dir,
            f"{task}_predicted_mixed_pruning_candidates_{profile_mode}.csv",
        )

        topk_path = os.path.join(
            mode_dir,
            f"{task}_predicted_mixed_pruning_top{top_k}_{profile_mode}.csv",
        )

        shared_summary_df.to_csv(shared_summary_path, index=False)
        feasible_df.to_csv(feasible_path, index=False)
        pred_df.to_csv(pred_path, index=False)
        topk_df.to_csv(topk_path, index=False)

        print("\nShared measured optimal:")
        print(shared_summary_df.to_string(index=False))

        print(f"\nPredicted mixed-pruning top-{top_k}:")
        if topk_df.empty:
            print("[Warning] no predicted candidates.")
        else:
            print(topk_df.to_string(index=False))

        print("\nSaved:")
        print(f"  {shared_summary_path}")
        print(f"  {feasible_path}")
        print(f"  {pred_path}")
        print(f"  {topk_path}")

        outputs.append({
            "task": task,
            "profile_mode": profile_mode,
            "shared_summary": shared_summary,
            "shared_summary_path": shared_summary_path,
            "feasible_path": feasible_path,
            "pred_path": pred_path,
            "topk_path": topk_path,
        })

    return outputs


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)

    all_shared_summaries = []

    fmnist_outputs = export_single_task(
        csv_path=FMNIST_CSV,
        task="fmnist",
        baseline_val_acc=BASELINE_VAL_ACC["fmnist"],
        epsilon=EPSILON,
        pred_acc_margin=PRED_ACC_MARGIN,
        export_dir=EXPORT_DIR,
        top_k=TOP_K,
    )

    cifar10_outputs = export_single_task(
        csv_path=CIFAR10_CSV,
        task="cifar10",
        baseline_val_acc=BASELINE_VAL_ACC["cifar10"],
        epsilon=EPSILON,
        pred_acc_margin=PRED_ACC_MARGIN,
        export_dir=EXPORT_DIR,
        top_k=TOP_K,
    )

    for item in fmnist_outputs + cifar10_outputs:
        all_shared_summaries.append(item["shared_summary"])

    all_summary_df = pd.DataFrame(all_shared_summaries)
    all_summary_path = os.path.join(EXPORT_DIR, "all_profile_modes_shared_measured_summary.csv")
    all_summary_df.to_csv(all_summary_path, index=False)

    print("\n=== All profile modes shared measured summary ===")
    print(all_summary_df.to_string(index=False))
    print(f"\nSaved combined summary:\n  {all_summary_path}")


if __name__ == "__main__":
    main()