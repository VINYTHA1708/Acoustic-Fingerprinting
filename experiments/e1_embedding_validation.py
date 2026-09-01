"""Experiment E1 — Embedding Validation.

Scientifically validates whether the trained contrastive embedding pipeline
produces meaningful separation between healthy and abnormal held-out recordings.

This script is analysis-only. It reads the pre-computed evaluation_results.csv
and does NOT rerun BEATs, preprocessing, DSP, fusion, ProjectionHead, profile
building, or LearnedHealthAnalyzer.

Usage:
    python experiments/e1_embedding_validation.py

Metric interpretation (from src/learned_drift/metrics.py):
    normalized_euclidean:
        L2 norm of the z-score vector  z = (embedding - mean) / std.
        Measures how far the recording deviates from the healthy profile in
        standard-deviation units across all 256 dimensions.
        DIRECTION: larger value = greater deviation = more likely abnormal.

    normalized_manhattan:
        L1 norm of the z-score vector.
        Sum of absolute per-dimension z-scores.
        DIRECTION: larger value = greater deviation = more likely abnormal.

    normalized_cosine:
        Cosine similarity of the z-score vector against the all-ones (uniform)
        direction.  This measures whether the z-score deviations are uniformly
        distributed across dimensions.  It is NOT a similarity to the healthy
        profile mean.  Values can be negative (z-scores cancel out).
        DIRECTION: ambiguous — inspect AUC; invert score if AUC < 0.5.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# E1 constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "E1"
MACHINE_TYPE = "pump"
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]

INPUT_CSV = Path("experiments/results/e1/evaluation_results.csv")
OUTPUT_DIR = Path("experiments/results/e1/embedding_validation")

EXPECTED_NORMAL = 566
EXPECTED_ABNORMAL = 456
EXPECTED_TOTAL = 1022

REQUIRED_COLUMNS = {
    "machine_type", "machine_id", "filename", "true_label",
    "health_score", "health_percentage", "health_state",
    "normalized_euclidean", "normalized_manhattan", "normalized_cosine",
}

DRIFT_METRICS = ["normalized_euclidean", "normalized_manhattan", "normalized_cosine"]
ALL_METRICS = DRIFT_METRICS + ["health_score", "health_percentage_float"]

# For each drift metric: True means larger value = more likely abnormal.
# normalized_cosine direction is determined empirically from AUC.
_METRIC_LARGER_IS_ABNORMAL: dict[str, bool] = {
    "normalized_euclidean": True,
    "normalized_manhattan": True,
    "normalized_cosine": None,  # determined at runtime
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_duplicates(df: pd.DataFrame) -> None:
    # The unique key is (machine_type, machine_id, filename, true_label).
    # The same filename can legitimately appear in both normal/ and abnormal/
    # subdirectories in the MIMII dataset, so filename alone is not unique.
    dup_mask = df.duplicated(subset=["machine_type", "machine_id", "filename", "true_label"])
    if dup_mask.any():
        raise ValueError(f"Duplicate rows detected: {dup_mask.sum()} duplicates.")


def validate_csv(df: pd.DataFrame, path: Path) -> None:
    """Fail early with clear errors if the CSV does not meet E1 expectations."""
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    unexpected_labels = set(df["true_label"].unique()) - {"normal", "abnormal"}
    if unexpected_labels:
        raise ValueError(f"Unexpected true_label values: {unexpected_labels}")

    total = len(df)
    if total != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} rows, got {total}.")

    n_normal = int((df["true_label"] == "normal").sum())
    n_abnormal = int((df["true_label"] == "abnormal").sum())
    if n_normal != EXPECTED_NORMAL:
        raise ValueError(f"Expected {EXPECTED_NORMAL} normal rows, got {n_normal}.")
    if n_abnormal != EXPECTED_ABNORMAL:
        raise ValueError(f"Expected {EXPECTED_ABNORMAL} abnormal rows, got {n_abnormal}.")

    _check_duplicates(df)

    for col in DRIFT_METRICS:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaN values.")
        if np.isinf(df[col].values).any():
            raise ValueError(f"Column '{col}' contains Inf values.")

    for mid in MACHINE_IDS:
        sub = df[df["machine_id"] == mid]
        if (sub["true_label"] == "normal").sum() == 0:
            raise ValueError(f"{mid}: no normal recordings found.")
        if (sub["true_label"] == "abnormal").sum() == 0:
            raise ValueError(f"{mid}: no abnormal recordings found.")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _stats(values: pd.Series) -> dict:
    """Return count/mean/std/median/min/max as native Python floats."""
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """Cohen's d using pooled standard deviation.

    d = (mean_a - mean_b) / s_pooled
    s_pooled = sqrt(((n_a-1)*s_a^2 + (n_b-1)*s_b^2) / (n_a + n_b - 2))

    Returns 0.0 if pooled std is effectively zero.
    """
    n_a, n_b = len(group_a), len(group_b)
    s_a, s_b = float(group_a.std(ddof=1)), float(group_b.std(ddof=1))
    pooled_var = ((n_a - 1) * s_a ** 2 + (n_b - 1) * s_b ** 2) / (n_a + n_b - 2)
    s_pooled = float(np.sqrt(pooled_var))
    if s_pooled < 1e-12:
        return 0.0
    return float((group_a.mean() - group_b.mean()) / s_pooled)


def compute_auc(
    df: pd.DataFrame,
    metric: str,
    larger_is_abnormal: bool,
) -> float:
    """Compute ROC-AUC for one metric. Positive class = abnormal.

    If larger_is_abnormal is True, use the metric directly as the anomaly score.
    If False, negate the metric so that larger score = more likely abnormal.
    """
    y_true = (df["true_label"] == "abnormal").astype(int)
    scores = df[metric].values
    if not larger_is_abnormal:
        scores = -scores
    return float(roc_auc_score(y_true, scores))


def determine_cosine_direction(df: pd.DataFrame) -> bool:
    """Determine whether larger normalized_cosine = more likely abnormal.

    Computes AUC in both directions and picks the one with AUC >= 0.5.
    Returns True if larger = abnormal, False if smaller = abnormal.
    """
    y_true = (df["true_label"] == "abnormal").astype(int)
    scores = df["normalized_cosine"].values
    auc_direct = float(roc_auc_score(y_true, scores))
    return auc_direct >= 0.5


# ---------------------------------------------------------------------------
# Overall analysis
# ---------------------------------------------------------------------------

def overall_metric_statistics(df: pd.DataFrame) -> list[dict]:
    rows = []
    normal = df[df["true_label"] == "normal"]
    abnormal = df[df["true_label"] == "abnormal"]

    # Parse health_percentage string → float (e.g. "95.6%" → 95.6)
    df = df.copy()
    df["health_percentage_float"] = (
        df["health_percentage"].str.rstrip("%").astype(float)
    )
    normal = df[df["true_label"] == "normal"]
    abnormal = df[df["true_label"] == "abnormal"]

    for metric in DRIFT_METRICS + ["health_score"]:
        for label, group in [("normal", normal), ("abnormal", abnormal)]:
            s = _stats(group[metric])
            rows.append({"metric": metric, "label": label, **s})

    for label, group in [("normal", normal), ("abnormal", abnormal)]:
        s = _stats(group["health_percentage_float"])
        rows.append({"metric": "health_percentage", "label": label, **s})

    return rows


def per_machine_metric_statistics(df: pd.DataFrame) -> list[dict]:
    rows = []
    df = df.copy()
    df["health_percentage_float"] = (
        df["health_percentage"].str.rstrip("%").astype(float)
    )
    machine_ids = sorted(df["machine_id"].unique())
    for mid in machine_ids:
        sub = df[df["machine_id"] == mid]
        normal = sub[sub["true_label"] == "normal"]
        abnormal = sub[sub["true_label"] == "abnormal"]
        for metric in DRIFT_METRICS + ["health_score", "health_percentage_float"]:
            display_metric = "health_percentage" if metric == "health_percentage_float" else metric
            for label, group in [("normal", normal), ("abnormal", abnormal)]:
                s = _stats(group[metric])
                rows.append({
                    "machine_type": MACHINE_TYPE,
                    "machine_id": mid,
                    "metric": display_metric,
                    "label": label,
                    **s,
                })
    return rows


# ---------------------------------------------------------------------------
# AUC + Cohen's d
# ---------------------------------------------------------------------------

def overall_auc_results(df: pd.DataFrame, cosine_larger_is_abnormal: bool) -> list[dict]:
    rows = []
    directions = {
        "normalized_euclidean": True,
        "normalized_manhattan": True,
        "normalized_cosine": cosine_larger_is_abnormal,
    }
    normal = df[df["true_label"] == "normal"]
    abnormal = df[df["true_label"] == "abnormal"]

    for metric in DRIFT_METRICS:
        larger_is_abnormal = directions[metric]
        auc = compute_auc(df, metric, larger_is_abnormal)
        # Cohen's d: abnormal - normal (positive d = abnormal has higher values)
        d = cohens_d(abnormal[metric], normal[metric])
        rows.append({
            "metric": metric,
            "roc_auc": round(auc, 6),
            "cohens_d": round(d, 6),
            "expected_abnormal_direction": "larger" if larger_is_abnormal else "smaller",
        })
    return rows


def per_machine_auc_results(df: pd.DataFrame, cosine_larger_is_abnormal: bool) -> list[dict]:
    rows = []
    directions = {
        "normalized_euclidean": True,
        "normalized_manhattan": True,
        "normalized_cosine": cosine_larger_is_abnormal,
    }
    machine_ids = sorted(df["machine_id"].unique())
    for mid in machine_ids:
        sub = df[df["machine_id"] == mid]
        normal = sub[sub["true_label"] == "normal"]
        abnormal = sub[sub["true_label"] == "abnormal"]
        for metric in DRIFT_METRICS:
            larger_is_abnormal = directions[metric]
            try:
                auc = compute_auc(sub, metric, larger_is_abnormal)
            except ValueError:
                auc = float("nan")
            d = cohens_d(abnormal[metric], normal[metric]) if len(normal) > 1 and len(abnormal) > 1 else float("nan")
            rows.append({
                "machine_type": MACHINE_TYPE,
                "machine_id": mid,
                "metric": metric,
                "roc_auc": round(auc, 6) if not np.isnan(auc) else None,
                "cohens_d": round(d, 6) if not np.isnan(d) else None,
            })
    return rows


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def build_summary_json(
    df: pd.DataFrame,
    overall_auc: list[dict],
    per_machine_auc: list[dict],
) -> dict:
    machine_ids = sorted(df["machine_id"].unique().tolist())

    overall_results = {
        row["metric"]: {
            "roc_auc": row["roc_auc"],
            "cohens_d": row["cohens_d"],
            "expected_abnormal_direction": row["expected_abnormal_direction"],
        }
        for row in overall_auc
    }

    per_machine_results: dict[str, dict] = {}
    for row in per_machine_auc:
        mid = row["machine_id"]
        if mid not in per_machine_results:
            per_machine_results[mid] = {}
        per_machine_results[mid][row["metric"]] = {
            "roc_auc": row["roc_auc"],
            "cohens_d": row["cohens_d"],
        }

    return {
        "experiment_id": EXPERIMENT_ID,
        "machine_type": MACHINE_TYPE,
        "input_csv": str(INPUT_CSV),
        "total_recordings": EXPECTED_TOTAL,
        "normal_count": EXPECTED_NORMAL,
        "abnormal_count": EXPECTED_ABNORMAL,
        "machine_ids": machine_ids,
        "metrics_analyzed": DRIFT_METRICS,
        "overall_results": overall_results,
        "per_machine_results": per_machine_results,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_separator(char: str = "=", width: int = 50) -> None:
    print(char * width)


def print_metric_interpretation(cosine_larger_is_abnormal: bool) -> None:
    _print_separator()
    print("Metric Interpretation")
    _print_separator()
    print()
    print("normalized_euclidean:")
    print("  L2 norm of the z-score vector (embedding - mean) / std.")
    print("  Measures total deviation from the healthy profile in std units.")
    print("  DIRECTION: larger value = greater deviation = more likely abnormal.")
    print()
    print("normalized_manhattan:")
    print("  L1 norm of the z-score vector. Sum of absolute per-dimension z-scores.")
    print("  DIRECTION: larger value = greater deviation = more likely abnormal.")
    print()
    print("normalized_cosine:")
    print("  Cosine similarity of the z-score vector vs the all-ones (uniform) direction.")
    print("  Measures whether deviations are uniformly distributed across dimensions.")
    print("  NOT a similarity to the healthy profile mean. Can be negative.")
    cosine_dir = "larger" if cosine_larger_is_abnormal else "smaller"
    print(f"  DIRECTION (empirical): {cosine_dir} value = more likely abnormal.")
    print()


def print_overall_separation(
    df: pd.DataFrame,
    overall_auc: list[dict],
) -> None:
    _print_separator()
    print("Overall Separation (Normal vs Abnormal)")
    _print_separator()
    normal = df[df["true_label"] == "normal"]
    abnormal = df[df["true_label"] == "abnormal"]
    for row in overall_auc:
        metric = row["metric"]
        n_mean, n_std = normal[metric].mean(), normal[metric].std()
        a_mean, a_std = abnormal[metric].mean(), abnormal[metric].std()
        print(f"\n  {metric}")
        print(f"    Normal   : {n_mean:.4f} ± {n_std:.4f}")
        print(f"    Abnormal : {a_mean:.4f} ± {a_std:.4f}")
        print(f"    Cohen's d: {row['cohens_d']:.4f}")
        print(f"    ROC-AUC  : {row['roc_auc']:.4f}")
    print()


def print_per_machine_separation(
    df: pd.DataFrame,
    per_machine_auc: list[dict],
) -> None:
    _print_separator()
    print("Per-Machine Separation")
    _print_separator()
    machine_ids = sorted(df["machine_id"].unique())
    auc_lookup = {(r["machine_id"], r["metric"]): r for r in per_machine_auc}
    for mid in machine_ids:
        print(f"\n  Machine: {mid}")
        sub = df[df["machine_id"] == mid]
        normal = sub[sub["true_label"] == "normal"]
        abnormal = sub[sub["true_label"] == "abnormal"]
        for metric in DRIFT_METRICS:
            row = auc_lookup.get((mid, metric), {})
            n_mean, n_std = normal[metric].mean(), normal[metric].std()
            a_mean, a_std = abnormal[metric].mean(), abnormal[metric].std()
            auc_val = row.get("roc_auc", "N/A")
            d_val = row.get("cohens_d", "N/A")
            auc_str = f"{auc_val:.4f}" if isinstance(auc_val, float) else str(auc_val)
            d_str = f"{d_val:.4f}" if isinstance(d_val, float) else str(d_val)
            print(f"    {metric}")
            print(f"      Normal   : {n_mean:.4f} ± {n_std:.4f}  (n={len(normal)})")
            print(f"      Abnormal : {a_mean:.4f} ± {a_std:.4f}  (n={len(abnormal)})")
            print(f"      Cohen's d: {d_str}   ROC-AUC: {auc_str}")
    print()


def print_health_summary(df: pd.DataFrame) -> None:
    _print_separator()
    print("Health Summary")
    _print_separator()
    df = df.copy()
    df["health_percentage_float"] = df["health_percentage"].str.rstrip("%").astype(float)
    for label in ["normal", "abnormal"]:
        group = df[df["true_label"] == label]
        hs_mean, hs_std = group["health_score"].mean(), group["health_score"].std()
        hp_mean, hp_std = group["health_percentage_float"].mean(), group["health_percentage_float"].std()
        print(f"\n  {label.capitalize()}:")
        print(f"    health_score      : {hs_mean:.2f} ± {hs_std:.2f}")
        print(f"    health_percentage : {hp_mean:.2f}% ± {hp_std:.2f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    # --- Load and validate ---
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}\n"
            "Run experiments/e1_evaluate.py first."
        )

    df = pd.read_csv(INPUT_CSV)
    validate_csv(df, INPUT_CSV)

    # --- Determine cosine direction empirically ---
    cosine_larger_is_abnormal = determine_cosine_direction(df)

    # --- Compute analyses ---
    overall_stats = overall_metric_statistics(df)
    per_machine_stats = per_machine_metric_statistics(df)
    overall_auc = overall_auc_results(df, cosine_larger_is_abnormal)
    per_machine_auc = per_machine_auc_results(df, cosine_larger_is_abnormal)
    summary = build_summary_json(df, overall_auc, per_machine_auc)

    # --- Write outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_stats_path = OUTPUT_DIR / "overall_metric_statistics.csv"
    pd.DataFrame(overall_stats).to_csv(overall_stats_path, index=False)

    per_machine_stats_path = OUTPUT_DIR / "per_machine_metric_statistics.csv"
    pd.DataFrame(per_machine_stats).to_csv(per_machine_stats_path, index=False)

    overall_auc_path = OUTPUT_DIR / "overall_auc.csv"
    pd.DataFrame(overall_auc).to_csv(overall_auc_path, index=False)

    per_machine_auc_path = OUTPUT_DIR / "per_machine_auc.csv"
    pd.DataFrame(per_machine_auc).to_csv(per_machine_auc_path, index=False)

    summary_path = OUTPUT_DIR / "embedding_validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # --- Console output ---
    machine_ids = sorted(df["machine_id"].unique())

    _print_separator()
    print(f"Experiment ID : {EXPERIMENT_ID}")
    print(f"Stage         : Embedding Validation")
    _print_separator()
    print()
    print("Input:")
    print(f"  {INPUT_CSV}")
    print()
    print("Dataset:")
    print(f"  Normal recordings   : {EXPECTED_NORMAL}")
    print(f"  Abnormal recordings : {EXPECTED_ABNORMAL}")
    print(f"  Total               : {EXPECTED_TOTAL}")
    print()
    print("Machine IDs:")
    for mid in machine_ids:
        print(f"  {mid}")
    print()

    print_metric_interpretation(cosine_larger_is_abnormal)
    print_overall_separation(df, overall_auc)
    print_per_machine_separation(df, per_machine_auc)
    print_health_summary(df)

    # --- Best metric by AUC ---
    best = max(overall_auc, key=lambda r: r["roc_auc"])
    _print_separator()
    print("Best-Performing Metric (by ROC-AUC)")
    _print_separator()
    print(f"  {best['metric']}  AUC={best['roc_auc']:.4f}  Cohen's d={best['cohens_d']:.4f}")
    print()

    _print_separator()
    print("Output Files")
    _print_separator()
    for p in [overall_stats_path, per_machine_stats_path, overall_auc_path, per_machine_auc_path, summary_path]:
        print(f"  {p}")
    print()


if __name__ == "__main__":
    main()
