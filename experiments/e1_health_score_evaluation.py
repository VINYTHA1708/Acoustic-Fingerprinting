"""Experiment E1 — Health Score Evaluation (Phase 6.4).

Evaluates how well ``combined_health_score`` separates Normal from Abnormal
recordings using:

  - Descriptive statistics per label (mean, median, std, min, max, count)
  - Mann-Whitney U test (two-sided) and p-value
  - Rank-biserial correlation (effect size)
  - ROC-AUC  (anomaly_score = 100 - combined_health_score)

Usage:
    python experiments/e1_health_score_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.e1_health_calibration import (
    compute_health_scores,
    compute_healthy_reference,
    load_csv,
)
from experiments.e1_health_score_calculation import (
    COMBINED_SCORE_COLUMN,
    compute_combined_health_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "E1"
STAGE = "Health Score Evaluation"

INPUT_CSV = Path("experiments/results/e1/evaluation_results.csv")
OUTPUT_DIR = Path("experiments/results/e1/health_calibration")
EVALUATION_CSV = OUTPUT_DIR / "health_score_evaluation.csv"

EVALUATION_COLUMNS = [
    "normal_count", "normal_mean", "normal_median", "normal_std",
    "normal_min", "normal_max",
    "abnormal_count", "abnormal_mean", "abnormal_median", "abnormal_std",
    "abnormal_min", "abnormal_max",
    "u_statistic", "p_value",
    "rank_biserial_correlation",
    "roc_auc",
]

REQUIRED_COLUMNS = {"true_label", COMBINED_SCORE_COLUMN}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_combined_df(df: pd.DataFrame) -> None:
    """Validate that *df* has the columns needed for evaluation.

    Raises:
        ValueError: If the DataFrame is empty or missing required columns.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not pd.api.types.is_numeric_dtype(df[COMBINED_SCORE_COLUMN]):
        raise ValueError(f"Column '{COMBINED_SCORE_COLUMN}' is not numeric.")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate how well ``combined_health_score`` separates the two labels.

    Args:
        df: DataFrame containing ``true_label`` and ``combined_health_score``.
            Labels must include at least one ``"normal"`` and one ``"abnormal"``
            row.

    Returns:
        Single-row DataFrame with columns defined by :data:`EVALUATION_COLUMNS`.

    Raises:
        ValueError: If required columns are missing, the DataFrame is empty,
            or either label group is absent.
    """
    validate_combined_df(df)

    normal = df.loc[df["true_label"] == "normal", COMBINED_SCORE_COLUMN]
    abnormal = df.loc[df["true_label"] == "abnormal", COMBINED_SCORE_COLUMN]

    if normal.empty:
        raise ValueError("No 'normal' recordings found.")
    if abnormal.empty:
        raise ValueError("No 'abnormal' recordings found.")

    u_stat, p_val = mannwhitneyu(normal, abnormal, alternative="two-sided")
    rbc = 1.0 - (2.0 * float(u_stat)) / (len(normal) * len(abnormal))

    # Anomaly score: higher value = more likely abnormal
    anomaly_scores = 100.0 - df[COMBINED_SCORE_COLUMN]
    binary_labels = (df["true_label"] == "abnormal").astype(int)
    auc = float(roc_auc_score(binary_labels, anomaly_scores))

    row = {
        "normal_count":   int(normal.count()),
        "normal_mean":    float(normal.mean()),
        "normal_median":  float(normal.median()),
        "normal_std":     float(normal.std()),
        "normal_min":     float(normal.min()),
        "normal_max":     float(normal.max()),
        "abnormal_count": int(abnormal.count()),
        "abnormal_mean":  float(abnormal.mean()),
        "abnormal_median": float(abnormal.median()),
        "abnormal_std":   float(abnormal.std()),
        "abnormal_min":   float(abnormal.min()),
        "abnormal_max":   float(abnormal.max()),
        "u_statistic":    float(u_stat),
        "p_value":        float(p_val),
        "rank_biserial_correlation": rbc,
        "roc_auc":        auc,
    }
    return pd.DataFrame([row], columns=EVALUATION_COLUMNS)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_evaluation(df: pd.DataFrame, path: Path) -> None:
    """Save the evaluation DataFrame to CSV, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "=", width: int = 50) -> None:
    print(char * width)


def print_evaluation(ev: pd.DataFrame) -> None:
    """Print a formatted evaluation summary."""
    row = ev.iloc[0]
    _sep()
    print(f"Experiment ID : {EXPERIMENT_ID}")
    print(f"Stage         : {STAGE}")
    _sep()
    print()
    _sep()
    print("Health Score Evaluation — Normal vs Abnormal")
    _sep()
    print(
        f"  {'Normal':<10}  "
        f"count={int(row['normal_count']):>4}  "
        f"mean={row['normal_mean']:>6.2f}  "
        f"median={row['normal_median']:>6.2f}  "
        f"std={row['normal_std']:>6.2f}  "
        f"min={row['normal_min']:>6.2f}  "
        f"max={row['normal_max']:>6.2f}"
    )
    print(
        f"  {'Abnormal':<10}  "
        f"count={int(row['abnormal_count']):>4}  "
        f"mean={row['abnormal_mean']:>6.2f}  "
        f"median={row['abnormal_median']:>6.2f}  "
        f"std={row['abnormal_std']:>6.2f}  "
        f"min={row['abnormal_min']:>6.2f}  "
        f"max={row['abnormal_max']:>6.2f}"
    )
    print()
    sig_tag = "*" if row["p_value"] < 0.05 else " "
    print(f"  Mann-Whitney U : {row['u_statistic']:.1f}")
    print(f"  p-value        : {row['p_value']:.4e}  {sig_tag}")
    print(f"  Effect size    : {row['rank_biserial_correlation']:+.4f}  (rank-biserial r)")
    print(f"  ROC-AUC        : {row['roc_auc']:.4f}")
    print("  (* p < 0.05)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_csv(INPUT_CSV)
    ref = compute_healthy_reference(df)
    scored = compute_health_scores(df, ref)
    combined = compute_combined_health_score(scored)

    ev = compute_evaluation(combined)
    print_evaluation(ev)
    save_evaluation(ev, EVALUATION_CSV)
    print(f"Saved: {EVALUATION_CSV}")
    print()


if __name__ == "__main__":
    main()
