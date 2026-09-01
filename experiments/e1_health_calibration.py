"""Experiment E1 — Healthy Reference Calibration (Phase 6.1 + 6.2).

Phase 6.1: Reads the pre-computed evaluation_results.csv, filters to normal
recordings only, and computes IQR-based reference statistics for each drift metric.

Phase 6.2: Maps each recording's drift values to per-metric health scores [0, 100]
using the healthy reference upper_threshold for scaling.

Usage:
    python experiments/e1_health_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "E1"
STAGE = "Healthy Reference Calibration & Health Score Mapping"

INPUT_CSV = Path("experiments/results/e1/evaluation_results.csv")
OUTPUT_DIR = Path("experiments/results/e1/health_calibration")
HEALTHY_REFERENCE_CSV = OUTPUT_DIR / "healthy_reference.csv"
HEALTH_SCORES_CSV = OUTPUT_DIR / "health_scores.csv"

REQUIRED_COLUMNS = {
    "true_label",
    "normalized_euclidean",
    "normalized_manhattan",
    "normalized_cosine",
}

DRIFT_METRICS = ["normalized_euclidean", "normalized_manhattan", "normalized_cosine"]

REFERENCE_COLUMNS = [
    "metric", "count", "mean", "std", "median",
    "q1", "q3", "lower_threshold", "upper_threshold",
]

HEALTH_SCORE_COLUMNS = [
    "euclidean_health_score",
    "manhattan_health_score",
    "cosine_health_score",
]

_METRIC_TO_HEALTH_COL = {
    "normalized_euclidean": "euclidean_health_score",
    "normalized_manhattan": "manhattan_health_score",
    "normalized_cosine": "cosine_health_score",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_csv(df: pd.DataFrame) -> None:
    """Validate the evaluation CSV before calibration.

    Raises:
        ValueError: On any structural or content problem.
    """
    if df.empty:
        raise ValueError("CSV is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col in DRIFT_METRICS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is not numeric.")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    """Load and validate the evaluation CSV.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If validation fails.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {path}\n"
            "Run experiments/e1_evaluate.py first."
        )
    df = pd.read_csv(path)
    validate_csv(df)
    return df


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_healthy_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Compute IQR-based reference statistics from normal recordings only.

    Args:
        df: Full evaluation DataFrame (may contain both normal and abnormal rows).

    Returns:
        DataFrame with one row per drift metric and columns:
        metric, count, mean, std, median, q1, q3,
        lower_threshold, upper_threshold.

    Raises:
        ValueError: If required columns are missing or no normal recordings exist.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    normal = df[df["true_label"] == "normal"]
    if normal.empty:
        raise ValueError("No normal recordings found in the DataFrame.")

    rows = []
    for metric in DRIFT_METRICS:
        s = normal[metric]
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        rows.append({
            "metric": metric,
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(s.median()),
            "q1": q1,
            "q3": q3,
            "lower_threshold": q1 - 1.5 * iqr,
            "upper_threshold": q3 + 1.5 * iqr,
        })
    return pd.DataFrame(rows, columns=REFERENCE_COLUMNS)


# ---------------------------------------------------------------------------
# Phase 6.2 — Per-Metric Health Score Mapping
# ---------------------------------------------------------------------------

def compute_health_scores(df: pd.DataFrame, healthy_reference: pd.DataFrame) -> pd.DataFrame:
    """Add per-metric health score columns to the recordings DataFrame.

    For each drift metric the score is:
        health_score = clip(100 * (1 - drift / upper_threshold), 0, 100)

    Args:
        df: Full evaluation DataFrame containing the three drift metric columns.
        healthy_reference: Output of :func:`compute_healthy_reference` — one row
            per metric with an ``upper_threshold`` column.

    Returns:
        Copy of *df* with three additional columns:
        ``euclidean_health_score``, ``manhattan_health_score``,
        ``cosine_health_score``.

    Raises:
        ValueError: If required drift columns are missing from *df*, or if
            *healthy_reference* is missing expected metrics.
    """
    missing = set(DRIFT_METRICS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing drift columns in recordings: {sorted(missing)}")

    ref_metrics = set(healthy_reference["metric"].tolist())
    missing_ref = set(DRIFT_METRICS) - ref_metrics
    if missing_ref:
        raise ValueError(f"Missing metrics in healthy_reference: {sorted(missing_ref)}")

    thresholds = (
        healthy_reference.set_index("metric")["upper_threshold"].to_dict()
    )

    result = df.copy()
    for metric, col in _METRIC_TO_HEALTH_COL.items():
        upper = thresholds[metric]
        if upper == 0:
            result[col] = 0.0
        else:
            raw = 100.0 * (1.0 - result[metric] / upper)
            result[col] = raw.clip(0.0, 100.0)
    return result


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_healthy_reference(df: pd.DataFrame, path: Path) -> None:
    """Save the healthy reference DataFrame to CSV, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_health_scores(df: pd.DataFrame, path: Path) -> None:
    """Save the health scores DataFrame to CSV, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "=", width: int = 50) -> None:
    print(char * width)


def print_healthy_reference(ref: pd.DataFrame) -> None:
    _sep()
    print(f"Experiment ID : {EXPERIMENT_ID}")
    print(f"Stage         : {STAGE}")
    _sep()
    print()
    _sep()
    print("Healthy Reference Calibration (normal recordings only)")
    _sep()
    for _, row in ref.iterrows():
        print(f"\n  {row['metric']}")
        print(f"    count            : {int(row['count'])}")
        print(f"    mean             : {row['mean']:.4f}")
        print(f"    std              : {row['std']:.4f}")
        print(f"    median           : {row['median']:.4f}")
        print(f"    Q1               : {row['q1']:.4f}")
        print(f"    Q3               : {row['q3']:.4f}")
        print(f"    lower_threshold  : {row['lower_threshold']:.4f}")
        print(f"    upper_threshold  : {row['upper_threshold']:.4f}")
    print()


def print_health_scores(df: pd.DataFrame) -> None:
    """Print summary statistics for the three per-metric health score columns."""
    _sep()
    print("Per-Metric Health Score Summary")
    _sep()
    for col in HEALTH_SCORE_COLUMNS:
        s = df[col]
        print(
            f"  {col:<28}  "
            f"count={int(s.count()):>4}  "
            f"mean={s.mean():>6.2f}  "
            f"std={s.std():>6.2f}  "
            f"min={s.min():>6.2f}  "
            f"median={s.median():>6.2f}  "
            f"max={s.max():>6.2f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_csv(INPUT_CSV)
    ref = compute_healthy_reference(df)
    print_healthy_reference(ref)
    save_healthy_reference(ref, HEALTHY_REFERENCE_CSV)
    print(f"Saved: {HEALTHY_REFERENCE_CSV}")
    print()

    scored = compute_health_scores(df, ref)
    print_health_scores(scored)
    save_health_scores(scored, HEALTH_SCORES_CSV)
    print(f"Saved: {HEALTH_SCORES_CSV}")
    print()


if __name__ == "__main__":
    main()
