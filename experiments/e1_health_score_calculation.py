"""Experiment E1 — Health Score Calculation (Phase 6.3).

Derives a single combined health score [0, 100] per recording from the three
per-metric health scores produced in Phase 6.2.

Formula:
    combined_health_score = mean(euclidean_health_score,
                                 manhattan_health_score,
                                 cosine_health_score)

The combined score is the primary health indicator used in downstream phases.

Usage:
    python experiments/e1_health_score_calculation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.e1_health_calibration import (
    DRIFT_METRICS,
    HEALTH_SCORE_COLUMNS,
    REQUIRED_COLUMNS,
    compute_health_scores,
    compute_healthy_reference,
    load_csv,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "E1"
STAGE = "Health Score Calculation"

INPUT_CSV = Path("experiments/results/e1/evaluation_results.csv")
OUTPUT_DIR = Path("experiments/results/e1/health_calibration")
COMBINED_HEALTH_CSV = OUTPUT_DIR / "combined_health_scores.csv"

COMBINED_SCORE_COLUMN = "combined_health_score"

# Weights applied to [euclidean, manhattan, cosine] health scores.
# Equal weighting by default; must sum to 1.0.
WEIGHTS: dict[str, float] = {
    "euclidean_health_score": 1 / 3,
    "manhattan_health_score": 1 / 3,
    "cosine_health_score":    1 / 3,
}

COMBINED_COLUMNS = [COMBINED_SCORE_COLUMN]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_combined_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add a single ``combined_health_score`` column to a scored DataFrame.

    Expects the three per-metric health score columns produced by
    :func:`~experiments.e1_health_calibration.compute_health_scores` to
    already be present.

    The combined score is the weighted mean of the three per-metric scores,
    clipped to [0, 100].

    Args:
        df: DataFrame that already contains the three per-metric health score
            columns (``euclidean_health_score``, ``manhattan_health_score``,
            ``cosine_health_score``).

    Returns:
        Copy of *df* with one additional column: ``combined_health_score``.

    Raises:
        ValueError: If any of the three per-metric score columns are missing.
    """
    missing = set(HEALTH_SCORE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing per-metric health score columns: {sorted(missing)}"
        )

    result = df.copy()
    combined = sum(result[col] * w for col, w in WEIGHTS.items())
    result[COMBINED_SCORE_COLUMN] = combined.clip(0.0, 100.0)
    return result


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_combined_health_scores(df: pd.DataFrame, path: Path) -> None:
    """Save the combined health scores DataFrame to CSV, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "=", width: int = 50) -> None:
    print(char * width)


def print_combined_health_scores(df: pd.DataFrame) -> None:
    """Print a summary of combined health scores split by true_label."""
    _sep()
    print(f"Experiment ID : {EXPERIMENT_ID}")
    print(f"Stage         : {STAGE}")
    _sep()
    print()
    _sep()
    print("Combined Health Score Summary")
    _sep()

    s_all = df[COMBINED_SCORE_COLUMN]
    print(
        f"  {'All recordings':<18}  "
        f"count={int(s_all.count()):>4}  "
        f"mean={s_all.mean():>6.2f}  "
        f"std={s_all.std():>6.2f}  "
        f"min={s_all.min():>6.2f}  "
        f"median={s_all.median():>6.2f}  "
        f"max={s_all.max():>6.2f}"
    )

    if "true_label" in df.columns:
        for label in ("normal", "abnormal"):
            subset = df.loc[df["true_label"] == label, COMBINED_SCORE_COLUMN]
            if subset.empty:
                continue
            print(
                f"  {label.capitalize():<18}  "
                f"count={int(subset.count()):>4}  "
                f"mean={subset.mean():>6.2f}  "
                f"std={subset.std():>6.2f}  "
                f"min={subset.min():>6.2f}  "
                f"median={subset.median():>6.2f}  "
                f"max={subset.max():>6.2f}"
            )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_csv(INPUT_CSV)
    ref = compute_healthy_reference(df)
    scored = compute_health_scores(df, ref)
    combined = compute_combined_health_score(scored)
    print_combined_health_scores(combined)
    save_combined_health_scores(combined, COMBINED_HEALTH_CSV)
    print(f"Saved: {COMBINED_HEALTH_CSV}")
    print()


if __name__ == "__main__":
    main()
