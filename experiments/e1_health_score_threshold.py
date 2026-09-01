"""Experiment E1 — Health Score Threshold Calibration (Phase 6.5).

Selects an optimal threshold for ``combined_health_score`` using Youden's J
statistic (maximises sensitivity + specificity − 1 over all ROC thresholds).

Score direction:
    anomaly_score = 100 − combined_health_score   (higher → more abnormal)
    predicted abnormal  ⟺  combined_health_score < threshold

Classification metrics reported:
    accuracy, precision, recall, F1-score,
    sensitivity (= recall for abnormal), specificity (= recall for normal),
    confusion matrix (tp, fp, tn, fn)

Usage:
    python experiments/e1_health_score_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

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
STAGE = "Health Score Threshold Calibration"

INPUT_CSV = Path("experiments/results/e1/evaluation_results.csv")
OUTPUT_DIR = Path("experiments/results/e1/health_calibration")
THRESHOLD_CSV = OUTPUT_DIR / "health_score_threshold.csv"

REQUIRED_COLUMNS = {"true_label", COMBINED_SCORE_COLUMN}

THRESHOLD_COLUMNS = [
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "sensitivity",
    "specificity",
    "tp", "fp", "tn", "fn",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_df(df: pd.DataFrame) -> None:
    """Validate that *df* has the columns needed for threshold calibration.

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

def find_optimal_threshold(df: pd.DataFrame) -> float:
    """Return the Youden-J optimal threshold on ``combined_health_score``.

    Uses ``anomaly_score = 100 − combined_health_score`` so that higher
    anomaly scores correspond to abnormal recordings, then maps the selected
    ROC threshold back to the health-score space.

    Args:
        df: DataFrame with ``true_label`` and ``combined_health_score``.

    Returns:
        Optimal health-score threshold (float).  Recordings with
        ``combined_health_score < threshold`` are predicted abnormal.

    Raises:
        ValueError: If required columns are missing, the DataFrame is empty,
            or either label is absent.
    """
    validate_df(df)

    normal = df.loc[df["true_label"] == "normal", COMBINED_SCORE_COLUMN]
    abnormal = df.loc[df["true_label"] == "abnormal", COMBINED_SCORE_COLUMN]
    if normal.empty:
        raise ValueError("No 'normal' recordings found.")
    if abnormal.empty:
        raise ValueError("No 'abnormal' recordings found.")

    binary_labels = (df["true_label"] == "abnormal").astype(int)
    anomaly_scores = 100.0 - df[COMBINED_SCORE_COLUMN]

    fpr, tpr, roc_thresholds = roc_curve(binary_labels, anomaly_scores)
    j_scores = tpr - fpr                          # Youden's J = sensitivity + specificity − 1
    best_idx = int(np.argmax(j_scores))
    optimal_anomaly_threshold = float(roc_thresholds[best_idx])

    # Convert back to health-score space
    return 100.0 - optimal_anomaly_threshold


def compute_threshold_metrics(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Classify every recording using *threshold* and compute metrics.

    Predicted label: ``abnormal`` if ``combined_health_score < threshold``,
    else ``normal``.

    Args:
        df: DataFrame with ``true_label`` and ``combined_health_score``.
        threshold: Health-score decision boundary.

    Returns:
        Single-row DataFrame with columns defined by :data:`THRESHOLD_COLUMNS`.

    Raises:
        ValueError: If required columns are missing or the DataFrame is empty.
    """
    validate_df(df)

    y_true = (df["true_label"] == "abnormal").astype(int)
    y_pred = (df[COMBINED_SCORE_COLUMN] < threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = float(tp) / (float(tp) + float(fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn) / (float(tn) + float(fp)) if (tn + fp) > 0 else 0.0

    row = {
        "threshold":   threshold,
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score":    float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    return pd.DataFrame([row], columns=THRESHOLD_COLUMNS)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_threshold_results(df: pd.DataFrame, path: Path) -> None:
    """Save the threshold results DataFrame to CSV, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "=", width: int = 50) -> None:
    print(char * width)


def print_threshold_results(results: pd.DataFrame) -> None:
    """Print a formatted threshold calibration summary."""
    row = results.iloc[0]
    _sep()
    print(f"Experiment ID : {EXPERIMENT_ID}")
    print(f"Stage         : {STAGE}")
    _sep()
    print()
    _sep()
    print("Optimal Threshold & Classification Metrics")
    _sep()
    print(f"  Threshold    : {row['threshold']:.4f}")
    print()
    print(f"  Accuracy     : {row['accuracy']:.4f}")
    print(f"  Precision    : {row['precision']:.4f}")
    print(f"  Recall       : {row['recall']:.4f}")
    print(f"  F1-score     : {row['f1_score']:.4f}")
    print()
    print(f"  Sensitivity  : {row['sensitivity']:.4f}  (true positive rate)")
    print(f"  Specificity  : {row['specificity']:.4f}  (true negative rate)")
    print()
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                  Pred Normal  Pred Abnormal")
    print(f"  Actual Normal   {int(row['tn']):>11}  {int(row['fp']):>13}")
    print(f"  Actual Abnormal {int(row['fn']):>11}  {int(row['tp']):>13}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_csv(INPUT_CSV)
    ref = compute_healthy_reference(df)
    scored = compute_health_scores(df, ref)
    combined = compute_combined_health_score(scored)

    threshold = find_optimal_threshold(combined)
    results = compute_threshold_metrics(combined, threshold)

    print_threshold_results(results)
    save_threshold_results(results, THRESHOLD_CSV)
    print(f"Saved: {THRESHOLD_CSV}")
    print()


if __name__ == "__main__":
    main()
