"""Generate per_type_summary.json for Phase 11 (seed 42 CSVs).

Reads the committed Phase 11 per-type evaluation CSVs, recomputes
ROC-AUC and Cohen's d using the same definitions as phase9_evaluate.py,
verifies total samples == 5522, and writes the artifact.

Usage:
    python experiments/generate_phase11_per_type_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PHASE11_SEED42_DIR = Path("experiments/results/phase11/seed_42")
OUTPUT_PATH = Path("experiments/results/phase11/per_type_summary.json")
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
EXPECTED_TOTAL = 5522


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sa = float(np.std(a, ddof=1))
    sb = float(np.std(b, ddof=1))
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-12 else 0.0


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        auc = float(roc_auc_score(y_true, scores))
        if not np.isnan(auc) and auc < 0.5:
            auc = float(roc_auc_score(y_true, -scores))
        return auc
    except Exception:
        return float("nan")


def compute_type_metrics(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["normalized_euclidean"] = pd.to_numeric(df["normalized_euclidean"], errors="coerce")
    df = df.dropna(subset=["normalized_euclidean"])

    y_true = (df["true_label"] == "abnormal").astype(int).values
    scores = df["normalized_euclidean"].values.astype(float)

    normal_vals = df.loc[df["true_label"] == "normal", "normalized_euclidean"].values.astype(float)
    abnormal_vals = df.loc[df["true_label"] == "abnormal", "normalized_euclidean"].values.astype(float)

    return {
        "roc_auc": round(_roc_auc(y_true, scores), 6),
        "cohens_d": round(_cohens_d(abnormal_vals, normal_vals), 6),
    }


def main() -> None:
    per_type: dict[str, dict] = {}
    grand_total = 0

    for mt in MACHINE_TYPES:
        csv_path = PHASE11_SEED42_DIR / f"evaluation_{mt}.csv"
        df = pd.read_csv(csv_path)

        n_normal = int((df["true_label"] == "normal").sum())
        n_abnormal = int((df["true_label"] == "abnormal").sum())
        total = n_normal + n_abnormal
        grand_total += total

        metrics = compute_type_metrics(df)

        per_type[mt] = {
            "total_samples": total,
            "normal_samples": n_normal,
            "abnormal_samples": n_abnormal,
            "roc_auc": metrics["roc_auc"],
            "cohens_d": metrics["cohens_d"],
        }

        print(f"{mt}: total={total}  normal={n_normal}  abnormal={n_abnormal}"
              f"  ROC-AUC={metrics['roc_auc']:.6f}  Cohen's d={metrics['cohens_d']:.6f}")

    print(f"\nGrand total: {grand_total}")
    assert grand_total == EXPECTED_TOTAL, (
        f"VERIFICATION FAILED: expected {EXPECTED_TOTAL}, got {grand_total}"
    )
    print(f"VERIFICATION PASSED: total samples == {EXPECTED_TOTAL}")

    artifact = {
        "experiment": "phase11",
        "seed": 42,
        "source_csvs": [
            str(PHASE11_SEED42_DIR / f"evaluation_{mt}.csv")
            for mt in MACHINE_TYPES
        ],
        "primary_metric": "normalized_euclidean",
        "total_samples": grand_total,
        "verification": {
            "expected_total": EXPECTED_TOTAL,
            "passed": grand_total == EXPECTED_TOTAL,
        },
        "per_type": per_type,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)

    print(f"\nArtifact saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
