"""Regenerate evaluation_summary.json from the four per-machine-type CSVs.

Only the pump CSV has been repaired.  Fan, slider, and valve still contain
their smoke-test zero rows (same as when the original summary was produced),
so the only change in the summary will be the pump metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR  = Path("experiments/results/phase9")
SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.json"
CHECKPOINT   = "models/contrastive/phase9/best_projection_head.pt"

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]
METRICS       = ["normalized_euclidean", "normalized_manhattan", "normalized_cosine"]


def _cohens_d(a, b) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-12 else 0.0


def _roc_auc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def _compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty or set(df["true_label"].unique()) < {"normal", "abnormal"}:
        return {}
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    df = df.dropna(subset=METRICS)
    if df.empty:
        return {}
    y_true = (df["true_label"] == "abnormal").astype(int).values
    results = {}
    for m in METRICS:
        scores = df[m].values.astype(float)
        auc = _roc_auc(y_true, scores)
        if not np.isnan(auc) and auc < 0.5:
            auc = _roc_auc(y_true, -scores)
        normal_vals   = df.loc[df["true_label"] == "normal",   m].values.astype(float)
        abnormal_vals = df.loc[df["true_label"] == "abnormal", m].values.astype(float)
        d = _cohens_d(abnormal_vals, normal_vals)
        results[m] = {"roc_auc": round(auc, 6), "cohens_d": round(d, 6)}
    return results


def main() -> None:
    # Load existing summary to preserve smoke_test flag and split params
    with SUMMARY_PATH.open(encoding="utf-8") as fh:
        existing = json.load(fh)

    summary: dict = {}
    all_rows: list[dict] = []

    for mt in MACHINE_TYPES:
        csv_path = RESULTS_DIR / f"evaluation_{mt}.csv"
        df = pd.read_csv(csv_path)

        n_normal   = int((df["true_label"] == "normal").sum())
        n_abnormal = int((df["true_label"] == "abnormal").sum())

        type_metrics = _compute_metrics(df.copy())

        per_id_metrics: dict[str, dict] = {}
        for mid in MACHINE_IDS:
            id_df = df[df["machine_id"] == mid]
            if not id_df.empty:
                per_id_metrics[mid] = _compute_metrics(id_df.copy())

        summary[mt] = {
            "n_normal":        n_normal,
            "n_abnormal":      n_abnormal,
            "overall_metrics": type_metrics,
            "per_id_metrics":  per_id_metrics,
        }
        all_rows.extend(df.to_dict(orient="records"))

        print(f"  {mt}: normal={n_normal}  abnormal={n_abnormal}")
        for m, v in type_metrics.items():
            print(f"    {m}: AUC={v['roc_auc']:.4f}  d={v['cohens_d']:.4f}")

    all_df = pd.DataFrame(all_rows)
    overall_metrics = _compute_metrics(all_df.copy())

    summary["overall"] = {
        "n_normal":        int((all_df["true_label"] == "normal").sum()),
        "n_abnormal":      int((all_df["true_label"] == "abnormal").sum()),
        "overall_metrics": overall_metrics,
    }

    summary_meta = {
        "experiment_id": existing["experiment_id"],
        "checkpoint":    existing["checkpoint"],
        "smoke_test":    existing["smoke_test"],
        "split":         existing["split"],
        "results":       summary,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary_meta, fh, indent=2)

    print(f"\nSaved: {SUMMARY_PATH}")
    print("Overall:")
    for m, v in overall_metrics.items():
        print(f"  {m}: AUC={v['roc_auc']:.4f}  d={v['cohens_d']:.4f}")


if __name__ == "__main__":
    main()
