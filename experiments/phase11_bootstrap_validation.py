"""
Phase 11 — Clean Bootstrap Validation (Seed 42)

Loads ONLY experiments/results/phase11/seed_42/evaluation_results.csv
and computes a nonparametric percentile bootstrap 95% CI for ROC-AUC.

Score column : normalized_euclidean  (higher = more anomalous)
Label column : true_label            ("normal" -> 0, "abnormal" -> 1)

No Phase 9 data, no sentinel removal, no score flipping.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────

SOURCE_CSV  = Path("experiments/results/phase11/seed_42/evaluation_results.csv")
OUT_DIR     = Path("experiments/results/phase11/bootstrap_validation")
SCORE_COL   = "normalized_euclidean"
LABEL_COL   = "true_label"
N_BOOTSTRAP = 2000
RNG_SEED    = 42

# ── Load ──────────────────────────────────────────────────────────────────────

def _load() -> pd.DataFrame:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")
    df = pd.read_csv(SOURCE_CSV)
    for col in (SCORE_COL, LABEL_COL):
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
    if df[SCORE_COL].isna().any():
        raise ValueError(f"'{SCORE_COL}' contains NaN values.")
    if df[LABEL_COL].isna().any():
        raise ValueError(f"'{LABEL_COL}' contains NaN values.")
    invalid = set(df[LABEL_COL].unique()) - {"normal", "abnormal"}
    if invalid:
        raise ValueError(f"'{LABEL_COL}' contains unexpected values: {invalid}")
    return df


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _bootstrap(scores: np.ndarray, labels: np.ndarray) -> list[float]:
    rng  = np.random.default_rng(RNG_SEED)
    aucs: list[float] = []
    while len(aucs) < N_BOOTSTRAP:
        idx = rng.integers(0, len(scores), len(scores))
        s, l = scores[idx], labels[idx]
        if len(np.unique(l)) < 2:   # skip single-class samples
            continue
        aucs.append(float(roc_auc_score(l, s)))
    return aucs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _load()

    # Binary labels: "abnormal" -> 1, "normal" -> 0
    y_true = (df[LABEL_COL] == "abnormal").astype(int).values
    scores = df[SCORE_COL].values.astype(float)

    n_total    = len(df)
    n_normal   = int((df[LABEL_COL] == "normal").sum())
    n_abnormal = int((df[LABEL_COL] == "abnormal").sum())

    point_auc = float(roc_auc_score(y_true, scores))

    bootstrap_aucs = _bootstrap(scores, y_true)
    b_arr = np.array(bootstrap_aucs)
    b_mean = float(b_arr.mean())
    b_std  = float(b_arr.std(ddof=1))
    ci_lo, ci_hi = np.percentile(b_arr, [2.5, 97.5])

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results = {
        "source_file":             str(SOURCE_CSV),
        "score_column":            SCORE_COL,
        "label_column":            LABEL_COL,
        "total_samples":           n_total,
        "normal_samples":          n_normal,
        "abnormal_samples":        n_abnormal,
        "point_estimate_roc_auc":  round(point_auc, 6),
        "bootstrap_mean":          round(b_mean, 6),
        "bootstrap_std":           round(b_std, 6),
        "ci_lower_95":             round(float(ci_lo), 6),
        "ci_upper_95":             round(float(ci_hi), 6),
        "n_bootstrap":             N_BOOTSTRAP,
        "rng_seed":                RNG_SEED,
        "valid_bootstrap_samples": len(bootstrap_aucs),
    }

    json_path = OUT_DIR / "bootstrap_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "bootstrap_summary.csv"
    pd.DataFrame([results]).to_csv(csv_path, index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\nCLEAN PHASE 11 SEED 42 BOOTSTRAP VALIDATION")
    print(f"Source:                  {SOURCE_CSV}")
    print(f"Total samples:           {n_total}")
    print(f"Normal:                  {n_normal}")
    print(f"Abnormal:                {n_abnormal}")
    print(f"Point estimate ROC-AUC:  {point_auc:.6f}")
    print(f"Bootstrap mean:          {b_mean:.6f}")
    print(f"Bootstrap std:           {b_std:.6f}")
    print(f"95% CI:                  [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"Bootstrap iterations:    {len(bootstrap_aucs)}")
    print(f"Seed:                    {RNG_SEED}")
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
