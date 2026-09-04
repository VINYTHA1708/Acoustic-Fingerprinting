"""Phase 9 — Step 8: Repair 8 zero-metric rows in evaluation_pump.csv.

Root cause: smoke-test profiles (1 embedding → std=0) produced zero normalized
metrics for exactly 8 rows.  This script recomputes only those rows using the
existing valid profiles and the existing contrastive checkpoint.

No retraining, no profile rebuilding, no fusion-cache modification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.loader import DatasetLoader
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
from src.learned_profile.serializer import LearnedProfileSerializer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR    = Path("experiments/results/phase9")
PROFILE_DIR    = RESULTS_DIR / "profiles"
CHECKPOINT     = Path("models/contrastive/phase9/best_projection_head.pt")
DATASET_ROOT   = Path("data/raw/MIMII")
CSV_IN         = RESULTS_DIR / "evaluation_pump.csv"
CSV_REPAIRED   = RESULTS_DIR / "evaluation_pump_repaired.csv"
REPORT_PATH    = RESULTS_DIR / "comparison_e1" / "step8_repair_report.txt"

AFFECTED = [
    ("pump", "id_00", "00000764.wav", "normal"),
    ("pump", "id_02", "00000095.wav", "normal"),
    ("pump", "id_04", "00000081.wav", "normal"),
    ("pump", "id_06", "00000048.wav", "normal"),
    ("pump", "id_00", "00000000.wav", "abnormal"),
    ("pump", "id_02", "00000000.wav", "abnormal"),
    ("pump", "id_04", "00000000.wav", "abnormal"),
    ("pump", "id_06", "00000000.wav", "abnormal"),
]

METRIC_COLS = [
    "health_score", "health_percentage", "health_state",
    "normalized_euclidean", "normalized_manhattan", "normalized_cosine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_profiles(serializer: LearnedProfileSerializer) -> dict:
    profiles = {}
    for mid in ["id_00", "id_02", "id_04", "id_06"]:
        stem = f"phase9_pump_{mid}_learned_profile"
        p = serializer.load_npz(PROFILE_DIR / f"{stem}.npz")
        profiles[("pump", mid)] = p
        n = len(p.embeddings)
        std_min = float(np.min(np.abs(p.std_vector)))
        print(f"  Loaded profile pump/{mid}: {n} embeddings, min|std|={std_min:.6f}")
    return profiles


def _find_record(all_recordings, machine_type, machine_id, filename):
    for r in all_recordings:
        if r.machine_type == machine_type and r.machine_id == machine_id and r.filename == filename:
            return r
    raise ValueError(f"Recording not found: {machine_type}/{machine_id}/{filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 9 — Step 8: Repair zero-metric rows")
    print("=" * 60)

    # 1. Load CSV
    df = pd.read_csv(CSV_IN)
    original_rows = len(df)
    print(f"\nLoaded {CSV_IN}: {original_rows} rows")

    # 2. Find the 8 affected rows
    mask = pd.Series([False] * len(df))
    for mt, mid, fn, lbl in AFFECTED:
        row_mask = (
            (df["machine_type"] == mt) &
            (df["machine_id"]   == mid) &
            (df["filename"]     == fn) &
            (df["true_label"]   == lbl)
        )
        mask |= row_mask

    affected_df = df[mask]
    n_found = len(affected_df)
    print(f"Found {n_found} affected rows (expected 8)")
    if n_found != 8:
        raise RuntimeError(f"Expected exactly 8 affected rows, found {n_found}. Aborting.")

    print("\nAffected rows (old metrics):")
    for _, row in affected_df.iterrows():
        print(
            f"  {row['machine_id']}/{row['filename']} [{row['true_label']}]"
            f"  norm_euclid={row['normalized_euclidean']}"
            f"  norm_manhat={row['normalized_manhattan']}"
            f"  norm_cosine={row['normalized_cosine']}"
        )

    # 3. Load valid profiles
    print("\nLoading valid profiles...")
    serializer = LearnedProfileSerializer()
    profiles = _load_profiles(serializer)

    # 4. Load recordings
    print("\nLoading dataset...")
    loader = DatasetLoader(DATASET_ROOT)
    all_recordings = loader.get_all_files()

    # 5. Recompute metrics for the 8 rows
    print("\nRecomputing metrics...")
    analyzer = LearnedHealthAnalyzer(checkpoint_path=CHECKPOINT)

    old_metrics: list[dict] = []
    new_metrics: list[dict] = []

    repaired_df = df.copy()

    for mt, mid, fn, lbl in AFFECTED:
        record = _find_record(all_recordings, mt, mid, fn)
        profile = profiles[(mt, mid)]

        # Capture old values
        row_idx = repaired_df.index[
            (repaired_df["machine_type"] == mt) &
            (repaired_df["machine_id"]   == mid) &
            (repaired_df["filename"]     == fn) &
            (repaired_df["true_label"]   == lbl)
        ][0]
        old_row = repaired_df.loc[row_idx]
        old_metrics.append({
            "key": f"{mid}/{fn} [{lbl}]",
            "health_score":         old_row["health_score"],
            "normalized_euclidean": old_row["normalized_euclidean"],
            "normalized_manhattan": old_row["normalized_manhattan"],
            "normalized_cosine":    old_row["normalized_cosine"],
        })

        # Recompute
        result = analyzer.analyze(record, profile)
        print(
            f"  {mid}/{fn} [{lbl}]"
            f"  score={result.health_score:.4f}"
            f"  norm_euclid={result.normalized_euclidean:.6f}"
            f"  norm_manhat={result.normalized_manhattan:.6f}"
            f"  norm_cosine={result.normalized_cosine:.6f}"
        )

        new_metrics.append({
            "key": f"{mid}/{fn} [{lbl}]",
            "health_score":         result.health_score,
            "normalized_euclidean": result.normalized_euclidean,
            "normalized_manhattan": result.normalized_manhattan,
            "normalized_cosine":    result.normalized_cosine,
        })

        # Patch the row
        repaired_df.at[row_idx, "health_score"]         = result.health_score
        repaired_df.at[row_idx, "health_percentage"]    = result.health_percentage
        repaired_df.at[row_idx, "health_state"]         = result.health_state
        repaired_df.at[row_idx, "normalized_euclidean"] = result.normalized_euclidean
        repaired_df.at[row_idx, "normalized_manhattan"] = result.normalized_manhattan
        repaired_df.at[row_idx, "normalized_cosine"]    = result.normalized_cosine

    # 6. Validate repaired CSV
    print("\nValidating repaired CSV...")
    assert len(repaired_df) == original_rows, "Row count changed!"

    changed = (repaired_df[METRIC_COLS] != df[METRIC_COLS]).any(axis=1).sum()
    assert changed == 8, f"Expected 8 changed rows, got {changed}"

    for col in ["normalized_euclidean", "normalized_manhattan"]:
        neg = (repaired_df[col] < 0).sum()
        assert neg == 0, f"Negative values in {col}: {neg}"

    for col in METRIC_COLS[:4]:  # numeric cols
        nan_count = repaired_df[col].isna().sum()
        inf_count = np.isinf(pd.to_numeric(repaired_df[col], errors="coerce")).sum()
        assert nan_count == 0, f"NaN in {col}: {nan_count}"
        assert inf_count == 0, f"Inf in {col}: {inf_count}"

    print(f"  Row count unchanged: {len(repaired_df)}")
    print(f"  Rows modified: {changed}")
    print("  No NaN or Inf detected")
    print("  No negative Euclidean or Manhattan distances")
    print("  VALIDATION PASSED")

    # 7. Save repaired CSV
    repaired_df.to_csv(CSV_REPAIRED, index=False)
    print(f"\nSaved repaired CSV: {CSV_REPAIRED}")

    # 8. Overwrite production CSV
    repaired_df.to_csv(CSV_IN, index=False)
    print(f"Updated production CSV: {CSV_IN}")

    # 9. Write report
    _write_report(old_metrics, new_metrics, original_rows, changed)
    print(f"\nReport saved: {REPORT_PATH}")
    print("\nSTEP 8B COMPLETE")


def _write_report(old_metrics, new_metrics, total_rows, changed_rows) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 60,
        "Phase 9 — Step 8 Repair Report",
        "=" * 60,
        "",
        f"Total rows in evaluation_pump.csv : {total_rows}",
        f"Rows repaired                      : {changed_rows}",
        "",
        "Old vs New Metrics for 8 Repaired Rows",
        "-" * 60,
    ]
    for old, new in zip(old_metrics, new_metrics):
        lines += [
            f"\n  {old['key']}",
            f"    health_score         : {old['health_score']:.6f}  →  {new['health_score']:.6f}",
            f"    normalized_euclidean : {old['normalized_euclidean']:.6f}  →  {new['normalized_euclidean']:.6f}",
            f"    normalized_manhattan : {old['normalized_manhattan']:.6f}  →  {new['normalized_manhattan']:.6f}",
            f"    normalized_cosine    : {old['normalized_cosine']:.6f}  →  {new['normalized_cosine']:.6f}",
        ]
    lines += [
        "",
        "Validation Results",
        "-" * 60,
        "  Row count unchanged          : PASS",
        "  Exactly 8 rows modified      : PASS",
        "  No NaN or Inf                : PASS",
        "  No negative Euclidean/Manhat : PASS",
        "",
        "Smoke-Test Contamination Fix (Step 8D)",
        "-" * 60,
        "  phase9_evaluate.py: _evaluate_split() now skips writing to",
        "  the production CSV when --smoke-test is active.  Smoke-test",
        "  results are written to a separate file:",
        "    evaluation_{machine_type}_smoketest.csv",
        "  The production CSV is never opened in append mode during a",
        "  smoke-test run.",
        "",
        "Final Status",
        "-" * 60,
        "  STEP 8 PASSED",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
