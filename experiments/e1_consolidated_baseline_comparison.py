"""Experiment E1 — Consolidated Baseline Comparison.

Loads the per-method result CSVs produced by Phases 7.2 and 7.3, combines
them into a single comparison table, and computes per-method summary
statistics.

Inputs:
    experiments/results/e1/baseline_comparison/baseline_results.csv
    experiments/results/e1/baseline_comparison/main_method_results.csv

Outputs:
    experiments/results/e1/baseline_comparison/consolidated_comparison.csv
        One row per (method × machine_id).  Identical schema to the input
        files plus an ``auroc_delta_vs_best_baseline`` column that is
        populated only for the main method rows.

    experiments/results/e1/baseline_comparison/method_summary.csv
        One row per method with:
            mean_auroc, mean_separation_ratio,
            auroc_improvement_over_best_baseline   (main method only, else "")

Usage:
    python experiments/e1_consolidated_baseline_comparison.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DIR = Path("experiments/results/e1/baseline_comparison")

BASELINE_CSV   = _DIR / "baseline_results.csv"
MAIN_CSV       = _DIR / "main_method_results.csv"
CONSOLIDATED_CSV = _DIR / "consolidated_comparison.csv"
SUMMARY_CSV    = _DIR / "method_summary.csv"

MAIN_METHOD_ID = "contrastive_main"

CONSOLIDATED_COLUMNS = [
    "baseline_id",
    "baseline_name",
    "machine_id",
    "n_normal",
    "n_abnormal",
    "auroc",
    "separation_ratio",
    "auroc_delta_vs_best_baseline",   # filled only for main method rows
]

SUMMARY_COLUMNS = [
    "baseline_id",
    "baseline_name",
    "mean_auroc",
    "mean_separation_ratio",
    "auroc_improvement_over_best_baseline",  # main method only, else ""
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    """Read a CSV file and return rows as a list of dicts with typed values.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "baseline_id":    row["baseline_id"],
                "baseline_name":  row["baseline_name"],
                "machine_id":     row["machine_id"],
                "n_normal":       int(row["n_normal"]),
                "n_abnormal":     int(row["n_abnormal"]),
                "auroc":          float(row["auroc"]),
                "separation_ratio": float(row["separation_ratio"]),
            })
    return rows


def save_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    """Write *rows* to *path* as a CSV with the given column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def build_consolidated(all_rows: list[dict]) -> list[dict]:
    """Add ``auroc_delta_vs_best_baseline`` to every row.

    For baseline rows the field is an empty string.
    For main-method rows it is ``main_auroc − best_baseline_auroc`` per
    machine ID, rounded to 4 decimal places.

    Args:
        all_rows: Combined rows from both input CSVs (typed values).

    Returns:
        New list of dicts with the extra column appended.
    """
    # Index baseline AUROCs per machine_id → best (max) across all baselines
    best_baseline: dict[str, float] = {}
    for row in all_rows:
        if row["baseline_id"] == MAIN_METHOD_ID:
            continue
        mid = row["machine_id"]
        best_baseline[mid] = max(best_baseline.get(mid, 0.0), row["auroc"])

    consolidated = []
    for row in all_rows:
        mid = row["machine_id"]
        if row["baseline_id"] == MAIN_METHOD_ID and mid in best_baseline:
            delta = round(row["auroc"] - best_baseline[mid], 4)
        else:
            delta = ""
        consolidated.append({**row, "auroc_delta_vs_best_baseline": delta})
    return consolidated


def build_summary(all_rows: list[dict]) -> list[dict]:
    """Compute per-method summary statistics.

    For each unique ``baseline_id``:
        - ``mean_auroc``: arithmetic mean of AUROC across machine IDs
        - ``mean_separation_ratio``: arithmetic mean of separation ratio
        - ``auroc_improvement_over_best_baseline``: for the main method only,
          mean of (main_auroc − best_baseline_auroc) per machine ID; empty
          string for all other methods.

    Args:
        all_rows: Combined rows from both input CSVs (typed values).

    Returns:
        List of summary dicts, one per method, in the order they first appear.
    """
    # Group rows by baseline_id, preserving insertion order
    groups: dict[str, list[dict]] = {}
    for row in all_rows:
        groups.setdefault(row["baseline_id"], []).append(row)

    # Best baseline AUROC per machine_id
    best_baseline: dict[str, float] = {}
    for bid, rows in groups.items():
        if bid == MAIN_METHOD_ID:
            continue
        for row in rows:
            mid = row["machine_id"]
            best_baseline[mid] = max(best_baseline.get(mid, 0.0), row["auroc"])

    summary = []
    for bid, rows in groups.items():
        aurocs = [r["auroc"] for r in rows]
        seps   = [r["separation_ratio"] for r in rows]
        mean_auroc = round(float(np.mean(aurocs)), 4)
        mean_sep   = round(float(np.mean(seps)),   4)

        if bid == MAIN_METHOD_ID:
            deltas = [
                r["auroc"] - best_baseline[r["machine_id"]]
                for r in rows
                if r["machine_id"] in best_baseline
            ]
            improvement = round(float(np.mean(deltas)), 4) if deltas else ""
        else:
            improvement = ""

        summary.append({
            "baseline_id":   bid,
            "baseline_name": rows[0]["baseline_name"],
            "mean_auroc":    mean_auroc,
            "mean_separation_ratio": mean_sep,
            "auroc_improvement_over_best_baseline": improvement,
        })
    return summary


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inputs(baseline_rows: list[dict], main_rows: list[dict]) -> None:
    """Raise ValueError if the two input files are inconsistent.

    Checks:
        - Both files share the same set of machine IDs.
        - No method ID appears in both files.
        - All AUROC values are in [0, 1].
    """
    baseline_mids = {r["machine_id"] for r in baseline_rows}
    main_mids     = {r["machine_id"] for r in main_rows}
    if baseline_mids != main_mids:
        raise ValueError(
            f"Machine ID mismatch between input files: "
            f"baseline={sorted(baseline_mids)}  main={sorted(main_mids)}"
        )

    baseline_ids = {r["baseline_id"] for r in baseline_rows}
    if MAIN_METHOD_ID in baseline_ids:
        raise ValueError(
            f"Main method ID '{MAIN_METHOD_ID}' must not appear in baseline_results.csv"
        )

    for row in baseline_rows + main_rows:
        a = row["auroc"]
        if not (0.0 <= a <= 1.0):
            raise ValueError(
                f"AUROC={a} out of [0,1] for {row['baseline_id']}/{row['machine_id']}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    baseline_rows = load_csv(BASELINE_CSV)
    main_rows     = load_csv(MAIN_CSV)

    validate_inputs(baseline_rows, main_rows)

    all_rows = baseline_rows + main_rows

    consolidated = build_consolidated(all_rows)
    summary      = build_summary(all_rows)

    save_csv(consolidated, CONSOLIDATED_CSV, CONSOLIDATED_COLUMNS)
    save_csv(summary,      SUMMARY_CSV,      SUMMARY_COLUMNS)

    # -----------------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("Experiment E1 — Consolidated Baseline Comparison")
    print("=" * 72)
    print()
    print(f"{'Method':<32} {'Machine':<8} {'AUROC':>7} {'Sep':>7} {'dAUROC':>9}")
    print("-" * 72)

    prev_id = None
    for row in consolidated:
        if row["baseline_id"] != prev_id and prev_id is not None:
            print()
        prev_id = row["baseline_id"]
        delta_str = (
            f"{row['auroc_delta_vs_best_baseline']:>+.4f}"
            if row["auroc_delta_vs_best_baseline"] != ""
            else "        -"
        )
        print(
            f"{row['baseline_id']:<32} {row['machine_id']:<8} "
            f"{row['auroc']:>7.4f} {row['separation_ratio']:>7.4f} {delta_str:>9}"
        )

    print()
    print("=" * 72)
    print("Method Summary")
    print("=" * 72)
    print(f"{'Method':<32} {'Mean AUROC':>10} {'Mean Sep':>9} {'AUROC+':>9}")
    print("-" * 65)
    for row in summary:
        imp = (
            f"{row['auroc_improvement_over_best_baseline']:>+.4f}"
            if row["auroc_improvement_over_best_baseline"] != ""
            else "         -"
        )
        print(
            f"{row['baseline_id']:<32} {row['mean_auroc']:>10.4f} "
            f"{row['mean_separation_ratio']:>9.4f} {imp:>9}"
        )

    print()
    print("Saved:")
    print(f"  {CONSOLIDATED_CSV}")
    print(f"  {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
