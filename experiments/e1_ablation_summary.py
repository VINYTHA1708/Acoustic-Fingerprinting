"""Experiment E1 — Ablation Study Summary.

Loads experiments/results/e1/ablation_study/ablation_results.csv,
validates its structure and metric ranges, computes per-configuration
summary statistics, and writes a summary CSV.

Inputs:
    experiments/results/e1/ablation_study/ablation_results.csv

Outputs:
    experiments/results/e1/ablation_study/ablation_summary.csv
        One row per ablation configuration with:
            ablation_id, ablation_name,
            mean_auroc, mean_separation_ratio,
            auroc_delta_vs_fm   (AUROC − FM_full_method mean; 0.0 for FM itself)

Usage:
    python experiments/e1_ablation_summary.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.e1_ablation_definition import ABLATIONS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DIR = Path("experiments/results/e1/ablation_study")

ABLATION_CSV = _DIR / "ablation_results.csv"
SUMMARY_CSV  = _DIR / "ablation_summary.csv"

FM_ID = "FM_full_method"

EXPECTED_ABLATION_COUNT  = 5
EXPECTED_MACHINE_COUNT   = 4
EXPECTED_ROW_COUNT       = EXPECTED_ABLATION_COUNT * EXPECTED_MACHINE_COUNT  # 20

SUMMARY_COLUMNS = [
    "ablation_id",
    "ablation_name",
    "mean_auroc",
    "mean_separation_ratio",
    "auroc_delta_vs_fm",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    """Read the ablation results CSV and return typed rows.

    Args:
        path: Path to the CSV file.

    Returns:
        List of dicts with typed values (int counts, float metrics).

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ablation results not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "ablation_id":       row["ablation_id"],
                "ablation_name":     row["ablation_name"],
                "machine_id":        row["machine_id"],
                "n_normal":          int(row["n_normal"]),
                "n_abnormal":        int(row["n_abnormal"]),
                "auroc":             float(row["auroc"]),
                "separation_ratio":  float(row["separation_ratio"]),
            })
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    """Write summary rows to *path* as a CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_rows(rows: list[dict]) -> None:
    """Validate structure and metric ranges of the loaded rows.

    Checks:
        - Exactly EXPECTED_ROW_COUNT rows.
        - Exactly EXPECTED_ABLATION_COUNT distinct ablation IDs.
        - Each ablation ID covers exactly EXPECTED_MACHINE_COUNT machine IDs.
        - All AUROC values are in [0, 1].
        - All separation ratios are non-negative.
        - n_normal and n_abnormal are consistent across configurations for
          each machine ID (same test set used by all five configurations).

    Raises:
        ValueError: On any structural or metric violation.
    """
    if len(rows) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ROW_COUNT} rows, got {len(rows)}"
        )

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["ablation_id"], []).append(row)

    if len(groups) != EXPECTED_ABLATION_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ABLATION_COUNT} ablation IDs, "
            f"got {len(groups)}: {sorted(groups)}"
        )

    for aid, arows in groups.items():
        if len(arows) != EXPECTED_MACHINE_COUNT:
            raise ValueError(
                f"Ablation '{aid}' has {len(arows)} rows, "
                f"expected {EXPECTED_MACHINE_COUNT}"
            )

    for row in rows:
        auroc = row["auroc"]
        sep   = row["separation_ratio"]
        if not (0.0 <= auroc <= 1.0):
            raise ValueError(
                f"AUROC={auroc} out of [0,1] for "
                f"{row['ablation_id']}/{row['machine_id']}"
            )
        if sep < 0.0:
            raise ValueError(
                f"separation_ratio={sep} < 0 for "
                f"{row['ablation_id']}/{row['machine_id']}"
            )

    # Sample counts must be identical across configurations for each machine ID
    counts_by_machine: dict[str, tuple[int, int]] = {}
    for row in rows:
        mid = row["machine_id"]
        counts = (row["n_normal"], row["n_abnormal"])
        if mid in counts_by_machine:
            if counts_by_machine[mid] != counts:
                raise ValueError(
                    f"Inconsistent sample counts for machine '{mid}': "
                    f"saw {counts_by_machine[mid]} and {counts}"
                )
        else:
            counts_by_machine[mid] = counts


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def build_summary(rows: list[dict]) -> list[dict]:
    """Compute per-ablation summary statistics.

    For each ablation configuration:
        - mean_auroc: arithmetic mean of AUROC across all machine IDs.
        - mean_separation_ratio: arithmetic mean of separation ratio.
        - auroc_delta_vs_fm: mean_auroc − FM mean_auroc.
                             0.0 for FM_full_method itself.

    Args:
        rows: Typed rows from load_csv.

    Returns:
        List of summary dicts in the order ablations first appear in *rows*.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["ablation_id"], []).append(row)

    fm_mean_auroc: float | None = None
    summaries: list[dict] = []

    for aid, arows in groups.items():
        aurocs = [r["auroc"] for r in arows]
        seps   = [r["separation_ratio"] for r in arows]
        mean_auroc = round(float(np.mean(aurocs)), 4)
        mean_sep   = round(float(np.mean(seps)),   4)

        if aid == FM_ID:
            fm_mean_auroc = mean_auroc

        summaries.append({
            "ablation_id":          aid,
            "ablation_name":        arows[0]["ablation_name"],
            "mean_auroc":           mean_auroc,
            "mean_separation_ratio": mean_sep,
            "_aurocs":              aurocs,   # temporary; removed below
        })

    # Second pass: fill auroc_delta_vs_fm now that FM mean is known
    result = []
    for s in summaries:
        aurocs = s.pop("_aurocs")
        delta = round(s["mean_auroc"] - fm_mean_auroc, 4) if fm_mean_auroc is not None else 0.0
        result.append({**s, "auroc_delta_vs_fm": delta})

    return result


def build_per_machine_delta(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Return AUROC delta vs FM_full_method per ablation per machine ID.

    Args:
        rows: Typed rows from load_csv.

    Returns:
        Nested dict: {ablation_id: {machine_id: delta}}.
        FM_full_method maps to all-zero deltas.
    """
    fm_auroc: dict[str, float] = {
        r["machine_id"]: r["auroc"]
        for r in rows
        if r["ablation_id"] == FM_ID
    }

    result: dict[str, dict[str, float]] = {}
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["ablation_id"], []).append(row)

    for aid, arows in groups.items():
        result[aid] = {
            r["machine_id"]: round(r["auroc"] - fm_auroc.get(r["machine_id"], 0.0), 4)
            for r in arows
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows = load_csv(ABLATION_CSV)
    validate_rows(rows)
    summary = build_summary(rows)
    per_machine_delta = build_per_machine_delta(rows)

    save_csv(summary, SUMMARY_CSV)

    machine_ids = sorted({r["machine_id"] for r in rows})

    print("=" * 72)
    print("Experiment E1 — Ablation Study Summary")
    print("=" * 72)
    print()

    # Per-machine AUROC table
    col_w = 26
    print(f"{'Ablation':<{col_w}}", end="")
    for mid in machine_ids:
        print(f"  {mid:>8}", end="")
    print(f"  {'Mean':>8}  {'dFM':>8}  {'MeanSep':>8}")
    print("-" * 72)

    for s in summary:
        aid = s["ablation_id"]
        print(f"{aid:<{col_w}}", end="")
        for mid in machine_ids:
            auroc = next(r["auroc"] for r in rows if r["ablation_id"] == aid and r["machine_id"] == mid)
            print(f"  {auroc:>8.4f}", end="")
        delta_str = f"{s['auroc_delta_vs_fm']:>+.4f}" if aid != FM_ID else "  0.0000"
        print(f"  {s['mean_auroc']:>8.4f}  {delta_str:>8}  {s['mean_separation_ratio']:>8.4f}")

    print()
    print("=" * 72)
    print("Per-machine AUROC delta vs FM_full_method")
    print("=" * 72)
    print(f"{'Ablation':<{col_w}}", end="")
    for mid in machine_ids:
        print(f"  {mid:>8}", end="")
    print(f"  {'Mean Δ':>8}")
    print("-" * 72)

    for s in summary:
        aid = s["ablation_id"]
        if aid == FM_ID:
            continue
        print(f"{aid:<{col_w}}", end="")
        for mid in machine_ids:
            d = per_machine_delta[aid].get(mid, float("nan"))
            print(f"  {d:>+8.4f}", end="")
        print(f"  {s['auroc_delta_vs_fm']:>+8.4f}")

    print()
    print(f"Summary saved to: {SUMMARY_CSV}")
    print()

    # Highlight A2 vs FM
    fm_row  = next(s for s in summary if s["ablation_id"] == FM_ID)
    a2_row  = next((s for s in summary if s["ablation_id"] == "A2_no_dsp"), None)
    if a2_row:
        print("Note — A2_no_dsp (BEATs-only) vs FM_full_method:")
        print(f"  FM  mean AUROC = {fm_row['mean_auroc']:.4f}")
        print(f"  A2  mean AUROC = {a2_row['mean_auroc']:.4f}")
        print(f"  A2 − FM        = {a2_row['auroc_delta_vs_fm']:+.4f}")
        print()


if __name__ == "__main__":
    main()
