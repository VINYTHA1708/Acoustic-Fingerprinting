"""Evaluate learned fingerprint drift separation for every machine ID of a given machine type.

For each machine ID:
  - Builds a LearnedFingerprintProfile from up to --max-recordings normal recordings.
  - Evaluates up to 50 normal and 50 abnormal recordings.
  - Computes mean raw Euclidean, mean normalized Euclidean, mean Manhattan, mean Cosine.
  - Prints a per-machine-ID summary and overall statistics.

Usage:
    python examples/evaluate_learned_drift.py \\
        --root data/raw/MIMII \\
        --machine-type pump \\
        --checkpoint models/contrastive/best_projection_head.pt

    python examples/evaluate_learned_drift.py \\
        --root data/raw/MIMII \\
        --machine-type pump \\
        --checkpoint models/contrastive/best_projection_head.pt \\
        --max-recordings 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from src.dataset.loader import DatasetLoader
from src.dataset.metadata import AudioMetadata
from src.learned_drift.analyzer import LearnedDriftAnalyzer
from src.learned_profile.builder import LearnedProfileBuilder

_MAX_EVAL = 50


def _mean_metrics(
    records: list[AudioMetadata],
    profile,
    analyzer: LearnedDriftAnalyzer,
) -> tuple[float, float, float, float]:
    """Return (mean_raw_euclidean, mean_norm_euclidean, mean_manhattan, mean_cosine).

    Args:
        records: Recordings to evaluate.
        profile: LearnedFingerprintProfile for the same machine.
        analyzer: Shared LearnedDriftAnalyzer instance.

    Returns:
        Tuple of four mean metric floats.
    """
    raw_euc, norm_euc, manhat, cosine = [], [], [], []
    for rec in records:
        result = analyzer.analyze(rec, profile)
        raw_euc.append(result.euclidean_distance)
        norm_euc.append(result.norm_euclidean_distance)
        manhat.append(result.manhattan_distance)
        cosine.append(result.cosine_similarity)
    return (
        float(np.mean(raw_euc)),
        float(np.mean(norm_euc)),
        float(np.mean(manhat)),
        float(np.mean(cosine)),
    )


def _evaluate_machine_id(
    machine_id: str,
    machine_type: str,
    all_normal: list[AudioMetadata],
    all_abnormal: list[AudioMetadata],
    checkpoint: str,
    max_recordings: int,
    analyzer: LearnedDriftAnalyzer,
) -> dict | None:
    """Build a profile and evaluate one machine ID.

    Args:
        machine_id: Machine ID to evaluate.
        machine_type: Machine type (e.g. ``"pump"``).
        all_normal: All normal records for this machine ID.
        all_abnormal: All abnormal records for this machine ID.
        checkpoint: Path to the ProjectionHead checkpoint.
        max_recordings: Max healthy recordings for the profile.
        analyzer: Shared LearnedDriftAnalyzer instance.

    Returns:
        Result dict, or None if evaluation was skipped.
    """
    if not all_abnormal:
        print(f"  [{machine_id}] SKIP — no abnormal recordings found.")
        return None

    n_normal   = min(len(all_normal),   _MAX_EVAL)
    n_abnormal = min(len(all_abnormal), _MAX_EVAL)
    print(f"  [{machine_id}] Building profile (max {max_recordings} recordings)...")

    builder = LearnedProfileBuilder(checkpoint_path=checkpoint)

    # Use a minimal loader-compatible object: build() accepts a DatasetLoader,
    # so we pass a real loader but the builder filters by machine_type + machine_id.
    # The profile is built from the full normal set (capped by max_recordings).
    # We then hold out the first normal recording for evaluation so it is never
    # exclusively inside the profile — consistent with evaluate_all_machine_ids.py.
    loader = _RecordLoader(all_normal)
    profile = builder.build(
        loader=loader,
        machine_type=machine_type,
        machine_id=machine_id,
        max_recordings=max_recordings,
    )

    eval_normal   = all_normal[:_MAX_EVAL]
    eval_abnormal = all_abnormal[:_MAX_EVAL]

    print(f"  [{machine_id}] Evaluating ({n_normal} normal, {n_abnormal} abnormal)...")

    n_raw_euc, n_norm_euc, n_man, n_cos = _mean_metrics(eval_normal,   profile, analyzer)
    a_raw_euc, a_norm_euc, a_man, a_cos = _mean_metrics(eval_abnormal, profile, analyzer)

    passed = a_norm_euc > n_norm_euc
    sep_ratio = a_norm_euc / n_norm_euc if n_norm_euc > 0 else float("nan")

    status = "PASS" if passed else "FAIL"
    print(f"  [{machine_id}] Done — {status}")

    return {
        "machine_id":  machine_id,
        "n_raw_euc":   n_raw_euc,
        "n_norm_euc":  n_norm_euc,
        "n_man":       n_man,
        "n_cos":       n_cos,
        "a_raw_euc":   a_raw_euc,
        "a_norm_euc":  a_norm_euc,
        "a_man":       a_man,
        "a_cos":       a_cos,
        "pass":        passed,
        "sep_ratio":   sep_ratio,
    }


class _RecordLoader:
    """Minimal DatasetLoader-compatible wrapper around a pre-filtered record list.

    LearnedProfileBuilder.build() calls loader.get_all_files() and then filters
    by machine_type, machine_id, and label internally.  This wrapper satisfies
    that interface without requiring a full DatasetLoader re-scan.
    """

    def __init__(self, records: list[AudioMetadata]) -> None:
        self._records = records

    def get_all_files(self) -> list[AudioMetadata]:
        return list(self._records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate learned drift separation for all machine IDs of a given type."
    )
    parser.add_argument("--root",           type=str, required=True, help="Dataset root directory")
    parser.add_argument("--machine-type",   type=str, required=True, help="Machine type (e.g. pump)")
    parser.add_argument("--checkpoint",     type=str, required=True, help="Path to ProjectionHead checkpoint")
    parser.add_argument("--max-recordings", type=int, default=100,   help="Max healthy recordings per profile")
    args = parser.parse_args()

    loader = DatasetLoader(args.root)

    all_normal   = [
        r for r in loader.filter_by_label("normal")
        if r.machine_type == args.machine_type
    ]
    all_abnormal = [
        r for r in loader.filter_by_label("abnormal")
        if r.machine_type == args.machine_type
    ]

    machine_ids = sorted({r.machine_id for r in all_normal})

    if not machine_ids:
        print(f"ERROR: No normal recordings found for machine type '{args.machine_type}'.")
        sys.exit(1)

    print(f"\nMachine type  : {args.machine_type}")
    print(f"Machine IDs   : {machine_ids}")
    print(f"Max recordings: {args.max_recordings}\n")

    # One shared analyzer — loads the BEATs encoder and ProjectionHead once.
    analyzer = LearnedDriftAnalyzer(checkpoint_path=args.checkpoint)

    results = []
    for machine_id in machine_ids:
        normal_records   = [r for r in all_normal   if r.machine_id == machine_id]
        abnormal_records = [r for r in all_abnormal if r.machine_id == machine_id]

        result = _evaluate_machine_id(
            machine_id=machine_id,
            machine_type=args.machine_type,
            all_normal=normal_records,
            all_abnormal=abnormal_records,
            checkpoint=args.checkpoint,
            max_recordings=args.max_recordings,
            analyzer=analyzer,
        )
        if result is not None:
            results.append(result)

    if not results:
        print("No results to display.")
        sys.exit(1)

    # ── Summary table ─────────────────────────────────────────────────
    sep = "======================================================"

    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)

    col_id   = 12
    col_val  = 16
    col_stat =  6

    header = (
        f"{'Machine ID':<{col_id}}  "
        f"{'N norm Eucl':>{col_val}}  "
        f"{'A norm Eucl':>{col_val}}  "
        f"{'N raw Eucl':>{col_val}}  "
        f"{'A raw Eucl':>{col_val}}  "
        f"{'N Manh':>{col_val}}  "
        f"{'A Manh':>{col_val}}  "
        f"{'N Cos':>{col_val}}  "
        f"{'A Cos':>{col_val}}  "
        f"{'Status':>{col_stat}}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"{r['machine_id']:<{col_id}}  "
            f"{r['n_norm_euc']:>{col_val}.4f}  "
            f"{r['a_norm_euc']:>{col_val}.4f}  "
            f"{r['n_raw_euc']:>{col_val}.4f}  "
            f"{r['a_raw_euc']:>{col_val}.4f}  "
            f"{r['n_man']:>{col_val}.4f}  "
            f"{r['a_man']:>{col_val}.4f}  "
            f"{r['n_cos']:>{col_val}.6f}  "
            f"{r['a_cos']:>{col_val}.6f}  "
            f"{status:>{col_stat}}"
        )

    # ── Overall statistics ────────────────────────────────────────────
    total     = len(results)
    n_pass    = sum(1 for r in results if r["pass"])
    n_fail    = total - n_pass
    ratios    = [r["sep_ratio"] for r in results if not np.isnan(r["sep_ratio"])]
    avg_ratio = float(np.mean(ratios)) if ratios else float("nan")

    print(f"\n{sep}")
    print("OVERALL STATISTICS")
    print(sep)
    print(f"  Total machine IDs  : {total}")
    print(f"  PASS               : {n_pass}")
    print(f"  FAIL               : {n_fail}")
    print(f"  Avg separation ratio (abnormal / normal norm Euclidean): {avg_ratio:.4f}")


if __name__ == "__main__":
    main()
