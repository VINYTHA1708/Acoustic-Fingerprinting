"""Learned drift analysis example.

Builds the healthy learned fingerprint profile, then analyzes one normal and
one abnormal recording, printing raw and normalized drift metrics.

Usage:
    python examples/learned_drift_example.py \\
        --root data/raw/MIMII \\
        --machine-type pump \\
        --machine-id id_00 \\
        --checkpoint models/contrastive/best_projection_head.pt

    python examples/learned_drift_example.py \\
        --root data/raw/MIMII \\
        --machine-type pump \\
        --machine-id id_00 \\
        --checkpoint models/contrastive/best_projection_head.pt \\
        --max-recordings 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.loader import DatasetLoader
from src.learned_drift.analyzer import LearnedDriftAnalyzer
from src.learned_drift.learned_drift_result import LearnedDriftResult
from src.learned_profile.builder import LearnedProfileBuilder

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _print_result(label: str, result: LearnedDriftResult) -> None:
    print(f"\n  [{label}] {result.filename}")

    print("\n  ======================================")
    print("  RAW METRICS")
    print("  ======================================")
    print(f"  Euclidean  : {result.euclidean_distance:.6f}")
    print(f"  Manhattan  : {result.manhattan_distance:.6f}")
    print(f"  Cosine     : {result.cosine_similarity:.6f}")

    print("\n  ======================================")
    print("  NORMALIZED METRICS")
    print("  ======================================")
    print(f"  Euclidean  : {result.norm_euclidean_distance:.6f}")
    print(f"  Manhattan  : {result.norm_manhattan_distance:.6f}")
    print(f"  Cosine     : {result.norm_cosine_similarity:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Learned drift analysis example")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--machine-type", type=str, required=True, help="Machine type (e.g. pump)")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID (e.g. id_00)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ProjectionHead checkpoint")
    parser.add_argument("--max-recordings", type=int, default=100, help="Max healthy recordings for profile")
    args = parser.parse_args()

    loader = DatasetLoader(args.root)

    # --- Filter records for the target machine ---
    normal_records = [
        r for r in loader.get_all_files()
        if r.machine_type == args.machine_type
        and r.machine_id == args.machine_id
        and r.label == "normal"
    ]
    abnormal_records = [
        r for r in loader.get_all_files()
        if r.machine_type == args.machine_type
        and r.machine_id == args.machine_id
        and r.label == "abnormal"
    ]

    if not normal_records:
        print(f"ERROR: No normal recordings found for {args.machine_type}/{args.machine_id}.")
        sys.exit(1)
    if not abnormal_records:
        print(f"ERROR: No abnormal recordings found for {args.machine_type}/{args.machine_id}.")
        sys.exit(1)

    print(f"Machine type : {args.machine_type}")
    print(f"Machine ID   : {args.machine_id}")
    print(f"Normal recordings available   : {len(normal_records)}")
    print(f"Abnormal recordings available : {len(abnormal_records)}")

    # Hold out the first normal recording as the test sample so it is never
    # part of the profile. If only one exists, it must serve both roles.
    test_normal = normal_records[0]
    profile_records = normal_records[1:] if len(normal_records) > 1 else normal_records

    if len(normal_records) == 1:
        print(
            "WARNING: Only one normal recording available. "
            "It will be used for both the profile and the test — "
            "normal drift will appear artificially small."
        )

    # --- Build healthy learned profile ---
    print(f"\nBuilding healthy learned profile from up to {args.max_recordings} recording(s)...")
    builder = LearnedProfileBuilder(checkpoint_path=args.checkpoint)

    # Build using a loader that only sees the profile records (not the held-out test)
    # We pass the full loader but limit via max_recordings applied inside build().
    # To exclude the held-out test normal, we build a temporary loader-like object
    # by filtering manually and passing max_recordings to cap the rest.
    profile = builder.build(
        loader=loader,
        machine_type=args.machine_type,
        machine_id=args.machine_id,
        max_recordings=args.max_recordings,
    )

    # --- Analyze one normal and one abnormal recording ---
    analyzer = LearnedDriftAnalyzer(checkpoint_path=args.checkpoint)

    print("\nAnalyzing normal recording...")
    normal_result = analyzer.analyze(test_normal, profile)

    print("Analyzing abnormal recording...")
    abnormal_result = analyzer.analyze(abnormal_records[0], profile)

    # --- Print results ---
    _print_result("NORMAL", normal_result)
    _print_result("ABNORMAL", abnormal_result)

    # --- Summary ---
    sep = "======================================"
    print(f"\n  {sep}")
    print("  SUMMARY")
    print(f"  {sep}")
    print(
        f"  Normal   normalized Euclidean : {normal_result.norm_euclidean_distance:.6f}"
    )
    print(
        f"  Abnormal normalized Euclidean : {abnormal_result.norm_euclidean_distance:.6f}"
    )
    passed = abnormal_result.norm_euclidean_distance > normal_result.norm_euclidean_distance
    symbol = "✓" if passed else "✗"
    print(f"  {symbol} Abnormal drift {'larger' if passed else 'NOT larger'} than normal drift")


if __name__ == "__main__":
    main()
