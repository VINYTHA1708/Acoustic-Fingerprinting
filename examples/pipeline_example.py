"""Machine health pipeline example.

Builds the healthy learned fingerprint profile, then evaluates up to 50 normal
and 50 abnormal recordings using MachineHealthPipeline and reports mean health
scores for each group.

Usage:
    python examples/pipeline_example.py \\
        --root data/raw/MIMII \\
        --machine-type pump \\
        --machine-id id_00 \\
        --checkpoint models/contrastive/best_projection_head.pt

    python examples/pipeline_example.py \\
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
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
from src.learned_profile.builder import LearnedProfileBuilder
from src.pipeline.pipeline import MachineHealthPipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

_SEP = "===================================="
_EVAL_LIMIT = 50


def main() -> None:
    parser = argparse.ArgumentParser(description="Machine health pipeline example")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--machine-type", type=str, required=True, help="Machine type (e.g. pump)")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID (e.g. id_00)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ProjectionHead checkpoint")
    parser.add_argument("--max-recordings", type=int, default=100, help="Max healthy recordings for profile")
    args = parser.parse_args()

    loader = DatasetLoader(args.root)

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

    print(f"\nBuilding healthy learned profile from up to {args.max_recordings} recording(s)...")
    builder = LearnedProfileBuilder(checkpoint_path=args.checkpoint)
    profile = builder.build(
        loader=loader,
        machine_type=args.machine_type,
        machine_id=args.machine_id,
        max_recordings=args.max_recordings,
    )

    drift_analyzer = LearnedDriftAnalyzer(checkpoint_path=args.checkpoint)
    health_analyzer = LearnedHealthAnalyzer(checkpoint_path=args.checkpoint)
    pipeline = MachineHealthPipeline(
        profile=profile,
        drift_analyzer=drift_analyzer,
        health_analyzer=health_analyzer,
    )

    eval_normal = normal_records[:_EVAL_LIMIT]
    eval_abnormal = abnormal_records[:_EVAL_LIMIT]

    print(f"\nEvaluating {len(eval_normal)} normal recording(s)...")
    normal_scores: list[float] = []
    for rec in eval_normal:
        report = pipeline.analyze(rec)
        normal_scores.append(report.health_score)

    print(f"Evaluating {len(eval_abnormal)} abnormal recording(s)...")
    abnormal_scores: list[float] = []
    for rec in eval_abnormal:
        report = pipeline.analyze(rec)
        abnormal_scores.append(report.health_score)

    mean_normal = sum(normal_scores) / len(normal_scores)
    mean_abnormal = sum(abnormal_scores) / len(abnormal_scores)

    print(f"\n  {_SEP}")
    print("  NORMAL (AVERAGE)")
    print(f"  {_SEP}")
    print(f"  Mean Health Score      : {mean_normal:.2f}")
    print(f"  Mean Health Percentage : {mean_normal:.1f}%")

    print(f"\n  {_SEP}")
    print("  ABNORMAL (AVERAGE)")
    print(f"  {_SEP}")
    print(f"  Mean Health Score      : {mean_abnormal:.2f}")
    print(f"  Mean Health Percentage : {mean_abnormal:.1f}%")

    print(f"\n  {_SEP}")
    print("  SUMMARY")
    print(f"  {_SEP}")
    passed = mean_normal > mean_abnormal
    symbol = "✓" if passed else "✗"
    msg = "Normal average health higher than abnormal" if passed else "Unexpected result"
    print(f"  {symbol} {msg}")


if __name__ == "__main__":
    main()
