"""Explainability example.

Builds a healthy learned fingerprint profile, runs InferencePipeline on one
held-out normal and one abnormal recording, then generates and prints a
rule-based explanation for each via ExplainabilityEngine.

Usage:
    python examples/explainability_example.py \\
        --root data/raw/MIMII \\
        --machine-type pump \\
        --machine-id id_00 \\
        --checkpoint models/contrastive/best_projection_head.pt

    python examples/explainability_example.py \\
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
from src.explainability import ExplainabilityEngine, ExplanationResult
from src.learned_drift.analyzer import LearnedDriftAnalyzer
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
from src.learned_profile.builder import LearnedProfileBuilder
from src.pipeline.pipeline import InferencePipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

_SEP = "===================================="


def _print_explanation(result: ExplanationResult) -> None:
    print(_SEP)
    print("Machine Explanation")
    print(_SEP)
    print()
    print(f"Machine Type    : {result.machine_type}")
    print(f"Machine ID      : {result.machine_id}")
    print(f"Filename        : {result.filename}")
    print()
    print(f"Health Score    : {result.health_score:.2f}")
    print(f"Health State    : {result.health_state}")
    print()
    print(f"Summary         : {result.summary}")
    if result.possible_causes:
        print("Possible Causes :")
        for cause in result.possible_causes:
            print(f"  - {cause}")
    else:
        print("Possible Causes : None")
    print(f"Recommendation  : {result.recommendation}")
    print()
    print(_SEP)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Explainability example")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--machine-type", type=str, required=True, help="Machine type (e.g. pump)")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID (e.g. id_00)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ProjectionHead checkpoint")
    parser.add_argument(
        "--max-recordings", type=int, default=100,
        help="Maximum number of healthy recordings used to build the profile (default: 100)",
    )
    args = parser.parse_args()

    loader = DatasetLoader(args.root)
    all_records = loader.get_all_files()

    normal_records = [
        r for r in all_records
        if r.machine_type == args.machine_type
        and r.machine_id == args.machine_id
        and r.label == "normal"
    ]
    abnormal_records = [
        r for r in all_records
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

    # Hold out the first normal recording for inference so it is never in the profile.
    inference_normal = normal_records[0]
    inference_abnormal = abnormal_records[0]

    print(f"Machine type        : {args.machine_type}")
    print(f"Machine ID          : {args.machine_id}")
    print(f"Normal recordings   : {len(normal_records)}")
    print(f"Abnormal recordings : {len(abnormal_records)}")
    print()

    # --- Build healthy learned profile ---
    print(f"Building healthy learned profile (up to {args.max_recordings} recordings)...")
    builder = LearnedProfileBuilder(checkpoint_path=args.checkpoint)
    profile = builder.build(
        loader=loader,
        machine_type=args.machine_type,
        machine_id=args.machine_id,
        max_recordings=args.max_recordings,
        exclude_filenames={inference_normal.filename},
    )
    print(f"Profile built — {len(profile.embeddings)} embeddings, dim={profile.embedding_dimension}")
    print()

    # --- Shared analyzers (reuse existing modules, no inference logic duplicated) ---
    pipeline = InferencePipeline(checkpoint_path=args.checkpoint)
    drift_analyzer = LearnedDriftAnalyzer(checkpoint_path=args.checkpoint)
    health_analyzer = LearnedHealthAnalyzer(checkpoint_path=args.checkpoint)
    engine = ExplainabilityEngine()

    for label, record in [("NORMAL", inference_normal), ("ABNORMAL", inference_abnormal)]:
        print(f"Analyzing {label} recording: {record.filename}")
        drift = drift_analyzer.analyze(record, profile)
        health = health_analyzer.analyze(record, profile)
        explanation = engine.explain(drift, health)
        _print_explanation(explanation)


if __name__ == "__main__":
    main()
