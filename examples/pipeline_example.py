"""End-to-end inference pipeline example.

Builds a healthy learned fingerprint profile, holds out the first normal
recording for inference, then runs InferencePipeline.analyze() and prints
a formatted machine health report.

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

Expected dimensions:
    DSP Dimension       : 153
    BEATs Dimension     : 768
    Fusion Dimension    : 921
    Embedding Dimension : 256
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.loader import DatasetLoader
from src.learned_profile.builder import LearnedProfileBuilder
from src.pipeline.pipeline import InferencePipeline
from src.pipeline.result import PipelineResult

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _print_report(result: PipelineResult) -> None:
    sep = "=" * 40
    thin = "-" * 40

    print(sep)
    print("Machine Health Report")
    print(sep)
    print()
    print(f"Machine Type        : {result.machine_type}")
    print(f"Machine ID          : {result.machine_id}")
    print(f"Filename            : {result.filename}")
    print()
    print(f"DSP Dimension       : {result.dsp_dimension}")
    print(f"BEATs Dimension     : {result.beats_dimension}")
    print(f"Fusion Dimension    : {result.fusion_dimension}")
    print(f"Embedding Dimension : {result.embedding_dimension}")
    print()
    print(thin)
    print("Raw Drift")
    print(thin)
    print()
    print(f"Euclidean           : {result.raw_euclidean:.6f}")
    print(f"Manhattan           : {result.raw_manhattan:.6f}")
    print(f"Cosine              : {result.raw_cosine:.6f}")
    print()
    print(thin)
    print("Normalized Drift")
    print(thin)
    print()
    print(f"Euclidean           : {result.normalized_euclidean:.6f}")
    print(f"Manhattan           : {result.normalized_manhattan:.6f}")
    print(f"Cosine              : {result.normalized_cosine:.6f}")
    print()
    print(thin)
    print("Health")
    print(thin)
    print()
    print(f"Score               : {result.health_score:.2f}")
    print(f"Percentage          : {result.health_percentage}")
    print(f"State               : {result.health_state}")
    print()
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end inference pipeline example")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--machine-type", type=str, required=True, help="Machine type (e.g. pump)")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID (e.g. id_00)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ProjectionHead checkpoint")
    parser.add_argument(
        "--max-recordings", type=int, default=100,
        help="Max healthy recordings used to build the profile (default: 100)",
    )
    args = parser.parse_args()

    loader = DatasetLoader(args.root)

    normal_records = [
        r for r in loader.get_all_files()
        if r.machine_type == args.machine_type
        and r.machine_id == args.machine_id
        and r.label == "normal"
    ]

    if not normal_records:
        print(
            f"ERROR: No normal recordings found for "
            f"{args.machine_type}/{args.machine_id}."
        )
        sys.exit(1)

    # Hold out the first normal recording for inference so it is never
    # included in the profile.
    inference_record = normal_records[0]

    if len(normal_records) == 1:
        print(
            "WARNING: Only one normal recording available. "
            "It will be used for both the profile and inference — "
            "drift will appear artificially small."
        )

    print(f"Machine type        : {args.machine_type}")
    print(f"Machine ID          : {args.machine_id}")
    print(f"Normal recordings   : {len(normal_records)}")
    print(f"Inference recording : {inference_record.filename}")
    print()

    # --- Build healthy learned profile (excluding the held-out recording) ---
    print(f"Building healthy learned profile (up to {args.max_recordings} recordings)...")
    builder = LearnedProfileBuilder(checkpoint_path=args.checkpoint)
    profile = builder.build(
        loader=loader,
        machine_type=args.machine_type,
        machine_id=args.machine_id,
        max_recordings=args.max_recordings,
        exclude_filenames={inference_record.filename},
    )
    print(f"Profile built — {len(profile.embeddings)} embeddings, dim={profile.embedding_dimension}")
    print()

    # --- Run inference pipeline ---
    print("Running inference pipeline...")
    pipeline = InferencePipeline(checkpoint_path=args.checkpoint)
    result = pipeline.analyze(inference_record, profile)

    # --- Print formatted report ---
    print()
    _print_report(result)


if __name__ == "__main__":
    main()
