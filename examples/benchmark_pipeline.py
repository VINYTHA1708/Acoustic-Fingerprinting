"""Pipeline benchmark example.

Builds a healthy learned fingerprint profile, selects one normal recording,
runs PipelineBenchmark, and prints a formatted per-stage timing report.

Usage:
    python examples/benchmark_pipeline.py \\
        --root data/raw/MIMII \\
        --machine-type pump \\
        --machine-id id_00 \\
        --checkpoint models/contrastive/best_projection_head.pt

    python examples/benchmark_pipeline.py \\
        --root data/raw/MIMII \\
        --machine-type pump \\
        --machine-id id_00 \\
        --checkpoint models/contrastive/best_projection_head.pt \\
        --max-recordings 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.benchmark import PipelineBenchmark
from src.benchmark.benchmark_result import BenchmarkResult
from src.dataset.loader import DatasetLoader
from src.learned_profile.builder import LearnedProfileBuilder

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _print_report(r: BenchmarkResult) -> None:
    sep = "=" * 36
    thin = "-" * 36

    ms = lambda s: f"{s * 1000:.3f} ms"  # noqa: E731

    print(sep)
    print("Pipeline Benchmark")
    print(sep)
    print(f"Machine Type        : {r.machine_type}")
    print(f"Machine ID          : {r.machine_id}")
    print(f"Filename            : {r.filename}")
    print(thin)
    print("Stage Times")
    print(thin)
    print(f"Preprocessing       : {ms(r.preprocessing_time)}")
    print(f"DSP Extraction      : {ms(r.dsp_time)}")
    print(f"BEATs               : {ms(r.beats_time)}")
    print(f"Fusion              : {ms(r.fusion_time)}")
    print(f"Projection          : {ms(r.projection_time)}")
    print(f"Drift               : {ms(r.drift_time)}")
    print(f"Health              : {ms(r.health_time)}")
    print(thin)
    print("Dimensions")
    print(thin)
    print(f"DSP                 : {r.dsp_dimension}")
    print(f"BEATs               : {r.beats_dimension}")
    print(f"Fusion              : {r.fusion_dimension}")
    print(f"Embedding           : {r.embedding_dimension}")
    print(thin)
    print("Cache")
    print(thin)
    print(f"Cache Hit           : {r.cache_hit}")
    print(thin)
    print("Total")
    print(thin)
    print(f"Total Time          : {ms(r.total_time)}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline benchmark example")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--machine-type", type=str, required=True, help="Machine type (e.g. pump)")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID (e.g. id_00)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ProjectionHead checkpoint")
    parser.add_argument(
        "--max-recordings", type=int, default=50,
        help="Max healthy recordings used to build the profile (default: 50)",
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

    # Hold out the first normal recording for benchmarking.
    benchmark_record = normal_records[0]

    print(f"Machine type        : {args.machine_type}")
    print(f"Machine ID          : {args.machine_id}")
    print(f"Normal recordings   : {len(normal_records)}")
    print(f"Benchmark recording : {benchmark_record.filename}")
    print()

    # --- Build healthy learned profile (excluding the benchmark recording) ---
    print(f"Building healthy learned profile (up to {args.max_recordings} recordings)...")
    builder = LearnedProfileBuilder(checkpoint_path=args.checkpoint)
    profile = builder.build(
        loader=loader,
        machine_type=args.machine_type,
        machine_id=args.machine_id,
        max_recordings=args.max_recordings,
        exclude_filenames={benchmark_record.filename},
    )
    print(f"Profile built — {len(profile.embeddings)} embeddings, dim={profile.embedding_dimension}")
    print()

    # --- Run benchmark ---
    print("Running benchmark...")
    bench = PipelineBenchmark(checkpoint_path=args.checkpoint)
    result = bench.benchmark(benchmark_record, profile)

    print()
    _print_report(result)


if __name__ == "__main__":
    main()
