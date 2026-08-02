"""Pipeline benchmark example.

Builds the healthy learned fingerprint profile, then evaluates up to
--max-recordings normal recordings through MachineHealthPipeline, measuring
the runtime of each stage externally using time.perf_counter().

Stages timed per recording:
    1. FusionCache retrieval   — cache.load_or_create()
    2. Drift analysis          — drift_analyzer.analyze()
    3. Health analysis         — health_analyzer.analyze()
    4. Total pipeline          — pipeline.analyze()

The sub-stage calls use the same FusionCache, so the fused vector is already
warm (on disk) by the time pipeline.analyze() runs — total time reflects the
realistic steady-state cost of the full pipeline.

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
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.loader import DatasetLoader
from src.learned_drift.analyzer import LearnedDriftAnalyzer
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
from src.learned_profile.builder import LearnedProfileBuilder
from src.pipeline.pipeline import MachineHealthPipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

_SEP = "========================================"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _stddev(values: list[float]) -> float:
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline benchmark")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--machine-type", type=str, required=True, help="Machine type (e.g. pump)")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID (e.g. id_00)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ProjectionHead checkpoint")
    parser.add_argument("--max-recordings", type=int, default=100, help="Max normal recordings to benchmark")
    args = parser.parse_args()

    loader = DatasetLoader(args.root)

    normal_records = [
        r for r in loader.get_all_files()
        if r.machine_type == args.machine_type
        and r.machine_id == args.machine_id
        and r.label == "normal"
    ]

    if not normal_records:
        print(f"ERROR: No normal recordings found for {args.machine_type}/{args.machine_id}.")
        sys.exit(1)

    records = normal_records[:args.max_recordings]

    print(f"Machine type        : {args.machine_type}")
    print(f"Machine ID          : {args.machine_id}")
    print(f"Recordings to bench : {len(records)}")

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

    # Access the shared FusionCache owned by the drift analyzer.
    cache = drift_analyzer._cache

    cache_times: list[float] = []
    drift_times: list[float] = []
    health_times: list[float] = []
    total_times: list[float] = []

    print(f"\nBenchmarking {len(records)} recording(s)...")
    for idx, rec in enumerate(records, start=1):
        # --- Stage 1: FusionCache retrieval ---
        t0 = time.perf_counter()
        cache.load_or_create(rec)
        cache_times.append(time.perf_counter() - t0)

        # --- Stage 2: Drift analysis ---
        t0 = time.perf_counter()
        drift_analyzer.analyze(rec, profile)
        drift_times.append(time.perf_counter() - t0)

        # --- Stage 3: Health analysis ---
        t0 = time.perf_counter()
        health_analyzer.analyze(rec, profile)
        health_times.append(time.perf_counter() - t0)

        # --- Stage 4: Total pipeline ---
        t0 = time.perf_counter()
        pipeline.analyze(rec)
        total_times.append(time.perf_counter() - t0)

        if idx % 10 == 0 or idx == len(records):
            print(f"  Processed {idx}/{len(records)}")

    n = len(total_times)
    mean_total = _mean(total_times)
    rps = 1.0 / mean_total if mean_total > 0 else 0.0

    print(f"\n  {_SEP}")
    print("  PIPELINE BENCHMARK")
    print(f"  {_SEP}")
    print(f"  Recordings evaluated          : {n}")
    print(f"  Average cache retrieval time  : {_mean(cache_times) * 1000:.2f} ms")
    print(f"  Average drift analysis time   : {_mean(drift_times) * 1000:.2f} ms")
    print(f"  Average health analysis time  : {_mean(health_times) * 1000:.2f} ms")
    print(f"  Average total pipeline time   : {mean_total * 1000:.2f} ms")
    print(f"  Minimum total time            : {min(total_times) * 1000:.2f} ms")
    print(f"  Maximum total time            : {max(total_times) * 1000:.2f} ms")
    print(f"  Standard deviation            : {_stddev(total_times) * 1000:.2f} ms")
    print(f"  Recordings per second         : {rps:.2f}")
    print(f"  {_SEP}")


if __name__ == "__main__":
    main()
