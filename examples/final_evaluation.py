"""Final evaluation across all machine IDs for a given machine type.

For each machine ID:
    1. Build a healthy learned profile.
    2. Evaluate up to 50 normal and 50 abnormal recordings.
    3. Compute average drift, health, and inference time.
    4. Compute separation ratio and PASS/FAIL.

Prints a per-machine-ID table followed by overall results.

Usage:
    python examples/final_evaluation.py \\
        --root data/raw/MIMII \\
        --machine-type pump \\
        --checkpoint models/contrastive/best_projection_head.pt

    python examples/final_evaluation.py \\
        --root data/raw/MIMII \\
        --machine-type pump \\
        --checkpoint models/contrastive/best_projection_head.pt \\
        --max-recordings 100
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.loader import DatasetLoader
from src.learned_drift.analyzer import LearnedDriftAnalyzer
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
from src.learned_profile.builder import LearnedProfileBuilder
from src.pipeline.pipeline import MachineHealthPipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

_EVAL_LIMIT = 50
_SEP = "========================================"


@dataclass
class _MachineResult:
    machine_id: str
    mean_normal_raw_euclid: float
    mean_abnormal_raw_euclid: float
    mean_normal_norm_euclid: float
    mean_abnormal_norm_euclid: float
    mean_normal_health: float
    mean_abnormal_health: float
    separation_ratio: float
    avg_inference_ms: float
    passed: bool


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _evaluate_machine(
    loader: DatasetLoader,
    machine_type: str,
    machine_id: str,
    checkpoint: str,
    max_recordings: int,
) -> _MachineResult | None:
    """Build profile and evaluate one machine ID. Returns None if data is missing."""
    normal_records = [
        r for r in loader.get_all_files()
        if r.machine_type == machine_type
        and r.machine_id == machine_id
        and r.label == "normal"
    ]
    abnormal_records = [
        r for r in loader.get_all_files()
        if r.machine_type == machine_type
        and r.machine_id == machine_id
        and r.label == "abnormal"
    ]

    if not normal_records or not abnormal_records:
        print(f"  WARNING: Skipping {machine_id} — missing normal or abnormal recordings.")
        return None

    builder = LearnedProfileBuilder(checkpoint_path=checkpoint)
    profile = builder.build(
        loader=loader,
        machine_type=machine_type,
        machine_id=machine_id,
        max_recordings=max_recordings,
    )

    drift_analyzer = LearnedDriftAnalyzer(checkpoint_path=checkpoint)
    health_analyzer = LearnedHealthAnalyzer(checkpoint_path=checkpoint)
    pipeline = MachineHealthPipeline(
        profile=profile,
        drift_analyzer=drift_analyzer,
        health_analyzer=health_analyzer,
    )

    eval_normal = normal_records[:_EVAL_LIMIT]
    eval_abnormal = abnormal_records[:_EVAL_LIMIT]

    normal_raw_euclid: list[float] = []
    normal_norm_euclid: list[float] = []
    normal_health: list[float] = []
    inference_times: list[float] = []

    for rec in eval_normal:
        t0 = time.perf_counter()
        report = pipeline.analyze(rec)
        inference_times.append(time.perf_counter() - t0)
        normal_raw_euclid.append(report.euclidean_distance)
        normal_norm_euclid.append(report.normalized_euclidean_distance)
        normal_health.append(report.health_score)

    abnormal_raw_euclid: list[float] = []
    abnormal_norm_euclid: list[float] = []
    abnormal_health: list[float] = []

    for rec in eval_abnormal:
        t0 = time.perf_counter()
        report = pipeline.analyze(rec)
        inference_times.append(time.perf_counter() - t0)
        abnormal_raw_euclid.append(report.euclidean_distance)
        abnormal_norm_euclid.append(report.normalized_euclidean_distance)
        abnormal_health.append(report.health_score)

    mean_normal_norm = _mean(normal_norm_euclid)
    mean_abnormal_norm = _mean(abnormal_norm_euclid)
    separation_ratio = (
        mean_abnormal_norm / mean_normal_norm if mean_normal_norm > 0 else 0.0
    )

    passed = (
        mean_abnormal_norm > mean_normal_norm
        and _mean(normal_health) > _mean(abnormal_health)
    )

    return _MachineResult(
        machine_id=machine_id,
        mean_normal_raw_euclid=_mean(normal_raw_euclid),
        mean_abnormal_raw_euclid=_mean(abnormal_raw_euclid),
        mean_normal_norm_euclid=mean_normal_norm,
        mean_abnormal_norm_euclid=mean_abnormal_norm,
        mean_normal_health=_mean(normal_health),
        mean_abnormal_health=_mean(abnormal_health),
        separation_ratio=separation_ratio,
        avg_inference_ms=_mean(inference_times) * 1000,
        passed=passed,
    )


def _print_machine_table(results: list[_MachineResult]) -> None:
    col = 14
    header = (
        f"  {'Machine ID':<10} "
        f"{'Norm Drift':>{col}} "
        f"{'Norm Drift':>{col}} "
        f"{'Health':>{col}} "
        f"{'Health':>{col}} "
        f"{'Sep Ratio':>{col}} "
        f"{'Inf (ms)':>{col}} "
        f"{'Result':<8}"
    )
    subheader = (
        f"  {'':10} "
        f"{'(Normal)':>{col}} "
        f"{'(Abnormal)':>{col}} "
        f"{'(Normal)':>{col}} "
        f"{'(Abnormal)':>{col}} "
        f"{'':>{col}} "
        f"{'':>{col}} "
        f"{'':8}"
    )
    divider = "  " + "-" * (len(header) - 2)

    print(divider)
    print(header)
    print(subheader)
    print(divider)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"  {r.machine_id:<10} "
            f"{r.mean_normal_norm_euclid:>{col}.4f} "
            f"{r.mean_abnormal_norm_euclid:>{col}.4f} "
            f"{r.mean_normal_health:>{col}.2f} "
            f"{r.mean_abnormal_health:>{col}.2f} "
            f"{r.separation_ratio:>{col}.4f} "
            f"{r.avg_inference_ms:>{col}.2f} "
            f"{status:<8}"
        )

    print(divider)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final evaluation across all machine IDs")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--machine-type", type=str, required=True, help="Machine type (e.g. pump)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ProjectionHead checkpoint")
    parser.add_argument("--max-recordings", type=int, default=100, help="Max healthy recordings for profile")
    args = parser.parse_args()

    loader = DatasetLoader(args.root)
    machine_ids = loader.get_machine_ids(machine_type=args.machine_type)

    if not machine_ids:
        print(f"ERROR: No machine IDs found for machine type '{args.machine_type}'.")
        sys.exit(1)

    print(f"Machine type  : {args.machine_type}")
    print(f"Machine IDs   : {machine_ids}")
    print(f"Eval limit    : {_EVAL_LIMIT} normal / {_EVAL_LIMIT} abnormal per machine ID")

    results: list[_MachineResult] = []
    for machine_id in machine_ids:
        print(f"\n--- Evaluating {machine_id} ---")
        result = _evaluate_machine(
            loader=loader,
            machine_type=args.machine_type,
            machine_id=machine_id,
            checkpoint=args.checkpoint,
            max_recordings=args.max_recordings,
        )
        if result is not None:
            results.append(result)

    if not results:
        print("ERROR: No machine IDs could be evaluated.")
        sys.exit(1)

    print(f"\n\n  {_SEP}")
    print(f"  EVALUATION RESULTS — {args.machine_type.upper()}")
    print(f"  {_SEP}")
    _print_machine_table(results)

    pass_count = sum(1 for r in results if r.passed)
    fail_count = len(results) - pass_count
    avg_sep = _mean([r.separation_ratio for r in results])
    avg_normal_health = _mean([r.mean_normal_health for r in results])
    avg_abnormal_health = _mean([r.mean_abnormal_health for r in results])
    avg_inference_ms = _mean([r.avg_inference_ms for r in results])
    overall_passed = fail_count == 0

    print(f"\n  {_SEP}")
    print("  OVERALL RESULTS")
    print(f"  {_SEP}")
    print(f"  Machine IDs evaluated    : {len(results)}")
    print(f"  PASS count               : {pass_count}")
    print(f"  FAIL count               : {fail_count}")
    print(f"  Average separation ratio : {avg_sep:.4f}")
    print(f"  Average normal health    : {avg_normal_health:.2f}")
    print(f"  Average abnormal health  : {avg_abnormal_health:.2f}")
    print(f"  Average inference time   : {avg_inference_ms:.2f} ms")
    symbol = "✓" if overall_passed else "✗"
    status = "PASS" if overall_passed else "FAIL"
    print(f"  Overall Status           : {symbol} {status}")
    print(f"  {_SEP}")


if __name__ == "__main__":
    main()
