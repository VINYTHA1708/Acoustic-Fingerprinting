"""Diagnostic: distance distribution of normal vs abnormal recordings against the healthy profile.

Processes up to 50 normal and 50 abnormal recordings for one machine and prints
distribution statistics for Euclidean, Manhattan, and Cosine metrics.
Identifies the five normal recordings with the largest Euclidean distance and
the five abnormal recordings with the smallest Euclidean distance.

Usage:
    python examples/profile_distance_distribution.py --root path/to/dataset
    python examples/profile_distance_distribution.py --root path/to/dataset --machine-type fan --machine-id id_00
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from dataset import DatasetLoader
from feature_extraction import FeatureExtractor, FeatureVectorBuilder
from fingerprint import FingerprintGenerator
from preprocessing import PreprocessingPipeline
from profile import HealthyProfileBuilder
from drift import DriftAnalyzer

_MAX_SAMPLES = 50


@dataclass
class RecordingResult:
    filename: str
    euclidean: float
    manhattan: float
    cosine: float


def build_fingerprint(record, pipeline, extractor, vec_builder, generator):
    """Run the full preprocessing + extraction + generation pipeline for one record."""
    result   = pipeline.run(record.absolute_path)
    features = extractor.extract(result["waveform"])
    vector, names = vec_builder.build(features)
    return generator.generate(
        features=(vector, names),
        metadata=record,
        sample_rate=result["sample_rate"],
    )


def analyze_batch(records, profile, pipeline, extractor, vec_builder, generator, analyzer, label):
    """Process a batch of records and return a list of RecordingResult.

    Args:
        records: List of AudioMetadata records to process.
        profile: HealthyFingerprintProfile to compare against.
        pipeline, extractor, vec_builder, generator, analyzer: shared pipeline objects.
        label: Display label used in progress output.

    Returns:
        List of RecordingResult, one per record.
    """
    results = []
    for i, record in enumerate(records):
        print(f"  Processing {label} {i + 1}/{len(records)}: {record.filename}", end="\r")
        fp     = build_fingerprint(record, pipeline, extractor, vec_builder, generator)
        drift  = analyzer.analyze(fp, profile)
        results.append(RecordingResult(
            filename=record.filename,
            euclidean=drift.euclidean_distance,
            manhattan=drift.manhattan_distance,
            cosine=drift.cosine_similarity,
        ))
    print()  # newline after progress
    return results


def print_stats(results: list[RecordingResult], label: str) -> None:
    """Print distribution statistics for a batch of results.

    Args:
        results: List of RecordingResult objects.
        label: Section header label.
    """
    euclid  = np.array([r.euclidean for r in results])
    manhat  = np.array([r.manhattan for r in results])
    cosine  = np.array([r.cosine    for r in results])

    print(f"\n{label} (n={len(results)})")
    print(f"  {'Metric':<22}  {'Mean':>10}  {'Median':>10}  {'Min':>10}  {'Max':>10}  {'Std':>10}")
    print(f"  {'-'*72}")

    for name, values in (("Euclidean", euclid), ("Manhattan", manhat), ("Cosine Similarity", cosine)):
        print(
            f"  {name:<22}  "
            f"{values.mean():>10.4f}  "
            f"{np.median(values):>10.4f}  "
            f"{values.min():>10.4f}  "
            f"{values.max():>10.4f}  "
            f"{values.std():>10.4f}"
        )


def print_outliers(results: list[RecordingResult], n: int, largest: bool, metric: str) -> None:
    """Print the n recordings with the largest or smallest Euclidean distance.

    Args:
        results: List of RecordingResult objects.
        n: Number of outliers to show.
        largest: If True, show largest distances; if False, show smallest.
        metric: Label for the metric column header.
    """
    values = np.array([r.euclidean for r in results])
    indices = np.argsort(values)[::-1][:n] if largest else np.argsort(values)[:n]
    direction = "largest" if largest else "smallest"

    print(f"\n  {n} recordings with {direction} Euclidean distance:")
    print(f"  {'Filename':<40}  {'Euclidean':>12}  {'Cosine':>10}")
    print(f"  {'-'*66}")
    for i in indices:
        r = results[i]
        print(f"  {r.filename:<40}  {r.euclidean:>12.4f}  {r.cosine:>10.6f}")


def main(root: str, machine_type: str | None, machine_id: str | None) -> None:
    loader = DatasetLoader(root)

    normal_records   = loader.filter_by_label("normal")
    abnormal_records = loader.filter_by_label("abnormal")

    if machine_type:
        normal_records   = [r for r in normal_records   if r.machine_type == machine_type]
        abnormal_records = [r for r in abnormal_records if r.machine_type == machine_type]
    if machine_id:
        normal_records   = [r for r in normal_records   if r.machine_id == machine_id]
        abnormal_records = [r for r in abnormal_records if r.machine_id == machine_id]

    if not normal_records:
        print("ERROR: No normal recordings found.")
        sys.exit(1)
    if not abnormal_records:
        print("ERROR: No abnormal recordings found.")
        sys.exit(1)

    # Lock to the first machine_type + machine_id that has both labels
    target_type = normal_records[0].machine_type
    target_id   = normal_records[0].machine_id
    normal_records   = [r for r in normal_records   if r.machine_type == target_type and r.machine_id == target_id]
    abnormal_records = [r for r in abnormal_records if r.machine_type == target_type and r.machine_id == target_id]

    if not abnormal_records:
        print(f"ERROR: No abnormal recordings for {target_type}/{target_id}.")
        sys.exit(1)

    print(f"\nMachine type : {target_type}")
    print(f"Machine ID   : {target_id}")
    print(f"Normal  available : {len(normal_records)}  (using up to {_MAX_SAMPLES})")
    print(f"Abnormal available: {len(abnormal_records)}  (using up to {_MAX_SAMPLES})")

    # ── Shared pipeline objects ───────────────────────────────────────
    pipeline    = PreprocessingPipeline(target_sr=16_000)
    extractor   = FeatureExtractor(sample_rate=16_000)
    vec_builder = FeatureVectorBuilder()
    generator   = FingerprintGenerator()
    analyzer    = DriftAnalyzer()

    # ── Build profile from held-out normal set ────────────────────────
    # Hold out the first normal recording so it can serve as a test sample.
    # The profile is built from normal_records[1:] (or all if only one exists).
    test_normal_record = normal_records[0]
    profile_records    = normal_records[1:] if len(normal_records) > 1 else normal_records

    if len(normal_records) == 1:
        print("WARNING: Only one normal recording — it is used for both profile and test.")

    print(f"\nBuilding healthy profile from {len(profile_records)} normal recording(s)...")
    profile_fps = [
        build_fingerprint(r, pipeline, extractor, vec_builder, generator)
        for r in profile_records
    ]
    profile = HealthyProfileBuilder().build(profile_fps)
    print(f"Profile built — {profile.number_of_samples} samples, {len(profile.feature_names)} features.")

    # ── Evaluate up to 50 normal recordings (including held-out test) ─
    # Re-insert the held-out record at position 0 for the evaluation batch.
    eval_normal   = ([test_normal_record] + profile_records)[:_MAX_SAMPLES]
    eval_abnormal = abnormal_records[:_MAX_SAMPLES]

    print(f"\nEvaluating {len(eval_normal)} normal recordings...")
    normal_results = analyze_batch(
        eval_normal, profile, pipeline, extractor, vec_builder, generator, analyzer, "normal"
    )

    print(f"Evaluating {len(eval_abnormal)} abnormal recordings...")
    abnormal_results = analyze_batch(
        eval_abnormal, profile, pipeline, extractor, vec_builder, generator, analyzer, "abnormal"
    )

    # ── Distribution statistics ───────────────────────────────────────
    print("\n" + "=" * 76)
    print("DISTANCE DISTRIBUTION STATISTICS")
    print("=" * 76)
    print_stats(normal_results,   "NORMAL   recordings vs healthy profile")
    print_stats(abnormal_results, "ABNORMAL recordings vs healthy profile")

    # ── Outlier inspection ────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("OUTLIER INSPECTION")
    print("=" * 76)
    print("\nNORMAL — recordings furthest from the profile (potential outliers):")
    print_outliers(normal_results,   n=5, largest=True,  metric="Euclidean")
    print("\nABNORMAL — recordings closest to the profile (hardest to detect):")
    print_outliers(abnormal_results, n=5, largest=False, metric="Euclidean")

    # ── Separation summary ────────────────────────────────────────────
    normal_euclid   = np.array([r.euclidean for r in normal_results])
    abnormal_euclid = np.array([r.euclidean for r in abnormal_results])
    overlap = np.sum(abnormal_euclid < normal_euclid.mean())

    print("\n" + "=" * 76)
    print("SEPARATION SUMMARY")
    print("=" * 76)
    print(f"  Normal   mean Euclidean : {normal_euclid.mean():.4f}")
    print(f"  Abnormal mean Euclidean : {abnormal_euclid.mean():.4f}")
    print(f"  Abnormal recordings below normal mean: {overlap}/{len(abnormal_results)}")
    if abnormal_euclid.mean() > normal_euclid.mean():
        print("  RESULT: Abnormal recordings are on average FURTHER from the profile. ✓")
    else:
        print("  RESULT: Abnormal recordings are on average CLOSER to the profile. ✗")
        print("          This may indicate a data distribution issue for this machine/ID.")
        print("          Try a different --machine-type or --machine-id.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile distance distribution diagnostic")
    parser.add_argument("--root", required=True, help="Dataset root directory.")
    parser.add_argument("--machine-type", default=None, help="Filter by machine type (optional).")
    parser.add_argument("--machine-id",   default=None, help="Filter by machine ID (optional).")
    args = parser.parse_args()
    main(args.root, args.machine_type, args.machine_id)
