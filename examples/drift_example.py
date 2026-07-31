"""Example: fingerprint drift analysis comparing normal vs abnormal recordings.

Usage:
    python examples/drift_example.py --root path/to/dataset
    python examples/drift_example.py --root path/to/dataset --machine-type fan --machine-id id_00
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import DatasetLoader
from feature_extraction import FeatureExtractor, FeatureVectorBuilder
from fingerprint import FingerprintGenerator
from preprocessing import PreprocessingPipeline
from profile import HealthyProfileBuilder
from drift import DriftAnalyzer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def build_fingerprint(record, pipeline, extractor, vec_builder, generator):
    """Preprocess one recording and return its AcousticFingerprint.

    Args:
        record: ``AudioMetadata`` from the dataset loader.
        pipeline: ``PreprocessingPipeline`` instance.
        extractor: ``FeatureExtractor`` instance.
        vec_builder: ``FeatureVectorBuilder`` instance.
        generator: ``FingerprintGenerator`` instance.

    Returns:
        ``AcousticFingerprint`` for the recording.
    """
    result = pipeline.run(record.absolute_path)
    features = extractor.extract(result["waveform"])
    vector, names = vec_builder.build(features)
    return generator.generate(features=(vector, names), metadata=record, sample_rate=result["sample_rate"])


def print_drift(label: str, result) -> None:
    """Print a formatted drift summary for one result.

    Args:
        label: Display label (e.g. ``"NORMAL"`` or ``"ABNORMAL"``).
        result: ``DriftResult`` instance.
    """
    print(f"\n  [{label}] {result.filename}")

    print("\n  RAW METRICS")
    print(f"    Cosine Similarity  : {result.cosine_similarity:.6f}")
    print(f"    Euclidean Distance : {result.euclidean_distance:.6f}")
    print(f"    Manhattan Distance : {result.manhattan_distance:.6f}")

    print("\n  NORMALIZED METRICS")
    print(f"    Cosine Similarity  : {result.norm_cosine_similarity:.6f}")
    print(f"    Euclidean Distance : {result.norm_euclidean_distance:.6f}")
    print(f"    Manhattan Distance : {result.norm_manhattan_distance:.6f}")

    abs_z = np.abs(result.z_score_vector)
    top_idx = np.argsort(abs_z)[::-1][:10]
    print(f"\n    Top 10 features by |z-score|:")
    print(f"    {'Feature name':<40}  z-score")
    print(f"    {'-' * 54}")
    for i in top_idx:
        print(f"    {result.feature_names[i]:<40}  {result.z_score_vector[i]:>8.4f}")


def main(root: str, machine_type: str | None, machine_id: str | None) -> None:
    loader = DatasetLoader(root)

    # ── Select target machine ─────────────────────────────────────────
    normal_records   = loader.filter_by_label("normal")
    abnormal_records = loader.filter_by_label("abnormal")

    if machine_type:
        normal_records   = [r for r in normal_records   if r.machine_type == machine_type]
        abnormal_records = [r for r in abnormal_records if r.machine_type == machine_type]
    if machine_id:
        normal_records   = [r for r in normal_records   if r.machine_id == machine_id]
        abnormal_records = [r for r in abnormal_records if r.machine_id == machine_id]

    if not normal_records:
        print("ERROR: No normal recordings found for the specified machine.")
        sys.exit(1)
    if not abnormal_records:
        print("ERROR: No abnormal recordings found for the specified machine.")
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
    print(f"Normal recordings available   : {len(normal_records)}")
    print(f"Abnormal recordings available : {len(abnormal_records)}")

    # ── Shared pipeline objects ───────────────────────────────────────
    pipeline    = PreprocessingPipeline(target_sr=16_000)
    extractor   = FeatureExtractor(sample_rate=16_000)
    vec_builder = FeatureVectorBuilder()
    generator   = FingerprintGenerator()

    # ── Split normal recordings: profile set vs held-out test ──────────
    # Hold out normal_records[0] as the test normal so it is never part
    # of the profile. If only one normal recording exists, it must serve
    # both roles — warn the user that the result will be artificially low.
    test_normal_record = normal_records[0]
    profile_records    = normal_records[1:] if len(normal_records) > 1 else normal_records

    if len(normal_records) == 1:
        print("WARNING: Only one normal recording available. "
              "It will be used for both the profile and the test — "
              "normal drift will appear artificially small.")

    print(f"\nBuilding healthy profile from {len(profile_records)} normal recording(s)...")
    profile_fps = [
        build_fingerprint(r, pipeline, extractor, vec_builder, generator)
        for r in profile_records
    ]
    profile = HealthyProfileBuilder().build(profile_fps)

    # ── Analyze the held-out normal and one abnormal recording ────────
    analyzer = DriftAnalyzer()

    normal_fp   = build_fingerprint(test_normal_record,   pipeline, extractor, vec_builder, generator)
    abnormal_fp = build_fingerprint(abnormal_records[0], pipeline, extractor, vec_builder, generator)

    normal_result   = analyzer.analyze(normal_fp,   profile)
    abnormal_result = analyzer.analyze(abnormal_fp, profile)

    # ── Report ────────────────────────────────────────────────────────
    print("\nDrift Analysis Results:")
    print_drift("NORMAL",   normal_result)
    print_drift("ABNORMAL", abnormal_result)

    print("\nSummary (abnormal drift should be larger):")
    print(f"  RAW       Euclidean — normal: {normal_result.euclidean_distance:.4f}  "
          f"abnormal: {abnormal_result.euclidean_distance:.4f}  "
          f"{'✓ larger' if abnormal_result.euclidean_distance > normal_result.euclidean_distance else '✗ not larger'}")
    print(f"  NORMALIZED Euclidean — normal: {normal_result.norm_euclidean_distance:.4f}  "
          f"abnormal: {abnormal_result.norm_euclidean_distance:.4f}  "
          f"{'✓ larger' if abnormal_result.norm_euclidean_distance > normal_result.norm_euclidean_distance else '✗ not larger'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint drift analysis example")
    parser.add_argument("--root", required=True, help="Dataset root directory.")
    parser.add_argument("--machine-type", default=None, help="Filter by machine type (optional).")
    parser.add_argument("--machine-id",   default=None, help="Filter by machine ID (optional).")
    args = parser.parse_args()
    main(args.root, args.machine_type, args.machine_id)
