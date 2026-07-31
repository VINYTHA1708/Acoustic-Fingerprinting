"""Compare raw distances vs z-score normalized distances for normal and abnormal recordings.

Evaluates up to 50 normal and 50 abnormal recordings for one machine ID and
prints mean Euclidean, Manhattan, and Cosine for both raw and normalized
representations, then reports which gives better separation.

Usage:
    python examples/compare_distance_metrics.py --root path/to/dataset
    python examples/compare_distance_metrics.py --root path/to/dataset --machine-type pump --machine-id id_02
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from dataset import DatasetLoader
from drift import DriftAnalyzer
from feature_extraction import FeatureExtractor, FeatureVectorBuilder
from fingerprint import FingerprintGenerator
from preprocessing import PreprocessingPipeline
from profile import HealthyProfileBuilder

_MAX_SAMPLES = 50
_DEFAULT_MACHINE_TYPE = "pump"
_DEFAULT_MACHINE_ID = "id_00"


def build_fingerprint(record, pipeline, extractor, vec_builder, generator):
    result = pipeline.run(record.absolute_path)
    features = extractor.extract(result["waveform"])
    vector, names = vec_builder.build(features)
    return generator.generate(
        features=(vector, names),
        metadata=record,
        sample_rate=result["sample_rate"],
    )


def normalized_distances(vector, mean, std):
    """Compute Euclidean, Manhattan, Cosine of z-score normalized vector vs zero vector.

    Args:
        vector: Current feature vector (float32 ndarray).
        mean: Profile mean vector.
        std: Profile std vector.

    Returns:
        Tuple of (euclidean, manhattan, cosine) floats.
    """
    safe_std = np.where(std == 0.0, 1.0, std)
    z = (vector - mean) / safe_std  # distance from zero vector = norm of z

    euclidean = float(np.linalg.norm(z))
    manhattan = float(np.sum(np.abs(z)))

    norm_z = np.linalg.norm(z)
    # cosine similarity between z and zero vector is undefined; use cosine
    # similarity between z and the all-ones unit direction as a proxy for
    # how uniformly spread the deviation is — but the standard interpretation
    # here is cosine similarity of z against the mean direction (ones vector).
    ones = np.ones_like(z)
    norm_ones = np.linalg.norm(ones)
    cosine = float(np.dot(z, ones) / (norm_z * norm_ones)) if norm_z > 0 else 0.0

    return euclidean, manhattan, cosine


def evaluate_batch(records, profile, pipeline, extractor, vec_builder, generator, analyzer):
    """Return arrays of (raw_euc, raw_man, raw_cos, norm_euc, norm_man, norm_cos)."""
    raw_euc, raw_man, raw_cos = [], [], []
    norm_euc, norm_man, norm_cos = [], [], []

    for i, record in enumerate(records):
        print(f"  {i + 1}/{len(records)}: {record.filename}", end="\r")
        fp = build_fingerprint(record, pipeline, extractor, vec_builder, generator)
        drift = analyzer.analyze(fp, profile)

        raw_euc.append(drift.euclidean_distance)
        raw_man.append(drift.manhattan_distance)
        raw_cos.append(drift.cosine_similarity)

        ne, nm, nc = normalized_distances(
            fp.feature_vector, profile.mean_vector, profile.std_vector
        )
        norm_euc.append(ne)
        norm_man.append(nm)
        norm_cos.append(nc)

    print()
    return (
        np.array(raw_euc),  np.array(raw_man),  np.array(raw_cos),
        np.array(norm_euc), np.array(norm_man), np.array(norm_cos),
    )


def separation_ratio(abnormal_mean, normal_mean):
    return abnormal_mean / normal_mean if normal_mean > 0 else float("nan")


def main(root: str, machine_type: str, machine_id: str) -> None:
    loader = DatasetLoader(root)

    normal_records   = loader.filter_by_label("normal")
    abnormal_records = loader.filter_by_label("abnormal")

    normal_records   = [r for r in normal_records   if r.machine_type == machine_type and r.machine_id == machine_id]
    abnormal_records = [r for r in abnormal_records if r.machine_type == machine_type and r.machine_id == machine_id]

    if not normal_records:
        print(f"ERROR: No normal recordings found for {machine_type}/{machine_id}.")
        sys.exit(1)
    if not abnormal_records:
        print(f"ERROR: No abnormal recordings found for {machine_type}/{machine_id}.")
        sys.exit(1)

    print(f"\nMachine type : {machine_type}")
    print(f"Machine ID   : {machine_id}")
    print(f"Normal  available : {len(normal_records)}")
    print(f"Abnormal available: {len(abnormal_records)}")

    pipeline    = PreprocessingPipeline(target_sr=16_000)
    extractor   = FeatureExtractor(sample_rate=16_000)
    vec_builder = FeatureVectorBuilder()
    generator   = FingerprintGenerator()
    analyzer    = DriftAnalyzer()

    # Build profile from all normal recordings except the held-out first one
    profile_records = normal_records[1:] if len(normal_records) > 1 else normal_records
    print(f"\nBuilding healthy profile from {len(profile_records)} normal recording(s)...")
    profile_fps = [
        build_fingerprint(r, pipeline, extractor, vec_builder, generator)
        for r in profile_records
    ]
    profile = HealthyProfileBuilder().build(profile_fps)
    print(f"Profile built — {profile.number_of_samples} samples, {len(profile.feature_names)} features.")

    # Evaluation batches: re-insert held-out normal at position 0
    eval_normal   = ([normal_records[0]] + profile_records)[:_MAX_SAMPLES]
    eval_abnormal = abnormal_records[:_MAX_SAMPLES]

    print(f"\nEvaluating {len(eval_normal)} normal recordings...")
    n_re, n_rm, n_rc, n_ne, n_nm, n_nc = evaluate_batch(
        eval_normal, profile, pipeline, extractor, vec_builder, generator, analyzer
    )

    print(f"Evaluating {len(eval_abnormal)} abnormal recordings...")
    a_re, a_rm, a_rc, a_ne, a_nm, a_nc = evaluate_batch(
        eval_abnormal, profile, pipeline, extractor, vec_builder, generator, analyzer
    )

    # ── RAW DISTANCES ─────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print("RAW DISTANCES")
    print("=" * 45)
    print(f"  {'Metric':<22}  {'Normal':>10}  {'Abnormal':>10}")
    print(f"  {'-'*44}")
    print(f"  {'Mean Euclidean':<22}  {n_re.mean():>10.4f}  {a_re.mean():>10.4f}")
    print(f"  {'Mean Manhattan':<22}  {n_rm.mean():>10.4f}  {a_rm.mean():>10.4f}")
    print(f"  {'Mean Cosine':<22}  {n_rc.mean():>10.6f}  {a_rc.mean():>10.6f}")

    # ── NORMALIZED DISTANCES ──────────────────────────────────────────
    print("\n" + "=" * 45)
    print("NORMALIZED DISTANCES  (z-score vs zero vector)")
    print("=" * 45)
    print(f"  {'Metric':<22}  {'Normal':>10}  {'Abnormal':>10}")
    print(f"  {'-'*44}")
    print(f"  {'Mean Euclidean':<22}  {n_ne.mean():>10.4f}  {a_ne.mean():>10.4f}")
    print(f"  {'Mean Manhattan':<22}  {n_nm.mean():>10.4f}  {a_nm.mean():>10.4f}")
    print(f"  {'Mean Cosine':<22}  {n_nc.mean():>10.6f}  {a_nc.mean():>10.6f}")

    # ── Separation comparison ─────────────────────────────────────────
    raw_ratio  = separation_ratio(a_re.mean(), n_re.mean())
    norm_ratio = separation_ratio(a_ne.mean(), n_ne.mean())

    raw_pass  = a_re.mean() > n_re.mean()
    norm_pass = a_ne.mean() > n_ne.mean()

    print("\n" + "=" * 45)
    print("SEPARATION COMPARISON  (Euclidean, abnormal / normal)")
    print("=" * 45)
    print(f"  {'Representation':<24}  {'Ratio':>8}  {'Separated?':>10}")
    print(f"  {'-'*44}")
    print(f"  {'Raw':<24}  {raw_ratio:>8.4f}  {'YES ✓' if raw_pass else 'NO ✗':>10}")
    print(f"  {'Normalized (z-score)':<24}  {norm_ratio:>8.4f}  {'YES ✓' if norm_pass else 'NO ✗':>10}")

    if raw_pass and norm_pass:
        better = "Raw" if raw_ratio >= norm_ratio else "Normalized (z-score)"
        print(f"\n  Both representations separate the classes.")
        print(f"  Better separation: {better}  (ratio {max(raw_ratio, norm_ratio):.4f})")
    elif raw_pass:
        print(f"\n  Only RAW distances separate the classes.")
    elif norm_pass:
        print(f"\n  Only NORMALIZED distances separate the classes.")
    else:
        print(f"\n  Neither representation separates the classes for this machine/ID.")
        print(f"  Consider trying a different --machine-type or --machine-id.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare raw vs z-score normalized distances for DSP fingerprints."
    )
    parser.add_argument("--root",         required=True,                    help="Dataset root directory.")
    parser.add_argument("--machine-type", default=_DEFAULT_MACHINE_TYPE,   help="Machine type (default: pump).")
    parser.add_argument("--machine-id",   default=_DEFAULT_MACHINE_ID,     help="Machine ID (default: id_00).")
    args = parser.parse_args()
    main(args.root, args.machine_type, args.machine_id)
