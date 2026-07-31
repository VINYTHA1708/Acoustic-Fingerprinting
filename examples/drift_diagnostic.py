"""Diagnostic: prints the exact vectors used in drift similarity calculations.

Usage:
    python examples/drift_diagnostic.py --root path/to/dataset
    python examples/drift_diagnostic.py --root path/to/dataset --machine-type fan --machine-id id_00
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import DatasetLoader
from feature_extraction import FeatureExtractor, FeatureVectorBuilder
from fingerprint import FingerprintGenerator
from fingerprint.fingerprint import AcousticFingerprint
from preprocessing import PreprocessingPipeline
from profile import HealthyProfileBuilder
from profile.profile import HealthyFingerprintProfile


def build_fp(record, pipeline, extractor, vb, gen):
    r = pipeline.run(record.absolute_path)
    feats = extractor.extract(r["waveform"])
    vec, names = vb.build(feats)
    return gen.generate(features=(vec, names), metadata=record, sample_rate=r["sample_rate"])


def inspect_similarity(
    fp: AcousticFingerprint,
    profile: HealthyFingerprintProfile,
    label: str,
) -> None:
    """Print the exact arrays that will be passed into each similarity function."""

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC — {label}")
    print(f"{'='*60}")

    # --- What _profile_as_fingerprint() produces ---
    profile_fp_vector = profile.mean_vector.astype("float32")

    left  = fp.feature_vector.astype("float32")
    right = profile_fp_vector

    print(f"\nleft  vector source : fingerprint.feature_vector")
    print(f"right vector source : profile.mean_vector  (via _profile_as_fingerprint)")
    print(f"left  shape         : {left.shape}")
    print(f"right shape         : {right.shape}")

    # --- Confirm right is mean, not std/min/max ---
    is_mean = np.allclose(right, profile.mean_vector.astype("float32"))
    is_std  = np.allclose(right, profile.std_vector.astype("float32"))
    is_min  = np.allclose(right, profile.min_vector.astype("float32"))
    is_max  = np.allclose(right, profile.max_vector.astype("float32"))
    print(f"\nright == profile.mean_vector : {is_mean}")
    print(f"right == profile.std_vector  : {is_std}  <-- must be False")
    print(f"right == profile.min_vector  : {is_min}  <-- must be False")
    print(f"right == profile.max_vector  : {is_max}  <-- must be False")

    # --- Feature name identity and ordering ---
    names_identical = fp.feature_names == profile.feature_names
    print(f"\nfeature_names identical (==) : {names_identical}")
    if not names_identical:
        for i, (fn, pn) in enumerate(zip(fp.feature_names, profile.feature_names)):
            if fn != pn:
                print(f"  MISMATCH at index {i}: fp='{fn}'  profile='{pn}'")

    # --- Manually compute the three metrics ---
    norm_l = np.linalg.norm(left)
    norm_r = np.linalg.norm(right)
    cosine   = float(np.dot(left, right) / (norm_l * norm_r)) if norm_l > 1e-10 and norm_r > 1e-10 else 0.0
    euclid   = float(np.linalg.norm(left - right))
    manhattan = float(np.sum(np.abs(left - right)))

    print(f"\nManually computed metrics (left=fp.feature_vector, right=profile.mean_vector):")
    print(f"  Cosine similarity  : {cosine:.6f}")
    print(f"  Euclidean distance : {euclid:.6f}")
    print(f"  Manhattan distance : {manhattan:.6f}")

    # --- Show first 5 values of each vector for spot-check ---
    print(f"\nFirst 5 values — left  (fp.feature_vector) : {left[:5]}")
    print(f"First 5 values — right (profile.mean_vector): {right[:5]}")

    # --- Profile internals sanity check ---
    print(f"\nProfile internals:")
    print(f"  mean_vector[:5]  : {profile.mean_vector[:5]}")
    print(f"  std_vector[:5]   : {profile.std_vector[:5]}")
    print(f"  number_of_samples: {profile.number_of_samples}")
    print(f"  machine_type     : {profile.machine_type}")
    print(f"  machine_id       : {profile.machine_id}")


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

    if not normal_records or not abnormal_records:
        print("ERROR: Need at least one normal and one abnormal recording.")
        sys.exit(1)

    target_type = normal_records[0].machine_type
    target_id   = normal_records[0].machine_id
    normal_records   = [r for r in normal_records   if r.machine_type == target_type and r.machine_id == target_id]
    abnormal_records = [r for r in abnormal_records if r.machine_type == target_type and r.machine_id == target_id]

    pipeline    = PreprocessingPipeline(target_sr=16_000)
    extractor   = FeatureExtractor(sample_rate=16_000)
    vec_builder = FeatureVectorBuilder()
    generator   = FingerprintGenerator()

    # Build profile from held-out set (exclude index 0)
    profile_records = normal_records[1:] if len(normal_records) > 1 else normal_records
    profile_fps = [build_fp(r, pipeline, extractor, vec_builder, generator) for r in profile_records]
    profile = HealthyProfileBuilder().build(profile_fps)

    normal_fp   = build_fp(normal_records[0],   pipeline, extractor, vec_builder, generator)
    abnormal_fp = build_fp(abnormal_records[0], pipeline, extractor, vec_builder, generator)

    inspect_similarity(normal_fp,   profile, "NORMAL recording vs profile")
    inspect_similarity(abnormal_fp, profile, "ABNORMAL recording vs profile")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift diagnostic")
    parser.add_argument("--root", required=True)
    parser.add_argument("--machine-type", default=None)
    parser.add_argument("--machine-id",   default=None)
    args = parser.parse_args()
    main(args.root, args.machine_type, args.machine_id)
