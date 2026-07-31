"""Example: build a HealthyFingerprintProfile from MIMII normal recordings.

Usage:
    python examples/profile_example.py --root path/to/dataset
    python examples/profile_example.py --root path/to/dataset --machine-type fan --machine-id id_00
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import DatasetLoader
from feature_extraction import FeatureExtractor, FeatureVectorBuilder
from fingerprint import FingerprintGenerator
from preprocessing import PreprocessingPipeline
from profile import HealthyProfileBuilder, ProfileSerializer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _check(condition: bool, label: str) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")


def main(root: str, machine_type: str | None, machine_id: str | None) -> None:
    # ── 1. Locate normal recordings ───────────────────────────────────
    loader = DatasetLoader(root)
    records = loader.filter_by_label("normal")

    if not records:
        print(f"ERROR: No normal recordings found under: {Path(root).resolve()}")
        sys.exit(1)

    if machine_type:
        records = [r for r in records if r.machine_type == machine_type]
    if machine_id:
        records = [r for r in records if r.machine_id == machine_id]

    if not records:
        print("ERROR: No normal recordings match the specified machine_type / machine_id.")
        sys.exit(1)

    # Use the first machine_type + machine_id combination found
    target_type = records[0].machine_type
    target_id   = records[0].machine_id
    records = [r for r in records if r.machine_type == target_type and r.machine_id == target_id]

    print(f"\nMachine type : {target_type}")
    print(f"Machine ID   : {target_id}")
    print(f"Normal recordings found: {len(records)}")

    # ── 2–4. Preprocess → extract → generate fingerprints ─────────────
    pipeline  = PreprocessingPipeline(target_sr=16_000)
    extractor = FeatureExtractor(sample_rate=16_000)
    builder   = FeatureVectorBuilder()
    generator = FingerprintGenerator()

    fingerprints = []
    for record in records:
        result   = pipeline.run(record.absolute_path)
        features = extractor.extract(result["waveform"])
        vector, names = builder.build(features)
        fp = generator.generate(
            features=(vector, names),
            metadata=record,
            sample_rate=result["sample_rate"],
        )
        fingerprints.append(fp)

    # ── 5. Build HealthyFingerprintProfile ────────────────────────────
    profile_builder = HealthyProfileBuilder()
    profile = profile_builder.build(fingerprints)

    print(f"\nFeature count      : {len(profile.feature_names)}")
    print(f"Mean vector length : {len(profile.mean_vector)}")

    print(f"\n{'Feature name':<40}  Mean        Std")
    print("-" * 68)
    for name, mean_val, std_val in zip(
        profile.feature_names[:10],
        profile.mean_vector[:10],
        profile.std_vector[:10],
    ):
        print(f"  {name:<38}  {mean_val:>10.6f}  {std_val:>10.6f}")

    # ── 6–8. Save, load, verify ───────────────────────────────────────
    serializer = ProfileSerializer()
    with tempfile.TemporaryDirectory() as tmp:
        json_path = Path(tmp) / "profile.json"
        npz_path  = Path(tmp) / "profile.npz"

        serializer.save_json(profile, json_path)
        serializer.save_npz(profile, npz_path)

        loaded_json = serializer.load_json(json_path)
        loaded_npz  = serializer.load_npz(npz_path)

        print("\nVerification:")
        for tag, loaded in (("JSON", loaded_json), ("NPZ", loaded_npz)):
            _check(loaded.machine_type     == profile.machine_type,     f"{tag} machine_type matches")
            _check(loaded.machine_id       == profile.machine_id,       f"{tag} machine_id matches")
            _check(loaded.number_of_samples == profile.number_of_samples, f"{tag} number_of_samples matches")
            _check(loaded.feature_names    == profile.feature_names,    f"{tag} feature_names match")
            _check(np.allclose(loaded.mean_vector, profile.mean_vector), f"{tag} mean_vector matches")
            _check(np.allclose(loaded.std_vector,  profile.std_vector),  f"{tag} std_vector matches")
            _check(np.allclose(loaded.min_vector,  profile.min_vector),  f"{tag} min_vector matches")
            _check(np.allclose(loaded.max_vector,  profile.max_vector),  f"{tag} max_vector matches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Healthy Fingerprint Profile example")
    parser.add_argument("--root", required=True, help="Dataset root directory.")
    parser.add_argument("--machine-type", default=None, help="Filter by machine type (optional).")
    parser.add_argument("--machine-id",   default=None, help="Filter by machine ID (optional).")
    args = parser.parse_args()
    main(args.root, args.machine_type, args.machine_id)
