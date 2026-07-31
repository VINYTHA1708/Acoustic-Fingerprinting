"""Example: acoustic fingerprint generation, serialization, and similarity.

Usage:
    python examples/fingerprint_example.py --file path/to/recording.wav
    python examples/fingerprint_example.py --root path/to/dataset
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import DatasetLoader
from dataset.metadata import AudioMetadata
from feature_extraction import FeatureExtractor, FeatureVectorBuilder
from fingerprint import AcousticFingerprint, FingerprintGenerator, FingerprintSerializer, FingerprintSimilarity
from preprocessing import PreprocessingPipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def resolve_wav_and_metadata(file_path: str | None, root: str | None):
    """Return ``(wav_path, AudioMetadata | None)`` from CLI arguments.

    Args:
        file_path: Explicit ``.wav`` path, or ``None``.
        root: Dataset root to scan, or ``None``.

    Returns:
        Tuple of ``(Path, AudioMetadata | None)``.
    """
    if file_path is not None:
        wav = Path(file_path).resolve()
        if not wav.exists():
            print(f"ERROR: File not found: {wav}")
            sys.exit(1)
        return wav, None

    if root is not None:
        loader = DatasetLoader(root)
        normal = loader.filter_by_label("normal")
        if not normal:
            print(f"ERROR: No normal recordings found under: {Path(root).resolve()}")
            sys.exit(1)
        record = normal[0]
        return record.absolute_path, record

    print("ERROR: Provide either --file or --root.")
    sys.exit(1)


def _check(condition: bool, label: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")


def main(file_path: str | None, root: str | None) -> None:
    wav, metadata = resolve_wav_and_metadata(file_path, root)
    print(f"\nSelected file: {wav}")

    # ── 1. Preprocess ─────────────────────────────────────────────────
    pipeline = PreprocessingPipeline(target_sr=16_000)
    result = pipeline.run(wav)

    # ── 2. Extract DSP features ───────────────────────────────────────
    extractor = FeatureExtractor(sample_rate=result["sample_rate"])
    features = extractor.extract(result["waveform"])

    # ── 3. Build feature vector ───────────────────────────────────────
    builder = FeatureVectorBuilder()
    vector, names = builder.build(features)

    # ── 4. Create AcousticFingerprint ─────────────────────────────────
    # Build a minimal AudioMetadata if --file was used without a dataset root
    if metadata is None:
        metadata = AudioMetadata(
            machine_type="unknown",
            machine_id="unknown",
            label="normal",
            filename=wav.name,
            relative_path=Path(wav.name),
            absolute_path=wav,
        )

    generator = FingerprintGenerator()
    fingerprint = generator.generate(
        features=(vector, names),
        metadata=metadata,
        sample_rate=result["sample_rate"],
    )

    print(f"\nNumber of features  : {len(fingerprint.feature_names)}")
    print(f"Feature vector length: {len(fingerprint.feature_vector)}")

    # ── 5. Save as JSON and NPZ ───────────────────────────────────────
    serializer = FingerprintSerializer()
    with tempfile.TemporaryDirectory() as tmp:
        json_path = Path(tmp) / "fingerprint.json"
        npz_path  = Path(tmp) / "fingerprint.npz"

        serializer.save_json(fingerprint, json_path)
        serializer.save_npz(fingerprint, npz_path)

        # ── 6. Load both back ─────────────────────────────────────────
        loaded_json = serializer.load_json(json_path)
        loaded_npz  = serializer.load_npz(npz_path)

        # ── 7. Verify round-trip ──────────────────────────────────────
        print("\nVerification:")
        for tag, loaded in (("JSON", loaded_json), ("NPZ", loaded_npz)):
            _check(loaded.machine_type == fingerprint.machine_type,  f"{tag} machine_type matches")
            _check(loaded.machine_id   == fingerprint.machine_id,    f"{tag} machine_id matches")
            _check(loaded.label        == fingerprint.label,         f"{tag} label matches")
            _check(loaded.feature_names == fingerprint.feature_names, f"{tag} feature_names match")
            _check(np.array_equal(loaded.feature_vector, fingerprint.feature_vector), f"{tag} vectors identical")

        # ── 8. Similarity (original vs JSON-loaded) ───────────────────
        sim = FingerprintSimilarity()
        cosine    = sim.cosine_similarity(fingerprint, loaded_json)
        euclidean = sim.euclidean_distance(fingerprint, loaded_json)
        manhattan = sim.manhattan_distance(fingerprint, loaded_json)

    print(f"\nSimilarity (original vs loaded):")
    print(f"  Cosine Similarity  : {cosine:.6f}")
    print(f"  Euclidean Distance : {euclidean:.6f}")
    print(f"  Manhattan Distance : {manhattan:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acoustic fingerprint generation example",
        epilog="Provide --file OR --root.",
    )
    parser.add_argument("--file", default=None, help="Path to a specific .wav file.")
    parser.add_argument("--root", default=None, help="Dataset root; auto-selects first normal recording.")
    args = parser.parse_args()

    if args.file is None and args.root is None:
        parser.error("Provide at least one of --file or --root.")

    main(args.file, args.root)
