"""Example: DSP feature extraction on one recording.

Usage:
    # From a specific file:
    python examples/feature_extraction_example.py --file path/to/recording.wav

    # Auto-select first normal recording from a dataset root:
    python examples/feature_extraction_example.py --root path/to/dataset
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import DatasetLoader
from feature_extraction import FeatureExtractor, FeatureVectorBuilder
from preprocessing import PreprocessingPipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def resolve_wav(file_path: str | None, root: str | None) -> Path:
    """Resolve the WAV file to process from CLI arguments.

    Args:
        file_path: Explicit path to a ``.wav`` file, or ``None``.
        root: Dataset root directory to scan, or ``None``.

    Returns:
        Resolved absolute ``Path`` to the selected ``.wav`` file.
    """
    if file_path is not None:
        wav = Path(file_path).resolve()
        if not wav.exists():
            print(f"ERROR: File not found: {wav}")
            sys.exit(1)
        if not wav.is_file():
            print(f"ERROR: Path is not a file: {wav}")
            sys.exit(1)
        return wav

    if root is not None:
        loader = DatasetLoader(root)
        normal = loader.filter_by_label("normal")
        if not normal:
            print(f"ERROR: No normal recordings found under: {Path(root).resolve()}")
            sys.exit(1)
        return normal[0].absolute_path

    print("ERROR: Provide either --file or --root.")
    sys.exit(1)


def main(file_path: str | None, root: str | None) -> None:
    wav = resolve_wav(file_path, root)

    print(f"\nSelected file: {wav}")

    # ── Preprocess ────────────────────────────────────────────────────
    pipeline = PreprocessingPipeline(target_sr=16_000)
    result = pipeline.run(wav)

    # ── Extract features ──────────────────────────────────────────────
    extractor = FeatureExtractor(sample_rate=result["sample_rate"])
    features = extractor.extract(result["waveform"])

    builder = FeatureVectorBuilder()
    vector, names = builder.build(features)

    # ── Report ────────────────────────────────────────────────────────
    print(f"\nNumber of extracted features : {len(features)}")
    print(f"Feature vector length        : {len(vector)}")
    print(f"\n{'Feature name':<40}  Value")
    print("-" * 58)
    for name, value in zip(names[:15], vector[:15]):
        print(f"  {name:<38}  {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DSP feature extraction example",
        epilog="Provide --file OR --root (not both).",
    )
    parser.add_argument("--file", default=None, help="Path to a specific .wav file.")
    parser.add_argument("--root", default=None, help="Dataset root directory; auto-selects the first normal recording.")
    args = parser.parse_args()

    if args.file is None and args.root is None:
        parser.error("Provide at least one of --file or --root.")

    main(args.file, args.root)
