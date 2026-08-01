"""BEATs embedding extraction example.

Usage:
    # From a single file:
    python examples/beats_example.py --file path/to/audio.wav

    # From a dataset root (auto-selects first normal recording):
    python examples/beats_example.py --root data/raw/MIMII
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.beats import BEATsEncoder
from src.preprocessing.pipeline import PreprocessingPipeline

_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "beats" / "BEATs_iter3_plus_AS2M.pt"


def _select_file_from_root(root: str) -> Path:
    from src.dataset.loader import DatasetLoader

    loader = DatasetLoader(root)
    normal = loader.filter_by_label("normal")
    if not normal:
        raise FileNotFoundError(f"No normal recordings found under: {root}")
    return Path(normal[0].absolute_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="BEATs embedding extraction example")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single .wav file")
    group.add_argument("--root", type=str, help="Dataset root; auto-selects first normal recording")
    args = parser.parse_args()

    if args.root:
        audio_path = _select_file_from_root(args.root)
    else:
        audio_path = Path(args.file)

    print(f"Selected file : {audio_path}")

    pipeline = PreprocessingPipeline(target_sr=16_000)
    result = pipeline.run(audio_path)

    encoder = BEATsEncoder(_CHECKPOINT)
    embedding = encoder.encode(
        waveform=result["waveform"],
        sample_rate=result["sample_rate"],
        filename=audio_path.name,
    )

    print(f"Embedding dimension : {embedding.embedding_dim}")
    print(f"Embedding shape     : {embedding.vector.shape}")
    print(f"First 10 values     : {embedding.vector[:10]}")


if __name__ == "__main__":
    main()
