"""Example: demonstrates the PreprocessingPipeline on one MIMII recording.

Usage:
    python examples/preprocessing_example.py --file data/MIMII/fan/id_00/normal/00000000.wav
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import PreprocessingPipeline
from dataset import DatasetLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main(file_path: str | None, root: str) -> None:
    # Resolve a file to process
    if file_path:
        wav = Path(file_path)
    else:
        loader = DatasetLoader(root)
        normal = loader.filter_by_label("normal")
        if not normal:
            print("No normal recordings found under:", root)
            sys.exit(1)
        wav = normal[0].absolute_path

    print(f"\nFile : {wav}")

    # Load raw to capture original sample rate before pipeline resamples
    import librosa
    _, original_sr = librosa.load(wav, sr=None, mono=False)

    pipeline = PreprocessingPipeline(target_sr=16_000)
    result = pipeline.run(wav)

    print(f"Original sample rate : {original_sr} Hz")
    print(f"Final sample rate    : {result['sample_rate']} Hz")
    print(f"Waveform shape       : {result['waveform'].shape}")
    print(f"Spectrogram shape    : {result['spectrogram'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing pipeline example")
    parser.add_argument("--file", default=None, help="Path to a specific .wav file")
    parser.add_argument("--root", default="data/MIMII", help="Dataset root (used if --file is omitted)")
    args = parser.parse_args()
    main(args.file, args.root)
