"""Fusion Fingerprint extraction example.

Runs the full Version 2 pipeline:
    DatasetLoader → PreprocessingPipeline → DSP features → BEATs embedding → FusionBuilder

Usage:
    python examples/fusion_example.py --root data/raw/MIMII
    python examples/fusion_example.py --file path/to/audio.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.beats import BEATsEncoder
from src.dataset.loader import DatasetLoader
from src.feature_extraction.extractor import FeatureExtractor
from src.feature_extraction.feature_vector import FeatureVectorBuilder
from src.fusion import FusionBuilder
from src.preprocessing.pipeline import PreprocessingPipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "beats" / "BEATs_iter3_plus_AS2M.pt"


def _resolve_file(args: argparse.Namespace) -> tuple[Path, str, str, str]:
    """Return (audio_path, machine_type, machine_id, label)."""
    if args.file:
        return Path(args.file).resolve(), "", "", ""

    loader = DatasetLoader(args.root)
    normal = loader.filter_by_label("normal")
    if not normal:
        print(f"ERROR: No normal recordings found under: {args.root}")
        sys.exit(1)
    rec = normal[0]
    return rec.absolute_path, rec.machine_type, rec.machine_id, rec.label


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion Fingerprint extraction example")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single .wav file")
    group.add_argument("--root", type=str, help="Dataset root; auto-selects first normal recording")
    args = parser.parse_args()

    audio_path, machine_type, machine_id, label = _resolve_file(args)
    print(f"Selected file     : {audio_path}")

    # ── Preprocess ────────────────────────────────────────────────────
    pipeline = PreprocessingPipeline(target_sr=16_000)
    result = pipeline.run(audio_path)

    # ── DSP features ──────────────────────────────────────────────────
    extractor = FeatureExtractor(sample_rate=result["sample_rate"])
    features = extractor.extract(result["waveform"])
    dsp_vector, dsp_names = FeatureVectorBuilder().build(features)

    # ── BEATs embedding ───────────────────────────────────────────────
    encoder = BEATsEncoder(_CHECKPOINT)
    embedding = encoder.encode(
        waveform=result["waveform"],
        sample_rate=result["sample_rate"],
        filename=audio_path.name,
    )

    # ── Fusion ────────────────────────────────────────────────────────
    fused = FusionBuilder().build(
        dsp_vector=dsp_vector,
        dsp_feature_names=dsp_names,
        beats_embedding=embedding,
        machine_type=machine_type,
        machine_id=machine_id,
        label=label,
    )

    print(f"DSP dimension     : {len(fused.dsp_feature_vector)}")
    print(f"BEATs dimension   : {len(fused.beats_embedding)}")
    print(f"Fusion dimension  : {len(fused.fused_feature_vector)}")


if __name__ == "__main__":
    main()
