"""Contrastive inference example.

Loads one recording, obtains its fused feature vector via FusionCache, loads
the trained ProjectionHead checkpoint, and generates one learned fingerprint.

Usage:
    python examples/contrastive_inference_example.py --root data/raw/MIMII
    python examples/contrastive_inference_example.py --file path/to/audio.wav
    python examples/contrastive_inference_example.py \\
        --root data/raw/MIMII --checkpoint models/contrastive/best_projection_head.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.beats.encoder import BEATsEncoder
from src.contrastive_learning.inference import ContrastiveInference
from src.contrastive_learning.model import ProjectionHead
from src.dataset.loader import DatasetLoader
from src.dataset.metadata import AudioMetadata, extract_metadata
from src.feature_extraction.extractor import FeatureExtractor
from src.feature_extraction.feature_vector import FeatureVectorBuilder
from src.fusion.cache import FusionCache
from src.fusion.fusion import FusionBuilder
from src.preprocessing.pipeline import PreprocessingPipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

_BEATS_CHECKPOINT = (
    Path(__file__).resolve().parent.parent / "models" / "beats" / "BEATs_iter3_plus_AS2M.pt"
)
_CACHE_ROOT = Path(__file__).resolve().parent.parent / "data" / "fusion_cache"
_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent / "models" / "contrastive" / "best_projection_head.pt"
)


def _build_cache() -> FusionCache:
    return FusionCache(
        cache_root=_CACHE_ROOT,
        pipeline=PreprocessingPipeline(target_sr=16_000),
        extractor=FeatureExtractor(sample_rate=16_000),
        vec_builder=FeatureVectorBuilder(),
        encoder=BEATsEncoder(_BEATS_CHECKPOINT),
        fusion=FusionBuilder(),
    )


def _resolve_record(args: argparse.Namespace) -> AudioMetadata:
    if args.file:
        path = Path(args.file).resolve()
        meta = extract_metadata(path, path.parent.parent.parent)
        if meta is None:
            meta = AudioMetadata(
                machine_type="unknown",
                machine_id="id_00",
                label="normal",
                filename=path.name,
                relative_path=path.name,
                absolute_path=path,
            )
        return meta

    loader = DatasetLoader(args.root)
    normal = loader.filter_by_label("normal")
    if not normal:
        print(f"ERROR: No normal recordings found under: {args.root}")
        sys.exit(1)
    return normal[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive inference example")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single .wav file")
    group.add_argument("--root", type=str, help="Dataset root; auto-selects first normal recording")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_DEFAULT_CHECKPOINT),
        help="Path to trained ProjectionHead checkpoint (default: models/contrastive/best_projection_head.pt)",
    )
    args = parser.parse_args()

    rec = _resolve_record(args)
    print(f"Selected file   : {rec.absolute_path}")

    # Obtain fused feature vector
    cache = _build_cache()
    fused = cache.load_or_create(rec)

    # Load trained head and run inference
    head = ProjectionHead()
    inference = ContrastiveInference(
        projection_head=head,
        checkpoint_path=args.checkpoint,
    )
    fingerprint = inference.generate_fingerprint(fused)

    l2_norm = float(np.linalg.norm(fingerprint))

    print(f"Input dimension : {len(fused.fused_feature_vector)}")
    print(f"Output dimension: {fingerprint.shape[0]}")
    print(f"Output L2 norm  : {l2_norm:.6f}")


if __name__ == "__main__":
    main()
