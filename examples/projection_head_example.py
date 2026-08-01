"""Projection head example.

Loads one recording, obtains its fused feature vector via FusionCache,
passes it through ProjectionHead, and prints a summary.

Usage:
    python examples/projection_head_example.py --root data/raw/MIMII
    python examples/projection_head_example.py --file path/to/audio.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.beats.encoder import BEATsEncoder
from src.contrastive_learning.model import ProjectionHead
from src.dataset.loader import DatasetLoader
from src.dataset.metadata import AudioMetadata, extract_metadata
from src.feature_extraction.extractor import FeatureExtractor
from src.feature_extraction.feature_vector import FeatureVectorBuilder
from src.fusion.cache import FusionCache
from src.fusion.fusion import FusionBuilder
from src.preprocessing.pipeline import PreprocessingPipeline

_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "beats" / "BEATs_iter3_plus_AS2M.pt"
_CACHE_ROOT = Path(__file__).resolve().parent.parent / "data" / "fusion_cache"


def _build_cache() -> FusionCache:
    return FusionCache(
        cache_root=_CACHE_ROOT,
        pipeline=PreprocessingPipeline(target_sr=16_000),
        extractor=FeatureExtractor(sample_rate=16_000),
        vec_builder=FeatureVectorBuilder(),
        encoder=BEATsEncoder(_CHECKPOINT),
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
    parser = argparse.ArgumentParser(description="Projection head example")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single .wav file")
    group.add_argument("--root", type=str, help="Dataset root; auto-selects first normal recording")
    args = parser.parse_args()

    rec = _resolve_record(args)
    print(f"Selected file   : {rec.absolute_path}")

    cache = _build_cache()
    fused = cache.load_or_create(rec)

    fused_tensor = torch.tensor(fused.fused_feature_vector, dtype=torch.float32)

    head = ProjectionHead()
    head.eval()
    with torch.no_grad():
        embedding = head(fused_tensor)

    l2_norm = embedding.norm(p=2).item()

    print(f"Input dimension : {fused_tensor.shape[0]}")
    print(f"Output dimension: {embedding.shape[0]}")
    print(f"Output L2 norm  : {l2_norm:.6f}")


if __name__ == "__main__":
    main()
