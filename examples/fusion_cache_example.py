"""Fusion cache example.

Demonstrates FusionCache behaviour on a single recording:
  - First run:  cache miss  → computes and saves the fused vector.
  - Second run: cache hit   → loads from disk instantly.

Usage:
    python examples/fusion_cache_example.py --root data/raw/MIMII
    python examples/fusion_cache_example.py --file path/to/audio.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.beats.encoder import BEATsEncoder
from src.dataset.loader import DatasetLoader
from src.dataset.metadata import AudioMetadata, extract_metadata
from src.feature_extraction.extractor import FeatureExtractor
from src.feature_extraction.feature_vector import FeatureVectorBuilder
from src.fusion.cache import FusionCache
from src.fusion.fusion import FusionBuilder
from src.preprocessing.pipeline import PreprocessingPipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

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
        # Synthesise a minimal AudioMetadata for a standalone file
        meta = extract_metadata(path, path.parent.parent.parent)
        if meta is None:
            # Fallback: treat as unknown machine
            from src.dataset.metadata import AudioMetadata as AM
            meta = AM(
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


def _run(label: str, rec: AudioMetadata, cache: FusionCache) -> None:
    print(f"\n--- {label} ---")
    t0 = time.perf_counter()
    fused = cache.load_or_create(rec, verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"Fusion dimension : {len(fused.fused_feature_vector)}")
    print(f"Time             : {elapsed:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion cache example")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single .wav file")
    group.add_argument("--root", type=str, help="Dataset root; auto-selects first normal recording")
    args = parser.parse_args()

    rec = _resolve_record(args)
    print(f"Selected file : {rec.absolute_path}")

    cache = _build_cache()

    _run("First run", rec, cache)
    _run("Second run", rec, cache)


if __name__ == "__main__":
    main()
