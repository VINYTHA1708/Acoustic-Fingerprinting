"""NT-Xent loss example.

Loads a small batch of normal recordings via FusionCache, passes each through
ProjectionHead to obtain two sets of embeddings (anchor and paired), then
computes one NT-Xent loss value.

Usage:
    python examples/ntxent_loss_example.py --root data/raw/MIMII
    python examples/ntxent_loss_example.py --root data/raw/MIMII --machine-type pump --machine-id id_00 --batch-size 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.beats.encoder import BEATsEncoder
from src.contrastive_learning.dataset import ContrastiveDataset
from src.contrastive_learning.loss import NTXentLoss
from src.contrastive_learning.model import ProjectionHead
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


def main() -> None:
    parser = argparse.ArgumentParser(description="NT-Xent loss example")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--machine-type", type=str, default=None)
    parser.add_argument("--machine-id", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8, help="Number of pairs (default: 8)")
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    # Build dataset and collect up to batch_size positive pairs
    dataset = ContrastiveDataset(
        dataset_root=args.root,
        cache_root=_CACHE_ROOT,
        machine_type=args.machine_type,
        machine_id=args.machine_id,
        max_recordings=args.batch_size * 4,  # ensure enough recordings for pairs
    )

    pairs = dataset.positive_pairs[: args.batch_size]
    if len(pairs) < 2:
        print(
            f"ERROR: Need at least 2 positive pairs for NT-Xent loss, "
            f"got {len(pairs)}. Try a larger --batch-size or more recordings."
        )
        sys.exit(1)

    # Stack anchor and paired fused vectors into tensors
    anchors = torch.from_numpy(
        np.stack([p.anchor.fused_feature_vector for p in pairs])
    ).float()
    paired = torch.from_numpy(
        np.stack([p.paired.fused_feature_vector for p in pairs])
    ).float()

    # Project both batches through the shared ProjectionHead
    head = ProjectionHead()
    head.eval()
    with torch.no_grad():
        emb_a = head(anchors)
        emb_b = head(paired)

    # Compute NT-Xent loss
    criterion = NTXentLoss(temperature=args.temperature)
    loss = criterion(emb_a, emb_b)

    print(f"Batch size          : {len(pairs)}")
    print(f"Embedding dimension : {emb_a.shape[1]}")
    print(f"Temperature         : {args.temperature}")
    print(f"NT-Xent loss        : {loss.item():.6f}")


if __name__ == "__main__":
    main()
