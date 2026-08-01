"""Contrastive Learning module — Version 3.

Provides the dataset infrastructure for contrastive fingerprint learning.

SDD v4 §2 (Version 3):
    Positive pairs: same machine, different recordings.
    Negative pairs: different machine_id or different machine_type.
    Only normal recordings are used during training.

Public API:
    ContrastiveDataset — encodes all normal recordings and builds pairs
    ContrastivePair    — dataclass holding (anchor, paired, label)
"""

from .dataset import ContrastiveDataset, ContrastivePair

__all__ = ["ContrastiveDataset", "ContrastivePair"]
