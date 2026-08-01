"""Contrastive Learning module — Version 3.

Provides the dataset infrastructure for contrastive fingerprint learning.

SDD v4 §2 (Version 3):
    Positive pairs: same machine, different recordings.
    Negative pairs: different machine_id or different machine_type.
    Only normal recordings are used during training.

Public API:
    ContrastiveDataset  — encodes all normal recordings and builds pairs
    ContrastivePair     — dataclass holding (anchor, paired, label)
    NTXentLoss          — NT-Xent (InfoNCE) contrastive loss
    ProjectionHead      — small trainable head over the Fusion Fingerprint
    ContrastiveTrainer  — training pipeline (fit, history, checkpointing)
    EpochResult         — per-epoch training/validation loss record
"""

from .dataset import ContrastiveDataset, ContrastivePair
from .loss import NTXentLoss
from .model import ProjectionHead
from .trainer import ContrastiveTrainer, EpochResult

__all__ = [
    "ContrastiveDataset",
    "ContrastivePair",
    "NTXentLoss",
    "ProjectionHead",
    "ContrastiveTrainer",
    "EpochResult",
]
