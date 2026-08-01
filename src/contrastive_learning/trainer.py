"""ContrastiveTrainer — training pipeline for contrastive fingerprint learning.

SDD v4 §10 (Version 3):
    Train a small contrastive head over the Fusion Fingerprint.
    Positive pairs: same machine, different recordings.
    Single NT-Xent objective — no identity/health split.

Only positive pairs are used during training: the anchor and paired embeddings
from each ContrastivePair form the two views fed to NTXentLoss.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import torch
import torch.optim as optim

from .dataset import ContrastiveDataset, ContrastivePair
from .loss import NTXentLoss
from .model import ProjectionHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint schema
# ---------------------------------------------------------------------------

class _CheckpointDict(TypedDict):
    epoch: int
    model_state_dict: dict
    optimizer_state_dict: dict
    validation_loss: float


# ---------------------------------------------------------------------------
# History entry
# ---------------------------------------------------------------------------

@dataclass
class EpochResult:
    """Training and validation loss for one epoch.

    Attributes:
        epoch: 1-based epoch index.
        training_loss: Mean NT-Xent loss over all training batches.
        validation_loss: Mean NT-Xent loss over all validation batches.
    """

    epoch: int
    training_loss: float
    validation_loss: float


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ContrastiveTrainer:
    """Training pipeline for :class:`ProjectionHead` using NT-Xent loss.

    Args:
        head: The :class:`ProjectionHead` to train.
        criterion: The :class:`NTXentLoss` instance.
        learning_rate: Adam learning rate. Defaults to ``1e-3``.
        batch_size: Number of pairs per mini-batch. Defaults to ``32``.
        epochs: Number of full passes over the training pairs. Defaults to ``5``.
        checkpoint_dir: Directory where the best checkpoint is saved.
                        Defaults to ``models/contrastive``.
        val_split: Fraction of positive pairs reserved for validation.
                   Defaults to ``0.2``.
        seed: Random seed for pair shuffling and split. Defaults to ``42``.
    """

    def __init__(
        self,
        head: ProjectionHead,
        criterion: NTXentLoss,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 5,
        checkpoint_dir: str | Path = Path("models") / "contrastive",
        val_split: float = 0.2,
        seed: int = 42,
    ) -> None:
        self._head = head
        self._criterion = criterion
        self._batch_size = batch_size
        self._epochs = epochs
        self._checkpoint_dir = Path(checkpoint_dir)
        self._val_split = val_split
        self._rng = random.Random(seed)

        self._optimizer = optim.Adam(head.parameters(), lr=learning_rate)
        self._best_val_loss: float = math.inf
        self._history: list[EpochResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, dataset: ContrastiveDataset) -> None:
        """Train the projection head on positive pairs from *dataset*.

        Splits positive pairs into train/validation sets, runs for
        ``self._epochs`` epochs, prints per-epoch losses, and saves a
        checkpoint whenever validation loss improves.

        Args:
            dataset: A fully-constructed :class:`ContrastiveDataset`.
        """
        pairs = list(dataset.positive_pairs)
        if len(pairs) < 2:
            raise ValueError(
                f"Need at least 2 positive pairs to train, got {len(pairs)}."
            )

        self._rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * self._val_split))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]

        if len(train_pairs) < 2:
            raise ValueError(
                f"Too few training pairs after validation split: {len(train_pairs)}. "
                "Increase max_recordings or reduce val_split."
            )

        logger.info(
            "Training pairs: %d  |  Validation pairs: %d", len(train_pairs), len(val_pairs)
        )

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self._epochs + 1):
            train_loss = self._run_epoch(train_pairs, training=True)
            val_loss = self._run_epoch(val_pairs, training=False)

            result = EpochResult(epoch=epoch, training_loss=train_loss, validation_loss=val_loss)
            self._history.append(result)

            print(f"Epoch {epoch}/{self._epochs}")
            print(f"  Training loss   : {train_loss:.4f}")
            print(f"  Validation loss : {val_loss:.4f}")

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)
                logger.info("Checkpoint saved at epoch %d (val_loss=%.4f)", epoch, val_loss)

    def history(self) -> dict[str, list[float]]:
        """Return training and validation loss lists.

        Returns:
            Dict with keys ``"training_losses"`` and ``"validation_losses"``,
            each a list of floats with one entry per completed epoch.
        """
        return {
            "training_losses": [r.training_loss for r in self._history],
            "validation_losses": [r.validation_loss for r in self._history],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_epoch(self, pairs: list[ContrastivePair], *, training: bool) -> float:
        """Run one full pass over *pairs* and return the mean loss.

        Args:
            pairs: List of :class:`ContrastivePair` objects to iterate over.
            training: If True, performs backprop and optimizer step.

        Returns:
            Mean NT-Xent loss across all batches in this pass.
        """
        self._head.train(training)
        shuffled = list(pairs)
        if training:
            self._rng.shuffle(shuffled)

        total_loss = 0.0
        n_batches = 0

        for batch in self._make_batches(shuffled):
            anchors = torch.tensor(
                [p.anchor.fused_feature_vector for p in batch], dtype=torch.float32
            )
            paired = torch.tensor(
                [p.paired.fused_feature_vector for p in batch], dtype=torch.float32
            )

            if training:
                self._optimizer.zero_grad()
                emb_a = self._head(anchors)
                emb_b = self._head(paired)
                loss = self._criterion(emb_a, emb_b)
                loss.backward()
                self._optimizer.step()
            else:
                with torch.no_grad():
                    emb_a = self._head(anchors)
                    emb_b = self._head(paired)
                    loss = self._criterion(emb_a, emb_b)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

    def _make_batches(self, pairs: list[ContrastivePair]) -> list[list[ContrastivePair]]:
        """Partition *pairs* into mini-batches of size ``self._batch_size``.

        The final batch is dropped if it would contain fewer than 2 pairs,
        since NTXentLoss requires batch_size >= 2.

        Args:
            pairs: Flat list of pairs to partition.

        Returns:
            List of batches, each a list of :class:`ContrastivePair`.
        """
        batches = []
        for start in range(0, len(pairs), self._batch_size):
            batch = pairs[start : start + self._batch_size]
            if len(batch) >= 2:
                batches.append(batch)
        return batches

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Persist the best checkpoint to disk.

        Saves projection head weights, optimizer state, epoch index, and
        validation loss into a single file via :func:`torch.save`.

        Args:
            epoch: Current epoch (1-based).
            val_loss: Validation loss that triggered this save.
        """
        checkpoint: _CheckpointDict = {
            "epoch": epoch,
            "model_state_dict": self._head.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "validation_loss": val_loss,
        }
        path = self._checkpoint_dir / "best_projection_head.pt"
        torch.save(checkpoint, path)
