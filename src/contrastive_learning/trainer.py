"""ContrastiveTrainer — training pipeline for contrastive fingerprint learning.

SDD v4 §10 (Version 4):
    Train a small contrastive head over the Fusion Fingerprint.
    Positive pairs: same machine, different recordings.
    Single NT-Xent objective — no identity/health split.

Machine-aware batching
-----------------------
Each batch contains at most ONE positive pair per (machine_type, machine_id).
This ensures that every non-positive embedding in the batch belongs to a
different machine, eliminating false negatives in the NT-Xent loss.

Recording-level train/validation split
---------------------------------------
The split is performed inside ContrastiveDataset before pair generation.
ContrastiveTrainer consumes dataset.train_positive_pairs and
dataset.val_positive_pairs directly — it never re-splits pairs itself.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from .dataset import ContrastiveDataset, ContrastivePair
from .loss import NTXentLoss
from .model import ProjectionHead
from .serializer import ContrastiveSerializer

logger = logging.getLogger(__name__)


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
        learning_rate: Adam learning rate. Must be > 0. Defaults to ``1e-3``.
        batch_size: Number of pairs per mini-batch. Must be >= 2. Defaults to ``32``.
        epochs: Number of full passes over the training pairs. Must be > 0.
                Defaults to ``5``.
        checkpoint_dir: Directory where the best checkpoint is saved.
                        Defaults to ``models/contrastive``.
        val_split: Fraction of positive pairs reserved for validation.
                   Must be in (0, 1). Defaults to ``0.2``.
        seed: Random seed for pair shuffling. Defaults to ``42``.

    Raises:
        ValueError: If any parameter is outside its valid range.
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
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if batch_size < 2:
            raise ValueError(f"batch_size must be >= 2, got {batch_size}")
        if epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {epochs}")
        if not (0 < val_split < 1):
            raise ValueError(f"val_split must be in (0, 1), got {val_split}")

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

        Consumes ``dataset.train_positive_pairs`` and
        ``dataset.val_positive_pairs`` (recording-level split already applied
        inside :class:`ContrastiveDataset`).  Runs for ``self._epochs`` epochs,
        prints per-epoch losses, and saves a checkpoint whenever validation
        loss improves.

        Args:
            dataset: A fully-constructed :class:`ContrastiveDataset`.

        Raises:
            ValueError: If there are fewer than 2 training or validation pairs.
        """
        train_pairs = dataset.train_positive_pairs
        val_pairs = dataset.val_positive_pairs

        if len(train_pairs) < 2:
            raise ValueError(
                f"Need at least 2 training pairs, got {len(train_pairs)}. "
                "Increase max_recordings or reduce val_split."
            )
        if len(val_pairs) < 2:
            raise ValueError(
                f"Need at least 2 validation pairs, got {len(val_pairs)}. "
                "Increase max_recordings or reduce val_split."
            )

        logger.info(
            "Training pairs: %d  |  Validation pairs: %d",
            len(train_pairs),
            len(val_pairs),
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

        Batches are constructed with machine-aware sampling: each batch
        contains at most one positive pair per (machine_type, machine_id),
        eliminating false negatives in the NT-Xent loss.

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

        for batch in self._make_machine_aware_batches(shuffled):
            anchors = torch.from_numpy(
                np.stack([p.anchor.fused_feature_vector for p in batch])
            ).float()

            paired = torch.from_numpy(
                np.stack([p.paired.fused_feature_vector for p in batch])
            ).float()

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

    def _make_machine_aware_batches(
        self, pairs: list[ContrastivePair]
    ) -> list[list[ContrastivePair]]:
        """Partition *pairs* into batches with at most one pair per machine.

        Algorithm:
          1. Group pairs by (machine_type, machine_id).
          2. Round-robin across machines, taking one pair per machine per
             batch slot, until ``batch_size`` slots are filled or all
             machines are exhausted for this batch.
          3. Repeat until all pairs are consumed.
          4. Drop the final batch if it contains fewer than 2 pairs
             (NTXentLoss requires batch_size >= 2).

        Args:
            pairs: Flat list of :class:`ContrastivePair` objects.

        Returns:
            List of batches, each a list of :class:`ContrastivePair`.
        """
        # Group by machine key, preserving the shuffled order within each group
        groups: dict[tuple[str, str], list[ContrastivePair]] = {}
        for p in pairs:
            key = (p.anchor.machine_type, p.anchor.machine_id)
            groups.setdefault(key, []).append(p)

        # Ordered list of machine keys for round-robin
        machine_keys = list(groups.keys())
        # Per-machine cursor
        cursors: dict[tuple[str, str], int] = {k: 0 for k in machine_keys}

        batches: list[list[ContrastivePair]] = []

        while True:
            batch: list[ContrastivePair] = []
            # Round-robin: one pair per machine until batch is full
            for key in machine_keys:
                if len(batch) >= self._batch_size:
                    break
                idx = cursors[key]
                if idx < len(groups[key]):
                    batch.append(groups[key][idx])
                    cursors[key] = idx + 1

            if not batch:
                break  # all pairs consumed

            if len(batch) >= 2:
                batches.append(batch)

            # Stop if every machine's cursor is exhausted
            if all(cursors[k] >= len(groups[k]) for k in machine_keys):
                break

        return batches

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Persist the best checkpoint to disk via :class:`ContrastiveSerializer`.

        Args:
            epoch: Current epoch (1-based).
            val_loss: Validation loss that triggered this save.
        """
        ContrastiveSerializer.save_checkpoint(
            path=self._checkpoint_dir / "best_projection_head.pt",
            model_state_dict=self._head.state_dict(),
            optimizer_state_dict=self._optimizer.state_dict(),
            epoch=epoch,
            validation_loss=val_loss,
        )
