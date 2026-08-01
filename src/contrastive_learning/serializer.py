"""ContrastiveSerializer — single checkpoint interface for contrastive learning.

Provides save_checkpoint / load_checkpoint as the canonical persistence layer
for the entire contrastive learning module.  Both ContrastiveTrainer and
ContrastiveInference are designed to use this interface.

Checkpoint schema (torch.save / torch.load):
    {
        "model_state_dict":     dict          # required
        "epoch":                int           # required
        "validation_loss":      float         # required
        "optimizer_state_dict": dict | None   # optional
        "config":               dict          # optional, defaults to {}
    }

The schema is a strict superset of the format previously written by
ContrastiveTrainer._save_checkpoint, so checkpoints produced by either
source are loadable by ContrastiveInference without modification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Keys that must be present in every checkpoint written or read by this class.
_REQUIRED_KEYS: frozenset[str] = frozenset({"model_state_dict", "epoch", "validation_loss"})


class ContrastiveSerializer:
    """Save and load ProjectionHead checkpoints.

    Designed to be reused by :class:`~contrastive_learning.trainer.ContrastiveTrainer`
    and :class:`~contrastive_learning.inference.ContrastiveInference` so that
    checkpoint serialization logic lives in exactly one place.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def save_checkpoint(
        path: str | Path,
        model_state_dict: dict,
        epoch: int,
        validation_loss: float,
        optimizer_state_dict: dict | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Persist a checkpoint to *path* using :func:`torch.save`.

        Args:
            path: Destination ``.pt`` file.  Parent directories are created
                  automatically.
            model_state_dict: ``ProjectionHead.state_dict()`` output.
            epoch: 1-based epoch index at the time of saving.
            validation_loss: Validation loss that triggered this save.
            optimizer_state_dict: ``optimizer.state_dict()`` output.  Pass
                                  ``None`` to omit (e.g. inference-only saves).
            config: Arbitrary JSON-serialisable configuration dictionary
                    (e.g. ``{"temperature": 0.1, "batch_size": 32}``).
                    Defaults to an empty dict when omitted.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": model_state_dict,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "optimizer_state_dict": optimizer_state_dict,
            "config": config if config is not None else {},
        }

        torch.save(checkpoint, path)
        logger.info(
            "Checkpoint saved: '%s'  (epoch=%d, val_loss=%.4f)",
            path,
            epoch,
            validation_loss,
        )

    @staticmethod
    def load_checkpoint(path: str | Path) -> dict[str, Any]:
        """Load a checkpoint from *path* using :func:`torch.load`.

        Args:
            path: Source ``.pt`` file written by :meth:`save_checkpoint` or
                  by ``ContrastiveTrainer._save_checkpoint``.

        Returns:
            Dictionary with keys:

            - ``"model_state_dict"`` — ``dict``
            - ``"epoch"``            — ``int``
            - ``"validation_loss"``  — ``float``
            - ``"optimizer_state_dict"`` — ``dict | None``
            - ``"config"``           — ``dict`` (empty dict if absent in file)

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If any required key is missing from the checkpoint file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: '{path}'")

        checkpoint: dict[str, Any] = torch.load(path, map_location="cpu")

        missing = _REQUIRED_KEYS - checkpoint.keys()
        if missing:
            raise KeyError(
                f"Checkpoint at '{path}' is missing required keys: {sorted(missing)}. "
                f"Found keys: {sorted(checkpoint.keys())}"
            )

        # Back-fill optional keys so callers never need to guard with .get()
        checkpoint.setdefault("optimizer_state_dict", None)
        checkpoint.setdefault("config", {})

        logger.info(
            "Checkpoint loaded: '%s'  (epoch=%s, val_loss=%s)",
            path,
            checkpoint["epoch"],
            checkpoint["validation_loss"],
        )
        return checkpoint
