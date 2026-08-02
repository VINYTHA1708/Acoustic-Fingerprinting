"""ContrastiveInference — generates learned fingerprints from a trained ProjectionHead.

SDD v4 §11 (Version 3):
    At inference time, extract DSP + BEATs → Fusion Fingerprint, then pass
    through the trained contrastive head to obtain the final embedding used
    for drift analysis.

Checkpoint loading is delegated to :class:`ContrastiveSerializer`, which is
the single checkpoint interface for the entire contrastive learning module.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from ..fusion.fused_vector import FusedFeatureVector
from .model import ProjectionHead
from .serializer import ContrastiveSerializer

logger = logging.getLogger(__name__)

_FUSED_DIM = 921
_OUTPUT_DIM = 256
_NORM_TOLERANCE = 1e-3


class ContrastiveInference:
    """Generates 256-dimensional learned fingerprints from a trained ProjectionHead.

    Loads the best checkpoint saved by :class:`ContrastiveTrainer`, sets the
    model to evaluation mode, and disables gradient computation for all
    subsequent calls.

    Args:
        projection_head: A :class:`ProjectionHead` instance (weights will be
                         overwritten by the checkpoint).
        checkpoint_path: Path to the ``.pt`` checkpoint written by
                         :class:`ContrastiveTrainer`.

    Raises:
        FileNotFoundError: If *checkpoint_path* does not exist.
        KeyError: If the checkpoint file is missing ``"model_state_dict"``.
    """

    def __init__(
        self,
        projection_head: ProjectionHead,
        checkpoint_path: str | Path,
    ) -> None:
        checkpoint = ContrastiveSerializer.load_checkpoint(checkpoint_path)

        projection_head.load_state_dict(checkpoint["model_state_dict"])
        projection_head.eval()

        self._head = projection_head
        logger.info(
            "Loaded checkpoint from '%s' (epoch=%s, val_loss=%s)",
            checkpoint_path,
            checkpoint["epoch"],
            checkpoint["validation_loss"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_fingerprint(self, fused_vector: FusedFeatureVector) -> np.ndarray:
        """Project a fused feature vector into a 256-dimensional learned fingerprint.

        Args:
            fused_vector: A :class:`~fusion.fused_vector.FusedFeatureVector`
                          whose ``fused_feature_vector`` field has shape ``(921,)``.

        Returns:
            L2-normalised ``numpy.ndarray`` of shape ``(256,)``, dtype ``float32``.

        Raises:
            ValueError: If the input dimension is not 921, the output dimension
                        is not 256, or the output contains NaN, Inf, or has an
                        L2 norm that deviates from 1 by more than 1e-3.
        """
        raw = fused_vector.fused_feature_vector
        if raw.shape[0] != _FUSED_DIM:
            raise ValueError(
                f"fused_feature_vector must have dimension {_FUSED_DIM}, got {raw.shape[0]}"
            )

        x = torch.from_numpy(raw).float()

        with torch.no_grad():
            embedding = self._head(x)  # ProjectionHead validates NaN/Inf/dim internally

        result: np.ndarray = embedding.numpy().astype(np.float32)

        self._validate_output(result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_output(out: np.ndarray) -> None:
        if out.shape[0] != _OUTPUT_DIM:
            raise ValueError(
                f"Output dimension must be {_OUTPUT_DIM}, got {out.shape[0]}"
            )
        if np.isnan(out).any():
            raise ValueError("Generated fingerprint contains NaN values")
        if np.isinf(out).any():
            raise ValueError("Generated fingerprint contains Inf values")
        norm = float(np.linalg.norm(out))
        if abs(norm - 1.0) > _NORM_TOLERANCE:
            raise ValueError(
                f"Generated fingerprint L2 norm must be ≈ 1.0, got {norm:.6f}"
            )
