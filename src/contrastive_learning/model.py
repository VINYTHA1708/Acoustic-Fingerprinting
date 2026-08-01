"""ProjectionHead — small trainable head over the Fusion Fingerprint.

SDD v4 §5:
    Only the small contrastive head is trained; BEATs and DSP extractors are frozen.

Architecture:
    Input (921) → Linear(512) → ReLU → Linear(256) → L2 normalisation → Output (256)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_INPUT_DIM = 921
_OUTPUT_DIM = 256


class ProjectionHead(nn.Module):
    """Projection head for contrastive fingerprint learning.

    Maps a 921-dimensional Fusion Fingerprint to a 256-dimensional
    L2-normalised embedding.

    Args:
        input_dim: Must be 921 (DSP 153 + BEATs 768).
        output_dim: Must be 256.

    Raises:
        ValueError: If input_dim != 921 or output_dim != 256.
    """

    def __init__(self, input_dim: int = _INPUT_DIM, output_dim: int = _OUTPUT_DIM) -> None:
        if input_dim != _INPUT_DIM:
            raise ValueError(f"input_dim must be {_INPUT_DIM}, got {input_dim}")
        if output_dim != _OUTPUT_DIM:
            raise ValueError(f"output_dim must be {_OUTPUT_DIM}, got {output_dim}")

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalise the input.

        Args:
            x: Float tensor of shape ``(B, 921)`` or ``(921,)``.

        Returns:
            L2-normalised tensor of shape ``(B, 256)`` or ``(256,)``.

        Raises:
            ValueError: If the output contains NaN, Inf, or has wrong dimension.
        """
        out = F.normalize(self.net(x), p=2, dim=-1)
        self._validate_output(out)
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_weights(self, path: str | Path) -> None:
        """Save model weights to *path* using :func:`torch.save`.

        Args:
            path: Destination file path (e.g. ``models/projection_head.pt``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str | Path) -> None:
        """Load model weights from *path* using :func:`torch.load`.

        Args:
            path: Source file path written by :meth:`save_weights`.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")
        self.load_state_dict(torch.load(path, map_location="cpu"))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_output(out: torch.Tensor) -> None:
        if out.shape[-1] != _OUTPUT_DIM:
            raise ValueError(f"Output dimension must be {_OUTPUT_DIM}, got {out.shape[-1]}")
        if torch.isnan(out).any():
            raise ValueError("ProjectionHead output contains NaN values")
        if torch.isinf(out).any():
            raise ValueError("ProjectionHead output contains Inf values")
