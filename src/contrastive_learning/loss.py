"""NTXentLoss — NT-Xent (InfoNCE) loss for contrastive fingerprint learning.

SDD v4 §2 (Version 3):
    Positive pairs: same machine, different recordings.
    Negative pairs: different machine_id or machine_type.
    Single contrastive objective — no identity/health split.

Given N anchor embeddings and N paired embeddings, the 2N vectors form a
similarity matrix.  For each anchor i, its positive is the corresponding
paired embedding i; all other 2N-2 embeddings in the batch are negatives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_EMBEDDING_DIM = 256


class NTXentLoss(nn.Module):
    """NT-Xent (InfoNCE) loss for contrastive fingerprint learning.

    Args:
        temperature: Softmax temperature τ > 0. Lower values produce sharper
                     distributions. Defaults to 0.1.

    Raises:
        ValueError: If temperature <= 0.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss over a batch of paired embeddings.

        Args:
            embeddings_a: L2-normalised anchor embeddings, shape ``(N, 256)``.
            embeddings_b: L2-normalised paired embeddings, shape ``(N, 256)``.

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If shapes differ, embedding dim != 256, batch size < 2,
                        or either tensor contains NaN or Inf.
        """
        self._validate(embeddings_a, embeddings_b)

        n = embeddings_a.shape[0]

        # Concatenate into (2N, 256); first N are anchors, last N are pairs
        z = torch.cat([embeddings_a, embeddings_b], dim=0)  # (2N, 256)

        # Cosine similarity matrix (2N, 2N); embeddings are already L2-normalised
        sim = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive indices: for anchor i (row i), positive is at index i+N;
        # for paired i+N (row i+N), positive is at index i.
        labels = torch.cat([
            torch.arange(n, 2 * n, device=z.device),
            torch.arange(0, n, device=z.device),
        ])  # (2N,)

        loss = F.cross_entropy(sim, labels)
        return loss

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(a: torch.Tensor, b: torch.Tensor) -> None:
        if a.shape != b.shape:
            raise ValueError(
                f"embeddings_a and embeddings_b must have identical shape, "
                f"got {tuple(a.shape)} and {tuple(b.shape)}"
            )
        if a.ndim != 2:
            raise ValueError(f"Embeddings must be 2-D (N, {_EMBEDDING_DIM}), got {a.ndim}-D")
        if a.shape[1] != _EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension must be {_EMBEDDING_DIM}, got {a.shape[1]}"
            )
        if a.shape[0] < 2:
            raise ValueError(
                f"Batch size must be >= 2 for NT-Xent loss, got {a.shape[0]}"
            )
        if torch.isnan(a).any() or torch.isnan(b).any():
            raise ValueError("Embeddings contain NaN values")
        if torch.isinf(a).any() or torch.isinf(b).any():
            raise ValueError("Embeddings contain Inf values")
