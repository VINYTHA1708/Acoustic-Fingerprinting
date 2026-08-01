"""LearnedDriftMetrics — computes raw and normalized drift metrics for a learned embedding.

SDD v4 §7:
    Raw metrics: compare current_embedding directly against profile.mean_vector.
    Normalized metrics: operate on the z-score vector:
        normalized_vector = (current_embedding - mean_vector) / std_vector
    std == 0 dimensions are treated as zero deviation (no division by zero).
"""

from __future__ import annotations

import logging

import numpy as np

from ..learned_profile.learned_profile import LearnedFingerprintProfile

logger = logging.getLogger(__name__)

_STD_FLOOR = 1e-10


class LearnedDriftMetrics:
    """Computes raw and normalized drift metrics between a learned embedding and a profile.

    All inputs are expected to be 256-dimensional float32 arrays.
    """

    def compute(
        self,
        current_embedding: np.ndarray,
        profile: LearnedFingerprintProfile,
    ) -> tuple[float, float, float, float, float, float, np.ndarray]:
        """Compute all drift metrics for one embedding against a learned profile.

        Args:
            current_embedding: 256-dim float32 embedding from the ProjectionHead.
            profile: :class:`~learned_profile.learned_profile.LearnedFingerprintProfile`
                     for the same machine.

        Returns:
            Tuple of:
            - ``euclidean_distance``      (float) — raw
            - ``manhattan_distance``      (float) — raw
            - ``cosine_similarity``       (float) — raw
            - ``norm_euclidean_distance`` (float) — normalized
            - ``norm_manhattan_distance`` (float) — normalized
            - ``norm_cosine_similarity``  (float) — normalized
            - ``normalized_vector``       (float32 ndarray, shape ``(256,)``)

        Raises:
            ValueError: If ``current_embedding`` dimension does not match the profile.
        """
        current = current_embedding.astype(np.float32)
        mean = profile.mean_vector.astype(np.float32)
        std = profile.std_vector.astype(np.float32)

        if current.shape[0] != mean.shape[0]:
            raise ValueError(
                f"Embedding dimension {current.shape[0]} does not match "
                f"profile dimension {mean.shape[0]}."
            )

        # --- Raw metrics ---
        euclid = float(np.linalg.norm(current - mean))
        manhat = float(np.sum(np.abs(current - mean)))
        cosine = self._cosine(current, mean)

        # --- Normalized vector ---
        safe_std = np.where(std < _STD_FLOOR, 1.0, std)
        norm_vec = np.where(std < _STD_FLOOR, 0.0, (current - mean) / safe_std).astype(np.float32)

        # --- Normalized metrics (z-score vector vs zero) ---
        norm_euclid = float(np.linalg.norm(norm_vec))
        norm_manhat = float(np.sum(np.abs(norm_vec)))
        norm_cosine = self._cosine_vs_uniform(norm_vec)

        logger.debug(
            "Learned drift — raw: euclid=%.4f manhat=%.4f cosine=%.4f "
            "| norm: euclid=%.4f manhat=%.4f cosine=%.4f",
            euclid, manhat, cosine, norm_euclid, norm_manhat, norm_cosine,
        )
        return euclid, manhat, cosine, norm_euclid, norm_manhat, norm_cosine, norm_vec

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors; returns 0.0 if either is zero."""
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _cosine_vs_uniform(z: np.ndarray) -> float:
        """Cosine similarity of z-score vector against the all-ones (uniform) direction."""
        norm_z = float(np.linalg.norm(z))
        if norm_z == 0.0:
            return 0.0
        ones = np.ones_like(z)
        return float(np.dot(z, ones) / (norm_z * float(np.linalg.norm(ones))))
