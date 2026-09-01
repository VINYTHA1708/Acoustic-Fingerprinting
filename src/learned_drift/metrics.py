"""LearnedDriftMetrics — computes raw and normalized drift metrics for a learned embedding.

SDD v4 §7:
    Raw metrics: compare embedding directly against profile.mean_vector.
    Normalized metrics: operate on the z-score vector:
        z_score_vector = (embedding - mean_vector) / std_vector
    std == 0 dimensions are treated as zero deviation (no division by zero).
"""

from __future__ import annotations

import logging

import numpy as np

from ..learned_profile.learned_profile import LearnedFingerprintProfile

logger = logging.getLogger(__name__)

_STD_FLOOR = 1e-10
_EMBEDDING_DIM = 256


class LearnedDriftMetrics:
    """Computes raw and normalized drift metrics between a learned embedding and a profile.

    All inputs are expected to be 256-dimensional float32 arrays.
    """

    def compute(
        self,
        embedding: np.ndarray,
        profile: LearnedFingerprintProfile,
    ) -> tuple[
        float,
        float,
        float,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        np.ndarray,
    ]:
        """Compute all drift metrics for one embedding against a learned profile.

        Args:
            embedding: 256-dim float32 embedding from the ProjectionHead.
            profile: :class:`~learned_profile.learned_profile.LearnedFingerprintProfile`
                     for the same machine.

        Returns:
            Tuple of:
            - ``cosine_similarity``          (float) — raw
            - ``euclidean_distance``         (float) — raw
            - ``manhattan_distance``         (float) — raw
            - ``z_score_vector``             (float32 ndarray, shape ``(256,)``)
            - ``absolute_difference_vector`` (float32 ndarray, shape ``(256,)``)
            - ``normalized_euclidean_distance`` (float)
            - ``normalized_manhattan_distance`` (float)
            - ``normalized_cosine_similarity``  (float)
            - ``normalized_vector``          (float32 ndarray, shape ``(256,)``)

        Raises:
            ValueError: On any validation failure.
        """
        self._validate(embedding, profile)

        emb = embedding.astype(np.float32)
        mean = profile.mean_vector.astype(np.float32)
        std = profile.std_vector.astype(np.float32)

        # Raw metrics
        cosine = self._cosine(emb, mean)
        euclidean = float(np.linalg.norm(emb - mean))
        manhattan = float(np.sum(np.abs(emb - mean)))

        # Absolute difference
        abs_diff = np.abs(emb - mean).astype(np.float32)

        # Z-score vector (also the normalized_vector)
        safe_std = np.where(std < _STD_FLOOR, 1.0, std)
        z = np.where(std < _STD_FLOOR, 0.0, (emb - mean) / safe_std).astype(np.float32)

        # Normalized metrics
        norm_euclidean = float(np.linalg.norm(z))
        norm_manhattan = float(np.sum(np.abs(z)))
        norm_cosine = self._cosine_vs_uniform(z)

        logger.debug(
            "Learned drift — raw: cosine=%.4f euclid=%.4f manhat=%.4f "
            "| norm: euclid=%.4f manhat=%.4f cosine=%.4f",
            cosine, euclidean, manhattan, norm_euclidean, norm_manhattan, norm_cosine,
        )

        return (
            cosine,
            euclidean,
            manhattan,
            z,           # z_score_vector
            abs_diff,    # absolute_difference_vector
            norm_euclidean,
            norm_manhattan,
            norm_cosine,
            z.copy(),    # normalized_vector (same as z_score_vector)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(embedding: np.ndarray, profile: LearnedFingerprintProfile) -> None:
        """Validate embedding and profile before metric computation."""
        if not isinstance(embedding, np.ndarray):
            raise ValueError("embedding must be a numpy ndarray.")
        if embedding.ndim != 1:
            raise ValueError(
                f"embedding must be one-dimensional, got ndim={embedding.ndim}."
            )
        if embedding.shape[0] != _EMBEDDING_DIM:
            raise ValueError(
                f"embedding must have shape ({_EMBEDDING_DIM},), got {embedding.shape}."
            )
        if np.isnan(embedding).any():
            raise ValueError("embedding must not contain NaN.")
        if np.isinf(embedding).any():
            raise ValueError("embedding must not contain Inf.")

        if profile.embedding_dimension != _EMBEDDING_DIM:
            raise ValueError(
                f"profile.embedding_dimension must be {_EMBEDDING_DIM}, "
                f"got {profile.embedding_dimension}."
            )
        if profile.mean_vector.shape != (_EMBEDDING_DIM,):
            raise ValueError(
                f"profile.mean_vector must have shape ({_EMBEDDING_DIM},), "
                f"got {profile.mean_vector.shape}."
            )
        if profile.std_vector.shape != (_EMBEDDING_DIM,):
            raise ValueError(
                f"profile.std_vector must have shape ({_EMBEDDING_DIM},), "
                f"got {profile.std_vector.shape}."
            )
        if np.isnan(profile.mean_vector).any() or np.isinf(profile.mean_vector).any():
            raise ValueError("profile.mean_vector must not contain NaN or Inf.")
        if np.isnan(profile.std_vector).any() or np.isinf(profile.std_vector).any():
            raise ValueError("profile.std_vector must not contain NaN or Inf.")

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors; returns 0.0 if either has zero norm."""
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
