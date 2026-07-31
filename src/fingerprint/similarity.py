"""FingerprintSimilarity: distance and similarity metrics between AcousticFingerprints."""

import logging

import numpy as np

from .fingerprint import AcousticFingerprint

logger = logging.getLogger(__name__)

_NORM_FLOOR = 1e-10


class FingerprintSimilarity:
    """Computes similarity and distance metrics between two ``AcousticFingerprint`` objects."""

    def cosine_similarity(self, a: AcousticFingerprint, b: AcousticFingerprint) -> float:
        """Compute cosine similarity between two fingerprint vectors.

        Args:
            a: First fingerprint.
            b: Second fingerprint.

        Returns:
            Cosine similarity in the range [-1, 1]. Returns 0.0 if either
            vector has zero norm.
        """
        va, vb = self._validated_pair(a, b)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a < _NORM_FLOOR or norm_b < _NORM_FLOOR:
            logger.warning("One or both fingerprint vectors have near-zero norm; returning 0.0.")
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    def euclidean_distance(self, a: AcousticFingerprint, b: AcousticFingerprint) -> float:
        """Compute Euclidean (L2) distance between two fingerprint vectors.

        Args:
            a: First fingerprint.
            b: Second fingerprint.

        Returns:
            Non-negative float Euclidean distance.
        """
        va, vb = self._validated_pair(a, b)
        return float(np.linalg.norm(va - vb))

    def manhattan_distance(self, a: AcousticFingerprint, b: AcousticFingerprint) -> float:
        """Compute Manhattan (L1) distance between two fingerprint vectors.

        Args:
            a: First fingerprint.
            b: Second fingerprint.

        Returns:
            Non-negative float Manhattan distance.
        """
        va, vb = self._validated_pair(a, b)
        return float(np.sum(np.abs(va - vb)))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validated_pair(
        a: AcousticFingerprint, b: AcousticFingerprint
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the two feature vectors after validating equal length.

        Args:
            a: First fingerprint.
            b: Second fingerprint.

        Returns:
            Tuple of ``(vector_a, vector_b)`` as float32 arrays.

        Raises:
            ValueError: If the vectors have different lengths.
        """
        if len(a.feature_vector) != len(b.feature_vector):
            raise ValueError(
                f"Vector length mismatch: {len(a.feature_vector)} vs {len(b.feature_vector)}."
            )
        return a.feature_vector.astype(np.float32), b.feature_vector.astype(np.float32)
