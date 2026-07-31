"""ProfileStatistics: aggregate statistics over a collection of feature vectors."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ProfileStatistics:
    """Computes descriptive statistics over a set of feature vectors.

    All vectors must share the same length. At least one vector is required.
    """

    def compute(
        self, vectors: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute mean, std, min, and max across a list of feature vectors.

        Args:
            vectors: List of 1-D float32 numpy arrays, all the same length.

        Returns:
            A tuple of ``(mean, std, min, max)`` — each a 1-D float32 array
            of the same length as the input vectors.

        Raises:
            ValueError: If ``vectors`` is empty or vectors have inconsistent lengths.
        """
        self._validate(vectors)
        matrix = np.stack(vectors, axis=0).astype(np.float32)  # (N, D)

        return (
            matrix.mean(axis=0),
            matrix.std(axis=0),
            matrix.min(axis=0),
            matrix.max(axis=0),
        )

    @staticmethod
    def _validate(vectors: list[np.ndarray]) -> None:
        """Validate that vectors is non-empty and all share the same length.

        Args:
            vectors: List of feature vectors to validate.

        Raises:
            ValueError: If the list is empty or lengths are inconsistent.
        """
        if not vectors:
            raise ValueError("At least one feature vector is required to compute statistics.")

        expected_len = len(vectors[0])
        for i, v in enumerate(vectors[1:], start=1):
            if len(v) != expected_len:
                raise ValueError(
                    f"Feature vector at index {i} has length {len(v)}, "
                    f"expected {expected_len}."
                )
