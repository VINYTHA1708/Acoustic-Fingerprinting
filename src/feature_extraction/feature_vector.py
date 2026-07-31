"""FeatureVectorBuilder: converts a named feature dict into an ordered numpy vector."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class FeatureVectorBuilder:
    """Converts a DSP feature dictionary into a deterministic, ordered numpy vector.

    The builder is stateless — the key order is determined by the dict's
    insertion order (guaranteed in Python 3.7+), which is fixed by
    :class:`~feature_extraction.extractor.FeatureExtractor`.

    This design allows future concatenation with BEATs embeddings:
    ``np.concatenate([dsp_vector, beats_embedding])``.
    """

    def build(self, features: dict[str, float]) -> tuple[np.ndarray, list[str]]:
        """Convert a feature dict to a 1-D float32 numpy vector.

        Args:
            features: Flat dict of ``{feature_name: float_value}`` as returned
                by :class:`~feature_extraction.extractor.FeatureExtractor`.

        Returns:
            A tuple of:
            - ``vector``: 1-D float32 numpy array of length ``len(features)``.
            - ``names``: List of feature names in the same order as ``vector``.

        Raises:
            ValueError: If ``features`` is empty.
        """
        if not features:
            raise ValueError("Feature dict is empty — nothing to build a vector from.")

        names = list(features.keys())
        vector = np.array(list(features.values()), dtype=np.float32)

        logger.debug("Feature vector built: length=%d.", len(vector))
        return vector, names
