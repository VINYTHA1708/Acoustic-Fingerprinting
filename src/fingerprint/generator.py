"""FingerprintGenerator: validates inputs and builds AcousticFingerprint instances."""

import logging

import numpy as np

from dataset.metadata import AudioMetadata
from .fingerprint import AcousticFingerprint

logger = logging.getLogger(__name__)

_REQUIRED_METADATA_FIELDS = ("machine_type", "machine_id", "label", "filename")


class FingerprintGenerator:
    """Builds an :class:`~fingerprint.fingerprint.AcousticFingerprint` from DSP features
    and recording metadata.

    Accepts either:
    - A raw feature dict ``{name: float}`` (as returned by ``FeatureExtractor``), or
    - A pre-built ``(vector, names)`` tuple (as returned by ``FeatureVectorBuilder``).
    """

    def generate(
        self,
        features: dict[str, float] | tuple[np.ndarray, list[str]],
        metadata: AudioMetadata,
        sample_rate: int,
    ) -> AcousticFingerprint:
        """Build an ``AcousticFingerprint`` from features and metadata.

        Args:
            features: Either a ``{feature_name: value}`` dict or a
                ``(vector, names)`` tuple from ``FeatureVectorBuilder``.
            metadata: ``AudioMetadata`` instance from the dataset module.
            sample_rate: Sample rate of the preprocessed waveform in Hz.

        Returns:
            A validated :class:`~fingerprint.fingerprint.AcousticFingerprint`.

        Raises:
            TypeError: If ``features`` is not a dict or a 2-tuple.
            ValueError: If the feature vector contains NaN/Inf values,
                is empty, or ``feature_names`` length mismatches the vector.
        """
        self._validate_metadata(metadata)
        vector, names = self._resolve_features(features)
        self._validate_vector(vector, names)

        fp = AcousticFingerprint(
            machine_type=metadata.machine_type,
            machine_id=metadata.machine_id,
            label=metadata.label,
            filename=metadata.filename,
            sample_rate=sample_rate,
            feature_names=names,
            feature_vector=vector,
        )
        logger.info(
            "Fingerprint created — machine=%s/%s label=%s features=%d",
            fp.machine_type, fp.machine_id, fp.label, len(fp.feature_names),
        )
        return fp

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_metadata(metadata: AudioMetadata) -> None:
        """Raise ValueError if any required metadata field is empty.

        Args:
            metadata: ``AudioMetadata`` to validate.

        Raises:
            ValueError: If a required field is missing or blank.
        """
        for field in _REQUIRED_METADATA_FIELDS:
            value = getattr(metadata, field, None)
            if not value:
                raise ValueError(f"Metadata field '{field}' is missing or empty.")

    @staticmethod
    def _resolve_features(
        features: dict[str, float] | tuple[np.ndarray, list[str]],
    ) -> tuple[np.ndarray, list[str]]:
        """Normalise the ``features`` argument to a ``(vector, names)`` pair.

        Args:
            features: Dict or ``(vector, names)`` tuple.

        Returns:
            ``(float32 ndarray, list[str])`` pair.

        Raises:
            TypeError: If the input type is not supported.
            ValueError: If the feature dict or tuple is empty.
        """
        if isinstance(features, dict):
            if not features:
                raise ValueError("Feature dict is empty.")
            names = list(features.keys())
            vector = np.array(list(features.values()), dtype=np.float32)
            return vector, names

        if isinstance(features, tuple) and len(features) == 2:
            vector, names = features
            if len(vector) == 0:
                raise ValueError("Feature vector is empty.")
            return np.asarray(vector, dtype=np.float32), list(names)

        raise TypeError(
            "features must be a dict[str, float] or a (ndarray, list[str]) tuple; "
            f"got {type(features).__name__}."
        )

    @staticmethod
    def _validate_vector(vector: np.ndarray, names: list[str]) -> None:
        """Validate the feature vector for length, NaN, and Inf.

        Args:
            vector: Float32 feature vector.
            names: Corresponding feature names.

        Raises:
            ValueError: On length mismatch, NaN, or Inf values.
        """
        if len(names) != len(vector):
            raise ValueError(
                f"feature_names length ({len(names)}) != vector length ({len(vector)})."
            )
        if np.any(np.isnan(vector)):
            raise ValueError("Feature vector contains NaN values.")
        if np.any(np.isinf(vector)):
            raise ValueError("Feature vector contains infinite values.")
