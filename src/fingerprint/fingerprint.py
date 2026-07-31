"""AcousticFingerprint dataclass — the core data structure of the fingerprint module."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AcousticFingerprint:
    """Represents the acoustic fingerprint of a single machine recording.

    Attributes:
        machine_type: Type of machine (e.g. ``fan``, ``pump``).
        machine_id: Specific machine identifier (e.g. ``id_00``).
        label: Recording condition — ``normal`` or ``abnormal``.
        filename: Source audio filename.
        sample_rate: Sample rate of the source recording in Hz.
        feature_names: Ordered list of DSP feature names.
        feature_vector: 1-D float32 numpy array of DSP feature values.
            Order matches ``feature_names`` exactly.
        created_at: ISO-8601 UTC timestamp of fingerprint creation.
    """

    machine_type: str
    machine_id: str
    label: str
    filename: str
    sample_rate: int
    feature_names: list[str]
    feature_vector: np.ndarray
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Serialise the fingerprint to a JSON-compatible dictionary.

        Returns:
            Dict with all metadata fields and the feature vector as a
            plain Python list of floats.
        """
        return {
            "machine_type": self.machine_type,
            "machine_id": self.machine_id,
            "label": self.label,
            "filename": self.filename,
            "sample_rate": self.sample_rate,
            "feature_names": self.feature_names,
            "feature_vector": self.feature_vector.tolist(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AcousticFingerprint":
        """Reconstruct an ``AcousticFingerprint`` from a serialised dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A fully reconstructed ``AcousticFingerprint`` instance.

        Raises:
            KeyError: If a required field is missing from ``data``.
            ValueError: If ``feature_names`` and ``feature_vector`` lengths differ.
        """
        required = {"machine_type", "machine_id", "label", "filename",
                    "sample_rate", "feature_names", "feature_vector", "created_at"}
        missing = required - data.keys()
        if missing:
            raise KeyError(f"Missing required fields in fingerprint dict: {missing}")

        vector = np.array(data["feature_vector"], dtype=np.float32)
        names: list[str] = data["feature_names"]

        if len(names) != len(vector):
            raise ValueError(
                f"feature_names length ({len(names)}) != "
                f"feature_vector length ({len(vector)})."
            )

        return cls(
            machine_type=data["machine_type"],
            machine_id=data["machine_id"],
            label=data["label"],
            filename=data["filename"],
            sample_rate=int(data["sample_rate"]),
            feature_names=names,
            feature_vector=vector,
            created_at=data["created_at"],
        )
