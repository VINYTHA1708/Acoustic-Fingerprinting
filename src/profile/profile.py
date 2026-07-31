"""HealthyFingerprintProfile dataclass — statistical summary of healthy recordings."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = {
    "machine_type", "machine_id", "number_of_samples",
    "feature_names", "mean_vector", "std_vector",
    "min_vector", "max_vector", "created_at",
}


@dataclass
class HealthyFingerprintProfile:
    """Statistical summary of healthy acoustic fingerprints for one machine.

    Attributes:
        machine_type: Type of machine (e.g. ``fan``, ``pump``).
        machine_id: Specific machine identifier (e.g. ``id_00``).
        number_of_samples: Number of healthy fingerprints used to build the profile.
        feature_names: Ordered list of DSP feature names.
        mean_vector: Per-feature mean across all healthy fingerprints.
        std_vector: Per-feature standard deviation across all healthy fingerprints.
        min_vector: Per-feature minimum across all healthy fingerprints.
        max_vector: Per-feature maximum across all healthy fingerprints.
        created_at: ISO-8601 UTC timestamp of profile creation.
    """

    machine_type: str
    machine_id: str
    number_of_samples: int
    feature_names: list[str]
    mean_vector: np.ndarray
    std_vector: np.ndarray
    min_vector: np.ndarray
    max_vector: np.ndarray
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Serialise the profile to a JSON-compatible dictionary.

        Returns:
            Dict with all metadata and vectors as plain Python lists.
        """
        return {
            "machine_type": self.machine_type,
            "machine_id": self.machine_id,
            "number_of_samples": self.number_of_samples,
            "feature_names": self.feature_names,
            "mean_vector": self.mean_vector.tolist(),
            "std_vector": self.std_vector.tolist(),
            "min_vector": self.min_vector.tolist(),
            "max_vector": self.max_vector.tolist(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HealthyFingerprintProfile":
        """Reconstruct a ``HealthyFingerprintProfile`` from a serialised dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A fully reconstructed ``HealthyFingerprintProfile`` instance.

        Raises:
            KeyError: If a required field is missing from ``data``.
        """
        missing = _REQUIRED_FIELDS - data.keys()
        if missing:
            raise KeyError(f"Missing required fields in profile dict: {missing}")

        return cls(
            machine_type=data["machine_type"],
            machine_id=data["machine_id"],
            number_of_samples=int(data["number_of_samples"]),
            feature_names=list(data["feature_names"]),
            mean_vector=np.array(data["mean_vector"], dtype=np.float32),
            std_vector=np.array(data["std_vector"], dtype=np.float32),
            min_vector=np.array(data["min_vector"], dtype=np.float32),
            max_vector=np.array(data["max_vector"], dtype=np.float32),
            created_at=data["created_at"],
        )
