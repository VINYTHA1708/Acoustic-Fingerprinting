"""DriftResult dataclass — output of a single fingerprint drift analysis.

V1 architecture note
---------------------
Raw metrics (euclidean_distance, manhattan_distance, cosine_similarity) compare
the current feature vector directly against the profile mean vector.

Normalized metrics (norm_euclidean_distance, norm_manhattan_distance,
norm_cosine_similarity) operate on the z-score normalized vector:

    normalized_vector = (current - profile.mean) / profile.std

where std == 0 dimensions are treated as zero deviation.

Validation experiments confirmed that normalized distances provide more
consistent separation between healthy and abnormal recordings across machine
IDs than raw distances alone. The normalized representation is therefore the
official input to the future Health Index module (Version 1).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = {
    "machine_type", "machine_id", "filename", "feature_names",
    "cosine_similarity", "euclidean_distance", "manhattan_distance",
    "z_score_vector", "absolute_difference_vector",
    "norm_euclidean_distance", "norm_manhattan_distance", "norm_cosine_similarity",
    "normalized_vector",
    "timestamp",
}


@dataclass
class DriftResult:
    """Result of comparing one recording's fingerprint against a healthy profile.

    Attributes:
        machine_type: Type of machine (e.g. ``fan``, ``pump``).
        machine_id: Specific machine identifier (e.g. ``id_00``).
        filename: Source audio filename of the current recording.
        feature_names: Ordered list of DSP feature names.

        Raw metrics — distance between current vector and profile mean vector:
        cosine_similarity: Cosine similarity (raw).
        euclidean_distance: Euclidean distance (raw).
        manhattan_distance: Manhattan distance (raw).

        Per-feature arrays:
        z_score_vector: Per-feature z-score: ``(current - mean) / std``.
        absolute_difference_vector: Per-feature absolute difference: ``|current - mean|``.

        Normalized metrics — distance of the z-score vector from the zero vector.
        These are the official inputs to the Health Index module.
        norm_euclidean_distance: Euclidean distance of normalized_vector from zero.
        norm_manhattan_distance: Manhattan distance of normalized_vector from zero.
        norm_cosine_similarity: Cosine similarity of normalized_vector vs mean direction.
        normalized_vector: The z-score normalized feature vector (float32 ndarray).

        timestamp: ISO-8601 UTC timestamp of drift computation.
    """

    machine_type: str
    machine_id: str
    filename: str
    feature_names: list[str]
    # Raw metrics
    cosine_similarity: float
    euclidean_distance: float
    manhattan_distance: float
    # Per-feature arrays
    z_score_vector: np.ndarray
    absolute_difference_vector: np.ndarray
    # Normalized metrics (official Health Index input — see module docstring)
    norm_euclidean_distance: float
    norm_manhattan_distance: float
    norm_cosine_similarity: float
    normalized_vector: np.ndarray
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Serialise the drift result to a JSON-compatible dictionary.

        Returns:
            Dict with all fields; numpy arrays converted to plain Python lists.
        """
        return {
            "machine_type": self.machine_type,
            "machine_id": self.machine_id,
            "filename": self.filename,
            "feature_names": self.feature_names,
            # Raw metrics
            "cosine_similarity": self.cosine_similarity,
            "euclidean_distance": self.euclidean_distance,
            "manhattan_distance": self.manhattan_distance,
            # Per-feature arrays
            "z_score_vector": self.z_score_vector.tolist(),
            "absolute_difference_vector": self.absolute_difference_vector.tolist(),
            # Normalized metrics
            "norm_euclidean_distance": self.norm_euclidean_distance,
            "norm_manhattan_distance": self.norm_manhattan_distance,
            "norm_cosine_similarity": self.norm_cosine_similarity,
            "normalized_vector": self.normalized_vector.tolist(),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DriftResult":
        """Reconstruct a ``DriftResult`` from a serialised dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A fully reconstructed ``DriftResult`` instance.

        Raises:
            KeyError: If a required field is missing from ``data``.
        """
        missing = _REQUIRED_FIELDS - data.keys()
        if missing:
            raise KeyError(f"Missing required fields in drift result dict: {missing}")

        return cls(
            machine_type=data["machine_type"],
            machine_id=data["machine_id"],
            filename=data["filename"],
            feature_names=list(data["feature_names"]),
            # Raw metrics
            cosine_similarity=float(data["cosine_similarity"]),
            euclidean_distance=float(data["euclidean_distance"]),
            manhattan_distance=float(data["manhattan_distance"]),
            # Per-feature arrays
            z_score_vector=np.array(data["z_score_vector"], dtype=np.float32),
            absolute_difference_vector=np.array(data["absolute_difference_vector"], dtype=np.float32),
            # Normalized metrics
            norm_euclidean_distance=float(data["norm_euclidean_distance"]),
            norm_manhattan_distance=float(data["norm_manhattan_distance"]),
            norm_cosine_similarity=float(data["norm_cosine_similarity"]),
            normalized_vector=np.array(data["normalized_vector"], dtype=np.float32),
            timestamp=data["timestamp"],
        )
