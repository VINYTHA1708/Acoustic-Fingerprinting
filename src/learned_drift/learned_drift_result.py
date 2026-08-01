"""LearnedDriftResult — output of a single learned fingerprint drift analysis.

Raw metrics compare the current 256-dim embedding directly against the profile
mean vector.

Normalized metrics operate on the z-score vector:

    normalized_vector = (current_embedding - profile.mean_vector) / profile.std_vector

where std == 0 dimensions are treated as zero deviation.

Normalized metrics are the official input to the Health Index module (SDD v4 §7).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = {
    "machine_type", "machine_id", "filename",
    "euclidean_distance", "manhattan_distance", "cosine_similarity",
    "norm_euclidean_distance", "norm_manhattan_distance", "norm_cosine_similarity",
    "normalized_vector",
    "created_at",
}


@dataclass
class LearnedDriftResult:
    """Result of comparing one recording's learned embedding against a healthy profile.

    Attributes:
        machine_type: Type of machine (e.g. ``"pump"``).
        machine_id: Specific machine identifier (e.g. ``"id_00"``).
        filename: Source audio filename of the current recording.

        Raw metrics — distance between current embedding and profile mean:
        euclidean_distance: Euclidean distance (raw).
        manhattan_distance: Manhattan distance (raw).
        cosine_similarity: Cosine similarity (raw).

        Normalized metrics — distance of the z-score vector from zero.
        These are the official inputs to the Health Index module.
        norm_euclidean_distance: Euclidean distance of normalized_vector from zero.
        norm_manhattan_distance: Manhattan distance of normalized_vector from zero.
        norm_cosine_similarity: Cosine similarity of normalized_vector vs uniform direction.

        normalized_vector: The z-score normalized embedding (float32 ndarray, shape (256,)).
        created_at: ISO-8601 UTC timestamp of drift computation.
    """

    machine_type: str
    machine_id: str
    filename: str
    # Raw metrics
    euclidean_distance: float
    manhattan_distance: float
    cosine_similarity: float
    # Normalized metrics (official Health Index input)
    norm_euclidean_distance: float
    norm_manhattan_distance: float
    norm_cosine_similarity: float
    normalized_vector: np.ndarray   # shape (256,), float32
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Serialise the drift result to a JSON-compatible dictionary.

        Returns:
            Dict with all fields; numpy arrays converted to plain Python lists.
        """
        return {
            "machine_type": self.machine_type,
            "machine_id": self.machine_id,
            "filename": self.filename,
            # Raw metrics
            "euclidean_distance": self.euclidean_distance,
            "manhattan_distance": self.manhattan_distance,
            "cosine_similarity": self.cosine_similarity,
            # Normalized metrics
            "norm_euclidean_distance": self.norm_euclidean_distance,
            "norm_manhattan_distance": self.norm_manhattan_distance,
            "norm_cosine_similarity": self.norm_cosine_similarity,
            "normalized_vector": self.normalized_vector.tolist(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearnedDriftResult":
        """Reconstruct a ``LearnedDriftResult`` from a serialised dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A fully reconstructed ``LearnedDriftResult`` instance.

        Raises:
            KeyError: If a required field is missing from ``data``.
        """
        missing = _REQUIRED_FIELDS - data.keys()
        if missing:
            raise KeyError(f"Missing required fields in learned drift result dict: {missing}")

        return cls(
            machine_type=data["machine_type"],
            machine_id=data["machine_id"],
            filename=data["filename"],
            # Raw metrics
            euclidean_distance=float(data["euclidean_distance"]),
            manhattan_distance=float(data["manhattan_distance"]),
            cosine_similarity=float(data["cosine_similarity"]),
            # Normalized metrics
            norm_euclidean_distance=float(data["norm_euclidean_distance"]),
            norm_manhattan_distance=float(data["norm_manhattan_distance"]),
            norm_cosine_similarity=float(data["norm_cosine_similarity"]),
            normalized_vector=np.array(data["normalized_vector"], dtype=np.float32),
            created_at=data["created_at"],
        )
