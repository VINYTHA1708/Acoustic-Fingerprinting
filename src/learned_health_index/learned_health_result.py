"""LearnedHealthResult — output of a single learned health index computation.

Health score is derived from normalized drift metrics (SDD v4 §8).
The normalized Euclidean distance is the primary anomaly score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

_REQUIRED_FIELDS = {
    "machine_type", "machine_id", "filename",
    "health_score", "health_percentage", "health_state",
    "normalized_euclidean", "normalized_manhattan", "normalized_cosine",
    "created_at",
}


@dataclass
class LearnedHealthResult:
    """Result of computing the health index from a learned drift analysis.

    Attributes:
        machine_type: Type of machine (e.g. ``"pump"``).
        machine_id: Specific machine identifier (e.g. ``"id_00"``).
        filename: Source audio filename of the analyzed recording.

        health_score: Bounded health score in [0, 100].
        health_percentage: Health percentage string (e.g. ``"82.5%"``).
        health_state: Qualitative state — ``EXCELLENT``, ``GOOD``, ``WARNING``, or ``CRITICAL``.

        normalized_euclidean: Normalized Euclidean distance used as primary input.
        normalized_manhattan: Normalized Manhattan distance.
        normalized_cosine: Normalized cosine similarity.

        created_at: ISO-8601 UTC timestamp of health computation.
    """

    machine_type: str
    machine_id: str
    filename: str
    health_score: float
    health_percentage: str
    health_state: str
    normalized_euclidean: float
    normalized_manhattan: float
    normalized_cosine: float
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Serialise the health result to a JSON-compatible dictionary."""
        return {
            "machine_type": self.machine_type,
            "machine_id": self.machine_id,
            "filename": self.filename,
            "health_score": self.health_score,
            "health_percentage": self.health_percentage,
            "health_state": self.health_state,
            "normalized_euclidean": self.normalized_euclidean,
            "normalized_manhattan": self.normalized_manhattan,
            "normalized_cosine": self.normalized_cosine,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearnedHealthResult":
        """Reconstruct a ``LearnedHealthResult`` from a serialised dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A fully reconstructed ``LearnedHealthResult`` instance.

        Raises:
            KeyError: If a required field is missing from ``data``.
        """
        missing = _REQUIRED_FIELDS - data.keys()
        if missing:
            raise KeyError(f"Missing required fields in learned health result dict: {missing}")

        return cls(
            machine_type=data["machine_type"],
            machine_id=data["machine_id"],
            filename=data["filename"],
            health_score=float(data["health_score"]),
            health_percentage=data["health_percentage"],
            health_state=data["health_state"],
            normalized_euclidean=float(data["normalized_euclidean"]),
            normalized_manhattan=float(data["normalized_manhattan"]),
            normalized_cosine=float(data["normalized_cosine"]),
            created_at=data["created_at"],
        )
