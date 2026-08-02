"""ExplanationResult — dataclass for a single rule-based anomaly explanation.

Produced by :class:`~explainability.explainer.ExplainabilityEngine` from a
:class:`~learned_drift.learned_drift_result.LearnedDriftResult` and a
:class:`~learned_health_index.learned_health_result.LearnedHealthResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

_REQUIRED_FIELDS = {
    "machine_type", "machine_id", "filename",
    "health_score", "health_state",
    "raw_euclidean", "normalized_euclidean",
    "summary", "possible_causes", "recommendation",
    "created_at",
}


@dataclass
class ExplanationResult:
    """Human-readable explanation for one anomaly detection result.

    Attributes:
        machine_type: Type of machine (e.g. ``"pump"``).
        machine_id: Specific machine identifier (e.g. ``"id_00"``).
        filename: Source audio filename of the analyzed recording.

        health_score: Bounded health score in [0, 100].
        health_state: Qualitative state — ``EXCELLENT``, ``GOOD``, ``WARNING``,
                      or ``CRITICAL``.

        raw_euclidean: Raw Euclidean distance between embedding and profile mean.
        normalized_euclidean: Normalized Euclidean distance (z-score vector norm).

        summary: One-sentence description of the machine's current condition.
        possible_causes: List of potential root causes; empty for healthy states.
        recommendation: Suggested action for the operator.

        created_at: ISO-8601 UTC timestamp of explanation generation.
    """

    machine_type: str
    machine_id: str
    filename: str
    health_score: float
    health_state: str
    raw_euclidean: float
    normalized_euclidean: float
    summary: str
    possible_causes: list[str]
    recommendation: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Serialise the explanation to a JSON-compatible dictionary.

        Returns:
            Dict with all fields; ``possible_causes`` is a plain Python list.
        """
        return {
            "machine_type": self.machine_type,
            "machine_id": self.machine_id,
            "filename": self.filename,
            "health_score": self.health_score,
            "health_state": self.health_state,
            "raw_euclidean": self.raw_euclidean,
            "normalized_euclidean": self.normalized_euclidean,
            "summary": self.summary,
            "possible_causes": self.possible_causes,
            "recommendation": self.recommendation,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExplanationResult":
        """Reconstruct an ``ExplanationResult`` from a serialised dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A fully reconstructed ``ExplanationResult`` instance.

        Raises:
            KeyError: If a required field is missing from ``data``.
        """
        missing = _REQUIRED_FIELDS - data.keys()
        if missing:
            raise KeyError(f"Missing required fields in explanation result dict: {missing}")

        return cls(
            machine_type=data["machine_type"],
            machine_id=data["machine_id"],
            filename=data["filename"],
            health_score=float(data["health_score"]),
            health_state=data["health_state"],
            raw_euclidean=float(data["raw_euclidean"]),
            normalized_euclidean=float(data["normalized_euclidean"]),
            summary=data["summary"],
            possible_causes=list(data["possible_causes"]),
            recommendation=data["recommendation"],
            created_at=data["created_at"],
        )
