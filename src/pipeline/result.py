"""Pipeline result dataclasses.

Contains:
    MachineHealthReport — original full pipeline result (MachineHealthPipeline).
    PipelineResult      — end-to-end inference result (InferencePipeline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

_REQUIRED_FIELDS = {
    "machine_type", "machine_id", "filename",
    "dsp_dimension", "beats_dimension", "fusion_dimension", "learned_dimension",
    "euclidean_distance", "manhattan_distance", "cosine_similarity",
    "normalized_euclidean_distance", "normalized_manhattan_distance", "normalized_cosine_similarity",
    "health_score", "health_percentage", "health_state",
    "created_at",
}


@dataclass
class MachineHealthReport:
    """Full pipeline result for one recording.

    Attributes:
        machine_type: Type of machine (e.g. ``"pump"``).
        machine_id: Specific machine identifier (e.g. ``"id_00"``).
        filename: Source audio filename.

        dsp_dimension: Length of the DSP feature vector.
        beats_dimension: Length of the BEATs embedding.
        fusion_dimension: Length of the fused feature vector (DSP + BEATs).
        learned_dimension: Length of the learned embedding from the ProjectionHead.

        euclidean_distance: Raw Euclidean distance between embedding and profile mean.
        manhattan_distance: Raw Manhattan distance between embedding and profile mean.
        cosine_similarity: Raw cosine similarity between embedding and profile mean.

        normalized_euclidean_distance: Normalized Euclidean distance (z-score vector norm).
        normalized_manhattan_distance: Normalized Manhattan distance (z-score vector L1).
        normalized_cosine_similarity: Normalized cosine similarity.

        health_score: Bounded health score in [0, 100].
        health_percentage: Health percentage string (e.g. ``"82.5%"``).
        health_state: Qualitative state — ``EXCELLENT``, ``GOOD``, ``WARNING``, or ``CRITICAL``.

        created_at: ISO-8601 UTC timestamp of report creation.
    """

    machine_type: str
    machine_id: str
    filename: str
    # Dimensions
    dsp_dimension: int
    beats_dimension: int
    fusion_dimension: int
    learned_dimension: int
    # Raw drift metrics
    euclidean_distance: float
    manhattan_distance: float
    cosine_similarity: float
    # Normalized drift metrics
    normalized_euclidean_distance: float
    normalized_manhattan_distance: float
    normalized_cosine_similarity: float
    # Health
    health_score: float
    health_percentage: str
    health_state: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Serialise the report to a JSON-compatible dictionary."""
        return {
            "machine_type": self.machine_type,
            "machine_id": self.machine_id,
            "filename": self.filename,
            "dsp_dimension": self.dsp_dimension,
            "beats_dimension": self.beats_dimension,
            "fusion_dimension": self.fusion_dimension,
            "learned_dimension": self.learned_dimension,
            "euclidean_distance": self.euclidean_distance,
            "manhattan_distance": self.manhattan_distance,
            "cosine_similarity": self.cosine_similarity,
            "normalized_euclidean_distance": self.normalized_euclidean_distance,
            "normalized_manhattan_distance": self.normalized_manhattan_distance,
            "normalized_cosine_similarity": self.normalized_cosine_similarity,
            "health_score": self.health_score,
            "health_percentage": self.health_percentage,
            "health_state": self.health_state,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MachineHealthReport":
        """Reconstruct a ``MachineHealthReport`` from a serialised dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A fully reconstructed ``MachineHealthReport`` instance.

        Raises:
            KeyError: If a required field is missing from ``data``.
        """
        missing = _REQUIRED_FIELDS - data.keys()
        if missing:
            raise KeyError(f"Missing required fields in machine health report dict: {missing}")

        return cls(
            machine_type=data["machine_type"],
            machine_id=data["machine_id"],
            filename=data["filename"],
            dsp_dimension=int(data["dsp_dimension"]),
            beats_dimension=int(data["beats_dimension"]),
            fusion_dimension=int(data["fusion_dimension"]),
            learned_dimension=int(data["learned_dimension"]),
            euclidean_distance=float(data["euclidean_distance"]),
            manhattan_distance=float(data["manhattan_distance"]),
            cosine_similarity=float(data["cosine_similarity"]),
            normalized_euclidean_distance=float(data["normalized_euclidean_distance"]),
            normalized_manhattan_distance=float(data["normalized_manhattan_distance"]),
            normalized_cosine_similarity=float(data["normalized_cosine_similarity"]),
            health_score=float(data["health_score"]),
            health_percentage=data["health_percentage"],
            health_state=data["health_state"],
            created_at=data["created_at"],
        )


# ---------------------------------------------------------------------------
# PipelineResult — end-to-end InferencePipeline result
# ---------------------------------------------------------------------------

_PIPELINE_REQUIRED_FIELDS = {
    "machine_type", "machine_id", "filename",
    "dsp_dimension", "beats_dimension", "fusion_dimension", "embedding_dimension",
    "raw_euclidean", "raw_manhattan", "raw_cosine",
    "normalized_euclidean", "normalized_manhattan", "normalized_cosine",
    "health_score", "health_percentage", "health_state",
    "created_at",
}


@dataclass
class PipelineResult:
    """End-to-end inference result for one recording.

    Produced by :class:`~pipeline.pipeline.InferencePipeline`.

    Attributes:
        machine_type: Type of machine (e.g. ``"pump"``).
        machine_id: Specific machine identifier (e.g. ``"id_00"``).
        filename: Source audio filename.

        dsp_dimension: Length of the DSP feature vector (153).
        beats_dimension: Length of the BEATs embedding (768).
        fusion_dimension: Length of the fused feature vector (921).
        embedding_dimension: Length of the learned embedding from the ProjectionHead (256).

        raw_euclidean: Raw Euclidean distance between embedding and profile mean.
        raw_manhattan: Raw Manhattan distance between embedding and profile mean.
        raw_cosine: Raw cosine similarity between embedding and profile mean.

        normalized_euclidean: Normalized Euclidean distance (z-score vector norm).
        normalized_manhattan: Normalized Manhattan distance (z-score vector L1).
        normalized_cosine: Normalized cosine similarity.

        health_score: Bounded health score in [0, 100].
        health_percentage: Health percentage string (e.g. ``"82.5%"``).
        health_state: Qualitative state — ``EXCELLENT``, ``GOOD``, ``WARNING``, or ``CRITICAL``.

        created_at: ISO-8601 UTC timestamp of result creation.
    """

    machine_type: str
    machine_id: str
    filename: str
    # Dimensions
    dsp_dimension: int
    beats_dimension: int
    fusion_dimension: int
    embedding_dimension: int
    # Raw drift metrics
    raw_euclidean: float
    raw_manhattan: float
    raw_cosine: float
    # Normalized drift metrics
    normalized_euclidean: float
    normalized_manhattan: float
    normalized_cosine: float
    # Health
    health_score: float
    health_percentage: str
    health_state: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Serialise the result to a JSON-compatible dictionary."""
        return {
            "machine_type": self.machine_type,
            "machine_id": self.machine_id,
            "filename": self.filename,
            "dsp_dimension": self.dsp_dimension,
            "beats_dimension": self.beats_dimension,
            "fusion_dimension": self.fusion_dimension,
            "embedding_dimension": self.embedding_dimension,
            "raw_euclidean": self.raw_euclidean,
            "raw_manhattan": self.raw_manhattan,
            "raw_cosine": self.raw_cosine,
            "normalized_euclidean": self.normalized_euclidean,
            "normalized_manhattan": self.normalized_manhattan,
            "normalized_cosine": self.normalized_cosine,
            "health_score": self.health_score,
            "health_percentage": self.health_percentage,
            "health_state": self.health_state,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineResult":
        """Reconstruct a ``PipelineResult`` from a serialised dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A fully reconstructed ``PipelineResult`` instance.

        Raises:
            KeyError: If a required field is missing from ``data``.
        """
        missing = _PIPELINE_REQUIRED_FIELDS - data.keys()
        if missing:
            raise KeyError(f"Missing required fields in pipeline result dict: {missing}")

        return cls(
            machine_type=data["machine_type"],
            machine_id=data["machine_id"],
            filename=data["filename"],
            dsp_dimension=int(data["dsp_dimension"]),
            beats_dimension=int(data["beats_dimension"]),
            fusion_dimension=int(data["fusion_dimension"]),
            embedding_dimension=int(data["embedding_dimension"]),
            raw_euclidean=float(data["raw_euclidean"]),
            raw_manhattan=float(data["raw_manhattan"]),
            raw_cosine=float(data["raw_cosine"]),
            normalized_euclidean=float(data["normalized_euclidean"]),
            normalized_manhattan=float(data["normalized_manhattan"]),
            normalized_cosine=float(data["normalized_cosine"]),
            health_score=float(data["health_score"]),
            health_percentage=data["health_percentage"],
            health_state=data["health_state"],
            created_at=data["created_at"],
        )
