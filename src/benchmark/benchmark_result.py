"""BenchmarkResult — per-recording timing and dimension report from PipelineBenchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

_REQUIRED_FIELDS = {
    "machine_type", "machine_id", "filename",
    "preprocessing_time", "dsp_time", "beats_time", "fusion_time",
    "projection_time", "drift_time", "health_time", "total_time",
    "cache_hit",
    "dsp_dimension", "beats_dimension", "fusion_dimension", "embedding_dimension",
    "created_at",
}


@dataclass
class BenchmarkResult:
    """Per-recording timing and dimension report produced by :class:`~benchmark.benchmark.PipelineBenchmark`.

    All ``*_time`` fields are in seconds as returned by ``time.perf_counter()``.

    Attributes:
        machine_type: Type of machine (e.g. ``"pump"``).
        machine_id: Specific machine identifier (e.g. ``"id_00"``).
        filename: Source audio filename.

        preprocessing_time: Time for audio preprocessing (load → resample → normalize).
        dsp_time: Time for DSP feature extraction.
        beats_time: Time for BEATs encoding.
        fusion_time: Time for fusion vector construction.
        projection_time: Time for ProjectionHead inference.
        drift_time: Time for learned drift metric computation.
        health_time: Time for health score computation.
        total_time: Wall-clock time for the entire inference call.

        cache_hit: True if the fused vector was loaded from disk cache.

        dsp_dimension: Length of the DSP feature vector (153).
        beats_dimension: Length of the BEATs embedding (768).
        fusion_dimension: Length of the fused feature vector (921).
        embedding_dimension: Length of the learned embedding (256).

        created_at: ISO-8601 UTC timestamp of benchmark creation.
    """

    machine_type: str
    machine_id: str
    filename: str
    # Stage times (seconds)
    preprocessing_time: float
    dsp_time: float
    beats_time: float
    fusion_time: float
    projection_time: float
    drift_time: float
    health_time: float
    total_time: float
    # Cache
    cache_hit: bool
    # Dimensions
    dsp_dimension: int
    beats_dimension: int
    fusion_dimension: int
    embedding_dimension: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Serialise the result to a JSON-compatible dictionary."""
        return {
            "machine_type": self.machine_type,
            "machine_id": self.machine_id,
            "filename": self.filename,
            "preprocessing_time": self.preprocessing_time,
            "dsp_time": self.dsp_time,
            "beats_time": self.beats_time,
            "fusion_time": self.fusion_time,
            "projection_time": self.projection_time,
            "drift_time": self.drift_time,
            "health_time": self.health_time,
            "total_time": self.total_time,
            "cache_hit": self.cache_hit,
            "dsp_dimension": self.dsp_dimension,
            "beats_dimension": self.beats_dimension,
            "fusion_dimension": self.fusion_dimension,
            "embedding_dimension": self.embedding_dimension,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkResult":
        """Reconstruct a ``BenchmarkResult`` from a serialised dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A fully reconstructed ``BenchmarkResult`` instance.

        Raises:
            KeyError: If a required field is missing from ``data``.
        """
        missing = _REQUIRED_FIELDS - data.keys()
        if missing:
            raise KeyError(f"Missing required fields in benchmark result dict: {missing}")

        return cls(
            machine_type=data["machine_type"],
            machine_id=data["machine_id"],
            filename=data["filename"],
            preprocessing_time=float(data["preprocessing_time"]),
            dsp_time=float(data["dsp_time"]),
            beats_time=float(data["beats_time"]),
            fusion_time=float(data["fusion_time"]),
            projection_time=float(data["projection_time"]),
            drift_time=float(data["drift_time"]),
            health_time=float(data["health_time"]),
            total_time=float(data["total_time"]),
            cache_hit=bool(data["cache_hit"]),
            dsp_dimension=int(data["dsp_dimension"]),
            beats_dimension=int(data["beats_dimension"]),
            fusion_dimension=int(data["fusion_dimension"]),
            embedding_dimension=int(data["embedding_dimension"]),
            created_at=data["created_at"],
        )
