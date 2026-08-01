"""LearnedDriftSerializer — save and load LearnedDriftResult to JSON and NPZ."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .learned_drift_result import LearnedDriftResult

logger = logging.getLogger(__name__)


class LearnedDriftSerializer:
    """Saves and loads :class:`~learned_drift.learned_drift_result.LearnedDriftResult` objects.

    Supports two formats:

    - **JSON** — human-readable; all metadata and vectors stored as lists.
    - **NPZ** — compact binary; vectors as float32 arrays, metadata as scalars.
    """

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def save_json(self, result: LearnedDriftResult, path: str | Path) -> None:
        """Serialise a drift result to a JSON file.

        Args:
            result: The drift result to save.
            path: Destination file path (created or overwritten).

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        logger.info("LearnedDriftResult saved as JSON: %s", path)

    def load_json(self, path: str | Path) -> LearnedDriftResult:
        """Load a drift result from a JSON file.

        Args:
            path: Path to the JSON file produced by :meth:`save_json`.

        Returns:
            Reconstructed :class:`~learned_drift.learned_drift_result.LearnedDriftResult`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            KeyError: If required fields are missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON learned drift result file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("LearnedDriftResult loaded from JSON: %s", path)
        return LearnedDriftResult.from_dict(data)

    # ------------------------------------------------------------------
    # NPZ
    # ------------------------------------------------------------------

    def save_npz(self, result: LearnedDriftResult, path: str | Path) -> None:
        """Serialise a drift result to a compressed NumPy NPZ file.

        Args:
            result: The drift result to save.
            path: Destination file path (``.npz`` extension added if absent).

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            normalized_vector=result.normalized_vector,
            machine_type=result.machine_type,
            machine_id=result.machine_id,
            filename=result.filename,
            euclidean_distance=result.euclidean_distance,
            manhattan_distance=result.manhattan_distance,
            cosine_similarity=result.cosine_similarity,
            norm_euclidean_distance=result.norm_euclidean_distance,
            norm_manhattan_distance=result.norm_manhattan_distance,
            norm_cosine_similarity=result.norm_cosine_similarity,
            created_at=result.created_at,
        )
        logger.info("LearnedDriftResult saved as NPZ: %s", path)

    def load_npz(self, path: str | Path) -> LearnedDriftResult:
        """Load a drift result from an NPZ file.

        Args:
            path: Path to the NPZ file produced by :meth:`save_npz`.

        Returns:
            Reconstructed :class:`~learned_drift.learned_drift_result.LearnedDriftResult`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            suffixed = path.with_suffix(".npz")
            if suffixed.exists():
                path = suffixed
            else:
                raise FileNotFoundError(f"NPZ learned drift result file not found: {path}")

        data = np.load(path, allow_pickle=True)
        logger.info("LearnedDriftResult loaded from NPZ: %s", path)

        return LearnedDriftResult.from_dict({
            "machine_type": str(data["machine_type"]),
            "machine_id": str(data["machine_id"]),
            "filename": str(data["filename"]),
            "euclidean_distance": float(data["euclidean_distance"]),
            "manhattan_distance": float(data["manhattan_distance"]),
            "cosine_similarity": float(data["cosine_similarity"]),
            "norm_euclidean_distance": float(data["norm_euclidean_distance"]),
            "norm_manhattan_distance": float(data["norm_manhattan_distance"]),
            "norm_cosine_similarity": float(data["norm_cosine_similarity"]),
            "normalized_vector": data["normalized_vector"].tolist(),
            "created_at": str(data["created_at"]),
        })
