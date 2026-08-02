"""LearnedHealthSerializer — save and load LearnedHealthResult to JSON and NPZ."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .learned_health_result import LearnedHealthResult

logger = logging.getLogger(__name__)


class LearnedHealthSerializer:
    """Saves and loads :class:`~learned_health_index.learned_health_result.LearnedHealthResult` objects.

    Supports two formats:

    - **JSON** — human-readable; all fields stored as plain Python types.
    - **NPZ** — compact binary; scalar fields stored as NumPy scalars.
    """

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def save_json(self, result: LearnedHealthResult, path: str | Path) -> None:
        """Serialise a health result to a JSON file.

        Args:
            result: The health result to save.
            path: Destination file path (created or overwritten).

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        logger.info("LearnedHealthResult saved as JSON: %s", path)

    def load_json(self, path: str | Path) -> LearnedHealthResult:
        """Load a health result from a JSON file.

        Args:
            path: Path to the JSON file produced by :meth:`save_json`.

        Returns:
            Reconstructed :class:`~learned_health_index.learned_health_result.LearnedHealthResult`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            KeyError: If required fields are missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON learned health result file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("LearnedHealthResult loaded from JSON: %s", path)
        return LearnedHealthResult.from_dict(data)

    # ------------------------------------------------------------------
    # NPZ
    # ------------------------------------------------------------------

    def save_npz(self, result: LearnedHealthResult, path: str | Path) -> None:
        """Serialise a health result to a compressed NumPy NPZ file.

        Args:
            result: The health result to save.
            path: Destination file path (``.npz`` extension added if absent).

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            machine_type=result.machine_type,
            machine_id=result.machine_id,
            filename=result.filename,
            health_score=result.health_score,
            health_percentage=result.health_percentage,
            health_state=result.health_state,
            normalized_euclidean=result.normalized_euclidean,
            normalized_manhattan=result.normalized_manhattan,
            normalized_cosine=result.normalized_cosine,
            created_at=result.created_at,
        )
        logger.info("LearnedHealthResult saved as NPZ: %s", path)

    def load_npz(self, path: str | Path) -> LearnedHealthResult:
        """Load a health result from an NPZ file.

        Args:
            path: Path to the NPZ file produced by :meth:`save_npz`.

        Returns:
            Reconstructed :class:`~learned_health_index.learned_health_result.LearnedHealthResult`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            suffixed = path.with_suffix(".npz")
            if suffixed.exists():
                path = suffixed
            else:
                raise FileNotFoundError(f"NPZ learned health result file not found: {path}")

        data = np.load(path, allow_pickle=True)
        logger.info("LearnedHealthResult loaded from NPZ: %s", path)

        return LearnedHealthResult.from_dict({
            "machine_type": str(data["machine_type"]),
            "machine_id": str(data["machine_id"]),
            "filename": str(data["filename"]),
            "health_score": float(data["health_score"]),
            "health_percentage": str(data["health_percentage"]),
            "health_state": str(data["health_state"]),
            "normalized_euclidean": float(data["normalized_euclidean"]),
            "normalized_manhattan": float(data["normalized_manhattan"]),
            "normalized_cosine": float(data["normalized_cosine"]),
            "created_at": str(data["created_at"]),
        })
