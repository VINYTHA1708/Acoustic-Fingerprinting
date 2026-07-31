"""DriftSerializer: save and load DriftResult to JSON and NPZ."""

import json
import logging
from pathlib import Path

import numpy as np

from .drift_result import DriftResult

logger = logging.getLogger(__name__)


class DriftSerializer:
    """Saves and loads :class:`~drift.drift_result.DriftResult` objects.

    Supports two formats:

    - **JSON** — human-readable; all metadata and vectors stored as lists.
    - **NPZ** — compact binary; vectors as float32 arrays, metadata as scalars.
    """

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def save_json(self, result: DriftResult, path: str | Path) -> None:
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
        logger.info("DriftResult saved as JSON: %s", path)

    def load_json(self, path: str | Path) -> DriftResult:
        """Load a drift result from a JSON file.

        Args:
            path: Path to the JSON file produced by :meth:`save_json`.

        Returns:
            Reconstructed :class:`~drift.drift_result.DriftResult`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            KeyError: If required fields are missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON drift result file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("DriftResult loaded from JSON: %s", path)
        return DriftResult.from_dict(data)

    # ------------------------------------------------------------------
    # NPZ
    # ------------------------------------------------------------------

    def save_npz(self, result: DriftResult, path: str | Path) -> None:
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
            z_score_vector=result.z_score_vector,
            absolute_difference_vector=result.absolute_difference_vector,
            normalized_vector=result.normalized_vector,
            feature_names=np.array(result.feature_names, dtype=object),
            machine_type=result.machine_type,
            machine_id=result.machine_id,
            filename=result.filename,
            cosine_similarity=result.cosine_similarity,
            euclidean_distance=result.euclidean_distance,
            manhattan_distance=result.manhattan_distance,
            norm_euclidean_distance=result.norm_euclidean_distance,
            norm_manhattan_distance=result.norm_manhattan_distance,
            norm_cosine_similarity=result.norm_cosine_similarity,
            timestamp=result.timestamp,
        )
        logger.info("DriftResult saved as NPZ: %s", path)

    def load_npz(self, path: str | Path) -> DriftResult:
        """Load a drift result from an NPZ file.

        Args:
            path: Path to the NPZ file produced by :meth:`save_npz`.

        Returns:
            Reconstructed :class:`~drift.drift_result.DriftResult`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            suffixed = path.with_suffix(".npz")
            if suffixed.exists():
                path = suffixed
            else:
                raise FileNotFoundError(f"NPZ drift result file not found: {path}")

        data = np.load(path, allow_pickle=True)
        logger.info("DriftResult loaded from NPZ: %s", path)

        return DriftResult.from_dict({
            "machine_type": str(data["machine_type"]),
            "machine_id": str(data["machine_id"]),
            "filename": str(data["filename"]),
            "feature_names": data["feature_names"].tolist(),
            "cosine_similarity": float(data["cosine_similarity"]),
            "euclidean_distance": float(data["euclidean_distance"]),
            "manhattan_distance": float(data["manhattan_distance"]),
            "z_score_vector": data["z_score_vector"].tolist(),
            "absolute_difference_vector": data["absolute_difference_vector"].tolist(),
            "norm_euclidean_distance": float(data["norm_euclidean_distance"]),
            "norm_manhattan_distance": float(data["norm_manhattan_distance"]),
            "norm_cosine_similarity": float(data["norm_cosine_similarity"]),
            "normalized_vector": data["normalized_vector"].tolist(),
            "timestamp": str(data["timestamp"]),
        })
