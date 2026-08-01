"""FusedVectorSerializer: save and load FusedFeatureVector to JSON and NPZ formats."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .fused_vector import FusedFeatureVector

logger = logging.getLogger(__name__)


class FusedVectorSerializer:
    """Saves and loads :class:`~fusion.fused_vector.FusedFeatureVector` objects.

    Supports two formats:

    - **JSON** — human-readable; all metadata and arrays stored as lists.
    - **NPZ** — compact binary; arrays as float32, metadata as scalar arrays.
    """

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def save_json(self, fused: FusedFeatureVector, path: str | Path) -> None:
        """Serialise a fused vector to a JSON file.

        Args:
            fused: The fused vector to save.
            path: Destination file path (created or overwritten).

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(fused.to_dict(), fh, indent=2)
        logger.info("Fused vector saved as JSON: %s", path)

    def load_json(self, path: str | Path) -> FusedFeatureVector:
        """Load a fused vector from a JSON file.

        Args:
            path: Path to the JSON file produced by :meth:`save_json`.

        Returns:
            Reconstructed :class:`~fusion.fused_vector.FusedFeatureVector`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            KeyError: If required fields are missing from the file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON fused vector file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Fused vector loaded from JSON: %s", path)
        return FusedFeatureVector.from_dict(data)

    # ------------------------------------------------------------------
    # NPZ
    # ------------------------------------------------------------------

    def save_npz(self, fused: FusedFeatureVector, path: str | Path) -> None:
        """Serialise a fused vector to a compressed NumPy NPZ file.

        Args:
            fused: The fused vector to save.
            path: Destination file path (``.npz`` extension added if absent).

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            dsp_feature_vector=fused.dsp_feature_vector,
            beats_embedding=fused.beats_embedding,
            fused_feature_vector=fused.fused_feature_vector,
            dsp_feature_names=np.array(fused.dsp_feature_names, dtype=object),
            machine_type=fused.machine_type,
            machine_id=fused.machine_id,
            label=fused.label,
            filename=fused.filename,
            sample_rate=fused.sample_rate,
            created_at=fused.created_at,
        )
        logger.info("Fused vector saved as NPZ: %s", path)

    def load_npz(self, path: str | Path) -> FusedFeatureVector:
        """Load a fused vector from an NPZ file.

        Args:
            path: Path to the NPZ file produced by :meth:`save_npz`.

        Returns:
            Reconstructed :class:`~fusion.fused_vector.FusedFeatureVector`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            suffixed = path.with_suffix(".npz")
            if suffixed.exists():
                path = suffixed
            else:
                raise FileNotFoundError(f"NPZ fused vector file not found: {path}")

        data = np.load(path, allow_pickle=True)
        logger.info("Fused vector loaded from NPZ: %s", path)

        return FusedFeatureVector.from_dict({
            "machine_type": str(data["machine_type"]),
            "machine_id": str(data["machine_id"]),
            "label": str(data["label"]),
            "filename": str(data["filename"]),
            "sample_rate": int(data["sample_rate"]),
            "dsp_feature_names": data["dsp_feature_names"].tolist(),
            "dsp_feature_vector": data["dsp_feature_vector"].tolist(),
            "beats_embedding": data["beats_embedding"].tolist(),
            "fused_feature_vector": data["fused_feature_vector"].tolist(),
            "created_at": str(data["created_at"]),
        })
