"""FingerprintSerializer: save and load AcousticFingerprint to JSON and NPZ formats."""

import json
import logging
from pathlib import Path

import numpy as np

from .fingerprint import AcousticFingerprint

logger = logging.getLogger(__name__)

# Metadata fields stored alongside the vector in NPZ files.
_NPZ_METADATA_KEYS = ("machine_type", "machine_id", "label", "filename", "sample_rate", "created_at")


class FingerprintSerializer:
    """Saves and loads :class:`~fingerprint.fingerprint.AcousticFingerprint` objects.

    Supports two formats:

    - **JSON** — human-readable; stores all metadata, feature names, and the
      feature vector as a list of floats.
    - **NPZ** — compact binary; stores the feature vector as a float32 array,
      feature names as a string array, and all metadata fields as scalar arrays.
    """

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def save_json(self, fingerprint: AcousticFingerprint, path: str | Path) -> None:
        """Serialise a fingerprint to a JSON file.

        Args:
            fingerprint: The fingerprint to save.
            path: Destination file path (created or overwritten).

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(fingerprint.to_dict(), fh, indent=2)
        logger.info("Fingerprint saved as JSON: %s", path)

    def load_json(self, path: str | Path) -> AcousticFingerprint:
        """Load a fingerprint from a JSON file.

        Args:
            path: Path to the JSON file produced by :meth:`save_json`.

        Returns:
            Reconstructed :class:`~fingerprint.fingerprint.AcousticFingerprint`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            KeyError: If required fields are missing from the file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON fingerprint file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Fingerprint loaded from JSON: %s", path)
        return AcousticFingerprint.from_dict(data)

    # ------------------------------------------------------------------
    # NPZ
    # ------------------------------------------------------------------

    def save_npz(self, fingerprint: AcousticFingerprint, path: str | Path) -> None:
        """Serialise a fingerprint to a compressed NumPy NPZ file.

        Args:
            fingerprint: The fingerprint to save.
            path: Destination file path (the ``.npz`` extension is added by
                ``numpy.savez_compressed`` if absent).

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            feature_vector=fingerprint.feature_vector,
            feature_names=np.array(fingerprint.feature_names, dtype=object),
            machine_type=fingerprint.machine_type,
            machine_id=fingerprint.machine_id,
            label=fingerprint.label,
            filename=fingerprint.filename,
            sample_rate=fingerprint.sample_rate,
            created_at=fingerprint.created_at,
        )
        logger.info("Fingerprint saved as NPZ: %s", path)

    def load_npz(self, path: str | Path) -> AcousticFingerprint:
        """Load a fingerprint from an NPZ file.

        Args:
            path: Path to the NPZ file produced by :meth:`save_npz`.

        Returns:
            Reconstructed :class:`~fingerprint.fingerprint.AcousticFingerprint`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            # numpy appends .npz automatically; try the suffixed path too
            suffixed = path.with_suffix(".npz")
            if suffixed.exists():
                path = suffixed
            else:
                raise FileNotFoundError(f"NPZ fingerprint file not found: {path}")

        data = np.load(path, allow_pickle=True)
        logger.info("Fingerprint loaded from NPZ: %s", path)

        return AcousticFingerprint.from_dict({
            "machine_type": str(data["machine_type"]),
            "machine_id": str(data["machine_id"]),
            "label": str(data["label"]),
            "filename": str(data["filename"]),
            "sample_rate": int(data["sample_rate"]),
            "feature_names": data["feature_names"].tolist(),
            "feature_vector": data["feature_vector"].tolist(),
            "created_at": str(data["created_at"]),
        })
