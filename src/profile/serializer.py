"""ProfileSerializer: save and load HealthyFingerprintProfile to JSON and NPZ."""

import json
import logging
from pathlib import Path

import numpy as np

from .profile import HealthyFingerprintProfile

logger = logging.getLogger(__name__)

_VECTOR_KEYS = ("mean_vector", "std_vector", "min_vector", "max_vector")
_META_KEYS = ("machine_type", "machine_id", "number_of_samples", "created_at")


class ProfileSerializer:
    """Saves and loads :class:`~profile.profile.HealthyFingerprintProfile` objects.

    Supports two formats:

    - **JSON** — human-readable; all metadata and vectors stored as lists.
    - **NPZ** — compact binary; vectors as float32 arrays, metadata as scalars.
    """

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def save_json(self, profile: HealthyFingerprintProfile, path: str | Path) -> None:
        """Serialise a profile to a JSON file.

        Args:
            profile: The profile to save.
            path: Destination file path (created or overwritten).

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(profile.to_dict(), fh, indent=2)
        logger.info("Profile saved as JSON: %s", path)

    def load_json(self, path: str | Path) -> HealthyFingerprintProfile:
        """Load a profile from a JSON file.

        Args:
            path: Path to the JSON file produced by :meth:`save_json`.

        Returns:
            Reconstructed :class:`~profile.profile.HealthyFingerprintProfile`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            KeyError: If required fields are missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON profile file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Profile loaded from JSON: %s", path)
        return HealthyFingerprintProfile.from_dict(data)

    # ------------------------------------------------------------------
    # NPZ
    # ------------------------------------------------------------------

    def save_npz(self, profile: HealthyFingerprintProfile, path: str | Path) -> None:
        """Serialise a profile to a compressed NumPy NPZ file.

        Args:
            profile: The profile to save.
            path: Destination file path (``.npz`` extension added if absent).

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean_vector=profile.mean_vector,
            std_vector=profile.std_vector,
            min_vector=profile.min_vector,
            max_vector=profile.max_vector,
            feature_names=np.array(profile.feature_names, dtype=object),
            machine_type=profile.machine_type,
            machine_id=profile.machine_id,
            number_of_samples=profile.number_of_samples,
            created_at=profile.created_at,
        )
        logger.info("Profile saved as NPZ: %s", path)

    def load_npz(self, path: str | Path) -> HealthyFingerprintProfile:
        """Load a profile from an NPZ file.

        Args:
            path: Path to the NPZ file produced by :meth:`save_npz`.

        Returns:
            Reconstructed :class:`~profile.profile.HealthyFingerprintProfile`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            suffixed = path.with_suffix(".npz")
            if suffixed.exists():
                path = suffixed
            else:
                raise FileNotFoundError(f"NPZ profile file not found: {path}")

        data = np.load(path, allow_pickle=True)
        logger.info("Profile loaded from NPZ: %s", path)

        return HealthyFingerprintProfile.from_dict({
            "machine_type": str(data["machine_type"]),
            "machine_id": str(data["machine_id"]),
            "number_of_samples": int(data["number_of_samples"]),
            "feature_names": data["feature_names"].tolist(),
            "mean_vector": data["mean_vector"].tolist(),
            "std_vector": data["std_vector"].tolist(),
            "min_vector": data["min_vector"].tolist(),
            "max_vector": data["max_vector"].tolist(),
            "created_at": str(data["created_at"]),
        })
