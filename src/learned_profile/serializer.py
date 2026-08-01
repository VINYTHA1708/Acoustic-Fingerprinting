"""LearnedProfileSerializer — save and load LearnedFingerprintProfile to JSON and NPZ."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .learned_profile import LearnedFingerprintProfile

logger = logging.getLogger(__name__)

_META_KEYS = ("machine_type", "machine_id", "embedding_dimension", "created_at")


class LearnedProfileSerializer:
    """Saves and loads :class:`~learned_profile.learned_profile.LearnedFingerprintProfile` objects.

    Supports two formats:

    - **JSON** — human-readable; all metadata and vectors stored as lists.
    - **NPZ** — compact binary; vectors as float32 arrays, metadata as scalars.
    """

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def save_json(self, profile: LearnedFingerprintProfile, path: str | Path) -> None:
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
        logger.info("Learned profile saved as JSON: %s", path)

    def load_json(self, path: str | Path) -> LearnedFingerprintProfile:
        """Load a profile from a JSON file.

        Args:
            path: Path to the JSON file produced by :meth:`save_json`.

        Returns:
            Reconstructed :class:`~learned_profile.learned_profile.LearnedFingerprintProfile`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            KeyError: If required fields are missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON learned profile file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Learned profile loaded from JSON: %s", path)
        return LearnedFingerprintProfile.from_dict(data)

    # ------------------------------------------------------------------
    # NPZ
    # ------------------------------------------------------------------

    def save_npz(self, profile: LearnedFingerprintProfile, path: str | Path) -> None:
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
            embeddings=profile.embeddings,
            mean_vector=profile.mean_vector,
            std_vector=profile.std_vector,
            machine_type=profile.machine_type,
            machine_id=profile.machine_id,
            embedding_dimension=profile.embedding_dimension,
            created_at=profile.created_at,
        )
        logger.info("Learned profile saved as NPZ: %s", path)

    def load_npz(self, path: str | Path) -> LearnedFingerprintProfile:
        """Load a profile from an NPZ file.

        Args:
            path: Path to the NPZ file produced by :meth:`save_npz`.

        Returns:
            Reconstructed :class:`~learned_profile.learned_profile.LearnedFingerprintProfile`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            suffixed = path.with_suffix(".npz")
            if suffixed.exists():
                path = suffixed
            else:
                raise FileNotFoundError(f"NPZ learned profile file not found: {path}")

        data = np.load(path, allow_pickle=True)
        logger.info("Learned profile loaded from NPZ: %s", path)

        return LearnedFingerprintProfile.from_dict({
            "machine_type": str(data["machine_type"]),
            "machine_id": str(data["machine_id"]),
            "embedding_dimension": int(data["embedding_dimension"]),
            "embeddings": data["embeddings"].tolist(),
            "mean_vector": data["mean_vector"].tolist(),
            "std_vector": data["std_vector"].tolist(),
            "created_at": str(data["created_at"]),
        })
