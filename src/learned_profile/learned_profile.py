"""LearnedFingerprintProfile — dataclass for the healthy learned fingerprint profile.

SDD v4 §5, §6:
    After contrastive training, every normal recording is projected through the
    trained ProjectionHead to produce a 256-dimensional L2-normalised embedding.
    The collection of these embeddings for one machine forms the Learned
    Fingerprint Profile used for drift analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

_REQUIRED_FIELDS = {
    "machine_type", "machine_id", "embedding_dimension",
    "mean_vector", "std_vector", "created_at",
}


@dataclass
class LearnedFingerprintProfile:
    """Healthy learned fingerprint profile for one machine.

    Attributes:
        machine_type: Type of machine (e.g. ``"pump"``).
        machine_id: Specific machine identifier (e.g. ``"id_00"``).
        embedding_dimension: Dimensionality of each embedding (always 256).
        embeddings: All healthy learned embeddings, shape ``(N, 256)``.
        mean_vector: Per-dimension mean across all embeddings, shape ``(256,)``.
        std_vector: Per-dimension std across all embeddings, shape ``(256,)``.
        created_at: ISO-8601 UTC timestamp of profile creation.
    """

    machine_type: str
    machine_id: str
    embedding_dimension: int
    embeddings: np.ndarray       # shape (N, 256), float32
    mean_vector: np.ndarray      # shape (256,), float32
    std_vector: np.ndarray       # shape (256,), float32
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Serialise the profile to a JSON-compatible dictionary.

        Returns:
            Dict with all metadata and vectors as plain Python lists.
        """
        return {
            "machine_type": self.machine_type,
            "machine_id": self.machine_id,
            "embedding_dimension": self.embedding_dimension,
            "embeddings": self.embeddings.tolist(),
            "mean_vector": self.mean_vector.tolist(),
            "std_vector": self.std_vector.tolist(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearnedFingerprintProfile":
        """Reconstruct a ``LearnedFingerprintProfile`` from a serialised dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A fully reconstructed ``LearnedFingerprintProfile`` instance.

        Raises:
            KeyError: If a required field is missing from ``data``.
        """
        missing = _REQUIRED_FIELDS - data.keys()
        if missing:
            raise KeyError(f"Missing required fields in learned profile dict: {missing}")

        return cls(
            machine_type=data["machine_type"],
            machine_id=data["machine_id"],
            embedding_dimension=int(data["embedding_dimension"]),
            embeddings=np.array(data["embeddings"], dtype=np.float32),
            mean_vector=np.array(data["mean_vector"], dtype=np.float32),
            std_vector=np.array(data["std_vector"], dtype=np.float32),
            created_at=data["created_at"],
        )
