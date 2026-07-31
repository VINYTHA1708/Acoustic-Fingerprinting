"""BEATsEmbedding — dataclass for a single BEATs audio embedding.

SDD v4 §4.1:
    The BEATs deep block produces a 768-dim frozen embedding per audio clip.
    This dataclass is the output contract of BEATsEncoder.encode() and the
    input contract of the Fusion Fingerprint builder (Version 2).

Design note:
    BEATsEmbedding is intentionally kept separate from AcousticFingerprint.
    The fusion step (src/fingerprint/fusion/) concatenates the DSP feature
    vector with the BEATsEmbedding.vector to produce the Fusion Fingerprint.
    This keeps the DSP and deep blocks independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

_EXPECTED_DIM = 768  # BEATs output dimensionality (fixed by the pretrained model)


@dataclass
class BEATsEmbedding:
    """A single 768-dim BEATs audio embedding with provenance metadata.

    Attributes:
        vector: The 768-dim float32 embedding produced by the frozen BEATs encoder.
        embedding_dim: Dimensionality of the vector (always 768).
        filename: Source audio filename this embedding was computed from.
        machine_type: Machine type label (e.g. ``"pump"``).
        machine_id: Machine identifier (e.g. ``"id_00"``).
        sample_rate: Sample rate of the source waveform (expected: 16 000 Hz).
        created_at: ISO-8601 UTC timestamp of embedding creation.
    """

    vector: np.ndarray          # shape (768,), dtype float32
    embedding_dim: int
    filename: str
    machine_type: str
    machine_id: str
    sample_rate: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Validate vector shape and dtype on construction."""
        if self.vector.ndim != 1 or len(self.vector) != _EXPECTED_DIM:
            raise ValueError(
                f"BEATsEmbedding.vector must have shape ({_EXPECTED_DIM},), "
                f"got {self.vector.shape}."
            )
        if self.vector.dtype != np.float32:
            self.vector = self.vector.astype(np.float32)

    def to_dict(self) -> dict:
        """Serialise the embedding to a JSON-compatible dictionary.

        Returns:
            Dict with all fields; vector converted to a plain Python list.
        """
        return {
            "vector": self.vector.tolist(),
            "embedding_dim": self.embedding_dim,
            "filename": self.filename,
            "machine_type": self.machine_type,
            "machine_id": self.machine_id,
            "sample_rate": self.sample_rate,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BEATsEmbedding":
        """Reconstruct a BEATsEmbedding from a serialised dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Returns:
            A fully reconstructed ``BEATsEmbedding`` instance.
        """
        return cls(
            vector=np.array(data["vector"], dtype=np.float32),
            embedding_dim=int(data["embedding_dim"]),
            filename=data["filename"],
            machine_type=data["machine_type"],
            machine_id=data["machine_id"],
            sample_rate=int(data["sample_rate"]),
            created_at=data["created_at"],
        )
