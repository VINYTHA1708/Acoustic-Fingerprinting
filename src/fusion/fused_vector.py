"""FusedFeatureVector — dataclass for a single DSP + BEATs fused feature vector.

SDD v4 §4.1:
    Fusion Fingerprint = DSP Features ⊕ BEATs Embedding (768-dim, frozen)
    DSP features always appear first in the concatenated vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np


@dataclass
class FusedFeatureVector:
    """A single fused feature vector combining DSP features and BEATs embedding.

    Attributes:
        machine_type: Machine type label (e.g. ``"pump"``).
        machine_id: Machine identifier (e.g. ``"id_00"``).
        label: Recording condition — ``"normal"`` or ``"abnormal"``.
        filename: Source audio filename.
        sample_rate: Sample rate of the source recording in Hz.
        dsp_feature_names: Ordered list of DSP feature names.
        dsp_feature_vector: 1-D float32 array of DSP feature values.
        beats_embedding: 1-D float32 array of BEATs embedding values (768-dim).
        fused_feature_vector: Concatenation of DSP vector followed by BEATs embedding.
        created_at: ISO-8601 UTC timestamp of creation.
    """

    machine_type: str
    machine_id: str
    label: str
    filename: str
    sample_rate: int
    dsp_feature_names: list[str]
    dsp_feature_vector: np.ndarray    # shape (D,), float32
    beats_embedding: np.ndarray       # shape (768,), float32
    fused_feature_vector: np.ndarray  # shape (D + 768,), float32
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "machine_type": self.machine_type,
            "machine_id": self.machine_id,
            "label": self.label,
            "filename": self.filename,
            "sample_rate": self.sample_rate,
            "dsp_feature_names": self.dsp_feature_names,
            "dsp_feature_vector": self.dsp_feature_vector.tolist(),
            "beats_embedding": self.beats_embedding.tolist(),
            "fused_feature_vector": self.fused_feature_vector.tolist(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FusedFeatureVector":
        """Reconstruct a FusedFeatureVector from a serialised dictionary.

        Args:
            data: Dict as produced by :meth:`to_dict`.

        Raises:
            KeyError: If a required field is missing from ``data``.
        """
        required = {
            "machine_type", "machine_id", "label", "filename", "sample_rate",
            "dsp_feature_names", "dsp_feature_vector", "beats_embedding",
            "fused_feature_vector", "created_at",
        }
        missing = required - data.keys()
        if missing:
            raise KeyError(f"Missing required fields in fused vector dict: {missing}")

        return cls(
            machine_type=data["machine_type"],
            machine_id=data["machine_id"],
            label=data["label"],
            filename=data["filename"],
            sample_rate=int(data["sample_rate"]),
            dsp_feature_names=list(data["dsp_feature_names"]),
            dsp_feature_vector=np.array(data["dsp_feature_vector"], dtype=np.float32),
            beats_embedding=np.array(data["beats_embedding"], dtype=np.float32),
            fused_feature_vector=np.array(data["fused_feature_vector"], dtype=np.float32),
            created_at=data["created_at"],
        )
