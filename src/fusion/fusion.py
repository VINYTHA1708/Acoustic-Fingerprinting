"""FusionBuilder — concatenates DSP features and BEATs embedding into a Fusion Fingerprint.

SDD v4 §4.1:
    Fingerprint = DSP Features ⊕ BEATs Embedding
    DSP features always appear first. BEATs is strictly additive — the DSP
    pathway from Version 1 is never removed.
"""

from __future__ import annotations

import logging

import numpy as np

from ..beats.embedding import BEATsEmbedding
from .fused_vector import FusedFeatureVector

logger = logging.getLogger(__name__)

_BEATS_DIM = 768


class FusionBuilder:
    """Builds a :class:`FusedFeatureVector` from a DSP vector and a BEATs embedding.

    The fused vector is the simple concatenation::

        fused = np.concatenate([dsp_feature_vector, beats_embedding.vector])

    DSP features always occupy the first ``D`` dimensions; BEATs occupies the
    trailing 768 dimensions.
    """

    def build(
        self,
        dsp_vector: np.ndarray,
        dsp_feature_names: list[str],
        beats_embedding: BEATsEmbedding,
        machine_type: str = "",
        machine_id: str = "",
        label: str = "",
    ) -> FusedFeatureVector:
        """Fuse a DSP feature vector with a BEATs embedding.

        Args:
            dsp_vector: 1-D float32 DSP feature vector, shape ``(D,)``.
                        Produced by :class:`~feature_extraction.feature_vector.FeatureVectorBuilder`.
            dsp_feature_names: Ordered feature names matching ``dsp_vector``.
            beats_embedding: :class:`~beats.embedding.BEATsEmbedding` with a
                             768-dim float32 vector.
            machine_type: Machine type label (e.g. ``"pump"``).
            machine_id: Machine identifier (e.g. ``"id_00"``).
            label: Recording condition — ``"normal"`` or ``"abnormal"``.

        Returns:
            :class:`FusedFeatureVector` with DSP features first, BEATs last.

        Raises:
            ValueError: If the DSP vector is empty, the BEATs embedding is not
                        768-dim, or either vector contains NaN / Inf values.
        """
        self._validate(dsp_vector, beats_embedding)

        fused = np.concatenate([dsp_vector, beats_embedding.vector]).astype(np.float32)

        expected_len = len(dsp_vector) + _BEATS_DIM
        if len(fused) != expected_len:
            raise ValueError(
                f"Fused vector length {len(fused)} != expected {expected_len} "
                f"(DSP {len(dsp_vector)} + BEATs {_BEATS_DIM})."
            )

        logger.info(
            "Fusion complete — DSP=%d, BEATs=%d, fused=%d",
            len(dsp_vector), _BEATS_DIM, len(fused),
        )

        return FusedFeatureVector(
            machine_type=machine_type,
            machine_id=machine_id,
            label=label,
            filename=beats_embedding.filename,
            sample_rate=beats_embedding.sample_rate,
            dsp_feature_names=list(dsp_feature_names),
            dsp_feature_vector=dsp_vector.astype(np.float32),
            beats_embedding=beats_embedding.vector,
            fused_feature_vector=fused,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(self, dsp_vector: np.ndarray, beats_embedding: BEATsEmbedding) -> None:
        if dsp_vector.ndim != 1 or len(dsp_vector) == 0:
            raise ValueError(
                f"DSP vector must be a non-empty 1-D array, got shape {dsp_vector.shape}."
            )
        if beats_embedding.vector.ndim != 1 or len(beats_embedding.vector) != _BEATS_DIM:
            raise ValueError(
                f"BEATs embedding must be 1-D with {_BEATS_DIM} dimensions, "
                f"got shape {beats_embedding.vector.shape}."
            )
        if not np.isfinite(dsp_vector).all():
            raise ValueError("DSP feature vector contains NaN or Inf values.")
        if not np.isfinite(beats_embedding.vector).all():
            raise ValueError("BEATs embedding contains NaN or Inf values.")
