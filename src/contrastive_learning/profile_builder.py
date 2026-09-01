"""LearnedFingerprintProfileBuilder — builds a LearnedFingerprintProfile from fused vectors.

Accepts already-computed :class:`~fusion.fused_vector.FusedFeatureVector` objects,
passes each through :class:`ContrastiveInference`, and aggregates the resulting
256-dimensional embeddings into a :class:`~learned_profile.learned_profile.LearnedFingerprintProfile`.

No training, dataset loading, or checkpoint creation is performed here.
"""

from __future__ import annotations

import logging

import numpy as np

from ..fusion.fused_vector import FusedFeatureVector
from ..learned_profile.learned_profile import LearnedFingerprintProfile
from .inference import ContrastiveInference

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 256


class LearnedFingerprintProfileBuilder:
    """Builds a healthy :class:`LearnedFingerprintProfile` for one machine.

    Accepts pre-computed fused vectors and projects each through the trained
    :class:`ContrastiveInference` head to produce 256-dimensional embeddings.
    Mean and standard deviation are computed across all embeddings.

    Args:
        inference: A :class:`ContrastiveInference` instance loaded with a
                   trained checkpoint.
    """

    def __init__(self, inference: ContrastiveInference) -> None:
        self._inference = inference

    def build(
        self,
        machine_type: str,
        machine_id: str,
        fused_vectors: list[FusedFeatureVector],
    ) -> LearnedFingerprintProfile:
        """Build a healthy learned fingerprint profile from fused vectors.

        Args:
            machine_type: Expected machine type for all vectors (e.g. ``"pump"``).
            machine_id: Expected machine ID for all vectors (e.g. ``"id_00"``).
            fused_vectors: Non-empty list of :class:`FusedFeatureVector` objects,
                           all belonging to *machine_type* / *machine_id*.

        Returns:
            :class:`LearnedFingerprintProfile` with embeddings, mean, and std.

        Raises:
            ValueError: If *fused_vectors* is empty, contains vectors from a
                        different machine, or if any generated embedding is
                        invalid (wrong dimension, NaN, or Inf).
        """
        if not fused_vectors:
            raise ValueError("fused_vectors must not be empty.")

        for fv in fused_vectors:
            if fv.machine_type != machine_type:
                raise ValueError(
                    f"Expected machine_type '{machine_type}', "
                    f"got '{fv.machine_type}' in vector '{fv.filename}'."
                )
            if fv.machine_id != machine_id:
                raise ValueError(
                    f"Expected machine_id '{machine_id}', "
                    f"got '{fv.machine_id}' in vector '{fv.filename}'."
                )

        embeddings: list[np.ndarray] = []
        for fv in fused_vectors:
            emb = self._inference.generate_fingerprint(fv)

            if emb.shape[0] != _EMBEDDING_DIM:
                raise ValueError(
                    f"Expected embedding dimension {_EMBEDDING_DIM}, got {emb.shape[0]} "
                    f"for '{fv.filename}'."
                )
            if np.isnan(emb).any():
                raise ValueError(
                    f"Embedding for '{fv.filename}' contains NaN values."
                )
            if np.isinf(emb).any():
                raise ValueError(
                    f"Embedding for '{fv.filename}' contains Inf values."
                )

            embeddings.append(emb)

        matrix = np.stack(embeddings, axis=0).astype(np.float32)   # (N, 256)
        mean_vec = matrix.mean(axis=0).astype(np.float32)           # (256,)
        std_vec = matrix.std(axis=0).astype(np.float32)             # (256,)

        logger.info(
            "Learned profile built — %s/%s  recordings=%d  dim=%d",
            machine_type, machine_id, len(embeddings), _EMBEDDING_DIM,
        )

        return LearnedFingerprintProfile(
            machine_type=machine_type,
            machine_id=machine_id,
            embedding_dimension=_EMBEDDING_DIM,
            embeddings=matrix,
            mean_vector=mean_vec,
            std_vector=std_vec,
        )
