"""LearnedHealthAnalyzer — computes health index for one recording.

Pipeline (SDD v4 §11):
    record
    → LearnedDriftAnalyzer  (preprocessing + inference + drift metrics)
    → LearnedHealthCalculator  (profile-derived scale)
    → LearnedHealthResult

Reuses FusionCache and ContrastiveInference via LearnedDriftAnalyzer.
No pipeline logic is duplicated.

Profile-derived scale:
    For each healthy embedding e_i in profile.embeddings, compute
    z_i = (e_i − mean_vector) / std_vector, then take mean ‖z_i‖.
    This is the expected normalized Euclidean distance for a healthy
    recording and is used as the machine-specific normalization scale.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..dataset.metadata import AudioMetadata
from ..learned_drift.analyzer import LearnedDriftAnalyzer
from ..learned_profile.learned_profile import LearnedFingerprintProfile
from .calculator import LearnedHealthCalculator
from .learned_health_result import LearnedHealthResult

logger = logging.getLogger(__name__)

_STD_FLOOR = 1e-10


class LearnedHealthAnalyzer:
    """Analyzes the health of a recording against a :class:`LearnedFingerprintProfile`.

    Delegates preprocessing and inference to :class:`~learned_drift.analyzer.LearnedDriftAnalyzer`,
    then converts the normalized drift metrics into a health score via
    :class:`LearnedHealthCalculator`.

    The normalization scale is derived per-machine from the profile embeddings:
    it is the mean ‖z_i‖ of all healthy embeddings, where
    z_i = (embedding_i − mean_vector) / std_vector.

    Args:
        checkpoint_path: Path to the trained ProjectionHead ``.pt`` checkpoint.
        beats_checkpoint: Path to the BEATs model checkpoint.
                          Defaults to the project standard location.
        cache_root: Directory for the FusionCache.
                    Defaults to ``data/fusion_cache``.
        thresholds: State band thresholds passed to :class:`LearnedHealthCalculator`.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        beats_checkpoint: str | Path | None = None,
        cache_root: str | Path | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self._drift_analyzer = LearnedDriftAnalyzer(
            checkpoint_path=checkpoint_path,
            beats_checkpoint=beats_checkpoint,
            cache_root=cache_root,
        )
        self._calculator = LearnedHealthCalculator(thresholds=thresholds)

    @staticmethod
    def _profile_healthy_norm(profile: LearnedFingerprintProfile) -> float:
        """Compute the mean ‖z_i‖ of all healthy embeddings in the profile.

        This is the expected normalized Euclidean distance for a healthy
        recording and serves as the machine-specific scale.
        """
        mean = profile.mean_vector.astype(np.float32)
        std = profile.std_vector.astype(np.float32)
        safe_std = np.where(std < _STD_FLOOR, 1.0, std)
        z = np.where(
            std < _STD_FLOOR,
            0.0,
            (profile.embeddings.astype(np.float32) - mean) / safe_std,
        )  # shape (N, 256)
        norms = np.linalg.norm(z, axis=1)  # shape (N,)
        return float(norms.mean())

    def analyze(
        self,
        record: AudioMetadata,
        profile: LearnedFingerprintProfile,
    ) -> LearnedHealthResult:
        """Compute the health index for one recording.

        Args:
            record: :class:`~dataset.metadata.AudioMetadata` for the recording to analyze.
            profile: :class:`~learned_profile.learned_profile.LearnedFingerprintProfile`
                     for the same machine.

        Returns:
            :class:`LearnedHealthResult` with health score, percentage, and state.
        """
        drift = self._drift_analyzer.analyze(record, profile)
        healthy_norm = self._profile_healthy_norm(profile)

        health_score, health_percentage, health_state = self._calculator.calculate(
            normalized_euclidean=drift.norm_euclidean_distance,
            normalized_manhattan=drift.norm_manhattan_distance,
            normalized_cosine=drift.norm_cosine_similarity,
            profile_healthy_norm=healthy_norm,
        )

        result = LearnedHealthResult(
            machine_type=record.machine_type,
            machine_id=record.machine_id,
            filename=record.filename,
            health_score=health_score,
            health_percentage=health_percentage,
            health_state=health_state,
            normalized_euclidean=drift.norm_euclidean_distance,
            normalized_manhattan=drift.norm_manhattan_distance,
            normalized_cosine=drift.norm_cosine_similarity,
        )

        logger.info(
            "Learned health — %s/%s '%s'  score=%.2f  state=%s",
            result.machine_type, result.machine_id, result.filename,
            result.health_score, result.health_state,
        )
        return result
