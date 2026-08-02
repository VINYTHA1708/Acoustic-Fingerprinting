"""MachineHealthPipeline — end-to-end pipeline for one recording.

SDD v4 §11:
    FusionCache.load_or_create()
        ↓
    ContrastiveInference
        ↓
    LearnedDriftAnalyzer
        ↓
    LearnedHealthAnalyzer
        ↓
    MachineHealthReport

No preprocessing, DSP, BEATs, fusion, inference, drift, or health logic is
duplicated here. All computation is delegated to the existing modules.
"""

from __future__ import annotations

import logging

from ..dataset.metadata import AudioMetadata
from ..learned_drift.analyzer import LearnedDriftAnalyzer
from ..learned_health_index.analyzer import LearnedHealthAnalyzer
from ..learned_profile.learned_profile import LearnedFingerprintProfile
from .result import MachineHealthReport

logger = logging.getLogger(__name__)


class MachineHealthPipeline:
    """Runs the full health analysis pipeline for one recording.

    Accepts pre-constructed analyzer instances so that expensive resources
    (BEATs model, ProjectionHead checkpoint, FusionCache) are loaded once
    and reused across multiple calls to :meth:`analyze`.

    Args:
        profile: :class:`~learned_profile.learned_profile.LearnedFingerprintProfile`
                 for the target machine.
        drift_analyzer: A ready :class:`~learned_drift.analyzer.LearnedDriftAnalyzer`.
        health_analyzer: A ready :class:`~learned_health_index.analyzer.LearnedHealthAnalyzer`.
    """

    def __init__(
        self,
        profile: LearnedFingerprintProfile,
        drift_analyzer: LearnedDriftAnalyzer,
        health_analyzer: LearnedHealthAnalyzer,
    ) -> None:
        self._profile = profile
        self._drift_analyzer = drift_analyzer
        self._health_analyzer = health_analyzer

    def analyze(self, record: AudioMetadata) -> MachineHealthReport:
        """Run the full pipeline for one recording and return a health report.

        Delegates to :class:`~learned_drift.analyzer.LearnedDriftAnalyzer` for
        drift metrics and :class:`~learned_health_index.analyzer.LearnedHealthAnalyzer`
        for the health score. Dimension metadata is read from the fused vector
        produced internally by the drift analyzer's FusionCache.

        Args:
            record: :class:`~dataset.metadata.AudioMetadata` for the recording to analyze.

        Returns:
            :class:`MachineHealthReport` with all metrics and health fields populated.
        """
        drift = self._drift_analyzer.analyze(record, self._profile)
        health = self._health_analyzer.analyze(record, self._profile)

        # Retrieve dimension metadata from the cached fused vector.
        fused = self._drift_analyzer._cache.load_or_create(record)
        dsp_dim = int(fused.dsp_feature_vector.shape[0])
        beats_dim = int(fused.beats_embedding.shape[0])
        fusion_dim = int(fused.fused_feature_vector.shape[0])
        learned_dim = int(self._profile.embedding_dimension)

        report = MachineHealthReport(
            machine_type=record.machine_type,
            machine_id=record.machine_id,
            filename=record.filename,
            dsp_dimension=dsp_dim,
            beats_dimension=beats_dim,
            fusion_dimension=fusion_dim,
            learned_dimension=learned_dim,
            euclidean_distance=drift.euclidean_distance,
            manhattan_distance=drift.manhattan_distance,
            cosine_similarity=drift.cosine_similarity,
            normalized_euclidean_distance=drift.norm_euclidean_distance,
            normalized_manhattan_distance=drift.norm_manhattan_distance,
            normalized_cosine_similarity=drift.norm_cosine_similarity,
            health_score=health.health_score,
            health_percentage=health.health_percentage,
            health_state=health.health_state,
        )

        logger.info(
            "MachineHealthPipeline — %s/%s '%s'  score=%.2f  state=%s",
            report.machine_type, report.machine_id, report.filename,
            report.health_score, report.health_state,
        )
        return report
