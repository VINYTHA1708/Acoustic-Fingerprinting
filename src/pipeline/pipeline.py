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


# ---------------------------------------------------------------------------
# InferencePipeline — self-contained end-to-end pipeline
# ---------------------------------------------------------------------------

from pathlib import Path as _Path

from ..beats.encoder import BEATsEncoder as _BEATsEncoder
from ..contrastive_learning.inference import ContrastiveInference as _ContrastiveInference
from ..contrastive_learning.model import ProjectionHead as _ProjectionHead
from ..feature_extraction.extractor import FeatureExtractor as _FeatureExtractor
from ..feature_extraction.feature_vector import FeatureVectorBuilder as _FeatureVectorBuilder
from ..fusion.cache import FusionCache as _FusionCache
from ..fusion.fusion import FusionBuilder as _FusionBuilder
from ..learned_drift.analyzer import LearnedDriftAnalyzer as _LearnedDriftAnalyzer
from ..learned_health_index.analyzer import LearnedHealthAnalyzer as _LearnedHealthAnalyzer
from ..preprocessing.pipeline import PreprocessingPipeline as _PreprocessingPipeline
from .result import PipelineResult

_BEATS_DEFAULT = (
    _Path(__file__).resolve().parents[2] / "models" / "beats" / "BEATs_iter3_plus_AS2M.pt"
)
_CACHE_DEFAULT = _Path(__file__).resolve().parents[2] / "data" / "fusion_cache"

_pipeline_logger = logging.getLogger(__name__ + ".InferencePipeline")


class InferencePipeline:
    """Self-contained end-to-end inference pipeline for one recording.

    Constructs all required sub-components internally so callers only need
    to supply paths.  Expensive resources (BEATs model, ProjectionHead,
    FusionCache) are loaded once at construction time and reused across
    every call to :meth:`analyze`.

    All computation is delegated to the existing modules:

    - :class:`~fusion.cache.FusionCache` — load or compute the fused vector
    - :class:`~learned_drift.analyzer.LearnedDriftAnalyzer` — drift metrics
    - :class:`~learned_health_index.analyzer.LearnedHealthAnalyzer` — health score

    No preprocessing, DSP, BEATs, fusion, drift, or health logic is duplicated.

    Args:
        checkpoint_path: Path to the trained ProjectionHead ``.pt`` checkpoint.
        beats_checkpoint: Path to the BEATs model checkpoint.
                          Defaults to ``models/beats/BEATs_iter3_plus_AS2M.pt``.
        cache_root: Directory for the FusionCache.
                    Defaults to ``data/fusion_cache``.
    """

    def __init__(
        self,
        checkpoint_path: str | _Path,
        beats_checkpoint: str | _Path | None = None,
        cache_root: str | _Path | None = None,
    ) -> None:
        beats_ckpt = _Path(beats_checkpoint) if beats_checkpoint else _BEATS_DEFAULT
        _cache = _Path(cache_root) if cache_root else _CACHE_DEFAULT

        # Build the shared FusionCache once — reused by both analyzers.
        _cache_instance = _FusionCache(
            cache_root=_cache,
            pipeline=_PreprocessingPipeline(target_sr=16_000),
            extractor=_FeatureExtractor(sample_rate=16_000),
            vec_builder=_FeatureVectorBuilder(),
            encoder=_BEATsEncoder(beats_ckpt),
            fusion=_FusionBuilder(),
        )

        self._drift_analyzer = _LearnedDriftAnalyzer(
            checkpoint_path=checkpoint_path,
            beats_checkpoint=beats_ckpt,
            cache_root=_cache,
        )
        self._health_analyzer = _LearnedHealthAnalyzer(
            checkpoint_path=checkpoint_path,
            beats_checkpoint=beats_ckpt,
            cache_root=_cache,
        )
        # Keep a reference to the cache for dimension introspection.
        self._cache = _cache_instance

    def analyze(
        self,
        audio_record: "AudioMetadata",
        learned_profile: "LearnedFingerprintProfile",
    ) -> PipelineResult:
        """Run the full pipeline for one recording and return a :class:`PipelineResult`.

        Delegates to :class:`~learned_drift.analyzer.LearnedDriftAnalyzer` for
        drift metrics and :class:`~learned_health_index.analyzer.LearnedHealthAnalyzer`
        for the health score.  Dimension metadata is read from the fused vector
        produced by the FusionCache.

        Args:
            audio_record: :class:`~dataset.metadata.AudioMetadata` for the recording.
            learned_profile: :class:`~learned_profile.learned_profile.LearnedFingerprintProfile`
                             for the same machine.

        Returns:
            :class:`PipelineResult` with all metrics and health fields populated.
        """
        drift = self._drift_analyzer.analyze(audio_record, learned_profile)
        health = self._health_analyzer.analyze(audio_record, learned_profile)

        fused = self._drift_analyzer._cache.load_or_create(audio_record)
        dsp_dim = int(fused.dsp_feature_vector.shape[0])
        beats_dim = int(fused.beats_embedding.shape[0])
        fusion_dim = int(fused.fused_feature_vector.shape[0])
        embedding_dim = int(learned_profile.embedding_dimension)

        result = PipelineResult(
            machine_type=audio_record.machine_type,
            machine_id=audio_record.machine_id,
            filename=audio_record.filename,
            dsp_dimension=dsp_dim,
            beats_dimension=beats_dim,
            fusion_dimension=fusion_dim,
            embedding_dimension=embedding_dim,
            raw_euclidean=drift.euclidean_distance,
            raw_manhattan=drift.manhattan_distance,
            raw_cosine=drift.cosine_similarity,
            normalized_euclidean=drift.norm_euclidean_distance,
            normalized_manhattan=drift.norm_manhattan_distance,
            normalized_cosine=drift.norm_cosine_similarity,
            health_score=health.health_score,
            health_percentage=health.health_percentage,
            health_state=health.health_state,
        )

        _pipeline_logger.info(
            "InferencePipeline — %s/%s '%s'  score=%.2f  state=%s",
            result.machine_type, result.machine_id, result.filename,
            result.health_score, result.health_state,
        )
        return result
