"""LearnedDriftAnalyzer — runs the full pipeline for one recording and returns a LearnedDriftResult.

SDD v4 §11 (Version 3):
    Pipeline per recording:
        Audio → Preprocessing → DSP → BEATs → Fusion → ProjectionHead → 256-dim embedding
        → Learned Drift Metrics → LearnedDriftResult

Reuses FusionCache and ContrastiveInference — no pipeline logic is duplicated.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..beats.encoder import BEATsEncoder
from ..contrastive_learning.inference import ContrastiveInference
from ..contrastive_learning.model import ProjectionHead
from ..dataset.metadata import AudioMetadata
from ..feature_extraction.extractor import FeatureExtractor
from ..feature_extraction.feature_vector import FeatureVectorBuilder
from ..fusion.cache import FusionCache
from ..fusion.fusion import FusionBuilder
from ..learned_profile.learned_profile import LearnedFingerprintProfile
from ..preprocessing.pipeline import PreprocessingPipeline
from .learned_drift_result import LearnedDriftResult
from .metrics import LearnedDriftMetrics

logger = logging.getLogger(__name__)

_BEATS_CHECKPOINT_REL = (
    Path(__file__).resolve().parents[2] / "models" / "beats" / "BEATs_iter3_plus_AS2M.pt"
)
_CACHE_ROOT_REL = Path(__file__).resolve().parents[2] / "data" / "fusion_cache"


class LearnedDriftAnalyzer:
    """Analyzes drift between a new recording and a :class:`LearnedFingerprintProfile`.

    Runs the full pipeline (Preprocessing → DSP → BEATs → Fusion → ProjectionHead)
    for each recording, then computes raw and normalized drift metrics against the
    provided healthy profile.

    Args:
        checkpoint_path: Path to the trained ProjectionHead ``.pt`` checkpoint.
        beats_checkpoint: Path to the BEATs model checkpoint.
                          Defaults to the project standard location.
        cache_root: Directory for the FusionCache.
                    Defaults to ``data/fusion_cache``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        beats_checkpoint: str | Path | None = None,
        cache_root: str | Path | None = None,
    ) -> None:
        beats_ckpt = Path(beats_checkpoint) if beats_checkpoint else _BEATS_CHECKPOINT_REL
        _cache = Path(cache_root) if cache_root else _CACHE_ROOT_REL

        pipeline = PreprocessingPipeline(target_sr=16_000)
        extractor = FeatureExtractor(sample_rate=16_000)
        vec_builder = FeatureVectorBuilder()
        encoder = BEATsEncoder(beats_ckpt)
        fusion = FusionBuilder()

        self._cache = FusionCache(
            cache_root=_cache,
            pipeline=pipeline,
            extractor=extractor,
            vec_builder=vec_builder,
            encoder=encoder,
            fusion=fusion,
        )

        head = ProjectionHead()
        self._inference = ContrastiveInference(
            projection_head=head,
            checkpoint_path=checkpoint_path,
        )

        self._metrics = LearnedDriftMetrics()

    def analyze(
        self,
        record: AudioMetadata,
        profile: LearnedFingerprintProfile,
    ) -> LearnedDriftResult:
        """Analyze drift for one recording against a healthy learned profile.

        Args:
            record: :class:`~dataset.metadata.AudioMetadata` for the recording to analyze.
            profile: :class:`~learned_profile.learned_profile.LearnedFingerprintProfile`
                     for the same machine.

        Returns:
            :class:`LearnedDriftResult` containing all raw and normalized drift metrics.

        Raises:
            ValueError: If the recording's machine type or ID does not match the profile.
        """
        if record.machine_type != profile.machine_type:
            raise ValueError(
                f"machine_type mismatch: record='{record.machine_type}' "
                f"vs profile='{profile.machine_type}'."
            )
        if record.machine_id != profile.machine_id:
            raise ValueError(
                f"machine_id mismatch: record='{record.machine_id}' "
                f"vs profile='{profile.machine_id}'."
            )

        fused = self._cache.load_or_create(record)
        embedding = self._inference.generate_fingerprint(fused)

        euclid, manhat, cosine, norm_euclid, norm_manhat, norm_cosine, norm_vec = (
            self._metrics.compute(embedding, profile)
        )

        result = LearnedDriftResult(
            machine_type=record.machine_type,
            machine_id=record.machine_id,
            filename=record.filename,
            euclidean_distance=euclid,
            manhattan_distance=manhat,
            cosine_similarity=cosine,
            norm_euclidean_distance=norm_euclid,
            norm_manhattan_distance=norm_manhat,
            norm_cosine_similarity=norm_cosine,
            normalized_vector=norm_vec,
        )

        logger.info(
            "Learned drift — %s/%s '%s'  raw_euclid=%.4f | norm_euclid=%.4f",
            result.machine_type, result.machine_id, result.filename,
            result.euclidean_distance, result.norm_euclidean_distance,
        )
        return result
