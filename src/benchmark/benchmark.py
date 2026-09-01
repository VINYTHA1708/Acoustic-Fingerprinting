"""PipelineBenchmark — per-stage inference timing for one recording.

Measures wall-clock time for each pipeline stage using ``time.perf_counter()``:

    Preprocessing → DSP → BEATs → Fusion → ProjectionHead → Drift → Health

Each stage is timed independently by calling the underlying sub-components
directly.  No pipeline logic is duplicated — all computation is delegated to
the existing modules.

Total time wraps the entire inference sequence end-to-end.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from ..beats.encoder import BEATsEncoder
from ..contrastive_learning.inference import ContrastiveInference
from ..contrastive_learning.model import ProjectionHead
from ..dataset.metadata import AudioMetadata
from ..feature_extraction.extractor import FeatureExtractor
from ..feature_extraction.feature_vector import FeatureVectorBuilder
from ..fusion.cache import FusionCache
from ..fusion.fusion import FusionBuilder
from ..learned_drift.metrics import LearnedDriftMetrics
from ..learned_health_index.analyzer import LearnedHealthAnalyzer
from ..learned_health_index.calculator import LearnedHealthCalculator
from ..learned_profile.learned_profile import LearnedFingerprintProfile
from ..preprocessing.pipeline import PreprocessingPipeline
from .benchmark_result import BenchmarkResult

_BEATS_DEFAULT = (
    Path(__file__).resolve().parents[2] / "models" / "beats" / "BEATs_iter3_plus_AS2M.pt"
)
_CACHE_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "fusion_cache"

class PipelineBenchmark:
    """Measures per-stage inference time for one recording.

    Constructs all sub-components once at construction time and reuses them
    across calls to :meth:`benchmark`.

    Args:
        checkpoint_path: Path to the trained ProjectionHead ``.pt`` checkpoint.
        beats_checkpoint: Path to the BEATs model checkpoint.
                          Defaults to ``models/beats/BEATs_iter3_plus_AS2M.pt``.
        cache_root: Directory for the FusionCache.
                    Defaults to ``data/fusion_cache``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        beats_checkpoint: str | Path | None = None,
        cache_root: str | Path | None = None,
    ) -> None:
        beats_ckpt = Path(beats_checkpoint) if beats_checkpoint else _BEATS_DEFAULT
        _cache = Path(cache_root) if cache_root else _CACHE_DEFAULT

        self._pipeline = PreprocessingPipeline(target_sr=16_000)
        self._extractor = FeatureExtractor(sample_rate=16_000)
        self._vec_builder = FeatureVectorBuilder()
        self._encoder = BEATsEncoder(beats_ckpt)
        self._fusion = FusionBuilder()

        self._cache = FusionCache(
            cache_root=_cache,
            pipeline=self._pipeline,
            extractor=self._extractor,
            vec_builder=self._vec_builder,
            encoder=self._encoder,
            fusion=self._fusion,
        )

        head = ProjectionHead()
        self._inference = ContrastiveInference(
            projection_head=head,
            checkpoint_path=checkpoint_path,
        )

        self._drift_metrics = LearnedDriftMetrics()
        self._health_calculator = LearnedHealthCalculator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def benchmark(
        self,
        audio_record: AudioMetadata,
        learned_profile: LearnedFingerprintProfile,
    ) -> BenchmarkResult:
        """Run the full pipeline for one recording and return per-stage timings.

        Each stage is timed individually with ``time.perf_counter()``.
        Total time wraps the entire sequence.

        When the fused vector is already cached, preprocessing / DSP / BEATs /
        fusion times reflect the disk-load path (very fast).  ``cache_hit``
        is set accordingly.

        Args:
            audio_record: :class:`~dataset.metadata.AudioMetadata` for the recording.
            learned_profile: :class:`~learned_profile.learned_profile.LearnedFingerprintProfile`
                             for the same machine.

        Returns:
            :class:`BenchmarkResult` with all stage times and dimension metadata.
        """
        cache_hit = self._cache.exists(audio_record)

        t_total_start = time.perf_counter()

        if cache_hit:
            # Load from cache — attribute the disk-load time to fusion stage;
            # preprocessing / DSP / BEATs are effectively zero (skipped).
            t0 = time.perf_counter()
            fused = self._cache.load(audio_record)
            t_cache_load = time.perf_counter() - t0

            preprocessing_time = 0.0
            dsp_time = 0.0
            beats_time = 0.0
            fusion_time = t_cache_load
        else:
            # --- Preprocessing ---
            t0 = time.perf_counter()
            result = self._pipeline.run(audio_record.absolute_path)
            preprocessing_time = time.perf_counter() - t0

            # --- DSP extraction ---
            t0 = time.perf_counter()
            features = self._extractor.extract(result["waveform"])
            dsp_vector, dsp_names = self._vec_builder.build(features)
            dsp_time = time.perf_counter() - t0

            # --- BEATs encoding ---
            t0 = time.perf_counter()
            embedding = self._encoder.encode(
                waveform=result["waveform"],
                sample_rate=result["sample_rate"],
                filename=audio_record.filename,
            )
            beats_time = time.perf_counter() - t0

            # --- Fusion ---
            t0 = time.perf_counter()
            fused = self._fusion.build(
                dsp_vector=dsp_vector,
                dsp_feature_names=dsp_names,
                beats_embedding=embedding,
                machine_type=audio_record.machine_type,
                machine_id=audio_record.machine_id,
                label=audio_record.label,
            )
            self._cache.save(fused, audio_record)
            fusion_time = time.perf_counter() - t0

        # --- ProjectionHead inference ---
        t0 = time.perf_counter()
        current_embedding = self._inference.generate_fingerprint(fused)
        projection_time = time.perf_counter() - t0

        # --- Drift metrics ---
        t0 = time.perf_counter()
        (
             _cosine,
    _euclid,
    _manhat,
    _z_score_vector,
    _absolute_difference_vector,
    norm_euclid,
    _norm_manhat,
    _norm_cosine,
    _norm_vec,
) = self._drift_metrics.compute(
    current_embedding,
    learned_profile,
)
        drift_time = time.perf_counter() - t0

        # --- Health score ---
        t0 = time.perf_counter()
        mu_norm, sigma_norm = LearnedHealthAnalyzer._profile_norm_stats(learned_profile)
        self._health_calculator.calculate(
            normalized_euclidean=norm_euclid,
            normalized_manhattan=_norm_manhat,
            normalized_cosine=_norm_cosine,
            profile_healthy_norm=mu_norm,
            profile_healthy_norm_std=sigma_norm,
        )
        health_time = time.perf_counter() - t0

        total_time = time.perf_counter() - t_total_start

        return BenchmarkResult(
            machine_type=audio_record.machine_type,
            machine_id=audio_record.machine_id,
            filename=audio_record.filename,
            preprocessing_time=preprocessing_time,
            dsp_time=dsp_time,
            beats_time=beats_time,
            fusion_time=fusion_time,
            projection_time=projection_time,
            drift_time=drift_time,
            health_time=health_time,
            total_time=total_time,
            cache_hit=cache_hit,
            dsp_dimension=int(fused.dsp_feature_vector.shape[0]),
            beats_dimension=int(fused.beats_embedding.shape[0]),
            fusion_dimension=int(fused.fused_feature_vector.shape[0]),
            embedding_dimension=int(learned_profile.embedding_dimension),
        )

