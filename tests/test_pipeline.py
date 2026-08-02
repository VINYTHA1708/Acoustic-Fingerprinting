"""Tests for InferencePipeline, PipelineResult, and PipelineBenchmark
(src/pipeline/, src/benchmark/).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from conftest import CONTRASTIVE_CHECKPOINT
import importlib

PipelineResult = importlib.import_module("src.pipeline.result").PipelineResult

_VALID_STATES = {"EXCELLENT", "GOOD", "WARNING", "CRITICAL"}


@pytest.fixture(scope="module")
def pipeline_result(first_normal_record, learned_profile):
    import importlib
    InferencePipeline = importlib.import_module("src.pipeline.pipeline").InferencePipeline

    pipeline = InferencePipeline(checkpoint_path=CONTRASTIVE_CHECKPOINT)
    return pipeline.analyze(first_normal_record, learned_profile)


@pytest.fixture(scope="module")
def benchmark_result(first_normal_record, learned_profile):
    import importlib
    PipelineBenchmark = importlib.import_module("src.benchmark.benchmark").PipelineBenchmark

    bench = PipelineBenchmark(checkpoint_path=CONTRASTIVE_CHECKPOINT)
    return bench.benchmark(first_normal_record, learned_profile)


class TestInferencePipeline:
    """InferencePipeline execution and output tests."""

    def test_executes_successfully(self, pipeline_result):
        """InferencePipeline.analyze() returns a PipelineResult without raising."""
        assert pipeline_result is not None

    def test_dsp_dimension_is_153(self, pipeline_result):
        """DSP dimension is exactly 153."""
        assert pipeline_result.dsp_dimension == 153

    def test_beats_dimension_is_768(self, pipeline_result):
        """BEATs dimension is exactly 768."""
        assert pipeline_result.beats_dimension == 768

    def test_fusion_dimension_is_921(self, pipeline_result):
        """Fusion dimension is exactly 921."""
        assert pipeline_result.fusion_dimension == 921

    def test_embedding_dimension_is_256(self, pipeline_result):
        """Embedding dimension is exactly 256."""
        assert pipeline_result.embedding_dimension == 256

    def test_health_score_lower_bound(self, pipeline_result):
        """Health score is >= 0."""
        assert pipeline_result.health_score >= 0.0

    def test_health_score_upper_bound(self, pipeline_result):
        """Health score is <= 100."""
        assert pipeline_result.health_score <= 100.0

    def test_raw_euclidean_is_finite(self, pipeline_result):
        """Raw Euclidean drift metric is finite."""
        assert math.isfinite(pipeline_result.raw_euclidean)

    def test_raw_manhattan_is_finite(self, pipeline_result):
        """Raw Manhattan drift metric is finite."""
        assert math.isfinite(pipeline_result.raw_manhattan)

    def test_raw_cosine_is_finite(self, pipeline_result):
        """Raw cosine drift metric is finite."""
        assert math.isfinite(pipeline_result.raw_cosine)

    def test_normalized_euclidean_is_finite(self, pipeline_result):
        """Normalized Euclidean drift metric is finite."""
        assert math.isfinite(pipeline_result.normalized_euclidean)

    def test_normalized_manhattan_is_finite(self, pipeline_result):
        """Normalized Manhattan drift metric is finite."""
        assert math.isfinite(pipeline_result.normalized_manhattan)

    def test_normalized_cosine_is_finite(self, pipeline_result):
        """Normalized cosine drift metric is finite."""
        assert math.isfinite(pipeline_result.normalized_cosine)

    def test_health_state_is_valid(self, pipeline_result):
        """Health state is one of the four valid bands."""
        assert pipeline_result.health_state in _VALID_STATES


class TestPipelineResultSerialization:
    """PipelineResult to_dict / from_dict round-trip."""

    def test_to_dict_from_dict_round_trip(self, pipeline_result):
        """to_dict() → from_dict() reconstructs an identical PipelineResult."""
        data = pipeline_result.to_dict()
        restored = PipelineResult.from_dict(data)

        assert restored.machine_type == pipeline_result.machine_type
        assert restored.machine_id == pipeline_result.machine_id
        assert restored.filename == pipeline_result.filename
        assert restored.dsp_dimension == pipeline_result.dsp_dimension
        assert restored.beats_dimension == pipeline_result.beats_dimension
        assert restored.fusion_dimension == pipeline_result.fusion_dimension
        assert restored.embedding_dimension == pipeline_result.embedding_dimension
        assert math.isclose(restored.health_score, pipeline_result.health_score)
        assert restored.health_state == pipeline_result.health_state


class TestPipelineBenchmark:
    """PipelineBenchmark execution and output tests."""

    def test_benchmark_executes_successfully(self, benchmark_result):
        """PipelineBenchmark.benchmark() returns a BenchmarkResult without raising."""
        import importlib
        BenchmarkResult = importlib.import_module("src.benchmark.benchmark_result").BenchmarkResult

        assert isinstance(benchmark_result, BenchmarkResult)

    def test_total_time_is_positive(self, benchmark_result):
        """Total benchmark time is > 0."""
        assert benchmark_result.total_time > 0.0

    def test_projection_time_is_non_negative(self, benchmark_result):
        """Projection stage time is >= 0."""
        assert benchmark_result.projection_time >= 0.0

    def test_drift_time_is_non_negative(self, benchmark_result):
        """Drift stage time is >= 0."""
        assert benchmark_result.drift_time >= 0.0

    def test_health_time_is_non_negative(self, benchmark_result):
        """Health stage time is >= 0."""
        assert benchmark_result.health_time >= 0.0

    def test_cache_hit_is_bool(self, benchmark_result):
        """cache_hit field is a boolean."""
        assert isinstance(benchmark_result.cache_hit, bool)

    def test_dimensions_match_pipeline(self, benchmark_result):
        """Benchmark reports the same dimensions as the pipeline."""
        assert benchmark_result.dsp_dimension == 153
        assert benchmark_result.beats_dimension == 768
        assert benchmark_result.fusion_dimension == 921
        assert benchmark_result.embedding_dimension == 256

    def test_benchmark_result_serialization_round_trip(self, benchmark_result):
        """BenchmarkResult to_dict() → from_dict() round-trip preserves all fields."""
        import importlib
        BenchmarkResult = importlib.import_module("src.benchmark.benchmark_result").BenchmarkResult

        data = benchmark_result.to_dict()
        restored = BenchmarkResult.from_dict(data)

        assert restored.machine_type == benchmark_result.machine_type
        assert restored.cache_hit == benchmark_result.cache_hit
        assert math.isclose(restored.total_time, benchmark_result.total_time)
        assert restored.dsp_dimension == benchmark_result.dsp_dimension
