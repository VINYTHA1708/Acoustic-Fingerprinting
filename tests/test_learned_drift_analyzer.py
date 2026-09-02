"""Unit tests for LearnedDriftAnalyzer (src/learned_drift/analyzer.py).

All heavy dependencies (BEATsEncoder, ProjectionHead, FusionCache,
ContrastiveInference, PreprocessingPipeline, etc.) are mocked.
No audio files, checkpoints, or filesystem access required.

sys.modules stubs for librosa / torch are installed at module level so that
the src package tree can be imported without those libraries being present.
"""

from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------
# Install lightweight stubs BEFORE any src.* import so that the eager
# __init__ chains (librosa, torch, torchaudio) do not raise ImportError.
# ---------------------------------------------------------------------------

def _stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# Pre-import real librosa so the stub loop below does not shadow it.
# librosa is installed in this environment; importing it here ensures
# sys.modules already contains the real module before the guard check.
try:
    import librosa as _librosa_real          # noqa: F401
    import librosa.feature as _lf_real       # noqa: F401
    import librosa.effects as _le_real       # noqa: F401
    import librosa.core as _lc_real          # noqa: F401
except ImportError:
    pass  # not installed — stubs will be used as before

for _name in [
    "librosa", "librosa.feature", "librosa.effects", "librosa.core",
    "torch", "torch.nn", "torch.nn.functional", "torch.optim",
    "torch.utils", "torch.utils.data",
    "torchaudio", "torchaudio.transforms",
    "torchaudio.compliance", "torchaudio.compliance.kaldi",
    "faiss",
    # third_party/beats modules added to sys.path by beats/encoder.py
    "BEATs", "backbone",
]:
    if _name not in sys.modules:
        _stub(_name)

_nn = sys.modules["torch.nn"]
sys.modules["torch"].nn = _nn

# Populate torch.nn with the class names that third_party/beats/BEATs.py imports
for _attr in [
    "Module", "Linear", "LayerNorm", "Dropout", "MultiheadAttention",
    "Conv1d", "Conv2d", "BatchNorm1d", "BatchNorm2d", "GELU", "ReLU",
    "Sequential", "Embedding", "Parameter", "ModuleList",
]:
    if not hasattr(_nn, _attr):
        setattr(_nn, _attr, type(_attr, (object,), {}))

# BEATs stub needs BEATs and BEATsConfig classes
_beats_mod = sys.modules["BEATs"]
if not hasattr(_beats_mod, "BEATs"):
    _beats_mod.BEATs = type("BEATs", (object,), {})
if not hasattr(_beats_mod, "BEATsConfig"):
    _beats_mod.BEATsConfig = type("BEATsConfig", (object,), {})

# ---------------------------------------------------------------------------
# Now safe to import src modules
# ---------------------------------------------------------------------------

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.dataset.metadata import AudioMetadata
from src.fusion.fused_vector import FusedFeatureVector
from src.learned_drift.learned_drift_result import LearnedDriftResult
from src.learned_profile.learned_profile import LearnedFingerprintProfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 256
_FAKE_CHECKPOINT = Path("/fake/checkpoint.pt")
_PATCH_BASE = "src.learned_drift.analyzer"

# ---------------------------------------------------------------------------
# Synthetic test data helpers
# ---------------------------------------------------------------------------


def _make_record(machine_type: str = "pump", machine_id: str = "id_00") -> AudioMetadata:
    return AudioMetadata(
        machine_type=machine_type,
        machine_id=machine_id,
        label="normal",
        filename="00000000.wav",
        relative_path=Path(f"{machine_type}/{machine_id}/normal/00000000.wav"),
        absolute_path=Path(f"/data/{machine_type}/{machine_id}/normal/00000000.wav"),
    )


def _make_profile(
    machine_type: str = "pump", machine_id: str = "id_00"
) -> LearnedFingerprintProfile:
    rng = np.random.default_rng(0)
    return LearnedFingerprintProfile(
        machine_type=machine_type,
        machine_id=machine_id,
        embedding_dimension=DIM,
        embeddings=rng.random((5, DIM)).astype(np.float32),
        mean_vector=rng.random(DIM).astype(np.float32),
        std_vector=(rng.random(DIM).astype(np.float32) + 0.1),
    )


def _make_fused_vector() -> FusedFeatureVector:
    rng = np.random.default_rng(1)
    dsp = rng.random(153).astype(np.float32)
    beats = rng.random(768).astype(np.float32)
    return FusedFeatureVector(
        machine_type="pump",
        machine_id="id_00",
        label="normal",
        filename="00000000.wav",
        sample_rate=16_000,
        dsp_feature_names=[f"f{i}" for i in range(153)],
        dsp_feature_vector=dsp,
        beats_embedding=beats,
        fused_feature_vector=np.concatenate([dsp, beats]),
    )


def _make_embedding() -> np.ndarray:
    rng = np.random.default_rng(2)
    emb = rng.random(DIM).astype(np.float32)
    emb /= np.linalg.norm(emb)   # L2-normalised, as ProjectionHead produces
    return emb


def _make_norm_vec() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.random(DIM).astype(np.float32)


def _metrics_return(embedding: np.ndarray, norm_vec: np.ndarray) -> tuple:
    """Fake 9-tuple matching LearnedDriftMetrics.compute() return signature:
    (cosine, euclid, manhat, z, abs_diff, norm_euclid, norm_manhat, norm_cosine, norm_vec)
    """
    z = norm_vec.copy()
    abs_diff = np.abs(embedding).astype(np.float32)
    return (0.91, 1.23, 45.6, z, abs_diff, 0.55, 12.3, 0.77, norm_vec)


# ---------------------------------------------------------------------------
# Builder: instantiate LearnedDriftAnalyzer with all heavy deps mocked
# ---------------------------------------------------------------------------


def _build_analyzer(
    fused: FusedFeatureVector,
    embedding: np.ndarray,
    metrics_ret: tuple,
):
    """Return (analyzer, mock_cache, mock_inference, mock_metrics)."""
    from src.learned_drift.analyzer import LearnedDriftAnalyzer

    mock_cache = MagicMock()
    mock_cache.load_or_create.return_value = fused

    mock_inference = MagicMock()
    mock_inference.generate_fingerprint.return_value = embedding

    mock_metrics = MagicMock()
    mock_metrics.compute.return_value = metrics_ret

    with (
        patch(f"{_PATCH_BASE}.PreprocessingPipeline"),
        patch(f"{_PATCH_BASE}.FeatureExtractor"),
        patch(f"{_PATCH_BASE}.FeatureVectorBuilder"),
        patch(f"{_PATCH_BASE}.BEATsEncoder"),
        patch(f"{_PATCH_BASE}.FusionBuilder"),
        patch(f"{_PATCH_BASE}.FusionCache", return_value=mock_cache),
        patch(f"{_PATCH_BASE}.ProjectionHead"),
        patch(f"{_PATCH_BASE}.ContrastiveInference", return_value=mock_inference),
        patch(f"{_PATCH_BASE}.LearnedDriftMetrics", return_value=mock_metrics),
    ):
        analyzer = LearnedDriftAnalyzer(checkpoint_path=_FAKE_CHECKPOINT)

    # Inject mocks directly so they survive outside the patch context
    analyzer._cache = mock_cache
    analyzer._inference = mock_inference
    analyzer._metrics = mock_metrics
    return analyzer, mock_cache, mock_inference, mock_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def record() -> AudioMetadata:
    return _make_record()


@pytest.fixture
def profile() -> LearnedFingerprintProfile:
    return _make_profile()


@pytest.fixture
def fused() -> FusedFeatureVector:
    return _make_fused_vector()


@pytest.fixture
def embedding() -> np.ndarray:
    return _make_embedding()


@pytest.fixture
def norm_vec() -> np.ndarray:
    return _make_norm_vec()


@pytest.fixture
def metrics_ret(embedding, norm_vec) -> tuple:
    return _metrics_return(embedding, norm_vec)


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------


class TestLearnedDriftAnalyzerInit:

    def test_init_succeeds_with_mocked_deps(self, fused, embedding, metrics_ret):
        """LearnedDriftAnalyzer.__init__ completes without error when deps are mocked."""
        analyzer, _, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        assert analyzer is not None


# ---------------------------------------------------------------------------
# 2–4. analyze() return type and call counts
# ---------------------------------------------------------------------------


class TestAnalyzeCallBehavior:

    def test_returns_learned_drift_result(self, record, profile, fused, embedding, metrics_ret):
        """analyze() returns a LearnedDriftResult instance."""
        analyzer, _, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        result = analyzer.analyze(record, profile)
        assert isinstance(result, LearnedDriftResult)

    def test_cache_load_or_create_called_once(
        self, record, profile, fused, embedding, metrics_ret
    ):
        """FusionCache.load_or_create() is called exactly once per analyze() call."""
        analyzer, mock_cache, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        analyzer.analyze(record, profile)
        mock_cache.load_or_create.assert_called_once()

    def test_generate_fingerprint_called_once(
        self, record, profile, fused, embedding, metrics_ret
    ):
        """ContrastiveInference.generate_fingerprint() is called exactly once."""
        analyzer, _, mock_inference, _ = _build_analyzer(fused, embedding, metrics_ret)
        analyzer.analyze(record, profile)
        mock_inference.generate_fingerprint.assert_called_once()

    def test_metrics_compute_called_once(self, record, profile, fused, embedding, metrics_ret):
        """LearnedDriftMetrics.compute() is called exactly once."""
        analyzer, _, _, mock_metrics = _build_analyzer(fused, embedding, metrics_ret)
        analyzer.analyze(record, profile)
        mock_metrics.compute.assert_called_once()


# ---------------------------------------------------------------------------
# 5–8. Correct arguments passed to each dependency
# ---------------------------------------------------------------------------


class TestAnalyzeArgumentPassing:

    def test_cache_receives_exact_record(self, record, profile, fused, embedding, metrics_ret):
        """FusionCache.load_or_create() receives the exact AudioMetadata record."""
        analyzer, mock_cache, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        analyzer.analyze(record, profile)
        mock_cache.load_or_create.assert_called_once_with(record)

    def test_generate_fingerprint_receives_fused_vector(
        self, record, profile, fused, embedding, metrics_ret
    ):
        """generate_fingerprint() receives the FusedFeatureVector returned by the cache."""
        analyzer, _, mock_inference, _ = _build_analyzer(fused, embedding, metrics_ret)
        analyzer.analyze(record, profile)
        mock_inference.generate_fingerprint.assert_called_once_with(fused)

    def test_metrics_compute_receives_embedding_and_profile(
        self, record, profile, fused, embedding, metrics_ret
    ):
        """metrics.compute() receives the embedding from inference and the exact profile."""
        analyzer, _, _, mock_metrics = _build_analyzer(fused, embedding, metrics_ret)
        analyzer.analyze(record, profile)
        mock_metrics.compute.assert_called_once_with(embedding, profile)


# ---------------------------------------------------------------------------
# 9–11. Result metadata fields
# ---------------------------------------------------------------------------


class TestResultMetadataFields:

    def test_result_machine_type_matches_record(
        self, record, profile, fused, embedding, metrics_ret
    ):
        """result.machine_type matches record.machine_type."""
        analyzer, _, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        result = analyzer.analyze(record, profile)
        assert result.machine_type == record.machine_type

    def test_result_machine_id_matches_record(
        self, record, profile, fused, embedding, metrics_ret
    ):
        """result.machine_id matches record.machine_id."""
        analyzer, _, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        result = analyzer.analyze(record, profile)
        assert result.machine_id == record.machine_id

    def test_result_filename_matches_record(
        self, record, profile, fused, embedding, metrics_ret
    ):
        """result.filename matches record.filename."""
        analyzer, _, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        result = analyzer.analyze(record, profile)
        assert result.filename == record.filename


# ---------------------------------------------------------------------------
# 12–14. Raw and normalized metric values transferred correctly
# ---------------------------------------------------------------------------


class TestResultMetricValues:

    def test_raw_euclidean_transferred(self, record, profile, fused, embedding, metrics_ret):
        """euclidean_distance matches index 1 of the metrics tuple."""
        analyzer, _, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        result = analyzer.analyze(record, profile)
        assert result.euclidean_distance == metrics_ret[1]

    def test_raw_manhattan_transferred(self, record, profile, fused, embedding, metrics_ret):
        """manhattan_distance matches index 2 of the metrics tuple."""
        analyzer, _, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        result = analyzer.analyze(record, profile)
        assert result.manhattan_distance == metrics_ret[2]

    def test_raw_cosine_transferred(self, record, profile, fused, embedding, metrics_ret):
        """cosine_similarity matches index 0 of the metrics tuple."""
        analyzer, _, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        result = analyzer.analyze(record, profile)
        assert result.cosine_similarity == metrics_ret[0]

    def test_norm_euclidean_transferred(self, record, profile, fused, embedding, metrics_ret):
        """norm_euclidean_distance matches index 5 of the metrics tuple."""
        analyzer, _, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        result = analyzer.analyze(record, profile)
        assert result.norm_euclidean_distance == metrics_ret[5]

    def test_norm_manhattan_transferred(self, record, profile, fused, embedding, metrics_ret):
        """norm_manhattan_distance matches index 6 of the metrics tuple."""
        analyzer, _, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        result = analyzer.analyze(record, profile)
        assert result.norm_manhattan_distance == metrics_ret[6]

    def test_norm_cosine_transferred(self, record, profile, fused, embedding, metrics_ret):
        """norm_cosine_similarity matches index 7 of the metrics tuple."""
        analyzer, _, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        result = analyzer.analyze(record, profile)
        assert result.norm_cosine_similarity == metrics_ret[7]

    def test_normalized_vector_transferred(
        self, record, profile, fused, embedding, metrics_ret
    ):
        """normalized_vector matches index 8 (norm_vec) of the metrics tuple."""
        analyzer, _, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        result = analyzer.analyze(record, profile)
        np.testing.assert_array_equal(result.normalized_vector, metrics_ret[8])


# ---------------------------------------------------------------------------
# 15–18. Mismatch validation — ValueError raised before any dep is called
# ---------------------------------------------------------------------------


class TestMismatchValidation:

    def test_machine_type_mismatch_raises_value_error(self, fused, embedding, metrics_ret):
        """machine_type mismatch raises ValueError."""
        record = _make_record(machine_type="fan", machine_id="id_00")
        profile = _make_profile(machine_type="pump", machine_id="id_00")
        analyzer, _, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        with pytest.raises(ValueError, match="machine_type"):
            analyzer.analyze(record, profile)

    def test_machine_id_mismatch_raises_value_error(self, fused, embedding, metrics_ret):
        """machine_id mismatch raises ValueError."""
        record = _make_record(machine_type="pump", machine_id="id_99")
        profile = _make_profile(machine_type="pump", machine_id="id_00")
        analyzer, _, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        with pytest.raises(ValueError, match="machine_id"):
            analyzer.analyze(record, profile)

    def test_machine_type_mismatch_no_cache_call(self, fused, embedding, metrics_ret):
        """On machine_type mismatch, cache.load_or_create() is NOT called."""
        record = _make_record(machine_type="fan", machine_id="id_00")
        profile = _make_profile(machine_type="pump", machine_id="id_00")
        analyzer, mock_cache, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        with pytest.raises(ValueError):
            analyzer.analyze(record, profile)
        mock_cache.load_or_create.assert_not_called()

    def test_machine_type_mismatch_no_inference_call(self, fused, embedding, metrics_ret):
        """On machine_type mismatch, generate_fingerprint() is NOT called."""
        record = _make_record(machine_type="fan", machine_id="id_00")
        profile = _make_profile(machine_type="pump", machine_id="id_00")
        analyzer, _, mock_inference, _ = _build_analyzer(fused, embedding, metrics_ret)
        with pytest.raises(ValueError):
            analyzer.analyze(record, profile)
        mock_inference.generate_fingerprint.assert_not_called()

    def test_machine_type_mismatch_no_metrics_call(self, fused, embedding, metrics_ret):
        """On machine_type mismatch, metrics.compute() is NOT called."""
        record = _make_record(machine_type="fan", machine_id="id_00")
        profile = _make_profile(machine_type="pump", machine_id="id_00")
        analyzer, _, _, mock_metrics = _build_analyzer(fused, embedding, metrics_ret)
        with pytest.raises(ValueError):
            analyzer.analyze(record, profile)
        mock_metrics.compute.assert_not_called()

    def test_machine_id_mismatch_no_cache_call(self, fused, embedding, metrics_ret):
        """On machine_id mismatch, cache.load_or_create() is NOT called."""
        record = _make_record(machine_type="pump", machine_id="id_99")
        profile = _make_profile(machine_type="pump", machine_id="id_00")
        analyzer, mock_cache, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        with pytest.raises(ValueError):
            analyzer.analyze(record, profile)
        mock_cache.load_or_create.assert_not_called()

    def test_machine_id_mismatch_no_inference_call(self, fused, embedding, metrics_ret):
        """On machine_id mismatch, generate_fingerprint() is NOT called."""
        record = _make_record(machine_type="pump", machine_id="id_99")
        profile = _make_profile(machine_type="pump", machine_id="id_00")
        analyzer, _, mock_inference, _ = _build_analyzer(fused, embedding, metrics_ret)
        with pytest.raises(ValueError):
            analyzer.analyze(record, profile)
        mock_inference.generate_fingerprint.assert_not_called()

    def test_machine_id_mismatch_no_metrics_call(self, fused, embedding, metrics_ret):
        """On machine_id mismatch, metrics.compute() is NOT called."""
        record = _make_record(machine_type="pump", machine_id="id_99")
        profile = _make_profile(machine_type="pump", machine_id="id_00")
        analyzer, _, _, mock_metrics = _build_analyzer(fused, embedding, metrics_ret)
        with pytest.raises(ValueError):
            analyzer.analyze(record, profile)
        mock_metrics.compute.assert_not_called()


# ---------------------------------------------------------------------------
# 19–21. Exception propagation
# ---------------------------------------------------------------------------


class TestExceptionPropagation:

    def test_cache_exception_propagates(self, record, profile, fused, embedding, metrics_ret):
        """Exceptions from FusionCache.load_or_create() propagate unchanged."""
        analyzer, mock_cache, _, _ = _build_analyzer(fused, embedding, metrics_ret)
        mock_cache.load_or_create.side_effect = RuntimeError("cache failure")
        with pytest.raises(RuntimeError, match="cache failure"):
            analyzer.analyze(record, profile)

    def test_inference_exception_propagates(
        self, record, profile, fused, embedding, metrics_ret
    ):
        """Exceptions from ContrastiveInference.generate_fingerprint() propagate unchanged."""
        analyzer, _, mock_inference, _ = _build_analyzer(fused, embedding, metrics_ret)
        mock_inference.generate_fingerprint.side_effect = RuntimeError("inference failure")
        with pytest.raises(RuntimeError, match="inference failure"):
            analyzer.analyze(record, profile)

    def test_metrics_exception_propagates(self, record, profile, fused, embedding, metrics_ret):
        """Exceptions from LearnedDriftMetrics.compute() propagate unchanged."""
        analyzer, _, _, mock_metrics = _build_analyzer(fused, embedding, metrics_ret)
        mock_metrics.compute.side_effect = ValueError("metrics failure")
        with pytest.raises(ValueError, match="metrics failure"):
            analyzer.analyze(record, profile)
