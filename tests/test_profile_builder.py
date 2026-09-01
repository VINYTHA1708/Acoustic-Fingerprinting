"""Tests for LearnedFingerprintProfileBuilder
(src/contrastive_learning/profile_builder.py).

All tests are fully isolated — no real audio, BEATs model, dataset, or
checkpoint is used.  ContrastiveInference.generate_fingerprint is mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.contrastive_learning.profile_builder import LearnedFingerprintProfileBuilder
from src.fusion.fused_vector import FusedFeatureVector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 256
_MACHINE_TYPE = "pump"
_MACHINE_ID = "id_00"


def _make_fused(
    machine_type: str = _MACHINE_TYPE,
    machine_id: str = _MACHINE_ID,
    filename: str = "rec_00.wav",
) -> FusedFeatureVector:
    """Return a minimal FusedFeatureVector with fake numpy arrays."""
    return FusedFeatureVector(
        machine_type=machine_type,
        machine_id=machine_id,
        label="normal",
        filename=filename,
        sample_rate=16_000,
        dsp_feature_names=["f0"],
        dsp_feature_vector=np.zeros(153, dtype=np.float32),
        beats_embedding=np.zeros(768, dtype=np.float32),
        fused_feature_vector=np.zeros(921, dtype=np.float32),
    )


def _l2_unit(dim: int = _DIM, seed: int = 0) -> np.ndarray:
    """Return a deterministic L2-normalised float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _make_inference(embeddings: list[np.ndarray]) -> MagicMock:
    """Return a mock ContrastiveInference whose generate_fingerprint returns
    each embedding in *embeddings* on successive calls."""
    mock = MagicMock()
    mock.generate_fingerprint.side_effect = embeddings
    return mock


def _make_builder(embeddings: list[np.ndarray]) -> LearnedFingerprintProfileBuilder:
    return LearnedFingerprintProfileBuilder(_make_inference(embeddings))


def _default_vectors(n: int = 3) -> list[FusedFeatureVector]:
    return [_make_fused(filename=f"rec_{i:02d}.wav") for i in range(n)]


def _default_embeddings(n: int = 3) -> list[np.ndarray]:
    return [_l2_unit(seed=i) for i in range(n)]


# ---------------------------------------------------------------------------
# Successful profile creation
# ---------------------------------------------------------------------------


class TestProfileBuilderSuccess:

    def test_builds_successfully(self):
        """build() returns a non-None LearnedFingerprintProfile."""
        profile = _make_builder(_default_embeddings()).build(
            _MACHINE_TYPE, _MACHINE_ID, _default_vectors()
        )
        assert profile is not None

    def test_correct_machine_type(self):
        profile = _make_builder(_default_embeddings()).build(
            _MACHINE_TYPE, _MACHINE_ID, _default_vectors()
        )
        assert profile.machine_type == _MACHINE_TYPE

    def test_correct_machine_id(self):
        profile = _make_builder(_default_embeddings()).build(
            _MACHINE_TYPE, _MACHINE_ID, _default_vectors()
        )
        assert profile.machine_id == _MACHINE_ID

    def test_embedding_dimension_is_256(self):
        profile = _make_builder(_default_embeddings()).build(
            _MACHINE_TYPE, _MACHINE_ID, _default_vectors()
        )
        assert profile.embedding_dimension == 256

    def test_embeddings_shape(self):
        n = 4
        profile = _make_builder(_default_embeddings(n)).build(
            _MACHINE_TYPE, _MACHINE_ID, _default_vectors(n)
        )
        assert profile.embeddings.shape == (n, 256)

    def test_mean_vector_shape(self):
        profile = _make_builder(_default_embeddings()).build(
            _MACHINE_TYPE, _MACHINE_ID, _default_vectors()
        )
        assert profile.mean_vector.shape == (256,)

    def test_std_vector_shape(self):
        profile = _make_builder(_default_embeddings()).build(
            _MACHINE_TYPE, _MACHINE_ID, _default_vectors()
        )
        assert profile.std_vector.shape == (256,)

    def test_embeddings_dtype_float32(self):
        profile = _make_builder(_default_embeddings()).build(
            _MACHINE_TYPE, _MACHINE_ID, _default_vectors()
        )
        assert profile.embeddings.dtype == np.float32

    def test_mean_vector_dtype_float32(self):
        profile = _make_builder(_default_embeddings()).build(
            _MACHINE_TYPE, _MACHINE_ID, _default_vectors()
        )
        assert profile.mean_vector.dtype == np.float32

    def test_std_vector_dtype_float32(self):
        profile = _make_builder(_default_embeddings()).build(
            _MACHINE_TYPE, _MACHINE_ID, _default_vectors()
        )
        assert profile.std_vector.dtype == np.float32


# ---------------------------------------------------------------------------
# Mean and std correctness
# ---------------------------------------------------------------------------


class TestProfileBuilderStatistics:

    def _build_with_known_embeddings(self, embs: list[np.ndarray]):
        vectors = [_make_fused(filename=f"rec_{i}.wav") for i in range(len(embs))]
        return _make_builder(embs).build(_MACHINE_TYPE, _MACHINE_ID, vectors)

    def test_mean_vector_correct(self):
        embs = [_l2_unit(seed=i) for i in range(5)]
        profile = self._build_with_known_embeddings(embs)
        expected = np.stack(embs, axis=0).mean(axis=0).astype(np.float32)
        np.testing.assert_array_almost_equal(profile.mean_vector, expected)

    def test_std_vector_correct(self):
        embs = [_l2_unit(seed=i) for i in range(5)]
        profile = self._build_with_known_embeddings(embs)
        expected = np.stack(embs, axis=0).std(axis=0).astype(np.float32)
        np.testing.assert_array_almost_equal(profile.std_vector, expected)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestProfileBuilderValidation:

    def test_empty_fused_vectors_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _make_builder([]).build(_MACHINE_TYPE, _MACHINE_ID, [])

    def test_machine_type_mismatch_raises(self):
        wrong = _make_fused(machine_type="valve")
        with pytest.raises(ValueError, match="machine_type"):
            _make_builder([_l2_unit()]).build(_MACHINE_TYPE, _MACHINE_ID, [wrong])

    def test_machine_id_mismatch_raises(self):
        wrong = _make_fused(machine_id="id_99")
        with pytest.raises(ValueError, match="machine_id"):
            _make_builder([_l2_unit()]).build(_MACHINE_TYPE, _MACHINE_ID, [wrong])

    def test_invalid_embedding_dimension_raises(self):
        bad_emb = np.ones(128, dtype=np.float32)  # wrong dim
        with pytest.raises(ValueError, match="dimension"):
            _make_builder([bad_emb]).build(_MACHINE_TYPE, _MACHINE_ID, [_make_fused()])

    def test_nan_embedding_raises(self):
        nan_emb = np.full(_DIM, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            _make_builder([nan_emb]).build(_MACHINE_TYPE, _MACHINE_ID, [_make_fused()])

    def test_inf_embedding_raises(self):
        inf_emb = np.full(_DIM, np.inf, dtype=np.float32)
        with pytest.raises(ValueError, match="Inf"):
            _make_builder([inf_emb]).build(_MACHINE_TYPE, _MACHINE_ID, [_make_fused()])
