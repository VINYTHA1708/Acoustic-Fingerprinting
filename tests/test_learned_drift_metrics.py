"""Tests for LearnedDriftMetrics.

All tests use synthetic NumPy arrays only — no audio, BEATs, PyTorch, or checkpoints.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.learned_drift.metrics import LearnedDriftMetrics, _STD_FLOOR
from src.learned_profile.learned_profile import LearnedFingerprintProfile

DIM = 256


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_profile(
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    dim: int = DIM,
    embedding_dim: int = DIM,
) -> LearnedFingerprintProfile:
    rng = np.random.default_rng(0)
    mean = mean if mean is not None else rng.random(dim).astype(np.float32)
    std = std if std is not None else (rng.random(dim).astype(np.float32) + 0.1)
    embeddings = rng.random((5, dim)).astype(np.float32)
    return LearnedFingerprintProfile(
        machine_type="pump",
        machine_id="id_00",
        embedding_dimension=embedding_dim,
        embeddings=embeddings,
        mean_vector=mean,
        std_vector=std,
    )


@pytest.fixture
def metrics() -> LearnedDriftMetrics:
    return LearnedDriftMetrics()


@pytest.fixture
def profile() -> LearnedFingerprintProfile:
    return _make_profile()


@pytest.fixture
def embedding() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random(DIM).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Successful computation
# ---------------------------------------------------------------------------

def test_successful_computation(metrics, embedding, profile):
    result = metrics.compute(embedding, profile)
    assert result is not None


# ---------------------------------------------------------------------------
# 2. All returned scalar metrics are finite
# ---------------------------------------------------------------------------

def test_all_scalars_finite(metrics, embedding, profile):
    cosine, euclid, manhat, z, abs_diff, norm_euclid, norm_manhat, norm_cosine, norm_vec = (
        metrics.compute(embedding, profile)
    )
    for val in (cosine, euclid, manhat, norm_euclid, norm_manhat, norm_cosine):
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# 3. Correct return tuple length
# ---------------------------------------------------------------------------

def test_return_tuple_length(metrics, embedding, profile):
    result = metrics.compute(embedding, profile)
    assert len(result) == 9


# ---------------------------------------------------------------------------
# 4. z_score_vector shape is (256,)
# ---------------------------------------------------------------------------

def test_z_score_vector_shape(metrics, embedding, profile):
    _, _, _, z, _, _, _, _, _ = metrics.compute(embedding, profile)
    assert z.shape == (DIM,)


# ---------------------------------------------------------------------------
# 5. absolute_difference_vector shape is (256,)
# ---------------------------------------------------------------------------

def test_abs_diff_shape(metrics, embedding, profile):
    _, _, _, _, abs_diff, _, _, _, _ = metrics.compute(embedding, profile)
    assert abs_diff.shape == (DIM,)


# ---------------------------------------------------------------------------
# 6. normalized_vector shape is (256,)
# ---------------------------------------------------------------------------

def test_normalized_vector_shape(metrics, embedding, profile):
    *_, norm_vec = metrics.compute(embedding, profile)
    assert norm_vec.shape == (DIM,)


# ---------------------------------------------------------------------------
# 7. All vector outputs are float32
# ---------------------------------------------------------------------------

def test_vector_dtypes_float32(metrics, embedding, profile):
    _, _, _, z, abs_diff, _, _, _, norm_vec = metrics.compute(embedding, profile)
    assert z.dtype == np.float32
    assert abs_diff.dtype == np.float32
    assert norm_vec.dtype == np.float32


# ---------------------------------------------------------------------------
# 8. Euclidean distance calculation is correct
# ---------------------------------------------------------------------------

def test_euclidean_distance_correct(metrics):
    emb = np.ones(DIM, dtype=np.float32) * 2.0
    mean = np.ones(DIM, dtype=np.float32)
    std = np.ones(DIM, dtype=np.float32)
    profile = _make_profile(mean=mean, std=std)
    _, euclid, _, _, _, _, _, _, _ = metrics.compute(emb, profile)
    expected = float(np.linalg.norm(emb - mean))
    assert abs(euclid - expected) < 1e-5


# ---------------------------------------------------------------------------
# 9. Manhattan distance calculation is correct
# ---------------------------------------------------------------------------

def test_manhattan_distance_correct(metrics):
    emb = np.ones(DIM, dtype=np.float32) * 3.0
    mean = np.ones(DIM, dtype=np.float32)
    std = np.ones(DIM, dtype=np.float32)
    profile = _make_profile(mean=mean, std=std)
    _, _, manhat, _, _, _, _, _, _ = metrics.compute(emb, profile)
    expected = float(np.sum(np.abs(emb - mean)))
    assert abs(manhat - expected) < 1e-4


# ---------------------------------------------------------------------------
# 10. Absolute difference calculation is correct
# ---------------------------------------------------------------------------

def test_abs_diff_correct(metrics):
    emb = np.ones(DIM, dtype=np.float32) * 5.0
    mean = np.ones(DIM, dtype=np.float32) * 2.0
    std = np.ones(DIM, dtype=np.float32)
    profile = _make_profile(mean=mean, std=std)
    _, _, _, _, abs_diff, _, _, _, _ = metrics.compute(emb, profile)
    expected = np.abs(emb - mean).astype(np.float32)
    np.testing.assert_allclose(abs_diff, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# 11. Normalized vector calculation is correct
# ---------------------------------------------------------------------------

def test_normalized_vector_correct(metrics):
    emb = np.ones(DIM, dtype=np.float32) * 3.0
    mean = np.ones(DIM, dtype=np.float32)
    std = np.ones(DIM, dtype=np.float32) * 2.0
    profile = _make_profile(mean=mean, std=std)
    _, _, _, z, _, _, _, _, norm_vec = metrics.compute(emb, profile)
    expected = ((emb - mean) / std).astype(np.float32)
    np.testing.assert_allclose(z, expected, rtol=1e-6)
    np.testing.assert_allclose(norm_vec, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# 12. Normalized Euclidean distance is correct
# ---------------------------------------------------------------------------

def test_normalized_euclidean_correct(metrics):
    emb = np.ones(DIM, dtype=np.float32) * 3.0
    mean = np.ones(DIM, dtype=np.float32)
    std = np.ones(DIM, dtype=np.float32) * 2.0
    profile = _make_profile(mean=mean, std=std)
    _, _, _, z, _, norm_euclid, _, _, _ = metrics.compute(emb, profile)
    expected = float(np.linalg.norm(z))
    assert abs(norm_euclid - expected) < 1e-5


# ---------------------------------------------------------------------------
# 13. Normalized Manhattan distance is correct
# ---------------------------------------------------------------------------

def test_normalized_manhattan_correct(metrics):
    emb = np.ones(DIM, dtype=np.float32) * 3.0
    mean = np.ones(DIM, dtype=np.float32)
    std = np.ones(DIM, dtype=np.float32) * 2.0
    profile = _make_profile(mean=mean, std=std)
    _, _, _, z, _, _, norm_manhat, _, _ = metrics.compute(emb, profile)
    expected = float(np.sum(np.abs(z)))
    assert abs(norm_manhat - expected) < 1e-4


# ---------------------------------------------------------------------------
# 14. Zero embedding norm does not crash
# ---------------------------------------------------------------------------

def test_zero_embedding_norm_no_crash(metrics):
    emb = np.zeros(DIM, dtype=np.float32)
    profile = _make_profile()
    result = metrics.compute(emb, profile)
    assert len(result) == 9


# ---------------------------------------------------------------------------
# 15. Zero profile mean norm does not crash
# ---------------------------------------------------------------------------

def test_zero_profile_mean_norm_no_crash(metrics, embedding):
    mean = np.zeros(DIM, dtype=np.float32)
    profile = _make_profile(mean=mean)
    result = metrics.compute(embedding, profile)
    assert len(result) == 9


# ---------------------------------------------------------------------------
# 16. Zero normalized vector returns normalized cosine similarity 0.0
# ---------------------------------------------------------------------------

def test_zero_normalized_vector_cosine_is_zero(metrics):
    # embedding == mean → z-score vector is all zeros
    mean = np.ones(DIM, dtype=np.float32) * 0.5
    std = np.ones(DIM, dtype=np.float32)
    emb = mean.copy()
    profile = _make_profile(mean=mean, std=std)
    _, _, _, _, _, _, _, norm_cosine, _ = metrics.compute(emb, profile)
    assert norm_cosine == 0.0


# ---------------------------------------------------------------------------
# 17. Wrong embedding dimension raises ValueError
# ---------------------------------------------------------------------------

def test_wrong_embedding_dimension_raises(metrics, profile):
    bad_emb = np.ones(128, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        metrics.compute(bad_emb, profile)


# ---------------------------------------------------------------------------
# 18. Non-1D embedding raises ValueError
# ---------------------------------------------------------------------------

def test_non_1d_embedding_raises(metrics, profile):
    bad_emb = np.ones((DIM, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="one-dimensional"):
        metrics.compute(bad_emb, profile)


# ---------------------------------------------------------------------------
# 19. NaN embedding raises ValueError
# ---------------------------------------------------------------------------

def test_nan_embedding_raises(metrics, profile):
    bad_emb = np.ones(DIM, dtype=np.float32)
    bad_emb[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        metrics.compute(bad_emb, profile)


# ---------------------------------------------------------------------------
# 20. Inf embedding raises ValueError
# ---------------------------------------------------------------------------

def test_inf_embedding_raises(metrics, profile):
    bad_emb = np.ones(DIM, dtype=np.float32)
    bad_emb[5] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        metrics.compute(bad_emb, profile)


# ---------------------------------------------------------------------------
# 21. Wrong profile embedding_dimension raises ValueError
# ---------------------------------------------------------------------------

def test_wrong_profile_embedding_dimension_raises(metrics, embedding):
    profile = _make_profile(embedding_dim=128)
    with pytest.raises(ValueError, match="embedding_dimension"):
        metrics.compute(embedding, profile)


# ---------------------------------------------------------------------------
# 22. Wrong mean_vector shape raises ValueError
# ---------------------------------------------------------------------------

def test_wrong_mean_vector_shape_raises(metrics, embedding):
    bad_mean = np.ones(128, dtype=np.float32)
    profile = _make_profile(mean=bad_mean)
    with pytest.raises(ValueError, match="mean_vector"):
        metrics.compute(embedding, profile)


# ---------------------------------------------------------------------------
# 23. Wrong std_vector shape raises ValueError
# ---------------------------------------------------------------------------

def test_wrong_std_vector_shape_raises(metrics, embedding):
    bad_std = np.ones(128, dtype=np.float32)
    profile = _make_profile(std=bad_std)
    with pytest.raises(ValueError, match="std_vector"):
        metrics.compute(embedding, profile)


# ---------------------------------------------------------------------------
# 24. NaN in profile mean_vector raises ValueError
# ---------------------------------------------------------------------------

def test_nan_in_mean_vector_raises(metrics, embedding):
    mean = np.ones(DIM, dtype=np.float32)
    mean[0] = np.nan
    profile = _make_profile(mean=mean)
    with pytest.raises(ValueError, match="mean_vector"):
        metrics.compute(embedding, profile)


# ---------------------------------------------------------------------------
# 25. Inf in profile mean_vector raises ValueError
# ---------------------------------------------------------------------------

def test_inf_in_mean_vector_raises(metrics, embedding):
    mean = np.ones(DIM, dtype=np.float32)
    mean[0] = np.inf
    profile = _make_profile(mean=mean)
    with pytest.raises(ValueError, match="mean_vector"):
        metrics.compute(embedding, profile)


# ---------------------------------------------------------------------------
# 26. NaN in profile std_vector raises ValueError
# ---------------------------------------------------------------------------

def test_nan_in_std_vector_raises(metrics, embedding):
    std = np.ones(DIM, dtype=np.float32)
    std[0] = np.nan
    profile = _make_profile(std=std)
    with pytest.raises(ValueError, match="std_vector"):
        metrics.compute(embedding, profile)


# ---------------------------------------------------------------------------
# 27. Inf in profile std_vector raises ValueError
# ---------------------------------------------------------------------------

def test_inf_in_std_vector_raises(metrics, embedding):
    std = np.ones(DIM, dtype=np.float32)
    std[0] = np.inf
    profile = _make_profile(std=std)
    with pytest.raises(ValueError, match="std_vector"):
        metrics.compute(embedding, profile)


# ---------------------------------------------------------------------------
# 28. std values below _STD_FLOOR do not cause division by zero
# ---------------------------------------------------------------------------

def test_std_below_floor_no_division_by_zero(metrics, embedding):
    std = np.full(DIM, _STD_FLOOR / 10, dtype=np.float32)  # all below floor
    profile = _make_profile(std=std)
    _, _, _, z, _, _, _, _, norm_vec = metrics.compute(embedding, profile)
    assert not np.isnan(z).any()
    assert not np.isinf(z).any()
    assert not np.isnan(norm_vec).any()
    assert not np.isinf(norm_vec).any()
    # All deviations treated as zero
    np.testing.assert_array_equal(z, np.zeros(DIM, dtype=np.float32))
