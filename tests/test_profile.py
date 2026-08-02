"""Tests for LearnedProfileBuilder and LearnedProfileSerializer
(src/learned_profile/).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import CONTRASTIVE_CHECKPOINT, DATASET_ROOT, MACHINE_ID, MACHINE_TYPE


class TestLearnedFingerprintProfile:
    """Profile construction and shape tests."""

    def test_profile_builds_successfully(self, learned_profile):
        """LearnedProfileBuilder.build() returns a non-None profile."""
        assert learned_profile is not None

    def test_mean_vector_shape(self, learned_profile):
        """Profile mean_vector has shape (256,)."""
        assert learned_profile.mean_vector.shape == (256,)

    def test_std_vector_shape(self, learned_profile):
        """Profile std_vector has shape (256,)."""
        assert learned_profile.std_vector.shape == (256,)

    def test_embeddings_matrix_columns(self, learned_profile):
        """Profile embeddings matrix has 256 columns."""
        assert learned_profile.embeddings.shape[1] == 256

    def test_invalid_machine_raises_value_error(self, dataset_loader):
        """Building a profile for a non-existent machine raises ValueError."""
        import importlib
        LearnedProfileBuilder = importlib.import_module("src.learned_profile.builder").LearnedProfileBuilder

        builder = LearnedProfileBuilder(checkpoint_path=CONTRASTIVE_CHECKPOINT)
        with pytest.raises(ValueError):
            builder.build(
                loader=dataset_loader,
                machine_type="nonexistent_type",
                machine_id="id_99",
                max_recordings=5,
            )


class TestLearnedProfileSerializer:
    """Serializer JSON and NPZ round-trip tests."""

    def test_json_round_trip(self, learned_profile):
        """JSON save → load preserves machine metadata and vector values."""
        import importlib
        LearnedProfileSerializer = importlib.import_module("src.learned_profile.serializer").LearnedProfileSerializer

        serializer = LearnedProfileSerializer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            serializer.save_json(learned_profile, path)
            loaded = serializer.load_json(path)

        assert loaded.machine_type == learned_profile.machine_type
        assert loaded.machine_id == learned_profile.machine_id
        assert loaded.embedding_dimension == learned_profile.embedding_dimension
        np.testing.assert_array_almost_equal(loaded.mean_vector, learned_profile.mean_vector)
        np.testing.assert_array_almost_equal(loaded.std_vector, learned_profile.std_vector)

    def test_npz_round_trip(self, learned_profile):
        """NPZ save → load preserves machine metadata and vector values."""
        import importlib
        LearnedProfileSerializer = importlib.import_module("src.learned_profile.serializer").LearnedProfileSerializer

        serializer = LearnedProfileSerializer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.npz"
            serializer.save_npz(learned_profile, path)
            loaded = serializer.load_npz(path)

        assert loaded.machine_type == learned_profile.machine_type
        assert loaded.machine_id == learned_profile.machine_id
        assert loaded.embedding_dimension == learned_profile.embedding_dimension
        np.testing.assert_array_almost_equal(loaded.mean_vector, learned_profile.mean_vector)
        np.testing.assert_array_almost_equal(loaded.std_vector, learned_profile.std_vector)
