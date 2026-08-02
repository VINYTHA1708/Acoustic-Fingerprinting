"""Tests for BEATsEncoder (src/beats/encoder.py)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import BEATS_CHECKPOINT


class TestBEATsEncoder:
    """BEATsEncoder unit tests."""

    def test_loads_successfully(self, beats_encoder):
        """BEATsEncoder constructs without raising."""
        assert beats_encoder is not None

    def test_embedding_dimension_is_768(self, cached_fused_vector):
        """The BEATs embedding stored in the cached fused vector is 768-dim."""
        assert cached_fused_vector.beats_embedding.shape == (768,)

    def test_embedding_no_nan(self, cached_fused_vector):
        """BEATs embedding contains no NaN values."""
        assert not np.isnan(cached_fused_vector.beats_embedding).any()

    def test_embedding_no_inf(self, cached_fused_vector):
        """BEATs embedding contains no Inf values."""
        assert not np.isinf(cached_fused_vector.beats_embedding).any()

    def test_invalid_checkpoint_raises_file_not_found(self):
        """Passing a non-existent checkpoint path raises FileNotFoundError."""
        from src.beats.encoder import BEATsEncoder

        with pytest.raises(FileNotFoundError):
            BEATsEncoder("/nonexistent/path/beats.pt")
