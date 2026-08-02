"""Tests for FusionBuilder (src/fusion/fusion.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load(rel: str):
    """Load a module directly from its .py file, bypassing __init__.py."""
    path = _ROOT / rel
    spec = importlib.util.spec_from_file_location(rel.replace("/", ".").replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load only the leaf modules we need — no __init__.py chain triggered.
_fused_vector_mod = _load("src/fusion/fused_vector.py")
_fusion_mod = _load("src/fusion/fusion.py")
_embedding_mod = _load("src/beats/embedding.py")

FusionBuilder = _fusion_mod.FusionBuilder
BEATsEmbedding = _embedding_mod.BEATsEmbedding


@pytest.fixture(scope="module")
def builder():
    return FusionBuilder()


@pytest.fixture(scope="module")
def valid_dsp():
    return np.random.default_rng(0).random(153).astype(np.float32)


@pytest.fixture(scope="module")
def valid_beats_embedding():
    vec = np.random.default_rng(1).random(768).astype(np.float32)
    return BEATsEmbedding(
        vector=vec,
        embedding_dim=768,
        filename="test.wav",
        machine_type="pump",
        machine_id="id_00",
        sample_rate=16_000,
    )


class TestFusionBuilder:
    """FusionBuilder unit tests."""

    def test_fused_vector_dimension_is_921(self, builder, valid_dsp, valid_beats_embedding):
        """FusionBuilder produces a 921-dimensional fused vector."""
        fused = builder.build(
            dsp_vector=valid_dsp,
            dsp_feature_names=[f"f{i}" for i in range(153)],
            beats_embedding=valid_beats_embedding,
        )
        assert fused.fused_feature_vector.shape == (921,)

    def test_dsp_features_are_first(self, builder, valid_dsp, valid_beats_embedding):
        """The first 153 elements of the fused vector equal the DSP vector."""
        fused = builder.build(
            dsp_vector=valid_dsp,
            dsp_feature_names=[f"f{i}" for i in range(153)],
            beats_embedding=valid_beats_embedding,
        )
        np.testing.assert_array_equal(fused.fused_feature_vector[:153], valid_dsp)

    def test_beats_features_are_second(self, builder, valid_dsp, valid_beats_embedding):
        """The last 768 elements of the fused vector equal the BEATs embedding."""
        fused = builder.build(
            dsp_vector=valid_dsp,
            dsp_feature_names=[f"f{i}" for i in range(153)],
            beats_embedding=valid_beats_embedding,
        )
        np.testing.assert_array_equal(fused.fused_feature_vector[153:], valid_beats_embedding.vector)

    def test_invalid_dsp_empty_raises_value_error(self, builder, valid_beats_embedding):
        """An empty DSP vector raises ValueError."""
        with pytest.raises(ValueError):
            builder.build(
                dsp_vector=np.array([], dtype=np.float32),
                dsp_feature_names=[],
                beats_embedding=valid_beats_embedding,
            )

    def test_invalid_beats_wrong_dim_raises_value_error(self, builder, valid_dsp):
        """A BEATs embedding with wrong dimension raises ValueError."""
        bad_vec = np.random.default_rng(2).random(512).astype(np.float32)

        with pytest.raises(ValueError):
            BEATsEmbedding(
                vector=bad_vec,
                embedding_dim=512,
                filename="test.wav",
                machine_type="pump",
                machine_id="id_00",
                sample_rate=16000,
            )
