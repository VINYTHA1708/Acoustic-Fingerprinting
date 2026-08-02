"""Tests for ProjectionHead (src/contrastive_learning/model.py)."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent


def _load(rel: str):
    path = _ROOT / rel
    spec = importlib.util.spec_from_file_location(rel.replace("/", ".").replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ProjectionHead = _load("src/contrastive_learning/model.py").ProjectionHead


@pytest.fixture(scope="module")
def fresh_head():
    """An untrained ProjectionHead with default random weights."""
    torch.manual_seed(42)
    return ProjectionHead()


@pytest.fixture(scope="module")
def sample_input():
    torch.manual_seed(0)
    return torch.randn(4, 921)


class TestProjectionHead:
    """ProjectionHead unit tests."""

    def test_output_dimension_is_256(self, fresh_head, sample_input):
        """ProjectionHead output has exactly 256 dimensions."""
        with torch.no_grad():
            out = fresh_head(sample_input)
        assert out.shape == (4, 256)

    def test_output_is_l2_normalized(self, fresh_head, sample_input):
        """Each output row has L2 norm ≈ 1.0."""
        with torch.no_grad():
            out = fresh_head(sample_input)
        norms = torch.linalg.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_invalid_input_dim_raises_value_error(self):
        """ProjectionHead constructed with wrong input_dim raises ValueError."""
        with pytest.raises(ValueError):
            ProjectionHead(input_dim=512)

    def test_save_and_load_weights_preserve_parameters(self, fresh_head, sample_input):
        """save_weights / load_weights round-trip preserves all parameters."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head.pt"
            fresh_head.save_weights(path)

            loaded = ProjectionHead()
            loaded.load_weights(path)

        for p_orig, p_loaded in zip(fresh_head.parameters(), loaded.parameters()):
            assert torch.equal(p_orig, p_loaded)
