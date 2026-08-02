"""Tests for NTXentLoss (src/contrastive_learning/loss.py)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent


def _load(rel: str):
    path = _ROOT / rel
    spec = importlib.util.spec_from_file_location(rel.replace("/", ".").replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NTXentLoss = _load("src/contrastive_learning/loss.py").NTXentLoss


@pytest.fixture(scope="module")
def criterion():
    return NTXentLoss(temperature=0.1)


def _l2_batch(n: int, dim: int = 256, seed: int = 0) -> torch.Tensor:
    """Return an (n, dim) batch of L2-normalised random embeddings."""
    torch.manual_seed(seed)
    return F.normalize(torch.randn(n, dim), p=2, dim=1)


class TestNTXentLoss:
    """NTXentLoss unit tests."""

    def test_returns_scalar_tensor(self, criterion):
        """NTXentLoss forward returns a 0-dim scalar tensor."""
        a, b = _l2_batch(4), _l2_batch(4, seed=1)
        loss = criterion(a, b)
        assert loss.ndim == 0

    def test_loss_is_finite(self, criterion):
        """NTXentLoss value is finite for a valid batch."""
        a, b = _l2_batch(8), _l2_batch(8, seed=2)
        loss = criterion(a, b)
        assert math.isfinite(loss.item())

    def test_batch_size_less_than_2_raises_value_error(self, criterion):
        """Batch size of 1 raises ValueError."""
        a = _l2_batch(1)
        with pytest.raises(ValueError, match="Batch size"):
            criterion(a, a)

    def test_mismatched_embedding_dimensions_raise_value_error(self, criterion):
        """Embeddings with different shapes raise ValueError."""
        a = _l2_batch(4)
        b = _l2_batch(4)[:, :128]  # wrong dim
        with pytest.raises(ValueError):
            criterion(a, b)

    def test_invalid_temperature_raises_value_error(self):
        """Temperature <= 0 raises ValueError at construction."""
        with pytest.raises(ValueError, match="temperature"):
            NTXentLoss(temperature=0.0)

        with pytest.raises(ValueError, match="temperature"):
            NTXentLoss(temperature=-0.5)
