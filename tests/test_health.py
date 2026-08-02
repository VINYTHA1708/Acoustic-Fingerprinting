"""Tests for LearnedHealthAnalyzer and LearnedHealthSerializer
(src/learned_health_index/).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

import importlib

LearnedHealthResult = importlib.import_module("src.learned_health_index.learned_health_result").LearnedHealthResult

_VALID_STATES = {"EXCELLENT", "GOOD", "WARNING", "CRITICAL"}


class TestLearnedHealthAnalyzer:
    """LearnedHealthAnalyzer unit tests."""

    def test_returns_learned_health_result(self, health_result):
        """analyze() returns a LearnedHealthResult instance."""
        assert isinstance(health_result, LearnedHealthResult)

    def test_health_score_lower_bound(self, health_result):
        """Health score is >= 0."""
        assert health_result.health_score >= 0.0

    def test_health_score_upper_bound(self, health_result):
        """Health score is <= 100."""
        assert health_result.health_score <= 100.0

    def test_health_percentage_lower_bound(self, health_result):
        """Health percentage numeric value is >= 0."""
        pct = float(health_result.health_percentage.rstrip("%"))
        assert pct >= 0.0

    def test_health_percentage_upper_bound(self, health_result):
        """Health percentage numeric value is <= 100."""
        pct = float(health_result.health_percentage.rstrip("%"))
        assert pct <= 100.0

    def test_health_state_is_valid(self, health_result):
        """Health state is one of EXCELLENT, GOOD, WARNING, CRITICAL."""
        assert health_result.health_state in _VALID_STATES


class TestLearnedHealthSerializer:
    """LearnedHealthSerializer JSON and NPZ round-trip tests."""

    def test_json_round_trip(self, health_result):
        """JSON save → load preserves all fields."""
        import importlib
        LearnedHealthSerializer = importlib.import_module("src.learned_health_index.serializer").LearnedHealthSerializer

        serializer = LearnedHealthSerializer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "health.json"
            serializer.save_json(health_result, path)
            loaded = serializer.load_json(path)

        assert loaded.machine_type == health_result.machine_type
        assert loaded.machine_id == health_result.machine_id
        assert loaded.filename == health_result.filename
        assert math.isclose(loaded.health_score, health_result.health_score)
        assert loaded.health_percentage == health_result.health_percentage
        assert loaded.health_state == health_result.health_state

    def test_npz_round_trip(self, health_result):
        """NPZ save → load preserves all fields."""
        import importlib
        LearnedHealthSerializer = importlib.import_module("src.learned_health_index.serializer").LearnedHealthSerializer

        serializer = LearnedHealthSerializer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "health.npz"
            serializer.save_npz(health_result, path)
            loaded = serializer.load_npz(path)

        assert loaded.machine_type == health_result.machine_type
        assert loaded.machine_id == health_result.machine_id
        assert math.isclose(loaded.health_score, health_result.health_score)
        assert loaded.health_percentage == health_result.health_percentage
        assert loaded.health_state == health_result.health_state
