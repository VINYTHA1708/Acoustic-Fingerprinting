"""Tests for LearnedDriftAnalyzer and LearnedDriftSerializer
(src/learned_drift/).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import CONTRASTIVE_CHECKPOINT, MACHINE_ID, MACHINE_TYPE
import importlib

LearnedDriftResult = importlib.import_module("src.learned_drift.learned_drift_result").LearnedDriftResult


class TestLearnedDriftAnalyzer:
    """LearnedDriftAnalyzer unit tests."""

    def test_returns_learned_drift_result(self, drift_result):
        """analyze() returns a LearnedDriftResult instance."""
        assert isinstance(drift_result, LearnedDriftResult)

    def test_raw_euclidean_is_finite(self, drift_result):
        """Raw Euclidean distance is a finite float."""
        assert math.isfinite(drift_result.euclidean_distance)

    def test_raw_manhattan_is_finite(self, drift_result):
        """Raw Manhattan distance is a finite float."""
        assert math.isfinite(drift_result.manhattan_distance)

    def test_raw_cosine_is_finite(self, drift_result):
        """Raw cosine similarity is a finite float."""
        assert math.isfinite(drift_result.cosine_similarity)

    def test_normalized_euclidean_is_finite(self, drift_result):
        """Normalized Euclidean distance is a finite float."""
        assert math.isfinite(drift_result.norm_euclidean_distance)

    def test_normalized_manhattan_is_finite(self, drift_result):
        """Normalized Manhattan distance is a finite float."""
        assert math.isfinite(drift_result.norm_manhattan_distance)

    def test_normalized_cosine_is_finite(self, drift_result):
        """Normalized cosine similarity is a finite float."""
        assert math.isfinite(drift_result.norm_cosine_similarity)

    def test_machine_type_mismatch_raises_value_error(self, first_normal_record, learned_profile):
        """Analyzing a record whose machine_type differs from the profile raises ValueError."""
        import importlib
        AudioMetadata = importlib.import_module("src.dataset.metadata").AudioMetadata
        LearnedDriftAnalyzer = importlib.import_module("src.learned_drift.analyzer").LearnedDriftAnalyzer

        wrong_record = AudioMetadata(
            machine_type="fan",           # mismatch
            machine_id=first_normal_record.machine_id,
            label=first_normal_record.label,
            filename=first_normal_record.filename,
            relative_path=first_normal_record.relative_path,
            absolute_path=first_normal_record.absolute_path,
        )
        analyzer = LearnedDriftAnalyzer(checkpoint_path=CONTRASTIVE_CHECKPOINT)
        with pytest.raises(ValueError, match="machine_type"):
            analyzer.analyze(wrong_record, learned_profile)

    def test_machine_id_mismatch_raises_value_error(self, first_normal_record, learned_profile):
        """Analyzing a record whose machine_id differs from the profile raises ValueError."""
        import importlib
        AudioMetadata = importlib.import_module("src.dataset.metadata").AudioMetadata
        LearnedDriftAnalyzer = importlib.import_module("src.learned_drift.analyzer").LearnedDriftAnalyzer

        wrong_record = AudioMetadata(
            machine_type=first_normal_record.machine_type,
            machine_id="id_99",           # mismatch
            label=first_normal_record.label,
            filename=first_normal_record.filename,
            relative_path=first_normal_record.relative_path,
            absolute_path=first_normal_record.absolute_path,
        )
        analyzer = LearnedDriftAnalyzer(checkpoint_path=CONTRASTIVE_CHECKPOINT)
        with pytest.raises(ValueError, match="machine_id"):
            analyzer.analyze(wrong_record, learned_profile)


class TestLearnedDriftSerializer:
    """LearnedDriftSerializer JSON and NPZ round-trip tests."""

    def test_json_round_trip(self, drift_result):
        """JSON save → load preserves all scalar fields and the normalized vector."""
        import importlib
        LearnedDriftSerializer = importlib.import_module("src.learned_drift.serializer").LearnedDriftSerializer

        serializer = LearnedDriftSerializer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drift.json"
            serializer.save_json(drift_result, path)
            loaded = serializer.load_json(path)

        assert loaded.machine_type == drift_result.machine_type
        assert loaded.machine_id == drift_result.machine_id
        assert loaded.filename == drift_result.filename
        assert math.isclose(loaded.euclidean_distance, drift_result.euclidean_distance)
        assert math.isclose(loaded.norm_euclidean_distance, drift_result.norm_euclidean_distance)
        np.testing.assert_array_almost_equal(loaded.normalized_vector, drift_result.normalized_vector)

    def test_npz_round_trip(self, drift_result):
        """NPZ save → load preserves all scalar fields and the normalized vector."""
        import importlib
        LearnedDriftSerializer = importlib.import_module("src.learned_drift.serializer").LearnedDriftSerializer

        serializer = LearnedDriftSerializer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drift.npz"
            serializer.save_npz(drift_result, path)
            loaded = serializer.load_npz(path)

        assert loaded.machine_type == drift_result.machine_type
        assert loaded.machine_id == drift_result.machine_id
        assert math.isclose(loaded.euclidean_distance, drift_result.euclidean_distance)
        assert math.isclose(loaded.norm_euclidean_distance, drift_result.norm_euclidean_distance)
        np.testing.assert_array_almost_equal(loaded.normalized_vector, drift_result.normalized_vector)
