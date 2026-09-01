"""Tests for experiments/e1_baseline_evaluation.py.

No BEATs, no audio files, no MIMII dataset required.
Uses synthetic data and mocks throughout.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.e1_baseline_evaluation as script
from experiments.e1_baseline_definition import PROTOCOL, get_all_baselines
from src.dataset.metadata import AudioMetadata
from src.dataset.split import DatasetSplitter

MACHINE_IDS = list(PROTOCOL.machine_ids)
MACHINE_TYPE = PROTOCOL.machine_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(machine_id: str, label: str, idx: int) -> AudioMetadata:
    p = Path(f"/fake/pump/{machine_id}/{label}/{idx:08d}.wav")
    return AudioMetadata(
        machine_type=MACHINE_TYPE,
        machine_id=machine_id,
        label=label,
        filename=p.name,
        relative_path=Path(f"pump/{machine_id}/{label}/{idx:08d}.wav"),
        absolute_path=p,
    )


def _make_recordings(n_normal: int = 100, n_abnormal: int = 20) -> list[AudioMetadata]:
    recs = []
    for mid in MACHINE_IDS:
        for i in range(n_normal):
            recs.append(_make_meta(mid, "normal", i))
        for i in range(n_abnormal):
            recs.append(_make_meta(mid, "abnormal", i))
    return recs


def _split(recordings):
    return DatasetSplitter(
        train_ratio=PROTOCOL.train_ratio,
        profile_ratio=PROTOCOL.profile_ratio,
        seed=PROTOCOL.seed,
    ).split(recordings)


# ---------------------------------------------------------------------------
# compute_auroc
# ---------------------------------------------------------------------------

class TestComputeAUROC:
    def test_perfect_separation_returns_one(self):
        scores = np.array([0.1, 0.2, 0.9, 1.0], dtype=np.float64)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        assert script.compute_auroc(scores, labels) == pytest.approx(1.0, abs=1e-6)

    def test_random_scores_auroc_near_half(self):
        rng = np.random.default_rng(0)
        scores = rng.random(200)
        labels = (rng.random(200) > 0.5).astype(np.int32)
        auroc = script.compute_auroc(scores, labels)
        assert 0.3 <= auroc <= 0.7

    def test_inverted_scores_returns_zero(self):
        # Normal scores higher than abnormal → AUROC near 0
        scores = np.array([0.9, 1.0, 0.1, 0.2], dtype=np.float64)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        assert script.compute_auroc(scores, labels) == pytest.approx(0.0, abs=1e-6)

    def test_no_positives_returns_nan(self):
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([0, 0, 0])
        assert np.isnan(script.compute_auroc(scores, labels))

    def test_no_negatives_returns_nan(self):
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([1, 1, 1])
        assert np.isnan(script.compute_auroc(scores, labels))

    def test_auroc_in_unit_interval(self):
        rng = np.random.default_rng(42)
        scores = rng.random(100)
        labels = (rng.random(100) > 0.5).astype(np.int32)
        auroc = script.compute_auroc(scores, labels)
        assert 0.0 <= auroc <= 1.0


# ---------------------------------------------------------------------------
# compute_separation_ratio
# ---------------------------------------------------------------------------

class TestComputeSeparationRatio:
    def test_higher_abnormal_mean_gives_ratio_above_one(self):
        normal = np.array([1.0, 1.0, 1.0])
        abnormal = np.array([3.0, 3.0, 3.0])
        assert script.compute_separation_ratio(normal, abnormal) == pytest.approx(3.0)

    def test_equal_means_gives_ratio_one(self):
        normal = np.array([2.0, 2.0])
        abnormal = np.array([2.0, 2.0])
        assert script.compute_separation_ratio(normal, abnormal) == pytest.approx(1.0)

    def test_zero_normal_mean_returns_nan(self):
        normal = np.array([0.0, 0.0])
        abnormal = np.array([1.0, 2.0])
        assert np.isnan(script.compute_separation_ratio(normal, abnormal))

    def test_empty_arrays_returns_nan(self):
        assert np.isnan(script.compute_separation_ratio(np.array([]), np.array([1.0])))

    def test_ratio_is_non_negative(self):
        normal = np.array([1.0, 2.0, 3.0])
        abnormal = np.array([0.5, 1.0])
        ratio = script.compute_separation_ratio(normal, abnormal)
        assert ratio >= 0.0


# ---------------------------------------------------------------------------
# validate_results
# ---------------------------------------------------------------------------

class TestValidateResults:
    def _valid_row(self, baseline_id="B1_mfcc_distance", machine_id="id_00"):
        return {
            "baseline_id": baseline_id,
            "baseline_name": "Raw MFCC Distance",
            "machine_id": machine_id,
            "n_normal": 10,
            "n_abnormal": 5,
            "auroc": 0.75,
            "separation_ratio": 1.5,
        }

    def test_valid_rows_do_not_raise(self):
        rows = [self._valid_row(mid) for mid in MACHINE_IDS]
        script.validate_results(rows)  # must not raise

    def test_auroc_above_one_raises(self):
        row = self._valid_row()
        row["auroc"] = 1.5
        with pytest.raises(ValueError, match="AUROC"):
            script.validate_results([row])

    def test_auroc_below_zero_raises(self):
        row = self._valid_row()
        row["auroc"] = -0.1
        with pytest.raises(ValueError, match="AUROC"):
            script.validate_results([row])

    def test_negative_separation_ratio_raises(self):
        row = self._valid_row()
        row["separation_ratio"] = -1.0
        with pytest.raises(ValueError, match="separation_ratio"):
            script.validate_results([row])

    def test_nan_auroc_is_accepted(self):
        row = self._valid_row()
        row["auroc"] = float("nan")
        script.validate_results([row])  # NaN is allowed (no positives/negatives)

    def test_nan_separation_is_accepted(self):
        row = self._valid_row()
        row["separation_ratio"] = float("nan")
        script.validate_results([row])


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

class TestCSVOutput:
    def _make_rows(self, n: int = 4) -> list[dict]:
        return [
            {
                "baseline_id": "B1_mfcc_distance",
                "baseline_name": "Raw MFCC Distance",
                "machine_id": f"id_{i:02d}",
                "n_normal": 10,
                "n_abnormal": 5,
                "auroc": 0.80,
                "separation_ratio": 1.2,
            }
            for i in range(n)
        ]

    def test_csv_has_all_required_columns(self, tmp_path):
        rows = self._make_rows()
        out = tmp_path / "baseline_results.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=script.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with out.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            written = list(reader)
        for col in script.CSV_COLUMNS:
            assert col in written[0]

    def test_csv_row_count_matches_input(self, tmp_path):
        rows = self._make_rows(8)
        out = tmp_path / "baseline_results.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=script.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with out.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        assert len(written) == 8

    def test_csv_values_round_trip(self, tmp_path):
        rows = self._make_rows(1)
        rows[0]["auroc"] = 0.9123
        out = tmp_path / "baseline_results.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=script.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with out.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        assert float(written[0]["auroc"]) == pytest.approx(0.9123, abs=1e-4)


# ---------------------------------------------------------------------------
# Split isolation — profile_normal not in test set
# ---------------------------------------------------------------------------

class TestSplitIsolation:
    def test_profile_normal_not_in_test_set(self):
        split = _split(_make_recordings())
        profile_paths = {r.absolute_path for r in split.profile_normal}
        test_paths = (
            {r.absolute_path for r in split.test_normal}
            | {r.absolute_path for r in split.test_abnormal}
        )
        assert not profile_paths & test_paths

    def test_train_normal_not_in_test_set(self):
        split = _split(_make_recordings())
        train_paths = {r.absolute_path for r in split.train_normal}
        test_paths = (
            {r.absolute_path for r in split.test_normal}
            | {r.absolute_path for r in split.test_abnormal}
        )
        assert not train_paths & test_paths


# ---------------------------------------------------------------------------
# _evaluate_machine_id (mocked feature extraction)
# ---------------------------------------------------------------------------

class TestEvaluateMachineId:
    """Tests _evaluate_machine_id with mocked feature vectors."""

    def _make_records(self, machine_id: str, n_profile: int, n_normal: int, n_abnormal: int):
        profile = [_make_meta(machine_id, "normal", i) for i in range(n_profile)]
        test = (
            [(_make_meta(machine_id, "normal", i + n_profile), "normal") for i in range(n_normal)]
            + [(_make_meta(machine_id, "abnormal", i), "abnormal") for i in range(n_abnormal)]
        )
        return profile, test

    def _run_with_mock_vectors(self, baseline_id: str, profile_vec, test_normal_vec, test_abnormal_vec):
        """Patch all three feature extraction helpers to return controlled vectors."""
        profile_records, test_records = self._make_records("id_00", 5, 10, 10)

        def mock_dsp(rec, pipeline, extractor, vec_builder):
            return profile_vec if rec.label == "normal" else test_abnormal_vec

        def mock_stat(rec, pipeline, extractor):
            return profile_vec[:3] if rec.label == "normal" else test_abnormal_vec[:3]

        def mock_proj(rec, cache):
            return profile_vec[:256] if rec.label == "normal" else test_abnormal_vec[:256]

        with (
            patch.object(script, "_dsp_vector", side_effect=mock_dsp),
            patch.object(script, "_stat_vector", side_effect=mock_stat),
            patch.object(script, "_random_projection_vector", side_effect=mock_proj),
        ):
            return script._evaluate_machine_id(
                baseline_id=baseline_id,
                machine_id="id_00",
                profile_records=profile_records,
                test_records=test_records,
                pipeline=None,
                extractor=None,
                vec_builder=None,
                cache=None,
            )

    def test_b1_result_has_all_keys(self):
        profile_vec = np.zeros(153, dtype=np.float32)
        abnormal_vec = np.ones(153, dtype=np.float32) * 5.0
        result = self._run_with_mock_vectors("B1_mfcc_distance", profile_vec, profile_vec, abnormal_vec)
        for col in script.CSV_COLUMNS:
            assert col in result

    def test_b1_auroc_in_unit_interval(self):
        profile_vec = np.zeros(153, dtype=np.float32)
        abnormal_vec = np.ones(153, dtype=np.float32) * 5.0
        result = self._run_with_mock_vectors("B1_mfcc_distance", profile_vec, profile_vec, abnormal_vec)
        assert 0.0 <= result["auroc"] <= 1.0

    def test_b1_perfect_separation_auroc_one(self):
        # Normal vectors identical to profile → distance 0
        # Abnormal vectors far from profile → high distance
        profile_vec = np.zeros(153, dtype=np.float32)
        abnormal_vec = np.ones(153, dtype=np.float32) * 100.0
        result = self._run_with_mock_vectors("B1_mfcc_distance", profile_vec, profile_vec, abnormal_vec)
        assert result["auroc"] == pytest.approx(1.0, abs=1e-6)

    def test_b2_result_has_all_keys(self):
        profile_vec = np.zeros(256, dtype=np.float32)
        abnormal_vec = np.ones(256, dtype=np.float32) * 5.0
        result = self._run_with_mock_vectors("B2_stat_distance", profile_vec, profile_vec, abnormal_vec)
        for col in script.CSV_COLUMNS:
            assert col in result

    def test_b3_result_has_all_keys(self):
        profile_vec = np.zeros(256, dtype=np.float32)
        abnormal_vec = np.ones(256, dtype=np.float32) * 5.0
        result = self._run_with_mock_vectors("B3_random_projection", profile_vec, profile_vec, abnormal_vec)
        for col in script.CSV_COLUMNS:
            assert col in result

    def test_n_normal_and_n_abnormal_counts_correct(self):
        profile_vec = np.zeros(153, dtype=np.float32)
        abnormal_vec = np.ones(153, dtype=np.float32)
        result = self._run_with_mock_vectors("B1_mfcc_distance", profile_vec, profile_vec, abnormal_vec)
        assert result["n_normal"] == 10
        assert result["n_abnormal"] == 10

    def test_separation_ratio_above_one_when_abnormal_farther(self):
        # Profile mean = zeros; normal test = small offset (dist ~1); abnormal = large offset (dist ~100)
        # Use a dedicated mock that distinguishes by filename index
        profile_records, test_records = self._make_records("id_00", 5, 10, 10)
        normal_vec = np.ones(153, dtype=np.float32) * 0.01   # small distance from profile
        abnormal_vec = np.ones(153, dtype=np.float32) * 100.0  # large distance from profile
        profile_vec = np.zeros(153, dtype=np.float32)

        def mock_dsp(rec, pipeline, extractor, vec_builder):
            if rec in profile_records:
                return profile_vec
            return normal_vec if rec.label == "normal" else abnormal_vec

        with patch.object(script, "_dsp_vector", side_effect=mock_dsp):
            result = script._evaluate_machine_id(
                baseline_id="B1_mfcc_distance",
                machine_id="id_00",
                profile_records=profile_records,
                test_records=test_records,
                pipeline=None,
                extractor=None,
                vec_builder=None,
                cache=None,
            )
        assert result["separation_ratio"] > 1.0


# ---------------------------------------------------------------------------
# validate_inputs
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_missing_dataset_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "DATASET_ROOT", tmp_path / "missing_mimii")
        with pytest.raises(FileNotFoundError, match="MIMII"):
            script.validate_inputs()

    def test_missing_beats_checkpoint_raises(self, tmp_path, monkeypatch):
        # Dataset exists but BEATs checkpoint does not
        dataset = tmp_path / "MIMII"
        dataset.mkdir()
        monkeypatch.setattr(script, "DATASET_ROOT", dataset)
        monkeypatch.setattr(script, "BEATS_CHECKPOINT", tmp_path / "missing.pt")
        with pytest.raises(FileNotFoundError, match="BEATs"):
            script.validate_inputs()

    def test_both_present_does_not_raise(self, tmp_path, monkeypatch):
        dataset = tmp_path / "MIMII"
        dataset.mkdir()
        ckpt = tmp_path / "BEATs.pt"
        ckpt.write_bytes(b"fake")
        monkeypatch.setattr(script, "DATASET_ROOT", dataset)
        monkeypatch.setattr(script, "BEATS_CHECKPOINT", ckpt)
        script.validate_inputs()  # must not raise


# ---------------------------------------------------------------------------
# Protocol alignment
# ---------------------------------------------------------------------------

class TestProtocolAlignment:
    def test_csv_columns_include_auroc_and_separation(self):
        assert "auroc" in script.CSV_COLUMNS
        assert "separation_ratio" in script.CSV_COLUMNS

    def test_csv_columns_include_sample_counts(self):
        assert "n_normal" in script.CSV_COLUMNS
        assert "n_abnormal" in script.CSV_COLUMNS

    def test_stat_keys_are_subset_of_extractor_output(self):
        """B2 stat keys must exist in FeatureExtractor output."""
        from src.feature_extraction.extractor import FeatureExtractor
        import numpy as np

        extractor = FeatureExtractor(sample_rate=16_000)
        dummy_waveform = np.zeros(16_000, dtype=np.float32)
        features = extractor.extract(dummy_waveform)
        for key in script._STAT_KEYS:
            assert key in features, f"Stat key '{key}' missing from FeatureExtractor output"

    def test_three_baselines_evaluated(self):
        assert len(get_all_baselines()) == 3
