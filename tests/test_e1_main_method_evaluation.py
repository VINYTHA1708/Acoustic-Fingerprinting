"""Tests for experiments/e1_main_method_evaluation.py.

No BEATs, no audio files, no MIMII dataset, no trained checkpoint required.
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

import experiments.e1_main_method_evaluation as script
from experiments.e1_baseline_definition import PROTOCOL
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


def _make_inference_mock():
    """Return a mock ContrastiveInference that returns deterministic embeddings."""
    mock = MagicMock()
    mock.generate_fingerprint.return_value = np.zeros(256, dtype=np.float32)
    return mock


def _make_cache_mock():
    """Return a mock FusionCache."""
    mock = MagicMock()
    mock.load_or_create.return_value = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# CSV schema alignment with baseline_results.csv
# ---------------------------------------------------------------------------

class TestCSVSchemaAlignment:
    """Main method CSV must use the same columns as baseline_results.csv."""

    def test_csv_columns_match_baseline_schema(self):
        from experiments.e1_baseline_evaluation import CSV_COLUMNS as baseline_cols
        assert script.CSV_COLUMNS == baseline_cols

    def test_method_id_field_present_in_rows(self):
        assert "baseline_id" in script.CSV_COLUMNS

    def test_method_name_field_present_in_rows(self):
        assert "baseline_name" in script.CSV_COLUMNS

    def test_all_required_metric_columns_present(self):
        for col in ("n_normal", "n_abnormal", "auroc", "separation_ratio"):
            assert col in script.CSV_COLUMNS


# ---------------------------------------------------------------------------
# evaluate_machine_id
# ---------------------------------------------------------------------------

class TestEvaluateMachineId:

    def _make_records(self, machine_id: str, n_profile: int, n_normal: int, n_abnormal: int):
        profile = [_make_meta(machine_id, "normal", i) for i in range(n_profile)]
        test = (
            [(_make_meta(machine_id, "normal", i + n_profile), "normal") for i in range(n_normal)]
            + [(_make_meta(machine_id, "abnormal", i), "abnormal") for i in range(n_abnormal)]
        )
        return profile, test

    def _run(self, profile_vec, normal_vec, abnormal_vec, machine_id="id_00",
             n_profile=5, n_normal=10, n_abnormal=10):
        profile_records, test_records = self._make_records(machine_id, n_profile, n_normal, n_abnormal)
        profile_set = set(id(r) for r in profile_records)

        def mock_embedding(rec, inference, cache):
            if id(rec) in profile_set:
                return profile_vec.copy()
            return normal_vec.copy() if rec.label == "normal" else abnormal_vec.copy()

        with patch.object(script, "_contrastive_embedding", side_effect=mock_embedding):
            return script.evaluate_machine_id(
                machine_id=machine_id,
                profile_records=profile_records,
                test_records=test_records,
                inference=None,
                cache=None,
            )

    def test_result_has_all_csv_columns(self):
        v = np.zeros(256, dtype=np.float32)
        result = self._run(v, v, v + 1.0)
        for col in script.CSV_COLUMNS:
            assert col in result

    def test_method_id_is_contrastive_main(self):
        v = np.zeros(256, dtype=np.float32)
        result = self._run(v, v, v + 1.0)
        assert result["baseline_id"] == script.METHOD_ID

    def test_method_name_is_set(self):
        v = np.zeros(256, dtype=np.float32)
        result = self._run(v, v, v + 1.0)
        assert result["baseline_name"] == script.METHOD_NAME

    def test_auroc_in_unit_interval(self):
        profile_vec = np.zeros(256, dtype=np.float32)
        abnormal_vec = np.ones(256, dtype=np.float32) * 5.0
        result = self._run(profile_vec, profile_vec, abnormal_vec)
        assert 0.0 <= result["auroc"] <= 1.0

    def test_perfect_separation_gives_auroc_one(self):
        # Normal embeddings identical to profile → distance 0
        # Abnormal embeddings far from profile → high distance
        profile_vec = np.zeros(256, dtype=np.float32)
        normal_vec = np.zeros(256, dtype=np.float32)
        abnormal_vec = np.ones(256, dtype=np.float32) * 100.0
        result = self._run(profile_vec, normal_vec, abnormal_vec)
        assert result["auroc"] == pytest.approx(1.0, abs=1e-6)

    def test_n_normal_count_correct(self):
        v = np.zeros(256, dtype=np.float32)
        result = self._run(v, v, v + 1.0, n_normal=12, n_abnormal=8)
        assert result["n_normal"] == 12

    def test_n_abnormal_count_correct(self):
        v = np.zeros(256, dtype=np.float32)
        result = self._run(v, v, v + 1.0, n_normal=12, n_abnormal=8)
        assert result["n_abnormal"] == 8

    def test_separation_ratio_above_one_when_abnormal_farther(self):
        profile_vec = np.zeros(256, dtype=np.float32)
        normal_vec = np.ones(256, dtype=np.float32) * 0.01
        abnormal_vec = np.ones(256, dtype=np.float32) * 100.0
        result = self._run(profile_vec, normal_vec, abnormal_vec)
        assert result["separation_ratio"] > 1.0

    def test_machine_id_preserved_in_result(self):
        v = np.zeros(256, dtype=np.float32)
        result = self._run(v, v, v + 1.0, machine_id="id_02")
        assert result["machine_id"] == "id_02"

    def test_embedding_dim_256_used(self):
        """Profile mean must be built from 256-dim vectors."""
        profile_records, test_records = self._make_records("id_00", 3, 5, 5)
        captured = []

        def mock_embedding(rec, inference, cache):
            v = np.zeros(256, dtype=np.float32)
            captured.append(v.shape[0])
            return v

        with patch.object(script, "_contrastive_embedding", side_effect=mock_embedding):
            script.evaluate_machine_id(
                machine_id="id_00",
                profile_records=profile_records,
                test_records=test_records,
                inference=None,
                cache=None,
            )
        assert all(d == 256 for d in captured)


# ---------------------------------------------------------------------------
# validate_inputs
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_missing_dataset_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "DATASET_ROOT", tmp_path / "missing_mimii")
        with pytest.raises(FileNotFoundError, match="MIMII"):
            script.validate_inputs()

    def test_missing_beats_checkpoint_raises(self, tmp_path, monkeypatch):
        dataset = tmp_path / "MIMII"
        dataset.mkdir()
        monkeypatch.setattr(script, "DATASET_ROOT", dataset)
        monkeypatch.setattr(script, "BEATS_CHECKPOINT", tmp_path / "missing_beats.pt")
        monkeypatch.setattr(script, "CONTRASTIVE_CHECKPOINT", tmp_path / "missing_ckpt.pt")
        with pytest.raises(FileNotFoundError, match="BEATs"):
            script.validate_inputs()

    def test_missing_contrastive_checkpoint_raises(self, tmp_path, monkeypatch):
        dataset = tmp_path / "MIMII"
        dataset.mkdir()
        beats = tmp_path / "BEATs.pt"
        beats.write_bytes(b"fake")
        monkeypatch.setattr(script, "DATASET_ROOT", dataset)
        monkeypatch.setattr(script, "BEATS_CHECKPOINT", beats)
        monkeypatch.setattr(script, "CONTRASTIVE_CHECKPOINT", tmp_path / "missing_ckpt.pt")
        with pytest.raises(FileNotFoundError, match="[Cc]ontrastive"):
            script.validate_inputs()

    def test_all_present_does_not_raise(self, tmp_path, monkeypatch):
        dataset = tmp_path / "MIMII"
        dataset.mkdir()
        beats = tmp_path / "BEATs.pt"
        beats.write_bytes(b"fake")
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"fake")
        monkeypatch.setattr(script, "DATASET_ROOT", dataset)
        monkeypatch.setattr(script, "BEATS_CHECKPOINT", beats)
        monkeypatch.setattr(script, "CONTRASTIVE_CHECKPOINT", ckpt)
        script.validate_inputs()  # must not raise


# ---------------------------------------------------------------------------
# validate_results
# ---------------------------------------------------------------------------

class TestValidateResults:
    def _valid_row(self, machine_id="id_00"):
        return {
            "baseline_id": script.METHOD_ID,
            "baseline_name": script.METHOD_NAME,
            "machine_id": machine_id,
            "n_normal": 152,
            "n_abnormal": 143,
            "auroc": 0.85,
            "separation_ratio": 2.1,
        }

    def test_valid_rows_do_not_raise(self):
        rows = [self._valid_row(mid) for mid in MACHINE_IDS]
        script.validate_results(rows)

    def test_auroc_above_one_raises(self):
        row = self._valid_row()
        row["auroc"] = 1.1
        with pytest.raises(ValueError, match="AUROC"):
            script.validate_results([row])

    def test_auroc_below_zero_raises(self):
        row = self._valid_row()
        row["auroc"] = -0.01
        with pytest.raises(ValueError, match="AUROC"):
            script.validate_results([row])

    def test_negative_separation_raises(self):
        row = self._valid_row()
        row["separation_ratio"] = -0.5
        with pytest.raises(ValueError, match="separation_ratio"):
            script.validate_results([row])

    def test_nan_auroc_accepted(self):
        row = self._valid_row()
        row["auroc"] = float("nan")
        script.validate_results([row])

    def test_nan_separation_accepted(self):
        row = self._valid_row()
        row["separation_ratio"] = float("nan")
        script.validate_results([row])


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

class TestCSVOutput:
    def _make_rows(self) -> list[dict]:
        return [
            {
                "baseline_id": script.METHOD_ID,
                "baseline_name": script.METHOD_NAME,
                "machine_id": mid,
                "n_normal": 152,
                "n_abnormal": 143,
                "auroc": 0.85,
                "separation_ratio": 2.1,
            }
            for mid in MACHINE_IDS
        ]

    def test_csv_has_all_columns(self, tmp_path):
        rows = self._make_rows()
        out = tmp_path / "main_method_results.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=script.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with out.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        for col in script.CSV_COLUMNS:
            assert col in written[0]

    def test_csv_row_count_is_four(self, tmp_path):
        rows = self._make_rows()
        out = tmp_path / "main_method_results.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=script.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with out.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        assert len(written) == len(MACHINE_IDS)

    def test_csv_auroc_round_trips(self, tmp_path):
        rows = self._make_rows()
        rows[0]["auroc"] = 0.7654
        out = tmp_path / "main_method_results.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=script.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with out.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        assert float(written[0]["auroc"]) == pytest.approx(0.7654, abs=1e-4)


# ---------------------------------------------------------------------------
# Split isolation — same guarantees as baseline evaluation
# ---------------------------------------------------------------------------

class TestSplitIsolation:
    def test_profile_not_in_test(self):
        split = _split(_make_recordings())
        profile_paths = {r.absolute_path for r in split.profile_normal}
        test_paths = (
            {r.absolute_path for r in split.test_normal}
            | {r.absolute_path for r in split.test_abnormal}
        )
        assert not profile_paths & test_paths

    def test_train_not_in_test(self):
        split = _split(_make_recordings())
        train_paths = {r.absolute_path for r in split.train_normal}
        test_paths = (
            {r.absolute_path for r in split.test_normal}
            | {r.absolute_path for r in split.test_abnormal}
        )
        assert not train_paths & test_paths


# ---------------------------------------------------------------------------
# Protocol alignment — same split parameters as baseline evaluation
# ---------------------------------------------------------------------------

class TestProtocolAlignment:
    def test_same_train_ratio_as_baseline(self):
        from experiments.e1_baseline_evaluation import PROTOCOL as baseline_proto
        assert PROTOCOL.train_ratio == baseline_proto.train_ratio

    def test_same_profile_ratio_as_baseline(self):
        from experiments.e1_baseline_evaluation import PROTOCOL as baseline_proto
        assert PROTOCOL.profile_ratio == baseline_proto.profile_ratio

    def test_same_seed_as_baseline(self):
        from experiments.e1_baseline_evaluation import PROTOCOL as baseline_proto
        assert PROTOCOL.seed == baseline_proto.seed

    def test_same_machine_ids_as_baseline(self):
        from experiments.e1_baseline_evaluation import PROTOCOL as baseline_proto
        assert set(PROTOCOL.machine_ids) == set(baseline_proto.machine_ids)

    def test_four_machine_ids_evaluated(self):
        assert len(PROTOCOL.machine_ids) == 4
