"""Lightweight tests for experiments/e1_evaluate.py.

No BEATs, no audio files, no MIMII dataset, no trained checkpoint required.
Uses synthetic AudioMetadata objects and mocks throughout.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.e1_evaluate as script
from src.dataset.metadata import AudioMetadata
from src.dataset.split import DatasetSplitter

MACHINE_TYPE = "pump"
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
TRAIN_RATIO = 0.70
PROFILE_RATIO = 0.15
SEED = 42


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
    return DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED).split(recordings)


def _fake_profile(machine_id: str):
    p = MagicMock()
    p.machine_type = MACHINE_TYPE
    p.machine_id = machine_id
    p.embedding_dimension = 256
    p.mean_vector = np.zeros(256, dtype=np.float32)
    p.std_vector = np.ones(256, dtype=np.float32)
    p.embeddings = np.zeros((10, 256), dtype=np.float32)
    return p


def _fake_health_result(record: AudioMetadata):
    r = MagicMock()
    r.machine_type = record.machine_type
    r.machine_id = record.machine_id
    r.filename = record.filename
    r.health_score = 85.0
    r.health_percentage = "85.0%"
    r.health_state = "GOOD"
    r.normalized_euclidean = 1.2
    r.normalized_manhattan = 10.5
    r.normalized_cosine = 0.9
    return r


# ---------------------------------------------------------------------------
# A: Evaluation dataset composition
# ---------------------------------------------------------------------------

class TestEvaluationDatasetComposition:
    def test_only_test_splits_in_evaluation(self):
        split = _split(_make_recordings())
        evaluation_records = (
            [(r, "normal") for r in split.test_normal]
            + [(r, "abnormal") for r in split.test_abnormal]
        )
        eval_paths = {r.absolute_path for r, _ in evaluation_records}
        train_paths = {r.absolute_path for r in split.train_normal}
        profile_paths = {r.absolute_path for r in split.profile_normal}

        assert not eval_paths & train_paths, "train_normal leaked into evaluation"
        assert not eval_paths & profile_paths, "profile_normal leaked into evaluation"

    def test_train_normal_excluded(self):
        split = _split(_make_recordings())
        evaluation_records = (
            [(r, "normal") for r in split.test_normal]
            + [(r, "abnormal") for r in split.test_abnormal]
        )
        eval_paths = {r.absolute_path for r, _ in evaluation_records}
        for r in split.train_normal:
            assert r.absolute_path not in eval_paths

    def test_profile_normal_excluded(self):
        split = _split(_make_recordings())
        evaluation_records = (
            [(r, "normal") for r in split.test_normal]
            + [(r, "abnormal") for r in split.test_abnormal]
        )
        eval_paths = {r.absolute_path for r, _ in evaluation_records}
        for r in split.profile_normal:
            assert r.absolute_path not in eval_paths


# ---------------------------------------------------------------------------
# B: Correct labels
# ---------------------------------------------------------------------------

class TestCorrectLabels:
    def test_test_normal_gets_normal_label(self):
        split = _split(_make_recordings())
        for record, label in [(r, "normal") for r in split.test_normal]:
            assert label == "normal"

    def test_test_abnormal_gets_abnormal_label(self):
        split = _split(_make_recordings())
        for record, label in [(r, "abnormal") for r in split.test_abnormal]:
            assert label == "abnormal"

    def test_no_label_mixing(self):
        split = _split(_make_recordings())
        normal_paths = {r.absolute_path for r in split.test_normal}
        abnormal_paths = {r.absolute_path for r in split.test_abnormal}
        assert not normal_paths & abnormal_paths


# ---------------------------------------------------------------------------
# C: Correct profile selection
# ---------------------------------------------------------------------------

class TestCorrectProfileSelection:
    def test_each_record_gets_matching_profile(self):
        split = _split(_make_recordings())
        profiles = {(MACHINE_TYPE, mid): _fake_profile(mid) for mid in MACHINE_IDS}

        evaluation_records = (
            [(r, "normal") for r in split.test_normal]
            + [(r, "abnormal") for r in split.test_abnormal]
        )
        for record, _ in evaluation_records:
            key = (record.machine_type, record.machine_id)
            profile = profiles[key]
            assert profile.machine_type == record.machine_type
            assert profile.machine_id == record.machine_id

    def test_wrong_machine_id_profile_not_used(self):
        split = _split(_make_recordings(n_normal=20))
        profiles = {(MACHINE_TYPE, mid): _fake_profile(mid) for mid in MACHINE_IDS}

        for record in split.test_normal:
            selected = profiles[(record.machine_type, record.machine_id)]
            assert selected.machine_id == record.machine_id


# ---------------------------------------------------------------------------
# D: Missing profile raises clearly
# ---------------------------------------------------------------------------

class TestMissingProfile:
    def test_missing_profile_raises_value_error(self):
        split = _split(_make_recordings(n_normal=20))
        # Only load profiles for id_00, id_02, id_04 — omit id_06
        profiles = {(MACHINE_TYPE, mid): _fake_profile(mid) for mid in MACHINE_IDS[:3]}

        evaluation_records = (
            [(r, "normal") for r in split.test_normal]
            + [(r, "abnormal") for r in split.test_abnormal]
        )
        with pytest.raises((KeyError, ValueError)):
            script._validate_profiles_cover_all_records(profiles, evaluation_records)

    def test_all_profiles_present_does_not_raise(self):
        split = _split(_make_recordings(n_normal=20))
        profiles = {(MACHINE_TYPE, mid): _fake_profile(mid) for mid in MACHINE_IDS}
        evaluation_records = (
            [(r, "normal") for r in split.test_normal]
            + [(r, "abnormal") for r in split.test_abnormal]
        )
        # Should not raise
        script._validate_profiles_cover_all_records(profiles, evaluation_records)


# ---------------------------------------------------------------------------
# E: Split count validation
# ---------------------------------------------------------------------------

class TestSplitCountValidation:
    def test_wrong_counts_raise_value_error(self):
        # Patch expected counts to something impossible
        original = script._EXPECTED_TEST_NORMAL.copy()
        try:
            script._EXPECTED_TEST_NORMAL["id_00"] = 9999
            split = _split(_make_recordings())
            with pytest.raises(ValueError, match="split count mismatch"):
                script._validate_split_counts(split)
        finally:
            script._EXPECTED_TEST_NORMAL.update(original)

    def test_correct_counts_do_not_raise(self):
        # Build a split with the exact E1 normal counts
        E1_NORMAL_COUNTS = {"id_00": 1006, "id_02": 1005, "id_04": 702, "id_06": 1036}
        E1_ABNORMAL_COUNTS = {"id_00": 143, "id_02": 111, "id_04": 100, "id_06": 102}
        recs = []
        for mid in MACHINE_IDS:
            for i in range(E1_NORMAL_COUNTS[mid]):
                recs.append(_make_meta(mid, "normal", i))
            for i in range(E1_ABNORMAL_COUNTS[mid]):
                recs.append(_make_meta(mid, "abnormal", i))
        split = _split(recs)
        # Should not raise
        script._validate_split_counts(split)


# ---------------------------------------------------------------------------
# F: Partition isolation validation
# ---------------------------------------------------------------------------

class TestPartitionIsolation:
    def test_overlapping_partitions_raise(self):
        from src.dataset.split import DatasetSplit

        # Construct a split where test_normal and train_normal share a record
        shared = _make_meta("id_00", "normal", 0)
        split = DatasetSplit(
            train_normal=[shared],
            profile_normal=[],
            test_normal=[shared],
            test_abnormal=[],
        )
        with pytest.raises(ValueError, match="ISOLATION FAIL"):
            script._validate_isolation(split)

    def test_clean_split_does_not_raise(self):
        split = _split(_make_recordings())
        script._validate_isolation(split)


# ---------------------------------------------------------------------------
# G: CSV output
# ---------------------------------------------------------------------------

class TestCSVOutput:
    def test_csv_has_required_columns(self, tmp_path):
        split = _split(_make_recordings(n_normal=10, n_abnormal=5))
        profiles = {(MACHINE_TYPE, mid): _fake_profile(mid) for mid in MACHINE_IDS}

        evaluation_records = (
            [(r, "normal") for r in split.test_normal]
            + [(r, "abnormal") for r in split.test_abnormal]
        )

        rows = []
        for record, true_label in evaluation_records:
            result = _fake_health_result(record)
            rows.append({
                "machine_type": result.machine_type,
                "machine_id": result.machine_id,
                "filename": result.filename,
                "true_label": true_label,
                "health_score": result.health_score,
                "health_percentage": result.health_percentage,
                "health_state": result.health_state,
                "normalized_euclidean": result.normalized_euclidean,
                "normalized_manhattan": result.normalized_manhattan,
                "normalized_cosine": result.normalized_cosine,
            })

        csv_path = tmp_path / "evaluation_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=script.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            written_rows = list(reader)

        assert len(written_rows) == len(evaluation_records)
        for col in script.CSV_COLUMNS:
            assert col in written_rows[0], f"Missing column: {col}"

    def test_csv_row_count_matches_evaluation_size(self, tmp_path):
        split = _split(_make_recordings(n_normal=10, n_abnormal=5))
        evaluation_records = (
            [(r, "normal") for r in split.test_normal]
            + [(r, "abnormal") for r in split.test_abnormal]
        )
        rows = [
            {col: "x" for col in script.CSV_COLUMNS}
            for _ in evaluation_records
        ]
        csv_path = tmp_path / "out.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=script.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        with csv_path.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        assert len(written) == len(evaluation_records)

    def test_csv_one_row_per_recording(self, tmp_path):
        split = _split(_make_recordings(n_normal=5, n_abnormal=3))
        evaluation_records = (
            [(r, "normal") for r in split.test_normal]
            + [(r, "abnormal") for r in split.test_abnormal]
        )
        rows = []
        for record, true_label in evaluation_records:
            rows.append({
                "machine_type": record.machine_type,
                "machine_id": record.machine_id,
                "filename": record.filename,
                "true_label": true_label,
                "health_score": 80.0,
                "health_percentage": "80.0%",
                "health_state": "GOOD",
                "normalized_euclidean": 1.0,
                "normalized_manhattan": 8.0,
                "normalized_cosine": 0.8,
            })
        csv_path = tmp_path / "out.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=script.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with csv_path.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        assert len(written) == len(evaluation_records)


# ---------------------------------------------------------------------------
# H: Checkpoint validation
# ---------------------------------------------------------------------------

class TestCheckpointValidation:
    def test_missing_checkpoint_raises_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "CHECKPOINT_PATH", tmp_path / "missing.pt")
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            script._validate_checkpoint()

    def test_existing_checkpoint_does_not_raise(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "best_projection_head.pt"
        ckpt.write_bytes(b"fake")
        monkeypatch.setattr(script, "CHECKPOINT_PATH", ckpt)
        script._validate_checkpoint()  # must not raise
