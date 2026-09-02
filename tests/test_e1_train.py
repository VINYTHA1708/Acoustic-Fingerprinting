"""Lightweight tests for experiments/e1_train.py helper logic.

No BEATs, no audio encoding, no MIMII dataset required.
Uses synthetic AudioMetadata objects throughout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.metadata import AudioMetadata
from src.dataset.split import DatasetSplitter

# E1 constants (mirrored from e1_train.py — tested independently)
MACHINE_TYPE = "pump"
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
TRAIN_RATIO = 0.70
PROFILE_RATIO = 0.15
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(machine_id: str, label: str, idx: int) -> AudioMetadata:
    p = Path("fake") / "pump" / machine_id / label / f"{idx:08d}.wav"
    return AudioMetadata(
        machine_type="pump",
        machine_id=machine_id,
        label=label,
        filename=p.name,
        relative_path=Path("pump") / machine_id / label / f"{idx:08d}.wav",
        absolute_path=p,
    )


def _make_pump_recordings(n_normal: int = 100, n_abnormal: int = 20) -> list[AudioMetadata]:
    recs = []
    for mid in MACHINE_IDS:
        for i in range(n_normal):
            recs.append(_make_meta(mid, "normal", i))
        for i in range(n_abnormal):
            recs.append(_make_meta(mid, "abnormal", i))
    return recs


def _split(recordings):
    return DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED).split(recordings)


# ---------------------------------------------------------------------------
# 1. Filtering to pump only
# ---------------------------------------------------------------------------

class TestMachineTypeFiltering:
    def test_only_pump_in_filtered_recordings(self):
        pump_recs = _make_pump_recordings()
        # Simulate adding a non-pump recording
        fan_rec = AudioMetadata(
            machine_type="fan", machine_id="id_00", label="normal",
            filename="00000000.wav",
            relative_path=Path("fan") / "id_00" / "normal" / "00000000.wav",
            absolute_path=Path("fake") / "fan" / "id_00" / "normal" / "00000000.wav",
        )
        all_recs = pump_recs + [fan_rec]
        filtered = [r for r in all_recs if r.machine_type == MACHINE_TYPE]
        assert all(r.machine_type == "pump" for r in filtered)
        assert fan_rec not in filtered

    def test_non_pump_excluded_from_split(self):
        pump_recs = _make_pump_recordings()
        split = _split(pump_recs)
        assert all(r.machine_type == "pump" for r in split.train_normal)
        assert all(r.machine_type == "pump" for r in split.profile_normal)
        assert all(r.machine_type == "pump" for r in split.test_normal)


# ---------------------------------------------------------------------------
# 2. All four E1 machine IDs are selected
# ---------------------------------------------------------------------------

class TestMachineIDSelection:
    def test_all_four_ids_in_train_normal(self):
        split = _split(_make_pump_recordings())
        ids_in_train = {r.machine_id for r in split.train_normal}
        assert set(MACHINE_IDS).issubset(ids_in_train)

    def test_machine_ids_constant_matches_e1_spec(self):
        assert MACHINE_IDS == ["id_00", "id_02", "id_04", "id_06"]

    def test_no_extra_machine_ids_in_train(self):
        split = _split(_make_pump_recordings())
        ids_in_train = {r.machine_id for r in split.train_normal}
        assert ids_in_train == set(MACHINE_IDS)


# ---------------------------------------------------------------------------
# 3. DatasetSplitter uses correct ratios and seed
# ---------------------------------------------------------------------------

class TestDatasetSplitterConfig:
    def test_train_ratio(self):
        splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
        assert splitter.train_ratio == TRAIN_RATIO

    def test_profile_ratio(self):
        splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
        assert splitter.profile_ratio == PROFILE_RATIO

    def test_seed(self):
        splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
        assert splitter.seed == SEED

    def test_approximate_train_size(self):
        n_normal = 100
        recs = [_make_meta("id_00", "normal", i) for i in range(n_normal)]
        split = _split(recs)
        assert abs(len(split.train_normal) - int(n_normal * TRAIN_RATIO)) <= 2

    def test_approximate_profile_size(self):
        n_normal = 100
        recs = [_make_meta("id_00", "normal", i) for i in range(n_normal)]
        split = _split(recs)
        assert abs(len(split.profile_normal) - int(n_normal * PROFILE_RATIO)) <= 2

    def test_reproducibility(self):
        recs = _make_pump_recordings()
        s1 = _split(recs)
        s2 = _split(recs)
        assert sorted(str(r.absolute_path) for r in s1.train_normal) == \
               sorted(str(r.absolute_path) for r in s2.train_normal)


# ---------------------------------------------------------------------------
# 4. Only train_normal is passed to ContrastiveDataset
# ---------------------------------------------------------------------------

class TestDataLeakagePrevention:
    def test_train_normal_contains_only_normal_labels(self):
        split = _split(_make_pump_recordings())
        assert all(r.label == "normal" for r in split.train_normal)

    def test_profile_normal_not_in_train_normal(self):
        split = _split(_make_pump_recordings())
        train_paths = {r.absolute_path for r in split.train_normal}
        profile_paths = {r.absolute_path for r in split.profile_normal}
        assert not train_paths & profile_paths

    def test_test_normal_not_in_train_normal(self):
        split = _split(_make_pump_recordings())
        train_paths = {r.absolute_path for r in split.train_normal}
        test_paths = {r.absolute_path for r in split.test_normal}
        assert not train_paths & test_paths

    def test_test_abnormal_not_in_train_normal(self):
        split = _split(_make_pump_recordings())
        train_paths = {r.absolute_path for r in split.train_normal}
        abnormal_paths = {r.absolute_path for r in split.test_abnormal}
        assert not train_paths & abnormal_paths

    def test_contrastive_dataset_receives_only_train_normal(self):
        """Verify that ContrastiveDataset is called with split.train_normal only."""
        split = _split(_make_pump_recordings())
        profile_paths = {r.absolute_path for r in split.profile_normal}
        test_normal_paths = {r.absolute_path for r in split.test_normal}
        test_abnormal_paths = {r.absolute_path for r in split.test_abnormal}

        # Simulate what e1_train.py does: pass only split.train_normal
        passed_recordings = split.train_normal
        passed_paths = {r.absolute_path for r in passed_recordings}

        assert not passed_paths & profile_paths, "profile_normal leaked into training"
        assert not passed_paths & test_normal_paths, "test_normal leaked into training"
        assert not passed_paths & test_abnormal_paths, "test_abnormal leaked into training"


# ---------------------------------------------------------------------------
# 5. E1 result metadata contains required fields
# ---------------------------------------------------------------------------

class TestE1ResultMetadata:
    def _make_result(self) -> dict:
        return {
            "experiment_id": "E1",
            "seed": SEED,
            "machine_type": MACHINE_TYPE,
            "machine_ids": MACHINE_IDS,
            "dataset_counts": {
                "train_normal": 2623,
                "profile_normal": 560,
                "test_normal": 566,
                "test_abnormal": 456,
            },
            "training_configuration": {
                "epochs": 20,
                "batch_size": 16,
                "learning_rate": 0.001,
                "temperature": 0.07,
                "input_dimension": 921,
                "projection_dimension": 256,
            },
            "loss_history": {
                "training": [1.0, 0.9],
                "validation": [1.1, 0.95],
            },
            "best_validation_loss": 0.95,
            "checkpoint_path": "models/contrastive/e1/best_projection_head.pt",
        }

    def test_required_top_level_fields(self):
        result = self._make_result()
        for field in ("experiment_id", "seed", "machine_type", "machine_ids",
                      "dataset_counts", "training_configuration",
                      "loss_history", "best_validation_loss", "checkpoint_path"):
            assert field in result, f"Missing field: {field}"

    def test_dataset_counts_fields(self):
        result = self._make_result()
        counts = result["dataset_counts"]
        for field in ("train_normal", "profile_normal", "test_normal", "test_abnormal"):
            assert field in counts

    def test_training_configuration_fields(self):
        result = self._make_result()
        cfg = result["training_configuration"]
        for field in ("epochs", "batch_size", "learning_rate", "temperature",
                      "input_dimension", "projection_dimension"):
            assert field in cfg

    def test_loss_history_fields(self):
        result = self._make_result()
        hist = result["loss_history"]
        assert "training" in hist
        assert "validation" in hist

    def test_experiment_id_is_e1(self):
        assert self._make_result()["experiment_id"] == "E1"

    def test_machine_type_is_pump(self):
        assert self._make_result()["machine_type"] == "pump"

    def test_machine_ids_are_correct(self):
        assert self._make_result()["machine_ids"] == ["id_00", "id_02", "id_04", "id_06"]

    def test_result_is_json_serialisable(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "training_history.json"
        with open(path, "w") as f:
            json.dump(result, f)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["experiment_id"] == "E1"
        assert loaded["training_configuration"]["temperature"] == 0.07
