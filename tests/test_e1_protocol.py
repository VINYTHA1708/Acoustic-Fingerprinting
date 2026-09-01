"""Tests for Experiment E1 dataset protocol.

Uses synthetic AudioMetadata objects — no BEATs, no audio encoding required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.dataset.metadata import AudioMetadata
from src.dataset.split import DatasetSplit, DatasetSplitter

TRAIN_RATIO = 0.70
PROFILE_RATIO = 0.15
SEED = 42


def _make_meta(machine_id: str, label: str, idx: int) -> AudioMetadata:
    p = Path(f"/fake/pump/{machine_id}/{label}/{idx:08d}.wav")
    return AudioMetadata(
        machine_type="pump",
        machine_id=machine_id,
        label=label,
        filename=p.name,
        relative_path=Path(f"pump/{machine_id}/{label}/{idx:08d}.wav"),
        absolute_path=p,
    )


def _make_dataset(n_normal: int = 100, n_abnormal: int = 20, ids=("id_00", "id_02", "id_04", "id_06")):
    recordings = []
    for mid in ids:
        for i in range(n_normal):
            recordings.append(_make_meta(mid, "normal", i))
        for i in range(n_abnormal):
            recordings.append(_make_meta(mid, "abnormal", i))
    return recordings


def _split(recordings) -> DatasetSplit:
    return DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED).split(recordings)


# ---------------------------------------------------------------------------
# A: No overlap between normal partitions
# ---------------------------------------------------------------------------

def test_no_overlap_between_normal_partitions():
    recordings = _make_dataset()
    split = _split(recordings)

    train = {r.absolute_path for r in split.train_normal}
    profile = {r.absolute_path for r in split.profile_normal}
    test_n = {r.absolute_path for r in split.test_normal}

    assert not train & profile, "train_normal and profile_normal overlap"
    assert not train & test_n, "train_normal and test_normal overlap"
    assert not profile & test_n, "profile_normal and test_normal overlap"


# ---------------------------------------------------------------------------
# B: Complete normal coverage
# ---------------------------------------------------------------------------

def test_complete_normal_coverage():
    recordings = _make_dataset()
    split = _split(recordings)

    all_normal = {r.absolute_path for r in recordings if r.label == "normal"}
    covered = (
        {r.absolute_path for r in split.train_normal}
        | {r.absolute_path for r in split.profile_normal}
        | {r.absolute_path for r in split.test_normal}
    )
    assert covered == all_normal


# ---------------------------------------------------------------------------
# C: Abnormal recordings do not appear in normal partitions
# ---------------------------------------------------------------------------

def test_abnormal_not_in_normal_partitions():
    recordings = _make_dataset()
    split = _split(recordings)

    abnormal_paths = {r.absolute_path for r in recordings if r.label == "abnormal"}
    normal_partition_paths = (
        {r.absolute_path for r in split.train_normal}
        | {r.absolute_path for r in split.profile_normal}
        | {r.absolute_path for r in split.test_normal}
    )
    assert not normal_partition_paths & abnormal_paths


# ---------------------------------------------------------------------------
# D: test_abnormal contains only abnormal recordings
# ---------------------------------------------------------------------------

def test_test_abnormal_contains_only_abnormal():
    recordings = _make_dataset()
    split = _split(recordings)

    all_abnormal = {r.absolute_path for r in recordings if r.label == "abnormal"}
    test_ab = {r.absolute_path for r in split.test_abnormal}
    assert test_ab == all_abnormal


# ---------------------------------------------------------------------------
# E: Reproducibility with seed 42
# ---------------------------------------------------------------------------

def test_reproducibility_with_seed_42():
    recordings = _make_dataset()
    split1 = _split(recordings)
    split2 = _split(recordings)

    assert sorted(str(r.absolute_path) for r in split1.train_normal) == \
           sorted(str(r.absolute_path) for r in split2.train_normal)
    assert sorted(str(r.absolute_path) for r in split1.profile_normal) == \
           sorted(str(r.absolute_path) for r in split2.profile_normal)
    assert sorted(str(r.absolute_path) for r in split1.test_normal) == \
           sorted(str(r.absolute_path) for r in split2.test_normal)


# ---------------------------------------------------------------------------
# Ratio sanity: approximate partition sizes
# ---------------------------------------------------------------------------

def test_partition_size_ratios():
    n_normal = 100
    recordings = _make_dataset(n_normal=n_normal, ids=("id_00",))
    split = _split(recordings)

    assert len(split.train_normal) == pytest.approx(n_normal * TRAIN_RATIO, abs=2)
    assert len(split.profile_normal) == pytest.approx(n_normal * PROFILE_RATIO, abs=2)
