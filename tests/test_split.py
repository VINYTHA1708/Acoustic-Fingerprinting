"""Tests for DatasetSplitter and DatasetSplit (src/dataset/split.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset.metadata import AudioMetadata
from src.dataset.split import DatasetSplit, DatasetSplitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(machine_type: str, machine_id: str, label: str, filename: str) -> AudioMetadata:
    """Construct a minimal AudioMetadata for testing (no real file required)."""
    rel = Path(machine_type) / machine_id / label / filename
    return AudioMetadata(
        machine_type=machine_type,
        machine_id=machine_id,
        label=label,
        filename=filename,
        relative_path=rel,
        absolute_path=Path("/fake") / rel,
    )


def _normal(n: int, machine_type: str = "pump", machine_id: str = "id_00") -> list[AudioMetadata]:
    return [_make(machine_type, machine_id, "normal", f"{i:08d}.wav") for i in range(n)]


def _abnormal(n: int, machine_type: str = "pump", machine_id: str = "id_00") -> list[AudioMetadata]:
    return [_make(machine_type, machine_id, "abnormal", f"ab_{i:08d}.wav") for i in range(n)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recordings_single_machine() -> list[AudioMetadata]:
    """20 normal + 5 abnormal recordings for a single machine."""
    return _normal(20) + _abnormal(5)


@pytest.fixture(scope="module")
def split_single(recordings_single_machine) -> DatasetSplit:
    return DatasetSplitter().split(recordings_single_machine)


# ---------------------------------------------------------------------------
# 1. Basic split
# ---------------------------------------------------------------------------

class TestBasicSplit:
    """Normal recordings are divided into three partitions; abnormal go to test_abnormal."""

    def test_returns_dataset_split(self, split_single):
        """split() returns a DatasetSplit instance."""
        assert isinstance(split_single, DatasetSplit)

    def test_train_normal_non_empty(self, split_single):
        """train_normal contains recordings."""
        assert len(split_single.train_normal) > 0

    def test_profile_normal_non_empty(self, split_single):
        """profile_normal contains recordings."""
        assert len(split_single.profile_normal) > 0

    def test_test_normal_non_empty(self, split_single):
        """test_normal contains recordings."""
        assert len(split_single.test_normal) > 0

    def test_all_normal_recordings_accounted_for(self, recordings_single_machine, split_single):
        """Every normal recording ends up in exactly one normal partition."""
        normal_input = {r.filename for r in recordings_single_machine if r.label == "normal"}
        normal_output = (
            {r.filename for r in split_single.train_normal}
            | {r.filename for r in split_single.profile_normal}
            | {r.filename for r in split_single.test_normal}
        )
        assert normal_input == normal_output

    def test_abnormal_recordings_go_to_test_abnormal(self, recordings_single_machine, split_single):
        """All abnormal recordings appear in test_abnormal."""
        abnormal_input = {r.filename for r in recordings_single_machine if r.label == "abnormal"}
        assert {r.filename for r in split_single.test_abnormal} == abnormal_input

    def test_no_abnormal_in_normal_partitions(self, split_single):
        """No abnormal recording leaks into any normal partition."""
        for partition in (split_single.train_normal, split_single.profile_normal, split_single.test_normal):
            assert all(r.label == "normal" for r in partition)

    def test_approximate_train_ratio(self, split_single):
        """train_normal is roughly 60 % of normal recordings (±1 recording tolerance)."""
        total = len(split_single.train_normal) + len(split_single.profile_normal) + len(split_single.test_normal)
        expected = int(total * 0.6)
        assert abs(len(split_single.train_normal) - expected) <= 1

    def test_approximate_profile_ratio(self, split_single):
        """profile_normal is roughly 20 % of normal recordings (±1 recording tolerance)."""
        total = len(split_single.train_normal) + len(split_single.profile_normal) + len(split_single.test_normal)
        expected = int(total * 0.2)
        assert abs(len(split_single.profile_normal) - expected) <= 1


# ---------------------------------------------------------------------------
# 2. No overlap
# ---------------------------------------------------------------------------

class TestNoOverlap:
    """No normal recording appears in more than one normal partition."""

    def test_train_profile_disjoint(self, split_single):
        train = {r.filename for r in split_single.train_normal}
        profile = {r.filename for r in split_single.profile_normal}
        assert train.isdisjoint(profile)

    def test_train_test_disjoint(self, split_single):
        train = {r.filename for r in split_single.train_normal}
        test = {r.filename for r in split_single.test_normal}
        assert train.isdisjoint(test)

    def test_profile_test_disjoint(self, split_single):
        profile = {r.filename for r in split_single.profile_normal}
        test = {r.filename for r in split_single.test_normal}
        assert profile.isdisjoint(test)


# ---------------------------------------------------------------------------
# 3. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Same seed → identical splits across two independent splitter instances."""

    def test_same_seed_produces_identical_train(self, recordings_single_machine):
        s1 = DatasetSplitter(seed=42).split(recordings_single_machine)
        s2 = DatasetSplitter(seed=42).split(recordings_single_machine)
        assert [r.filename for r in s1.train_normal] == [r.filename for r in s2.train_normal]

    def test_same_seed_produces_identical_profile(self, recordings_single_machine):
        s1 = DatasetSplitter(seed=42).split(recordings_single_machine)
        s2 = DatasetSplitter(seed=42).split(recordings_single_machine)
        assert [r.filename for r in s1.profile_normal] == [r.filename for r in s2.profile_normal]

    def test_same_seed_produces_identical_test_normal(self, recordings_single_machine):
        s1 = DatasetSplitter(seed=42).split(recordings_single_machine)
        s2 = DatasetSplitter(seed=42).split(recordings_single_machine)
        assert [r.filename for r in s1.test_normal] == [r.filename for r in s2.test_normal]


# ---------------------------------------------------------------------------
# 4. Different seed
# ---------------------------------------------------------------------------

class TestDifferentSeed:
    """A different seed is permitted to produce a different ordering."""

    def test_different_seed_may_differ(self, recordings_single_machine):
        """Splits with seed=0 and seed=99 are not required to match."""
        s1 = DatasetSplitter(seed=0).split(recordings_single_machine)
        s2 = DatasetSplitter(seed=99).split(recordings_single_machine)
        # At least one partition should differ in ordering (true for any non-trivial dataset)
        combined_s1 = [r.filename for r in s1.train_normal + s1.profile_normal + s1.test_normal]
        combined_s2 = [r.filename for r in s2.train_normal + s2.profile_normal + s2.test_normal]
        assert combined_s1 != combined_s2


# ---------------------------------------------------------------------------
# 5. Per-machine grouping
# ---------------------------------------------------------------------------

class TestPerMachineGrouping:
    """Both machine IDs contribute recordings to every normal partition."""

    @pytest.fixture
    def two_machine_split(self) -> DatasetSplit:
        recordings = (
            _normal(20, machine_id="id_00")
            + _normal(20, machine_id="id_02")
        )
        return DatasetSplitter().split(recordings)

    def test_both_ids_in_train(self, two_machine_split):
        ids = {r.machine_id for r in two_machine_split.train_normal}
        assert {"id_00", "id_02"} == ids

    def test_both_ids_in_profile(self, two_machine_split):
        ids = {r.machine_id for r in two_machine_split.profile_normal}
        assert {"id_00", "id_02"} == ids

    def test_both_ids_in_test_normal(self, two_machine_split):
        ids = {r.machine_id for r in two_machine_split.test_normal}
        assert {"id_00", "id_02"} == ids

# ---------------------------------------------------------------------------
# 6. Invalid label
# ---------------------------------------------------------------------------

class TestInvalidLabel:
    """An unrecognised label raises ValueError."""

    def test_unknown_label_raises_value_error(self):
        bad = [_make("pump", "id_00", "unknown", "bad.wav")]
        with pytest.raises(ValueError, match="unknown"):
            DatasetSplitter().split(bad)

    def test_error_message_contains_label(self):
        bad = [_make("pump", "id_00", "corrupted", "bad.wav")]
        with pytest.raises(ValueError, match="corrupted"):
            DatasetSplitter().split(bad)
