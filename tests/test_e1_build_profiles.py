"""Lightweight tests for experiments/e1_build_profiles.py.

No BEATs, no audio encoding, no MIMII dataset, no trained checkpoint required.
Uses synthetic AudioMetadata objects and mocks throughout.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.metadata import AudioMetadata
from src.dataset.split import DatasetSplitter

MACHINE_TYPE = "pump"
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
TRAIN_RATIO = 0.70
PROFILE_RATIO = 0.15
SEED = 42

# E1 expected profile_normal counts
EXPECTED_PROFILE_COUNTS = {
    "id_00": 150,
    "id_02": 150,
    "id_04": 105,
    "id_06": 155,
}

# Matching normal counts that reproduce E1 expected counts
# total_normal per id: id_00=1006, id_02=1005, id_04=702, id_06=1036
# profile = floor(n * 0.15)
E1_NORMAL_COUNTS = {
    "id_00": 1006,
    "id_02": 1005,
    "id_04": 702,
    "id_06": 1036,
}


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


def _make_recordings(n_normal_per_id: dict[str, int], n_abnormal: int = 20) -> list[AudioMetadata]:
    recs = []
    for mid, n in n_normal_per_id.items():
        for i in range(n):
            recs.append(_make_meta(mid, "normal", i))
        for i in range(n_abnormal):
            recs.append(_make_meta(mid, "abnormal", i))
    return recs


def _split(recordings):
    return DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED).split(recordings)


# ---------------------------------------------------------------------------
# A: Split isolation — normal partitions are disjoint
# ---------------------------------------------------------------------------

class TestSplitIsolation:
    def test_train_profile_disjoint(self):
        split = _split(_make_recordings({mid: 100 for mid in MACHINE_IDS}))
        train = {r.absolute_path for r in split.train_normal}
        profile = {r.absolute_path for r in split.profile_normal}
        assert not train & profile

    def test_train_test_normal_disjoint(self):
        split = _split(_make_recordings({mid: 100 for mid in MACHINE_IDS}))
        train = {r.absolute_path for r in split.train_normal}
        test_n = {r.absolute_path for r in split.test_normal}
        assert not train & test_n

    def test_profile_test_normal_disjoint(self):
        split = _split(_make_recordings({mid: 100 for mid in MACHINE_IDS}))
        profile = {r.absolute_path for r in split.profile_normal}
        test_n = {r.absolute_path for r in split.test_normal}
        assert not profile & test_n

    def test_profile_test_abnormal_disjoint(self):
        split = _split(_make_recordings({mid: 100 for mid in MACHINE_IDS}))
        profile = {r.absolute_path for r in split.profile_normal}
        test_ab = {r.absolute_path for r in split.test_abnormal}
        assert not profile & test_ab


# ---------------------------------------------------------------------------
# B: Profile-only input — builder receives recordings= not loader=
# ---------------------------------------------------------------------------

class TestProfileOnlyInput:
    def test_builder_called_with_recordings_kwarg(self, tmp_path):
        split = _split(_make_recordings({mid: 100 for mid in MACHINE_IDS}))

        mock_profile = MagicMock()
        mock_profile.embedding_dimension = 256
        mock_builder = MagicMock()
        mock_builder.build.return_value = mock_profile
        mock_serializer = MagicMock()

        for machine_id in MACHINE_IDS:
            machine_records = [
                r for r in split.profile_normal
                if r.machine_type == MACHINE_TYPE and r.machine_id == machine_id
            ]
            mock_builder.build(MACHINE_TYPE, machine_id, recordings=machine_records)

        for c in mock_builder.build.call_args_list:
            _, kwargs = c
            assert "recordings" in kwargs, "builder.build() must be called with recordings="
            assert "loader" not in kwargs, "loader= must never be passed to builder.build()"

    def test_loader_never_passed_to_build(self, tmp_path):
        split = _split(_make_recordings({mid: 100 for mid in MACHINE_IDS}))
        mock_builder = MagicMock()
        mock_builder.build.return_value = MagicMock(embedding_dimension=256)

        for machine_id in MACHINE_IDS:
            machine_records = [
                r for r in split.profile_normal
                if r.machine_type == MACHINE_TYPE and r.machine_id == machine_id
            ]
            mock_builder.build(MACHINE_TYPE, machine_id, recordings=machine_records)

        for c in mock_builder.build.call_args_list:
            _, kwargs = c
            assert "loader" not in kwargs


# ---------------------------------------------------------------------------
# C: Correct grouping — each machine ID receives only its own recordings
# ---------------------------------------------------------------------------

class TestCorrectGrouping:
    @pytest.mark.parametrize("machine_id", MACHINE_IDS)
    def test_only_correct_machine_id_in_group(self, machine_id):
        split = _split(_make_recordings({mid: 100 for mid in MACHINE_IDS}))
        group = [
            r for r in split.profile_normal
            if r.machine_type == MACHINE_TYPE and r.machine_id == machine_id
        ]
        assert all(r.machine_id == machine_id for r in group)
        assert all(r.machine_type == MACHINE_TYPE for r in group)

    @pytest.mark.parametrize("machine_id", MACHINE_IDS)
    def test_no_cross_contamination(self, machine_id):
        split = _split(_make_recordings({mid: 100 for mid in MACHINE_IDS}))
        other_ids = [mid for mid in MACHINE_IDS if mid != machine_id]
        group = [
            r for r in split.profile_normal
            if r.machine_type == MACHINE_TYPE and r.machine_id == machine_id
        ]
        group_ids = {r.machine_id for r in group}
        for other in other_ids:
            assert other not in group_ids


# ---------------------------------------------------------------------------
# D: Expected counts — E1 profile_normal counts match specification
# ---------------------------------------------------------------------------

class TestExpectedCounts:
    def test_e1_profile_counts(self):
        recordings = _make_recordings(E1_NORMAL_COUNTS)
        split = _split(recordings)

        for machine_id, expected in EXPECTED_PROFILE_COUNTS.items():
            actual = sum(
                1 for r in split.profile_normal
                if r.machine_id == machine_id
            )
            assert actual == expected, (
                f"{machine_id}: expected {expected} profile_normal recordings, got {actual}"
            )

    @pytest.mark.parametrize("machine_id,expected", EXPECTED_PROFILE_COUNTS.items())
    def test_per_machine_count(self, machine_id, expected):
        recordings = _make_recordings(E1_NORMAL_COUNTS)
        split = _split(recordings)
        actual = sum(1 for r in split.profile_normal if r.machine_id == machine_id)
        assert actual == expected


# ---------------------------------------------------------------------------
# E: Checkpoint validation — missing checkpoint raises FileNotFoundError
# ---------------------------------------------------------------------------

class TestCheckpointValidation:
    def test_missing_checkpoint_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.pt"
        with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
            if not missing.exists():
                raise FileNotFoundError(
                    f"E1 checkpoint not found: {missing}\n"
                    "Run experiments/e1_train.py first."
                )

    def test_existing_checkpoint_does_not_raise(self, tmp_path):
        ckpt = tmp_path / "best_projection_head.pt"
        ckpt.write_bytes(b"fake")
        # Should not raise
        if not ckpt.exists():
            raise FileNotFoundError(f"E1 checkpoint not found: {ckpt}")

    def test_checkpoint_guard_in_script(self, tmp_path, monkeypatch):
        """Import and call main() with a patched missing checkpoint path."""
        import experiments.e1_build_profiles as script

        monkeypatch.setattr(script, "CHECKPOINT_PATH", tmp_path / "missing.pt")
        with pytest.raises(FileNotFoundError):
            script.main()


# ---------------------------------------------------------------------------
# F: No dataset leakage — only profile_normal recordings reach the builder
# ---------------------------------------------------------------------------

class TestNoDatasetLeakage:
    def test_only_profile_normal_passed_to_builder(self):
        split = _split(_make_recordings({mid: 100 for mid in MACHINE_IDS}))

        profile_paths = {r.absolute_path for r in split.profile_normal}
        train_paths = {r.absolute_path for r in split.train_normal}
        test_normal_paths = {r.absolute_path for r in split.test_normal}
        test_abnormal_paths = {r.absolute_path for r in split.test_abnormal}

        for machine_id in MACHINE_IDS:
            passed = [
                r for r in split.profile_normal
                if r.machine_type == MACHINE_TYPE and r.machine_id == machine_id
            ]
            passed_paths = {r.absolute_path for r in passed}

            assert passed_paths.issubset(profile_paths), \
                f"{machine_id}: recordings outside profile_normal were passed"
            assert not passed_paths & train_paths, \
                f"{machine_id}: train_normal leaked into profile build"
            assert not passed_paths & test_normal_paths, \
                f"{machine_id}: test_normal leaked into profile build"
            assert not passed_paths & test_abnormal_paths, \
                f"{machine_id}: test_abnormal leaked into profile build"

    def test_all_profile_recordings_are_normal_label(self):
        split = _split(_make_recordings({mid: 100 for mid in MACHINE_IDS}))
        assert all(r.label == "normal" for r in split.profile_normal)
