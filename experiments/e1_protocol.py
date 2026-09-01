"""Experiment E1: Dataset protocol validation for acoustic fingerprinting.

Establishes and validates the reproducible train/profile/test split for
machine_type=pump using DatasetSplitter(train_ratio=0.70, profile_ratio=0.15, seed=42).

Usage:
    python experiments/e1_protocol.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplit, DatasetSplitter

DATASET_ROOT = Path("data/raw/MIMII")
MACHINE_TYPE = "pump"
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
TRAIN_RATIO = 0.70
PROFILE_RATIO = 0.15
SEED = 42


def build_split(recordings) -> DatasetSplit:
    splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
    return splitter.split(recordings)


def validate(split: DatasetSplit, all_recordings) -> None:
    train_paths = {r.absolute_path for r in split.train_normal}
    profile_paths = {r.absolute_path for r in split.profile_normal}
    test_normal_paths = {r.absolute_path for r in split.test_normal}
    test_abnormal_paths = {r.absolute_path for r in split.test_abnormal}

    # A: No overlap between normal partitions
    assert not train_paths & profile_paths, "FAIL A: train_normal ∩ profile_normal is non-empty"
    assert not train_paths & test_normal_paths, "FAIL A: train_normal ∩ test_normal is non-empty"
    assert not profile_paths & test_normal_paths, "FAIL A: profile_normal ∩ test_normal is non-empty"

    # B: Every normal recording appears exactly once
    all_normal_paths = {r.absolute_path for r in all_recordings if r.label == "normal"}
    covered = train_paths | profile_paths | test_normal_paths
    assert covered == all_normal_paths, "FAIL B: normal recordings not fully covered"

    # C: No abnormal recording in normal partitions
    all_abnormal_paths = {r.absolute_path for r in all_recordings if r.label == "abnormal"}
    assert not (train_paths | profile_paths | test_normal_paths) & all_abnormal_paths, \
        "FAIL C: abnormal recording found in normal partition"

    # D: test_abnormal contains only abnormal recordings
    assert test_abnormal_paths == all_abnormal_paths, \
        "FAIL D: test_abnormal does not match all abnormal recordings"

    # E: Reproducibility — split twice, compare path assignments
    split2 = build_split(all_recordings)
    assert (
        sorted(str(r.absolute_path) for r in split.train_normal) ==
        sorted(str(r.absolute_path) for r in split2.train_normal)
    ), "FAIL E: train_normal not reproducible"
    assert (
        sorted(str(r.absolute_path) for r in split.profile_normal) ==
        sorted(str(r.absolute_path) for r in split2.profile_normal)
    ), "FAIL E: profile_normal not reproducible"
    assert (
        sorted(str(r.absolute_path) for r in split.test_normal) ==
        sorted(str(r.absolute_path) for r in split2.test_normal)
    ), "FAIL E: test_normal not reproducible"

    print("All validation checks passed (A–E).")


def print_summary(split: DatasetSplit, recordings) -> None:
    print("=" * 50)
    print("Experiment ID : E1")
    print(f"Machine type  : {MACHINE_TYPE}")
    print(f"Machine IDs   : {MACHINE_IDS}")
    print(f"Seed          : {SEED}")
    print(f"Train ratio   : {TRAIN_RATIO}  |  Profile ratio: {PROFILE_RATIO}  |  Test ratio: {round(1 - TRAIN_RATIO - PROFILE_RATIO, 2)}")
    print("-" * 50)
    print(f"{'Machine ID':<12} {'train_normal':>12} {'profile_normal':>14} {'test_normal':>11} {'test_abnormal':>13}")
    print("-" * 50)

    total_train = total_profile = total_test_n = total_test_ab = 0

    for mid in MACHINE_IDS:
        tn = sum(1 for r in split.train_normal if r.machine_id == mid)
        pn = sum(1 for r in split.profile_normal if r.machine_id == mid)
        tsn = sum(1 for r in split.test_normal if r.machine_id == mid)
        tab = sum(1 for r in split.test_abnormal if r.machine_id == mid)
        print(f"{mid:<12} {tn:>12} {pn:>14} {tsn:>11} {tab:>13}")
        total_train += tn
        total_profile += pn
        total_test_n += tsn
        total_test_ab += tab

    print("-" * 50)
    print(f"{'TOTAL':<12} {total_train:>12} {total_profile:>14} {total_test_n:>11} {total_test_ab:>13}")
    print("=" * 50)


def main() -> None:
    loader = DatasetLoader(DATASET_ROOT)
    recordings = [r for r in loader.get_all_files() if r.machine_type == MACHINE_TYPE]

    split = build_split(recordings)
    print_summary(split, recordings)
    validate(split, recordings)


if __name__ == "__main__":
    main()
