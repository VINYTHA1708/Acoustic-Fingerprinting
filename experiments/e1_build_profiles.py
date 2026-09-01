"""Experiment E1 — Build healthy learned fingerprint profiles.

Uses only split.profile_normal recordings for each machine ID.
Checkpoint: models/contrastive/e1/best_projection_head.pt

Usage:
    python experiments/e1_build_profiles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplitter
from src.learned_profile.builder import LearnedProfileBuilder
from src.learned_profile.serializer import LearnedProfileSerializer

# ---------------------------------------------------------------------------
# E1 constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "E1"
DATASET_ROOT = Path("data/raw/MIMII")
MACHINE_TYPE = "pump"
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]

TRAIN_RATIO = 0.70
PROFILE_RATIO = 0.15
SEED = 42

CHECKPOINT_PATH = Path("models/contrastive/e1/best_projection_head.pt")
PROFILE_DIR = Path("experiments/results/e1/profiles")


# ---------------------------------------------------------------------------
# Split isolation check
# ---------------------------------------------------------------------------

def _verify_isolation(split) -> None:
    train_paths = {r.absolute_path for r in split.train_normal}
    profile_paths = {r.absolute_path for r in split.profile_normal}
    test_normal_paths = {r.absolute_path for r in split.test_normal}
    test_abnormal_paths = {r.absolute_path for r in split.test_abnormal}

    assert not (train_paths & profile_paths), \
        "ISOLATION FAIL: train_normal ∩ profile_normal is non-empty"
    assert not (train_paths & test_normal_paths), \
        "ISOLATION FAIL: train_normal ∩ test_normal is non-empty"
    assert not (profile_paths & test_normal_paths), \
        "ISOLATION FAIL: profile_normal ∩ test_normal is non-empty"
    assert not (profile_paths & test_abnormal_paths), \
        "ISOLATION FAIL: profile_normal ∩ test_abnormal is non-empty"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Checkpoint guard
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"E1 checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run experiments/e1_train.py first."
        )

    # 2. Load recordings and reproduce E1 split
    loader = DatasetLoader(DATASET_ROOT)
    all_recordings = [
        r for r in loader.get_all_files()
        if r.machine_type == MACHINE_TYPE and r.machine_id in MACHINE_IDS
    ]

    splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
    split = splitter.split(all_recordings)

    # 3. Enforce data isolation
    _verify_isolation(split)

    # 4. Summary header
    print("=" * 50)
    print(f"Experiment ID : {EXPERIMENT_ID}")
    print(f"Stage         : Healthy Profile Construction")
    print("=" * 50)
    print()
    print("Machine type:")
    print(MACHINE_TYPE)
    print()
    print("Checkpoint:")
    print(CHECKPOINT_PATH)
    print()
    print("Profile recordings:")
    for mid in MACHINE_IDS:
        count = sum(1 for r in split.profile_normal if r.machine_id == mid)
        print(f"  {mid} : {count}")
    print()
    print("=" * 50)
    print()

    # 5. Build one profile per machine ID using ONLY profile_normal
    builder = LearnedProfileBuilder(checkpoint_path=CHECKPOINT_PATH)
    serializer = LearnedProfileSerializer()
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    for machine_id in MACHINE_IDS:
        machine_records = [
            r for r in split.profile_normal
            if r.machine_type == MACHINE_TYPE and r.machine_id == machine_id
        ]

        profile = builder.build(
            MACHINE_TYPE,
            machine_id,
            recordings=machine_records,
        )

        print(f"Built profile: {MACHINE_TYPE}/{machine_id}")
        print(f"  Healthy recordings used : {len(machine_records)}")
        print(f"  Embedding dimension     : {profile.embedding_dimension}")

        # 6. Save (JSON + NPZ)
        stem = f"e1_{MACHINE_TYPE}_{machine_id}_learned_profile"
        json_path = PROFILE_DIR / f"{stem}.json"
        npz_path = PROFILE_DIR / f"{stem}.npz"

        serializer.save_json(profile, json_path)
        serializer.save_npz(profile, npz_path)

        print(f"  Saved JSON : {json_path}")
        print(f"  Saved NPZ  : {npz_path}")
        print()

    # 7. Metadata note
    print("Metadata note:")
    print("  LearnedFingerprintProfile does not have a dedicated metadata field.")
    print("  Experiment context is encoded in the file names (e1_ prefix) and")
    print("  the directory path (experiments/results/e1/profiles/).")
    print()
    print(f"  experiment_id    : {EXPERIMENT_ID}")
    print(f"  machine_type     : {MACHINE_TYPE}")
    print(f"  checkpoint_path  : {CHECKPOINT_PATH}")
    print(f"  split_seed       : {SEED}")
    print(f"  train_ratio      : {TRAIN_RATIO}")
    print(f"  profile_ratio    : {PROFILE_RATIO}")
    print()
    print("All E1 profiles saved to:", PROFILE_DIR)


if __name__ == "__main__":
    main()
