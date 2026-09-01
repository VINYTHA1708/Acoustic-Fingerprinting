"""Experiment E1 — Held-out evaluation of the trained acoustic fingerprinting system.

Evaluates only split.test_normal and split.test_abnormal recordings.
Reuses LearnedHealthAnalyzer, LearnedProfileSerializer, DatasetLoader, DatasetSplitter.

Usage:
    python experiments/e1_evaluate.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplitter
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
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
RESULTS_PATH = Path("experiments/results/e1/evaluation_results.csv")

# Expected held-out counts per machine ID
_EXPECTED_TEST_NORMAL = {"id_00": 152, "id_02": 152, "id_04": 106, "id_06": 156}
_EXPECTED_TEST_ABNORMAL = {"id_00": 143, "id_02": 111, "id_04": 100, "id_06": 102}

CSV_COLUMNS = [
    "machine_type", "machine_id", "filename", "true_label",
    "health_score", "health_percentage", "health_state",
    "normalized_euclidean", "normalized_manhattan", "normalized_cosine",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_checkpoint() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"E1 checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run experiments/e1_train.py first."
        )


def _validate_profiles() -> None:
    missing = []
    for mid in MACHINE_IDS:
        npz = PROFILE_DIR / f"e1_{MACHINE_TYPE}_{mid}_learned_profile.npz"
        if not npz.exists():
            missing.append(str(npz))
    if missing:
        raise FileNotFoundError(
            "Missing E1 profiles:\n" + "\n".join(missing) +
            "\nRun experiments/e1_build_profiles.py first."
        )


def _validate_split_counts(split) -> None:
    errors = []
    for mid in MACHINE_IDS:
        actual_n = sum(1 for r in split.test_normal if r.machine_id == mid)
        actual_ab = sum(1 for r in split.test_abnormal if r.machine_id == mid)
        if actual_n != _EXPECTED_TEST_NORMAL[mid]:
            errors.append(
                f"{mid} test_normal: expected {_EXPECTED_TEST_NORMAL[mid]}, got {actual_n}"
            )
        if actual_ab != _EXPECTED_TEST_ABNORMAL[mid]:
            errors.append(
                f"{mid} test_abnormal: expected {_EXPECTED_TEST_ABNORMAL[mid]}, got {actual_ab}"
            )
    if errors:
        raise ValueError("E1 split count mismatch:\n" + "\n".join(errors))


def _validate_isolation(split) -> None:
    train_paths = {r.absolute_path for r in split.train_normal}
    profile_paths = {r.absolute_path for r in split.profile_normal}
    test_normal_paths = {r.absolute_path for r in split.test_normal}
    test_abnormal_paths = {r.absolute_path for r in split.test_abnormal}

    checks = [
        (train_paths & profile_paths, "train_normal ∩ profile_normal"),
        (train_paths & test_normal_paths, "train_normal ∩ test_normal"),
        (profile_paths & test_normal_paths, "profile_normal ∩ test_normal"),
        (test_normal_paths & test_abnormal_paths, "test_normal ∩ test_abnormal"),
    ]
    for overlap, label in checks:
        if overlap:
            raise ValueError(f"ISOLATION FAIL: {label} is non-empty ({len(overlap)} files)")


def _validate_evaluation_records(evaluation_records) -> None:
    for record, _ in evaluation_records:
        if record.machine_type != MACHINE_TYPE:
            raise ValueError(
                f"Evaluation record has unexpected machine_type='{record.machine_type}'"
            )


def _validate_profiles_cover_all_records(profiles, evaluation_records) -> None:
    for record, _ in evaluation_records:
        key = (record.machine_type, record.machine_id)
        if key not in profiles:
            raise ValueError(
                f"No loaded profile for ({record.machine_type}, {record.machine_id})"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Pre-flight checks
    _validate_checkpoint()
    _validate_profiles()

    # 2. Load recordings and reproduce E1 split
    loader = DatasetLoader(DATASET_ROOT)
    all_recordings = [
        r for r in loader.get_all_files()
        if r.machine_type == MACHINE_TYPE and r.machine_id in MACHINE_IDS
    ]
    splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
    split = splitter.split(all_recordings)

    # 3. Validate split counts and partition isolation
    _validate_split_counts(split)
    _validate_isolation(split)

    # 4. Console header
    print("=" * 50)
    print(f"Experiment ID : {EXPERIMENT_ID}")
    print(f"Stage         : Evaluation")
    print("=" * 50)
    print()
    print(f"Machine type      : {MACHINE_TYPE}")
    print(f"Checkpoint        : {CHECKPOINT_PATH}")
    print(f"Profiles directory: {PROFILE_DIR}")
    print()
    print(f"Test normal count    : {len(split.test_normal)}")
    print(f"Test abnormal count  : {len(split.test_abnormal)}")
    print(f"Total evaluation     : {len(split.test_normal) + len(split.test_abnormal)}")
    print()
    print(f"{'Machine ID':<12} {'test_normal':>11} {'test_abnormal':>13}")
    print("-" * 40)
    for mid in MACHINE_IDS:
        tn = sum(1 for r in split.test_normal if r.machine_id == mid)
        tab = sum(1 for r in split.test_abnormal if r.machine_id == mid)
        print(f"{mid:<12} {tn:>11} {tab:>13}")
    print()

    # 5. Load profiles (NPZ)
    serializer = LearnedProfileSerializer()
    profiles: dict[tuple[str, str], object] = {}
    for mid in MACHINE_IDS:
        npz_path = PROFILE_DIR / f"e1_{MACHINE_TYPE}_{mid}_learned_profile.npz"
        profiles[(MACHINE_TYPE, mid)] = serializer.load_npz(npz_path)

    # 6. Build evaluation set — ONLY test_normal and test_abnormal
    evaluation_records = (
        [(record, "normal") for record in split.test_normal]
        + [(record, "abnormal") for record in split.test_abnormal]
    )

    # 7. Final record-level validations
    _validate_evaluation_records(evaluation_records)
    _validate_profiles_cover_all_records(profiles, evaluation_records)

    # 8. Run evaluation
    analyzer = LearnedHealthAnalyzer(checkpoint_path=CHECKPOINT_PATH)
    rows = []
    normal_count = 0
    abnormal_count = 0

    for i, (record, true_label) in enumerate(evaluation_records, start=1):
        if i % 50 == 0 or i == 1:
            print(f"  [{i}/{len(evaluation_records)}] {record.machine_type}/{record.machine_id} — {record.filename}")

        profile = profiles[(record.machine_type, record.machine_id)]
        result = analyzer.analyze(record, profile)

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

        if true_label == "normal":
            normal_count += 1
        else:
            abnormal_count += 1

    # 9. Save CSV
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # 10. Summary
    print()
    print("Evaluation completed successfully.")
    print()
    print(f"Normal recordings evaluated   : {normal_count}")
    print(f"Abnormal recordings evaluated : {abnormal_count}")
    print(f"Total recordings evaluated    : {normal_count + abnormal_count}")
    print()
    print("Results saved to:")
    print(f"  {RESULTS_PATH}")


if __name__ == "__main__":
    main()
