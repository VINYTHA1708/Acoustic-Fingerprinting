"""Phase 9 — Multi-Machine Contrastive Training on all four MIMII machine types.

Trains ONE shared ProjectionHead jointly on fan, pump, slider, and valve using
the identical architecture, hyperparameters, and training pipeline as e1_train.py.

Split protocol (per machine type, independently):
    DatasetSplitter(train_ratio=0.70, profile_ratio=0.15, seed=42)

Only train_normal recordings are passed to ContrastiveDataset.
profile_normal, test_normal, and test_abnormal are never touched.

Usage:
    python experiments/phase9_train.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contrastive_learning.dataset import ContrastiveDataset
from src.contrastive_learning.loss import NTXentLoss
from src.contrastive_learning.model import ProjectionHead
from src.contrastive_learning.trainer import ContrastiveTrainer
from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplit, DatasetSplitter

# ---------------------------------------------------------------------------
# Phase 9 constants  (hyperparameters identical to e1_train.py)
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "phase9"
DATASET_ROOT  = Path("data/raw/MIMII")
CACHE_ROOT    = Path("data/fusion_cache")

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]

TRAIN_RATIO   = 0.70
PROFILE_RATIO = 0.15
SEED          = 42

EPOCHS        = 20
BATCH_SIZE    = 16
LEARNING_RATE = 0.001
TEMPERATURE   = 0.07
INPUT_DIM     = 921
PROJECTION_DIM = 256

CHECKPOINT_DIR = Path("models/contrastive/phase9")

# ---------------------------------------------------------------------------
# Verified split counts (computed from the real dataset with seed=42).
# These are asserted at runtime to catch any dataset or splitter regression.
# ---------------------------------------------------------------------------

_EXPECTED_TRAIN_NORMAL: dict[str, int] = {
    "fan":    2851,
    "pump":   2623,
    "slider": 2240,
    "valve":  2582,
}

_EXPECTED_TRAIN_NORMAL_PER_ID: dict[str, dict[str, int]] = {
    "fan":    {"id_00": 707, "id_02": 711, "id_04": 723, "id_06": 710},
    "pump":   {"id_00": 704, "id_02": 703, "id_04": 491, "id_06": 725},
    "slider": {"id_00": 747, "id_02": 747, "id_04": 373, "id_06": 373},
    "valve":  {"id_00": 693, "id_02": 495, "id_04": 700, "id_06": 694},
}

_TOTAL_POOLED_TRAIN_NORMAL = sum(_EXPECTED_TRAIN_NORMAL.values())  # 10296


# ---------------------------------------------------------------------------
# Reproducibility  (identical to e1_train.py)
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Per-type validation  (mirrors e1_train.py checks, extended to all types)
# ---------------------------------------------------------------------------

def _validate_labels(split: DatasetSplit, machine_type: str) -> None:
    assert all(r.label == "normal" for r in split.train_normal), (
        f"FAIL A [{machine_type}]: train_normal contains non-normal recordings"
    )


def _validate_disjoint_sets(split: DatasetSplit, machine_type: str) -> None:
    train_paths = {r.absolute_path for r in split.train_normal}
    assert not train_paths & {r.absolute_path for r in split.profile_normal}, (
        f"FAIL B [{machine_type}]: train_normal ∩ profile_normal non-empty"
    )
    assert not train_paths & {r.absolute_path for r in split.test_normal}, (
        f"FAIL C [{machine_type}]: train_normal ∩ test_normal non-empty"
    )
    assert not train_paths & {r.absolute_path for r in split.test_abnormal}, (
        f"FAIL D [{machine_type}]: train_normal ∩ test_abnormal non-empty"
    )


def _validate_machine_coverage(split: DatasetSplit, machine_type: str) -> None:
    ids_in_train = {r.machine_id for r in split.train_normal}
    missing = set(MACHINE_IDS) - ids_in_train
    assert not missing, (
        f"FAIL E [{machine_type}]: machine IDs missing from train_normal: {missing}"
    )
    types_in_train = {r.machine_type for r in split.train_normal}
    assert types_in_train == {machine_type}, (
        f"FAIL F [{machine_type}]: unexpected machine types in split: {types_in_train}"
    )


def _validate_min_recordings(split: DatasetSplit, machine_type: str) -> None:
    for mid in MACHINE_IDS:
        count = sum(1 for r in split.train_normal if r.machine_id == mid)
        assert count >= 4, (
            f"FAIL G [{machine_type}/{mid}]: only {count} train_normal recordings "
            "(need ≥4 to guarantee ≥2 internal train and ≥2 internal val)"
        )


def _validate_split_counts(split: DatasetSplit, machine_type: str) -> None:
    """Assert train_normal totals match the verified Phase 9.1 counts."""
    actual_total = len(split.train_normal)
    expected_total = _EXPECTED_TRAIN_NORMAL[machine_type]
    assert actual_total == expected_total, (
        f"FAIL COUNT [{machine_type}]: train_normal={actual_total}, "
        f"expected {expected_total}"
    )
    for mid in MACHINE_IDS:
        actual = sum(1 for r in split.train_normal if r.machine_id == mid)
        expected = _EXPECTED_TRAIN_NORMAL_PER_ID[machine_type][mid]
        assert actual == expected, (
            f"FAIL COUNT [{machine_type}/{mid}]: train_normal={actual}, "
            f"expected {expected}"
        )


def _validate_per_type_split(split: DatasetSplit, machine_type: str) -> None:
    _validate_labels(split, machine_type)
    _validate_disjoint_sets(split, machine_type)
    _validate_machine_coverage(split, machine_type)
    _validate_min_recordings(split, machine_type)
    _validate_split_counts(split, machine_type)


# ---------------------------------------------------------------------------
# Cross-type validation on the pooled train_normal list
# ---------------------------------------------------------------------------

def _validate_pooled(
    pooled_train: list,
    splits: dict[str, DatasetSplit],
) -> None:
    """Assert the pooled list is clean and all four types are present."""
    # All records must be normal
    assert all(r.label == "normal" for r in pooled_train), (
        "FAIL POOL-A: pooled train_normal contains non-normal recordings"
    )

    # All four machine types must be present
    types_present = {r.machine_type for r in pooled_train}
    assert types_present == set(MACHINE_TYPES), (
        f"FAIL POOL-B: expected types {set(MACHINE_TYPES)}, got {types_present}"
    )

    # Total count must match the sum of per-type verified counts
    assert len(pooled_train) == _TOTAL_POOLED_TRAIN_NORMAL, (
        f"FAIL POOL-C: pooled count={len(pooled_train)}, "
        f"expected {_TOTAL_POOLED_TRAIN_NORMAL}"
    )

    # No path may appear in any type's profile_normal, test_normal, or test_abnormal
    pooled_paths = {r.absolute_path for r in pooled_train}
    for mt, split in splits.items():
        held_out = (
            {r.absolute_path for r in split.profile_normal}
            | {r.absolute_path for r in split.test_normal}
            | {r.absolute_path for r in split.test_abnormal}
        )
        overlap = pooled_paths & held_out
        assert not overlap, (
            f"FAIL POOL-D [{mt}]: {len(overlap)} pooled train paths appear "
            "in held-out partitions"
        )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(
    splits: dict[str, DatasetSplit],
    n_internal_train: int,
    n_internal_val: int,
) -> None:
    print("=" * 60)
    print(f"Experiment ID : {EXPERIMENT_ID}")
    print("=" * 60)
    print()
    print("Machine types :", MACHINE_TYPES)
    print("Machine IDs   :", MACHINE_IDS)
    print()
    print(f"{'Type':<8} {'train_n':>8} {'profile_n':>10} {'test_n':>8} {'test_ab':>8}")
    print("-" * 48)
    total_train = total_profile = total_test_n = total_test_ab = 0
    for mt in MACHINE_TYPES:
        s = splits[mt]
        tn  = len(s.train_normal)
        pn  = len(s.profile_normal)
        tsn = len(s.test_normal)
        tab = len(s.test_abnormal)
        print(f"{mt:<8} {tn:>8} {pn:>10} {tsn:>8} {tab:>8}")
        total_train   += tn
        total_profile += pn
        total_test_n  += tsn
        total_test_ab += tab
    print("-" * 48)
    print(f"{'TOTAL':<8} {total_train:>8} {total_profile:>10} {total_test_n:>8} {total_test_ab:>8}")
    print()
    print(f"Pooled train_normal          : {total_train}")
    print(f"Internal train pairs         : {n_internal_train}")
    print(f"Internal validation pairs    : {n_internal_val}")
    print()
    print(f"Seed          : {SEED}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Learning rate : {LEARNING_RATE}")
    print(f"Temperature   : {TEMPERATURE}")
    print()
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _set_seeds(SEED)

    # 1. Load all recordings from the MIMII root
    loader = DatasetLoader(DATASET_ROOT)
    all_recordings = loader.get_all_files()

    # 2. Split each machine type independently with the same splitter parameters
    splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
    splits: dict[str, DatasetSplit] = {}
    for mt in MACHINE_TYPES:
        type_recordings = [r for r in all_recordings if r.machine_type == mt]
        splits[mt] = splitter.split(type_recordings)

    # 3. Per-type validation (labels, disjointness, coverage, counts)
    for mt in MACHINE_TYPES:
        _validate_per_type_split(splits[mt], mt)

    # 4. Pool train_normal from all four types
    pooled_train_normal = [r for mt in MACHINE_TYPES for r in splits[mt].train_normal]

    # 5. Cross-type validation on the pooled list
    _validate_pooled(pooled_train_normal, splits)

    # 6. Build ContrastiveDataset from pooled train_normal ONLY
    dataset = ContrastiveDataset(
        recordings=pooled_train_normal,
        cache_root=CACHE_ROOT,
        seed=SEED,
        val_split=0.20,
    )

    n_internal_train = len(dataset.train_positive_pairs)
    n_internal_val   = len(dataset.val_positive_pairs)

    assert n_internal_train >= 2, (
        f"FAIL G: only {n_internal_train} internal training pairs"
    )
    assert n_internal_val >= 2, (
        f"FAIL H: only {n_internal_val} internal validation pairs"
    )

    _print_summary(splits, n_internal_train, n_internal_val)

    # 7. Construct components  (identical to e1_train.py)
    head      = ProjectionHead(input_dim=INPUT_DIM, output_dim=PROJECTION_DIM)
    criterion = NTXentLoss(temperature=TEMPERATURE)
    trainer   = ContrastiveTrainer(
        head=head,
        criterion=criterion,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
        seed=SEED,
    )

    # 8. Train
    trainer.fit(dataset)

    # 9. Save training history JSON
    hist     = trainer.history()
    best_val = min(hist["validation_losses"]) if hist["validation_losses"] else math.inf

    per_type_counts = {
        mt: {
            "train_normal":   len(splits[mt].train_normal),
            "profile_normal": len(splits[mt].profile_normal),
            "test_normal":    len(splits[mt].test_normal),
            "test_abnormal":  len(splits[mt].test_abnormal),
        }
        for mt in MACHINE_TYPES
    }

    result = {
        "experiment_id": EXPERIMENT_ID,
        "seed": SEED,
        "machine_types": MACHINE_TYPES,
        "machine_ids": MACHINE_IDS,
        "split_configuration": {
            "train_ratio":   TRAIN_RATIO,
            "profile_ratio": PROFILE_RATIO,
            "strategy":      "per_machine_type_independent",
        },
        "per_type_counts": per_type_counts,
        "total_pooled_train_normal": len(pooled_train_normal),
        "training_configuration": {
            "epochs":               EPOCHS,
            "batch_size":           BATCH_SIZE,
            "learning_rate":        LEARNING_RATE,
            "temperature":          TEMPERATURE,
            "input_dimension":      INPUT_DIM,
            "projection_dimension": PROJECTION_DIM,
        },
        "loss_history": {
            "training":   hist["training_losses"],
            "validation": hist["validation_losses"],
        },
        "best_validation_loss": best_val,
        "checkpoint_path": str(CHECKPOINT_DIR / "best_projection_head.pt"),
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history_path = CHECKPOINT_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nTraining history saved to : {history_path}")
    print(f"Best validation loss      : {best_val:.4f}")
    print(f"Checkpoint                : {CHECKPOINT_DIR / 'best_projection_head.pt'}")


if __name__ == "__main__":
    main()
