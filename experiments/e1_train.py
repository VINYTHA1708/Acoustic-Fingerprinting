"""Experiment E1 — Real Contrastive Training on MIMII pump recordings.

Usage:
    python experiments/e1_train.py
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
from src.dataset.split import DatasetSplitter

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

EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001
TEMPERATURE = 0.07
INPUT_DIM = 921
PROJECTION_DIM = 256

CHECKPOINT_DIR = Path("models/contrastive/e1")
RESULTS_DIR = Path("experiments/results/e1")
CACHE_ROOT = Path("data/fusion_cache")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_labels(split) -> None:
    assert all(r.label == "normal" for r in split.train_normal), \
        "FAIL A: train_normal contains non-normal recordings"


def _validate_disjoint_sets(split) -> None:
    train_paths = {r.absolute_path for r in split.train_normal}
    assert not train_paths & {r.absolute_path for r in split.profile_normal}, \
        "FAIL B: train_normal ∩ profile_normal non-empty"
    assert not train_paths & {r.absolute_path for r in split.test_normal}, \
        "FAIL C: train_normal ∩ test_normal non-empty"
    assert not train_paths & {r.absolute_path for r in split.test_abnormal}, \
        "FAIL D: train_normal ∩ test_abnormal non-empty"


def _validate_machine_coverage(split) -> None:
    ids_in_train = {r.machine_id for r in split.train_normal}
    missing = set(MACHINE_IDS) - ids_in_train
    assert not missing, f"FAIL E: machine IDs missing from train_normal: {missing}"

    types_in_train = {r.machine_type for r in split.train_normal}
    assert types_in_train == {"pump"}, f"FAIL F: unexpected machine types: {types_in_train}"


def _validate_min_recordings(split) -> None:
    for mid in MACHINE_IDS:
        count = sum(1 for r in split.train_normal if r.machine_id == mid)
        assert count >= 4, (
            f"FAIL G/H: machine {mid} has only {count} train_normal recordings "
            "(need ≥4 to guarantee ≥2 internal train and ≥2 internal val)"
        )


def _validate_split(split, train_normal_ids: set[str]) -> None:
    """Pre-training data leakage and integrity checks."""
    _validate_labels(split)
    _validate_disjoint_sets(split)
    _validate_machine_coverage(split)
    _validate_min_recordings(split)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(split, n_internal_train: int, n_internal_val: int) -> None:
    print("=" * 50)
    print(f"Experiment ID : {EXPERIMENT_ID}")
    print("=" * 50)
    print()
    print("Machine type:")
    print(MACHINE_TYPE)
    print()
    print("Machine IDs:")
    for mid in MACHINE_IDS:
        print(mid)
    print()
    print("Dataset split:")
    print(f"Train normal: {len(split.train_normal)}")
    print(f"Profile normal: {len(split.profile_normal)}")
    print(f"Test normal: {len(split.test_normal)}")
    print(f"Test abnormal: {len(split.test_abnormal)}")
    print()
    print("Contrastive training:")
    print()
    print("Internal train recordings:")
    print(n_internal_train)
    print()
    print("Internal validation recordings:")
    print(n_internal_val)
    print()
    print("Seed:")
    print(SEED)
    print()
    print("Epochs:")
    print(EPOCHS)
    print()
    print("Batch size:")
    print(BATCH_SIZE)
    print()
    print("Learning rate:")
    print(LEARNING_RATE)
    print()
    print("Temperature:")
    print(TEMPERATURE)
    print()
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _set_seeds(SEED)

    # 1. Load recordings, restrict to pump + E1 machine IDs
    loader = DatasetLoader(DATASET_ROOT)
    all_recordings = [
        r for r in loader.get_all_files()
        if r.machine_type == MACHINE_TYPE and r.machine_id in MACHINE_IDS
    ]

    # 2. E1 split
    splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
    split = splitter.split(all_recordings)

    # 3. Pre-training validation
    _validate_split(split, set(MACHINE_IDS))

    # 4. Build ContrastiveDataset from train_normal ONLY
    dataset = ContrastiveDataset(
        recordings=split.train_normal,
        cache_root=CACHE_ROOT,
        seed=SEED,
        val_split=0.20,
    )

    n_internal_train = len(dataset.train_positive_pairs)
    n_internal_val = len(dataset.val_positive_pairs)

    # Validate internal split sizes (G/H)
    assert n_internal_train >= 2, f"FAIL G: only {n_internal_train} internal training pairs"
    assert n_internal_val >= 2, f"FAIL H: only {n_internal_val} internal validation pairs"

    _print_summary(split, n_internal_train, n_internal_val)

    # 5. Construct components
    head = ProjectionHead(input_dim=INPUT_DIM, output_dim=PROJECTION_DIM)
    criterion = NTXentLoss(temperature=TEMPERATURE)
    trainer = ContrastiveTrainer(
        head=head,
        criterion=criterion,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
        seed=SEED,
    )

    # 6. Train
    trainer.fit(dataset)

    # 7. Save training history
    hist = trainer.history()
    best_val = min(hist["validation_losses"]) if hist["validation_losses"] else math.inf

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "seed": SEED,
        "machine_type": MACHINE_TYPE,
        "machine_ids": MACHINE_IDS,
        "dataset_counts": {
            "train_normal": len(split.train_normal),
            "profile_normal": len(split.profile_normal),
            "test_normal": len(split.test_normal),
            "test_abnormal": len(split.test_abnormal),
        },
        "training_configuration": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "temperature": TEMPERATURE,
            "input_dimension": INPUT_DIM,
            "projection_dimension": PROJECTION_DIM,
        },
        "loss_history": {
            "training": hist["training_losses"],
            "validation": hist["validation_losses"],
        },
        "best_validation_loss": best_val,
        "checkpoint_path": str(CHECKPOINT_DIR / "best_projection_head.pt"),
    }

    history_path = RESULTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nTraining history saved to: {history_path}")
    print(f"Best validation loss     : {best_val:.4f}")
    print(f"Checkpoint               : {CHECKPOINT_DIR / 'best_projection_head.pt'}")


if __name__ == "__main__":
    main()
