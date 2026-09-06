"""Phase 19 — BEATs-only ProjectionHead training (controlled ablation).

Trains a NEW 768→256 ProjectionHead on the IDENTICAL pooled multi-machine
normal training data, split, seed, epochs, and hyperparameters as Phase 9.

The only difference from Phase 9 training:
    - input_dim = 768  (BEATs embedding only, no DSP)
    - The fused vector's beats_embedding field is used instead of
      fused_feature_vector.

This resolves the E1 ablation confound where BEATs-only used a single-machine
head while the full method used a multi-machine head.

Checkpoint saved to: models/contrastive/phase19/best_projection_head_beats_only.pt

Usage:
    python experiments/phase19_beats_only_train.py
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contrastive_learning.dataset import ContrastiveDataset, ContrastivePair
from src.contrastive_learning.loss import NTXentLoss
from src.contrastive_learning.serializer import ContrastiveSerializer
from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplitter
from src.fusion.fused_vector import FusedFeatureVector

# ---------------------------------------------------------------------------
# Phase 19 constants — identical to phase9_train.py except INPUT_DIM
# ---------------------------------------------------------------------------

EXPERIMENT_ID   = "phase19"
DATASET_ROOT    = Path("data/raw/MIMII")
CACHE_ROOT      = Path("data/fusion_cache")

MACHINE_TYPES   = ["fan", "pump", "slider", "valve"]
MACHINE_IDS     = ["id_00", "id_02", "id_04", "id_06"]

TRAIN_RATIO     = 0.70
PROFILE_RATIO   = 0.15
SEED            = 42

EPOCHS          = 20
BATCH_SIZE      = 16
LEARNING_RATE   = 0.001
TEMPERATURE     = 0.07
INPUT_DIM       = 768   # BEATs-only (vs 921 for full method)
PROJECTION_DIM  = 256

CHECKPOINT_DIR  = Path("models/contrastive/phase19")
CHECKPOINT_NAME = "best_projection_head_beats_only.pt"

# Same verified totals as Phase 9 (same split, same data)
_EXPECTED_TRAIN_NORMAL: dict[str, int] = {
    "fan":    2851,
    "pump":   2623,
    "slider": 2240,
    "valve":  2582,
}
_TOTAL_POOLED = sum(_EXPECTED_TRAIN_NORMAL.values())  # 10296


# ---------------------------------------------------------------------------
# BEATs-only ProjectionHead  (768 → 512 → 256, L2-norm)
# ---------------------------------------------------------------------------

class BeatsOnlyProjectionHead(nn.Module):
    """768→256 projection head for BEATs-only contrastive learning.

    Architecture mirrors Phase 9 ProjectionHead exactly except input_dim=768.
    Linear(768→512) → ReLU → Linear(512→256) → L2-norm
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, PROJECTION_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


# ---------------------------------------------------------------------------
# BEATs-only pair wrapper
#
# ContrastiveDataset encodes via FusionCache → 921-dim fused vectors.
# We reuse that cache (no re-encoding) and simply swap each pair's
# fused_feature_vector for the 768-dim beats_embedding slice.
# ---------------------------------------------------------------------------

def _beats_only_pairs(pairs: list[ContrastivePair]) -> list[ContrastivePair]:
    """Return a copy of *pairs* with fused_feature_vector replaced by beats_embedding."""
    result = []
    for p in pairs:
        anchor = _beats_fused(p.anchor)
        paired = _beats_fused(p.paired)
        result.append(ContrastivePair(anchor=anchor, paired=paired, label=p.label))
    return result


def _beats_fused(fv: FusedFeatureVector) -> FusedFeatureVector:
    """Return a shallow copy of *fv* with fused_feature_vector = beats_embedding."""
    clone = copy.copy(fv)
    # dataclass is not frozen, so direct assignment works
    object.__setattr__(clone, "fused_feature_vector", fv.beats_embedding.copy())
    return clone


# ---------------------------------------------------------------------------
# Trainer (mirrors ContrastiveTrainer but uses BeatsOnlyProjectionHead)
# ---------------------------------------------------------------------------

class BeatsOnlyTrainer:
    """Trains BeatsOnlyProjectionHead with NT-Xent loss.

    Identical training loop to ContrastiveTrainer; machine-aware batching
    is preserved.
    """

    def __init__(
        self,
        head: BeatsOnlyProjectionHead,
        criterion: NTXentLoss,
    ) -> None:
        self._head = head
        self._criterion = criterion
        self._optimizer = optim.Adam(head.parameters(), lr=LEARNING_RATE)
        self._rng = random.Random(SEED)
        self._best_val_loss = math.inf
        self._history: list[dict] = []
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def fit(self, train_pairs: list[ContrastivePair], val_pairs: list[ContrastivePair]) -> None:
        for epoch in range(1, EPOCHS + 1):
            train_loss = self._run_epoch(train_pairs, training=True)
            val_loss   = self._run_epoch(val_pairs,   training=False)
            self._history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
            print(f"Epoch {epoch}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                ContrastiveSerializer.save_checkpoint(
                    path=CHECKPOINT_DIR / CHECKPOINT_NAME,
                    model_state_dict=self._head.state_dict(),
                    optimizer_state_dict=self._optimizer.state_dict(),
                    epoch=epoch,
                    validation_loss=val_loss,
                    config={"input_dim": INPUT_DIM, "projection_dim": PROJECTION_DIM},
                )
                print(f"  ✓ checkpoint saved (val={val_loss:.4f})")

    def history(self) -> list[dict]:
        return self._history

    def _run_epoch(self, pairs: list[ContrastivePair], *, training: bool) -> float:
        self._head.train(training)
        shuffled = list(pairs)
        if training:
            self._rng.shuffle(shuffled)

        total, n_batches = 0.0, 0
        with torch.set_grad_enabled(training):
            for batch in self._make_batches(shuffled):
                anchors = torch.from_numpy(
                    np.stack([p.anchor.fused_feature_vector for p in batch])
                ).float()
                paired = torch.from_numpy(
                    np.stack([p.paired.fused_feature_vector for p in batch])
                ).float()
                if training:
                    self._optimizer.zero_grad()
                emb_a = self._head(anchors)
                emb_b = self._head(paired)
                loss  = self._criterion(emb_a, emb_b)
                if training:
                    loss.backward()
                    self._optimizer.step()
                total    += loss.item()
                n_batches += 1
        return total / n_batches if n_batches else 0.0

    def _make_batches(self, pairs: list[ContrastivePair]) -> list[list[ContrastivePair]]:
        """Machine-aware batching: at most one pair per (machine_type, machine_id)."""
        groups: dict[tuple[str, str], list[ContrastivePair]] = {}
        for p in pairs:
            key = (p.anchor.machine_type, p.anchor.machine_id)
            groups.setdefault(key, []).append(p)

        queues  = [list(v) for v in groups.values()]
        batches: list[list[ContrastivePair]] = []
        while any(queues):
            batch: list[ContrastivePair] = []
            for q in queues:
                if q and len(batch) < BATCH_SIZE:
                    batch.append(q.pop(0))
            if len(batch) >= 2:
                batches.append(batch)
        return batches


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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _set_seeds(SEED)

    print("=" * 60)
    print(f"Experiment      : {EXPERIMENT_ID}")
    print(f"Condition       : BEATs-only  ({INPUT_DIM}→{PROJECTION_DIM})")
    print(f"Seed / Epochs   : {SEED} / {EPOCHS}")
    print(f"Batch / LR / T  : {BATCH_SIZE} / {LEARNING_RATE} / {TEMPERATURE}")
    print("=" * 60)

    # 1. Load all recordings
    loader = DatasetLoader(DATASET_ROOT)
    all_recordings = loader.get_all_files()

    # 2. Reproduce the identical Phase 9 split
    splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
    splits = {}
    for mt in MACHINE_TYPES:
        type_recs = [r for r in all_recordings if r.machine_type == mt]
        splits[mt] = splitter.split(type_recs)

    # 3. Pool train_normal — identical 10 296 recordings as Phase 9
    pooled_train = [r for mt in MACHINE_TYPES for r in splits[mt].train_normal]
    assert len(pooled_train) == _TOTAL_POOLED, (
        f"Pooled count {len(pooled_train)} != expected {_TOTAL_POOLED}"
    )
    for mt in MACHINE_TYPES:
        actual = len(splits[mt].train_normal)
        assert actual == _EXPECTED_TRAIN_NORMAL[mt], (
            f"{mt}: train_normal={actual}, expected {_EXPECTED_TRAIN_NORMAL[mt]}"
        )

    print(f"\nPooled train_normal : {len(pooled_train)} recordings")
    print(f"{'Type':<8} {'train_n':>8} {'profile_n':>10} {'test_n':>8} {'test_ab':>8}")
    print("-" * 48)
    for mt in MACHINE_TYPES:
        s = splits[mt]
        print(f"{mt:<8} {len(s.train_normal):>8} {len(s.profile_normal):>10} "
              f"{len(s.test_normal):>8} {len(s.test_abnormal):>8}")
    print()

    # 4. Build ContrastiveDataset (uses FusionCache — no re-encoding if cache exists)
    print("Building ContrastiveDataset (reuses fusion cache)...")
    dataset = ContrastiveDataset(
        recordings=pooled_train,
        cache_root=CACHE_ROOT,
        seed=SEED,
        val_split=0.20,
    )
    print(f"Train pairs : {len(dataset.train_positive_pairs)}")
    print(f"Val pairs   : {len(dataset.val_positive_pairs)}")

    # 5. Swap fused_feature_vector → beats_embedding in every pair
    print("\nSlicing BEATs embeddings from fused vectors...")
    train_pairs = _beats_only_pairs(dataset.train_positive_pairs)
    val_pairs   = _beats_only_pairs(dataset.val_positive_pairs)

    # Sanity-check: every vector must be 768-dim
    sample = train_pairs[0].anchor.fused_feature_vector
    assert sample.shape == (768,), f"Expected (768,), got {sample.shape}"
    print(f"Pair vector dimension confirmed: {sample.shape[0]}")

    # 6. Train
    print("\nTraining BEATs-only ProjectionHead...")
    head      = BeatsOnlyProjectionHead()
    criterion = NTXentLoss(temperature=TEMPERATURE)
    trainer   = BeatsOnlyTrainer(head, criterion)
    trainer.fit(train_pairs, val_pairs)

    # 7. Save training history
    hist = trainer.history()
    best_val = min(r["val"] for r in hist) if hist else math.inf

    result = {
        "experiment_id":  EXPERIMENT_ID,
        "condition":      "beats_only",
        "seed":           SEED,
        "machine_types":  MACHINE_TYPES,
        "machine_ids":    MACHINE_IDS,
        "split_configuration": {
            "train_ratio":   TRAIN_RATIO,
            "profile_ratio": PROFILE_RATIO,
            "strategy":      "per_machine_type_independent",
        },
        "total_pooled_train_normal": len(pooled_train),
        "training_configuration": {
            "epochs":               EPOCHS,
            "batch_size":           BATCH_SIZE,
            "learning_rate":        LEARNING_RATE,
            "temperature":          TEMPERATURE,
            "input_dimension":      INPUT_DIM,
            "projection_dimension": PROJECTION_DIM,
        },
        "loss_history": hist,
        "best_validation_loss": best_val,
        "checkpoint_path": str(CHECKPOINT_DIR / CHECKPOINT_NAME),
    }

    history_path = CHECKPOINT_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nBest validation loss : {best_val:.4f}")
    print(f"Checkpoint           : {CHECKPOINT_DIR / CHECKPOINT_NAME}")
    print(f"Training history     : {history_path}")


if __name__ == "__main__":
    main()
