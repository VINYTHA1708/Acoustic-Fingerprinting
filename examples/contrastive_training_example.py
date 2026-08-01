"""Contrastive training example.

Builds a ContrastiveDataset, trains a ProjectionHead with NTXentLoss for two
epochs, and prints the training history.

Usage:
    python examples/contrastive_training_example.py --root data/raw/MIMII
    python examples/contrastive_training_example.py \\
        --root data/raw/MIMII --machine-type pump --max-recordings 200 --epochs 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contrastive_learning.dataset import ContrastiveDataset
from src.contrastive_learning.loss import NTXentLoss
from src.contrastive_learning.model import ProjectionHead
from src.contrastive_learning.trainer import ContrastiveTrainer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "models" / "contrastive"
_CACHE_ROOT = Path(__file__).resolve().parent.parent / "data" / "fusion_cache"


def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive training example")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--machine-type", type=str, default="pump")
    parser.add_argument("--machine-id", type=str, default=None)
    parser.add_argument("--max-recordings", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    # --- Build dataset ---
    print("Building contrastive dataset...")
    dataset = ContrastiveDataset(
        dataset_root=args.root,
        cache_root=_CACHE_ROOT,
        machine_type=args.machine_type,
        machine_id=args.machine_id,
        max_recordings=args.max_recordings,
    )
    print(f"Positive pairs : {len(dataset.positive_pairs)}")
    print(f"Negative pairs : {len(dataset.negative_pairs)}")

    # --- Instantiate components ---
    head = ProjectionHead()
    criterion = NTXentLoss(temperature=args.temperature)
    trainer = ContrastiveTrainer(
        head=head,
        criterion=criterion,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_dir=_CHECKPOINT_DIR,
    )

    # --- Train ---
    print(f"\nTraining for {args.epochs} epoch(s)...\n")
    trainer.fit(dataset)

    # --- Print history ---
    hist = trainer.history()
    print("\nTraining history")
    for i, (tr, vl) in enumerate(
        zip(hist["training_losses"], hist["validation_losses"]), start=1
    ):
        print(f"  Epoch {i} | Training loss : {tr:.4f} | Validation loss : {vl:.4f}")

    print(f"\nBest checkpoint saved to : {_CHECKPOINT_DIR / 'best_projection_head.pt'}")


if __name__ == "__main__":
    main()
