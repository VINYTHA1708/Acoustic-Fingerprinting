"""Contrastive dataset example.

Loads a MIMII-style dataset, encodes all normal recordings, builds positive
and negative pairs, and prints a summary.

Usage:
    python examples/contrastive_dataset_example.py --root data/raw/MIMII
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contrastive_learning import ContrastiveDataset

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive dataset example")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    args = parser.parse_args()

    print("Building contrastive dataset — encoding all normal recordings...")
    dataset = ContrastiveDataset(dataset_root=args.root)

    print(f"\nMachine types          : {dataset.machine_types()}")
    print(f"Machine IDs            : {dataset.machine_ids()}")
    print(f"Normal recordings      : {dataset.normal_recording_count()}")
    print(f"Positive pairs         : {len(dataset.positive_pairs)}")
    print(f"Negative pairs         : {len(dataset.negative_pairs)}")

    print("\n--- Example pairs ---")
    examples = (
        dataset.positive_pairs[:2]
        + dataset.negative_pairs[:1]
    )
    for i, pair in enumerate(examples):
        kind = "positive" if pair.label == 1 else "negative"
        print(
            f"\nPair {i + 1} [{kind}]"
            f"\n  anchor : {pair.anchor.machine_type}/{pair.anchor.machine_id}  {pair.anchor.filename}"
            f"\n  paired : {pair.paired.machine_type}/{pair.paired.machine_id}  {pair.paired.filename}"
            f"\n  label  : {pair.label}"
        )


if __name__ == "__main__":
    main()
