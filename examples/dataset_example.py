"""Example: demonstrates DatasetLoader usage against the MIMII dataset.

Usage:
    python examples/dataset_example.py --root data/MIMII
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import DatasetLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main(root: str) -> None:
    loader = DatasetLoader(root)

    # ── Overview ──────────────────────────────────────────────────────
    loader.summary()

    # ── Machine types and IDs ─────────────────────────────────────────
    print("\nMachine types:", loader.get_machine_types())
    print("All machine IDs:", loader.get_machine_ids())

    # ── Filter by machine type ────────────────────────────────────────
    machine_type = loader.get_machine_types()[0]
    type_records = loader.filter_by_machine(machine_type)
    print(f"\n'{machine_type}' recordings: {len(type_records)}")
    print(f"  IDs: {loader.get_machine_ids(machine_type)}")

    # ── Filter by machine ID ──────────────────────────────────────────
    first_id = loader.get_machine_ids()[0]
    id_records = loader.filter_by_machine_id(first_id)
    print(f"\nRecordings for '{first_id}': {len(id_records)}")

    # ── Filter by label ───────────────────────────────────────────────
    normal = loader.filter_by_label("normal")
    abnormal = loader.filter_by_label("abnormal")
    print(f"\nNormal: {len(normal)}  |  Abnormal: {len(abnormal)}")

    # ── Inspect a single record ───────────────────────────────────────
    if normal:
        sample = normal[0]
        print("\nSample record:")
        print(f"  machine_type  : {sample.machine_type}")
        print(f"  machine_id    : {sample.machine_id}")
        print(f"  label         : {sample.label}")
        print(f"  filename      : {sample.filename}")
        print(f"  relative_path : {sample.relative_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DatasetLoader example")
    parser.add_argument(
        "--root",
        default="data/MIMII",
        help="Path to the dataset root directory (default: data/MIMII)",
    )
    args = parser.parse_args()
    main(args.root)
