"""Learned fingerprint profile example.

Builds the healthy learned fingerprint profile for one machine using a trained
ProjectionHead checkpoint, then prints a summary.

Usage:
    python examples/learned_profile_example.py \\
        --root data/raw/MIMII \\
        --machine-type pump \\
        --machine-id id_00 \\
        --checkpoint models/contrastive/best_projection_head.pt

Expected output:
    Machine type       : pump
    Machine ID         : id_00
    Healthy recordings : 100
    Embedding dimension: 256
    Mean shape         : (256,)
    Std shape          : (256,)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.loader import DatasetLoader
from src.learned_profile.builder import LearnedProfileBuilder
from src.learned_profile.serializer import LearnedProfileSerializer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "learned_profiles"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a healthy learned fingerprint profile")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--machine-type", type=str, required=True, help="Machine type (e.g. pump)")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID (e.g. id_00)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ProjectionHead checkpoint")
    parser.add_argument("--max-recordings", type=int, default=100, help="Max healthy recordings to process")
    args = parser.parse_args()

    loader = DatasetLoader(args.root)
    builder = LearnedProfileBuilder(checkpoint_path=args.checkpoint)

    print("Building healthy learned fingerprint profile...")
    profile = builder.build(
        loader=loader,
        machine_type=args.machine_type,
        machine_id=args.machine_id,
        max_recordings=args.max_recordings,
    )

    print(f"Machine type       : {profile.machine_type}")
    print(f"Machine ID         : {profile.machine_id}")
    print(f"Healthy recordings : {len(profile.embeddings)}")
    print(f"Embedding dimension: {profile.embedding_dimension}")
    print(f"Mean shape         : {profile.mean_vector.shape}")
    print(f"Std shape          : {profile.std_vector.shape}")

    serializer = LearnedProfileSerializer()
    out_stem = _OUTPUT_DIR / f"{profile.machine_type}_{profile.machine_id}_learned_profile"
    serializer.save_json(profile, out_stem.with_suffix(".json"))
    serializer.save_npz(profile, out_stem.with_suffix(".npz"))
    print(f"\nProfile saved to   : {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
