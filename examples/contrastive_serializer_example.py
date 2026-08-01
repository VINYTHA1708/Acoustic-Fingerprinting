"""Contrastive serializer example.

Instantiates a ProjectionHead, saves a checkpoint via ContrastiveSerializer,
loads it back, and verifies every required field is present and correct.

Usage:
    python examples/contrastive_serializer_example.py
    python examples/contrastive_serializer_example.py --checkpoint /tmp/test_ckpt.pt
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contrastive_learning.model import ProjectionHead
from src.contrastive_learning.serializer import ContrastiveSerializer

_DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent / "models" / "contrastive" / "serializer_test.pt"
)


def _check(label: str, condition: bool) -> None:
    status = "[PASS]" if condition else "[FAIL]"
    print(f"{status} {label}")
    if not condition:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive serializer example")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_DEFAULT_CHECKPOINT),
        help="Path to write/read the test checkpoint",
    )
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)

    # --- Build a ProjectionHead and capture its initial state ---
    head = ProjectionHead()
    original_state = {k: v.clone() for k, v in head.state_dict().items()}

    test_epoch = 3
    test_val_loss = 1.2345
    test_config = {"temperature": 0.1, "batch_size": 32, "learning_rate": 1e-3}

    # --- Save ---
    print(f"Saving checkpoint to: {checkpoint_path}\n")
    ContrastiveSerializer.save_checkpoint(
        path=checkpoint_path,
        model_state_dict=head.state_dict(),
        epoch=test_epoch,
        validation_loss=test_val_loss,
        optimizer_state_dict=None,   # omitted intentionally to test optional field
        config=test_config,
    )

    # --- Load ---
    ckpt = ContrastiveSerializer.load_checkpoint(checkpoint_path)

    # --- Verify required fields ---
    _check(
        "model_state_dict present and non-empty",
        isinstance(ckpt["model_state_dict"], dict) and len(ckpt["model_state_dict"]) > 0,
    )
    _check(
        "epoch matches",
        ckpt["epoch"] == test_epoch,
    )
    _check(
        "validation_loss matches",
        abs(ckpt["validation_loss"] - test_val_loss) < 1e-6,
    )
    _check(
        "config matches",
        ckpt["config"] == test_config,
    )
    _check(
        "optimizer_state_dict is None when omitted",
        ckpt["optimizer_state_dict"] is None,
    )

    # --- Verify weights round-trip correctly ---
    head2 = ProjectionHead()
    head2.load_state_dict(ckpt["model_state_dict"])
    weights_match = all(
        (head2.state_dict()[k] == original_state[k]).all().item()
        for k in original_state
    )
    _check("model weights round-trip correctly", weights_match)

    print("\nAll checks passed. ContrastiveSerializer is working correctly.")


if __name__ == "__main__":
    main()
