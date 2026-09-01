"""Experiment E1 — Main Method Evaluation.

Evaluates the trained contrastive acoustic fingerprinting method using the
identical dataset split, machine IDs, and evaluation protocol as Phase 7.2
(e1_baseline_evaluation.py).

For each machine ID (id_00, id_02, id_04, id_06):
    1. Build a per-machine healthy profile from split.profile_normal using
       LearnedProfileBuilder with the E1 trained checkpoint.
    2. Score every test recording (test_normal ∪ test_abnormal) by Euclidean
       distance between its 256-dim contrastive embedding and the profile mean.
    3. Compute AUROC and separation ratio using the same metric functions as
       the baseline evaluation.

Results are saved to:
    experiments/results/e1/baseline_comparison/main_method_results.csv

The CSV schema is identical to baseline_results.csv so Phase 7.4 can join
both files directly.

Usage:
    python experiments/e1_main_method_evaluation.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.e1_baseline_definition import PROTOCOL
from experiments.e1_baseline_evaluation import compute_auroc, compute_separation_ratio
from src.contrastive_learning.inference import ContrastiveInference
from src.contrastive_learning.model import ProjectionHead
from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplitter
from src.learned_profile.builder import LearnedProfileBuilder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("data/raw/MIMII")
BEATS_CHECKPOINT = Path("models/beats/BEATs_iter3_plus_AS2M.pt")
CONTRASTIVE_CHECKPOINT = Path("models/contrastive/e1/best_projection_head.pt")
RESULTS_PATH = Path("experiments/results/e1/baseline_comparison/main_method_results.csv")

METHOD_ID = "contrastive_main"
METHOD_NAME = "Contrastive Acoustic Fingerprinting"

CSV_COLUMNS = [
    "baseline_id",
    "baseline_name",
    "machine_id",
    "n_normal",
    "n_abnormal",
    "auroc",
    "separation_ratio",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_inputs() -> None:
    """Raise FileNotFoundError if required resources are missing."""
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"MIMII dataset not found: {DATASET_ROOT}")
    if not BEATS_CHECKPOINT.exists():
        raise FileNotFoundError(f"BEATs checkpoint not found: {BEATS_CHECKPOINT}")
    if not CONTRASTIVE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Contrastive checkpoint not found: {CONTRASTIVE_CHECKPOINT}\n"
            "Run experiments/e1_train.py first."
        )


def validate_results(rows: list[dict]) -> None:
    """Raise ValueError if any result row has invalid metric values."""
    for row in rows:
        auroc = row["auroc"]
        sep = row["separation_ratio"]
        if not (isinstance(auroc, float) and (0.0 <= auroc <= 1.0 or auroc != auroc)):
            raise ValueError(
                f"Invalid AUROC={auroc} for {row['baseline_id']}/{row['machine_id']}"
            )
        if not (isinstance(sep, float) and (sep >= 0.0 or sep != sep)):
            raise ValueError(
                f"Invalid separation_ratio={sep} for {row['baseline_id']}/{row['machine_id']}"
            )


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _contrastive_embedding(record, inference: ContrastiveInference, cache) -> np.ndarray:
    """Return the 256-dim L2-normalised contrastive embedding for one recording."""
    fused = cache.load_or_create(record)
    return inference.generate_fingerprint(fused)


# ---------------------------------------------------------------------------
# Per-machine evaluation
# ---------------------------------------------------------------------------

def evaluate_machine_id(
    machine_id: str,
    profile_records,
    test_records,
    inference: ContrastiveInference,
    cache,
) -> dict:
    """Evaluate the main method on one machine ID.

    Args:
        machine_id: Machine identifier (e.g. ``"id_00"``).
        profile_records: Normal recordings used to build the healthy profile mean.
        test_records: List of ``(AudioMetadata, label)`` tuples for scoring.
        inference: Loaded :class:`ContrastiveInference` instance.
        cache: :class:`FusionCache` for fused vector retrieval.

    Returns:
        Result dict with keys matching ``CSV_COLUMNS``.
    """
    # Build profile mean from profile_normal embeddings
    profile_vecs = [
        _contrastive_embedding(rec, inference, cache)
        for rec in profile_records
    ]
    profile_mean = np.mean(profile_vecs, axis=0).astype(np.float32)

    # Score every test recording
    scores, labels = [], []
    for rec, label in test_records:
        emb = _contrastive_embedding(rec, inference, cache)
        dist = float(np.linalg.norm(emb - profile_mean))
        scores.append(dist)
        labels.append(1 if label == "abnormal" else 0)

    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int32)

    normal_scores = scores_arr[labels_arr == 0]
    abnormal_scores = scores_arr[labels_arr == 1]

    auroc = compute_auroc(scores_arr, labels_arr)
    sep = compute_separation_ratio(normal_scores, abnormal_scores)

    return {
        "baseline_id": METHOD_ID,
        "baseline_name": METHOD_NAME,
        "machine_id": machine_id,
        "n_normal": int((labels_arr == 0).sum()),
        "n_abnormal": int((labels_arr == 1).sum()),
        "auroc": round(auroc, 4),
        "separation_ratio": round(sep, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    validate_inputs()

    proto = PROTOCOL
    loader = DatasetLoader(DATASET_ROOT)
    all_recordings = [
        r for r in loader.get_all_files()
        if r.machine_type == proto.machine_type
        and r.machine_id in proto.machine_ids
    ]

    splitter = DatasetSplitter(
        train_ratio=proto.train_ratio,
        profile_ratio=proto.profile_ratio,
        seed=proto.seed,
    )
    split = splitter.split(all_recordings)

    # Build shared inference components (BEATs + FusionCache inside builder)
    builder = LearnedProfileBuilder(checkpoint_path=CONTRASTIVE_CHECKPOINT)
    # Reuse the builder's internal cache and inference for scoring
    cache = builder._cache
    inference = builder._inference

    rows: list[dict] = []

    print("=" * 60)
    print("Experiment E1 — Main Method Evaluation")
    print("=" * 60)
    print(f"Method       : {METHOD_NAME}")
    print(f"Checkpoint   : {CONTRASTIVE_CHECKPOINT}")
    print(f"Machine type : {proto.machine_type}")
    print(f"Machine IDs  : {list(proto.machine_ids)}")
    print()

    for machine_id in proto.machine_ids:
        profile_records = [
            r for r in split.profile_normal if r.machine_id == machine_id
        ]
        test_records = (
            [(r, "normal") for r in split.test_normal if r.machine_id == machine_id]
            + [(r, "abnormal") for r in split.test_abnormal if r.machine_id == machine_id]
        )

        print(
            f"  {machine_id} — profile={len(profile_records)}  "
            f"test_normal={sum(1 for _, l in test_records if l == 'normal')}  "
            f"test_abnormal={sum(1 for _, l in test_records if l == 'abnormal')}",
            end="  ... ",
            flush=True,
        )

        row = evaluate_machine_id(
            machine_id=machine_id,
            profile_records=profile_records,
            test_records=test_records,
            inference=inference,
            cache=cache,
        )
        rows.append(row)
        print(f"AUROC={row['auroc']:.4f}  sep={row['separation_ratio']:.4f}")

    validate_results(rows)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print("Results saved to:", RESULTS_PATH)
    print("=" * 60)
    print()
    print(f"{'Machine':<8} {'N_norm':>7} {'N_abn':>6} {'AUROC':>7} {'Sep':>7}")
    print("-" * 40)
    for row in rows:
        print(
            f"{row['machine_id']:<8} "
            f"{row['n_normal']:>7} {row['n_abnormal']:>6} "
            f"{row['auroc']:>7.4f} {row['separation_ratio']:>7.4f}"
        )


if __name__ == "__main__":
    main()
