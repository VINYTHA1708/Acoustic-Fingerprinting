"""Experiment E1 — Baseline Evaluation.

Evaluates all three baselines (B1, B2, B3) defined in e1_baseline_definition.py
using the same MIMII pump dataset split and evaluation protocol.

For each baseline × machine ID:
    - Build a per-machine profile mean from split.profile_normal
    - Score every test recording (test_normal ∪ test_abnormal) by Euclidean
      distance to the profile mean
    - Compute AUROC and separation ratio

Results are saved to:
    experiments/results/e1/baseline_comparison/baseline_results.csv

Usage:
    python experiments/e1_baseline_evaluation.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.e1_baseline_definition import PROTOCOL, get_all_baselines
from src.beats.encoder import BEATsEncoder
from src.contrastive_learning.model import ProjectionHead
from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplitter
from src.feature_extraction.extractor import FeatureExtractor
from src.feature_extraction.feature_vector import FeatureVectorBuilder
from src.fusion.cache import FusionCache
from src.fusion.fusion import FusionBuilder
from src.preprocessing.pipeline import PreprocessingPipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("data/raw/MIMII")
BEATS_CHECKPOINT = Path("models/beats/BEATs_iter3_plus_AS2M.pt")
CACHE_ROOT = Path("data/fusion_cache")
RESULTS_PATH = Path("experiments/results/e1/baseline_comparison/baseline_results.csv")

CSV_COLUMNS = [
    "baseline_id",
    "baseline_name",
    "machine_id",
    "n_normal",
    "n_abnormal",
    "auroc",
    "separation_ratio",
]

# Stat feature keys used by B2
_STAT_KEYS = ("rms_mean", "zcr_mean", "spectral_centroid_mean")


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _build_pipeline_components():
    """Construct shared preprocessing and feature extraction components."""
    pipeline = PreprocessingPipeline(target_sr=16_000)
    extractor = FeatureExtractor(sample_rate=16_000)
    vec_builder = FeatureVectorBuilder()
    encoder = BEATsEncoder(BEATS_CHECKPOINT)
    fusion = FusionBuilder()
    cache = FusionCache(
        cache_root=CACHE_ROOT,
        pipeline=pipeline,
        extractor=extractor,
        vec_builder=vec_builder,
        encoder=encoder,
        fusion=fusion,
    )
    return pipeline, extractor, vec_builder, cache


def _dsp_vector(record, pipeline, extractor, vec_builder) -> np.ndarray:
    """Return the full 153-dim DSP feature vector for a recording."""
    result = pipeline.run(record.absolute_path)
    features = extractor.extract(result["waveform"])
    vec, _ = vec_builder.build(features)
    return vec


def _stat_vector(record, pipeline, extractor) -> np.ndarray:
    """Return the 3-dim [rms_mean, zcr_mean, spectral_centroid_mean] vector."""
    result = pipeline.run(record.absolute_path)
    features = extractor.extract(result["waveform"])
    return np.array([features[k] for k in _STAT_KEYS], dtype=np.float32)


def _random_projection_vector(record, cache) -> np.ndarray:
    """Return a 256-dim random-projection embedding (untrained ProjectionHead)."""
    import torch

    fused = cache.load_or_create(record)
    head = ProjectionHead()
    head.eval()
    with torch.no_grad():
        x = torch.from_numpy(fused.fused_feature_vector).float()
        emb = head(x)
    return emb.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC from anomaly scores and binary labels (1 = abnormal).

    Uses a threshold-free trapezoidal integration over all unique thresholds.

    Args:
        scores: 1-D array of anomaly scores (higher = more anomalous).
        labels: 1-D binary array (0 = normal, 1 = abnormal).

    Returns:
        AUROC in [0, 1].
    """
    thresholds = np.unique(scores)[::-1]
    tprs = []
    fprs = []
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)

    # Add (0,0) and (1,1) boundary points
    fprs = [0.0] + fprs + [1.0]
    tprs = [0.0] + tprs + [1.0]
    return float(np.trapezoid(tprs, fprs))


def compute_separation_ratio(
    normal_scores: np.ndarray, abnormal_scores: np.ndarray
) -> float:
    """Compute mean_abnormal / mean_normal distance ratio.

    Args:
        normal_scores: Anomaly scores for normal recordings.
        abnormal_scores: Anomaly scores for abnormal recordings.

    Returns:
        Separation ratio ≥ 0. Values > 1 indicate abnormal recordings score
        higher than normal ones on average.
    """
    mean_normal = float(normal_scores.mean()) if len(normal_scores) > 0 else 0.0
    mean_abnormal = float(abnormal_scores.mean()) if len(abnormal_scores) > 0 else 0.0
    if mean_normal == 0.0:
        return float("nan")
    return mean_abnormal / mean_normal


# ---------------------------------------------------------------------------
# Per-baseline evaluation
# ---------------------------------------------------------------------------

def _evaluate_machine_id(
    baseline_id: str,
    machine_id: str,
    profile_records,
    test_records,
    pipeline,
    extractor,
    vec_builder,
    cache,
) -> dict:
    """Evaluate one baseline on one machine ID.

    Returns a result dict with keys matching CSV_COLUMNS.
    """
    # --- Build profile mean ---
    profile_vecs = []
    for rec in profile_records:
        if baseline_id == "B1_mfcc_distance":
            v = _dsp_vector(rec, pipeline, extractor, vec_builder)
        elif baseline_id == "B2_stat_distance":
            v = _stat_vector(rec, pipeline, extractor)
        else:  # B3_random_projection
            v = _random_projection_vector(rec, cache)
        profile_vecs.append(v)

    profile_mean = np.mean(profile_vecs, axis=0).astype(np.float32)

    # --- Score test recordings ---
    scores = []
    labels = []
    for rec, label in test_records:
        if baseline_id == "B1_mfcc_distance":
            v = _dsp_vector(rec, pipeline, extractor, vec_builder)
        elif baseline_id == "B2_stat_distance":
            v = _stat_vector(rec, pipeline, extractor)
        else:
            v = _random_projection_vector(rec, cache)

        dist = float(np.linalg.norm(v - profile_mean))
        scores.append(dist)
        labels.append(1 if label == "abnormal" else 0)

    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int32)

    normal_scores = scores_arr[labels_arr == 0]
    abnormal_scores = scores_arr[labels_arr == 1]

    auroc = compute_auroc(scores_arr, labels_arr)
    sep = compute_separation_ratio(normal_scores, abnormal_scores)

    return {
        "baseline_id": baseline_id,
        "baseline_name": next(
            b.name for b in get_all_baselines() if b.baseline_id == baseline_id
        ),
        "machine_id": machine_id,
        "n_normal": int((labels_arr == 0).sum()),
        "n_abnormal": int((labels_arr == 1).sum()),
        "auroc": round(auroc, 4),
        "separation_ratio": round(sep, 4),
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_inputs() -> None:
    """Raise FileNotFoundError if required resources are missing."""
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"MIMII dataset not found: {DATASET_ROOT}")
    if not BEATS_CHECKPOINT.exists():
        raise FileNotFoundError(f"BEATs checkpoint not found: {BEATS_CHECKPOINT}")


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

    pipeline, extractor, vec_builder, cache = _build_pipeline_components()

    baselines = get_all_baselines()
    rows: list[dict] = []

    print("=" * 60)
    print("Experiment E1 — Baseline Evaluation")
    print("=" * 60)
    print(f"Machine type : {proto.machine_type}")
    print(f"Machine IDs  : {list(proto.machine_ids)}")
    print(f"Baselines    : {[b.baseline_id for b in baselines]}")
    print()

    for baseline in baselines:
        print(f"[{baseline.baseline_id}] {baseline.name}")
        for machine_id in proto.machine_ids:
            profile_records = [
                r for r in split.profile_normal
                if r.machine_id == machine_id
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

            row = _evaluate_machine_id(
                baseline_id=baseline.baseline_id,
                machine_id=machine_id,
                profile_records=profile_records,
                test_records=test_records,
                pipeline=pipeline,
                extractor=extractor,
                vec_builder=vec_builder,
                cache=cache,
            )
            rows.append(row)
            print(f"AUROC={row['auroc']:.4f}  sep={row['separation_ratio']:.4f}")

        print()

    validate_results(rows)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print("Results saved to:", RESULTS_PATH)
    print("=" * 60)
    print()
    print(f"{'Baseline':<28} {'Machine':<8} {'N_norm':>7} {'N_abn':>6} {'AUROC':>7} {'Sep':>7}")
    print("-" * 68)
    for row in rows:
        print(
            f"{row['baseline_id']:<28} {row['machine_id']:<8} "
            f"{row['n_normal']:>7} {row['n_abnormal']:>6} "
            f"{row['auroc']:>7.4f} {row['separation_ratio']:>7.4f}"
        )


if __name__ == "__main__":
    main()
