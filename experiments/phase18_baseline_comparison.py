"""Phase 18 — Baseline Comparison for Anomaly Detection.

Compares the proposed machine-specific healthy-profile method (Phase 9,
normalized Euclidean drift on 256-dim learned embeddings) against three
standard one-class anomaly detection baselines:

    1. One-Class SVM (OC-SVM)
    2. Isolation Forest (IF)
    3. k-Nearest Neighbours anomaly score (kNN)

Protocol (identical to Phase 9):
    - Same DatasetSplitter: train_ratio=0.70, profile_ratio=0.15, seed=42
    - Baselines fit on profile_normal embeddings only (same data as Phase 9
      healthy profile construction — no training data leakage)
    - Evaluated on test_normal + test_abnormal
    - Embeddings: 256-dim L2-normalised learned fingerprints from the frozen
      Phase 9 ProjectionHead (models/contrastive/phase9/best_projection_head.pt)
    - Embeddings loaded from the existing fusion cache — no BEATs/projection
      retraining
    - ROC-AUC primary metric, Cohen's d secondary metric
    - All four machine types: fan, pump, slider, valve
    - Per-machine-ID and per-machine-type results
    - Outputs saved to experiments/results/phase18_baseline_comparison/

Usage:
    # Smoke test (2 recordings per split per machine ID):
    python experiments/phase18_baseline_comparison.py --smoke-test

    # Full experiment:
    python experiments/phase18_baseline_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contrastive_learning.inference import ContrastiveInference
from src.contrastive_learning.model import ProjectionHead
from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplitter
from src.fusion.cache import FusionCache
from src.fusion.fusion import FusionBuilder
from src.feature_extraction.extractor import FeatureExtractor
from src.feature_extraction.feature_vector import FeatureVectorBuilder
from src.preprocessing.pipeline import PreprocessingPipeline
from src.beats.encoder import BEATsEncoder

# ---------------------------------------------------------------------------
# Constants — identical split to Phase 9
# ---------------------------------------------------------------------------

EXPERIMENT_ID   = "phase18_baseline_comparison"
DATASET_ROOT    = Path("data/raw/MIMII")
CHECKPOINT_PATH = Path("models/contrastive/phase9/best_projection_head.pt")
RESULTS_DIR     = Path("experiments/results/phase18_baseline_comparison")
CACHE_ROOT      = Path("data/fusion_cache")
BEATS_CKPT      = Path("models/beats/BEATs_iter3_plus_AS2M.pt")

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]

TRAIN_RATIO   = 0.70
PROFILE_RATIO = 0.15
SEED          = 42

BASELINES = ["ocsvm", "iforest", "knn"]

# Phase 9 ROC-AUC values (normalized_euclidean) from clean evaluation_results.csv
PHASE9_RESULTS = {
    "fan":    {"id_00": 0.531451, "id_02": 0.742768, "id_04": 0.690834, "id_06": 0.879746, "overall": 0.698489},
    "pump":   {"id_00": 0.828441, "id_02": 0.798305, "id_04": 0.953208, "id_06": 0.932378, "overall": 0.865713},
    "slider": {"id_00": 0.992777, "id_02": 0.937167, "id_04": 0.673186, "id_06": 0.798863, "overall": 0.884086},
    "valve":  {"id_00": 0.969132, "id_02": 0.769782, "id_04": 0.913333, "id_06": 0.733778, "overall": 0.833465},
    "overall": 0.789649,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_setup() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Phase 9 checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run experiments/phase9_train.py first."
        )
    if not CACHE_ROOT.exists():
        raise FileNotFoundError(f"Fusion cache not found: {CACHE_ROOT}")


def _validate_isolation(split) -> None:
    train_paths   = {r.absolute_path for r in split.train_normal}
    profile_paths = {r.absolute_path for r in split.profile_normal}
    test_n_paths  = {r.absolute_path for r in split.test_normal}
    for overlap, label in [
        (train_paths & profile_paths,  "train_normal ∩ profile_normal"),
        (train_paths & test_n_paths,   "train_normal ∩ test_normal"),
        (profile_paths & test_n_paths, "profile_normal ∩ test_normal"),
    ]:
        if overlap:
            raise ValueError(f"DATA LEAKAGE: {label} ({len(overlap)} files)")


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        auc = float(roc_auc_score(y_true, scores))
        # Flip if classifier is inverted (anomaly scores should be higher for abnormal)
        return auc if auc >= 0.5 else float(roc_auc_score(y_true, -scores))
    except Exception:
        return float("nan")


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sa = float(np.std(a, ddof=1))
    sb = float(np.std(b, ddof=1))
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _build_inference_pipeline() -> tuple[FusionCache, ContrastiveInference]:
    """Build shared FusionCache + ContrastiveInference (reuses disk cache)."""
    pipeline    = PreprocessingPipeline(target_sr=16_000)
    extractor   = FeatureExtractor(sample_rate=16_000)
    vec_builder = FeatureVectorBuilder()
    encoder     = BEATsEncoder(BEATS_CKPT)
    fusion      = FusionBuilder()

    cache = FusionCache(
        cache_root=CACHE_ROOT,
        pipeline=pipeline,
        extractor=extractor,
        vec_builder=vec_builder,
        encoder=encoder,
        fusion=fusion,
    )

    head = ProjectionHead()
    inference = ContrastiveInference(
        projection_head=head,
        checkpoint_path=CHECKPOINT_PATH,
    )
    return cache, inference


def _get_embeddings(
    recordings: list,
    cache: FusionCache,
    inference: ContrastiveInference,
    label: str = "",
) -> tuple[np.ndarray, list[str]]:
    """Return (N, 256) embedding matrix and list of filenames."""
    embeddings, filenames = [], []
    for rec in recordings:
        try:
            fused = cache.load_or_create(rec)
            emb   = inference.generate_fingerprint(fused)
            embeddings.append(emb)
            filenames.append(rec.filename)
        except Exception as exc:
            print(f"  [WARN] skipping {rec.filename} ({label}): {exc}")
    if not embeddings:
        return np.empty((0, 256), dtype=np.float32), []
    return np.stack(embeddings, axis=0).astype(np.float32), filenames


# ---------------------------------------------------------------------------
# Baseline fitting and scoring
# ---------------------------------------------------------------------------

def _fit_baselines(X_train: np.ndarray, seed: int = SEED) -> dict:
    """Fit OC-SVM, Isolation Forest, and kNN on healthy profile embeddings."""
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors

    models = {}

    # OC-SVM: RBF kernel, nu=0.1 (expects ~10% outliers in training)
    ocsvm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
    ocsvm.fit(X_train)
    models["ocsvm"] = ocsvm

    # Isolation Forest
    iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=seed)
    iforest.fit(X_train)
    models["iforest"] = iforest

    # kNN: store training data for distance scoring at test time
    k = min(5, len(X_train))
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X_train)
    models["knn"] = (knn, k)

    return models


def _score_baselines(models: dict, X_test: np.ndarray) -> dict[str, np.ndarray]:
    """Return anomaly scores (higher = more anomalous) for each baseline."""
    scores = {}

    # OC-SVM: decision_function returns signed distance; negate so higher = anomalous
    scores["ocsvm"] = -models["ocsvm"].decision_function(X_test)

    # Isolation Forest: score_samples returns negative anomaly score; negate
    scores["iforest"] = -models["iforest"].score_samples(X_test)

    # kNN: mean distance to k nearest neighbours in training set
    knn, k = models["knn"]
    dists, _ = knn.kneighbors(X_test, n_neighbors=k)
    scores["knn"] = dists.mean(axis=1)

    return scores


# ---------------------------------------------------------------------------
# Per-machine evaluation
# ---------------------------------------------------------------------------

def _evaluate_machine(
    mt: str,
    mid: str,
    profile_recs: list,
    test_normal_recs: list,
    test_abnormal_recs: list,
    cache: FusionCache,
    inference: ContrastiveInference,
    smoke_test: bool,
) -> dict | None:
    """Fit baselines on profile_normal, evaluate on test splits. Returns metrics dict."""

    if smoke_test:
        profile_recs       = profile_recs[:2]
        test_normal_recs   = test_normal_recs[:2]
        test_abnormal_recs = test_abnormal_recs[:2]

    if not profile_recs:
        print(f"  [SKIP] {mt}/{mid} — no profile_normal recordings")
        return None
    if not test_normal_recs or not test_abnormal_recs:
        print(f"  [SKIP] {mt}/{mid} — missing test_normal or test_abnormal")
        return None

    # Extract embeddings
    X_profile, _  = _get_embeddings(profile_recs,       cache, inference, "profile")
    X_test_n, fn  = _get_embeddings(test_normal_recs,   cache, inference, "test_normal")
    X_test_ab, fa = _get_embeddings(test_abnormal_recs, cache, inference, "test_abnormal")

    if len(X_profile) < 2 or len(X_test_n) == 0 or len(X_test_ab) == 0:
        print(f"  [SKIP] {mt}/{mid} — insufficient embeddings after extraction")
        return None

    print(f"  {mt}/{mid}: profile={len(X_profile)}  test_n={len(X_test_n)}  test_ab={len(X_test_ab)}")

    # Fit baselines on profile_normal only
    models = _fit_baselines(X_profile)

    # Score all test recordings
    X_test_all = np.vstack([X_test_n, X_test_ab])
    y_true = np.array([0] * len(X_test_n) + [1] * len(X_test_ab))
    filenames_all = fn + fa
    labels_all    = ["normal"] * len(X_test_n) + ["abnormal"] * len(X_test_ab)

    baseline_scores = _score_baselines(models, X_test_all)

    # Compute metrics per baseline
    metrics: dict[str, dict] = {}
    for name, scores in baseline_scores.items():
        auc = _roc_auc(y_true, scores)
        normal_scores   = scores[y_true == 0]
        abnormal_scores = scores[y_true == 1]
        d = _cohens_d(abnormal_scores, normal_scores)
        metrics[name] = {"roc_auc": round(float(auc), 6), "cohens_d": round(float(d), 6)}

    return {
        "machine_type":  mt,
        "machine_id":    mid,
        "n_profile":     len(X_profile),
        "n_test_normal": len(X_test_n),
        "n_test_abnormal": len(X_test_ab),
        "metrics":       metrics,
        # Per-recording rows for the CSV
        "rows": [
            {
                "machine_type": mt,
                "machine_id":   mid,
                "filename":     fname,
                "true_label":   lbl,
                "ocsvm_score":  float(baseline_scores["ocsvm"][i]),
                "iforest_score": float(baseline_scores["iforest"][i]),
                "knn_score":    float(baseline_scores["knn"][i]),
            }
            for i, (fname, lbl) in enumerate(zip(filenames_all, labels_all))
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(smoke_test: bool = False) -> None:
    _validate_setup()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"Experiment     : {EXPERIMENT_ID}")
    print(f"Stage          : Baseline Comparison{'  [SMOKE TEST]' if smoke_test else ''}")
    print("=" * 65)
    print(f"Checkpoint     : {CHECKPOINT_PATH}")
    print(f"Results dir    : {RESULTS_DIR}")
    print(f"Baselines      : {BASELINES}")
    print(f"Split          : train={TRAIN_RATIO}  profile={PROFILE_RATIO}  seed={SEED}")
    print()

    t0 = time.perf_counter()

    # 1. Load dataset and reproduce Phase 9 split
    loader = DatasetLoader(DATASET_ROOT)
    all_recordings = loader.get_all_files()

    splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
    splits: dict[str, object] = {}
    for mt in MACHINE_TYPES:
        type_recs = [r for r in all_recordings if r.machine_type == mt]
        splits[mt] = splitter.split(type_recs)
        _validate_isolation(splits[mt])

    # 2. Print split summary
    print(f"{'Type':<8} {'profile_n':>10} {'test_n':>8} {'test_ab':>8}")
    print("-" * 38)
    for mt in MACHINE_TYPES:
        s = splits[mt]
        print(f"{mt:<8} {len(s.profile_normal):>10} {len(s.test_normal):>8} {len(s.test_abnormal):>8}")
    print()

    # 3. Build shared inference pipeline (reuses fusion cache — no recomputation)
    print("Loading inference pipeline (fusion cache + ProjectionHead)...")
    cache, inference = _build_inference_pipeline()
    print()

    # 4. Evaluate per machine type and ID
    all_rows: list[dict] = []
    per_type_results: dict[str, dict] = {}

    for mt in MACHINE_TYPES:
        split = splits[mt]
        print(f"{'='*50}")
        print(f"Machine type: {mt}")
        print(f"{'='*50}")

        type_machine_results: list[dict] = []

        for mid in MACHINE_IDS:
            profile_recs       = [r for r in split.profile_normal   if r.machine_id == mid]
            test_normal_recs   = [r for r in split.test_normal      if r.machine_id == mid]
            test_abnormal_recs = [r for r in split.test_abnormal    if r.machine_id == mid]

            result = _evaluate_machine(
                mt, mid,
                profile_recs, test_normal_recs, test_abnormal_recs,
                cache, inference, smoke_test,
            )
            if result is None:
                continue

            type_machine_results.append(result)
            all_rows.extend(result["rows"])

            # Print per-ID metrics
            for baseline, m in result["metrics"].items():
                auc_s = f"{m['roc_auc']:.4f}"
                d_s   = f"{m['cohens_d']:.4f}"
                print(f"    {mid}  {baseline:<10}  AUC={auc_s}  Cohen's d={d_s}")

        if not type_machine_results:
            continue

        # Aggregate per-type: pool all test recordings across IDs
        # Collect per-baseline scores and labels across all IDs
        type_rows = [r for res in type_machine_results for r in res["rows"]]
        type_y    = np.array([1 if r["true_label"] == "abnormal" else 0 for r in type_rows])

        type_metrics: dict[str, dict] = {}
        for bl in BASELINES:
            sc_key = f"{bl}_score"
            sc = np.array([r[sc_key] for r in type_rows])
            auc = _roc_auc(type_y, sc)
            normal_sc   = sc[type_y == 0]
            abnormal_sc = sc[type_y == 1]
            d = _cohens_d(abnormal_sc, normal_sc)
            type_metrics[bl] = {"roc_auc": round(float(auc), 6), "cohens_d": round(float(d), 6)}

        print(f"\n  {mt} — overall:")
        for bl, m in type_metrics.items():
            p9 = PHASE9_RESULTS[mt]["overall"]
            delta = m["roc_auc"] - p9
            sign  = "+" if delta >= 0 else ""
            print(f"    {bl:<10}  AUC={m['roc_auc']:.4f}  Cohen's d={m['cohens_d']:.4f}  "
                  f"vs Phase9={p9:.4f} ({sign}{delta:.4f})")
        print()

        per_type_results[mt] = {
            "per_id": {res["machine_id"]: res["metrics"] for res in type_machine_results},
            "overall": type_metrics,
            "n_test_normal":   sum(r["n_test_normal"]   for r in type_machine_results),
            "n_test_abnormal": sum(r["n_test_abnormal"] for r in type_machine_results),
        }

    # 5. Overall pooled metrics
    print("=" * 65)
    print("Overall (all machine types pooled)")
    print("=" * 65)

    all_y = np.array([1 if r["true_label"] == "abnormal" else 0 for r in all_rows])
    overall_metrics: dict[str, dict] = {}
    for bl in BASELINES:
        sc_key = f"{bl}_score"
        sc = np.array([r[sc_key] for r in all_rows])
        auc = _roc_auc(all_y, sc)
        normal_sc   = sc[all_y == 0]
        abnormal_sc = sc[all_y == 1]
        d = _cohens_d(abnormal_sc, normal_sc)
        overall_metrics[bl] = {"roc_auc": round(float(auc), 6), "cohens_d": round(float(d), 6)}

    p9_overall = PHASE9_RESULTS["overall"]
    for bl, m in overall_metrics.items():
        delta = m["roc_auc"] - p9_overall
        sign  = "+" if delta >= 0 else ""
        print(f"  {bl:<10}  AUC={m['roc_auc']:.4f}  Cohen's d={m['cohens_d']:.4f}  "
              f"vs Phase9={p9_overall:.4f} ({sign}{delta:.4f})")
    print(f"  {'Phase9':<10}  AUC={p9_overall:.4f}  (proposed method, reference)")
    print()

    elapsed = time.perf_counter() - t0
    print(f"Total elapsed: {elapsed:.1f}s")
    print()

    # 6. Save outputs
    _save_outputs(per_type_results, overall_metrics, all_rows, smoke_test)


def _save_outputs(
    per_type_results: dict,
    overall_metrics: dict,
    all_rows: list[dict],
    smoke_test: bool,
) -> None:
    suffix = "_smoketest" if smoke_test else ""

    # --- comparison_results.json ---
    comparison = {
        "experiment_id":  EXPERIMENT_ID,
        "smoke_test":     smoke_test,
        "checkpoint":     str(CHECKPOINT_PATH),
        "split": {
            "train_ratio":   TRAIN_RATIO,
            "profile_ratio": PROFILE_RATIO,
            "seed":          SEED,
        },
        "baselines": BASELINES,
        "phase9_reference": PHASE9_RESULTS,
        "results": {
            "per_type": per_type_results,
            "overall":  overall_metrics,
        },
        "notes": (
            "Baselines fit on profile_normal embeddings (same partition as Phase 9 "
            "healthy profile). Evaluated on test_normal + test_abnormal. "
            "Embeddings: 256-dim L2-normalised from Phase 9 ProjectionHead. "
            "No retraining of BEATs or ProjectionHead."
        ),
    }
    json_path = RESULTS_DIR / f"comparison_results{suffix}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)

    # --- comparison_summary.csv (per-machine-type, per-baseline) ---
    csv_rows = []
    for mt, type_data in per_type_results.items():
        for bl, m in type_data["overall"].items():
            csv_rows.append({
                "machine_type":  mt,
                "machine_id":    "overall",
                "method":        bl,
                "roc_auc":       m["roc_auc"],
                "cohens_d":      m["cohens_d"],
                "phase9_roc_auc": PHASE9_RESULTS[mt]["overall"],
                "delta_vs_phase9": round(m["roc_auc"] - PHASE9_RESULTS[mt]["overall"], 6),
            })
        for mid, id_metrics in type_data["per_id"].items():
            for bl, m in id_metrics.items():
                p9_id = PHASE9_RESULTS[mt].get(mid, float("nan"))
                csv_rows.append({
                    "machine_type":  mt,
                    "machine_id":    mid,
                    "method":        bl,
                    "roc_auc":       m["roc_auc"],
                    "cohens_d":      m["cohens_d"],
                    "phase9_roc_auc": p9_id,
                    "delta_vs_phase9": round(m["roc_auc"] - p9_id, 6) if not np.isnan(p9_id) else float("nan"),
                })
    # Add Phase 9 rows for completeness
    for mt in per_type_results:
        csv_rows.append({
            "machine_type":  mt,
            "machine_id":    "overall",
            "method":        "phase9_proposed",
            "roc_auc":       PHASE9_RESULTS[mt]["overall"],
            "cohens_d":      float("nan"),
            "phase9_roc_auc": PHASE9_RESULTS[mt]["overall"],
            "delta_vs_phase9": 0.0,
        })
    # Overall rows
    for bl, m in overall_metrics.items():
        csv_rows.append({
            "machine_type":  "overall",
            "machine_id":    "overall",
            "method":        bl,
            "roc_auc":       m["roc_auc"],
            "cohens_d":      m["cohens_d"],
            "phase9_roc_auc": PHASE9_RESULTS["overall"],
            "delta_vs_phase9": round(m["roc_auc"] - PHASE9_RESULTS["overall"], 6),
        })
    csv_rows.append({
        "machine_type":  "overall",
        "machine_id":    "overall",
        "method":        "phase9_proposed",
        "roc_auc":       PHASE9_RESULTS["overall"],
        "cohens_d":      float("nan"),
        "phase9_roc_auc": PHASE9_RESULTS["overall"],
        "delta_vs_phase9": 0.0,
    })

    csv_path = RESULTS_DIR / f"comparison_summary{suffix}.csv"
    fieldnames = ["machine_type", "machine_id", "method", "roc_auc", "cohens_d",
                  "phase9_roc_auc", "delta_vs_phase9"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # --- per_recording_scores.csv ---
    rec_csv_path = RESULTS_DIR / f"per_recording_scores{suffix}.csv"
    rec_fields = ["machine_type", "machine_id", "filename", "true_label",
                  "ocsvm_score", "iforest_score", "knn_score"]
    with rec_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rec_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print("Outputs saved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {rec_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 18 — Baseline Comparison")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick smoke test: 2 recordings per split per machine ID.",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
