"""Phase 17 — Global vs Machine-Specific Healthy Profile Comparison.

Validates the central design choice of the acoustic anomaly detection method:
does using a *machine-specific* healthy profile outperform a single *global*
healthy profile built from all machines combined?

Methodology
-----------
Both conditions use the IDENTICAL:
  - Trained checkpoint  : models/contrastive/phase9/best_projection_head.pt
  - Dataset split       : DatasetSplitter(train_ratio=0.70, profile_ratio=0.15, seed=42)
  - Profile source      : profile_normal recordings ONLY (no leakage)
  - Anomaly score       : normalized_euclidean (z-score norm, primary metric)
  - ROC-AUC computation : sklearn roc_auc_score, flipped if AUC < 0.5 (same as phase9)
  - Test set            : test_normal + test_abnormal (same as phase9)

Condition A — Machine-Specific Profile (existing method, phase9 baseline)
    Each (machine_type, machine_id) pair has its own profile built from its
    own profile_normal recordings.  Scores are loaded directly from the
    already-computed phase9 evaluation CSVs to avoid redundant inference.

Condition B — Global Profile (new baseline)
    ONE profile is built by pooling ALL profile_normal embeddings from all
    machine types and IDs.  The global mean and std vectors are computed over
    this combined set.  Every test recording is scored against this single
    global profile using the same normalized_euclidean formula.

No model retraining.  No modification to existing results.

Existing files used
-------------------
  - experiments/results/phase9/evaluation_results.csv
      Source of machine-specific scores (Condition A) and test-set ground truth.
  - experiments/results/phase9/profiles/phase9_*_learned_profile.npz
      Pre-built per-machine profiles used to extract embeddings for the global
      profile construction (Condition B).
  - models/contrastive/phase9/best_projection_head.pt
      Loaded by LearnedProfileBuilder only if any phase9 profile NPZ is missing.

Results are written to:
  experiments/results/phase17_global_vs_specific/
    comparison_results.json   — full structured results artifact
    global_scores.csv         — per-recording global anomaly scores

Usage
-----
    python experiments/phase17_global_vs_specific_profile.py
    python experiments/phase17_global_vs_specific_profile.py --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplitter
from src.learned_drift.metrics import LearnedDriftMetrics
from src.learned_profile.builder import LearnedProfileBuilder
from src.learned_profile.learned_profile import LearnedFingerprintProfile
from src.learned_profile.serializer import LearnedProfileSerializer

# ---------------------------------------------------------------------------
# Constants — identical to phase9_evaluate.py
# ---------------------------------------------------------------------------

EXPERIMENT_ID    = "phase17"
DATASET_ROOT     = Path("data/raw/MIMII")
CHECKPOINT_PATH  = Path("models/contrastive/phase9/best_projection_head.pt")
PHASE9_RESULTS   = Path("experiments/results/phase9")
PHASE9_PROFILES  = PHASE9_RESULTS / "profiles"
RESULTS_DIR      = Path("experiments/results/phase17_global_vs_specific")

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]

TRAIN_RATIO   = 0.70
PROFILE_RATIO = 0.15
SEED          = 42

_STD_FLOOR = 1e-10   # matches LearnedDriftMetrics and final_method_config.json


# ---------------------------------------------------------------------------
# Helpers — identical logic to phase9_evaluate.py
# ---------------------------------------------------------------------------

def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        auc = float(roc_auc_score(y_true, scores))
        if not np.isnan(auc) and auc < 0.5:
            auc = float(roc_auc_score(y_true, -scores))
        return auc
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


def _validate_no_leakage(splits: dict) -> None:
    """Assert profile_normal and test sets are disjoint across all types."""
    for mt, split in splits.items():
        profile_paths = {r.absolute_path for r in split.profile_normal}
        test_paths    = {r.absolute_path for r in split.test_normal}
        test_ab_paths = {r.absolute_path for r in split.test_abnormal}
        train_paths   = {r.absolute_path for r in split.train_normal}
        for overlap, label in [
            (profile_paths & train_paths,   f"{mt}: profile_normal ∩ train_normal"),
            (profile_paths & test_paths,    f"{mt}: profile_normal ∩ test_normal"),
            (profile_paths & test_ab_paths, f"{mt}: profile_normal ∩ test_abnormal"),
        ]:
            if overlap:
                raise ValueError(f"LEAKAGE DETECTED — {label} ({len(overlap)} files)")


# ---------------------------------------------------------------------------
# Step 1 — Load phase9 machine-specific scores (Condition A)
# ---------------------------------------------------------------------------

def load_phase9_scores() -> list[dict]:
    """Load the already-computed phase9 evaluation results (Condition A).

    Returns rows with keys: machine_type, machine_id, filename, true_label,
    normalized_euclidean.
    """
    csv_path = PHASE9_RESULTS / "evaluation_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Phase 9 evaluation CSV not found: {csv_path}\n"
            "Run experiments/phase9_evaluate.py first."
        )
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "machine_type":        row["machine_type"],
                "machine_id":          row["machine_id"],
                "filename":            row["filename"],
                "true_label":          row["true_label"],
                "normalized_euclidean": float(row["normalized_euclidean"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Step 2 — Build global profile (Condition B)
# ---------------------------------------------------------------------------

def _load_or_build_machine_profile(
    mt: str,
    mid: str,
    split,
    builder: LearnedProfileBuilder,
    serializer: LearnedProfileSerializer,
    smoke_test: bool,
) -> LearnedFingerprintProfile:
    """Load the pre-built phase9 NPZ profile, or build it if missing."""
    npz_path = PHASE9_PROFILES / f"phase9_{mt}_{mid}_learned_profile.npz"
    if npz_path.exists():
        return serializer.load_npz(npz_path)

    # Fallback: build from profile_normal (should not happen in normal runs)
    print(f"  [WARN] Phase9 profile NPZ missing for {mt}/{mid} — building from scratch")
    recs = [r for r in split.profile_normal if r.machine_id == mid]
    if smoke_test:
        recs = recs[:2]
    return builder.build(mt, mid, recordings=recs)


def build_global_profile(
    splits: dict,
    smoke_test: bool,
) -> LearnedFingerprintProfile:
    """Pool all profile_normal embeddings and build one global profile.

    Embeddings are sourced from the pre-built phase9 per-machine NPZ profiles
    (which were themselves built from profile_normal only).  This avoids
    re-running inference and guarantees the same embeddings are used.

    The global profile uses machine_type='global' and machine_id='all' as
    identifiers; these are only used for serialisation labelling.
    """
    serializer = LearnedProfileSerializer()
    builder    = LearnedProfileBuilder(checkpoint_path=CHECKPOINT_PATH)

    all_embeddings: list[np.ndarray] = []
    profile_counts: dict[str, dict[str, int]] = {}

    for mt in MACHINE_TYPES:
        profile_counts[mt] = {}
        for mid in MACHINE_IDS:
            recs = [r for r in splits[mt].profile_normal if r.machine_id == mid]
            if not recs:
                continue
            profile = _load_or_build_machine_profile(
                mt, mid, splits[mt], builder, serializer, smoke_test
            )
            n = len(profile.embeddings)
            all_embeddings.append(profile.embeddings)
            profile_counts[mt][mid] = n
            print(f"  Loaded {mt}/{mid}: {n} profile embeddings")

    if not all_embeddings:
        raise ValueError("No profile embeddings found — cannot build global profile.")

    matrix   = np.concatenate(all_embeddings, axis=0).astype(np.float32)  # (N_total, 256)
    mean_vec = matrix.mean(axis=0)
    std_vec  = matrix.std(axis=0)

    total = len(matrix)
    print(f"\n  Global profile: {total} embeddings from "
          f"{sum(len(v) for v in profile_counts.values())} machine IDs")

    global_profile = LearnedFingerprintProfile(
        machine_type="global",
        machine_id="all",
        embedding_dimension=256,
        embeddings=matrix,
        mean_vector=mean_vec,
        std_vector=std_vec,
    )
    return global_profile, profile_counts


# ---------------------------------------------------------------------------
# Step 3 — Score test recordings against the global profile (Condition B)
# ---------------------------------------------------------------------------

def score_against_global_profile(
    phase9_rows: list[dict],
    global_profile: LearnedFingerprintProfile,
    machine_specific_profiles: dict[tuple[str, str], LearnedFingerprintProfile],
) -> list[dict]:
    """Compute global normalized_euclidean for every test recording.

    Embeddings are re-derived from the machine-specific profiles: for each
    test recording we look up its embedding from the phase9 per-machine
    profile's stored embeddings array.

    Since the phase9 evaluation CSV does not store raw embeddings, we must
    re-run inference.  We use LearnedProfileBuilder's FusionCache + inference
    pipeline, which hits the disk cache for all recordings that were already
    processed during phase9.
    """
    from src.beats.encoder import BEATsEncoder
    from src.contrastive_learning.inference import ContrastiveInference
    from src.contrastive_learning.model import ProjectionHead
    from src.dataset.loader import DatasetLoader
    from src.feature_extraction.extractor import FeatureExtractor
    from src.feature_extraction.feature_vector import FeatureVectorBuilder
    from src.fusion.cache import FusionCache
    from src.fusion.fusion import FusionBuilder
    from src.preprocessing.pipeline import PreprocessingPipeline

    _BEATS_CKPT  = Path("models/beats/BEATs_iter3_plus_AS2M.pt")
    _CACHE_ROOT  = Path("data/fusion_cache")

    pipeline    = PreprocessingPipeline(target_sr=16_000)
    extractor   = FeatureExtractor(sample_rate=16_000)
    vec_builder = FeatureVectorBuilder()
    encoder     = BEATsEncoder(_BEATS_CKPT)
    fusion      = FusionBuilder()
    cache       = FusionCache(
        cache_root=_CACHE_ROOT,
        pipeline=pipeline,
        extractor=extractor,
        vec_builder=vec_builder,
        encoder=encoder,
        fusion=fusion,
    )
    head      = ProjectionHead()
    inference = ContrastiveInference(projection_head=head, checkpoint_path=CHECKPOINT_PATH)
    metrics   = LearnedDriftMetrics()

    # Build a lookup: (machine_type, machine_id, filename) → AudioMetadata
    loader   = DatasetLoader(DATASET_ROOT)
    all_recs = loader.get_all_files()
    rec_index: dict[tuple[str, str, str], object] = {
        (r.machine_type, r.machine_id, r.filename): r for r in all_recs
    }

    global_rows: list[dict] = []
    total = len(phase9_rows)

    for i, row in enumerate(phase9_rows, 1):
        if i % 200 == 0 or i == 1:
            print(f"  Scoring [{i}/{total}] {row['machine_type']}/{row['machine_id']}/{row['filename']}")

        key = (row["machine_type"], row["machine_id"], row["filename"])
        rec = rec_index.get(key)
        if rec is None:
            print(f"  [WARN] Recording not found in dataset index: {key}")
            continue

        try:
            fused     = cache.load_or_create(rec)
            embedding = inference.generate_fingerprint(fused)
        except Exception as exc:
            print(f"  [WARN] Inference failed for {key}: {exc}")
            continue

        # Compute normalized_euclidean against the global profile directly.
        # LearnedDriftMetrics.compute() enforces no machine_type/id check.
        (_, _, _, _, _, norm_euclid, _, _, _) = metrics.compute(embedding, global_profile)

        global_rows.append({
            "machine_type":              row["machine_type"],
            "machine_id":                row["machine_id"],
            "filename":                  row["filename"],
            "true_label":                row["true_label"],
            "global_normalized_euclidean": norm_euclid,
            "specific_normalized_euclidean": row["normalized_euclidean"],
        })

    return global_rows


# ---------------------------------------------------------------------------
# Step 4 — Compute metrics
# ---------------------------------------------------------------------------

def compute_metrics_for_condition(
    rows: list[dict],
    score_key: str,
) -> dict:
    """Compute ROC-AUC and Cohen's d for a given score column."""
    if not rows:
        return {}
    labels = np.array([1 if r["true_label"] == "abnormal" else 0 for r in rows])
    scores = np.array([r[score_key] for r in rows], dtype=float)
    normal_scores   = scores[labels == 0]
    abnormal_scores = scores[labels == 1]
    return {
        "roc_auc":  round(_roc_auc(labels, scores), 6),
        "cohens_d": round(_cohens_d(abnormal_scores, normal_scores), 6),
        "n_normal":   int((labels == 0).sum()),
        "n_abnormal": int((labels == 1).sum()),
    }


def compute_all_metrics(
    scored_rows: list[dict],
) -> dict:
    """Compute per-type, per-id, and overall metrics for both conditions."""
    results: dict = {"per_type": {}, "overall": {}}

    for mt in MACHINE_TYPES:
        mt_rows = [r for r in scored_rows if r["machine_type"] == mt]
        if not mt_rows:
            continue
        results["per_type"][mt] = {
            "global":   compute_metrics_for_condition(mt_rows, "global_normalized_euclidean"),
            "specific": compute_metrics_for_condition(mt_rows, "specific_normalized_euclidean"),
            "per_id":   {},
        }
        for mid in MACHINE_IDS:
            id_rows = [r for r in mt_rows if r["machine_id"] == mid]
            if id_rows:
                results["per_type"][mt]["per_id"][mid] = {
                    "global":   compute_metrics_for_condition(id_rows, "global_normalized_euclidean"),
                    "specific": compute_metrics_for_condition(id_rows, "specific_normalized_euclidean"),
                }

    results["overall"]["global"]   = compute_metrics_for_condition(scored_rows, "global_normalized_euclidean")
    results["overall"]["specific"] = compute_metrics_for_condition(scored_rows, "specific_normalized_euclidean")
    return results


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(metrics: dict, profile_counts: dict) -> None:
    print()
    print("=" * 70)
    print("PHASE 17 — Global vs Machine-Specific Profile Comparison")
    print("=" * 70)
    print(f"  Checkpoint : {CHECKPOINT_PATH}")
    print(f"  Split      : train={TRAIN_RATIO}, profile={PROFILE_RATIO}, seed={SEED}")
    print(f"  Metric     : normalized_euclidean (primary, same as phase9)")
    print()

    # Profile size summary
    total_profile = sum(
        n for mid_counts in profile_counts.values() for n in mid_counts.values()
    )
    print(f"  Global profile built from {total_profile} profile_normal embeddings:")
    for mt in MACHINE_TYPES:
        counts = profile_counts.get(mt, {})
        per_id = "  ".join(f"{mid}:{n}" for mid, n in counts.items())
        print(f"    {mt:<8}  {per_id}")
    print()

    # Per-type table
    print(f"  {'Type':<8}  {'Global AUC':>12}  {'Specific AUC':>14}  {'Delta AUC':>10}  {'Winner':>10}")
    print("  " + "-" * 60)
    for mt in MACHINE_TYPES:
        if mt not in metrics["per_type"]:
            continue
        g = metrics["per_type"][mt]["global"]["roc_auc"]
        s = metrics["per_type"][mt]["specific"]["roc_auc"]
        delta = s - g
        winner = "SPECIFIC" if delta > 0 else ("GLOBAL" if delta < 0 else "TIE")
        print(f"  {mt:<8}  {g:>12.6f}  {s:>14.6f}  {delta:>+10.4f}  {winner:>10}")

    print("  " + "-" * 60)
    g_all = metrics["overall"]["global"]["roc_auc"]
    s_all = metrics["overall"]["specific"]["roc_auc"]
    delta_all = s_all - g_all
    winner_all = "SPECIFIC" if delta_all > 0 else ("GLOBAL" if delta_all < 0 else "TIE")
    print(f"  {'OVERALL':<8}  {g_all:>12.6f}  {s_all:>14.6f}  {delta_all:>+10.4f}  {winner_all:>10}")
    print()

    # Conclusion
    print("  CONCLUSION:")
    if delta_all > 0.005:
        print(f"  Machine-specific profiles IMPROVE ROC-AUC by {delta_all:+.4f} overall.")
        print("  The per-machine healthy reference is beneficial.")
    elif delta_all < -0.005:
        print(f"  Global profile OUTPERFORMS machine-specific by {-delta_all:.4f} overall.")
        print("  Unexpected: the global baseline is stronger.")
    else:
        print(f"  Results are approximately EQUAL (Delta={delta_all:+.4f}).")
        print("  Machine-specific profiling provides negligible benefit overall.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(smoke_test: bool = False) -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Phase 9 checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run experiments/phase9_train.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Experiment    : {EXPERIMENT_ID} — Global vs Machine-Specific Profile")
    print(f"Smoke test    : {smoke_test}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Reproduce the exact same split (for leakage validation only;
    #    actual scores come from phase9 CSV and phase9 profile NPZs)
    # ------------------------------------------------------------------
    print("\n[1/5] Reproducing dataset split (seed=42, same as phase9)...")
    loader         = DatasetLoader(DATASET_ROOT)
    all_recordings = loader.get_all_files()
    splitter       = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
    splits: dict   = {}
    for mt in MACHINE_TYPES:
        type_recs  = [r for r in all_recordings if r.machine_type == mt]
        splits[mt] = splitter.split(type_recs)

    _validate_no_leakage(splits)
    print("  No data leakage detected.")

    # Print split counts for verification
    print(f"\n  {'Type':<8} {'profile_n':>10} {'test_n':>8} {'test_ab':>8}")
    print("  " + "-" * 36)
    for mt in MACHINE_TYPES:
        s = splits[mt]
        print(f"  {mt:<8} {len(s.profile_normal):>10} {len(s.test_normal):>8} {len(s.test_abnormal):>8}")

    # ------------------------------------------------------------------
    # 2. Load Condition A scores from phase9 CSV
    # ------------------------------------------------------------------
    print("\n[2/5] Loading machine-specific scores from phase9 evaluation CSV...")
    phase9_rows = load_phase9_scores()
    if smoke_test:
        # Take a small balanced sample per machine type for speed
        sampled = []
        for mt in MACHINE_TYPES:
            mt_rows = [r for r in phase9_rows if r["machine_type"] == mt]
            normal_rows   = [r for r in mt_rows if r["true_label"] == "normal"][:5]
            abnormal_rows = [r for r in mt_rows if r["true_label"] == "abnormal"][:5]
            sampled.extend(normal_rows + abnormal_rows)
        phase9_rows = sampled
    n_normal   = sum(1 for r in phase9_rows if r["true_label"] == "normal")
    n_abnormal = sum(1 for r in phase9_rows if r["true_label"] == "abnormal")
    print(f"  Loaded {len(phase9_rows)} rows: {n_normal} normal, {n_abnormal} abnormal")

    # ------------------------------------------------------------------
    # 3. Build global profile (Condition B)
    # ------------------------------------------------------------------
    print("\n[3/5] Building global healthy profile from phase9 profile NPZs...")
    global_profile, profile_counts = build_global_profile(splits, smoke_test)

    # Save global profile
    global_npz  = RESULTS_DIR / "global_profile.npz"
    global_json = RESULTS_DIR / "global_profile.json"
    serializer  = LearnedProfileSerializer()
    serializer.save_npz(global_profile, global_npz)
    serializer.save_json(global_profile, global_json)
    print(f"  Global profile saved: {global_npz}")

    # ------------------------------------------------------------------
    # 4. Score test recordings against global profile
    # ------------------------------------------------------------------
    print("\n[4/5] Scoring test recordings against global profile...")
    scored_rows = score_against_global_profile(
        phase9_rows, global_profile, machine_specific_profiles={}
    )
    print(f"  Scored {len(scored_rows)} recordings")

    # Save global scores CSV
    csv_path = RESULTS_DIR / "global_scores.csv"
    csv_cols  = [
        "machine_type", "machine_id", "filename", "true_label",
        "global_normalized_euclidean", "specific_normalized_euclidean",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows(scored_rows)
    print(f"  Scores saved: {csv_path}")

    # ------------------------------------------------------------------
    # 5. Compute and report metrics
    # ------------------------------------------------------------------
    print("\n[5/5] Computing ROC-AUC and Cohen's d for both conditions...")
    metrics = compute_all_metrics(scored_rows)

    # Verify sample counts match phase9 (only in full run)
    if not smoke_test:
        p9_n_normal   = 2222  # from phase9 evaluation_summary.json
        p9_n_abnormal = 3300
        actual_n   = metrics["overall"]["specific"]["n_normal"]
        actual_ab  = metrics["overall"]["specific"]["n_abnormal"]
        if actual_n != p9_n_normal or actual_ab != p9_n_abnormal:
            print(f"  [WARN] Sample count mismatch vs phase9 reference "
                  f"(expected n={p9_n_normal}, ab={p9_n_abnormal}; "
                  f"got n={actual_n}, ab={actual_ab})")
        else:
            print(f"  Sample counts verified: {actual_n} normal, {actual_ab} abnormal "
                  f"(matches phase9 reference)")

    # Save JSON results artifact
    result_artifact = {
        "experiment_id":   EXPERIMENT_ID,
        "description":     "Global vs Machine-Specific Healthy Profile Comparison",
        "smoke_test":      smoke_test,
        "checkpoint":      str(CHECKPOINT_PATH),
        "phase9_csv_used": str(PHASE9_RESULTS / "evaluation_results.csv"),
        "phase9_profiles_used": str(PHASE9_PROFILES),
        "split": {
            "train_ratio":   TRAIN_RATIO,
            "profile_ratio": PROFILE_RATIO,
            "seed":          SEED,
        },
        "global_profile_embedding_count": int(len(global_profile.embeddings)),
        "profile_counts_per_machine": profile_counts,
        "metrics": metrics,
    }
    json_path = RESULTS_DIR / "comparison_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result_artifact, fh, indent=2)
    print(f"  Results artifact saved: {json_path}")

    # Console summary
    print_summary(metrics, profile_counts)

    print(f"\nAll results written to: {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 17: Global vs Machine-Specific Profile Comparison"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on a small subset (5 normal + 5 abnormal per machine type) for quick validation.",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
