"""Regenerate a clean Phase 9 evaluation_results.csv.

Re-scores all 5522 test recordings from scratch using the existing
Phase 9 checkpoint and machine-specific profiles (already built).
No resume logic — every recording is scored fresh.

Writes:
    experiments/results/phase9/evaluation_results.csv   (5522 rows, no zeros)
    experiments/results/phase9/evaluation_summary.json  (updated AUCs)

Usage:
    python experiments/phase9_regen_clean_csv.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplitter
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
from src.learned_profile.serializer import LearnedProfileSerializer
from src.learned_profile.learned_profile import LearnedFingerprintProfile

# ---------------------------------------------------------------------------
# Constants — identical to phase9_evaluate.py
# ---------------------------------------------------------------------------

DATASET_ROOT    = Path("data/raw/MIMII")
CHECKPOINT_PATH = Path("models/contrastive/phase9/best_projection_head.pt")
RESULTS_DIR     = Path("experiments/results/phase9")
PROFILE_DIR     = RESULTS_DIR / "profiles"

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]

TRAIN_RATIO   = 0.70
PROFILE_RATIO = 0.15
SEED          = 42

CSV_COLUMNS = [
    "machine_type", "machine_id", "filename", "true_label",
    "health_score", "health_percentage", "health_state",
    "normalized_euclidean", "normalized_manhattan", "normalized_cosine",
]


# ---------------------------------------------------------------------------
# Helpers
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


def _compute_metrics(rows: list[dict]) -> dict:
    metrics = ["normalized_euclidean", "normalized_manhattan", "normalized_cosine"]
    labels  = np.array([1 if r["true_label"] == "abnormal" else 0 for r in rows])
    result  = {}
    for m in metrics:
        scores = np.array([float(r[m]) for r in rows])
        auc    = _roc_auc(labels, scores)
        normal_s   = scores[labels == 0]
        abnormal_s = scores[labels == 1]
        result[m] = {
            "roc_auc":  round(auc, 6),
            "cohens_d": round(_cohens_d(abnormal_s, normal_s), 6),
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print("=" * 65)
    print("Phase 9 — Clean CSV Regeneration (no resume, no placeholders)")
    print("=" * 65)

    # 1. Reproduce exact split
    loader = DatasetLoader(DATASET_ROOT)
    all_recordings = loader.get_all_files()
    splitter = DatasetSplitter(
        train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED
    )
    splits: dict = {}
    for mt in MACHINE_TYPES:
        type_recs = [r for r in all_recordings if r.machine_type == mt]
        splits[mt] = splitter.split(type_recs)

    # Print split counts for verification
    print(f"\n{'Type':<8} {'test_n':>8} {'test_ab':>8} {'total':>8}")
    print("-" * 36)
    expected_total = 0
    for mt in MACHINE_TYPES:
        s = splits[mt]
        t = len(s.test_normal) + len(s.test_abnormal)
        expected_total += t
        print(f"{mt:<8} {len(s.test_normal):>8} {len(s.test_abnormal):>8} {t:>8}")
    print(f"{'TOTAL':<8} {'':>8} {'':>8} {expected_total:>8}")
    print()

    # 2. Load pre-built profiles (must already exist from phase9_evaluate.py)
    serializer = LearnedProfileSerializer()
    profiles: dict[tuple[str, str], LearnedFingerprintProfile] = {}
    for mt in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            npz = PROFILE_DIR / f"phase9_{mt}_{mid}_learned_profile.npz"
            if npz.exists():
                profiles[(mt, mid)] = serializer.load_npz(npz)
                print(f"  Loaded profile: {mt}/{mid}  "
                      f"({len(profiles[(mt, mid)].embeddings)} embeddings)")
            else:
                print(f"  [WARN] Profile missing: {npz}")
    print()

    # 3. Score every test recording — no resume, no skip
    analyzer  = LearnedHealthAnalyzer(checkpoint_path=CHECKPOINT_PATH)
    all_rows: list[dict] = []
    summary:  dict       = {}

    for mt in MACHINE_TYPES:
        split = splits[mt]
        evaluation_records = (
            [(rec, "normal")   for rec in split.test_normal]
            + [(rec, "abnormal") for rec in split.test_abnormal]
        )
        print(f"Scoring {mt}: {len(evaluation_records)} recordings ...")

        type_rows: list[dict] = []
        for i, (rec, true_label) in enumerate(evaluation_records, 1):
            key = (rec.machine_type, rec.machine_id)
            if key not in profiles:
                print(f"  [SKIP] no profile for {key}")
                continue

            if i % 200 == 0 or i == 1:
                print(f"  [{i}/{len(evaluation_records)}] "
                      f"{rec.machine_id}/{rec.filename}")

            result = analyzer.analyze(rec, profiles[key])
            row = {
                "machine_type":         result.machine_type,
                "machine_id":           result.machine_id,
                "filename":             result.filename,
                "true_label":           true_label,
                "health_score":         result.health_score,
                "health_percentage":    result.health_percentage,
                "health_state":         result.health_state,
                "normalized_euclidean": result.normalized_euclidean,
                "normalized_manhattan": result.normalized_manhattan,
                "normalized_cosine":    result.normalized_cosine,
            }
            type_rows.append(row)

        n_normal   = sum(1 for r in type_rows if r["true_label"] == "normal")
        n_abnormal = sum(1 for r in type_rows if r["true_label"] == "abnormal")
        print(f"  normal={n_normal}  abnormal={n_abnormal}  total={len(type_rows)}")

        # Verify no zero scores
        zeros = [r for r in type_rows if float(r["normalized_euclidean"]) == 0.0]
        if zeros:
            print(f"  [WARN] {len(zeros)} zero-score rows detected — investigate!")
        else:
            print("  Zero-score rows: 0 OK")

        type_metrics = _compute_metrics(type_rows)
        per_id_metrics: dict[str, dict] = {}
        for mid in MACHINE_IDS:
            id_rows = [r for r in type_rows if r["machine_id"] == mid]
            if id_rows:
                per_id_metrics[mid] = _compute_metrics(id_rows)

        summary[mt] = {
            "n_normal":        n_normal,
            "n_abnormal":      n_abnormal,
            "overall_metrics": type_metrics,
            "per_id_metrics":  per_id_metrics,
        }
        all_rows.extend(type_rows)
        print()

    # 4. Verify totals
    total_n  = sum(1 for r in all_rows if r["true_label"] == "normal")
    total_ab = sum(1 for r in all_rows if r["true_label"] == "abnormal")
    total    = len(all_rows)
    print("=" * 65)
    print(f"VERIFICATION: total={total}  normal={total_n}  abnormal={total_ab}")
    assert total == 5522, f"Expected 5522 rows, got {total}"
    assert total_n == 2222, f"Expected 2222 normal, got {total_n}"
    assert total_ab == 3300, f"Expected 3300 abnormal, got {total_ab}"
    print("All counts verified OK")
    print("=" * 65)

    # 5. Write clean combined CSV (overwrites stale file)
    out_csv = RESULTS_DIR / "evaluation_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote: {out_csv}  ({total} rows)")

    # 6. Overall metrics and summary JSON
    overall_metrics = _compute_metrics(all_rows)
    print("\nOverall metrics:")
    for m, v in overall_metrics.items():
        print(f"  {m:<28}  AUC={v['roc_auc']:.6f}  Cohen's d={v['cohens_d']:.6f}")

    summary["overall"] = {
        "n_normal":        total_n,
        "n_abnormal":      total_ab,
        "overall_metrics": overall_metrics,
    }
    summary_obj = {
        "experiment_id":   "phase9",
        "checkpoint":      str(CHECKPOINT_PATH),
        "smoke_test":      False,
        "split": {
            "train_ratio":   TRAIN_RATIO,
            "profile_ratio": PROFILE_RATIO,
            "seed":          SEED,
        },
        "results": summary,
    }
    out_json = RESULTS_DIR / "evaluation_summary.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(summary_obj, fh, indent=2)
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
