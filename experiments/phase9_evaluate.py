"""Phase 9 — Evaluation of the multi-machine shared ProjectionHead.

Follows the E1 evaluation methodology exactly:
  - Reproduce the same train/profile/test split (train_ratio=0.70,
    profile_ratio=0.15, seed=42) per machine type independently.
  - Build healthy profiles from profile_normal only.
  - Evaluate test_normal and test_abnormal.
  - Compute health_score, health_percentage, health_state,
    normalized_euclidean, normalized_manhattan, normalized_cosine.
  - Save per-machine-type CSVs and a combined CSV.
  - Compute ROC-AUC and Cohen's d per machine type and per machine ID.

Checkpoint: models/contrastive/phase9/best_projection_head.pt
Results:    experiments/results/phase9/

Usage:
    # Smoke test (1 recording per split per machine ID):
    python experiments/phase9_evaluate.py --smoke-test

    # Full evaluation:
    python experiments/phase9_evaluate.py
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
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
from src.learned_profile.builder import LearnedProfileBuilder
from src.learned_profile.serializer import LearnedProfileSerializer

# ---------------------------------------------------------------------------
# Phase 9 constants  (identical split parameters to phase9_train.py)
# ---------------------------------------------------------------------------

EXPERIMENT_ID   = "phase9"
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

def _validate_checkpoint() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Phase 9 checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run experiments/phase9_train.py first."
        )


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
            raise ValueError(f"ISOLATION FAIL: {label} ({len(overlap)} files)")


def _cohens_d(a, b) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-12 else 0.0


def _roc_auc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Profile building
# ---------------------------------------------------------------------------

def _build_profiles(
    splits: dict,
    builder: LearnedProfileBuilder,
    serializer: LearnedProfileSerializer,
    smoke_test: bool,
) -> dict[tuple[str, str], object]:
    """Build and save one profile per (machine_type, machine_id)."""
    profiles: dict[tuple[str, str], object] = {}
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    for mt in MACHINE_TYPES:
        split = splits[mt]
        ids_in_profile = {r.machine_id for r in split.profile_normal}
        for mid in MACHINE_IDS:
            if mid not in ids_in_profile:
                print(f"  [SKIP] {mt}/{mid} — no profile_normal recordings")
                continue

            recs = [r for r in split.profile_normal if r.machine_id == mid]
            if smoke_test:
                recs = recs[:1]

            profile = builder.build(mt, mid, recordings=recs)
            key = (mt, mid)
            profiles[key] = profile

            stem = f"phase9_{mt}_{mid}_learned_profile"
            serializer.save_npz(profile, PROFILE_DIR / f"{stem}.npz")
            serializer.save_json(profile, PROFILE_DIR / f"{stem}.json")

            print(f"  Built profile: {mt}/{mid}  ({len(recs)} recordings)")

    return profiles


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_split(
    mt: str,
    split,
    profiles: dict,
    analyzer: LearnedHealthAnalyzer,
    smoke_test: bool,
) -> list[dict]:

    # Smoke-test results go to a separate file so they never contaminate
    # the production evaluation CSV.
    if smoke_test:
        csv_path = RESULTS_DIR / f"evaluation_{mt}_smoketest.csv"
    else:
        csv_path = RESULTS_DIR / f"evaluation_{mt}.csv"

    # Load already completed recordings
    completed = set()
    existing_rows = []

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)

            for row in reader:
                completed.add(
                    (
                        row["machine_type"],
                        row["machine_id"],
                        row["filename"],
                        row["true_label"],
                    )
                )
                existing_rows.append(row)

        print(f"  Resuming: {len(existing_rows)} recordings already completed")

    test_normal = split.test_normal
    test_abnormal = split.test_abnormal

    if smoke_test:
        normal_sample = []
        abnormal_sample = []

        for mid in MACHINE_IDS:
            n_recs = [r for r in test_normal if r.machine_id == mid]
            a_recs = [r for r in test_abnormal if r.machine_id == mid]

            if n_recs:
                normal_sample.append(n_recs[0])

            if a_recs:
                abnormal_sample.append(a_recs[0])

        test_normal = normal_sample
        test_abnormal = abnormal_sample

    evaluation_records = (
        [(rec, "normal") for rec in test_normal]
        + [(rec, "abnormal") for rec in test_abnormal]
    )

    rows = existing_rows

    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=CSV_COLUMNS,
        )

        if not file_exists:
            writer.writeheader()

        for i, (record, true_label) in enumerate(evaluation_records, 1):

            record_key = (
                record.machine_type,
                record.machine_id,
                record.filename,
                true_label,
            )

            if record_key in completed:
                continue

            key = (
                record.machine_type,
                record.machine_id,
            )

            if key not in profiles:
                print(f"  [SKIP] no profile for {key}")
                continue

            if i % 100 == 0 or i == 1:
                print(
                    f"  [{mt}] [{i}/{len(evaluation_records)}] "
                    f"{record.machine_id}/{record.filename}"
                )

            result = analyzer.analyze(
                record,
                profiles[key],
            )

            row = {
                "machine_type": result.machine_type,
                "machine_id": result.machine_id,
                "filename": result.filename,
                "true_label": true_label,
                "health_score": result.health_score,
                "health_percentage": result.health_percentage,
                "health_state": result.health_state,
                "normalized_euclidean": result.normalized_euclidean,
                "normalized_manhattan": result.normalized_manhattan,
                "normalized_cosine": result.normalized_cosine,
            }

            # Save immediately
            writer.writerow(row)
            fh.flush()

            rows.append(row)
            completed.add(record_key)

    return rows
# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(rows: list[dict]) -> dict:
    """Compute ROC-AUC and Cohen's d for each drift metric."""
    import pandas as pd

    df = pd.DataFrame(rows)

    if df.empty or set(df["true_label"].unique()) < {"normal", "abnormal"}:
        return {}

    metrics = [
        "normalized_euclidean",
        "normalized_manhattan",
        "normalized_cosine",
    ]

    # Convert CSV-loaded string values to numeric
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    # Remove invalid numeric rows if any
    df = df.dropna(subset=metrics)

    if df.empty:
        return {}

    y_true = (df["true_label"] == "abnormal").astype(int).values

    results = {}

    for metric in metrics:

        scores = df[metric].values.astype(float)

        auc = _roc_auc(y_true, scores)

        if not np.isnan(auc) and auc < 0.5:
            auc = _roc_auc(y_true, -scores)

        normal_vals = (
            df.loc[
                df["true_label"] == "normal",
                metric
            ]
            .values
            .astype(float)
        )

        abnormal_vals = (
            df.loc[
                df["true_label"] == "abnormal",
                metric
            ]
            .values
            .astype(float)
        )

        d = _cohens_d(abnormal_vals, normal_vals)

        results[metric] = {
            "roc_auc": round(auc, 6),
            "cohens_d": round(d, 6),
        }

    return results

def _print_metrics(label: str, metrics: dict) -> None:
    print(f"\n  {label}")
    for metric, vals in metrics.items():
        auc = vals.get("roc_auc", "N/A")
        d   = vals.get("cohens_d", "N/A")
        auc_s = f"{auc:.4f}" if isinstance(auc, float) and not np.isnan(auc) else str(auc)
        d_s   = f"{d:.4f}"   if isinstance(d,   float) and not np.isnan(d)   else str(d)
        print(f"    {metric:<28}  AUC={auc_s}  Cohen's d={d_s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(smoke_test: bool = False) -> None:
    _validate_checkpoint()

    print("=" * 60)
    print(f"Experiment ID : {EXPERIMENT_ID}")
    print(f"Stage         : Evaluation{'  [SMOKE TEST]' if smoke_test else ''}")
    print("=" * 60)
    print(f"Checkpoint    : {CHECKPOINT_PATH}")
    print(f"Results dir   : {RESULTS_DIR}")
    print()

    # 1. Load all recordings
    loader = DatasetLoader(DATASET_ROOT)
    all_recordings = loader.get_all_files()

    # 2. Reproduce the same split per machine type (identical to phase9_train.py)
    splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
    splits: dict[str, object] = {}
    for mt in MACHINE_TYPES:
        type_recs = [r for r in all_recordings if r.machine_type == mt]
        splits[mt] = splitter.split(type_recs)
        _validate_isolation(splits[mt])

    # 3. Print split summary
    print(f"{'Type':<8} {'profile_n':>10} {'test_n':>8} {'test_ab':>8}")
    print("-" * 38)
    for mt in MACHINE_TYPES:
        s = splits[mt]
        print(f"{mt:<8} {len(s.profile_normal):>10} {len(s.test_normal):>8} {len(s.test_abnormal):>8}")
    print()

    # 4. Build profiles (profile_normal only)
    print("Building healthy profiles...")
    builder    = LearnedProfileBuilder(checkpoint_path=CHECKPOINT_PATH)
    serializer = LearnedProfileSerializer()
    profiles   = _build_profiles(splits, builder, serializer, smoke_test)
    print()

    # 5. Evaluate each machine type
    analyzer = LearnedHealthAnalyzer(checkpoint_path=CHECKPOINT_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    summary: dict = {}

    for mt in MACHINE_TYPES:
        print(f"Evaluating {mt}...")
        rows = _evaluate_split(mt, splits[mt], profiles, analyzer, smoke_test)


        n_normal   = sum(1 for r in rows if r["true_label"] == "normal")
        n_abnormal = sum(1 for r in rows if r["true_label"] == "abnormal")
        print(f"  normal={n_normal}  abnormal={n_abnormal}  total={len(rows)}")

        # Overall metrics for this type
        type_metrics = _compute_metrics(rows)
        _print_metrics(f"{mt} — overall", type_metrics)

        # Per-machine-ID metrics
        per_id_metrics: dict[str, dict] = {}
        for mid in MACHINE_IDS:
            id_rows = [r for r in rows if r["machine_id"] == mid]
            if id_rows:
                per_id_metrics[mid] = _compute_metrics(id_rows)
                _print_metrics(f"{mt}/{mid}", per_id_metrics[mid])

        summary[mt] = {
            "n_normal":        n_normal,
            "n_abnormal":      n_abnormal,
            "overall_metrics": type_metrics,
            "per_id_metrics":  per_id_metrics,
        }
        all_rows.extend(rows)
        print()

    # 6. Save combined CSV
    combined_path = RESULTS_DIR / "evaluation_results.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    # 7. Overall cross-type metrics
    print("=" * 60)
    print("Overall (all machine types combined)")
    print("=" * 60)
    overall_metrics = _compute_metrics(all_rows)
    _print_metrics("all types", overall_metrics)
    print()

    # 8. Save summary JSON
    summary["overall"] = {
        "n_normal":        sum(1 for r in all_rows if r["true_label"] == "normal"),
        "n_abnormal":      sum(1 for r in all_rows if r["true_label"] == "abnormal"),
        "overall_metrics": overall_metrics,
    }
    summary_meta = {
        "experiment_id":   EXPERIMENT_ID,
        "checkpoint":      str(CHECKPOINT_PATH),
        "smoke_test":      smoke_test,
        "split": {
            "train_ratio":   TRAIN_RATIO,
            "profile_ratio": PROFILE_RATIO,
            "seed":          SEED,
        },
        "results": summary,
    }
    summary_path = RESULTS_DIR / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_meta, fh, indent=2)

    print("Results saved to:")
    print(f"  {combined_path}")
    print(f"  {summary_path}")
    print(f"  {RESULTS_DIR}/evaluation_{{fan,pump,slider,valve}}.csv")
    print(f"  {PROFILE_DIR}/phase9_*_learned_profile.{{json,npz}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 9 evaluation")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick smoke test (1 recording per split per machine ID).",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
