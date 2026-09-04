"""Phase 11 — Multiple Seeds Stability Evaluation.

Runs the final Phase 9 method (multi-machine shared ProjectionHead) with
three random seeds: 42, 123, 2026.

For each seed:
  - Train a ProjectionHead using the same configuration as phase9_train.py
  - Evaluate on the same dataset split (train_ratio=0.70, profile_ratio=0.15)
  - Record ROC-AUC and Cohen's d for normalized_euclidean (primary metric)

Seed 42 reuses the existing Phase 9 checkpoint and evaluation results.
Seeds 123 and 2026 train new checkpoints and evaluate from scratch.

Results saved to: experiments/results/phase11/

Usage:
    python experiments/phase11_run.py
    python experiments/phase11_run.py --seeds 42 123 2026
    python experiments/phase11_run.py --skip-seed-42-train   # reuse existing checkpoint
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contrastive_learning.dataset import ContrastiveDataset
from src.contrastive_learning.loss import NTXentLoss
from src.contrastive_learning.model import ProjectionHead
from src.contrastive_learning.trainer import ContrastiveTrainer
from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplitter
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
from src.learned_profile.builder import LearnedProfileBuilder
from src.learned_profile.serializer import LearnedProfileSerializer

# ---------------------------------------------------------------------------
# Constants — identical to phase9_train.py / phase9_evaluate.py
# ---------------------------------------------------------------------------

DATASET_ROOT  = Path("data/raw/MIMII")
CACHE_ROOT    = Path("data/fusion_cache")

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]

TRAIN_RATIO   = 0.70
PROFILE_RATIO = 0.15

EPOCHS         = 20
BATCH_SIZE     = 16
LEARNING_RATE  = 0.001
TEMPERATURE    = 0.07
INPUT_DIM      = 921
PROJECTION_DIM = 256

RESULTS_DIR = Path("experiments/results/phase11")

CSV_COLUMNS = [
    "machine_type", "machine_id", "filename", "true_label",
    "health_score", "health_percentage", "health_state",
    "normalized_euclidean", "normalized_manhattan", "normalized_cosine",
]

PRIMARY_METRIC = "normalized_euclidean"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _cohens_d(a, b) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sa = float(np.std(a, ddof=1))
    sb = float(np.std(b, ddof=1))
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-12 else 0.0


def _roc_auc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        auc = float(roc_auc_score(y_true, scores))
        return auc if auc >= 0.5 else float(roc_auc_score(y_true, -scores))
    except Exception:
        return float("nan")


def _compute_overall_metrics(rows: list[dict]) -> dict:
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty or set(df["true_label"].unique()) < {"normal", "abnormal"}:
        return {}
    df[PRIMARY_METRIC] = pd.to_numeric(df[PRIMARY_METRIC], errors="coerce")
    df = df.dropna(subset=[PRIMARY_METRIC])
    if df.empty:
        return {}
    y_true = (df["true_label"] == "abnormal").astype(int).values
    scores = df[PRIMARY_METRIC].values.astype(float)
    auc = _roc_auc(y_true, scores)
    normal_vals   = df.loc[df["true_label"] == "normal",   PRIMARY_METRIC].values.astype(float)
    abnormal_vals = df.loc[df["true_label"] == "abnormal", PRIMARY_METRIC].values.astype(float)
    d = _cohens_d(abnormal_vals, normal_vals)
    return {"roc_auc": round(auc, 6), "cohens_d": round(d, 6)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train(seed: int, checkpoint_dir: Path) -> None:
    _set_seeds(seed)

    loader = DatasetLoader(DATASET_ROOT)
    all_recordings = loader.get_all_files()

    splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=seed)
    splits = {}
    for mt in MACHINE_TYPES:
        type_recs = [r for r in all_recordings if r.machine_type == mt]
        splits[mt] = splitter.split(type_recs)

    pooled_train = [r for mt in MACHINE_TYPES for r in splits[mt].train_normal]

    dataset = ContrastiveDataset(
        recordings=pooled_train,
        cache_root=CACHE_ROOT,
        seed=seed,
        val_split=0.20,
    )

    head      = ProjectionHead(input_dim=INPUT_DIM, output_dim=PROJECTION_DIM)
    criterion = NTXentLoss(temperature=TEMPERATURE)
    trainer   = ContrastiveTrainer(
        head=head,
        criterion=criterion,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )
    trainer.fit(dataset)

    hist     = trainer.history()
    best_val = min(hist["validation_losses"]) if hist["validation_losses"] else math.inf
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({"seed": seed, "best_validation_loss": best_val,
                   "loss_history": {"training": hist["training_losses"],
                                    "validation": hist["validation_losses"]}}, f, indent=2)
    print(f"  [seed={seed}] best_val_loss={best_val:.4f}  checkpoint={checkpoint_dir}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(seed: int, checkpoint_path: Path, seed_results_dir: Path) -> list[dict]:
    _set_seeds(seed)

    loader = DatasetLoader(DATASET_ROOT)
    all_recordings = loader.get_all_files()

    splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=seed)
    splits = {}
    for mt in MACHINE_TYPES:
        type_recs = [r for r in all_recordings if r.machine_type == mt]
        splits[mt] = splitter.split(type_recs)

    # Build profiles
    profile_dir = seed_results_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    builder    = LearnedProfileBuilder(checkpoint_path=checkpoint_path)
    serializer = LearnedProfileSerializer()
    profiles: dict[tuple[str, str], object] = {}

    for mt in MACHINE_TYPES:
        split = splits[mt]
        ids_in_profile = {r.machine_id for r in split.profile_normal}
        for mid in MACHINE_IDS:
            if mid not in ids_in_profile:
                continue
            recs = [r for r in split.profile_normal if r.machine_id == mid]
            profile = builder.build(mt, mid, recordings=recs)
            profiles[(mt, mid)] = profile
            stem = f"phase11_seed{seed}_{mt}_{mid}_learned_profile"
            serializer.save_npz(profile, profile_dir / f"{stem}.npz")
            serializer.save_json(profile, profile_dir / f"{stem}.json")
            print(f"  [seed={seed}] profile {mt}/{mid}  ({len(recs)} recs)")

    # Evaluate
    analyzer = LearnedHealthAnalyzer(checkpoint_path=checkpoint_path)
    all_rows: list[dict] = []

    for mt in MACHINE_TYPES:
        split = splits[mt]
        eval_recs = (
            [(r, "normal")   for r in split.test_normal]
            + [(r, "abnormal") for r in split.test_abnormal]
        )
        csv_path = seed_results_dir / f"evaluation_{mt}.csv"
        completed: set[tuple] = set()
        existing_rows: list[dict] = []

        if csv_path.exists():
            with csv_path.open("r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    completed.add((row["machine_type"], row["machine_id"],
                                   row["filename"], row["true_label"]))
                    existing_rows.append(row)
            print(f"  [seed={seed}] {mt}: resuming ({len(existing_rows)} done)")

        file_exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()

            for i, (record, true_label) in enumerate(eval_recs, 1):
                key_tuple = (record.machine_type, record.machine_id,
                             record.filename, true_label)
                if key_tuple in completed:
                    continue
                profile_key = (record.machine_type, record.machine_id)
                if profile_key not in profiles:
                    continue
                if i % 200 == 0 or i == 1:
                    print(f"  [seed={seed}] {mt} [{i}/{len(eval_recs)}]")
                result = analyzer.analyze(record, profiles[profile_key])
                row = {
                    "machine_type":        result.machine_type,
                    "machine_id":          result.machine_id,
                    "filename":            result.filename,
                    "true_label":          true_label,
                    "health_score":        result.health_score,
                    "health_percentage":   result.health_percentage,
                    "health_state":        result.health_state,
                    "normalized_euclidean": result.normalized_euclidean,
                    "normalized_manhattan": result.normalized_manhattan,
                    "normalized_cosine":   result.normalized_cosine,
                }
                writer.writerow(row)
                fh.flush()
                existing_rows.append(row)
                completed.add(key_tuple)

        all_rows.extend(existing_rows)

    # Save combined CSV
    combined_path = seed_results_dir / "evaluation_results.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    return all_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seeds: list[int], skip_seed42_train: bool) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    seed_metrics: dict[int, dict] = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Phase 11 — Seed {seed}")
        print(f"{'='*60}")

        seed_results_dir = RESULTS_DIR / f"seed_{seed}"
        seed_results_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_dir  = Path(f"models/contrastive/phase11/seed_{seed}")
        checkpoint_path = checkpoint_dir / "best_projection_head.pt"

        # ── Training ──────────────────────────────────────────────────────
        if seed == 42 and skip_seed42_train:
            # Reuse existing Phase 9 checkpoint
            phase9_ckpt = Path("models/contrastive/phase9/best_projection_head.pt")
            if phase9_ckpt.exists():
                checkpoint_path = phase9_ckpt
                print(f"  [seed=42] Reusing Phase 9 checkpoint: {phase9_ckpt}")
            else:
                print(f"  [seed=42] Phase 9 checkpoint not found, training fresh.")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                _train(seed, checkpoint_dir)
        else:
            if checkpoint_path.exists():
                print(f"  [seed={seed}] Checkpoint exists, skipping training.")
            else:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                print(f"  [seed={seed}] Training...")
                _train(seed, checkpoint_dir)

        # ── Evaluation ────────────────────────────────────────────────────
        print(f"  [seed={seed}] Evaluating...")
        all_rows = _evaluate(seed, checkpoint_path, seed_results_dir)

        metrics = _compute_overall_metrics(all_rows)
        seed_metrics[seed] = metrics

        n_normal   = sum(1 for r in all_rows if r["true_label"] == "normal")
        n_abnormal = sum(1 for r in all_rows if r["true_label"] == "abnormal")
        print(f"  [seed={seed}] n_normal={n_normal}  n_abnormal={n_abnormal}")
        print(f"  [seed={seed}] {PRIMARY_METRIC}:  "
              f"AUC={metrics.get('roc_auc', 'N/A'):.4f}  "
              f"Cohen's d={metrics.get('cohens_d', 'N/A'):.4f}")

        # Save per-seed summary
        seed_summary = {
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "n_normal": n_normal,
            "n_abnormal": n_abnormal,
            "primary_metric": PRIMARY_METRIC,
            "metrics": metrics,
        }
        with open(seed_results_dir / "seed_summary.json", "w") as f:
            json.dump(seed_summary, f, indent=2)

    # ── Aggregate across seeds ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 11 — Aggregate Results")
    print(f"{'='*60}")

    aucs = [seed_metrics[s]["roc_auc"]  for s in seeds if "roc_auc"  in seed_metrics[s]]
    ds   = [seed_metrics[s]["cohens_d"] for s in seeds if "cohens_d" in seed_metrics[s]]

    auc_mean = float(np.mean(aucs)) if aucs else float("nan")
    auc_std  = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
    d_mean   = float(np.mean(ds))   if ds   else float("nan")
    d_std    = float(np.std(ds,   ddof=1)) if len(ds)   > 1 else 0.0

    print(f"\n  Metric: {PRIMARY_METRIC}")
    cohens_d_header = "Cohen's d"
    print(f"  {'Seed':<8} {'ROC-AUC':>10} {cohens_d_header:>12}")
    print(f"  {'-'*32}")
    for s in seeds:
        m = seed_metrics.get(s, {})
        auc_s = f"{m.get('roc_auc', float('nan')):.4f}"
        d_s   = f"{m.get('cohens_d', float('nan')):.4f}"
        print(f"  {s:<8} {auc_s:>10} {d_s:>12}")
    print(f"  {'-'*32}")
    print(f"  {'mean':<8} {auc_mean:>10.4f} {d_mean:>12.4f}")
    print(f"  {'std':<8} {auc_std:>10.4f} {d_std:>12.4f}")

    # ── Save aggregate results ─────────────────────────────────────────────
    per_seed_rows = []
    for s in seeds:
        m = seed_metrics.get(s, {})
        per_seed_rows.append({
            "seed":      s,
            "roc_auc":   round(m.get("roc_auc",  float("nan")), 6),
            "cohens_d":  round(m.get("cohens_d", float("nan")), 6),
        })

    aggregate = {
        "experiment":    "phase11",
        "method":        "phase9_multi_machine_contrastive",
        "primary_metric": PRIMARY_METRIC,
        "seeds":         seeds,
        "per_seed":      per_seed_rows,
        "aggregate": {
            "roc_auc_mean": round(auc_mean, 6),
            "roc_auc_std":  round(auc_std,  6),
            "cohens_d_mean": round(d_mean,  6),
            "cohens_d_std":  round(d_std,   6),
        },
        "split": {
            "train_ratio":   TRAIN_RATIO,
            "profile_ratio": PROFILE_RATIO,
        },
        "training": {
            "epochs":        EPOCHS,
            "batch_size":    BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "temperature":   TEMPERATURE,
        },
    }

    out_json = RESULTS_DIR / "phase11_results.json"
    with open(out_json, "w") as f:
        json.dump(aggregate, f, indent=2)

    # CSV summary
    out_csv = RESULTS_DIR / "phase11_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "roc_auc", "cohens_d"])
        writer.writeheader()
        writer.writerows(per_seed_rows)
        writer.writerow({"seed": "mean", "roc_auc": round(auc_mean, 6), "cohens_d": round(d_mean, 6)})
        writer.writerow({"seed": "std",  "roc_auc": round(auc_std,  6), "cohens_d": round(d_std,  6)})

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 11: Multiple Seeds Evaluation")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 2026],
        help="Random seeds to evaluate (default: 42 123 2026)",
    )
    parser.add_argument(
        "--skip-seed-42-train", action="store_true", default=True,
        help="Reuse existing Phase 9 checkpoint for seed=42 (default: True)",
    )
    parser.add_argument(
        "--no-skip-seed-42-train", dest="skip_seed42_train", action="store_false",
        help="Force retrain for seed=42 instead of reusing Phase 9 checkpoint",
    )
    args = parser.parse_args()
    main(seeds=args.seeds, skip_seed42_train=args.skip_seed42_train)
