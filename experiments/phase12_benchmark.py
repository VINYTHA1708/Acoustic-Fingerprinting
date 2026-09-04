"""Phase 12 — Runtime Benchmark of the Phase 9 method.

Measures:
  1. Profile/training time  — derived from training_history.json (already run)
                              + live profile-build timing for one machine.
  2. Inference time per audio file — timed over N recordings (cache-hit path).
  3. Evaluation time for the dataset — timed over the full evaluation CSV rows.

All timings use time.perf_counter().  Each measurement is repeated where
feasible and the mean is reported.

Results saved to: experiments/results/phase12/
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplitter
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
from src.learned_profile.builder import LearnedProfileBuilder
from src.learned_profile.serializer import LearnedProfileSerializer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_ROOT    = Path("data/raw/MIMII")
CHECKPOINT_PATH = Path("models/contrastive/phase9/best_projection_head.pt")
RESULTS_DIR     = Path("experiments/results/phase12")
PHASE9_DIR      = Path("experiments/results/phase9")
TRAINING_HIST   = Path("models/contrastive/phase9/training_history.json")

TRAIN_RATIO   = 0.70
PROFILE_RATIO = 0.15
SEED          = 42

# Number of recordings to time for inference benchmark
N_INFERENCE_REPS = 30

# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------

def collect_environment() -> dict:
    return {
        "python_version":  platform.python_version(),
        "platform":        platform.platform(),
        "processor":       platform.processor(),
        "cpu_count":       str(getattr(platform, "cpu_count", lambda: "N/A")()),
        "torch_version":   torch.__version__,
        "cuda_available":  str(torch.cuda.is_available()),
        "cuda_device":     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "numpy_version":   np.__version__,
    }

# ---------------------------------------------------------------------------
# 1. Training time (from saved history)
# ---------------------------------------------------------------------------

def benchmark_training_time() -> dict:
    """Read training history and derive per-epoch and total training time.

    The Phase 9 training run did not instrument wall-clock time per epoch,
    so we derive a lower-bound estimate from the loss history length and
    the known dataset size, then report what is directly available.
    """
    with open(TRAINING_HIST, encoding="utf-8") as fh:
        hist = json.load(fh)

    n_epochs        = len(hist["loss_history"]["training"])
    total_train_recs = hist["total_pooled_train_normal"]   # 10296
    batch_size      = hist["training_configuration"]["batch_size"]  # 16

    # Estimate batches per epoch: pairs ≈ recordings (one pair per recording)
    # ContrastiveDataset uses 80% for train pairs
    approx_train_pairs = int(total_train_recs * 0.80)
    approx_batches_per_epoch = approx_train_pairs // batch_size

    return {
        "source":                    "training_history.json (Phase 9 run)",
        "note":                      "Wall-clock not instrumented during training; "
                                     "values derived from dataset/config.",
        "epochs":                    n_epochs,
        "total_pooled_train_normal": total_train_recs,
        "batch_size":                batch_size,
        "approx_train_pairs":        approx_train_pairs,
        "approx_batches_per_epoch":  approx_batches_per_epoch,
        "best_val_loss":             hist["best_validation_loss"],
        "final_train_loss":          hist["loss_history"]["training"][-1],
        "final_val_loss":            hist["loss_history"]["validation"][-1],
    }

# ---------------------------------------------------------------------------
# 2. Profile build time (live, one machine)
# ---------------------------------------------------------------------------

def benchmark_profile_build(splits: dict) -> dict:
    """Time building a healthy profile for pump/id_00 (profile_normal split)."""
    mt, mid = "pump", "id_00"
    recs = [r for r in splits[mt].profile_normal if r.machine_id == mid]

    builder = LearnedProfileBuilder(checkpoint_path=CHECKPOINT_PATH)

    t0 = time.perf_counter()
    profile = builder.build(mt, mid, recordings=recs)
    elapsed = time.perf_counter() - t0

    n = len(recs)
    return {
        "machine":              f"{mt}/{mid}",
        "n_recordings":         n,
        "total_time_s":         round(elapsed, 3),
        "mean_time_per_rec_s":  round(elapsed / n, 4) if n else 0.0,
    }

# ---------------------------------------------------------------------------
# 3. Inference time per audio file (cache-hit path)
# ---------------------------------------------------------------------------

def benchmark_inference(splits: dict, profile_cache: dict) -> dict:
    """Time analyzer.analyze() for N_INFERENCE_REPS recordings (cache-hit)."""
    analyzer = LearnedHealthAnalyzer(checkpoint_path=CHECKPOINT_PATH)

    # Use pump/id_00 test_normal recordings
    mt, mid = "pump", "id_00"
    recs = [r for r in splits[mt].test_normal if r.machine_id == mid]
    recs = recs[:N_INFERENCE_REPS]

    profile = profile_cache[(mt, mid)]

    times = []
    for rec in recs:
        t0 = time.perf_counter()
        analyzer.analyze(rec, profile)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return {
        "machine":          f"{mt}/{mid}",
        "n_recordings":     len(times),
        "mean_s":           round(float(times_arr.mean()), 4),
        "std_s":            round(float(times_arr.std()),  4),
        "min_s":            round(float(times_arr.min()),  4),
        "max_s":            round(float(times_arr.max()),  4),
        "median_s":         round(float(np.median(times_arr)), 4),
        "mean_ms":          round(float(times_arr.mean()) * 1000, 2),
        "note":             "Cache-hit path (fusion vector loaded from disk, "
                            "no BEATs/DSP recomputation)",
    }

# ---------------------------------------------------------------------------
# 4. Evaluation time (full dataset, from existing CSV)
# ---------------------------------------------------------------------------

def benchmark_evaluation(splits: dict, profile_cache: dict) -> dict:
    """Time running analyzer.analyze() over all test recordings for one machine type."""
    analyzer = LearnedHealthAnalyzer(checkpoint_path=CHECKPOINT_PATH)

    mt = "pump"
    split = splits[mt]
    all_recs = list(split.test_normal) + list(split.test_abnormal)

    t0 = time.perf_counter()
    for rec in all_recs:
        key = (rec.machine_type, rec.machine_id)
        if key in profile_cache:
            analyzer.analyze(rec, profile_cache[key])
    elapsed = time.perf_counter() - t0

    n = len(all_recs)
    return {
        "machine_type":         mt,
        "n_recordings":         n,
        "total_time_s":         round(elapsed, 3),
        "mean_time_per_rec_s":  round(elapsed / n, 4) if n else 0.0,
        "mean_time_per_rec_ms": round((elapsed / n) * 1000, 2) if n else 0.0,
        "note":                 "Cache-hit path; pump machine type only",
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 12 — Runtime Benchmark")
    print("=" * 60)

    # ── Environment ──────────────────────────────────────────────────
    print("\n[1/5] Collecting environment info...")
    env = collect_environment()
    for k, v in env.items():
        print(f"  {k:<22}: {v}")

    # ── Load dataset splits ──────────────────────────────────────────
    print("\n[2/5] Loading dataset splits...")
    loader   = DatasetLoader(DATASET_ROOT)
    all_recs = loader.get_all_files()
    splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
    splits   = {}
    for mt in ["fan", "pump", "slider", "valve"]:
        type_recs = [r for r in all_recs if r.machine_type == mt]
        splits[mt] = splitter.split(type_recs)
    print("  Splits loaded.")

    # ── Training time ────────────────────────────────────────────────
    print("\n[3/5] Benchmarking training time (from saved history)...")
    train_result = benchmark_training_time()
    for k, v in train_result.items():
        print(f"  {k:<36}: {v}")

    # ── Profile build time ───────────────────────────────────────────
    print("\n[4/5] Benchmarking profile build time (pump/id_00)...")
    profile_result = benchmark_profile_build(splits)
    print(f"  machine          : {profile_result['machine']}")
    print(f"  n_recordings     : {profile_result['n_recordings']}")
    print(f"  total_time_s     : {profile_result['total_time_s']}")
    print(f"  mean_per_rec_s   : {profile_result['mean_time_per_rec_s']}")

    # Build profiles for all pump IDs for inference/eval benchmarks
    print("\n  Building profiles for all pump IDs (for inference benchmark)...")
    builder    = LearnedProfileBuilder(checkpoint_path=CHECKPOINT_PATH)
    serializer = LearnedProfileSerializer()
    profile_cache: dict = {}
    for mid in ["id_00", "id_02", "id_04", "id_06"]:
        recs = [r for r in splits["pump"].profile_normal if r.machine_id == mid]
        if recs:
            profile_cache[("pump", mid)] = builder.build("pump", mid, recordings=recs)
            print(f"    pump/{mid}: {len(recs)} recordings")

    # ── Inference time ───────────────────────────────────────────────
    print(f"\n[5/5] Benchmarking inference time ({N_INFERENCE_REPS} recordings, cache-hit)...")
    infer_result = benchmark_inference(splits, profile_cache)
    print(f"  n_recordings     : {infer_result['n_recordings']}")
    print(f"  mean_s           : {infer_result['mean_s']}")
    print(f"  mean_ms          : {infer_result['mean_ms']}")
    print(f"  std_s            : {infer_result['std_s']}")
    print(f"  min_s            : {infer_result['min_s']}")
    print(f"  max_s            : {infer_result['max_s']}")
    print(f"  median_s         : {infer_result['median_s']}")

    # ── Evaluation time ──────────────────────────────────────────────
    print("\n[6/6] Benchmarking full evaluation time (pump, all test recordings)...")
    eval_result = benchmark_evaluation(splits, profile_cache)
    print(f"  machine_type     : {eval_result['machine_type']}")
    print(f"  n_recordings     : {eval_result['n_recordings']}")
    print(f"  total_time_s     : {eval_result['total_time_s']}")
    print(f"  mean_per_rec_s   : {eval_result['mean_time_per_rec_s']}")
    print(f"  mean_per_rec_ms  : {eval_result['mean_time_per_rec_ms']}")

    # ── Save results ─────────────────────────────────────────────────
    results = {
        "phase":       "phase12",
        "description": "Runtime benchmark of Phase 9 method",
        "environment": env,
        "training_time": train_result,
        "profile_build_time": profile_result,
        "inference_time_per_file": infer_result,
        "evaluation_time": eval_result,
    }

    out_json = RESULTS_DIR / "phase12_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # Flat CSV summary
    summary_rows = [
        {"metric": "profile_build_total_s",       "value": profile_result["total_time_s"],
         "unit": "s",  "n": profile_result["n_recordings"]},
        {"metric": "profile_build_mean_per_rec_s", "value": profile_result["mean_time_per_rec_s"],
         "unit": "s",  "n": profile_result["n_recordings"]},
        {"metric": "inference_mean_s",             "value": infer_result["mean_s"],
         "unit": "s",  "n": infer_result["n_recordings"]},
        {"metric": "inference_mean_ms",            "value": infer_result["mean_ms"],
         "unit": "ms", "n": infer_result["n_recordings"]},
        {"metric": "inference_std_s",              "value": infer_result["std_s"],
         "unit": "s",  "n": infer_result["n_recordings"]},
        {"metric": "inference_median_s",           "value": infer_result["median_s"],
         "unit": "s",  "n": infer_result["n_recordings"]},
        {"metric": "evaluation_total_s",           "value": eval_result["total_time_s"],
         "unit": "s",  "n": eval_result["n_recordings"]},
        {"metric": "evaluation_mean_per_rec_s",    "value": eval_result["mean_time_per_rec_s"],
         "unit": "s",  "n": eval_result["n_recordings"]},
        {"metric": "evaluation_mean_per_rec_ms",   "value": eval_result["mean_time_per_rec_ms"],
         "unit": "ms", "n": eval_result["n_recordings"]},
    ]
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "phase12_timing_summary.csv", index=False)

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  {out_json.name}")
    print(f"  phase12_timing_summary.csv")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 12 SUMMARY")
    print("=" * 60)
    print(f"  Training epochs              : {train_result['epochs']}")
    print(f"  Training recordings (pooled) : {train_result['total_pooled_train_normal']}")
    print(f"  Best validation loss         : {train_result['best_val_loss']:.4f}")
    print()
    print(f"  Profile build  ({profile_result['machine']}, {profile_result['n_recordings']} recs)")
    print(f"    Total time   : {profile_result['total_time_s']:.3f} s")
    print(f"    Per recording: {profile_result['mean_time_per_rec_s']:.4f} s")
    print()
    print(f"  Inference per file (cache-hit, n={infer_result['n_recordings']})")
    print(f"    Mean         : {infer_result['mean_ms']:.2f} ms")
    print(f"    Std          : {infer_result['std_s']*1000:.2f} ms")
    print(f"    Median       : {infer_result['median_s']*1000:.2f} ms")
    print()
    print(f"  Full evaluation ({eval_result['machine_type']}, n={eval_result['n_recordings']})")
    print(f"    Total time   : {eval_result['total_time_s']:.3f} s")
    print(f"    Per recording: {eval_result['mean_time_per_rec_ms']:.2f} ms")
    print()
    print("PHASE 12 PASSED")


if __name__ == "__main__":
    main()
