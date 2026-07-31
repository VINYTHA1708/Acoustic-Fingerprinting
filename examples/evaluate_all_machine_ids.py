"""Evaluate DSP fingerprint separation for every Pump machine ID.

For each machine ID under pump/, builds a HealthyFingerprintProfile from all
normal recordings except one held-out sample, then evaluates up to 50 normal
and 50 abnormal recordings. Prints a summary table and overall statistics.

Usage:
    python examples/evaluate_all_machine_ids.py --root path/to/dataset
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from dataset import DatasetLoader
from drift import DriftAnalyzer
from feature_extraction import FeatureExtractor, FeatureVectorBuilder
from fingerprint import FingerprintGenerator
from preprocessing import PreprocessingPipeline
from profile import HealthyProfileBuilder

_MAX_SAMPLES = 50
_MACHINE_TYPE = "pump"


def build_fingerprint(record, pipeline, extractor, vec_builder, generator):
    result = pipeline.run(record.absolute_path)
    features = extractor.extract(result["waveform"])
    vector, names = vec_builder.build(features)
    return generator.generate(
        features=(vector, names),
        metadata=record,
        sample_rate=result["sample_rate"],
    )


def mean_distances(records, profile, pipeline, extractor, vec_builder, generator, analyzer):
    """Return (mean_euclidean, mean_manhattan, mean_cosine) for a list of records."""
    euclidean, manhattan, cosine = [], [], []
    for record in records:
        fp = build_fingerprint(record, pipeline, extractor, vec_builder, generator)
        drift = analyzer.analyze(fp, profile)
        euclidean.append(drift.euclidean_distance)
        manhattan.append(drift.manhattan_distance)
        cosine.append(drift.cosine_similarity)
    return float(np.mean(euclidean)), float(np.mean(manhattan)), float(np.mean(cosine))


def evaluate_machine_id(machine_id, normal_records, abnormal_records,
                        pipeline, extractor, vec_builder, generator, analyzer):
    """Evaluate one machine ID and return a result dict."""
    # Hold out first normal recording; build profile from the rest
    profile_records = normal_records[1:] if len(normal_records) > 1 else normal_records
    profile_fps = [
        build_fingerprint(r, pipeline, extractor, vec_builder, generator)
        for r in profile_records
    ]
    profile = HealthyProfileBuilder().build(profile_fps)

    # Re-insert held-out normal at position 0 for evaluation
    eval_normal = ([normal_records[0]] + profile_records)[:_MAX_SAMPLES]
    eval_abnormal = abnormal_records[:_MAX_SAMPLES]

    n_euc, n_man, n_cos = mean_distances(
        eval_normal, profile, pipeline, extractor, vec_builder, generator, analyzer
    )
    a_euc, a_man, a_cos = mean_distances(
        eval_abnormal, profile, pipeline, extractor, vec_builder, generator, analyzer
    )

    return {
        "machine_id": machine_id,
        "n_euc": n_euc, "n_man": n_man, "n_cos": n_cos,
        "a_euc": a_euc, "a_man": a_man, "a_cos": a_cos,
        "pass": a_euc > n_euc,
        "separation_ratio": a_euc / n_euc if n_euc > 0 else float("nan"),
    }


def main(root: str) -> None:
    loader = DatasetLoader(root)

    all_normal   = [r for r in loader.filter_by_label("normal")   if r.machine_type == _MACHINE_TYPE]
    all_abnormal = [r for r in loader.filter_by_label("abnormal") if r.machine_type == _MACHINE_TYPE]

    machine_ids = sorted({r.machine_id for r in all_normal})

    if not machine_ids:
        print(f"ERROR: No normal recordings found for machine type '{_MACHINE_TYPE}'.")
        sys.exit(1)

    print(f"\nMachine type : {_MACHINE_TYPE}")
    print(f"Machine IDs  : {machine_ids}\n")

    pipeline    = PreprocessingPipeline(target_sr=16_000)
    extractor   = FeatureExtractor(sample_rate=16_000)
    vec_builder = FeatureVectorBuilder()
    generator   = FingerprintGenerator()
    analyzer    = DriftAnalyzer()

    results = []
    for machine_id in machine_ids:
        normal_records   = [r for r in all_normal   if r.machine_id == machine_id]
        abnormal_records = [r for r in all_abnormal if r.machine_id == machine_id]

        if not abnormal_records:
            print(f"  [{machine_id}] SKIP — no abnormal recordings found.")
            continue

        print(f"  [{machine_id}] Evaluating "
              f"({min(len(normal_records), _MAX_SAMPLES)} normal, "
              f"{min(len(abnormal_records), _MAX_SAMPLES)} abnormal)...")

        result = evaluate_machine_id(
            machine_id, normal_records, abnormal_records,
            pipeline, extractor, vec_builder, generator, analyzer,
        )
        results.append(result)
        status = "PASS" if result["pass"] else "FAIL"
        print(f"  [{machine_id}] Done — {status}")

    if not results:
        print("No results to display.")
        sys.exit(1)

    # ── Summary table ─────────────────────────────────────────────────
    col = {
        "id":    12,
        "neuc":  14, "aeuc": 14,
        "nman":  14, "aman": 14,
        "ncos":  12, "acos": 12,
        "stat":   6,
    }

    header = (
        f"{'Machine ID':<{col['id']}}  "
        f"{'Norm Eucl':>{col['neuc']}}  {'Abn Eucl':>{col['aeuc']}}  "
        f"{'Norm Manh':>{col['nman']}}  {'Abn Manh':>{col['aman']}}  "
        f"{'Norm Cos':>{col['ncos']}}  {'Abn Cos':>{col['acos']}}  "
        f"{'Status':>{col['stat']}}"
    )
    divider = "-" * len(header)

    print("\n" + "=" * len(header))
    print("SUMMARY TABLE")
    print("=" * len(header))
    print(header)
    print(divider)

    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"{r['machine_id']:<{col['id']}}  "
            f"{r['n_euc']:>{col['neuc']}.4f}  {r['a_euc']:>{col['aeuc']}.4f}  "
            f"{r['n_man']:>{col['nman']}.4f}  {r['a_man']:>{col['aman']}.4f}  "
            f"{r['n_cos']:>{col['ncos']}.6f}  {r['a_cos']:>{col['acos']}.6f}  "
            f"{status:>{col['stat']}}"
        )

    # ── Overall statistics ────────────────────────────────────────────
    total      = len(results)
    n_pass     = sum(1 for r in results if r["pass"])
    n_fail     = total - n_pass
    ratios     = [r["separation_ratio"] for r in results if not np.isnan(r["separation_ratio"])]
    avg_ratio  = float(np.mean(ratios)) if ratios else float("nan")

    print("\n" + "=" * len(header))
    print("OVERALL STATISTICS")
    print("=" * len(header))
    print(f"  Total machine IDs evaluated : {total}")
    print(f"  PASS (Abn Eucl > Norm Eucl) : {n_pass}")
    print(f"  FAIL                        : {n_fail}")
    print(f"  Average separation ratio    : {avg_ratio:.4f}  (abnormal / normal mean Euclidean)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Evaluate DSP fingerprint separation for all {_MACHINE_TYPE} machine IDs."
    )
    parser.add_argument("--root", required=True, help="Dataset root directory.")
    args = parser.parse_args()
    main(args.root)
