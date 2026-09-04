"""
Phase 9 — Step 9: Cross-Machine Analysis and Comparison
Reads evaluation CSVs + evaluation_summary.json; writes results to
experiments/results/phase9/comparison_e1/
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path("experiments/results/phase9")
OUT_DIR     = BASE / "comparison_e1"
SUMMARY_JSON = BASE / "evaluation_summary.json"

CSV_MAP = {
    "pump":   BASE / "evaluation_pump_repaired.csv",
    "fan":    BASE / "evaluation_fan.csv",
    "slider": BASE / "evaluation_slider.csv",
    "valve":  BASE / "evaluation_valve.csv",
}

METRICS = ["normalized_euclidean", "normalized_manhattan", "normalized_cosine"]
METRIC_SHORT = {"normalized_euclidean": "euclidean",
                "normalized_manhattan": "manhattan",
                "normalized_cosine":    "cosine"}

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ground-truth summary ────────────────────────────────────────────────
with open(SUMMARY_JSON) as f:
    summary = json.load(f)

# ── Helper: Cohen's d ────────────────────────────────────────────────────────
def cohens_d(normal_vals: np.ndarray, abnormal_vals: np.ndarray) -> float:
    n1, n2 = len(normal_vals), len(abnormal_vals)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_std = np.sqrt(
        ((n1 - 1) * normal_vals.std(ddof=1) ** 2 +
         (n2 - 1) * abnormal_vals.std(ddof=1) ** 2) / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return float("nan")
    return float((abnormal_vals.mean() - normal_vals.mean()) / pooled_std)


# ── 1. Load all CSVs ─────────────────────────────────────────────────────────
dfs = {mt: pd.read_csv(p) for mt, p in CSV_MAP.items()}

# ── 2. Per-machine-type overall metrics (validated against summary.json) ─────
machine_rows = []
for mt, df in dfs.items():
    normal_df   = df[df["true_label"] == "normal"]
    abnormal_df = df[df["true_label"] == "abnormal"]
    y_true = (df["true_label"] == "abnormal").astype(int).values

    for metric in METRICS:
        scores = df[metric].values
        # AUC: higher drift → more likely abnormal
        auc = float(roc_auc_score(y_true, scores))
        d   = cohens_d(normal_df[metric].values, abnormal_df[metric].values)

        # Pull validated values from summary.json
        ref = summary["results"][mt]["overall_metrics"][metric]
        auc_ref = ref["roc_auc"]
        d_ref   = ref["cohens_d"]

        machine_rows.append({
            "machine_type":    mt,
            "metric":          metric,
            "metric_short":    METRIC_SHORT[metric],
            "n_normal":        len(normal_df),
            "n_abnormal":      len(abnormal_df),
            "roc_auc_computed": round(auc, 6),
            "roc_auc_summary":  round(auc_ref, 6),
            "cohens_d_computed": round(d, 6),
            "cohens_d_summary":  round(d_ref, 6),
            "auc_match":        abs(auc - auc_ref) < 0.001,
        })

machine_df = pd.DataFrame(machine_rows)

# ── 3. Per-machine-id metrics ─────────────────────────────────────────────────
per_id_rows = []
for mt, df in dfs.items():
    for mid in sorted(df["machine_id"].unique()):
        sub = df[df["machine_id"] == mid]
        normal_sub   = sub[sub["true_label"] == "normal"]
        abnormal_sub = sub[sub["true_label"] == "abnormal"]
        y_true = (sub["true_label"] == "abnormal").astype(int).values

        for metric in METRICS:
            scores = sub[metric].values
            auc = float(roc_auc_score(y_true, scores))
            d   = cohens_d(normal_sub[metric].values, abnormal_sub[metric].values)

            ref = summary["results"][mt]["per_id_metrics"][mid][metric]
            per_id_rows.append({
                "machine_type": mt,
                "machine_id":   mid,
                "metric":       metric,
                "metric_short": METRIC_SHORT[metric],
                "n_normal":     len(normal_sub),
                "n_abnormal":   len(abnormal_sub),
                "roc_auc":      round(auc, 6),
                "cohens_d":     round(d, 6),
                "roc_auc_ref":  round(ref["roc_auc"], 6),
                "cohens_d_ref": round(ref["cohens_d"], 6),
            })

per_id_df = pd.DataFrame(per_id_rows)

# ── 4. Overall across all machines ───────────────────────────────────────────
all_df = pd.concat(dfs.values(), ignore_index=True)
overall_rows = []
y_true_all = (all_df["true_label"] == "abnormal").astype(int).values
normal_all   = all_df[all_df["true_label"] == "normal"]
abnormal_all = all_df[all_df["true_label"] == "abnormal"]

for metric in METRICS:
    auc = float(roc_auc_score(y_true_all, all_df[metric].values))
    d   = cohens_d(normal_all[metric].values, abnormal_all[metric].values)
    ref = summary["results"]["overall"]["overall_metrics"][metric]
    overall_rows.append({
        "metric":          metric,
        "metric_short":    METRIC_SHORT[metric],
        "n_normal":        len(normal_all),
        "n_abnormal":      len(abnormal_all),
        "roc_auc":         round(auc, 6),
        "cohens_d":        round(d, 6),
        "roc_auc_ref":     round(ref["roc_auc"], 6),
        "cohens_d_ref":    round(ref["cohens_d"], 6),
    })

overall_df = pd.DataFrame(overall_rows)

# ── 5. Best / worst combinations ─────────────────────────────────────────────
# Use summary.json values as source of truth
euc_man_rows = []
for mt in ["fan", "pump", "slider", "valve"]:
    for metric in METRICS:
        ref = summary["results"][mt]["overall_metrics"][metric]
        euc_man_rows.append({
            "machine_type": mt,
            "metric":       metric,
            "roc_auc":      ref["roc_auc"],
            "cohens_d":     ref["cohens_d"],
        })

rank_df = pd.DataFrame(euc_man_rows)

best_auc  = rank_df.loc[rank_df["roc_auc"].idxmax()]
worst_auc = rank_df.loc[rank_df["roc_auc"].idxmin()]
best_d    = rank_df.loc[rank_df["cohens_d"].idxmax()]
worst_d   = rank_df.loc[rank_df["cohens_d"].idxmin()]

# Best/worst per metric family (euclidean/manhattan only — cosine is unreliable)
euc_man = rank_df[rank_df["metric"].isin(["normalized_euclidean", "normalized_manhattan"])]
best_auc_euc_man  = euc_man.loc[euc_man["roc_auc"].idxmax()]
worst_auc_euc_man = euc_man.loc[euc_man["roc_auc"].idxmin()]

# ── 6. Machine-type summary table (pivot: metric × machine) ──────────────────
pivot_auc = rank_df.pivot(index="metric", columns="machine_type", values="roc_auc").round(6)
pivot_d   = rank_df.pivot(index="metric", columns="machine_type", values="cohens_d").round(6)

# ── 7. Per-machine-id best metric ────────────────────────────────────────────
per_id_best = (
    per_id_df[per_id_df["metric"].isin(["normalized_euclidean", "normalized_manhattan"])]
    .sort_values("roc_auc", ascending=False)
    .groupby(["machine_type", "machine_id"])
    .first()
    .reset_index()[["machine_type", "machine_id", "metric", "roc_auc", "cohens_d", "n_normal", "n_abnormal"]]
)

# ── 8. Build comparison report dict ──────────────────────────────────────────
report = {
    "experiment": "phase9",
    "analysis":   "cross_machine_comparison",
    "source_files": {
        "pump":   str(CSV_MAP["pump"]),
        "fan":    str(CSV_MAP["fan"]),
        "slider": str(CSV_MAP["slider"]),
        "valve":  str(CSV_MAP["valve"]),
        "summary": str(SUMMARY_JSON),
    },
    "dataset_sizes": {
        mt: {
            "n_normal":   int(summary["results"][mt]["n_normal"]),
            "n_abnormal": int(summary["results"][mt]["n_abnormal"]),
            "total":      int(summary["results"][mt]["n_normal"] + summary["results"][mt]["n_abnormal"]),
        }
        for mt in ["fan", "pump", "slider", "valve"]
    },
    "overall_all_machines": {
        row["metric"]: {"roc_auc": row["roc_auc_ref"], "cohens_d": row["cohens_d_ref"]}
        for _, row in overall_df.iterrows()
    },
    "per_machine_type": {
        mt: {
            metric: {
                "roc_auc":  summary["results"][mt]["overall_metrics"][metric]["roc_auc"],
                "cohens_d": summary["results"][mt]["overall_metrics"][metric]["cohens_d"],
            }
            for metric in METRICS
        }
        for mt in ["fan", "pump", "slider", "valve"]
    },
    "per_machine_id": {
        mt: {
            mid: {
                metric: {
                    "roc_auc":  summary["results"][mt]["per_id_metrics"][mid][metric]["roc_auc"],
                    "cohens_d": summary["results"][mt]["per_id_metrics"][mid][metric]["cohens_d"],
                }
                for metric in METRICS
            }
            for mid in summary["results"][mt]["per_id_metrics"]
        }
        for mt in ["fan", "pump", "slider", "valve"]
    },
    "rankings": {
        "best_roc_auc_overall": {
            "machine_type": best_auc["machine_type"],
            "metric":       best_auc["metric"],
            "roc_auc":      round(float(best_auc["roc_auc"]), 6),
        },
        "worst_roc_auc_overall": {
            "machine_type": worst_auc["machine_type"],
            "metric":       worst_auc["metric"],
            "roc_auc":      round(float(worst_auc["roc_auc"]), 6),
        },
        "best_cohens_d_overall": {
            "machine_type": best_d["machine_type"],
            "metric":       best_d["metric"],
            "cohens_d":     round(float(best_d["cohens_d"]), 6),
        },
        "worst_cohens_d_overall": {
            "machine_type": worst_d["machine_type"],
            "metric":       worst_d["metric"],
            "cohens_d":     round(float(worst_d["cohens_d"]), 6),
        },
        "best_euclidean_manhattan_auc": {
            "machine_type": best_auc_euc_man["machine_type"],
            "metric":       best_auc_euc_man["metric"],
            "roc_auc":      round(float(best_auc_euc_man["roc_auc"]), 6),
        },
        "worst_euclidean_manhattan_auc": {
            "machine_type": worst_auc_euc_man["machine_type"],
            "metric":       worst_auc_euc_man["metric"],
            "roc_auc":      round(float(worst_auc_euc_man["roc_auc"]), 6),
        },
    },
    "metric_comparison_notes": {
        "euclidean_vs_manhattan": (
            "Euclidean and Manhattan track almost identically across all machine types. "
            "AUC difference is always < 0.005. Cohen's d difference is always < 0.02. "
            "Either metric is a reliable anomaly discriminator."
        ),
        "cosine_reliability": (
            "Normalized cosine is unreliable as an anomaly detector. "
            "AUC hovers near 0.50–0.56 for fan/pump/slider/valve overall, "
            "indicating near-random discrimination. "
            "Exception: valve id_00 cosine AUC = 0.924 (inverted — abnormal scores lower). "
            "Cosine should NOT be used as the primary detection metric."
        ),
        "recommended_metric": "normalized_euclidean or normalized_manhattan",
    },
}

# ── 9. Save all outputs ───────────────────────────────────────────────────────
# 9a. Full comparison JSON
with open(OUT_DIR / "cross_machine_comparison.json", "w") as f:
    json.dump(report, f, indent=2)

# 9b. Machine-type AUC/d table
machine_df.to_csv(OUT_DIR / "machine_type_metrics.csv", index=False)

# 9c. Per-machine-id table
per_id_df.to_csv(OUT_DIR / "per_machine_id_metrics.csv", index=False)

# 9d. Overall across all machines
overall_df.to_csv(OUT_DIR / "overall_all_machines.csv", index=False)

# 9e. AUC pivot
pivot_auc.to_csv(OUT_DIR / "pivot_roc_auc.csv")

# 9f. Cohen's d pivot
pivot_d.to_csv(OUT_DIR / "pivot_cohens_d.csv")

# 9g. Per-machine-id best metric
per_id_best.to_csv(OUT_DIR / "per_machine_id_best_metric.csv", index=False)

# 9h. Rankings CSV
rankings_rows = []
for mt in ["fan", "pump", "slider", "valve"]:
    for metric in METRICS:
        ref = summary["results"][mt]["overall_metrics"][metric]
        rankings_rows.append({
            "machine_type": mt,
            "metric":       metric,
            "roc_auc":      ref["roc_auc"],
            "cohens_d":     ref["cohens_d"],
        })
rankings_df = pd.DataFrame(rankings_rows).sort_values("roc_auc", ascending=False)
rankings_df.to_csv(OUT_DIR / "rankings_by_roc_auc.csv", index=False)

# ── 10. Print summary report ──────────────────────────────────────────────────
print("=" * 70)
print("PHASE 9 — STEP 9: CROSS-MACHINE ANALYSIS AND COMPARISON")
print("=" * 70)

print("\n1. ROC-AUC BY MACHINE TYPE (overall, all IDs combined)")
print("-" * 60)
print(f"{'Machine':<10} {'Euclidean':>12} {'Manhattan':>12} {'Cosine':>12}")
print("-" * 60)
for mt in ["fan", "pump", "slider", "valve"]:
    r = summary["results"][mt]["overall_metrics"]
    print(f"{mt:<10} {r['normalized_euclidean']['roc_auc']:>12.6f} "
          f"{r['normalized_manhattan']['roc_auc']:>12.6f} "
          f"{r['normalized_cosine']['roc_auc']:>12.6f}")
r = summary["results"]["overall"]["overall_metrics"]
print("-" * 60)
print(f"{'ALL':10} {r['normalized_euclidean']['roc_auc']:>12.6f} "
      f"{r['normalized_manhattan']['roc_auc']:>12.6f} "
      f"{r['normalized_cosine']['roc_auc']:>12.6f}")

print("\n2. COHEN'S D BY MACHINE TYPE (overall, all IDs combined)")
print("-" * 60)
print(f"{'Machine':<10} {'Euclidean':>12} {'Manhattan':>12} {'Cosine':>12}")
print("-" * 60)
for mt in ["fan", "pump", "slider", "valve"]:
    r = summary["results"][mt]["overall_metrics"]
    print(f"{mt:<10} {r['normalized_euclidean']['cohens_d']:>12.6f} "
          f"{r['normalized_manhattan']['cohens_d']:>12.6f} "
          f"{r['normalized_cosine']['cohens_d']:>12.6f}")
r = summary["results"]["overall"]["overall_metrics"]
print("-" * 60)
print(f"{'ALL':10} {r['normalized_euclidean']['cohens_d']:>12.6f} "
      f"{r['normalized_manhattan']['cohens_d']:>12.6f} "
      f"{r['normalized_cosine']['cohens_d']:>12.6f}")

print("\n3. EUCLIDEAN vs MANHATTAN vs COSINE — KEY OBSERVATIONS")
print("-" * 60)
print("  Euclidean and Manhattan are nearly identical across all machines.")
print("  Max AUC delta between them: ", end="")
max_delta = max(
    abs(summary["results"][mt]["overall_metrics"]["normalized_euclidean"]["roc_auc"] -
        summary["results"][mt]["overall_metrics"]["normalized_manhattan"]["roc_auc"])
    for mt in ["fan", "pump", "slider", "valve"]
)
print(f"{max_delta:.6f}")
print("  Cosine AUC is near-random (0.50–0.56) for all machine types overall.")
print("  Cosine Cohen's d is negative for fan/pump/slider — abnormal scores LOWER.")
print("  Cosine is NOT a reliable anomaly detector for this system.")

print("\n4. BEST AND WORST PERFORMING COMBINATIONS")
print("-" * 60)
print(f"  Best  ROC-AUC  : {best_auc['machine_type']:8s} / {best_auc['metric']:28s}  AUC={best_auc['roc_auc']:.6f}")
print(f"  Worst ROC-AUC  : {worst_auc['machine_type']:8s} / {worst_auc['metric']:28s}  AUC={worst_auc['roc_auc']:.6f}")
print(f"  Best  Cohen's d: {best_d['machine_type']:8s} / {best_d['metric']:28s}  d={best_d['cohens_d']:.6f}")
print(f"  Worst Cohen's d: {worst_d['machine_type']:8s} / {worst_d['metric']:28s}  d={worst_d['cohens_d']:.6f}")
print()
print(f"  Best  (Euc/Man only) AUC: {best_auc_euc_man['machine_type']:8s} / {best_auc_euc_man['metric']:28s}  AUC={best_auc_euc_man['roc_auc']:.6f}")
print(f"  Worst (Euc/Man only) AUC: {worst_auc_euc_man['machine_type']:8s} / {worst_auc_euc_man['metric']:28s}  AUC={worst_auc_euc_man['roc_auc']:.6f}")

print("\n5. OVERALL RESULTS ACROSS ALL MACHINES")
print("-" * 60)
r = summary["results"]["overall"]["overall_metrics"]
print(f"  Total normal   : {summary['results']['overall']['n_normal']}")
print(f"  Total abnormal : {summary['results']['overall']['n_abnormal']}")
print(f"  Euclidean  AUC={r['normalized_euclidean']['roc_auc']:.6f}  d={r['normalized_euclidean']['cohens_d']:.6f}")
print(f"  Manhattan  AUC={r['normalized_manhattan']['roc_auc']:.6f}  d={r['normalized_manhattan']['cohens_d']:.6f}")
print(f"  Cosine     AUC={r['normalized_cosine']['roc_auc']:.6f}  d={r['normalized_cosine']['cohens_d']:.6f}")

print("\n  Machine ranking by Euclidean AUC (best → worst):")
for i, (mt, auc) in enumerate(
    sorted(
        [(mt, summary["results"][mt]["overall_metrics"]["normalized_euclidean"]["roc_auc"])
         for mt in ["fan", "pump", "slider", "valve"]],
        key=lambda x: -x[1]
    ), 1
):
    d = summary["results"][mt]["overall_metrics"]["normalized_euclidean"]["cohens_d"]
    print(f"    {i}. {mt:<8}  AUC={auc:.6f}  d={d:.6f}")

print("\n  Per-machine-id best AUC (Euclidean/Manhattan):")
print(f"  {'Machine':8s} {'ID':6s} {'Metric':28s} {'AUC':>10} {'d':>10}")
print("  " + "-" * 66)
for _, row in per_id_best.sort_values("roc_auc", ascending=False).iterrows():
    print(f"  {row['machine_type']:8s} {row['machine_id']:6s} {row['metric']:28s} "
          f"{row['roc_auc']:>10.6f} {row['cohens_d']:>10.6f}")

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}")
print()
