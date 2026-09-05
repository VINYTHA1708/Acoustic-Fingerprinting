"""Phase 14 — Final Tables and Figures (publication-ready)."""
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

OUT = pathlib.Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

# ── Exact values from Phases 7-13 ────────────────────────────────────────────

# Table 1: Performance comparison (E1 pump, consolidated_comparison.csv)
BASELINE_DATA = [
    ("B1 — Raw MFCC Distance",              [0.5068, 0.6670, 0.5816, 0.5416]),
    ("B2 — Statistical Feature Distance",   [0.4581, 0.6316, 0.5441, 0.5265]),
    ("B3 — Non-Contrastive Projection",     [0.5198, 0.5145, 0.5228, 0.5442]),
    ("Ours — Contrastive Fingerprinting",   [0.7836, 0.8046, 0.9578, 0.6851]),
]
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]

# Table 2: Multi-machine evaluation (phase9 final_method_config.json)
MULTI_MACHINE = {
    "fan":    {"roc_auc": 0.698577, "cohens_d": 0.738717,
               "ci_lo": 0.675629, "ci_hi": 0.721035},
    "pump":   {"roc_auc": 0.863535, "cohens_d": 1.424665,
               "ci_lo": 0.843819, "ci_hi": 0.886121},
    "slider": {"roc_auc": 0.881314, "cohens_d": 1.487442,
               "ci_lo": 0.866394, "ci_hi": 0.902679},
    "valve":  {"roc_auc": 0.828308, "cohens_d": 1.275274,
               "ci_lo": 0.810703, "ci_hi": 0.858098},
}
OVERALL_PHASE9 = {"roc_auc": 0.787522, "cohens_d": 1.061325,
                  "ci_lo": 0.777277, "ci_hi": 0.801181}

# Table 3: Ablation (ablation_results.csv — pump machine, mean over IDs)
ABLATION_RAW = {
    "Full Method (DSP+BEATs+Contrastive)": [0.7836, 0.8046, 0.9578, 0.6851],
    "A1 — No BEATs (DSP-only)":            [0.6862, 0.6255, 0.9592, 0.7191],
    "A2 — No DSP (BEATs-only)":            [0.8683, 0.8699, 0.9502, 0.7318],
    "A3 — No Contrastive Training":        [0.5586, 0.6550, 0.5852, 0.5764],
    "A4 — No ProjectionHead":              [0.5068, 0.6670, 0.5816, 0.5416],
}

# Table 4: Statistical results (step10_bootstrap_ci.csv + significance_tests)
STAT_DATA = [
    ("fan",    "norm_euclidean", 0.698507, 0.675629, 0.721035),
    ("pump",   "norm_euclidean", 0.865186, 0.843819, 0.886121),
    ("slider", "norm_euclidean", 0.884319, 0.866394, 0.902679),
    ("valve",  "norm_euclidean", 0.834061, 0.810703, 0.858098),
    ("overall","norm_euclidean", 0.789578, 0.777277, 0.801181),
]

# Table 5: Multi-seed stability (phase11_results.csv)
SEED_DATA = [
    (42,   0.789649, 1.068191),
    (123,  0.775694, 1.010870),
    (2026, 0.775575, 1.029714),
]
SEED_MEAN = (0.780306, 1.036258)
SEED_STD  = (0.008091, 0.029216)

# Table 6: Runtime benchmark (phase12_timing_summary.csv + benchmark_results.json)
RUNTIME_DATA = [
    ("Profile build (per recording)", "9.6 ms",  "150 recordings, pump/id_00"),
    ("Inference (cache-hit, mean)",   "5.86 ms", "30 recordings, pump/id_00"),
    ("Inference (cache-hit, median)", "5.40 ms", "30 recordings, pump/id_00"),
    ("Evaluation (per recording)",    "8.92 ms", "1022 recordings, pump"),
    ("Evaluation (1022 recs total)",  "9.118 s", "pump machine type"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
FONT = {"family": "DejaVu Sans", "size": 10}
matplotlib.rc("font", **FONT)
TITLE_SIZE = 12
LABEL_SIZE = 10
TICK_SIZE  = 9

def _save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIGURE] {p.name}")
    return p


# ── Table 1: Performance comparison ──────────────────────────────────────────
def make_table1():
    rows = []
    for method, aucs in BASELINE_DATA:
        mean_auc = np.mean(aucs)
        rows.append({
            "Method": method,
            "id_00 AUC": aucs[0],
            "id_02 AUC": aucs[1],
            "id_04 AUC": aucs[2],
            "id_06 AUC": aucs[3],
            "Mean AUC":  round(mean_auc, 4),
        })
    df = pd.DataFrame(rows)
    p = OUT / "table1_performance_comparison.csv"
    df.to_csv(p, index=False)
    print(f"  [TABLE]  {p.name}")
    return df


# ── Table 2: Multi-machine evaluation ────────────────────────────────────────
def make_table2():
    rows = []
    for mtype, v in MULTI_MACHINE.items():
        rows.append({
            "Machine Type": mtype.capitalize(),
            "ROC-AUC":      round(v["roc_auc"], 4),
            "95% CI Low":   round(v["ci_lo"], 4),
            "95% CI High":  round(v["ci_hi"], 4),
            "Cohen's d":    round(v["cohens_d"], 4),
        })
    rows.append({
        "Machine Type": "Overall",
        "ROC-AUC":      round(OVERALL_PHASE9["roc_auc"], 4),
        "95% CI Low":   round(OVERALL_PHASE9["ci_lo"], 4),
        "95% CI High":  round(OVERALL_PHASE9["ci_hi"], 4),
        "Cohen's d":    round(OVERALL_PHASE9["cohens_d"], 4),
    })
    df = pd.DataFrame(rows)
    p = OUT / "table2_multi_machine_evaluation.csv"
    df.to_csv(p, index=False)
    print(f"  [TABLE]  {p.name}")
    return df


# ── Table 3: Ablation ─────────────────────────────────────────────────────────
def make_table3():
    rows = []
    for method, aucs in ABLATION_RAW.items():
        mean_auc = np.mean(aucs)
        full_mean = np.mean(ABLATION_RAW["Full Method (DSP+BEATs+Contrastive)"])
        delta = round(mean_auc - full_mean, 4)
        rows.append({
            "Configuration":  method,
            "id_00 AUC":      aucs[0],
            "id_02 AUC":      aucs[1],
            "id_04 AUC":      aucs[2],
            "id_06 AUC":      aucs[3],
            "Mean AUC":       round(mean_auc, 4),
            "Delta vs Full":  delta,
        })
    df = pd.DataFrame(rows)
    p = OUT / "table3_ablation_study.csv"
    df.to_csv(p, index=False)
    print(f"  [TABLE]  {p.name}")
    return df


# ── Table 4: Statistical results ──────────────────────────────────────────────
def make_table4():
    rows = []
    for scope, metric, auc, ci_lo, ci_hi in STAT_DATA:
        rows.append({
            "Scope":       scope.capitalize(),
            "Metric":      metric,
            "AUC":         round(auc, 4),
            "95% CI":      f"[{ci_lo:.4f}, {ci_hi:.4f}]",
            "CI Width":    round(ci_hi - ci_lo, 4),
            "Significant": "Yes" if ci_lo > 0.5 else "No",
        })
    df = pd.DataFrame(rows)
    p = OUT / "table4_statistical_results.csv"
    df.to_csv(p, index=False)
    print(f"  [TABLE]  {p.name}")
    return df


# ── Table 5: Multi-seed stability ─────────────────────────────────────────────
def make_table5():
    rows = []
    for seed, auc, cd in SEED_DATA:
        rows.append({"Seed": seed, "ROC-AUC": auc, "Cohen's d": cd})
    rows.append({"Seed": "Mean", "ROC-AUC": SEED_MEAN[0], "Cohen's d": SEED_MEAN[1]})
    rows.append({"Seed": "Std",  "ROC-AUC": SEED_STD[0],  "Cohen's d": SEED_STD[1]})
    df = pd.DataFrame(rows)
    p = OUT / "table5_seed_stability.csv"
    df.to_csv(p, index=False)
    print(f"  [TABLE]  {p.name}")
    return df


# ── Table 6: Runtime benchmark ────────────────────────────────────────────────
def make_table6():
    rows = [{"Stage": s, "Time": t, "Notes": n} for s, t, n in RUNTIME_DATA]
    df = pd.DataFrame(rows)
    p = OUT / "table6_runtime_benchmark.csv"
    df.to_csv(p, index=False)
    print(f"  [TABLE]  {p.name}")
    return df


# ── Figure 1: Overall metric comparison (bar chart) ───────────────────────────
def make_fig1(df1):
    methods = df1["Method"].tolist()
    short = ["B1\nMFCC", "B2\nStat", "B3\nRand\nProj", "Ours\nContr."]
    means = df1["Mean AUC"].tolist()
    colors = ["#aec6cf", "#aec6cf", "#aec6cf", "#2c7bb6"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(short, means, color=colors, edgecolor="black", linewidth=0.7, width=0.55)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.9, label="Random (AUC=0.5)")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=TICK_SIZE, fontweight="bold")
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("Mean ROC-AUC (pump, 4 machine IDs)", fontsize=LABEL_SIZE)
    ax.set_title("Figure 1 — Overall Method Comparison (E1 Pump)", fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=TICK_SIZE)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _save(fig, "fig1_overall_metric_comparison.png")


# ── Figure 2: Per-machine-type performance ────────────────────────────────────
def make_fig2():
    types  = ["Fan", "Pump", "Slider", "Valve", "Overall"]
    aucs   = [MULTI_MACHINE["fan"]["roc_auc"],
              MULTI_MACHINE["pump"]["roc_auc"],
              MULTI_MACHINE["slider"]["roc_auc"],
              MULTI_MACHINE["valve"]["roc_auc"],
              OVERALL_PHASE9["roc_auc"]]
    ci_lo  = [MULTI_MACHINE["fan"]["ci_lo"],
              MULTI_MACHINE["pump"]["ci_lo"],
              MULTI_MACHINE["slider"]["ci_lo"],
              MULTI_MACHINE["valve"]["ci_lo"],
              OVERALL_PHASE9["ci_lo"]]
    ci_hi  = [MULTI_MACHINE["fan"]["ci_hi"],
              MULTI_MACHINE["pump"]["ci_hi"],
              MULTI_MACHINE["slider"]["ci_hi"],
              MULTI_MACHINE["valve"]["ci_hi"],
              OVERALL_PHASE9["ci_hi"]]
    yerr_lo = [a - l for a, l in zip(aucs, ci_lo)]
    yerr_hi = [h - a for a, h in zip(aucs, ci_hi)]
    colors = ["#4dac26", "#2c7bb6", "#d7191c", "#fdae61", "#555555"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(types))
    bars = ax.bar(x, aucs, color=colors, edgecolor="black", linewidth=0.7, width=0.55,
                  yerr=[yerr_lo, yerr_hi], capsize=5, error_kw={"elinewidth": 1.2})
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.9, label="Random baseline")
    for xi, val in zip(x, aucs):
        ax.text(xi, val + max(yerr_hi) + 0.012, f"{val:.3f}",
                ha="center", va="bottom", fontsize=TICK_SIZE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=TICK_SIZE)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("ROC-AUC (95% CI)", fontsize=LABEL_SIZE)
    ax.set_title("Figure 2 — Per-Machine-Type Performance (Phase 9, seed=42)", fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _save(fig, "fig2_per_machine_type_performance.png")


# ── Figure 3: Ablation comparison ────────────────────────────────────────────
def make_fig3(df3):
    configs = df3["Configuration"].tolist()
    short = ["Full\nMethod", "A1\nNo BEATs", "A2\nNo DSP", "A3\nNo\nContr.", "A4\nNo\nProjHead"]
    means = df3["Mean AUC"].tolist()
    colors = ["#2c7bb6", "#fdae61", "#fdae61", "#d7191c", "#d7191c"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(short, means, color=colors, edgecolor="black", linewidth=0.7, width=0.55)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.9, label="Random (AUC=0.5)")
    ax.axhline(means[0], color="#2c7bb6", linestyle=":", linewidth=1.2, label=f"Full method ({means[0]:.3f})")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=TICK_SIZE, fontweight="bold")
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("Mean ROC-AUC (pump, 4 machine IDs)", fontsize=LABEL_SIZE)
    ax.set_title("Figure 3 — Ablation Study Comparison", fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return _save(fig, "fig3_ablation_comparison.png")


# ── Figure 4: Multi-seed stability ───────────────────────────────────────────
def make_fig4():
    seeds = [42, 123, 2026]
    aucs  = [s[1] for s in SEED_DATA]
    cds   = [s[2] for s in SEED_DATA]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # AUC panel
    ax = axes[0]
    ax.bar([str(s) for s in seeds], aucs, color=["#2c7bb6", "#4dac26", "#d7191c"],
           edgecolor="black", linewidth=0.7, width=0.5)
    ax.axhline(SEED_MEAN[0], color="black", linestyle="--", linewidth=1.2,
               label=f"Mean={SEED_MEAN[0]:.4f}")
    ax.fill_between([-0.5, 2.5],
                    SEED_MEAN[0] - SEED_STD[0], SEED_MEAN[0] + SEED_STD[0],
                    alpha=0.15, color="black", label=f"±1 SD ({SEED_STD[0]:.4f})")
    for i, val in enumerate(aucs):
        ax.text(i, val + 0.002, f"{val:.4f}", ha="center", va="bottom",
                fontsize=TICK_SIZE, fontweight="bold")
    ax.set_ylim(0.75, 0.82)
    ax.set_xlabel("Random Seed", fontsize=LABEL_SIZE)
    ax.set_ylabel("ROC-AUC", fontsize=LABEL_SIZE)
    ax.set_title("ROC-AUC Across Seeds", fontsize=TITLE_SIZE - 1, fontweight="bold")
    ax.legend(fontsize=TICK_SIZE - 1)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Cohen's d panel
    ax = axes[1]
    ax.bar([str(s) for s in seeds], cds, color=["#2c7bb6", "#4dac26", "#d7191c"],
           edgecolor="black", linewidth=0.7, width=0.5)
    ax.axhline(SEED_MEAN[1], color="black", linestyle="--", linewidth=1.2,
               label=f"Mean={SEED_MEAN[1]:.4f}")
    ax.fill_between([-0.5, 2.5],
                    SEED_MEAN[1] - SEED_STD[1], SEED_MEAN[1] + SEED_STD[1],
                    alpha=0.15, color="black", label=f"±1 SD ({SEED_STD[1]:.4f})")
    for i, val in enumerate(cds):
        ax.text(i, val + 0.003, f"{val:.4f}", ha="center", va="bottom",
                fontsize=TICK_SIZE, fontweight="bold")
    ax.set_ylim(0.95, 1.12)
    ax.set_xlabel("Random Seed", fontsize=LABEL_SIZE)
    ax.set_ylabel("Cohen's d", fontsize=LABEL_SIZE)
    ax.set_title("Cohen's d Across Seeds", fontsize=TITLE_SIZE - 1, fontweight="bold")
    ax.legend(fontsize=TICK_SIZE - 1)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Figure 4 — Multi-Seed Stability (Phase 11, seeds 42 / 123 / 2026)",
                 fontsize=TITLE_SIZE, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig4_seed_stability.png")


# ── Summary markdown ──────────────────────────────────────────────────────────
def make_summary(tables, figures):
    lines = [
        "# Phase 14 — Final Tables and Figures",
        "",
        "Publication-ready tables and figures generated from exact Phase 7–13 results.",
        "No values were invented or estimated.",
        "",
        "## Tables",
        "",
        "| # | File | Description |",
        "|---|------|-------------|",
        "| 1 | table1_performance_comparison.csv | Method comparison vs baselines (E1 pump, 4 IDs) |",
        "| 2 | table2_multi_machine_evaluation.csv | Per-machine-type AUC + 95% CI + Cohen's d |",
        "| 3 | table3_ablation_study.csv | Ablation: component contribution (pump, 4 IDs) |",
        "| 4 | table4_statistical_results.csv | AUC + 95% bootstrap CI per machine type |",
        "| 5 | table5_seed_stability.csv | Multi-seed stability (seeds 42, 123, 2026) |",
        "| 6 | table6_runtime_benchmark.csv | CPU runtime benchmark (cache-hit path) |",
        "",
        "## Figures",
        "",
        "| # | File | Description |",
        "|---|------|-------------|",
        "| 1 | fig1_overall_metric_comparison.png | Bar chart: method vs baselines (mean AUC) |",
        "| 2 | fig2_per_machine_type_performance.png | Bar chart: AUC per machine type with 95% CI |",
        "| 3 | fig3_ablation_comparison.png | Bar chart: ablation component analysis |",
        "| 4 | fig4_seed_stability.png | Dual panel: AUC and Cohen's d across 3 seeds |",
        "",
        "## Key Results",
        "",
        f"- **Overall ROC-AUC** (Phase 9, seed=42): {OVERALL_PHASE9['roc_auc']:.4f}",
        f"  95% CI: [{OVERALL_PHASE9['ci_lo']:.4f}, {OVERALL_PHASE9['ci_hi']:.4f}]",
        f"- **Overall Cohen's d**: {OVERALL_PHASE9['cohens_d']:.4f}",
        f"- **Seed stability** (3 seeds): AUC = {SEED_MEAN[0]:.4f} ± {SEED_STD[0]:.4f}",
        f"- **Best machine type**: Slider (AUC={MULTI_MACHINE['slider']['roc_auc']:.4f})",
        f"- **Inference latency** (cache-hit): {5.86:.2f} ms mean",
        "",
        "## Data Sources",
        "",
        "| Table/Figure | Source file(s) |",
        "|---|---|",
        "| Table 1, Fig 1 | experiments/results/e1/baseline_comparison/consolidated_comparison.csv |",
        "| Table 2, Fig 2 | experiments/results/phase13/final_method_config.json |",
        "| Table 3, Fig 3 | experiments/results/e1/ablation_study/ablation_results.csv |",
        "| Table 4       | experiments/results/phase9/comparison_e1/step10_bootstrap_ci.csv |",
        "| Table 5, Fig 4 | experiments/results/phase11/phase11_results.csv |",
        "| Table 6       | experiments/results/phase12/phase12_timing_summary.csv |",
    ]
    p = OUT / "phase14_summary.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [SUMMARY] {p.name}")
    return p


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("PHASE 14 — Final Tables and Figures")
    print("=" * 60)

    print("\n[1/6] Generating tables...")
    df1 = make_table1()
    df2 = make_table2()
    df3 = make_table3()
    df4 = make_table4()
    df5 = make_table5()
    df6 = make_table6()

    print("\n[2/6] Generating figures...")
    f1 = make_fig1(df1)
    f2 = make_fig2()
    f3 = make_fig3(df3)
    f4 = make_fig4()

    print("\n[3/6] Writing summary...")
    make_summary(
        [df1, df2, df3, df4, df5, df6],
        [f1, f2, f3, f4],
    )

    print("\n[4/6] Verifying output files...")
    expected = [
        "table1_performance_comparison.csv",
        "table2_multi_machine_evaluation.csv",
        "table3_ablation_study.csv",
        "table4_statistical_results.csv",
        "table5_seed_stability.csv",
        "table6_runtime_benchmark.csv",
        "fig1_overall_metric_comparison.png",
        "fig2_per_machine_type_performance.png",
        "fig3_ablation_comparison.png",
        "fig4_seed_stability.png",
        "phase14_summary.md",
    ]
    all_ok = True
    for name in expected:
        p = OUT / name
        exists = p.exists()
        size   = p.stat().st_size if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  {status:7s}  {name}  ({size:,} bytes)")
        if not exists:
            all_ok = False

    print("\n" + "=" * 60)
    print("PHASE 14 COMPLETE" if all_ok else "PHASE 14 COMPLETED WITH ERRORS")
    print("=" * 60)

    print("\n── FINAL SUMMARY ──────────────────────────────────────────")
    print("\nTABLES CREATED:")
    print("  1. table1_performance_comparison.csv  — Method vs baselines (pump, 4 IDs)")
    print("  2. table2_multi_machine_evaluation.csv — AUC + 95% CI + Cohen's d per type")
    print("  3. table3_ablation_study.csv           — Ablation component analysis")
    print("  4. table4_statistical_results.csv      — AUC + bootstrap 95% CI")
    print("  5. table5_seed_stability.csv           — Seeds 42/123/2026 stability")
    print("  6. table6_runtime_benchmark.csv        — CPU runtime (cache-hit path)")
    print("\nFIGURES CREATED:")
    print("  1. fig1_overall_metric_comparison.png  — Bar: method vs baselines")
    print("  2. fig2_per_machine_type_performance.png — Bar: per-type AUC + CI")
    print("  3. fig3_ablation_comparison.png        — Bar: ablation components")
    print("  4. fig4_seed_stability.png             — Dual panel: AUC + Cohen's d")
    print("\nSUMMARY:")
    print("  phase14_summary.md")
    print(f"\nAll outputs in: {OUT}")


if __name__ == "__main__":
    main()
