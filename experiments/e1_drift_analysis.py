"""Experiment E1 — Healthy vs Abnormal Drift Analysis (Phase 5).

Analysis-only script. Reads the pre-computed evaluation_results.csv and
validates its structure. Statistical comparisons and plots are added in
later steps.

Usage:
    python experiments/e1_drift_analysis.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "E1"
STAGE = "Healthy vs Abnormal Drift Analysis"

INPUT_CSV = Path("experiments/results/e1/evaluation_results.csv")
OUTPUT_DIR = Path("experiments/results/e1/drift_analysis")
OVERALL_STATS_CSV = OUTPUT_DIR / "overall_drift_statistics.csv"
PER_MACHINE_STATS_CSV = OUTPUT_DIR / "per_machine_drift_statistics.csv"
OVERALL_SIG_CSV = OUTPUT_DIR / "overall_significance.csv"
PER_MACHINE_SIG_CSV = OUTPUT_DIR / "per_machine_significance.csv"
OVERALL_EFFECT_CSV = OUTPUT_DIR / "overall_effect_sizes.csv"
PER_MACHINE_EFFECT_CSV = OUTPUT_DIR / "per_machine_effect_sizes.csv"
PLOTS_DIR = OUTPUT_DIR / "plots"
OVERALL_SUMMARY_CSV = OUTPUT_DIR / "overall_results_summary.csv"
PER_MACHINE_SUMMARY_CSV = OUTPUT_DIR / "per_machine_results_summary.csv"

REQUIRED_COLUMNS = {
    "machine_type", "machine_id", "filename", "true_label",
    "health_score", "health_percentage", "health_state",
    "normalized_euclidean", "normalized_manhattan", "normalized_cosine",
}

DRIFT_METRICS = ["normalized_euclidean", "normalized_manhattan", "normalized_cosine"]
VALID_LABELS = {"normal", "abnormal"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_csv(df: pd.DataFrame) -> None:
    """Validate the evaluation CSV before any analysis.

    Raises:
        ValueError: On any structural or content problem.
    """
    if df.empty:
        raise ValueError("CSV is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    unexpected = set(df["true_label"].unique()) - VALID_LABELS
    if unexpected:
        raise ValueError(f"Unexpected true_label values: {unexpected}")

    for col in DRIFT_METRICS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is not numeric.")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    """Load and validate the evaluation CSV.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If validation fails.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {path}\n"
            "Run experiments/e1_evaluate.py first."
        )
    df = pd.read_csv(path)
    validate_csv(df)
    return df


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "=", width: int = 50) -> None:
    print(char * width)


def print_header() -> None:
    _sep()
    print(f"Experiment ID : {EXPERIMENT_ID}")
    print(f"Stage         : {STAGE}")
    _sep()
    print()


def print_dataset_summary(df: pd.DataFrame) -> None:
    n_normal = int((df["true_label"] == "normal").sum())
    n_abnormal = int((df["true_label"] == "abnormal").sum())
    machine_ids = sorted(df["machine_id"].unique())

    print(f"Input  : {INPUT_CSV}")
    print()
    print(f"Total recordings   : {len(df)}")
    print(f"Normal recordings  : {n_normal}")
    print(f"Abnormal recordings: {n_abnormal}")
    print()
    print("Machine IDs:")
    for mid in machine_ids:
        print(f"  {mid}")
    print()


# ---------------------------------------------------------------------------
# Phase 5.2 — Overall Healthy vs Abnormal Drift Statistics
# ---------------------------------------------------------------------------

_STAT_FUNCS = {
    "count": "count",
    "mean": "mean",
    "std": "std",
    "median": "median",
    "min": "min",
    "max": "max",
    "q1": lambda x: x.quantile(0.25),
    "q3": lambda x: x.quantile(0.75),
}


def compute_overall_drift_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per (metric, label) combination."""
    rows = []
    for metric in DRIFT_METRICS:
        for label in ("normal", "abnormal"):
            subset = df.loc[df["true_label"] == label, metric]
            row = {"metric": metric, "label": label}
            for stat_name, func in _STAT_FUNCS.items():
                row[stat_name] = subset.agg(func) if isinstance(func, str) else func(subset)
            rows.append(row)
    return pd.DataFrame(rows)


def save_overall_drift_statistics(stats: pd.DataFrame, path: Path) -> None:
    """Create output directory and save statistics CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(path, index=False)


def print_overall_drift_statistics(stats: pd.DataFrame) -> None:
    _sep()
    print("Overall Healthy vs Abnormal Drift Statistics")
    _sep()
    for metric in DRIFT_METRICS:
        print(f"\n  {metric}")
        subset = stats[stats["metric"] == metric]
        for _, row in subset.iterrows():
            label = row["label"].capitalize()
            print(
                f"    {label:<10}  count={int(row['count']):>4}  "
                f"mean={row['mean']:>9.4f}  std={row['std']:>9.4f}  "
                f"median={row['median']:>9.4f}  "
                f"min={row['min']:>9.4f}  max={row['max']:>9.4f}  "
                f"Q1={row['q1']:>9.4f}  Q3={row['q3']:>9.4f}"
            )
    print()


# ---------------------------------------------------------------------------
# Phase 5.3 — Per-Machine Drift Statistics
# ---------------------------------------------------------------------------

def compute_per_machine_drift_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per (machine_id, metric, label) combination."""
    rows = []
    for machine_id in sorted(df["machine_id"].unique()):
        machine_df = df[df["machine_id"] == machine_id]
        for metric in DRIFT_METRICS:
            for label in ("normal", "abnormal"):
                subset = machine_df.loc[machine_df["true_label"] == label, metric]
                if subset.empty:
                    continue
                row = {"machine_id": machine_id, "metric": metric, "label": label}
                for stat_name, func in _STAT_FUNCS.items():
                    row[stat_name] = subset.agg(func) if isinstance(func, str) else func(subset)
                rows.append(row)
    return pd.DataFrame(rows)


def save_per_machine_drift_statistics(stats: pd.DataFrame, path: Path) -> None:
    """Create output directory and save per-machine statistics CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(path, index=False)


def print_per_machine_drift_statistics(stats: pd.DataFrame) -> None:
    _sep()
    print("Per-Machine Drift Statistics")
    _sep()
    for machine_id in sorted(stats["machine_id"].unique()):
        print(f"\n  {machine_id}")
        for metric in DRIFT_METRICS:
            print(f"    {metric}")
            subset = stats[
                (stats["machine_id"] == machine_id) & (stats["metric"] == metric)
            ]
            for _, row in subset.iterrows():
                label = row["label"].capitalize()
                print(
                    f"      {label:<10}  count={int(row['count']):>4}  "
                    f"mean={row['mean']:>9.4f}  std={row['std']:>9.4f}  "
                    f"median={row['median']:>9.4f}  "
                    f"min={row['min']:>9.4f}  max={row['max']:>9.4f}  "
                    f"Q1={row['q1']:>9.4f}  Q3={row['q3']:>9.4f}"
                )
    print()


# ---------------------------------------------------------------------------
# Phase 5.4 — Statistical Significance Testing (Mann-Whitney U)
# ---------------------------------------------------------------------------

def _mann_whitney_row(normal: pd.Series, abnormal: pd.Series) -> dict:
    """Run a two-sided Mann-Whitney U test and return a result dict."""
    stat, p = mannwhitneyu(normal, abnormal, alternative="two-sided")
    return {"u_statistic": float(stat), "p_value": float(p)}


def compute_overall_significance(df: pd.DataFrame) -> pd.DataFrame:
    """One row per metric: U statistic and p-value across all recordings."""
    rows = []
    for metric in DRIFT_METRICS:
        normal = df.loc[df["true_label"] == "normal", metric]
        abnormal = df.loc[df["true_label"] == "abnormal", metric]
        row = {"metric": metric, "n_normal": len(normal), "n_abnormal": len(abnormal)}
        row.update(_mann_whitney_row(normal, abnormal))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_per_machine_significance(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (machine_id, metric): U statistic and p-value."""
    rows = []
    for machine_id in sorted(df["machine_id"].unique()):
        mdf = df[df["machine_id"] == machine_id]
        for metric in DRIFT_METRICS:
            normal = mdf.loc[mdf["true_label"] == "normal", metric]
            abnormal = mdf.loc[mdf["true_label"] == "abnormal", metric]
            if normal.empty or abnormal.empty:
                continue
            row = {
                "machine_id": machine_id, "metric": metric,
                "n_normal": len(normal), "n_abnormal": len(abnormal),
            }
            row.update(_mann_whitney_row(normal, abnormal))
            rows.append(row)
    return pd.DataFrame(rows)


def save_significance(sig: pd.DataFrame, path: Path) -> None:
    """Create output directory and save significance CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sig.to_csv(path, index=False)


def print_overall_significance(sig: pd.DataFrame) -> None:
    _sep()
    print("Overall Statistical Significance (Mann-Whitney U, two-sided)")
    _sep()
    for _, row in sig.iterrows():
        sig_tag = "*" if row["p_value"] < 0.05 else " "
        print(
            f"  {row['metric']:<26}  "
            f"n_normal={int(row['n_normal']):>4}  n_abnormal={int(row['n_abnormal']):>4}  "
            f"U={row['u_statistic']:>12.1f}  p={row['p_value']:.4e}  {sig_tag}"
        )
    print("  (* p < 0.05)")
    print()


def print_per_machine_significance(sig: pd.DataFrame) -> None:
    _sep()
    print("Per-Machine Statistical Significance (Mann-Whitney U, two-sided)")
    _sep()
    for machine_id in sorted(sig["machine_id"].unique()):
        print(f"\n  {machine_id}")
        subset = sig[sig["machine_id"] == machine_id]
        for _, row in subset.iterrows():
            sig_tag = "*" if row["p_value"] < 0.05 else " "
            print(
                f"    {row['metric']:<26}  "
                f"n_normal={int(row['n_normal']):>4}  n_abnormal={int(row['n_abnormal']):>4}  "
                f"U={row['u_statistic']:>10.1f}  p={row['p_value']:.4e}  {sig_tag}"
            )
    print("  (* p < 0.05)")
    print()


# ---------------------------------------------------------------------------
# Phase 5.5 — Effect Size Analysis (Rank-Biserial Correlation)
# ---------------------------------------------------------------------------

def _rank_biserial(u_statistic: float, n_normal: int, n_abnormal: int) -> float:
    """Rank-biserial correlation from Mann-Whitney U: r = 1 - 2U / (n1 * n2)."""
    return 1.0 - (2.0 * u_statistic) / (n_normal * n_abnormal)


def compute_overall_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per metric: rank-biserial correlation across all recordings."""
    rows = []
    for metric in DRIFT_METRICS:
        normal = df.loc[df["true_label"] == "normal", metric]
        abnormal = df.loc[df["true_label"] == "abnormal", metric]
        n_n, n_a = len(normal), len(abnormal)
        mw = _mann_whitney_row(normal, abnormal)
        rows.append({
            "metric": metric,
            "n_normal": n_n,
            "n_abnormal": n_a,
            "rank_biserial_correlation": _rank_biserial(mw["u_statistic"], n_n, n_a),
        })
    return pd.DataFrame(rows)


def compute_per_machine_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (machine_id, metric): rank-biserial correlation."""
    rows = []
    for machine_id in sorted(df["machine_id"].unique()):
        mdf = df[df["machine_id"] == machine_id]
        for metric in DRIFT_METRICS:
            normal = mdf.loc[mdf["true_label"] == "normal", metric]
            abnormal = mdf.loc[mdf["true_label"] == "abnormal", metric]
            if normal.empty or abnormal.empty:
                continue
            n_n, n_a = len(normal), len(abnormal)
            mw = _mann_whitney_row(normal, abnormal)
            rows.append({
                "machine_id": machine_id,
                "metric": metric,
                "n_normal": n_n,
                "n_abnormal": n_a,
                "rank_biserial_correlation": _rank_biserial(mw["u_statistic"], n_n, n_a),
            })
    return pd.DataFrame(rows)


def save_effect_sizes(effect: pd.DataFrame, path: Path) -> None:
    """Create output directory and save effect sizes CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    effect.to_csv(path, index=False)


def print_overall_effect_sizes(effect: pd.DataFrame) -> None:
    _sep()
    print("Overall Effect Sizes (Rank-Biserial Correlation)")
    _sep()
    for _, row in effect.iterrows():
        print(
            f"  {row['metric']:<26}  "
            f"n_normal={int(row['n_normal']):>4}  n_abnormal={int(row['n_abnormal']):>4}  "
            f"r={row['rank_biserial_correlation']:>+.4f}"
        )
    print()


def print_per_machine_effect_sizes(effect: pd.DataFrame) -> None:
    _sep()
    print("Per-Machine Effect Sizes (Rank-Biserial Correlation)")
    _sep()
    for machine_id in sorted(effect["machine_id"].unique()):
        print(f"\n  {machine_id}")
        subset = effect[effect["machine_id"] == machine_id]
        for _, row in subset.iterrows():
            print(
                f"    {row['metric']:<26}  "
                f"n_normal={int(row['n_normal']):>4}  n_abnormal={int(row['n_abnormal']):>4}  "
                f"r={row['rank_biserial_correlation']:>+.4f}"
            )
    print()


# ---------------------------------------------------------------------------
# Phase 5.7 — Consolidated Results Summary
# ---------------------------------------------------------------------------

_SUMMARY_COLS = [
    "metric",
    "normal_count", "normal_mean", "normal_std", "normal_median",
    "abnormal_count", "abnormal_mean", "abnormal_std", "abnormal_median",
    "u_statistic", "p_value", "rank_biserial",
]


def compute_overall_results_summary(
    overall_stats: pd.DataFrame,
    overall_significance: pd.DataFrame,
    overall_effect_sizes: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Phase 5.2, 5.4, and 5.5 results into one row per metric."""
    normal = (
        overall_stats[overall_stats["label"] == "normal"]
        [["metric", "count", "mean", "std", "median"]]
        .rename(columns={"count": "normal_count", "mean": "normal_mean",
                         "std": "normal_std", "median": "normal_median"})
    )
    abnormal = (
        overall_stats[overall_stats["label"] == "abnormal"]
        [["metric", "count", "mean", "std", "median"]]
        .rename(columns={"count": "abnormal_count", "mean": "abnormal_mean",
                         "std": "abnormal_std", "median": "abnormal_median"})
    )
    sig = overall_significance[["metric", "u_statistic", "p_value"]]
    effect = overall_effect_sizes[["metric", "rank_biserial_correlation"]].rename(
        columns={"rank_biserial_correlation": "rank_biserial"}
    )
    summary = (
        normal
        .merge(abnormal, on="metric")
        .merge(sig, on="metric")
        .merge(effect, on="metric")
    )
    return summary[_SUMMARY_COLS].reset_index(drop=True)


_PM_SUMMARY_COLS = [
    "machine_id", "metric",
    "normal_count", "normal_mean", "normal_std", "normal_median",
    "abnormal_count", "abnormal_mean", "abnormal_std", "abnormal_median",
    "u_statistic", "p_value", "rank_biserial",
]


def compute_per_machine_results_summary(
    per_machine_stats: pd.DataFrame,
    per_machine_significance: pd.DataFrame,
    per_machine_effect_sizes: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Phase 5.3, 5.4, and 5.5 results into one row per (machine_id, metric)."""
    normal = (
        per_machine_stats[per_machine_stats["label"] == "normal"]
        [["machine_id", "metric", "count", "mean", "std", "median"]]
        .rename(columns={"count": "normal_count", "mean": "normal_mean",
                         "std": "normal_std", "median": "normal_median"})
    )
    abnormal = (
        per_machine_stats[per_machine_stats["label"] == "abnormal"]
        [["machine_id", "metric", "count", "mean", "std", "median"]]
        .rename(columns={"count": "abnormal_count", "mean": "abnormal_mean",
                         "std": "abnormal_std", "median": "abnormal_median"})
    )
    sig = per_machine_significance[["machine_id", "metric", "u_statistic", "p_value"]]
    effect = per_machine_effect_sizes[["machine_id", "metric", "rank_biserial_correlation"]].rename(
        columns={"rank_biserial_correlation": "rank_biserial"}
    )
    summary = (
        normal
        .merge(abnormal, on=["machine_id", "metric"])
        .merge(sig, on=["machine_id", "metric"])
        .merge(effect, on=["machine_id", "metric"])
    )
    return summary[_PM_SUMMARY_COLS].reset_index(drop=True)


def save_results_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Create output directory and save results summary CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)


def print_overall_results_summary(summary: pd.DataFrame) -> None:
    _sep()
    print("Consolidated Overall Results Summary")
    _sep()
    for _, row in summary.iterrows():
        sig_tag = "*" if row["p_value"] < 0.05 else " "
        print(
            f"  {row['metric']:<26}  "
            f"normal_mean={row['normal_mean']:>9.4f}  "
            f"abnormal_mean={row['abnormal_mean']:>9.4f}  "
            f"p={row['p_value']:.4e}  "
            f"r={row['rank_biserial']:>+.4f}  {sig_tag}"
        )
    print("  (* p < 0.05)")
    print()


def print_per_machine_results_summary(summary: pd.DataFrame) -> None:
    _sep()
    print("Consolidated Per-Machine Results Summary")
    _sep()
    for machine_id in sorted(summary["machine_id"].unique()):
        print(f"\n  {machine_id}")
        subset = summary[summary["machine_id"] == machine_id]
        for _, row in subset.iterrows():
            sig_tag = "*" if row["p_value"] < 0.05 else " "
            print(
                f"    {row['metric']:<26}  "
                f"normal_mean={row['normal_mean']:>9.4f}  "
                f"abnormal_mean={row['abnormal_mean']:>9.4f}  "
                f"p={row['p_value']:.4e}  "
                f"r={row['rank_biserial']:>+.4f}  {sig_tag}"
            )
    print("  (* p < 0.05)")
    print()


# ---------------------------------------------------------------------------
# Phase 5.6 — Visualization of Healthy vs Abnormal Drift
# ---------------------------------------------------------------------------

_LABEL_ORDER = ["normal", "abnormal"]


def plot_overall_distribution(df: pd.DataFrame, metric: str) -> plt.Figure:
    """Boxplot comparing Normal vs Abnormal for one metric across all machines."""
    fig, ax = plt.subplots()
    data = [df.loc[df["true_label"] == lbl, metric].values for lbl in _LABEL_ORDER]
    ax.boxplot(data, tick_labels=["Normal", "Abnormal"])
    ax.set_title(f"Overall Distribution — {metric}")
    ax.set_xlabel("Label")
    ax.set_ylabel(metric)
    return fig


def save_overall_distribution_plot(df: pd.DataFrame, metric: str) -> None:
    """Save the overall distribution boxplot to PLOTS_DIR."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_overall_distribution(df, metric)
    fig.savefig(PLOTS_DIR / f"overall_{metric}.png", dpi=300)
    plt.close(fig)


def plot_per_machine_distribution(df: pd.DataFrame, metric: str) -> plt.Figure:
    """Grouped boxplots of Normal vs Abnormal per machine ID for one metric."""
    machine_ids = sorted(df["machine_id"].unique())
    n = len(machine_ids)
    fig, ax = plt.subplots(figsize=(max(6, 2 * n), 5))

    positions_normal = [i * 3 + 1 for i in range(n)]
    positions_abnormal = [i * 3 + 2 for i in range(n)]

    bp_normal = ax.boxplot(
        [df.loc[(df["machine_id"] == mid) & (df["true_label"] == "normal"), metric].values
         for mid in machine_ids],
        positions=positions_normal, widths=0.6, patch_artist=True,
        boxprops=dict(facecolor="steelblue"),
        medianprops=dict(color="white"),
    )
    bp_abnormal = ax.boxplot(
        [df.loc[(df["machine_id"] == mid) & (df["true_label"] == "abnormal"), metric].values
         for mid in machine_ids],
        positions=positions_abnormal, widths=0.6, patch_artist=True,
        boxprops=dict(facecolor="tomato"),
        medianprops=dict(color="white"),
    )

    ax.set_xticks([i * 3 + 1.5 for i in range(n)])
    ax.set_xticklabels(machine_ids)
    ax.set_title(f"Per-Machine Distribution — {metric}")
    ax.set_xlabel("Machine ID")
    ax.set_ylabel(metric)
    ax.legend(
        [bp_normal["boxes"][0], bp_abnormal["boxes"][0]],
        ["Normal", "Abnormal"],
    )
    return fig


def save_per_machine_distribution_plot(df: pd.DataFrame, metric: str) -> None:
    """Save the per-machine distribution boxplot to PLOTS_DIR."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_per_machine_distribution(df, metric)
    fig.savefig(PLOTS_DIR / f"per_machine_{metric}.png", dpi=300)
    plt.close(fig)


def generate_all_plots(df: pd.DataFrame) -> None:
    """Generate all six distribution plots (overall + per-machine for each metric)."""
    for metric in DRIFT_METRICS:
        save_overall_distribution_plot(df, metric)
        save_per_machine_distribution_plot(df, metric)


def print_visualization_summary() -> None:
    """Print the saved plot filenames and output directory."""
    _sep()
    print("Visualization Summary")
    _sep()
    print(f"Output directory: {PLOTS_DIR}")
    print()
    for metric in DRIFT_METRICS:
        print(f"  overall_{metric}.png")
        print(f"  per_machine_{metric}.png")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    df = load_csv(INPUT_CSV)
    print_header()
    print_dataset_summary(df)

    stats = compute_overall_drift_statistics(df)
    print_overall_drift_statistics(stats)
    save_overall_drift_statistics(stats, OVERALL_STATS_CSV)
    print(f"Saved: {OVERALL_STATS_CSV}")
    print()

    per_machine = compute_per_machine_drift_statistics(df)
    print_per_machine_drift_statistics(per_machine)
    save_per_machine_drift_statistics(per_machine, PER_MACHINE_STATS_CSV)
    print(f"Saved: {PER_MACHINE_STATS_CSV}")
    print()

    overall_sig = compute_overall_significance(df)
    print_overall_significance(overall_sig)
    save_significance(overall_sig, OVERALL_SIG_CSV)
    print(f"Saved: {OVERALL_SIG_CSV}")
    print()

    per_machine_sig = compute_per_machine_significance(df)
    print_per_machine_significance(per_machine_sig)
    save_significance(per_machine_sig, PER_MACHINE_SIG_CSV)
    print(f"Saved: {PER_MACHINE_SIG_CSV}")
    print()

    overall_effect = compute_overall_effect_sizes(df)
    print_overall_effect_sizes(overall_effect)
    save_effect_sizes(overall_effect, OVERALL_EFFECT_CSV)
    print(f"Saved: {OVERALL_EFFECT_CSV}")
    print()

    per_machine_effect = compute_per_machine_effect_sizes(df)
    print_per_machine_effect_sizes(per_machine_effect)
    save_effect_sizes(per_machine_effect, PER_MACHINE_EFFECT_CSV)
    print(f"Saved: {PER_MACHINE_EFFECT_CSV}")
    print()

    generate_all_plots(df)
    print_visualization_summary()

    overall_summary = compute_overall_results_summary(stats, overall_sig, overall_effect)
    print_overall_results_summary(overall_summary)
    save_results_summary(overall_summary, OVERALL_SUMMARY_CSV)
    print(f"Saved: {OVERALL_SUMMARY_CSV}")
    print()

    per_machine_summary = compute_per_machine_results_summary(
        per_machine, per_machine_sig, per_machine_effect
    )
    print_per_machine_results_summary(per_machine_summary)
    save_results_summary(per_machine_summary, PER_MACHINE_SUMMARY_CSV)
    print(f"Saved: {PER_MACHINE_SUMMARY_CSV}")
    print()


if __name__ == "__main__":
    main()
