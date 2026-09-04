"""
Phase 9.1 — E1 vs Phase 9 Comparability Validation

Purpose:
    Verify whether E1 and Phase 9 evaluated the same Pump test recordings
    before comparing anomaly detection metrics.

Inputs:
    experiments/results/e1/evaluation_results.csv
    experiments/results/phase9/evaluation_pump.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score


E1_RESULTS = Path(
    "experiments/results/e1/evaluation_results.csv"
)

PHASE9_RESULTS = Path(
    "experiments/results/phase9/evaluation_pump.csv"
)

COMPARISON_DIR = Path(
    "experiments/results/phase9/comparison_e1"
)

KEY_COLUMNS = [
    "machine_type",
    "machine_id",
    "filename",
    "true_label",
]

METRICS = [
    "normalized_euclidean",
    "normalized_manhattan",
    "normalized_cosine",
]

MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]


# ── Step 1 helpers ────────────────────────────────────────────────────────────

def load_results(path: Path, name: str) -> pd.DataFrame:
    """Load and validate an evaluation CSV."""

    if not path.exists():
        raise FileNotFoundError(
            f"{name} results not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(f"\n{name}")
    print("-" * 50)
    print(f"Path : {path}")
    print(f"Rows : {len(df)}")
    print(f"Columns:")

    for column in df.columns:
        print(f"  - {column}")

    missing = [
        column
        for column in KEY_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )

    return df


def validate_duplicates(
    df: pd.DataFrame,
    name: str,
) -> None:
    """Check for duplicate evaluation records."""

    duplicate_mask = df.duplicated(
        subset=KEY_COLUMNS,
        keep=False,
    )

    duplicates = df[duplicate_mask]

    if duplicates.empty:
        print(f"Duplicate records : 0")
    else:
        print(
            f"Duplicate records : {len(duplicates)}"
        )

        print(
            duplicates[KEY_COLUMNS]
            .head(10)
            .to_string(index=False)
        )

        raise ValueError(
            f"{name} contains duplicate evaluation records."
        )


def summarize_dataset(
    df: pd.DataFrame,
    name: str,
) -> None:
    """Print dataset summary."""

    print(f"\n{name} summary")
    print("-" * 50)

    print("\nMachine types:")
    print(
        df["machine_type"]
        .value_counts()
        .to_string()
    )

    print("\nMachine IDs:")
    print(
        df["machine_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nLabels:")
    print(
        df["true_label"]
        .value_counts()
        .to_string()
    )


def compare_records(
    e1: pd.DataFrame,
    phase9: pd.DataFrame,
) -> bool:
    """Compare exact evaluation recording sets."""

    e1_keys = set(
        map(
            tuple,
            e1[KEY_COLUMNS].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    phase9_keys = set(
        map(
            tuple,
            phase9[KEY_COLUMNS].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    only_e1 = e1_keys - phase9_keys
    only_phase9 = phase9_keys - e1_keys

    print("\nExact recording comparison")
    print("-" * 50)

    print(f"E1 unique records      : {len(e1_keys)}")
    print(f"Phase 9 unique records : {len(phase9_keys)}")

    print(
        f"Only in E1             : {len(only_e1)}"
    )

    print(
        f"Only in Phase 9        : {len(only_phase9)}"
    )

    if not only_e1 and not only_phase9:

        print(
            "\nPASS: E1 and Phase 9 evaluated "
            "exactly the same recordings."
        )

        return True

    print(
        "\nFAIL: E1 and Phase 9 did not evaluate "
        "the same recording set."
    )

    if only_e1:

        print("\nExamples only in E1:")

        for record in sorted(only_e1)[:10]:
            print(f"  {record}")

    if only_phase9:

        print("\nExamples only in Phase 9:")

        for record in sorted(only_phase9)[:10]:
            print(f"  {record}")

    return False


# ── Diagnostic ───────────────────────────────────────────────────────────────

KNOWN_PHASE9_AUCS = {
    "normalized_euclidean": 0.8586,
    "normalized_manhattan": 0.8592,
    "normalized_cosine":    0.5031,
}


def run_diagnostic(e1: pd.DataFrame, phase9: pd.DataFrame) -> None:
    """Investigate Phase 9 all-zero metric rows."""

    print("\n" + "=" * 60)
    print("Phase 9.1 — Diagnostic: Zero-Metric Rows")
    print("=" * 60)

    p9 = phase9.copy()
    for m in METRICS:
        p9[m] = pd.to_numeric(p9[m], errors="coerce")

    zero_mask = (p9[METRICS] == 0.0).all(axis=1)
    zero_rows = p9[zero_mask]

    print(f"\nPhase 9 rows where all three metrics are 0.0 : {len(zero_rows)}")

    if not zero_rows.empty:
        detail_cols = KEY_COLUMNS + METRICS
        print(zero_rows[detail_cols].to_string(index=False))

        print("\nChecking those recordings in E1:")
        e1_lookup = e1.set_index(KEY_COLUMNS)
        for _, row in zero_rows.iterrows():
            key = tuple(row[c] for c in KEY_COLUMNS)
            if key in e1_lookup.index:
                e1_row = e1_lookup.loc[key]
                vals = {m: e1_row[m] for m in METRICS}
                print(f"  FOUND   {key}  E1 metrics: {vals}")
            else:
                print(f"  MISSING {key}")

    print("\nPhase 9 ROC-AUC (all rows, no filtering):")
    labels = _binary_labels(p9)
    print(f"  Rows used : {len(p9)}")
    for metric in METRICS:
        scores = p9[metric].values
        try:
            auc = _compute_auc(scores, labels)
        except Exception as exc:
            print(f"  {metric:<28} AUC = ERROR ({exc})")
            continue
        known = KNOWN_PHASE9_AUCS[metric]
        match = "MATCHES" if abs(auc - known) < 1e-3 else f"DIFFERS (expected {known:.4f})"
        print(f"  {metric:<28} AUC = {auc:.4f}  {match}")

    print("=" * 60)


# ── Step 2 helpers ────────────────────────────────────────────────────────────

def _binary_labels(df: pd.DataFrame) -> np.ndarray:
    """Return 1 for abnormal, 0 for normal."""
    return (df["true_label"] == "abnormal").astype(int).values


def _compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC; if < 0.5, try negated scores and return the better value."""
    auc = roc_auc_score(labels, scores)
    if auc < 0.5:
        auc_neg = roc_auc_score(labels, -scores)
        auc = max(auc, auc_neg)
    return float(auc)


def _cohens_d(scores: np.ndarray, labels: np.ndarray) -> float:
    """Cohen's d: (mean_abnormal - mean_normal) / pooled_std."""
    abn = scores[labels == 1]
    nor = scores[labels == 0]
    pooled_std = np.sqrt(
        (np.std(abn, ddof=1) ** 2 + np.std(nor, ddof=1) ** 2) / 2
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(abn) - np.mean(nor)) / pooled_std)


def _compute_row(
    level: str,
    machine_id: str,
    metric: str,
    e1_sub: pd.DataFrame,
    p9_sub: pd.DataFrame,
) -> dict:
    labels_e1 = _binary_labels(e1_sub)
    labels_p9 = _binary_labels(p9_sub)

    e1_scores = e1_sub[metric].values
    p9_scores = p9_sub[metric].values

    e1_auc = _compute_auc(e1_scores, labels_e1)
    p9_auc = _compute_auc(p9_scores, labels_p9)

    e1_d = _cohens_d(e1_scores, labels_e1)
    p9_d = _cohens_d(p9_scores, labels_p9)

    return {
        "comparison_level": level,
        "machine_id": machine_id,
        "metric": metric,
        "e1_roc_auc": round(e1_auc, 6),
        "phase9_roc_auc": round(p9_auc, 6),
        "roc_auc_difference": round(p9_auc - e1_auc, 6),
        "e1_cohens_d": round(e1_d, 6),
        "phase9_cohens_d": round(p9_d, 6),
        "cohens_d_difference": round(p9_d - e1_d, 6),
    }


def _prepare_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Convert metric columns to numeric."""
    df = df.copy()
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return df.reset_index(drop=True)


def compute_metric_comparison(
    e1: pd.DataFrame,
    phase9: pd.DataFrame,
) -> pd.DataFrame:
    """Build the 15-row comparison DataFrame."""

    e1 = _prepare_metrics(e1)
    phase9 = _prepare_metrics(phase9)

    rows = []

    # Overall (3 rows)
    for metric in METRICS:
        rows.append(
            _compute_row(
                "overall",
                "all",
                metric,
                e1,
                phase9,
            )
        )

    # Per machine ID (12 rows)
    for mid in MACHINE_IDS:
        e1_sub = e1[e1["machine_id"] == mid]
        p9_sub = phase9[phase9["machine_id"] == mid]
        for metric in METRICS:
            rows.append(
                _compute_row(
                    "per_machine",
                    mid,
                    metric,
                    e1_sub,
                    p9_sub,
                )
            )

    return pd.DataFrame(rows)


def print_comparison(df: pd.DataFrame) -> None:
    """Print a readable summary of the comparison table."""

    print("\nROC-AUC and Cohen's d Comparison")
    print("-" * 50)

    for _, row in df.iterrows():
        level = row["comparison_level"]
        mid = row["machine_id"]
        metric = row["metric"]
        label = f"{level} / {mid} / {metric}"

        auc_diff = row["roc_auc_difference"]
        d_diff = row["cohens_d_difference"]

        print(
            f"\n  {label}"
            f"\n    AUC : E1={row['e1_roc_auc']:.4f}  "
            f"Phase9={row['phase9_roc_auc']:.4f}  "
            f"diff={auc_diff:+.4f}"
            f"\n    d   : E1={row['e1_cohens_d']:.4f}  "
            f"Phase9={row['phase9_cohens_d']:.4f}  "
            f"diff={d_diff:+.4f}"
        )


def save_comparison(df: pd.DataFrame) -> None:
    """Save CSV and JSON to the comparison output directory."""

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = COMPARISON_DIR / "metric_comparison.csv"
    json_path = COMPARISON_DIR / "metric_comparison.json"

    df.to_csv(csv_path, index=False)

    records = df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    print(f"\nSaved CSV  : {csv_path}")
    print(f"Saved JSON : {json_path}")
    print(f"Total rows : {len(df)}")


# ── Step 3 ────────────────────────────────────────────────────────────────────

def interpret_comparison(df: pd.DataFrame) -> dict:
    """Build and save a plain-language interpretation of the comparison."""

    overall = df[df["comparison_level"] == "overall"].set_index("metric")
    per_machine = df[df["comparison_level"] == "per_machine"]

    # ── Overall section ───────────────────────────────────────────────────────
    overall_summary = {}
    for metric in METRICS:
        row = overall.loc[metric]
        overall_summary[metric] = {
            "e1_roc_auc":          round(row["e1_roc_auc"], 6),
            "phase9_roc_auc":      round(row["phase9_roc_auc"], 6),
            "roc_auc_difference":  round(row["roc_auc_difference"], 6),
            "e1_cohens_d":         round(row["e1_cohens_d"], 6),
            "phase9_cohens_d":     round(row["phase9_cohens_d"], 6),
            "cohens_d_difference": round(row["cohens_d_difference"], 6),
        }

    # ── Per-machine counts ────────────────────────────────────────────────────
    n_cases = len(per_machine)  # 12
    auc_improved  = int((per_machine["roc_auc_difference"]  > 0).sum())
    auc_regressed = int((per_machine["roc_auc_difference"]  < 0).sum())
    d_improved    = int((per_machine["cohens_d_difference"] > 0).sum())
    d_regressed   = int((per_machine["cohens_d_difference"] < 0).sum())

    # ── Extremes ──────────────────────────────────────────────────────────────
    best_auc_row  = per_machine.loc[per_machine["roc_auc_difference"].idxmax()]
    worst_auc_row = per_machine.loc[per_machine["roc_auc_difference"].idxmin()]
    best_d_row    = per_machine.loc[per_machine["cohens_d_difference"].idxmax()]
    worst_d_row   = per_machine.loc[per_machine["cohens_d_difference"].idxmin()]

    def _extreme(row: pd.Series, diff_col: str) -> dict:
        return {
            "machine_id": row["machine_id"],
            "metric":     row["metric"],
            "difference": round(row[diff_col], 6),
        }

    summary = {
        "overall": overall_summary,
        "per_machine_cases_total": n_cases,
        "roc_auc": {
            "improved":  auc_improved,
            "regressed": auc_regressed,
            "unchanged": n_cases - auc_improved - auc_regressed,
            "best_improvement":  _extreme(best_auc_row,  "roc_auc_difference"),
            "largest_regression": _extreme(worst_auc_row, "roc_auc_difference"),
        },
        "cohens_d": {
            "improved":  d_improved,
            "regressed": d_regressed,
            "unchanged": n_cases - d_improved - d_regressed,
            "best_improvement":  _extreme(best_d_row,  "cohens_d_difference"),
            "largest_regression": _extreme(worst_d_row, "cohens_d_difference"),
        },
        "interpretation": _write_interpretation(overall_summary, auc_improved,
                                                 auc_regressed, d_improved,
                                                 d_regressed, n_cases),
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = COMPARISON_DIR / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # ── Print ─────────────────────────────────────────────────────────────────
    _print_interpretation(summary, summary_path)

    return summary


def _write_interpretation(
    overall: dict,
    auc_improved: int,
    auc_regressed: int,
    d_improved: int,
    d_regressed: int,
    n_cases: int,
) -> str:
    lines = []

    # Euclidean / Manhattan overall direction
    for m in ("normalized_euclidean", "normalized_manhattan"):
        diff = overall[m]["roc_auc_difference"]
        direction = "higher" if diff > 0 else "lower"
        lines.append(
            f"Phase 9 overall ROC-AUC for {m} is {direction} than E1 "
            f"(diff={diff:+.4f})."
        )

    # Cosine overall
    diff_cos = overall["normalized_cosine"]["roc_auc_difference"]
    lines.append(
        f"Phase 9 overall ROC-AUC for normalized_cosine is "
        f"{'higher' if diff_cos > 0 else 'lower'} than E1 "
        f"(diff={diff_cos:+.4f}); cosine separation is weak in both systems."
    )

    # Per-machine counts
    lines.append(
        f"Across {n_cases} per-machine metric cases: "
        f"Phase 9 ROC-AUC improved in {auc_improved}, "
        f"regressed in {auc_regressed}."
    )
    lines.append(
        f"Cohen's d improved in {d_improved} cases, "
        f"regressed in {d_regressed} cases."
    )

    # Balanced verdict
    if auc_improved > auc_regressed:
        lines.append(
            "Phase 9 shows a net ROC-AUC improvement over E1 across "
            "per-machine cases, but regressions exist for specific "
            "machine/metric combinations."
        )
    elif auc_improved < auc_regressed:
        lines.append(
            "Phase 9 shows a net ROC-AUC regression relative to E1 "
            "across per-machine cases."
        )
    else:
        lines.append(
            "Phase 9 and E1 show equal numbers of per-machine "
            "ROC-AUC improvements and regressions."
        )

    return " ".join(lines)


def _print_interpretation(summary: dict, saved_path: Path) -> None:
    print("\n" + "=" * 60)
    print("Phase 9.1 — Step 3: Comparison Interpretation Summary")
    print("=" * 60)

    print("\nOverall metrics (Phase 9 vs E1):")
    print("-" * 50)
    for metric, vals in summary["overall"].items():
        print(
            f"  {metric}"
            f"\n    ROC-AUC : E1={vals['e1_roc_auc']:.4f}  "
            f"Phase9={vals['phase9_roc_auc']:.4f}  "
            f"diff={vals['roc_auc_difference']:+.4f}"
            f"\n    Cohen's d: E1={vals['e1_cohens_d']:.4f}  "
            f"Phase9={vals['phase9_cohens_d']:.4f}  "
            f"diff={vals['cohens_d_difference']:+.4f}"
        )

    n = summary["per_machine_cases_total"]
    auc = summary["roc_auc"]
    d   = summary["cohens_d"]

    print(f"\nPer-machine cases ({n} total):")
    print("-" * 50)
    print(
        f"  ROC-AUC  — improved: {auc['improved']}  "
        f"regressed: {auc['regressed']}  "
        f"unchanged: {auc['unchanged']}"
    )
    print(
        f"  Cohen's d — improved: {d['improved']}  "
        f"regressed: {d['regressed']}  "
        f"unchanged: {d['unchanged']}"
    )

    bi = auc["best_improvement"]
    br = auc["largest_regression"]
    print(
        f"\n  Best ROC-AUC improvement : "
        f"{bi['machine_id']} / {bi['metric']}  diff={bi['difference']:+.4f}"
    )
    print(
        f"  Largest ROC-AUC regression: "
        f"{br['machine_id']} / {br['metric']}  diff={br['difference']:+.4f}"
    )

    di = d["best_improvement"]
    dr = d["largest_regression"]
    print(
        f"\n  Best Cohen's d improvement : "
        f"{di['machine_id']} / {di['metric']}  diff={di['difference']:+.4f}"
    )
    print(
        f"  Largest Cohen's d regression: "
        f"{dr['machine_id']} / {dr['metric']}  diff={dr['difference']:+.4f}"
    )

    print(f"\nInterpretation:")
    print("-" * 50)
    for sentence in summary["interpretation"].split(". "):
        sentence = sentence.strip()
        if sentence:
            print(f"  {sentence.rstrip('.')}. ")

    print(f"\nSaved summary : {saved_path}")
    print("=" * 60)


# ── Step 4 helpers ───────────────────────────────────────────────────────────

def _paired_merge(e1: pd.DataFrame, phase9: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on KEY_COLUMNS, verify exactly 1022 matched rows."""
    left  = _prepare_metrics(e1)
    right = _prepare_metrics(phase9)

    merged = left.merge(
        right,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_e1", "_p9"),
    )

    if len(merged) != 1022:
        raise ValueError(
            f"Paired merge produced {len(merged)} rows; expected 1022."
        )

    unmatched_left  = len(left)  - len(merged)
    unmatched_right = len(right) - len(merged)
    if unmatched_left or unmatched_right:
        raise ValueError(
            f"Unmatched rows - E1: {unmatched_left}, Phase 9: {unmatched_right}."
        )

    for m in METRICS:
        merged[f"diff_{m}"] = merged[f"{m}_p9"] - merged[f"{m}_e1"]

    return merged


def _diff_stats(values: pd.Series) -> dict:
    """Return a compact stats dict for a difference series."""
    return {
        "n":        int(values.notna().sum()),
        "mean":     round(float(values.mean()),   6),
        "std":      round(float(values.std()),    6),
        "min":      round(float(values.min()),    6),
        "max":      round(float(values.max()),    6),
        "phase9_higher": int((values > 0).sum()),
        "e1_higher":     int((values < 0).sum()),
        "equal":         int((values == 0).sum()),
    }


def _build_summary_rows(
    merged: pd.DataFrame,
) -> list[dict]:
    """Produce one summary row per (group_type, group_value, metric)."""
    rows = []

    groups: list[tuple[str, str, pd.DataFrame]] = [
        ("overall", "all", merged),
    ]
    for lbl in ("normal", "abnormal"):
        groups.append(("label", lbl, merged[merged["true_label"] == lbl]))
    for mid in MACHINE_IDS:
        groups.append(("machine", mid, merged[merged["machine_id"] == mid]))

    for group_type, group_value, subset in groups:
        for m in METRICS:
            stats = _diff_stats(subset[f"diff_{m}"])
            rows.append({
                "group_type":  group_type,
                "group_value": group_value,
                "metric":      m,
                **stats,
            })

    return rows


def _print_paired_summary(summary_df: pd.DataFrame) -> None:
    print("\nPaired difference summary  (phase9 - e1)")
    print("-" * 50)

    for group_type in ("overall", "label", "machine"):
        subset = summary_df[summary_df["group_type"] == group_type]
        print(f"\n  [{group_type}]")
        for _, row in subset.iterrows():
            print(
                f"    {row['group_value']:<10}  {row['metric']:<26}"
                f"  mean={row['mean']:+.4f}  std={row['std']:.4f}"
                f"  phase9_higher={row['phase9_higher']}  e1_higher={row['e1_higher']}"
                f"  equal={row['equal']}"
            )


def run_paired_analysis(
    e1: pd.DataFrame,
    phase9: pd.DataFrame,
) -> None:
    """Step 4: paired recording-level difference analysis."""

    print("\n" + "=" * 60)
    print("Phase 9.1 -- E1 vs Phase 9 Comparison")
    print("Step 4: Paired Recording-Level Analysis")
    print("=" * 60)

    merged = _paired_merge(e1, phase9)

    print(f"\nPaired merge verification")
    print("-" * 50)
    print(f"  Matched recordings : {len(merged)}")
    print(f"  Duplicates in merge: 0  (guaranteed by Step 1)")
    print(f"  Unmatched rows     : 0")
    print(f"  PASS: exactly 1022 recordings matched.")

    summary_rows = _build_summary_rows(merged)
    summary_df   = pd.DataFrame(summary_rows)

    _print_paired_summary(summary_df)

    # ── Save ──────────────────────────────────────────────────────────────────
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    paired_csv   = COMPARISON_DIR / "paired_recording_comparison.csv"
    summary_csv  = COMPARISON_DIR / "paired_difference_summary.csv"
    summary_json = COMPARISON_DIR / "paired_difference_summary.json"

    # paired CSV: key columns + per-metric e1/p9/diff columns
    paired_cols = (
        KEY_COLUMNS
        + [f"{m}_e1" for m in METRICS]
        + [f"{m}_p9" for m in METRICS]
        + [f"diff_{m}" for m in METRICS]
    )
    merged[paired_cols].to_csv(paired_csv, index=False)

    summary_df.to_csv(summary_csv, index=False)

    records = summary_df.to_dict(orient="records")
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    print(f"\nSaved paired CSV     : {paired_csv}")
    print(f"Saved summary CSV    : {summary_csv}")
    print(f"Saved summary JSON   : {summary_json}")
    print(f"Paired rows          : {len(merged)}")
    print(f"Summary rows         : {len(summary_df)}")


# ── Step 5 helpers ───────────────────────────────────────────────────────────

ALPHA = 0.05


def _wilcoxon_test(values: np.ndarray) -> dict:
    """Run Wilcoxon signed-rank test on paired differences."""
    nonzero = values[values != 0]
    if len(nonzero) < 10:
        return {"stat": None, "p_value": None, "verdict": "INSUFFICIENT_DATA"}
    stat, p = wilcoxon(nonzero, alternative="two-sided")
    verdict = "SIGNIFICANT" if p < ALPHA else "NOT_SIGNIFICANT"
    return {
        "stat":    round(float(stat), 4),
        "p_value": round(float(p), 6),
        "verdict": verdict,
    }


def run_significance_testing(
    e1: pd.DataFrame,
    phase9: pd.DataFrame,
) -> None:
    """Step 5: Wilcoxon signed-rank tests on paired differences."""

    print("\n" + "=" * 60)
    print("Phase 9.1 -- E1 vs Phase 9 Comparison")
    print("Step 5: Paired Statistical Significance Testing")
    print("=" * 60)

    merged = _paired_merge(e1, phase9)

    groups = [
        ("overall",  "all",      merged),
        ("label",    "normal",   merged[merged["true_label"] == "normal"]),
        ("label",    "abnormal", merged[merged["true_label"] == "abnormal"]),
    ]

    results = []

    print(f"\nWilcoxon signed-rank test  (two-sided, alpha={ALPHA})")
    print("-" * 50)

    for group_type, group_value, subset in groups:
        print(f"\n  [{group_type} / {group_value}]  n={len(subset)}")
        for m in METRICS:
            diffs = subset[f"diff_{m}"].dropna().values
            res = _wilcoxon_test(diffs)
            if res["stat"] is None:
                print(f"    {m:<28}  INSUFFICIENT_DATA")
            else:
                print(
                    f"    {m:<28}"
                    f"  stat={res['stat']:.1f}"
                    f"  p={res['p_value']:.6f}"
                    f"  -> {res['verdict']}"
                )
            results.append({
                "group_type":  group_type,
                "group_value": group_value,
                "metric":      m,
                "n":           int(len(diffs)),
                **res,
            })

    # ── Save ──────────────────────────────────────────────────────────────────
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    out_json = COMPARISON_DIR / "significance_tests.json"
    out_csv  = COMPARISON_DIR / "significance_tests.csv"

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    pd.DataFrame(results).to_csv(out_csv, index=False)

    print(f"\nSaved JSON : {out_json}")
    print(f"Saved CSV  : {out_csv}")
    print(f"Rows       : {len(results)}")


# ── Step 6 helpers ──────────────────────────────────────────────────────────

# Rank-biserial correlation from Wilcoxon signed-rank test:
#   r = 1 - (2 * W) / (n_nonzero * (n_nonzero + 1))
# where W is the Wilcoxon statistic (sum of ranks of negative differences)
# and n_nonzero is the number of non-zero paired differences.
#
# This is the standard matched-pairs rank-biserial correlation
# (Kerby 2014; equivalent to Glass rank-biserial for signed-rank tests).
# Ranges from -1 to +1:
#   positive r  -> Phase 9 tends to produce higher values
#   negative r  -> E1 tends to produce higher values
#
# Magnitude thresholds (Cohen 1988 / Rosenthal 1991 conventions):
#   |r| <  0.10  -> negligible
#   |r| <  0.30  -> small
#   |r| <  0.50  -> moderate
#   |r| >= 0.50  -> large

_ES_THRESHOLDS = [(0.10, "negligible"), (0.30, "small"), (0.50, "moderate")]


def _effect_size_magnitude(abs_r: float) -> str:
    for threshold, label in _ES_THRESHOLDS:
        if abs_r < threshold:
            return label
    return "large"


def _rank_biserial(diffs: np.ndarray) -> float | None:
    """Rank-biserial correlation computed from the Wilcoxon W statistic."""
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n < 10:
        return None
    stat, _ = wilcoxon(nonzero, alternative="two-sided")
    return float(1.0 - (2.0 * stat) / (n * (n + 1)))


def _practical_direction(mean_diff: float, pct_p9_higher: float) -> str:
    """Direction requires both mean and majority-win to agree."""
    if mean_diff > 0 and pct_p9_higher > 50.0:
        return "Phase 9 tends higher"
    if mean_diff < 0 and pct_p9_higher < 50.0:
        return "E1 tends higher"
    return "No clear practical direction"


def run_practical_effect_analysis(
    e1: pd.DataFrame,
    phase9: pd.DataFrame,
) -> None:
    """Step 6: Practical effect size and direction analysis."""

    print("\n" + "=" * 60)
    print("PHASE 9 \u2014 STEP 6")
    print("Practical Effect Size and Direction Analysis")
    print("=" * 60)

    merged = _paired_merge(e1, phase9)

    groups = [
        ("overall",  merged),
        ("normal",   merged[merged["true_label"] == "normal"]),
        ("abnormal", merged[merged["true_label"] == "abnormal"]),
    ]

    rows = []

    for group_name, subset in groups:
        print(f"\n  [{group_name}]  n={len(subset)}")
        print("  " + "-" * 56)

        for m in METRICS:
            diffs = subset[f"diff_{m}"].dropna().values
            n = len(diffs)

            mean_diff   = float(np.mean(diffs))
            median_diff = float(np.median(diffs))
            std_diff    = float(np.std(diffs, ddof=1))

            p9_higher = int((diffs > 0).sum())
            e1_higher = int((diffs < 0).sum())
            pct_p9    = round(100.0 * p9_higher / n, 2) if n > 0 else 0.0
            pct_e1    = round(100.0 * e1_higher / n, 2) if n > 0 else 0.0

            r = _rank_biserial(diffs)
            abs_r     = abs(r) if r is not None else None
            magnitude = _effect_size_magnitude(abs_r) if abs_r is not None else "INSUFFICIENT_DATA"
            direction = _practical_direction(mean_diff, pct_p9)

            print(
                f"    {m}\n"
                f"      n={n}  mean_diff={mean_diff:+.6f}  median_diff={median_diff:+.6f}\n"
                f"      pct_phase9_higher={pct_p9:.1f}%  pct_e1_higher={pct_e1:.1f}%\n"
                f"      effect_size(r)={r:+.4f}  |r|={abs_r:.4f}  magnitude={magnitude}\n"
                f"      direction: {direction}"
            )

            rows.append({
                "group":                 group_name,
                "metric":               m,
                "n":                    n,
                "mean_diff":            round(mean_diff,   6),
                "median_diff":          round(median_diff, 6),
                "std_diff":             round(std_diff,    6),
                "phase9_higher":        p9_higher,
                "e1_higher":            e1_higher,
                "pct_phase9_higher":    pct_p9,
                "pct_e1_higher":        pct_e1,
                "effect_size_r":        round(r, 6) if r is not None else None,
                "abs_effect_size_r":    round(abs_r, 6) if abs_r is not None else None,
                "effect_size_magnitude": magnitude,
                "direction":            direction,
            })

    result_df = pd.DataFrame(rows)

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    out_csv  = COMPARISON_DIR / "practical_effect_analysis.csv"
    out_json = COMPARISON_DIR / "practical_effect_analysis.json"

    result_df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result_df.to_dict(orient="records"), fh, indent=2)

    print(f"\nSaved CSV  : {out_csv}")
    print(f"Saved JSON : {out_json}")
    print(f"Rows       : {len(result_df)}  (3 groups x 3 metrics)")


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> None:

    print("=" * 60)
    print("Phase 9.1 — E1 vs Phase 9 Comparison")
    print("Step 1: Evaluation Dataset Validation")
    print("=" * 60)

    e1 = load_results(
        E1_RESULTS,
        "E1",
    )

    phase9 = load_results(
        PHASE9_RESULTS,
        "Phase 9",
    )

    print()

    validate_duplicates(e1, "E1")

    validate_duplicates(
        phase9,
        "Phase 9",
    )

    summarize_dataset(e1, "E1")

    summarize_dataset(
        phase9,
        "Phase 9",
    )

    records_match = compare_records(
        e1,
        phase9,
    )

    print("\n" + "=" * 60)

    if records_match:

        print(
            "RESULT: FAIR COMPARISON POSSIBLE"
        )

        print(
            "\nNext step:"
        )

        print(
            "Compare ROC-AUC and Cohen's d."
        )

    else:

        print(
            "RESULT: RECORDING SETS DIFFER"
        )

        print(
            "\nDo not directly compare metrics "
            "until the split difference is explained."
        )

    print("=" * 60)

    if not records_match:
        return

    # ── Step 2 Diagnostic ─────────────────────────────────────────────────────

    run_diagnostic(e1, phase9)

    # ── Step 2: ROC-AUC and Cohen's d Comparison ─────────────────────────────

    print("\n" + "=" * 60)
    print("Phase 9.1 — E1 vs Phase 9 Comparison")
    print("Step 2: ROC-AUC and Cohen's d Comparison")
    print("=" * 60)

    comparison_df = compute_metric_comparison(e1, phase9)

    print_comparison(comparison_df)

    save_comparison(comparison_df)

    print("\n" + "=" * 60)
    print("Step 2 complete.")
    print("=" * 60)

    # ── Step 3: Interpretation Summary ─────────────────────────────────

    interpret_comparison(comparison_df)

    print("\n" + "=" * 60)
    print("Step 3 complete.")
    print("=" * 60)

    # ── Step 4: Paired Recording-Level Analysis ────────────────────────

    run_paired_analysis(e1, phase9)

    print("\n" + "=" * 60)
    print("Step 4 complete.")
    print("=" * 60)

    # ── Step 5: Paired Statistical Significance Testing ─────────────────

    run_significance_testing(e1, phase9)

    print("\n" + "=" * 60)
    print("Step 5 complete.")
    print("=" * 60)

    # ── Step 6: Practical Effect Size and Direction Analysis ────────────────

    run_practical_effect_analysis(e1, phase9)

    print("\n" + "=" * 60)
    print("Step 6 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
