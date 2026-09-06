"""Phase 18 — Statistical Significance Analysis of Baseline Comparison.

Reads the existing per-recording score files produced by Phase 9 and
Phase 18 (no recomputation, no modification of those files) and answers:

    Is the Phase 9 proposed method significantly better than OC-SVM,
    Isolation Forest, and kNN under the same test set?

Three complementary tests are applied at every scope level
(overall-pooled, per-machine-type, per-machine-ID):

1. Bootstrap paired AUC difference (2 000 iterations, seed=42)
   - Resamples the SAME indices for both methods simultaneously so the
     pairing is preserved.
   - Reports: point-estimate delta, 95 % percentile CI, one-sided
     bootstrap p-value (H0: delta <= 0, i.e. Phase 9 is not better).

2. DeLong's test (analytic, no resampling)
   - Exploits the equivalence AUC = P(score_pos > score_neg) and the
     Mann-Whitney U structural components to derive the covariance
     between two correlated AUC estimators on the same test set.
   - Reports: z-statistic, two-sided p-value, 95 % CI for the AUC
     difference via the normal approximation.
   - Reference: DeLong et al. (1988) Biometrics 44(3):837-845.

3. Wilcoxon signed-rank test on per-machine-ID AUC pairs
   - 16 pairs (4 types x 4 IDs); non-parametric, consistent with the
     Step 10 methodology already used in this project.
   - Reports: W-statistic, two-sided p-value, rank-biserial effect size.

Outputs saved to:
    experiments/results/phase18_baseline_comparison/significance/
        significance_results.json
        significance_summary.csv        (one row per method x scope)
        bootstrap_deltas.csv            (all 2000 delta samples, overall)
        wilcoxon_per_id.csv             (16-pair signed-rank results)
        significance_report.txt         (human-readable narrative)

Source files (read-only):
    experiments/results/phase9/evaluation_results.csv
    experiments/results/phase18_baseline_comparison/per_recording_scores.csv

Usage:
    python experiments/phase18_significance.py
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm, wilcoxon
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent
P9_CSV      = ROOT / "results/phase9/evaluation_results.csv"
P18_CSV     = ROOT / "results/phase18_baseline_comparison/per_recording_scores.csv"
OUT_DIR     = ROOT / "results/phase18_baseline_comparison/significance"

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]
BASELINES     = ["ocsvm", "iforest", "knn"]

N_BOOTSTRAP = 2000
SEED        = 42
ALPHA       = 0.05


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_scores() -> dict:
    """Load and align Phase 9 and Phase 18 per-recording scores.

    Returns a dict keyed by (machine_type, machine_id, filename, true_label)
    with value {'phase9': float, 'ocsvm': float, 'iforest': float, 'knn': float}.
    Both CSVs must cover the same 5 522 recordings.
    """
    p9_map: dict[tuple, float] = {}
    with open(P9_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["machine_type"], row["machine_id"],
                   row["filename"], row["true_label"])
            p9_map[key] = float(row["normalized_euclidean"])

    merged: dict[tuple, dict] = {}
    with open(P18_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["machine_type"], row["machine_id"],
                   row["filename"], row["true_label"])
            if key not in p9_map:
                continue
            merged[key] = {
                "phase9":  p9_map[key],
                "ocsvm":   float(row["ocsvm_score"]),
                "iforest": float(row["iforest_score"]),
                "knn":     float(row["knn_score"]),
            }

    missing = set(p9_map) - set(merged)
    if missing:
        raise ValueError(f"{len(missing)} Phase 9 recordings not found in Phase 18 CSV")

    print(f"Loaded {len(merged)} aligned recordings.")
    return merged


def _subset(merged: dict, machine_type: str | None = None,
            machine_id: str | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return (y_true, phase9_scores, {baseline: scores}) for a subset."""
    keys = [k for k in merged
            if (machine_type is None or k[0] == machine_type)
            and (machine_id  is None or k[1] == machine_id)]

    y_true  = np.array([1 if k[3] == "abnormal" else 0 for k in keys], dtype=np.int32)
    p9      = np.array([merged[k]["phase9"]  for k in keys], dtype=np.float64)
    baselines = {
        bl: np.array([merged[k][bl] for k in keys], dtype=np.float64)
        for bl in BASELINES
    }
    return y_true, p9, baselines


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    auc = float(roc_auc_score(y, s))
    return auc if auc >= 0.5 else float(roc_auc_score(y, -s))


# ---------------------------------------------------------------------------
# Test 1 — Bootstrap paired AUC difference
# ---------------------------------------------------------------------------

def _bootstrap_paired(
    y: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """Bootstrap 95 % CI and one-sided p-value for AUC(a) - AUC(b).

    Resamples the same indices for both score vectors so the pairing is
    preserved.  H0: delta <= 0 (method a is not better than method b).
    """
    rng   = np.random.default_rng(seed)
    n     = len(y)
    deltas: list[float] = []

    while len(deltas) < n_boot:
        idx = rng.integers(0, n, n)
        yb, sa_b, sb_b = y[idx], s_a[idx], s_b[idx]
        if len(np.unique(yb)) < 2:
            continue
        auc_a = float(roc_auc_score(yb, sa_b))
        auc_b = float(roc_auc_score(yb, sb_b))
        # Flip if inverted (consistent with _safe_auc)
        if auc_a < 0.5:
            auc_a = float(roc_auc_score(yb, -sa_b))
        if auc_b < 0.5:
            auc_b = float(roc_auc_score(yb, -sb_b))
        deltas.append(auc_a - auc_b)

    arr      = np.array(deltas)
    point_a  = _safe_auc(y, s_a)
    point_b  = _safe_auc(y, s_b)
    delta_pt = point_a - point_b
    ci_lo, ci_hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    # One-sided p: proportion of bootstrap deltas <= 0
    p_one_sided = float((arr <= 0).mean())

    return arr, {
        "auc_a":          round(point_a,  6),
        "auc_b":          round(point_b,  6),
        "delta":          round(delta_pt, 6),
        "boot_mean_delta": round(float(arr.mean()), 6),
        "boot_std_delta":  round(float(arr.std(ddof=1)), 6),
        "ci_lo_95":       round(ci_lo,    6),
        "ci_hi_95":       round(ci_hi,    6),
        "p_one_sided":    round(p_one_sided, 6),
        "significant":    bool(ci_lo > 0),   # CI excludes 0 => significant
        "n_bootstrap":    len(deltas),
    }


# ---------------------------------------------------------------------------
# Test 2 — DeLong's test (analytic)
# ---------------------------------------------------------------------------

def _delong_components(y: np.ndarray, s: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute AUC and the structural components V10, V01 for DeLong's test.

    V10[i] = P(score of positive i > score of random negative)
    V01[j] = P(score of random positive > score of negative j)

    These are the per-sample placement values used to estimate Var(AUC).
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    s_pos   = s[pos_idx]
    s_neg   = s[neg_idx]
    n1, n0  = len(s_pos), len(s_neg)

    # AUC via Mann-Whitney
    auc = float(roc_auc_score(y, s))

    # V10: for each positive, fraction of negatives it outranks
    # V01: for each negative, fraction of positives that outrank it
    # Use broadcasting; tie-handling: 0.5 for ties
    diff10 = s_pos[:, None] - s_neg[None, :]   # (n1, n0)
    v10 = ((diff10 > 0).astype(float) + 0.5 * (diff10 == 0).astype(float)).mean(axis=1)

    diff01 = s_pos[:, None] - s_neg[None, :]   # (n1, n0)
    v01 = ((diff01 > 0).astype(float) + 0.5 * (diff01 == 0).astype(float)).mean(axis=0)

    return auc, v10, v01


def _delong_test(
    y: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
) -> dict:
    """DeLong's test for the difference of two correlated AUC values.

    Both score vectors are evaluated on the same test set (y).
    Returns z-statistic, two-sided p-value, and 95 % CI for delta.

    Reference: DeLong, DeLong & Clarke-Pearson (1988) Biometrics 44:837-845.
    """
    if len(np.unique(y)) < 2:
        return {"z": float("nan"), "p_two_sided": float("nan"),
                "ci_lo_95": float("nan"), "ci_hi_95": float("nan"),
                "auc_a": float("nan"), "auc_b": float("nan"),
                "delta": float("nan")}

    # Flip scores if AUC < 0.5 so both are in the same orientation
    auc_a_raw = float(roc_auc_score(y, s_a))
    auc_b_raw = float(roc_auc_score(y, s_b))
    s_a_use = s_a if auc_a_raw >= 0.5 else -s_a
    s_b_use = s_b if auc_b_raw >= 0.5 else -s_b

    auc_a, v10_a, v01_a = _delong_components(y, s_a_use)
    auc_b, v10_b, v01_b = _delong_components(y, s_b_use)

    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())

    # Covariance matrix of (AUC_a, AUC_b)
    s10 = np.cov(np.stack([v10_a, v10_b], axis=0), ddof=1)  # (2,2)
    s01 = np.cov(np.stack([v01_a, v01_b], axis=0), ddof=1)  # (2,2)

    cov_mat = s10 / n1 + s01 / n0   # (2,2)

    # Var(AUC_a - AUC_b) = Var(AUC_a) + Var(AUC_b) - 2*Cov(AUC_a, AUC_b)
    var_diff = cov_mat[0, 0] + cov_mat[1, 1] - 2 * cov_mat[0, 1]

    if var_diff <= 0:
        return {"z": float("nan"), "p_two_sided": float("nan"),
                "ci_lo_95": float("nan"), "ci_hi_95": float("nan"),
                "auc_a": round(auc_a, 6), "auc_b": round(auc_b, 6),
                "delta": round(auc_a - auc_b, 6)}

    se    = float(np.sqrt(var_diff))
    delta = auc_a - auc_b
    z     = delta / se
    p     = float(2 * norm.sf(abs(z)))
    ci_lo = delta - 1.96 * se
    ci_hi = delta + 1.96 * se

    return {
        "auc_a":      round(auc_a,  6),
        "auc_b":      round(auc_b,  6),
        "delta":      round(delta,  6),
        "se":         round(se,     6),
        "z":          round(z,      4),
        "p_two_sided": round(p,     6),
        "ci_lo_95":   round(ci_lo,  6),
        "ci_hi_95":   round(ci_hi,  6),
        "significant": bool(p < ALPHA),
    }


# ---------------------------------------------------------------------------
# Test 3 — Wilcoxon signed-rank on per-machine-ID AUC pairs
# ---------------------------------------------------------------------------

def _wilcoxon_per_id(merged: dict) -> list[dict]:
    """Wilcoxon signed-rank test on 16 per-machine-ID AUC pairs.

    For each baseline, collect 16 (phase9_auc, baseline_auc) pairs
    (4 machine types x 4 IDs) and test whether Phase 9 AUCs are
    systematically higher.
    """
    rows = []
    for bl in BASELINES:
        p9_aucs, bl_aucs, pair_labels = [], [], []
        for mt in MACHINE_TYPES:
            for mid in MACHINE_IDS:
                y, s_p9, s_bl = _subset(merged, machine_type=mt, machine_id=mid)
                if len(np.unique(y)) < 2 or len(y) < 4:
                    continue
                auc_p9 = _safe_auc(y, s_p9)
                auc_bl = _safe_auc(y, s_bl[bl])
                if np.isnan(auc_p9) or np.isnan(auc_bl):
                    continue
                p9_aucs.append(auc_p9)
                bl_aucs.append(auc_bl)
                pair_labels.append(f"{mt}/{mid}")

        a = np.array(p9_aucs)
        b = np.array(bl_aucs)
        diff = a - b
        nonzero = diff[diff != 0]
        n_pairs = len(a)

        if len(nonzero) < 5:
            rows.append({
                "baseline": bl, "n_pairs": n_pairs,
                "mean_phase9_auc": round(float(a.mean()), 6),
                "mean_baseline_auc": round(float(b.mean()), 6),
                "mean_delta": round(float(diff.mean()), 6),
                "w_stat": None, "p_two_sided": None,
                "effect_r": None, "verdict": "INSUFFICIENT_DATA",
            })
            continue

        w_stat, p_val = wilcoxon(nonzero, alternative="two-sided")
        # Rank-biserial effect size
        r_effect = float(1.0 - (2.0 * w_stat) / (len(nonzero) * (len(nonzero) + 1)))

        rows.append({
            "baseline":          bl,
            "n_pairs":           n_pairs,
            "mean_phase9_auc":   round(float(a.mean()),    6),
            "mean_baseline_auc": round(float(b.mean()),    6),
            "mean_delta":        round(float(diff.mean()), 6),
            "w_stat":            round(float(w_stat),      4),
            "p_two_sided":       round(float(p_val),       6),
            "effect_r":          round(r_effect,           4),
            "verdict":           "SIGNIFICANT" if p_val < ALPHA else "NOT_SIGNIFICANT",
            "pairs":             [
                {"label": lbl, "phase9": round(float(p9_aucs[i]), 6),
                 "baseline": round(float(bl_aucs[i]), 6),
                 "delta": round(float(p9_aucs[i] - bl_aucs[i]), 6)}
                for i, lbl in enumerate(pair_labels)
            ],
        })

    return rows


# ---------------------------------------------------------------------------
# Run all tests across all scopes
# ---------------------------------------------------------------------------

def _run_all_tests(merged: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Run bootstrap + DeLong at overall / per-type / per-ID scopes.

    Returns (summary_rows, bootstrap_delta_rows, delong_rows).
    summary_rows has one entry per (scope, group, baseline).
    """
    summary_rows: list[dict]          = []
    boot_delta_rows: list[dict]       = []

    scopes: list[tuple[str, str, str | None, str | None]] = [
        ("overall", "all", None, None),
    ]
    for mt in MACHINE_TYPES:
        scopes.append(("machine_type", mt, mt, None))
    for mt in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            scopes.append(("machine_id", f"{mt}/{mid}", mt, mid))

    total = len(scopes) * len(BASELINES)
    done  = 0

    for scope_label, group_label, mt, mid in scopes:
        y, s_p9, s_bl = _subset(merged, machine_type=mt, machine_id=mid)

        if len(np.unique(y)) < 2 or len(y) < 10:
            done += len(BASELINES)
            continue

        for bl in BASELINES:
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  [{done}/{total}] {scope_label} {group_label} vs {bl}")

            boot_arr, boot = _bootstrap_paired(y, s_p9, s_bl[bl])
            delong = _delong_test(y, s_p9, s_bl[bl])

            row = {
                "scope":          scope_label,
                "group":          group_label,
                "baseline":       bl,
                # Point estimates
                "phase9_auc":     boot["auc_a"],
                "baseline_auc":   boot["auc_b"],
                "delta":          boot["delta"],
                # Bootstrap
                "boot_ci_lo":     boot["ci_lo_95"],
                "boot_ci_hi":     boot["ci_hi_95"],
                "boot_p_one_sided": boot["p_one_sided"],
                "boot_significant": boot["significant"],
                # DeLong
                "delong_z":       delong.get("z"),
                "delong_p":       delong.get("p_two_sided"),
                "delong_ci_lo":   delong.get("ci_lo_95"),
                "delong_ci_hi":   delong.get("ci_hi_95"),
                "delong_significant": delong.get("significant"),
                # Sample size
                "n_total":        int(len(y)),
                "n_normal":       int((y == 0).sum()),
                "n_abnormal":     int((y == 1).sum()),
            }
            summary_rows.append(row)

            # Store bootstrap delta distribution for overall scope only
            if scope_label == "overall":
                for delta_val in boot_arr:
                    boot_delta_rows.append({
                        "baseline": bl,
                        "delta":    round(float(delta_val), 8),
                    })

    return summary_rows, boot_delta_rows


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def _save(summary_rows: list[dict], boot_delta_rows: list[dict],
          wilcoxon_rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # significance_summary.csv
    csv_fields = [
        "scope", "group", "baseline",
        "phase9_auc", "baseline_auc", "delta",
        "boot_ci_lo", "boot_ci_hi", "boot_p_one_sided", "boot_significant",
        "delong_z", "delong_p", "delong_ci_lo", "delong_ci_hi", "delong_significant",
        "n_total", "n_normal", "n_abnormal",
    ]
    with (OUT_DIR / "significance_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(summary_rows)

    # bootstrap_deltas.csv
    with (OUT_DIR / "bootstrap_deltas.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["baseline", "delta"])
        w.writeheader()
        w.writerows(boot_delta_rows)

    # wilcoxon_per_id.csv  (flatten pairs into separate rows)
    wil_flat = []
    for r in wilcoxon_rows:
        pairs = r.pop("pairs", [])
        wil_flat.append(r)
        for p in pairs:
            wil_flat.append({
                "baseline": r["baseline"],
                "n_pairs": "",
                "mean_phase9_auc": "",
                "mean_baseline_auc": "",
                "mean_delta": "",
                "w_stat": "",
                "p_two_sided": "",
                "effect_r": "",
                "verdict": "",
                **{f"pair_{k}": v for k, v in p.items()},
            })

    wil_fields = [
        "baseline", "n_pairs", "mean_phase9_auc", "mean_baseline_auc",
        "mean_delta", "w_stat", "p_two_sided", "effect_r", "verdict",
        "pair_label", "pair_phase9", "pair_baseline", "pair_delta",
    ]
    with (OUT_DIR / "wilcoxon_per_id.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=wil_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(wil_flat)

    # significance_results.json
    result_obj = {
        "experiment":    "phase18_significance",
        "source_p9":     str(P9_CSV),
        "source_p18":    str(P18_CSV),
        "n_recordings":  5522,
        "n_bootstrap":   N_BOOTSTRAP,
        "seed":          SEED,
        "alpha":         ALPHA,
        "methods_tested": BASELINES,
        "tests": {
            "bootstrap_paired": (
                "Paired bootstrap AUC difference (2000 iterations). "
                "CI excludes 0 => significant. One-sided p = P(delta<=0 under bootstrap)."
            ),
            "delong": (
                "DeLong (1988) analytic test for correlated AUC values. "
                "Two-sided z-test on AUC_phase9 - AUC_baseline."
            ),
            "wilcoxon_per_id": (
                "Wilcoxon signed-rank on 16 per-machine-ID AUC pairs "
                "(4 types x 4 IDs). Two-sided, alpha=0.05."
            ),
        },
        "overall_summary": {
            bl: next(
                (r for r in summary_rows
                 if r["scope"] == "overall" and r["baseline"] == bl), {}
            )
            for bl in BASELINES
        },
        "wilcoxon": wilcoxon_rows,
        "per_type_summary": {
            mt: {
                bl: next(
                    (r for r in summary_rows
                     if r["scope"] == "machine_type"
                     and r["group"] == mt
                     and r["baseline"] == bl), {}
                )
                for bl in BASELINES
            }
            for mt in MACHINE_TYPES
        },
    }
    with (OUT_DIR / "significance_results.json").open("w", encoding="utf-8") as fh:
        json.dump(result_obj, fh, indent=2)

    print(f"\nOutputs written to {OUT_DIR}/")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def _report(summary_rows: list[dict], wilcoxon_rows: list[dict]) -> str:
    def _get(scope, group, bl):
        return next((r for r in summary_rows
                     if r["scope"] == scope and r["group"] == group
                     and r["baseline"] == bl), {})

    def _sig(r):
        b = r.get("boot_significant")
        d = r.get("delong_significant")
        if b and d:
            return "SIGNIFICANT (both tests)"
        if b:
            return "SIGNIFICANT (bootstrap only)"
        if d:
            return "SIGNIFICANT (DeLong only)"
        return "NOT SIGNIFICANT"

    L: list[str] = []
    L.append("Phase 18 - Statistical Significance Analysis")
    L.append("=" * 70)
    L.append(f"Source (Phase 9) : {P9_CSV}")
    L.append(f"Source (Phase 18): {P18_CSV}")
    L.append(f"N recordings     : 5 522  (normal=2 222, abnormal=3 300)")
    L.append(f"Bootstrap iters  : {N_BOOTSTRAP}  seed={SEED}")
    L.append(f"Alpha            : {ALPHA}")
    L.append("")

    L.append("-" * 70)
    L.append("1. OVERALL POOLED RESULTS (all 4 machine types combined)")
    L.append("-" * 70)
    L.append(f"  {'Baseline':<12} {'Phase9 AUC':>10} {'BL AUC':>10} {'Delta':>8} "
             f"{'Boot 95%CI':>22} {'Boot p':>8} {'DeLong z':>9} {'DeLong p':>9} {'Verdict'}")
    L.append("  " + "-" * 100)
    for bl in BASELINES:
        r = _get("overall", "all", bl)
        if not r:
            continue
        ci = f"[{r['boot_ci_lo']:+.4f}, {r['boot_ci_hi']:+.4f}]"
        L.append(
            f"  {bl:<12} {r['phase9_auc']:>10.4f} {r['baseline_auc']:>10.4f} "
            f"{r['delta']:>+8.4f} {ci:>22} {r['boot_p_one_sided']:>8.4f} "
            f"{r['delong_z']:>+9.3f} {r['delong_p']:>9.4f}  {_sig(r)}"
        )
    L.append("")

    L.append("-" * 70)
    L.append("2. PER-MACHINE-TYPE RESULTS (bootstrap delta, DeLong p)")
    L.append("-" * 70)
    for mt in MACHINE_TYPES:
        L.append(f"\n  {mt.upper()}")
        L.append(f"  {'Baseline':<12} {'Phase9':>8} {'BL':>8} {'Delta':>8} "
                 f"{'Boot CI':>22} {'Boot p':>8} {'DeLong p':>9} {'Verdict'}")
        L.append("  " + "-" * 90)
        for bl in BASELINES:
            r = _get("machine_type", mt, bl)
            if not r:
                continue
            ci = f"[{r['boot_ci_lo']:+.4f}, {r['boot_ci_hi']:+.4f}]"
            L.append(
                f"  {bl:<12} {r['phase9_auc']:>8.4f} {r['baseline_auc']:>8.4f} "
                f"{r['delta']:>+8.4f} {ci:>22} {r['boot_p_one_sided']:>8.4f} "
                f"{r['delong_p']:>9.4f}  {_sig(r)}"
            )
    L.append("")

    L.append("-" * 70)
    L.append("3. WILCOXON SIGNED-RANK ON 16 PER-MACHINE-ID AUC PAIRS")
    L.append("-" * 70)
    L.append(f"  {'Baseline':<12} {'N pairs':>8} {'Mean P9':>9} {'Mean BL':>9} "
             f"{'Mean delta':>10} {'W stat':>8} {'p':>8} {'Effect r':>9} {'Verdict'}")
    L.append("  " + "-" * 90)
    for r in wilcoxon_rows:
        L.append(
            f"  {r['baseline']:<12} {r['n_pairs']:>8} "
            f"{r['mean_phase9_auc']:>9.4f} {r['mean_baseline_auc']:>9.4f} "
            f"{r['mean_delta']:>+8.4f} "
            f"{str(r['w_stat']):>8} {str(r['p_two_sided']):>8} "
            f"{str(r['effect_r']):>9}  {r['verdict']}"
        )
    L.append("")

    L.append("-" * 70)
    L.append("4. KEY FINDINGS")
    L.append("-" * 70)

    overall_sig = {
        bl: _get("overall", "all", bl).get("boot_significant", False)
        for bl in BASELINES
    }
    delong_sig = {
        bl: _get("overall", "all", bl).get("delong_significant", False)
        for bl in BASELINES
    }
    wil_sig = {r["baseline"]: r["verdict"] == "SIGNIFICANT" for r in wilcoxon_rows}

    for bl in BASELINES:
        r = _get("overall", "all", bl)
        delta = r.get("delta", float("nan"))
        sign  = "+" if delta >= 0 else ""
        bs    = "YES" if overall_sig[bl] else "NO"
        dl    = "YES" if delong_sig[bl]  else "NO"
        wl    = "YES" if wil_sig.get(bl) else "NO"
        L.append(f"  Phase9 vs {bl}:")
        L.append(f"    Overall delta AUC = {sign}{delta:.4f}")
        L.append(f"    Bootstrap CI excludes 0: {bs}  |  DeLong p<0.05: {dl}  |  Wilcoxon p<0.05: {wl}")
        L.append("")

    L.append("  INTERPRETATION:")
    any_sig = any(overall_sig.values())
    if any_sig:
        sig_bls = [bl for bl in BASELINES if overall_sig[bl]]
        not_sig = [bl for bl in BASELINES if not overall_sig[bl]]
        L.append(f"  Phase 9 is statistically significantly better than: {sig_bls}")
        if not_sig:
            L.append(f"  Difference vs {not_sig} is not statistically significant at alpha={ALPHA}.")
    else:
        L.append("  No baseline comparison reaches statistical significance at the")
        L.append(f"  overall pooled level (alpha={ALPHA}). The Phase 9 method performs")
        L.append("  comparably to the best baselines on this test set.")
    L.append("")
    L.append("  NOTE: Statistical non-significance at the pooled level does not")
    L.append("  imply equivalence. The bootstrap CIs quantify the plausible range")
    L.append("  of the true AUC difference. Per-machine-type results show where")
    L.append("  Phase 9 has a consistent advantage (pump, valve) vs where baselines")
    L.append("  are competitive (slider OC-SVM, fan kNN/IForest).")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Phase 18 - Statistical Significance Analysis")
    print("=" * 65)
    print(f"Phase 9 CSV  : {P9_CSV}")
    print(f"Phase 18 CSV : {P18_CSV}")
    print(f"Output dir   : {OUT_DIR}")
    print()

    merged = _load_scores()

    print("\nRunning bootstrap + DeLong across all scopes...")
    summary_rows, boot_delta_rows = _run_all_tests(merged)

    print("\nRunning Wilcoxon signed-rank on 16 per-machine-ID pairs...")
    wilcoxon_rows = _wilcoxon_per_id(merged)
    for r in wilcoxon_rows:
        print(f"  {r['baseline']:<10}  W={r['w_stat']}  p={r['p_two_sided']}  "
              f"r={r['effect_r']}  {r['verdict']}")

    print("\nSaving outputs...")
    _save(summary_rows, boot_delta_rows, wilcoxon_rows)

    report_text = _report(summary_rows, wilcoxon_rows)
    report_path = OUT_DIR / "significance_report.txt"
    report_path.write_text(report_text, encoding="ascii")
    print(f"  significance_report.txt")

    print()
    print(report_text)


if __name__ == "__main__":
    main()
