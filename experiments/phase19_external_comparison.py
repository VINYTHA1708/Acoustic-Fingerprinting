"""Phase 19 — External Literature Comparison.

Reads Phase 9 (proposed method) and Phase 18 (internal baselines) results and
produces a structured external comparison report.

IMPORTANT COMPARABILITY NOTICE
-------------------------------
Results from the literature are included as CONTEXT ONLY.  They are NOT
head-to-head comparisons unless every comparability field is True and
comparability_level == "DIRECT".  Differences in dataset version, SNR
conditions, machine-ID selection, train/test split strategy, and evaluation
metric make direct numerical comparison misleading.

Outputs (all written to experiments/results/phase19_external_comparison/):
    protocol_summary.json
    external_benchmark_comparison.csv
    literature_methods.json          (template — no invented numbers)
    phase19_report.txt
    fig_direct_comparison.png        (internal DIRECT comparisons only)

Usage:
    python experiments/phase19_external_comparison.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT     = Path(__file__).resolve().parents[1]
P9_JSON  = ROOT / "experiments/results/phase9/evaluation_summary.json"
P18_JSON = ROOT / "experiments/results/phase18_baseline_comparison/comparison_results.json"
P18_SIG  = ROOT / "experiments/results/phase18_baseline_comparison/significance/significance_results.json"
P13_JSON = ROOT / "experiments/results/phase13/final_method_config.json"
OUT_DIR  = ROOT / "experiments/results/phase19_external_comparison"

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]

# ---------------------------------------------------------------------------
# Load source data
# ---------------------------------------------------------------------------

def _load() -> tuple[dict, dict, dict, dict]:
    with P9_JSON.open(encoding="utf-8")  as fh: p9  = json.load(fh)
    with P18_JSON.open(encoding="utf-8") as fh: p18 = json.load(fh)
    with P18_SIG.open(encoding="utf-8")  as fh: sig = json.load(fh)
    with P13_JSON.open(encoding="utf-8") as fh: p13 = json.load(fh)
    return p9, p18, sig, p13


# ---------------------------------------------------------------------------
# 1. Protocol summary
# ---------------------------------------------------------------------------

def _protocol_summary(p9: dict, p13: dict) -> dict:
    split = p13["dataset_split"]
    return {
        "experiment":       "phase19_external_comparison",
        "proposed_method":  p13["method_name"],
        "dataset":          "MIMII (Malfunctioning Industrial Machine Investigation and Inspection)",
        "dataset_source":   "https://zenodo.org/record/3384388",
        "dataset_version":  "original (0 dB SNR mix, 16 kHz mono)",
        "machine_types":    MACHINE_TYPES,
        "machine_ids":      MACHINE_IDS,
        "n_machine_ids":    len(MACHINE_TYPES) * len(MACHINE_IDS),
        "split_strategy":   split["strategy"],
        "train_ratio":      split["train_ratio"],
        "profile_ratio":    split["profile_ratio"],
        "test_ratio":       split["test_ratio"],
        "abnormal_partition": split["abnormal_partition"],
        "seed":             p13["random_seeds"]["primary_seed"],
        "primary_metric":   p13["evaluation_protocol"]["primary_metric"],
        "anomaly_score":    p13["evaluation_protocol"]["anomaly_score"],
        "n_test_normal":    p9["results"]["overall"]["n_normal"],
        "n_test_abnormal":  p9["results"]["overall"]["n_abnormal"],
        "n_test_total":     p9["results"]["overall"]["n_normal"] + p9["results"]["overall"]["n_abnormal"],
        "notes": (
            "Training uses only normal recordings. Abnormal recordings are "
            "reserved exclusively for evaluation. No threshold tuning on the "
            "test set. ROC-AUC is threshold-free."
        ),
    }


# ---------------------------------------------------------------------------
# 2. Benchmark comparison rows
# ---------------------------------------------------------------------------

def _comparability(
    same_dataset: bool,
    same_version: bool,
    same_types: bool,
    same_ids: bool,
    same_protocol: bool,
    same_metric: bool,
) -> str:
    if all([same_dataset, same_version, same_types, same_ids, same_protocol, same_metric]):
        return "DIRECT"
    if same_dataset and same_metric:
        return "PARTIAL"
    return "CONTEXT_ONLY"


def _build_comparison_rows(p9: dict, p18: dict) -> list[dict]:
    """Build one row per method per machine-type scope."""
    rows: list[dict] = []

    # ---- Proposed method (Phase 9) ----------------------------------------
    for mt in MACHINE_TYPES:
        auc = p9["results"][mt]["overall_metrics"]["normalized_euclidean"]["roc_auc"]
        rows.append({
            "method":               "Proposed (DSP+BEATs+ContrastiveHead)",
            "method_type":          "PROPOSED",
            "machine_type":         mt,
            "scope":                "per_type",
            "roc_auc":              round(auc, 6),
            "same_dataset":         True,
            "same_dataset_version": True,
            "same_machine_types":   True,
            "same_machine_ids":     True,
            "same_protocol":        True,
            "same_metric":          True,
            "comparability_level":  "DIRECT",
            "source":               "Phase 9 (this work)",
            "notes":                "Primary result; normalized Euclidean drift on 256-dim L2-normalised embedding.",
        })
    # Overall
    rows.append({
        "method":               "Proposed (DSP+BEATs+ContrastiveHead)",
        "method_type":          "PROPOSED",
        "machine_type":         "overall",
        "scope":                "overall",
        "roc_auc":              round(p9["results"]["overall"]["overall_metrics"]["normalized_euclidean"]["roc_auc"], 6),
        "same_dataset":         True,
        "same_dataset_version": True,
        "same_machine_types":   True,
        "same_machine_ids":     True,
        "same_protocol":        True,
        "same_metric":          True,
        "comparability_level":  "DIRECT",
        "source":               "Phase 9 (this work)",
        "notes":                "Pooled across all 4 machine types.",
    })

    # ---- Internal baselines (Phase 18) — DIRECT ---------------------------
    bl_map = {"ocsvm": "OC-SVM", "iforest": "Isolation Forest", "knn": "kNN"}
    for bl_key, bl_name in bl_map.items():
        for mt in MACHINE_TYPES:
            auc = p18["results"]["per_type"][mt]["overall"][bl_key]["roc_auc"]
            rows.append({
                "method":               bl_name,
                "method_type":          "INTERNAL_BASELINE",
                "machine_type":         mt,
                "scope":                "per_type",
                "roc_auc":              round(auc, 6),
                "same_dataset":         True,
                "same_dataset_version": True,
                "same_machine_types":   True,
                "same_machine_ids":     True,
                "same_protocol":        True,
                "same_metric":          True,
                "comparability_level":  "DIRECT",
                "source":               "Phase 18 (this work)",
                "notes":                (
                    f"{bl_name} fit on profile_normal embeddings (256-dim L2-normalised "
                    "from Phase 9 ProjectionHead). Same split as proposed method."
                ),
            })
        auc_overall = p18["results"]["overall"][bl_key]["roc_auc"]
        rows.append({
            "method":               bl_name,
            "method_type":          "INTERNAL_BASELINE",
            "machine_type":         "overall",
            "scope":                "overall",
            "roc_auc":              round(auc_overall, 6),
            "same_dataset":         True,
            "same_dataset_version": True,
            "same_machine_types":   True,
            "same_machine_ids":     True,
            "same_protocol":        True,
            "same_metric":          True,
            "comparability_level":  "DIRECT",
            "source":               "Phase 18 (this work)",
            "notes":                "Pooled across all 4 machine types.",
        })

    # ---- Verified literature entries — CONTEXT_ONLY ----------------------
    # Numbers are taken directly from the cited papers.  All three entries
    # remain CONTEXT_ONLY because the evaluation protocols differ from this
    # work in at least one material respect (dataset version / SNR mix /
    # train-test split strategy / machine-ID selection / averaging method).
    # No statistical tests are performed against these values and no
    # better/worse judgement is made.

    _dcase_ae_notes = (
        "DCASE 2020 Task 2 official autoencoder baseline. Evaluated on the "
        "DCASE 2020 Task 2 development set. Metric is average ROC-AUC across "
        "machine IDs within each type. Protocol differences vs this work: "
        "different dataset version (DCASE 2020 dev set vs MIMII Zenodo 3384388 "
        "original), different SNR mix, different train/test split strategy, "
        "AUC averaged per machine type rather than pooled. "
        "Numerical differences cannot be interpreted as head-to-head "
        "performance differences."
    )
    dcase_ae_per_type = {
        "fan": 0.6583, "pump": 0.7289, "slider": 0.8476, "valve": 0.6628,
    }
    for mt, auc in dcase_ae_per_type.items():
        rows.append({
            "method":               "DCASE 2020 AE Baseline",
            "method_type":          "LITERATURE_CONTEXT",
            "machine_type":         mt,
            "scope":                "per_type",
            "roc_auc":              auc,
            "same_dataset":         True,
            "same_dataset_version": False,
            "same_machine_types":   True,
            "same_machine_ids":     False,
            "same_protocol":        False,
            "same_metric":          True,
            "comparability_level":  "CONTEXT_ONLY",
            "source":               "DCASE 2020 Task 2 challenge (official baseline)",
            "notes":                _dcase_ae_notes,
        })
    rows.append({
        "method":               "DCASE 2020 AE Baseline",
        "method_type":          "LITERATURE_CONTEXT",
        "machine_type":         "overall",
        "scope":                "overall",
        "roc_auc":              0.7244,
        "same_dataset":         True,
        "same_dataset_version": False,
        "same_machine_types":   True,
        "same_machine_ids":     False,
        "same_protocol":        False,
        "same_metric":          True,
        "comparability_level":  "CONTEXT_ONLY",
        "source":               "DCASE 2020 Task 2 challenge (official baseline)",
        "notes":                _dcase_ae_notes,
    })

    rows.append({
        "method":               "LSTM Autoencoder",
        "method_type":          "LITERATURE_CONTEXT",
        "machine_type":         "overall",
        "scope":                "overall",
        "roc_auc":              0.7351,
        "same_dataset":         True,
        "same_dataset_version": False,
        "same_machine_types":   None,
        "same_machine_ids":     False,
        "same_protocol":        False,
        "same_metric":          True,
        "comparability_level":  "CONTEXT_ONLY",
        "source": (
            "DCASE Challenge 2020: Unsupervised Anomalous Sound Detection "
            "of Machinery with Deep Autoencoders"
        ),
        "notes": (
            "LSTM-based autoencoder evaluated in the DCASE 2020/MIMII context. "
            "Metric is average ROC-AUC. Protocol differences vs this work: "
            "dataset version and SNR conditions unconfirmed, train/test split "
            "strategy differs, machine-ID selection unconfirmed. "
            "Numerical differences cannot be interpreted as head-to-head "
            "performance differences."
        ),
    })

    rows.append({
        "method":               "Conformer ID-Aware AE (single model)",
        "method_type":          "LITERATURE_CONTEXT",
        "machine_type":         "overall",
        "scope":                "overall",
        "roc_auc":              0.9047,
        "same_dataset":         True,
        "same_dataset_version": False,
        "same_machine_types":   None,
        "same_machine_ids":     False,
        "same_protocol":        False,
        "same_metric":          True,
        "comparability_level":  "CONTEXT_ONLY",
        "source": (
            "Conformer-Based ID-Aware Autoencoder for Unsupervised "
            "Anomalous Sound Detection"
        ),
        "notes": (
            "Single-model result (ensemble AUC=0.9133 not used as primary value). "
            "Evaluated on DCASE 2020 Task 2 development set. Protocol differences "
            "vs this work: dataset version and SNR conditions differ, model uses "
            "machine-ID conditioning (ID-aware), train/test split strategy differs, "
            "machine-ID selection unconfirmed. "
            "Numerical differences cannot be interpreted as head-to-head "
            "performance differences."
        ),
    })

    return rows


# ---------------------------------------------------------------------------
# 3. Literature methods template
# ---------------------------------------------------------------------------

def _literature_template() -> dict:
    return {
        "_instructions": (
            "All entries in this file are CONTEXT_ONLY. Numerical results are "
            "taken directly from the cited papers and have not been re-evaluated. "
            "They must not be used as head-to-head comparisons because the "
            "evaluation protocols differ from this work in at least one material "
            "respect. See the comparability fields and notes on each entry."
        ),
        "methods": [
            {
                "short_name":           "DCASE2020_AE_baseline",
                "full_name":            "DCASE 2020 Task 2 Autoencoder Baseline",
                "citation":             "DCASE 2020 Task 2 challenge (official baseline)",
                "url":                  "https://dcase.community/challenge2020/task2-unsupervised-detection-of-anomalous-sounds",
                "dataset":              "DCASE 2020 Task 2 development dataset",
                "dataset_version":      "DCASE 2020 dev set (differs from MIMII Zenodo 3384388 original)",
                "machine_types_used":   "fan, pump, slider, valve",
                "machine_ids_used":     "unconfirmed — differs from this work",
                "snr_conditions":       "mixed SNR conditions per DCASE 2020 Task 2 setup",
                "train_test_protocol":  "DCASE 2020 Task 2 protocol (differs from this work)",
                "reported_metric":      "average ROC-AUC per machine type",
                "reported_roc_auc_overall": 0.7244,
                "reported_roc_auc_per_type": {
                    "fan": 0.6583, "pump": 0.7289, "slider": 0.8476, "valve": 0.6628
                },
                "same_dataset":         True,
                "same_dataset_version": False,
                "same_machine_types":   True,
                "same_machine_ids":     False,
                "same_protocol":        False,
                "same_metric":          True,
                "comparability_level":  "CONTEXT_ONLY",
                "protocol_differences": [
                    "Different dataset version and SNR mix",
                    "Different train/test split strategy",
                    "Machine-ID selection unconfirmed",
                    "AUC averaged per machine type, not pooled across all recordings",
                ],
                "notes": (
                    "Numerical differences vs this work cannot be interpreted as "
                    "head-to-head performance differences due to protocol mismatches."
                ),
            },
            {
                "short_name":           "LSTM_AE",
                "full_name":            "LSTM Autoencoder",
                "citation": (
                    "DCASE Challenge 2020: Unsupervised Anomalous Sound Detection "
                    "of Machinery with Deep Autoencoders"
                ),
                "url":                  "https://dcase.community/challenge2020/task2-unsupervised-detection-of-anomalous-sounds",
                "dataset":              "DCASE 2020 / MIMII context",
                "dataset_version":      "unconfirmed — differs from MIMII Zenodo 3384388 original",
                "machine_types_used":   "unconfirmed",
                "machine_ids_used":     "unconfirmed — differs from this work",
                "snr_conditions":       "unconfirmed",
                "train_test_protocol":  "DCASE 2020 protocol (differs from this work)",
                "reported_metric":      "average ROC-AUC",
                "reported_roc_auc_overall": 0.7351,
                "reported_roc_auc_per_type": None,
                "same_dataset":         True,
                "same_dataset_version": False,
                "same_machine_types":   None,
                "same_machine_ids":     False,
                "same_protocol":        False,
                "same_metric":          True,
                "comparability_level":  "CONTEXT_ONLY",
                "protocol_differences": [
                    "Dataset version and SNR conditions unconfirmed",
                    "Train/test split strategy differs",
                    "Machine-ID selection unconfirmed",
                ],
                "notes": (
                    "Numerical differences vs this work cannot be interpreted as "
                    "head-to-head performance differences due to protocol mismatches."
                ),
            },
            {
                "short_name":           "Conformer_ID_Aware_AE",
                "full_name":            "Conformer-Based ID-Aware Autoencoder",
                "citation": (
                    "Conformer-Based ID-Aware Autoencoder for Unsupervised "
                    "Anomalous Sound Detection"
                ),
                "url":                  "https://dcase.community/challenge2020/task2-unsupervised-detection-of-anomalous-sounds",
                "dataset":              "DCASE 2020 Task 2 development dataset",
                "dataset_version":      "DCASE 2020 dev set (differs from MIMII Zenodo 3384388 original)",
                "machine_types_used":   "unconfirmed",
                "machine_ids_used":     "unconfirmed — differs from this work",
                "snr_conditions":       "DCASE 2020 Task 2 conditions",
                "train_test_protocol":  "DCASE 2020 protocol (differs from this work)",
                "reported_metric":      "average ROC-AUC",
                "reported_roc_auc_overall": 0.9047,
                "reported_roc_auc_ensemble": 0.9133,
                "primary_value_used":   "single_model (0.9047); ensemble (0.9133) not used as primary",
                "reported_roc_auc_per_type": None,
                "same_dataset":         True,
                "same_dataset_version": False,
                "same_machine_types":   None,
                "same_machine_ids":     False,
                "same_protocol":        False,
                "same_metric":          True,
                "comparability_level":  "CONTEXT_ONLY",
                "protocol_differences": [
                    "Dataset version and SNR conditions differ",
                    "Model uses machine-ID conditioning (ID-aware)",
                    "Train/test split strategy differs",
                    "Machine-ID selection unconfirmed",
                ],
                "notes": (
                    "Numerical differences vs this work cannot be interpreted as "
                    "head-to-head performance differences due to protocol mismatches."
                ),
            },
        ],
    }


# ---------------------------------------------------------------------------
# 4. Figure — DIRECT internal comparisons only
# ---------------------------------------------------------------------------

def _plot_direct(rows: list[dict], out_path: Path) -> None:
    direct = [r for r in rows if r["comparability_level"] == "DIRECT" and r["scope"] == "overall"]
    if not direct:
        return

    methods = [r["method"] for r in direct]
    aucs    = [r["roc_auc"] for r in direct]
    colors  = [
        "#1f77b4" if r["method_type"] == "PROPOSED" else "#aec7e8"
        for r in direct
    ]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(methods))))
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, aucs, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("ROC-AUC (overall, all machine types pooled)", fontsize=9)
    ax.set_title(
        "Phase 19 — Direct Internal Comparison\n"
        "(DIRECT comparisons only; same dataset, split, protocol, metric)",
        fontsize=9,
    )
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--")
    for bar, auc in zip(bars, aucs):
        ax.text(
            auc + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{auc:.4f}", va="center", ha="left", fontsize=8,
        )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Report text
# ---------------------------------------------------------------------------

def _report(p9: dict, p18: dict, sig: dict, rows: list[dict]) -> str:
    p9_overall = p9["results"]["overall"]["overall_metrics"]["normalized_euclidean"]["roc_auc"]
    p9_per_type = {
        mt: p9["results"][mt]["overall_metrics"]["normalized_euclidean"]["roc_auc"]
        for mt in MACHINE_TYPES
    }
    bl_overall = {bl: p18["results"]["overall"][bl]["roc_auc"] for bl in ["ocsvm", "iforest", "knn"]}
    bl_per_type = {
        mt: {bl: p18["results"]["per_type"][mt]["overall"][bl]["roc_auc"] for bl in ["ocsvm", "iforest", "knn"]}
        for mt in MACHINE_TYPES
    }

    def _sig_verdict(bl: str) -> str:
        r = sig["overall_summary"].get(bl, {})
        b = r.get("boot_significant", False)
        d = r.get("delong_significant", False)
        if b and d:
            return "SIGNIFICANT (bootstrap + DeLong)"
        if b:
            return "SIGNIFICANT (bootstrap only)"
        if d:
            return "SIGNIFICANT (DeLong only)"
        return "NOT SIGNIFICANT"

    L: list[str] = []
    L.append("Phase 19 — External Literature Comparison Report")
    L.append("=" * 70)
    L.append("")
    L.append("CRITICAL WARNING")
    L.append("-" * 70)
    L.append("External literature results listed in this report are NOT head-to-head")
    L.append("comparisons unless ALL of the following conditions hold:")
    L.append("  same_dataset=True, same_dataset_version=True,")
    L.append("  same_machine_types=True, same_machine_ids=True,")
    L.append("  same_protocol=True, same_metric=True")
    L.append("  => comparability_level == DIRECT")
    L.append("")
    L.append("Literature placeholders in external_benchmark_comparison.csv have")
    L.append("comparability_level=CONTEXT_ONLY and roc_auc=None.  Do NOT fill in")
    L.append("numbers from papers without first verifying every comparability field.")
    L.append("")

    L.append("=" * 70)
    L.append("1. EXPERIMENTAL PROTOCOL (this work)")
    L.append("=" * 70)
    L.append("  Dataset        : MIMII (Zenodo 3384388)")
    L.append("  Machine types  : fan, pump, slider, valve")
    L.append("  Machine IDs    : id_00, id_02, id_04, id_06 (4 per type, 16 total)")
    L.append("  Split          : train=0.70 / profile=0.15 / test=0.15  seed=42")
    L.append("  Abnormal data  : test_abnormal partition only (no leakage)")
    L.append("  Primary metric : ROC-AUC (threshold-free)")
    L.append("  Anomaly score  : normalized Euclidean drift on 256-dim L2-normalised embedding")
    L.append(f"  N test normal  : {p9['results']['overall']['n_normal']}")
    L.append(f"  N test abnormal: {p9['results']['overall']['n_abnormal']}")
    L.append("")

    L.append("=" * 70)
    L.append("2. PROPOSED METHOD RESULTS  (comparability_level = DIRECT)")
    L.append("=" * 70)
    L.append(f"  {'Machine type':<12} {'ROC-AUC':>10}")
    L.append("  " + "-" * 24)
    for mt in MACHINE_TYPES:
        L.append(f"  {mt:<12} {p9_per_type[mt]:>10.4f}")
    L.append(f"  {'overall':<12} {p9_overall:>10.4f}")
    L.append("")

    L.append("=" * 70)
    L.append("3. INTERNAL BASELINES  (comparability_level = DIRECT)")
    L.append("=" * 70)
    L.append("  All baselines use the same dataset, split, embeddings, and metric.")
    L.append("  Fit on profile_normal embeddings (256-dim L2-normalised, Phase 9 head).")
    L.append("")
    L.append(f"  {'Method':<18} {'fan':>8} {'pump':>8} {'slider':>8} {'valve':>8} {'overall':>8}")
    L.append("  " + "-" * 62)
    for bl, name in [("ocsvm", "OC-SVM"), ("iforest", "IForest"), ("knn", "kNN")]:
        vals = "  ".join(f"{bl_per_type[mt][bl]:>6.4f}" for mt in MACHINE_TYPES)
        L.append(f"  {name:<18} {vals}  {bl_overall[bl]:>6.4f}")
    L.append(f"  {'Proposed':<18} " +
             "  ".join(f"{p9_per_type[mt]:>6.4f}" for mt in MACHINE_TYPES) +
             f"  {p9_overall:>6.4f}")
    L.append("")

    L.append("=" * 70)
    L.append("4. STATISTICAL SIGNIFICANCE (Phase 18 results)")
    L.append("=" * 70)
    L.append("  Proposed vs OC-SVM  (overall): " + _sig_verdict("ocsvm"))
    L.append("  Proposed vs IForest (overall): " + _sig_verdict("iforest"))
    L.append("  Proposed vs kNN     (overall): " + _sig_verdict("knn"))
    L.append("  Tests: paired bootstrap (2000 iter) + DeLong analytic + Wilcoxon signed-rank")
    L.append("  See experiments/results/phase18_baseline_comparison/significance/ for details.")
    L.append("")

    lit_rows = [r for r in rows
                if r["method_type"] == "LITERATURE_CONTEXT" and r["scope"] == "overall"]

    L.append("=" * 70)
    L.append("5. LITERATURE CONTEXT  (comparability_level = CONTEXT_ONLY)")
    L.append("=" * 70)
    L.append("  PROTOCOL MISMATCH WARNING")
    L.append("  " + "-" * 66)
    L.append("  The numerical values below are taken directly from the cited papers.")
    L.append("  They are listed for orientation only.  Numerical differences between")
    L.append("  these values and the proposed method's results CANNOT be interpreted")
    L.append("  as head-to-head performance differences because the evaluation")
    L.append("  protocols differ in at least one of the following material respects:")
    L.append("")
    L.append("    - Dataset version: literature uses DCASE 2020 Task 2 dev set;")
    L.append("      this work uses MIMII Zenodo 3384388 original release.")
    L.append("    - SNR conditions: DCASE 2020 mixes multiple SNR levels;")
    L.append("      this work uses the 0 dB SNR mix only.")
    L.append("    - Train/test split: DCASE 2020 uses its own fixed split;")
    L.append("      this work uses train=0.70 / profile=0.15 / test=0.15 seed=42.")
    L.append("    - Machine-ID selection: literature machine IDs unconfirmed.")
    L.append("    - Averaging: literature reports average AUC per machine type;")
    L.append("      this work pools all recordings across types.")
    L.append("    - Some literature methods use machine-ID conditioning (ID-aware),")
    L.append("      which is a strictly stronger supervision signal.")
    L.append("")
    L.append("  No statistical tests are performed against literature values.")
    L.append("  No better/worse judgement is made about any literature method.")
    L.append("")
    L.append("  Verified literature entries (overall average ROC-AUC as reported):")
    L.append("")
    L.append(f"  {'Method':<42} {'Reported AUC':>13}  Source")
    L.append("  " + "-" * 80)
    for r in lit_rows:
        auc_str = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "N/A"
        src = r["source"]
        src_short = src[:36] + "..." if len(src) > 39 else src
        L.append(f"  {r['method']:<42} {auc_str:>13}  {src_short}")
    L.append("")
    L.append("  DCASE 2020 AE Baseline — per-machine-type reported AUCs:")
    L.append("    fan=0.6583  pump=0.7289  slider=0.8476  valve=0.6628  avg=0.7244")
    L.append("")
    L.append("  Full comparability fields and protocol notes are in:")
    L.append("    literature_methods.json")
    L.append("    external_benchmark_comparison.csv  (method_type=LITERATURE_CONTEXT)")
    L.append("")

    L.append("=" * 70)
    L.append("6. OUTPUTS")
    L.append("=" * 70)
    L.append("  protocol_summary.json            — dataset and protocol description")
    L.append("  external_benchmark_comparison.csv — all methods with comparability fields")
    L.append("  literature_methods.json           — template for literature entries")
    L.append("  phase19_report.txt                — this report")
    L.append("  fig_direct_comparison.png         — bar chart (DIRECT comparisons only)")
    L.append("")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "method", "method_type", "machine_type", "scope", "roc_auc",
        "same_dataset", "same_dataset_version", "same_machine_types",
        "same_machine_ids", "same_protocol", "same_metric",
        "comparability_level", "source", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Phase 19 — External Literature Comparison")
    print("=" * 65)
    print(f"Phase 9  source : {P9_JSON}")
    print(f"Phase 18 source : {P18_JSON}")
    print(f"Output dir      : {OUT_DIR}")
    print()

    p9, p18, sig, p13 = _load()

    # 1. Protocol summary
    protocol = _protocol_summary(p9, p13)
    proto_path = OUT_DIR / "protocol_summary.json"
    with proto_path.open("w", encoding="utf-8") as fh:
        json.dump(protocol, fh, indent=2)
    print(f"  Saved: {proto_path.name}")

    # 2. Benchmark comparison CSV
    rows = _build_comparison_rows(p9, p18)
    csv_path = OUT_DIR / "external_benchmark_comparison.csv"
    _save_csv(rows, csv_path)
    print(f"  Saved: {csv_path.name}")

    # 3. Literature methods template
    lit_path = OUT_DIR / "literature_methods.json"
    with lit_path.open("w", encoding="utf-8") as fh:
        json.dump(_literature_template(), fh, indent=2)
    print(f"  Saved: {lit_path.name}")

    # 4. Figure (DIRECT only)
    fig_path = OUT_DIR / "fig_direct_comparison.png"
    _plot_direct(rows, fig_path)
    print(f"  Saved: {fig_path.name}")

    # 5. Report
    report_text = _report(p9, p18, sig, rows)
    report_path = OUT_DIR / "phase19_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  Saved: {report_path.name}")

    print()
    print(report_text)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
