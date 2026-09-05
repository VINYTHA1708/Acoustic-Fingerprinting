"""Phase 15 — Explainability of the Final Frozen Method.

Explainability method:
  1. DSP feature-group contribution: For each recording, compute the mean
     absolute z-score within each named DSP feature group (MFCC, spectral,
     temporal, harmonic) using the fusion cache vectors and the healthy
     profile's mean/std. Higher z-score = larger deviation from healthy.

  2. Embedding dimension drift: Compute the per-dimension z-score vector
     z_i = (e_i - mu_i) / sigma_i for the 256-dim learned fingerprint.
     Identify the top-K dimensions with the largest |z_i| — these are the
     embedding dimensions that contribute most to the anomaly score.

  3. Drift decomposition: Report normalized_euclidean (primary anomaly score),
     health_score, and health_state for each recording.

No model retraining. No invented importance scores. All values are computed
directly from real cached fusion vectors and saved profiles.

Outputs (experiments/results/phase15/):
  - explainability_results.csv
  - explainability_summary.md
  - fig1_dsp_group_drift_by_label.png
  - fig2_top_embedding_dims_abnormal.png
  - fig3_health_score_vs_norm_euclidean.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fusion.serializer import FusedVectorSerializer

# ── Constants ────────────────────────────────────────────────────────────────

CHECKPOINT   = ROOT / "models" / "contrastive" / "phase9" / "best_projection_head.pt"
PROFILES_DIR = ROOT / "experiments" / "results" / "phase9" / "profiles"
EVAL_CSV     = ROOT / "experiments" / "results" / "e1" / "evaluation_results.csv"
CACHE_ROOT   = ROOT / "data" / "fusion_cache"
OUT_DIR      = ROOT / "experiments" / "results" / "phase15"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Representative subset: pump only (best-performing machine type, AUC=0.864)
MACHINE_TYPE = "pump"
MACHINE_IDS  = ["id_00", "id_02", "id_04", "id_06"]
N_PER_CLASS  = 15   # healthy + abnormal per machine ID
TOP_K_DIMS   = 20   # top embedding dimensions to report

# DSP feature group boundaries (indices into the 153-dim DSP vector)
DSP_GROUPS = {
    "MFCC (mean+std)":        (0,   40),
    "MFCC delta (mean+std)":  (40,  80),
    "MFCC delta2 (mean+std)": (80, 120),
    "Spectral":               (120, 136),
    "Temporal":               (136, 146),
    "Harmonic":               (146, 153),
}

# ── Helpers ──────────────────────────────────────────────────────────────────

_serializer = FusedVectorSerializer()


def load_profile_npz(machine_type: str, machine_id: str) -> dict:
    path = PROFILES_DIR / f"phase9_{machine_type}_{machine_id}_learned_profile.npz"
    data = np.load(path)
    return {
        "embeddings":  data["embeddings"].astype(np.float32),   # (N, 256)
        "mean_vector": data["mean_vector"].astype(np.float32),  # (256,)
        "std_vector":  data["std_vector"].astype(np.float32),   # (256,)
    }


def load_fused(machine_type: str, machine_id: str, label: str, filename: str):
    stem = Path(filename).stem
    path = CACHE_ROOT / machine_type / machine_id / label / f"{stem}.npz"
    if not path.exists():
        return None
    return _serializer.load_npz(path)


def dsp_group_zscores(dsp_vec: np.ndarray,
                      profile_dsp_mean: np.ndarray,
                      profile_dsp_std: np.ndarray) -> dict[str, float]:
    """Mean absolute z-score per DSP feature group."""
    safe_std = np.where(profile_dsp_std < 1e-10, 1.0, profile_dsp_std)
    z = np.where(profile_dsp_std < 1e-10, 0.0,
                 (dsp_vec - profile_dsp_mean) / safe_std)
    result = {}
    for group, (lo, hi) in DSP_GROUPS.items():
        result[group] = float(np.mean(np.abs(z[lo:hi])))
    return result


def embedding_dim_zscores(embedding: np.ndarray,
                          mean_vec: np.ndarray,
                          std_vec: np.ndarray) -> np.ndarray:
    """Per-dimension z-score for the 256-dim learned fingerprint."""
    safe_std = np.where(std_vec < 1e-10, 1.0, std_vec)
    return np.where(std_vec < 1e-10, 0.0,
                    (embedding - mean_vec) / safe_std).astype(np.float32)


# ── Load evaluation results ───────────────────────────────────────────────────

eval_df = pd.read_csv(EVAL_CSV)
eval_df = eval_df[eval_df["machine_type"] == MACHINE_TYPE].copy()

# ── Main loop ────────────────────────────────────────────────────────────────

print("\nPhase 15 - Explainability  ({})".format(MACHINE_TYPE))
print("=" * 60)

rows = []

for mid in MACHINE_IDS:
    print(f"\n  {MACHINE_TYPE}/{mid}")

    profile = load_profile_npz(MACHINE_TYPE, mid)
    mean_emb = profile["mean_vector"]
    std_emb  = profile["std_vector"]

    # Build DSP profile mean/std from all cached normal vectors
    normal_cache = list((CACHE_ROOT / MACHINE_TYPE / mid / "normal").glob("*.npz"))
    dsp_vecs = []
    for p in normal_cache:
        fv = _serializer.load_npz(p)
        dsp_vecs.append(fv.dsp_feature_vector)
    dsp_matrix     = np.stack(dsp_vecs, axis=0).astype(np.float32)
    dsp_mean       = dsp_matrix.mean(axis=0)
    dsp_std        = dsp_matrix.std(axis=0)
    dsp_feat_names = _serializer.load_npz(normal_cache[0]).dsp_feature_names

    # Sample recordings from eval CSV
    sub = eval_df[eval_df["machine_id"] == mid]
    normal_rows   = sub[sub["true_label"] == "normal"].head(N_PER_CLASS)
    abnormal_rows = sub[sub["true_label"] == "abnormal"].head(N_PER_CLASS)
    selected = pd.concat([normal_rows, abnormal_rows], ignore_index=True)

    for _, row in selected.iterrows():
        label    = row["true_label"]
        filename = row["filename"]
        fv = load_fused(MACHINE_TYPE, mid, label, filename)
        if fv is None:
            print(f"    SKIP (no cache): {filename}")
            continue

        dsp_vec   = fv.dsp_feature_vector
        emb_cache = fv.fused_feature_vector  # 921-dim; we need the embedding

        # Compute embedding via ProjectionHead (lazy import to avoid slow init
        # on every iteration — load once per machine ID below)
        # We use the pre-computed drift from eval_df directly for the primary
        # anomaly score; embedding z-scores are computed from profile.
        norm_euclid  = float(row["normalized_euclidean"])
        health_score = float(row["health_score"])
        health_state = str(row["health_state"])

        # DSP group contributions
        group_z = dsp_group_zscores(dsp_vec, dsp_mean, dsp_std)

        # Embedding z-scores: we need the actual embedding.
        # Load it from the profile embeddings matrix by matching filename
        # (profile only has profile_normal; for test recordings we must
        # recompute — but we can approximate using the fused vector directly
        # projected through the saved ProjectionHead).
        # We defer the full embedding computation to a single batch below.
        rows.append({
            "machine_type":   MACHINE_TYPE,
            "machine_id":     mid,
            "filename":       filename,
            "true_label":     label,
            "health_score":   round(health_score, 4),
            "health_state":   health_state,
            "norm_euclidean": round(norm_euclid, 4),
            **{f"dsp_group_{k.split('(')[0].strip().replace(' ','_')}": round(v, 6)
               for k, v in group_z.items()},
            "_dsp_vec":       dsp_vec,
        })

    print(f"    Collected {len([r for r in rows if r['machine_id']==mid])} recordings")

# ── Compute embeddings in batch via ProjectionHead ────────────────────────────

print("\nComputing learned embeddings via ProjectionHead...")
import torch
from src.contrastive_learning.model import ProjectionHead
from src.fusion.serializer import FusedVectorSerializer as _FS

head = ProjectionHead()
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
head.load_state_dict(ckpt["model_state_dict"])
head.eval()

for row in rows:
    mid      = row["machine_id"]
    label    = row["true_label"]
    filename = row["filename"]
    fv = load_fused(MACHINE_TYPE, mid, label, filename)
    if fv is None:
        row["top_dim_indices"] = []
        row["top_dim_abs_z"]   = []
        continue

    fused_t = torch.tensor(fv.fused_feature_vector, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = head(fused_t).squeeze(0).numpy()

    profile = load_profile_npz(MACHINE_TYPE, mid)
    z_vec   = embedding_dim_zscores(emb, profile["mean_vector"], profile["std_vector"])
    abs_z   = np.abs(z_vec)
    top_idx = np.argsort(abs_z)[::-1][:TOP_K_DIMS]

    row["top_dim_indices"] = top_idx.tolist()
    row["top_dim_abs_z"]   = abs_z[top_idx].tolist()
    row["norm_euclidean_recomputed"] = round(float(np.linalg.norm(z_vec)), 4)

print("  Done.")

# ── Build results DataFrame ───────────────────────────────────────────────────

dsp_group_cols = [c for c in rows[0].keys()
                  if c.startswith("dsp_group_")]

result_rows = []
for r in rows:
    result_rows.append({
        "machine_type":   r["machine_type"],
        "machine_id":     r["machine_id"],
        "filename":       r["filename"],
        "true_label":     r["true_label"],
        "health_score":   r["health_score"],
        "health_state":   r["health_state"],
        "norm_euclidean": r["norm_euclidean"],
        "norm_euclidean_recomputed": r.get("norm_euclidean_recomputed", np.nan),
        **{c: r[c] for c in dsp_group_cols},
        "top_embedding_dims":   str(r.get("top_dim_indices", [])),
        "top_embedding_abs_z":  str([round(v, 4) for v in r.get("top_dim_abs_z", [])]),
    })

results_df = pd.DataFrame(result_rows)
csv_path = OUT_DIR / "explainability_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}  ({len(results_df)} rows)")

# ── Figure 1: DSP group drift by label ───────────────────────────────────────

fig1_path = OUT_DIR / "fig1_dsp_group_drift_by_label.png"

group_short = {
    "dsp_group_MFCC":         "MFCC",
    "dsp_group_MFCC_delta":   "MFCC delta",
    "dsp_group_MFCC_delta2":  "MFCC delta2",
    "dsp_group_Spectral":     "Spectral",
    "dsp_group_Temporal":     "Temporal",
    "dsp_group_Harmonic":     "Harmonic",
}
group_cols = [c for c in dsp_group_cols if c in group_short]

normal_df   = results_df[results_df["true_label"] == "normal"]
abnormal_df = results_df[results_df["true_label"] == "abnormal"]

x      = np.arange(len(group_cols))
width  = 0.35
n_mean = [normal_df[c].mean()   for c in group_cols]
n_std  = [normal_df[c].std()    for c in group_cols]
a_mean = [abnormal_df[c].mean() for c in group_cols]
a_std  = [abnormal_df[c].std()  for c in group_cols]

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, n_mean, width, yerr=n_std, label="Normal",
       color="#2196F3", alpha=0.85, capsize=4)
ax.bar(x + width/2, a_mean, width, yerr=a_std, label="Abnormal",
       color="#F44336", alpha=0.85, capsize=4)
ax.set_xticks(x)
ax.set_xticklabels([group_short[c] for c in group_cols], fontsize=11)
ax.set_ylabel("Mean Absolute Z-Score", fontsize=12)
ax.set_title(
    "DSP Feature Group Deviation from Healthy Profile\n"
    f"(pump, {N_PER_CLASS} normal + {N_PER_CLASS} abnormal per machine ID)",
    fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(fig1_path, dpi=150)
plt.close(fig)
print(f"Saved: {fig1_path}")

# ── Figure 2: Top embedding dimensions for abnormal recordings ───────────────

fig2_path = OUT_DIR / "fig2_top_embedding_dims_abnormal.png"

# Aggregate top-dim votes across all abnormal recordings
dim_vote = np.zeros(256, dtype=np.float32)
for r in rows:
    if r["true_label"] == "abnormal" and r.get("top_dim_indices"):
        for idx, az in zip(r["top_dim_indices"], r["top_dim_abs_z"]):
            dim_vote[idx] += az

top20_idx  = np.argsort(dim_vote)[::-1][:20]
top20_vals = dim_vote[top20_idx]

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(20), top20_vals, color="#F44336", alpha=0.85)
ax.set_xticks(range(20))
ax.set_xticklabels([f"d{i}" for i in top20_idx], rotation=45, fontsize=9)
ax.set_ylabel("Cumulative |z-score| across abnormal recordings", fontsize=11)
ax.set_title(
    "Top 20 Embedding Dimensions Contributing to Abnormal Drift\n"
    "(pump, all machine IDs — Phase 9 frozen model)",
    fontsize=12)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(fig2_path, dpi=150)
plt.close(fig)
print(f"Saved: {fig2_path}")

# ── Figure 3: Health score vs normalised Euclidean ───────────────────────────

fig3_path = OUT_DIR / "fig3_health_score_vs_norm_euclidean.png"

fig, ax = plt.subplots(figsize=(7, 5))
for label, color, marker in [("normal", "#2196F3", "o"), ("abnormal", "#F44336", "^")]:
    sub = results_df[results_df["true_label"] == label]
    ax.scatter(sub["norm_euclidean"], sub["health_score"],
               c=color, marker=marker, alpha=0.7, s=50, label=label.capitalize())

ax.axhline(54.52, color="orange", linestyle="--", linewidth=1.5,
           label="Anomaly threshold (54.52)")
ax.set_xlabel("Normalized Euclidean Drift  ‖z‖₂", fontsize=12)
ax.set_ylabel("Health Score [0–100]", fontsize=12)
ax.set_title(
    "Health Score vs. Normalized Euclidean Drift\n"
    "(pump — Phase 13 frozen method)",
    fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(fig3_path, dpi=150)
plt.close(fig)
print(f"Saved: {fig3_path}")

# ── Compute summary statistics ────────────────────────────────────────────────

n_normal   = int((results_df["true_label"] == "normal").sum())
n_abnormal = int((results_df["true_label"] == "abnormal").sum())

norm_drift_normal   = results_df[results_df["true_label"]=="normal"]["norm_euclidean"]
norm_drift_abnormal = results_df[results_df["true_label"]=="abnormal"]["norm_euclidean"]

hs_normal   = results_df[results_df["true_label"]=="normal"]["health_score"]
hs_abnormal = results_df[results_df["true_label"]=="abnormal"]["health_score"]

# DSP group with highest separation
sep = {}
for c in group_cols:
    nm = normal_df[c].mean()
    am = abnormal_df[c].mean()
    sep[group_short[c]] = round(am - nm, 4)
most_sep_group = max(sep, key=lambda k: sep[k])

# Top 5 embedding dims
top5_dims = top20_idx[:5].tolist()

# ── Write summary markdown ────────────────────────────────────────────────────

md_path = OUT_DIR / "explainability_summary.md"
md = f"""# Phase 15 — Explainability of the Final Frozen Method

**Status:** Analysis only — no model retraining, no backend changes.
**Method:** Phase 13 frozen configuration (Phase 9 checkpoint, seed=42).
**Scope:** {MACHINE_TYPE} machine type, IDs {MACHINE_IDS},
{N_PER_CLASS} normal + {N_PER_CLASS} abnormal recordings per machine ID.

---

## Explainability Approach

This analysis explains *why* a recording is classified as healthy or abnormal
using three complementary lenses, all computed from real cached values:

### 1. DSP Feature Group Contribution
For each recording, the mean absolute z-score is computed within each named
DSP feature group (MFCC, MFCC-delta, MFCC-delta², Spectral, Temporal,
Harmonic) relative to the healthy population's mean and std derived from all
cached normal fusion vectors. A higher value indicates greater deviation from
the healthy acoustic signature in that feature group.

**Formula:**
```
group_z(g) = mean( |( x_i - mu_i ) / max(sigma_i, 1e-10)| )  for i in group g
```

### 2. Embedding Dimension Drift
For each recording, the per-dimension z-score of the 256-dim learned
fingerprint is computed against the healthy profile (mu, sigma):
```
z_i = (e_i - mu_i) / max(sigma_i, 1e-10)
```
The top-{TOP_K_DIMS} dimensions with the largest |z_i| are identified as the
embedding dimensions contributing most to the anomaly score.

### 3. Drift Decomposition
The primary anomaly score `normalized_euclidean = ‖z‖₂` and the derived
`health_score` are reported per recording, enabling direct traceability from
acoustic input to classification decision.

---

## Dataset Summary

| Partition | Count |
|---|---|
| Normal recordings | {n_normal} |
| Abnormal recordings | {n_abnormal} |
| Total | {n_normal + n_abnormal} |

---

## Key Findings

### Drift Separation (Normalized Euclidean)

| Group | Mean ± Std |
|---|---|
| Normal | {norm_drift_normal.mean():.3f} ± {norm_drift_normal.std():.3f} |
| Abnormal | {norm_drift_abnormal.mean():.3f} ± {norm_drift_abnormal.std():.3f} |

### Health Score Separation

| Group | Mean ± Std |
|---|---|
| Normal | {hs_normal.mean():.1f} ± {hs_normal.std():.1f} |
| Abnormal | {hs_abnormal.mean():.1f} ± {hs_abnormal.std():.1f} |

### DSP Feature Group Separation (Abnormal - Normal mean absolute z-score)

| Feature Group | Delta Mean |z| |
|---|---|
""" + "\n".join(f"| {k} | {v:+.4f} |" for k, v in sorted(sep.items(), key=lambda x: -x[1])) + f"""

**Most discriminative DSP group:** {most_sep_group}

### Top 5 Embedding Dimensions (by cumulative |z| across abnormal recordings)

Dimensions: {top5_dims}

These dimensions of the 256-dim learned fingerprint show the largest
systematic deviation in abnormal recordings relative to the healthy profile.
They represent the directions in the contrastively-learned embedding space
that are most sensitive to the acoustic changes associated with machine faults.

---

## Output Files

| File | Description |
|---|---|
| `explainability_results.csv` | Per-recording drift, health score, DSP group z-scores, top embedding dims |
| `explainability_summary.md` | This document |
| `fig1_dsp_group_drift_by_label.png` | DSP group mean absolute z-score: normal vs abnormal |
| `fig2_top_embedding_dims_abnormal.png` | Top 20 embedding dimensions contributing to abnormal drift |
| `fig3_health_score_vs_norm_euclidean.png` | Health score vs normalized Euclidean drift scatter |

---

## Methodological Notes

- All values are computed from real cached fusion vectors and saved Phase 9 profiles.
- No feature importance is invented or approximated.
- The DSP group z-scores are computed from the raw 153-dim DSP vectors in the
  fusion cache, not from the projected 256-dim embeddings.
- The embedding dimension analysis uses the actual ProjectionHead forward pass
  on cached fusion vectors — identical to the inference pipeline.
- The frozen method (Phase 13) is not modified in any way.
"""

md_path.write_text(md, encoding="utf-8")
print(f"Saved: {md_path}")

# ── Verify all outputs ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Phase 15 — Final Summary")
print("=" * 60)
expected = [
    OUT_DIR / "explainability_results.csv",
    OUT_DIR / "explainability_summary.md",
    OUT_DIR / "fig1_dsp_group_drift_by_label.png",
    OUT_DIR / "fig2_top_embedding_dims_abnormal.png",
    OUT_DIR / "fig3_health_score_vs_norm_euclidean.png",
]
all_ok = True
for p in expected:
    exists = p.exists()
    size   = p.stat().st_size if exists else 0
    status = "OK" if exists else "MISSING"
    print(f"  [{status}]  {p.name}  ({size:,} bytes)")
    if not exists:
        all_ok = False

print(f"\n  Recordings analysed : {len(results_df)}")
print(f"  Normal              : {n_normal}")
print(f"  Abnormal            : {n_abnormal}")
print(f"  Normal drift mean   : {norm_drift_normal.mean():.3f}")
print(f"  Abnormal drift mean : {norm_drift_abnormal.mean():.3f}")
print("  Most sep. DSP group : {}  (delta={:+.4f})".format(most_sep_group, sep[most_sep_group]))
print("  Top-5 emb dims      : {}".format(top5_dims))
print("\n  {}".format('ALL OUTPUTS VERIFIED' if all_ok else 'WARNING: SOME OUTPUTS MISSING'))
