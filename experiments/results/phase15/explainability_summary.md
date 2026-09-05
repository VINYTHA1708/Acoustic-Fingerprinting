# Phase 15 — Explainability of the Final Frozen Method

**Status:** Analysis only — no model retraining, no backend changes.
**Method:** Phase 13 frozen configuration (Phase 9 checkpoint, seed=42).
**Scope:** pump machine type, IDs ['id_00', 'id_02', 'id_04', 'id_06'],
15 normal + 15 abnormal recordings per machine ID.

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
The top-20 dimensions with the largest |z_i| are identified as the
embedding dimensions contributing most to the anomaly score.

### 3. Drift Decomposition
The primary anomaly score `normalized_euclidean = ‖z‖₂` and the derived
`health_score` are reported per recording, enabling direct traceability from
acoustic input to classification decision.

---

## Dataset Summary

| Partition | Count |
|---|---|
| Normal recordings | 60 |
| Abnormal recordings | 60 |
| Total | 120 |

---

## Key Findings

### Drift Separation (Normalized Euclidean)

| Group | Mean ± Std |
|---|---|
| Normal | 14.699 ± 11.079 |
| Abnormal | 38.266 ± 33.527 |

### Health Score Separation

| Group | Mean ± Std |
|---|---|
| Normal | 84.2 ± 28.7 |
| Abnormal | 49.6 ± 39.8 |

### DSP Feature Group Separation (Abnormal - Normal mean absolute z-score)

| Feature Group | Delta Mean |z| |
|---|---|
| MFCC delta | +0.2971 |
| Harmonic | +0.2836 |
| MFCC | +0.2322 |
| Temporal | +0.2315 |
| MFCC delta2 | +0.1494 |
| Spectral | -0.1210 |

**Most discriminative DSP group:** MFCC delta

### Top 5 Embedding Dimensions (by cumulative |z| across abnormal recordings)

Dimensions: [123, 189, 226, 202, 63]

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
