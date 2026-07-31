# System Design Document (SDD)

## Acoustic Fingerprinting of Industrial Machines for Predictive Failure Detection Without Labeled Data

## Version History

| Version | Description |
| ------- | ------------ |
| v1      | Initial architecture |
| v2      | Added BEATs, Fusion Fingerprint |
| v3      | Simplified architecture, Reference vs. Current Fingerprint |
| v4      | DSP implementation validation, Statistical Feature Normalization, Raw vs. Normalized Drift Metrics |

---

**Version:** 4.0
**Status:** DSP Baseline Implemented and Validated
**Owner:** [Your Name]

**Major changes from v3**

This revision incorporates the findings from the completed DSP implementation and validation experiments.

Key improvements:

- Introduces Statistical Feature Normalization into the Drift Analysis module.
- Separates Raw Drift Metrics from Normalized Drift Metrics.
- Uses Normalized Drift Metrics as the official input to the Health Index.
- Documents validation experiments performed on the MIMII Pump dataset.
- Updates the architecture based on experimental evidence rather than design assumptions.

---

## 1. Purpose and Scope

This document is the blueprint for implementing the project. It fixes the design decisions — representation, architecture, data flow, folder structure, pipelines, and evaluation strategy — _before_ any code is written.

Everything below traces back to the core research question:

> Not "Is this sound anomalous?" but **"How much has this specific recording drifted from this machine's healthy reference profile?"**

(Phrasing tightened slightly from earlier drafts — see §1.3 on dataset scope.)

### 1.1 Non-goals

- No supervised classification. The system never trains on labeled fault types.
- No reliance on abnormal recordings during training — abnormal audio is test-time-only.
- No black-box scoring or black-box explanations. Every health score and every explanation must be traceable to a computable, named DSP quantity.
- No unsolved research problem embedded as a load-bearing component (see Version History — this is why the Identity/Health split was removed in v3).

### 1.2 What changed in v4

This revision incorporates the findings from the completed DSP implementation and validation experiments. Unlike v3, which was a design-only revision, v4 is grounded in an implemented and tested pipeline.

| #   | Change                                                                                                                             | Type      | Section |
| --- | ----------------------------------------------------------------------------------------------------------------------------------- | --------- | ------- |
| 1   | Statistical Feature Normalization — normalize each feature against the Healthy Fingerprint Profile's mean/std before drift scoring | Mandatory | §7      |
| 2   | Raw vs. Normalized Drift Metrics — raw metrics retained for diagnostics; normalized metrics become the official drift metrics       | Mandatory | §7      |
| 3   | Validation Experiments — documents the DSP-only experiments run on the MIMII Pump dataset, with measured results                    | New       | §12.4   |
| 4   | Updated Health Index — now computed from normalized drift metrics rather than raw distances                                        | Mandatory | §8.1    |
| 5   | Architecture revision — pipeline diagram updated to reflect the implemented and validated DSP-only flow                            | Update    | §3, §1.4|

### 1.3 Scope caveat: what "drift" means given MIMII

MIMII provides machine-ID-tagged normal/anomalous recordings under varying operating conditions, but it does **not** contain the same physical unit recorded repeatedly as it degrades over weeks or months. So "fingerprint drift over time," as a literal longitudinal claim, cannot be validated on this dataset as-is.

The project therefore adopts this precise, defensible definition for all reported results:

> **Fingerprint Drift** = the distance between a machine's Healthy Fingerprint Profile (built from that machine's healthy recordings) and the fingerprint of any new recording from that same machine, under the recording's given test condition.

This is fully valid and testable on MIMII (healthy vs. anomalous recordings per machine ID). It does **not** claim to demonstrate true week-over-week predictive maintenance. The architecture is nonetheless designed so that if longitudinal recordings become available later (the project's own collected data, or a future dataset), the same pipeline extends directly to real time-series drift — no architectural change required, only new data. The Fingerprint Evolution view (§11.1) is explicitly labeled as a "designed-for" capability that MIMII does not yet let us validate end-to-end.

### 1.4 Implementation Validation and Design Evolution

Version 4 reflects the first implementation cycle of the project.

The DSP-only pipeline was fully implemented and experimentally validated using the MIMII dataset.

The implemented pipeline consists of:

```
Machine Audio
    │
    ▼
Audio Preprocessing
    │
    ▼
DSP Feature Extraction
    │
    ▼
Acoustic Fingerprint
    │
    ▼
Healthy Fingerprint Profile
    │
    ▼
Fingerprint Drift Analysis
    │
    ▼
Health Index
```

During validation, an important limitation of the original design was discovered.

The initial implementation compared raw DSP feature vectors directly using Euclidean, Manhattan, and Cosine distance.

Experimental evaluation showed that raw feature distances did not consistently separate healthy and abnormal recordings because the extracted DSP features have heterogeneous numerical scales.

A second validation experiment introduced statistical feature normalization using the healthy profile statistics:

```
normalized_feature = (current_feature − profile_mean) / profile_std
```

This significantly improved separation between healthy and abnormal recordings.

Therefore, statistical normalization becomes an official part of the system architecture in Version 4.

---

## 2. Development Strategy: Build in Layers (De-risking Plan)

Unchanged from v2 — still recommended as-is.

### Version 1 — DSP-only baseline (build and validate first)

```
Machine Audio
    │
    ▼
Audio Preprocessing
    │
    ▼
Log-Mel Spectrogram
    │
    ▼
MFCC + basic DSP descriptors
    │
    ▼
Simple Fingerprint (concatenated DSP vector)
    │
    ▼
Distance-based drift + naive health score
```

**Goal:** prove the end-to-end plumbing works on a purely classical, fully-interpretable feature set before adding any learned component, so that later problems can be attributed to BEATs/contrastive learning rather than infrastructure.

**Exit criteria for V1:** healthy-vs-healthy distances are small and stable; healthy-vs-faulty distances are visibly larger on at least the Fan and Pump subsets; dashboard renders a full run end-to-end.

### Version 2 — Add deep features and fusion

BEATs is introduced _alongside_ (not instead of) the DSP features, per §4.1. Strictly additive — the V1 DSP pathway is never deleted, since it remains the explainability backbone and half of the fusion fingerprint.

### Version 3 — Add contrastive learning

Once V2 is validated, add contrastive training over the Fusion Fingerprint (same-machine-different-time → similar; different-machine → dissimilar). This produces a single trained **Fusion Fingerprint** — there is no further split into sub-fingerprints at this stage (see §4.3 for why).

---

## 3. System Overview (Version 4 Architecture)

```
Machine Audio
        │
        ▼
Audio Preprocessing
        │
        ▼
Log-Mel Spectrogram
        │
        ▼
Feature Extraction
        │
        ▼
   ══════════════════
    Implemented (V1)
   ══════════════════
        │
        ▼
   DSP Features
        │
        ▼
Healthy Fingerprint Profile
        │
        ▼
Statistical Feature Normalization
        │
        ▼
Fingerprint Drift Analysis
     ├──────────────┐
     │              │
Raw Metrics   Normalized Metrics
     │              │
     └──────► Health Index ◄──────┘
        │
        ▼
   Dashboard


   ══════════════════════════
    Future Work (Version 2)
   ══════════════════════════

   DSP Features
        │
        ▼
   BEATs Encoder
        │
        ▼
  Fusion Fingerprint
        │
        ▼
Contrastive Fingerprint Learning
        │
        ▼
  (feeds Healthy Fingerprint Profile,
   Drift Analysis, and Explainability
   in place of the DSP-only path above)
```

The DSP-only path above the divider is implemented and experimentally validated (§12.4). The BEATs/fusion/contrastive path below the divider is the planned Version 2/3 extension described in §2 and §4–5; it has not yet been implemented or validated.

### Design principle: computed explainability, not forced explainability

Nothing in the fingerprint is _labeled_ by construction unless it is _computed_ by construction. Any component we want to call "frequency" must actually be a frequency-domain DSP quantity — never a block of neurons we merely hope specializes that way.

### Design principle (new in v3): no unsolved research problem as a load-bearing component

Any module that the _entire pipeline_ depends on must be implementable with well-understood techniques within the project timeline. Disentangling a stable "identity" signal from a drifting "health" signal with no supervision is an open research problem in its own right (related to slow-feature analysis / disentangled representation learning) — solving it is not a prerequisite for this project's core contribution, so it is removed as a dependency (§4.3).

---

## 4. Fingerprint Representation (Simplified)

### 4.1 Fusion design: Deep + DSP (unchanged from v2)

```
Fingerprint = Deep Features (BEATs) ⊕ DSP Features
```

| Block      | Contents                                                                                                                                   | Computed or Learned          | Purpose                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- | ---------------------------------------------------------------------------------------- |
| Deep block | 768-dim frozen BEATs embedding                                                                                                             | Learned (pretrained, frozen) | Captures rich, general timbral/spectral structure a hand-designed feature set would miss |
| DSP block  | MFCC (e.g., 20 coeffs, mean+std pooled), Spectral Centroid, Spectral Rolloff, RMS Energy, Harmonic-to-Noise Ratio / harmonic peak salience | **Computed**, not learned    | Gives every dimension a named, human-interpretable meaning                               |

The two blocks are concatenated (optionally after a small learned linear re-weighting during contrastive training) into a single **Fusion Fingerprint**.

### 4.2 Frequency descriptor is computed, not learned (unchanged from v2)

There is no learned "Frequency Head" carved out of BEATs. All four semantic descriptors used for explainability are computed directly from DSP:

| Descriptor           | Computed from                                                 |
| -------------------- | ------------------------------------------------------------- |
| Frequency Descriptor | Spectral centroid, spectral rolloff, dominant Mel-band energy |
| Temporal Descriptor  | Onset strength envelope / tempogram summary statistics        |
| Harmonic Descriptor  | Harmonic-to-noise ratio, harmonic peak salience (via HPSS)    |
| Energy Descriptor    | RMS energy, dynamic range, energy envelope statistics         |

These live inside the DSP block only. The BEATs deep block stays a single undivided 768-dim vector — it contributes to overall drift magnitude and contrastive discriminability but is never decomposed into semantic sub-parts.

### 4.3 Reference Fingerprint vs. Current Fingerprint (replaces Identity/Health split)

**Previous (v2) design — removed:** splitting the Fusion Fingerprint into a stable "Identity Fingerprint" and a drifting "Health Fingerprint" via a residual, gradient-reversal, or temporal-invariance mechanism. This is an unsupervised disentanglement problem with no guaranteed solution on this timeline, and it added risk without being necessary for the core contribution.

**v3 design:**

```
Fusion Fingerprint (single, undivided vector — no learned split)
        │
        ├── stored as Reference Fingerprint (built from healthy training clips → §6)
        │
        └── computed per new clip as Current Fingerprint (at inference time)

Fingerprint Drift = distance(Reference Fingerprint, Current Fingerprint)
```

There is exactly one learned fingerprint per clip. "Reference" and "Current" are not two different _kinds_ of fingerprint — they are the _same_ fingerprint computed at two different times (training-time healthy clips vs. inference-time new clips) and compared. This is simpler, requires no unproven disentanglement mechanism, and is much easier to explain and defend: _"we compare what the machine sounds like now to what it sounded like when healthy."_

Contrastive training (§2, Version 3) keeps its original, single objective: same-machine-different-time clips should map close together; different-machine clips should map apart. There is no second "invariance" objective and no residual computation.

### 4.4 Component-wise drift (retained from v2, now framed as directly comparing DSP sub-blocks)

Even though there is no learned sub-fingerprint split, the **DSP block** was always composed of named descriptors (§4.2), so component-wise drift analysis is retained by comparing each named descriptor between Reference and Current directly — no disentanglement needed, since these were separable inputs from the start, not something extracted from an entangled learned vector.

### 4.5 Acoustic Signature (new)

In addition to the numeric Fusion Fingerprint, store a small, human-readable summary per machine — the **Acoustic Signature** — derived directly from DSP, independent of any learned component:

```
Acoustic Signature (example: Fan_01)
├── Dominant Frequency:     100 Hz
├── Dominant Harmonic:      300 Hz
├── Average Energy (RMS):   0.81
└── Rotation Frequency:     49.8 Hz   (if extractable, e.g. via periodicity in onset envelope)
```

This is computed once per machine from its healthy reference recordings (and can be recomputed per new recording for comparison). It is not used as a model input — it exists purely so a factory engineer can compare machines or sanity-check the system's behavior without touching the AI pipeline at all. This directly supports non-technical trust in the system.

---

## 5. Model / Feature Architecture

```
Log-Mel Spectrogram
        │
        ├──────────────────────────────┐
        ▼                              ▼
  Classical DSP Extractors        BEATs (frozen)
  (MFCC, centroid, rolloff,        768-dim embedding
   RMS, harmonic salience)              │
        │                              │
        └────────── Fusion ────────────┘
                       │
                       ▼
         Contrastive Encoder (small trainable head
         over the fused vector — single objective,
         no identity/health split)
                       │
                       ▼
              Fusion Fingerprint (fixed schema, versioned)
                       │
              ┌────────┴────────┐
              ▼                 ▼
     stored as Reference   computed as Current
     (from healthy clips)   (at inference time)
```

Only the small contrastive head is trained; both BEATs and the DSP extractors are non-trainable/deterministic.

**Note:** This architecture corresponds to the planned Version 2/Version 3 implementation. The current Version 1 implementation uses only the DSP feature extraction pipeline described in Section 3.

---

## 6. Healthy Fingerprint Profile (renamed from "Healthy Cloud")

### 6.1 Naming rationale

Internally this remains a point-cloud / cluster representation (unchanged from v2's design intent — healthy sound varies naturally across morning/afternoon/night and load conditions, and a single centroid would discard that variance). The name shown on the dashboard and in stakeholder-facing text is **"Healthy Fingerprint Profile,"** since factory engineers recognize "profile" more readily than "cloud." Internal code/module names may keep "cloud" or "profile" interchangeably — the schema does not change.

### 6.2 Storage schema

```
FingerprintProfile   (internal implementation: still cluster/cloud-based)
├── machine_id: str
├── reference_fingerprints: float[N][F]         # every healthy Acoustic/Fusion Fingerprint sample, or a
│                                               # compressed cluster set (e.g., k-means, k=5-10)
├── cluster_centroids: float[k][F]              # optional compressed representation
├── cluster_covariances: float[k][...]           # for Mahalanobis-style distance / density estimate
├── distance_distribution: empirical CDF         # of intra-healthy distances, for §8 calibration
├── acoustic_signature: dict                      # dominant frequency/harmonic/energy/rotation freq (§4.5)
└── created_at / updated_at
```

where `F` is the fingerprint dimensionality. For the current DSP implementation, `F = 153`. Future fusion fingerprints will increase `F` accordingly once BEATs is integrated.

### 6.3 Construction

For the current implementation, compute the Acoustic Fingerprint for every healthy recording and retain the full set (or a compressed multi-cluster summary if large). This set — the Healthy Fingerprint Profile — is the Reference against which every Current Fingerprint is compared.

In future versions this becomes the Fusion Fingerprint after BEATs integration (§2, V2).

### 6.4 Backing store

FAISS index over the stored healthy point set per machine_id (nearest-neighbor search), plus a metadata store (SQLite/JSON) for the distance distribution used in Health Index calibration and for the Acoustic Signature.

---

## 7. Statistical Drift Analysis

The Drift Analysis module performs two independent analyses.

### 7.1 Drift computation flow

```
Current Fingerprint
        │
        ▼
Healthy Profile
(mean, std)
        │
        ▼
normalized_vector
   =
(current_vector - mean_vector)
   /
   std_vector
        │
        ▼
Normalized Euclidean
Normalized Manhattan
Normalized Cosine
```

### 7.2 Stage 1 — Raw Drift Metrics

Compute raw drift metrics.

These preserve the original fingerprint comparison and are useful for debugging, visualization and research.

Raw Metrics:

- Euclidean Distance
- Manhattan Distance
- Cosine Similarity

### 7.3 Stage 2 — Statistical Normalization

Compute statistical normalization.

For every feature:

```
normalized_feature = (current_feature − profile_mean) / profile_std
```

where `profile_mean` and `profile_std` are obtained from the Healthy Fingerprint Profile.

Division by zero is handled safely.

### 7.4 Stage 3 — Normalized Drift Metrics

Compute normalized metrics.

The normalized feature vector is compared against the healthy reference.

Normalized Metrics:

- Normalized Euclidean Distance
- Normalized Manhattan Distance
- Normalized Cosine Similarity

These metrics demonstrated better separation during validation and therefore become the official drift metrics used by downstream modules.

---

## 8. Health Index — Statistical, With Confidence

### 8.1 Statistical calibration (updated in v4)

Health Index is computed from the normalized drift metrics.

The normalized Euclidean distance is the primary anomaly score.

Additional statistical indicators include:

- Mean Absolute Z-score
- Maximum Absolute Z-score
- Normalized Manhattan Distance
- Normalized Cosine Similarity

The raw metrics remain available for diagnostics but are not used directly for health estimation.

Health % is still derived from the empirical distribution stored in the Fingerprint Profile (§6.2), not a hardcoded formula — the underlying percentile-rank calibration is unchanged from v2/v3, only the distance input has changed from raw to normalized:

```
health_pct = 100 * (1 − percentile_rank_outside_healthy_distribution)
```

A clip within the normal healthy-variation range scores near 100%; a clip in the extreme tail scores near 0%.

### 8.2 Status bands

| Health % | Status    | Statistical meaning                                             |
| -------- | --------- | --------------------------------------------------------------- |
| 90–100   | Excellent | Within normal healthy variation                                 |
| 75–89    | Good      | Slightly outside typical healthy variation, not yet significant |
| 50–74    | Warning   | Statistically significant deviation from healthy distribution   |
| 0–49     | Critical  | Extreme outlier relative to healthy distribution                |

### 8.3 Per-machine calibration

Calibrated per machine, since each unit's natural healthy variance differs.

### 8.4 Confidence score (new)

Alongside Health %, report a **Confidence** score reflecting how well-supported that health estimate is, based on:

- **Profile density near the Current Fingerprint** — if the healthy profile has few nearby reference samples (i.e., the current recording sits in a sparsely-sampled region of the healthy distribution, even if not far in raw distance), confidence is lower.
- **Sample size** — machines with few healthy training recordings get systematically lower confidence, surfaced explicitly rather than silently producing an overconfident score.
- **Agreement between DSP-only and BEATs-only sub-scores** — if the two feature families disagree substantially on drift magnitude, confidence is reduced (this is a cheap ensemble-agreement heuristic, not a new learned component).

Displayed as either a percentage or a qualitative band:

```
Health:      82%
Confidence:  96%   (or: Confidence: Low / Medium / High)
```

This is computed entirely from statistics already available in the Fingerprint Profile and DriftVector — no new model is trained to produce it.

---

## 9. Explainability — With Spectrogram Difference and Acoustic Signature

### 9.1 Component attribution (unchanged from v2)

Rank the four DriftVector DSP components by z-score relative to that component's own healthy-profile distance distribution. The top-ranked component drives the natural-language explanation.

### 9.2 Spectrogram difference visualization (unchanged from v2)

```
Healthy Reference Spectrogram (representative profile member, or profile-median)
        −
Current Spectrogram
        =
Difference Spectrogram (highlighted, e.g. red = increased energy, blue = decreased)
```

Cross-referenced against the top-ranked drift component.

### 9.3 Acoustic Signature comparison (new)

Alongside the plot, show the machine's Acoustic Signature (§4.5) for both the healthy reference and the current recording side by side:

```
                Reference     Current
Dominant Freq   100 Hz        118 Hz   ⚠
Dominant Harm.  300 Hz        301 Hz
Avg. Energy     0.81          0.85
Rotation Freq   49.8 Hz       49.7 Hz
```

This gives a fully non-AI, human-checkable cross-reference for the AI-driven explanation above it.

---

## 10. Training Pipeline (Staged per §2)

Version 1 has been completed and validated. Version 2 and Version 3 remain planned extensions and are documented here for completeness.

**V1 (DSP-only):**

```
1. Load healthy clips per machine
2. Preprocess (mono/16kHz/normalize/trim/fixed-length)
3. Compute log-mel spectrogram → MFCC + centroid + rolloff + RMS + harmonic salience
4. Concatenate into Simple Fingerprint
5. Build Healthy Fingerprint Profile per machine; validate drift/health index end-to-end
```

**V2 (add BEATs, fusion):**

```
6. Encode same clips with frozen BEATs → 768-dim embedding
7. Concatenate with DSP block → Fusion Fingerprint (no learning yet — validate fusion plumbing)
```

**V3 (add contrastive learning — single objective, no split):**

```
8. Train small contrastive head over the Fusion Fingerprint:
   - Positive pairs: same machine, different time segments
   - Negative pairs: different machines
9. Checkpoint best model by validation contrastive loss + Fingerprint Stability metric (§12)
```

## 11. Testing / Inference Pipeline

```
1. Load new clip (healthy or faulty, from test split or live upload)
2. Preprocess identically to training
3. Extract DSP block (deterministic) + BEATs embedding (frozen) → Current Fingerprint
4. Retrieve Healthy Fingerprint Profile + distance distribution + Acoustic Signature from storage (by machine_id)
5. Compute:
   • Raw Drift Metrics
   • Statistical Feature Normalization
   • Normalized Drift Metrics
   ↓
   Health Index
6. Map the normalized drift metrics to Health % via statistical confidence-interval scoring; compute Confidence (§8.4)
7. Generate explainability: ranked component attribution + spectrogram difference overlay + Acoustic Signature comparison
8. Return structured result to Dashboard
```

### 11.1 Fingerprint Evolution (new dashboard capability)

For machines with more than one historical recording, plot successive Current Fingerprint drift/health values over time:

```
Recording 1  →  Health: 100
Recording 2  →  Health: 98
Recording 3  →  Health: 94
Recording 4  →  Health: 89
```

**Important scope note (per §1.3):** MIMII does not provide true longitudinal same-unit degradation sequences, so this view cannot be validated against ground-truth gradual degradation using MIMII alone. It is implemented and demonstrated using whatever repeated-recording structure is available (e.g., multiple healthy/anomalous takes under the same machine ID), and explicitly documented as a "designed-for-future-data" capability — the architecture supports it out of the box the moment real longitudinal recordings are supplied.

---

## 12. Evaluation Strategy

### 12.1 Baselines for comparison

- Autoencoder (reconstruction error)
- Deep SVDD
- Isolation Forest

### 12.2 Detection metrics

- AUC, Precision, Recall, F1 (healthy vs. abnormal, thresholded on Health %)
- Inference time per clip, memory footprint

### 12.3 Project-specific metrics

- **Fingerprint Stability:** cosine similarity between Fusion Fingerprints of the same machine's healthy recordings taken at different times/conditions. High, tightly-clustered similarity supports the fingerprint's validity as a machine identity — reportable as a standalone contribution.
- **Fingerprint Visualization:** UMAP or t-SNE projection of fingerprints across machines and machine types; healthy clips from different machine types should form separated clusters, with faulty clips drifting away from their own machine's cluster.
- **Confidence calibration:** does reported Confidence actually correlate with correctness (e.g., are low-confidence predictions more often wrong)? Report as a reliability diagram if time permits.
- **Explainability precision:** does the flagged DSP component/frequency band correspond to the actual known fault type (validation-only ground truth, never used in training).
- Health trend behavior is reported descriptively (§11.1), with the MIMII scope caveat (§1.3) stated explicitly alongside any such figure.

---

### 12.4 Validation Experiments

The DSP-only implementation was experimentally evaluated before introducing any learned representation.

**Experiment 1 — Single Recording Drift**

Result: Raw distances occasionally ranked abnormal recordings closer to the healthy profile than normal recordings.

**Experiment 2 — Distance Distribution**

50 healthy, 50 abnormal recordings.

Result: Certain machine IDs showed overlapping distributions.

**Experiment 3 — Machine-wise Evaluation (Pump dataset, raw Euclidean distance)**

| Machine ID | Raw Euclidean (Normal) | Raw Euclidean (Abnormal) | Result |
| ---------- | ----------------------- | -------------------------- | ------ |
| id_00      | 557.18                  | 444.81                     | FAIL   |
| id_02      | 616.28                  | 727.37                     | PASS   |
| id_04      | 379.88                  | 453.96                     | PASS   |
| id_06      | 729.38                   | 693.11                     | FAIL   |

Result marked PASS where the abnormal mean distance exceeds the normal mean distance (i.e., raw distance correctly separates the classes); FAIL where it does not.

**Experiment 4 — Raw vs Normalized Distance (aggregate, id_00)**

| Metric               | Normal | Abnormal |
| --------------------- | ------ | -------- |
| Raw Euclidean         | 557.18 | 444.81   |
| Normalized Euclidean  | 13.75  | 14.23    |

**Experiment 5 — Cross Machine-ID Evaluation**

Four pump machine IDs (id_00, id_02, id_04, id_06) were evaluated independently.

Result: Normalized drift improved separation overall, although performance varied across machine IDs, highlighting differences in operating conditions and motivating future deep feature integration.

**Conclusion**

Statistical normalization significantly improved healthy/abnormal separation.

Although improvement was observed, separation was not perfect across all machine IDs. This motivates the addition of BEATs embeddings and contrastive fingerprint learning in the next development stage.

Version 4 therefore adopts normalized drift metrics as the official anomaly representation.

The validation results indicate that statistical normalization improves class separation for the implemented DSP feature representation. Future work will evaluate whether learned embeddings (BEATs) further improve robustness across all machine IDs.

---

## 13. Folder Structure

This reflects the actual implemented codebase for the DSP-only version. The BEATs/contrastive/explainability/evaluation modules described conceptually elsewhere in this document (fusion, contrastive learning, spectrogram-diff explainability, baseline comparisons) are not yet present in the repository and will be added as those stages are implemented (§2, V2/V3).

```
Acoustic-Fingerprinting/
├── src/
│   ├── dataset/            # dataset.py, MIMII loaders, split logic
│   ├── preprocessing/      # audio cleaning: mono/resample/normalize/trim/clip
│   ├── feature_extraction/ # DSP feature extraction: MFCC, centroid, rolloff, RMS, harmonic salience
│   ├── fingerprint/        # Acoustic Fingerprint construction (concatenated DSP vector)
│   ├── profile/            # Healthy Fingerprint Profile: mean/std per feature, distance distribution
│   └── drift/              # Raw Drift Metrics, Statistical Feature Normalization, Normalized Drift Metrics, Health Index
├── dashboard/                # Streamlit app
├── models/                   # saved checkpoints (reserved for future contrastive head, V3)
├── utils/                     # shared helpers (audio I/O, config loading)
├── configs/                   # YAML configs per experiment/version (v1/v2/v3/v4)
└── main.py                    # CLI entry point (train / infer / dashboard)
```

---

## 14. Technology Stack (Unchanged)

| Component                   | Technology                                                                                    |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| Language                    | Python                                                                                        |
| Deep Learning               | PyTorch                                                                                       |
| Audio processing            | librosa, torchaudio                                                                           |
| Spectrogram / DSP features  | librosa (MFCC, spectral centroid/rolloff, RMS, HPSS/harmonic salience)                        |
| Encoder                     | BEATs (pretrained, frozen)                                                                    |
| Contrastive learning        | PyTorch                                                                                       |
| Fingerprint profile storage | FAISS                                                                                         |
| Visualization               | UMAP or scikit-learn t-SNE, matplotlib/plotly                                                 |
| Dashboard                   | Streamlit                                                                                     |
| Explainability              | Template-based DSP attribution + spectrogram difference plots + Acoustic Signature comparison |
| Evaluation                  | scikit-learn                                                                                  |

---

## 15. Open Design Questions

1. Cluster count `k` for the compressed healthy profile representation (§6.2) — validate against raw-set nearest-neighbor drift to ensure compression doesn't lose calibration accuracy.
2. Fixed clip duration: confirm against actual MIMII clip lengths.
3. Whether DSP descriptor targets need per-machine-type normalization (Valve vs. Fan have very different natural spectra) — likely yes, to be validated in V1.
4. Choice of distance/density estimator over the healthy profile (k-NN distance vs. GMM likelihood vs. kernel density) — trade-off between calibration accuracy and inference latency.
5. Confidence score weighting (§8.4) — how to combine profile-density, sample-size, and DSP/BEATs-agreement signals into one number vs. showing them as separate sub-indicators; needs a small pilot to see which is more legible to a non-technical dashboard user.
6. Rotation Frequency extraction (§4.5) — confirm whether it's reliably extractable from the available recordings/machine types, or whether it should be marked "not available" for machine types where periodicity isn't clean (e.g., Valve).

---

## 16. Definition of Done (per stage, for tracking)

- [ ] **V1 (DSP-only):** dataset loads correctly; DSP-only Simple Fingerprint built; Healthy Fingerprint Profile constructed; drift + naive health index produce sane healthy-vs-faulty separation; dashboard renders end-to-end.
- [ ] **V2 (fusion):** BEATs embeddings generated and concatenated with DSP block without breaking V1 plumbing; Fusion Fingerprint schema versioned and stored.
- [ ] **V3 (contrastive):** contrastive training converges; same-machine similarity > different-machine similarity, verified quantitatively. No identity/health split is implemented or required.
- [ ] Fingerprint Profile stores/retrieves the healthy point set (not a single centroid) and its distance distribution correctly per machine_id.
- [ ] Fingerprint Drift Analysis returns a full Drift Vector (profile distance + 4 DSP-component distances + deep distance), computed as Reference-vs-Current, not identity-vs-health.
- [ ] Health Index calibrated using normalized drift metrics and validated through experimental evaluation.
- [ ] Confidence score computed and displayed alongside Health %.
- [ ] Acoustic Signature computed per machine and shown Reference-vs-Current in the dashboard.
- [ ] Explainability produces correct top-ranked component + a labeled spectrogram difference overlay.
- [ ] Fingerprint Stability measured and reported.
- [ ] UMAP/t-SNE fingerprint visualization produced across machine types and fault conditions.
- [ ] Fingerprint Evolution page implemented, with the MIMII longitudinal-scope caveat documented alongside it.
- [ ] Dashboard renders all required pages end-to-end on uploaded .wav.
- [ ] Evaluation report comparing against Autoencoder/Deep SVDD/Isolation Forest, including the project-specific metrics above, complete.

---

## Appendix A: Evolution of the Architecture

**Version 1**

```
Raw DSP Fingerprint
        │
        ▼
   Raw Distance
        │
        ▼
      Health
```

**Version 4**

```
Raw DSP Fingerprint
        │
        ▼
Healthy Statistical Profile
        │
        ▼
Feature Normalization
        │
        ▼
Normalized Drift Metrics
        │
        ▼
   Health Index
```

The architectural modification was introduced after experimental validation demonstrated improved separation between healthy and abnormal recordings.
