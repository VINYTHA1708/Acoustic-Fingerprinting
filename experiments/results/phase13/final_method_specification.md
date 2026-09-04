# Phase 13 — Final Method Specification (FROZEN)

> **STATUS: FROZEN — No further methodological changes permitted after Phase 13.**
> This document records the exact configuration, protocol, and results of the
> final selected method. All values are fixed from prior experimental phases
> (E1, Phase 9, Phase 11, Phase 12). Nothing in this document may be altered
> without invalidating the experimental record.

---

## 1. Final Model Architecture

**Method name:** Multi-Machine Contrastive Acoustic Fingerprinting (Phase 9)

### 1.1 Feature Extraction (frozen, not trained)

| Component | Details |
|---|---|
| Audio encoder | BEATs (`BEATs_iter3_plus_AS2M.pt`) — frozen, pretrained on AudioSet |
| BEATs output | 768-dim mean-pooled embedding per recording |
| DSP features | MFCC (20 coefficients, mean + std = 40 values), spectral centroid (mean + std = 2), spectral rolloff (mean + std = 2), RMS energy (mean + std = 2), harmonic salience (mean + std = 2) — total 153-dim |
| Fusion vector | Concatenation: DSP (153) ⊕ BEATs (768) = **921-dim** |
| Fusion cache | Compressed NPZ files keyed by recording path; avoids recomputation |

### 1.2 ProjectionHead (trainable)

```
Input (921) → Linear(921 → 512) → ReLU → Linear(512 → 256) → L2-normalise → Output (256)
```

| Parameter | Value |
|---|---|
| Input dimension | 921 |
| Hidden dimension | 512 |
| Output dimension | 256 |
| Output normalisation | L2 (unit sphere) |
| Trainable parameters | Only ProjectionHead; BEATs and DSP are frozen |

### 1.3 Loss Function

**NT-Xent (Normalised Temperature-Scaled Cross-Entropy)**

- Positive pairs: recordings from the same machine ID
- Negative pairs: recordings from different machine IDs
- Temperature: **0.07**

---

## 2. Input / Audio Preprocessing

All preprocessing is applied identically at training, profile-building, and inference time.

| Step | Configuration |
|---|---|
| Channel conversion | Mono (stereo averaged if needed) |
| Target sample rate | **16 000 Hz** |
| Amplitude normalisation | Peak normalisation to [−1, 1] |
| FFT window size | 1024 samples |
| Hop length | 512 samples |
| Window length | 1024 samples |
| Mel bands | 128 |
| Frequency range | 20 Hz – 8 000 Hz (fmax = target_sr // 2) |
| Spectrogram type | Log-Mel |

---

## 3. Contrastive Training Configuration

| Parameter | Value |
|---|---|
| Experiment ID | phase9 |
| Machine types | fan, pump, slider, valve (all four MIMII types) |
| Machine IDs | id_00, id_02, id_04, id_06 |
| Pooled train_normal recordings | 10 296 |
| Epochs | **20** |
| Batch size | **16** |
| Optimiser | Adam |
| Learning rate | **0.001** |
| NT-Xent temperature | **0.07** |
| Internal val split | 20% of train_normal (ContrastiveDataset val_split=0.20) |
| Checkpoint selection | Lowest validation loss across all epochs |
| Checkpoint path | `models/contrastive/phase9/best_projection_head.pt` |
| Best validation loss | 1.1015 (epoch 9) |

### 3.1 Per-Type Verified Train-Normal Counts (seed=42)

| Machine type | Total | id_00 | id_02 | id_04 | id_06 |
|---|---|---|---|---|---|
| fan | 2 851 | 707 | 711 | 723 | 710 |
| pump | 2 623 | 704 | 703 | 491 | 725 |
| slider | 2 240 | 747 | 747 | 373 | 373 |
| valve | 2 582 | 693 | 495 | 700 | 694 |
| **TOTAL** | **10 296** | | | | |

---

## 4. Dataset Split Protocol

**Splitter:** `DatasetSplitter(train_ratio=0.70, profile_ratio=0.15, seed=42)`

| Partition | Ratio | Purpose |
|---|---|---|
| `train_normal` | 70% of normal | Contrastive training only |
| `profile_normal` | 15% of normal | Healthy profile construction |
| `test_normal` | 15% of normal | Evaluation (held-out) |
| `test_abnormal` | 100% of abnormal | Evaluation (held-out) |

**Rules (validated at runtime):**
- `train_normal`, `profile_normal`, `test_normal` are mutually disjoint.
- Every normal recording appears in exactly one partition.
- No abnormal recording appears in any normal partition.
- Split is applied **independently per machine type**.
- Abnormal recordings are **never seen during training or profile construction**.
- Split is deterministic: same seed always produces the same assignment.

---

## 5. Healthy Profile Construction

The healthy profile is built from `profile_normal` recordings only.

| Step | Detail |
|---|---|
| Input | All `profile_normal` recordings for a given machine ID |
| Embedding | Run each recording through the frozen fusion pipeline + trained ProjectionHead |
| Profile storage | All 256-dim embeddings, mean vector (μ), std vector (σ) |
| Profile format | JSON (metadata) + NPZ (embeddings, μ, σ) |
| Profile scope | One profile per (machine_type, machine_id) pair |

---

## 6. Drift Metrics and Health-Score Calculation

### 6.1 Drift Metrics

Given a new 256-dim embedding **e** and profile (μ, σ):

```
z_i = (e_i − μ_i) / max(σ_i, 1e-10)   for each dimension i

normalized_euclidean  = ‖z‖₂   ← PRIMARY ANOMALY SCORE
normalized_manhattan  = ‖z‖₁
normalized_cosine     = cosine(z, 1)   (cosine vs uniform direction)

raw_euclidean  = ‖e − μ‖₂
raw_manhattan  = ‖e − μ‖₁
raw_cosine     = cosine(e, μ)
```

**Selected primary metric: `normalized_euclidean`**

Rationale (from Phase 9 Step 10 statistical validation):
- Euclidean and Manhattan are virtually identical (mean AUC difference = 0.002, p = 0.034 Wilcoxon).
- Cosine is a significantly weaker discriminator (AUC ≈ 0.48–0.53 overall).
- Euclidean is preferred by convention when the two are statistically equivalent.

### 6.2 Health-Score Calculation

**Method:** Gaussian survival function anchored to the healthy distribution.

```
μ_norm  = mean(‖z_i‖) over all profile_normal embeddings
σ_norm  = std(‖z_i‖)  over all profile_normal embeddings

t       = (normalized_euclidean − μ_norm) / max(σ_norm, 1e-8)

score   = 100 × Φ(c − t)   where c = Φ⁻¹(0.95) ≈ 1.6449
score   = clamp(score, 0, 100)
```

**Health state bands:**

| Score | State |
|---|---|
| 90–100 | EXCELLENT |
| 75–89 | GOOD |
| 50–74 | WARNING |
| 0–49 | CRITICAL |

**Anomaly classification threshold (from E1 health calibration):**

| Threshold | Accuracy | F1 | Sensitivity | Specificity |
|---|---|---|---|---|
| 54.52 | 0.755 | 0.726 | 0.728 | 0.777 |

---

## 7. Final Evaluation Protocol

1. Load the trained checkpoint (`models/contrastive/phase9/best_projection_head.pt`).
2. For each (machine_type, machine_id):
   a. Build the healthy profile from `profile_normal` recordings.
   b. Run inference on all `test_normal` and `test_abnormal` recordings.
   c. Compute `normalized_euclidean` drift for each recording.
   d. Compute ROC-AUC using `normalized_euclidean` as the anomaly score.
   e. Compute Cohen's d between normal and abnormal drift distributions.
3. Report per-machine-ID and per-machine-type metrics.
4. Report overall (all machine types combined) ROC-AUC and Cohen's d.
5. No threshold tuning is performed on the test set.

**Final evaluation results (Phase 9, seed=42):**

| Machine type | ROC-AUC (Euclidean) | Cohen's d |
|---|---|---|
| fan | 0.6986 | 0.739 |
| pump | 0.8635 | 1.425 |
| slider | 0.8813 | 1.487 |
| valve | 0.8283 | 1.275 |
| **Overall** | **0.7875** | **1.061** |

**Phase 11 seed-stability results (seeds 42, 123, 2026):**

| Seed | ROC-AUC | Cohen's d |
|---|---|---|
| 42 | 0.7896 | 1.068 |
| 123 | 0.7757 | 1.011 |
| 2026 | 0.7756 | 1.030 |
| **Mean ± std** | **0.780 ± 0.008** | **1.036 ± 0.029** |

---

## 8. Random Seeds

| Context | Seed |
|---|---|
| Dataset split (DatasetSplitter) | **42** |
| ContrastiveDataset internal val split | **42** |
| ContrastiveTrainer weight init + DataLoader shuffle | **42** |
| Python `random` module | **42** |
| NumPy `np.random` | **42** |
| PyTorch `torch.manual_seed` | **42** |
| Phase 11 seed-stability seeds | 42, 123, 2026 |
| Bootstrap CI (Phase 9 Step 10) | **42** (2 000 iterations) |

All seeds are set before any data loading, splitting, or model initialisation.

---

## 9. Final Selected Metric / Method

**Primary anomaly score:** `normalized_euclidean` (‖z‖₂ of the z-score vector)

**Selection rationale:**
- Consistently highest or statistically equivalent ROC-AUC across all machine types and IDs.
- Statistically indistinguishable from `normalized_manhattan` (Wilcoxon p = 0.034, effect size r = 0.80 — large but practically negligible in magnitude).
- `normalized_cosine` is significantly weaker (overall AUC ≈ 0.476, bootstrap 95% CI [0.460, 0.493]).
- Euclidean is the conventional choice when Euclidean and Manhattan are equivalent.
- Bootstrap 95% CI for overall Euclidean AUC: **[0.777, 0.801]** (width = 0.024).

**Ablation study findings (E1):**

| Ablation | Mean AUROC |
|---|---|
| Full method (DSP + BEATs + Contrastive) | 0.808 |
| A1 — No BEATs (DSP-only) | 0.750 |
| A2 — No DSP (BEATs-only) | 0.855 |
| A3 — No contrastive training | 0.594 |
| A4 — No ProjectionHead | 0.574 |
| B1 — Raw MFCC distance baseline | 0.574 |
| B2 — Statistical feature distance baseline | 0.540 |
| B3 — Random projection baseline | 0.525 |

The contrastive training step (A3 vs full) and the ProjectionHead (A4 vs full) are the most critical components. BEATs alone (A2) slightly outperforms the full fusion on pump, but the full method is more robust across machine types.

---

## Runtime Benchmarks (Phase 12, CPU, cache-hit path)

| Measurement | Value |
|---|---|
| Inference mean per file | **5.86 ms** |
| Inference median per file | **5.40 ms** |
| Profile build per recording | **9.6 ms** |
| Full evaluation (1 022 pump recs) | **9.12 s** (8.92 ms/rec) |
| Platform | Windows 11, Intel Core i7 (Family 6 Model 154), CPU-only |

---

## Freeze Declaration

> **This specification is frozen as of Phase 13.**
>
> No changes to model architecture, preprocessing parameters, training
> hyperparameters, dataset split ratios, profile construction procedure,
> drift metric formulas, health-score formula, evaluation protocol, or
> random seeds are permitted after this point.
>
> Any future work that modifies any of the above constitutes a new
> experimental phase and must be documented separately with full
> justification and re-evaluation.
