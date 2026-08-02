# Acoustic Fingerprinting of Industrial Machines

An explainable acoustic fingerprinting system for industrial machine health monitoring, combining classical Digital Signal Processing (DSP) features with deep audio representations from BEATs and contrastive learning.

---

## Overview

Industrial machines produce unique acoustic signatures during normal operation. As components wear, loosen, or become damaged, these signatures gradually change. Traditional fault detection systems require labeled fault examples during training, which are expensive to collect and cover only known failure modes.

This project takes a different approach. It learns only from healthy recordings and detects deviations by measuring how much a new recording has drifted from the machine's healthy acoustic reference profile.

The core research question is:

> **How much has this specific recording drifted from this machine's healthy reference profile?**

### Why BEATs + DSP Fusion

DSP features (MFCC, spectral centroid, spectral rolloff, RMS energy, harmonic salience) are computed directly from the audio signal. Every dimension has a named, human-interpretable meaning, which makes them the backbone of explainability. However, hand-designed features can miss subtle timbral and spectral structure.

BEATs (Bidirectional Encoder representation from Audio Transformers) is a self-supervised audio encoder pretrained on AudioSet. Its frozen 768-dimensional embeddings capture rich acoustic representations that complement the DSP features without requiring any labeled training data.

The two are concatenated into a 921-dimensional Fusion Vector, combining the interpretability of DSP with the representational power of deep audio embeddings.

### Contrastive Learning

A small trainable ProjectionHead maps the 921-dimensional Fusion Vector into a 256-dimensional L2-normalised learned fingerprint. It is trained with NT-Xent (Normalized Temperature-scaled Cross Entropy) loss using a single objective:

- **Positive pairs**: recordings from the same machine
- **Negative pairs**: recordings from different machines

This encourages the learned fingerprint space to cluster by machine identity, making drift from a healthy profile a reliable anomaly signal. Only the ProjectionHead is trained; BEATs and all DSP extractors remain frozen.

---

## System Architecture

```
Machine Audio (.wav)
        │
        ▼
Audio Preprocessing
(mono, 16 kHz, normalize, trim, fixed-length)
        │
        ▼
Log-Mel Spectrogram
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
DSP Feature Extraction              BEATs Encoder (frozen)
MFCC (20 coeffs, mean+std)          768-dim embedding
Spectral Centroid                         │
Spectral Rolloff                          │
RMS Energy                                │
Harmonic Salience                         │
(153-dim vector)                          │
        │                                 │
        └──────────── Fusion ─────────────┘
                          │
                          ▼
                  Fusion Vector (921-dim)
                  DSP (153) ⊕ BEATs (768)
                          │
                          ▼
              ProjectionHead (trainable)
              Linear → BN → ReLU → Linear → L2-norm
                          │
                          ▼
              Learned Fingerprint (256-dim, L2-normalised)
                          │
                ┌─────────┴──────────┐
                ▼                    ▼
        Stored as              Computed per
        Learned Profile        new recording
        (healthy mean/std      (inference time)
         + all embeddings)
                │                    │
                └────────┬───────────┘
                         ▼
               Drift Analysis
               Raw Metrics (Euclidean, Manhattan, Cosine)
               Statistical Normalization: z = (x − μ) / σ
               Normalized Metrics (official anomaly scores)
                         │
                         ▼
                   Health Index
               score = 100 × (1 − ‖z‖ / 2·μ_healthy)
               Bounded [0, 100]
                         │
                         ▼
              Machine Health Report
              (dimensions, drift, health score, state)
```

### Health State Bands

| Health Score | State     | Meaning                                          |
|-------------|-----------|--------------------------------------------------|
| 90 – 100    | EXCELLENT | Within normal healthy variation                  |
| 75 – 89     | GOOD      | Slightly outside typical healthy variation       |
| 50 – 74     | WARNING   | Statistically significant deviation              |
| 0 – 49      | CRITICAL  | Extreme outlier relative to healthy distribution |

---

## Repository Structure

```
Acoustic-Fingerprinting/
│
├── src/
│   ├── dataset/              # MIMII dataset loader and audio metadata extraction
│   ├── preprocessing/        # Audio cleaning: mono, resample, normalize, trim, clip
│   ├── feature_extraction/   # DSP extraction: MFCC, centroid, rolloff, RMS, harmonic
│   ├── beats/                # BEATs encoder wrapper (frozen, pretrained)
│   ├── fusion/               # Fusion vector construction and disk cache (FusionCache)
│   ├── contrastive_learning/ # ProjectionHead, NT-Xent loss, trainer, inference
│   ├── learned_profile/      # LearnedFingerprintProfile builder and serializer
│   ├── learned_drift/        # Raw and normalized drift metrics
│   ├── learned_health_index/ # Health score calculator and analyzer
│   └── pipeline/             # MachineHealthPipeline end-to-end orchestration
│
├── examples/                 # Standalone runnable scripts (see Examples section)
├── models/
│   ├── beats/                # BEATs checkpoint (BEATs_iter3_plus_AS2M.pt)
│   └── contrastive/          # Trained ProjectionHead checkpoints
├── data/
│   ├── raw/MIMII/            # Raw MIMII dataset
│   └── fusion_cache/         # Pre-computed FusionCache NPZ files
├── outputs/                  # Saved profiles, drift results, health reports
├── docs/sdd/                 # System Design Documents (v1–v4)
├── configs/                  # YAML experiment configurations
├── notebooks/                # Jupyter notebooks for exploration
├── tests/                    # Unit tests
├── requirements.txt
├── PROJECT_CONTEXT.md
└── README.md
```

### Key Source Modules

| Module | Responsibility |
|---|---|
| `dataset/` | Scans MIMII directory tree, extracts `AudioMetadata` per recording |
| `preprocessing/` | Converts raw audio to mono 16 kHz normalized fixed-length waveform |
| `feature_extraction/` | Computes 153-dim DSP feature vector from waveform |
| `beats/` | Wraps frozen BEATs model, produces 768-dim embedding |
| `fusion/` | Concatenates DSP + BEATs into 921-dim vector; caches to disk |
| `contrastive_learning/` | Trains and runs the 256-dim ProjectionHead |
| `learned_profile/` | Builds per-machine healthy profile from all normal embeddings |
| `learned_drift/` | Computes raw and z-score normalized drift metrics |
| `learned_health_index/` | Converts normalized drift into bounded health score |
| `pipeline/` | Single `MachineHealthPipeline.analyze()` call returns `MachineHealthReport` |

---

## Installation

Python 3.10 or later is required.

Clone the repository:

```bash
git clone https://github.com/VINYTHA1708/Acoustic-Fingerprinting.git
cd Acoustic-Fingerprinting
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Downloading the BEATs Checkpoint

The BEATs encoder requires a pretrained checkpoint that is not included in this repository.

Download `BEATs_iter3_plus_AS2M.pt` from the official Microsoft BEATs repository:

```
https://github.com/microsoft/unilm/tree/master/beats
```

Place the downloaded file at:

```
models/beats/BEATs_iter3_plus_AS2M.pt
```

The project resolves this path automatically. All modules that use BEATs default to this location.

---

## Dataset

This project uses the **MIMII (Malfunctioning Industrial Machine Investigation and Inspection Dataset)**.

Download MIMII from:

```
https://zenodo.org/record/3384388
```

Place the extracted dataset under `data/raw/MIMII/`. The expected directory structure is:

```
data/raw/MIMII/
├── fan/
│   ├── id_00/
│   │   ├── normal/
│   │   │   ├── 00000000.wav
│   │   │   └── ...
│   │   └── abnormal/
│   │       ├── 00000000.wav
│   │       └── ...
│   ├── id_02/
│   ├── id_04/
│   └── id_06/
├── pump/
│   ├── id_00/
│   └── ...
├── valve/
│   └── ...
└── slider/
    └── ...
```

Each machine type contains multiple machine IDs. Each machine ID contains `normal/` and `abnormal/` subdirectories of `.wav` recordings.

Training uses only `normal/` recordings. `abnormal/` recordings are reserved exclusively for evaluation.

---

## Training

Training the ProjectionHead requires the FusionCache to be populated first. The cache is built automatically on first use, but pre-populating it avoids recomputation during training.

Run contrastive training:

```bash
python examples/contrastive_training_example.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --max-recordings 200 \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --temperature 0.1
```

The best checkpoint is saved automatically to:

```
models/contrastive/best_projection_head.pt
```

Training arguments:

| Argument | Default | Description |
|---|---|---|
| `--root` | required | Dataset root directory |
| `--machine-type` | `pump` | Machine type to train on |
| `--machine-id` | all | Restrict to a single machine ID |
| `--max-recordings` | `200` | Maximum recordings per machine ID |
| `--epochs` | `2` | Number of training epochs |
| `--batch-size` | `32` | Batch size |
| `--learning-rate` | `1e-3` | Adam learning rate |
| `--temperature` | `0.1` | NT-Xent temperature |

---

## Examples

All example scripts are in `examples/` and are run from the project root.

### Dataset

```bash
python examples/dataset_example.py --root data/raw/MIMII
```

Scans the dataset and prints a summary of machine types, machine IDs, and recording counts.

### Preprocessing

```bash
python examples/preprocessing_example.py --root data/raw/MIMII
```

Runs the preprocessing pipeline on one recording and prints waveform statistics.

### Feature Extraction

```bash
python examples/feature_extraction_example.py --root data/raw/MIMII
```

Extracts the 153-dim DSP feature vector from one recording and prints each named feature.

### BEATs Embedding

```bash
python examples/beats_example.py --root data/raw/MIMII
# or
python examples/beats_example.py --file path/to/audio.wav
```

Encodes one recording with the frozen BEATs model and prints the 768-dim embedding shape.

### Fusion

```bash
python examples/fusion_example.py --root data/raw/MIMII
```

Builds the 921-dim Fusion Vector (DSP + BEATs) for one recording and prints dimension breakdown.

### Fusion Cache

```bash
python examples/fusion_cache_example.py --root data/raw/MIMII
```

Demonstrates cache miss (first run, computes and saves) vs cache hit (second run, loads from disk). Prints elapsed time for both.

### Contrastive Training

```bash
python examples/contrastive_training_example.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --epochs 10
```

Trains the ProjectionHead and prints per-epoch training and validation loss. Saves the best checkpoint.

### Contrastive Inference

```bash
python examples/contrastive_inference_example.py \
    --root data/raw/MIMII \
    --checkpoint models/contrastive/best_projection_head.pt
```

Runs the trained ProjectionHead on one recording and prints the 256-dim learned fingerprint.

### Learned Profile

```bash
python examples/learned_profile_example.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --machine-id id_00 \
    --checkpoint models/contrastive/best_projection_head.pt
```

Builds the healthy learned fingerprint profile for one machine and saves it to `outputs/learned_profiles/`.

### Learned Drift

```bash
python examples/learned_drift_example.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --machine-id id_00 \
    --checkpoint models/contrastive/best_projection_head.pt
```

Analyzes one normal and one abnormal recording. Prints raw and normalized drift metrics for both and checks that abnormal drift is larger.

### Learned Health Index

```bash
python examples/learned_health_example.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --machine-id id_00 \
    --checkpoint models/contrastive/best_projection_head.pt \
    --max-recordings 100
```

Evaluates up to 50 normal and 50 abnormal recordings. Reports mean health score and mean health percentage for each group, and checks that the normal average is higher.

### Full Pipeline

```bash
python examples/pipeline_example.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --machine-id id_00 \
    --checkpoint models/contrastive/best_projection_head.pt
```

Evaluates up to 50 normal and 50 abnormal recordings through `MachineHealthPipeline.analyze()`. Reports mean health scores and a pass/fail summary.

---

## Benchmark

```bash
python examples/benchmark_pipeline.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --machine-id id_00 \
    --checkpoint models/contrastive/best_projection_head.pt \
    --max-recordings 50
```

Measures per-stage inference time over up to `--max-recordings` normal recordings.

Each recording is timed across four stages:

| Stage | What is measured |
|---|---|
| Cache retrieval | `FusionCache.load_or_create()` — disk load of pre-computed fusion vector |
| Drift analysis | `LearnedDriftAnalyzer.analyze()` — ProjectionHead inference + drift metrics |
| Health analysis | `LearnedHealthAnalyzer.analyze()` — health score computation |
| Total pipeline | `MachineHealthPipeline.analyze()` — full end-to-end call |

Output includes average, minimum, maximum, standard deviation, and recordings per second for the total pipeline time. All times are reported in milliseconds.

---

## Final Evaluation

```bash
python examples/final_evaluation.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --checkpoint models/contrastive/best_projection_head.pt \
    --max-recordings 100
```

Runs a full evaluation across every machine ID for the specified machine type.

For each machine ID:

1. Builds a healthy learned profile from up to `--max-recordings` normal recordings.
2. Evaluates up to 50 normal and 50 abnormal recordings through `MachineHealthPipeline`.
3. Computes average raw Euclidean drift, normalized Euclidean drift, and health score for both groups.
4. Computes the **Separation Ratio** = mean abnormal normalized drift / mean normal normalized drift.
5. Measures average inference time per recording.

A machine ID **passes** if both conditions hold:

- Average abnormal normalized drift > average normal normalized drift
- Average normal health score > average abnormal health score

The per-machine table shows:

| Column | Description |
|---|---|
| Normal Drift | Mean normalized Euclidean distance for normal recordings |
| Abnormal Drift | Mean normalized Euclidean distance for abnormal recordings |
| Normal Health | Mean health score for normal recordings |
| Abnormal Health | Mean health score for abnormal recordings |
| Separation Ratio | Abnormal drift / Normal drift (> 1.0 is correct ordering) |
| Inf (ms) | Average inference time per recording in milliseconds |
| Result | PASS or FAIL |

Overall results summarize PASS/FAIL counts, average separation ratio, average health scores, and average inference time across all machine IDs.

---

## Results

The system was validated on the MIMII Pump dataset (SDD v4 §12.4).

**Key finding from validation experiments:**

Raw DSP feature distances did not consistently separate healthy and abnormal recordings due to heterogeneous feature scales. Statistical normalization using the healthy profile's mean and standard deviation significantly improved separation:

| Metric | Normal | Abnormal |
|---|---|---|
| Raw Euclidean (id_00) | 557.18 | 444.81 |
| Normalized Euclidean (id_00) | 13.75 | 14.23 |

Normalized drift metrics are the official anomaly representation used by the health index.

**Expected behavior on a well-trained model:**

- Normal recordings consistently receive higher health scores than abnormal recordings when averaged over 50 recordings per group.
- Separation ratio > 1.0 for most machine IDs.
- Health states for normal recordings cluster in GOOD to EXCELLENT; abnormal recordings cluster in WARNING to CRITICAL.

Performance varies across machine IDs due to differences in operating conditions, which motivates the BEATs + contrastive learning approach over DSP-only baselines.

---

## Future Work

- **Streamlit Dashboard** — interactive per-machine health monitoring with spectrogram difference visualization and acoustic signature comparison.
- **Explainability** — component-wise DSP attribution ranking the top drift contributor per recording with a labeled spectrogram difference overlay.
- **Evaluation against baselines** — AUC, precision, recall, and F1 comparison against Autoencoder, Deep SVDD, and Isolation Forest.
- **Fingerprint Stability metric** — cosine similarity between healthy embeddings of the same machine across different recording conditions.
- **UMAP / t-SNE visualization** — fingerprint projection across machine types and fault conditions.
- **Longitudinal drift tracking** — fingerprint evolution view plotting health score over successive recordings; designed to extend directly to real degradation datasets without architectural changes.
- **Confidence score** — per-prediction confidence based on profile density near the current fingerprint, sample size, and DSP/BEATs agreement.

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.1 |
| Audio Processing | librosa 0.11, torchaudio 2.1 |
| DSP Features | librosa (MFCC, spectral centroid/rolloff, RMS, HPSS) |
| Deep Audio Encoder | BEATs (Microsoft, pretrained frozen) |
| Similarity Search | FAISS |
| Visualization | matplotlib, UMAP, t-SNE |
| Dashboard | Streamlit |
| Evaluation | scikit-learn |

---

## License

This project is developed for academic research purposes.
