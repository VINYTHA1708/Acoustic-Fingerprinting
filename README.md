# Acoustic Fingerprinting of Industrial Machines

![Python](https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c?logo=pytorch&logoColor=white)
![Tests](https://github.com/VINYTHA1708/Acoustic-Fingerprinting/actions/workflows/tests.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)

An acoustic fingerprinting system for industrial machine health monitoring that combines classical Digital Signal Processing (DSP) features with deep audio representations from BEATs and contrastive learning.

Industrial machines produce unique acoustic signatures during normal operation. As components wear or become damaged, these signatures change. Traditional fault detection systems require labeled fault examples during training, which are expensive to collect and cover only known failure modes.

This project takes a different approach. It learns only from healthy recordings and detects deviations by measuring how much a new recording has drifted from the machine's healthy acoustic reference profile.

> **Core question:** How much has this specific recording drifted from this machine's healthy reference profile?

DSP features (MFCC, spectral centroid, spectral rolloff, RMS energy, harmonic salience) provide a 153-dimensional interpretable representation. BEATs (Bidirectional Encoder representation from Audio Transformers), a self-supervised audio encoder pretrained on AudioSet, contributes a frozen 768-dimensional embedding that captures rich timbral and spectral structure. The two are concatenated into a 921-dimensional Fusion Vector.

A small trainable ProjectionHead maps the Fusion Vector into a 256-dimensional L2-normalised learned fingerprint using NT-Xent contrastive loss. Positive pairs are recordings from the same machine; negative pairs are recordings from different machines. Only the ProjectionHead is trained — BEATs and all DSP extractors remain frozen.

---

## Features

- **Audio preprocessing** — mono conversion, resampling to 16 kHz, amplitude normalisation, log-Mel spectrogram generation
- **DSP feature extraction** — MFCC (20 coefficients, mean + std), spectral centroid, spectral rolloff, RMS energy, harmonic salience (153-dim vector)
- **BEATs encoder** — frozen pretrained BEATs model producing a 768-dim mean-pooled embedding per recording
- **Feature fusion** — concatenation of DSP and BEATs vectors into a 921-dim Fusion Vector
- **Fusion cache** — compressed NPZ disk cache keyed by recording path; avoids recomputing the full pipeline on repeated runs
- **Contrastive dataset** — automatic positive and negative pair construction from MIMII directory structure
- **NT-Xent loss** — normalised temperature-scaled cross-entropy loss for contrastive training
- **ProjectionHead** — trainable Linear → ReLU → Linear → L2-norm head mapping 921-dim to 256-dim
- **Contrastive training** — full training loop with train/validation split, per-epoch loss reporting, and best-checkpoint saving
- **Contrastive inference** — loads a trained checkpoint and generates 256-dim learned fingerprints
- **Learned fingerprint profile** — per-machine healthy profile storing all embeddings, mean vector, and std vector
- **Learned drift analysis** — raw and z-score normalised drift metrics (Euclidean, Manhattan, cosine) between a recording's embedding and the healthy profile
- **Learned health index** — bounded health score [0, 100] and qualitative state (EXCELLENT / GOOD / WARNING / CRITICAL) derived from normalised drift
- **End-to-end inference pipeline** — single `InferencePipeline.analyze()` call returning a structured `PipelineResult`
- **Pipeline benchmark** — per-stage wall-clock timing (preprocessing, DSP, BEATs, fusion, projection, drift, health) with cache-hit detection
- **Serialisation** — JSON and NPZ round-trip for profiles, drift results, health results, and pipeline results
- **Unit tests** — pytest suite covering all major modules

---

## Repository Structure

```
Acoustic-Fingerprinting/
│
├── src/
│   ├── preprocessing/          # Audio loading, resampling, normalisation, spectrogram
│   ├── feature_extraction/     # DSP extractors: MFCC, spectral, temporal, harmonic
│   ├── beats/                  # BEATsEncoder wrapper (frozen, pretrained)
│   ├── fusion/                 # FusionBuilder, FusedFeatureVector, FusionCache
│   ├── contrastive_learning/   # Dataset, ProjectionHead, NTXentLoss, Trainer, Inference
│   ├── learned_profile/        # LearnedProfileBuilder, LearnedFingerprintProfile
│   ├── learned_drift/          # LearnedDriftAnalyzer, LearnedDriftResult, metrics
│   ├── learned_health_index/   # LearnedHealthAnalyzer, LearnedHealthCalculator
│   ├── pipeline/               # InferencePipeline, MachineHealthPipeline, PipelineResult
│   ├── benchmark/              # PipelineBenchmark, BenchmarkResult
│   └── dataset/                # DatasetLoader, AudioMetadata, scanner
│
├── examples/                   # Standalone runnable scripts (one per module)
├── tests/                      # pytest unit tests
├── models/
│   ├── beats/                  # BEATs checkpoint (not tracked by Git)
│   └── contrastive/            # Trained ProjectionHead checkpoints
├── data/
│   ├── raw/MIMII/              # Raw MIMII dataset
│   └── fusion_cache/           # Pre-computed FusionCache NPZ files
├── outputs/
│   └── learned_profiles/       # Saved LearnedFingerprintProfile files
├── third_party/
│   └── beats/                  # Official Microsoft BEATs source (unmodified)
├── docs/
│   └── sdd/                    # System Design Documents (v3, v4)
├── requirements.txt
├── PROJECT_CONTEXT.md
└── README.md
```

---

## System Architecture

```
Machine Audio (.wav)
        │
        ▼
Audio Preprocessing
(mono, 16 kHz, normalise, log-Mel spectrogram)
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
DSP Feature Extraction              BEATs Encoder (frozen)
MFCC (20 coeffs, mean+std)          768-dim embedding
Spectral Centroid
Spectral Rolloff
RMS Energy
Harmonic Salience
(153-dim vector)
        │                                 │
        └──────────── Fusion ─────────────┘
                          │
                          ▼
                  Fusion Vector (921-dim)
                  DSP (153) ⊕ BEATs (768)
                          │
                          ▼
              ProjectionHead (trainable)
              Linear(921→512) → ReLU → Linear(512→256) → L2-norm
                          │
                          ▼
              Learned Fingerprint (256-dim, L2-normalised)
                          │
                ┌─────────┴──────────┐
                ▼                    ▼
        Healthy Profile         New Recording
        (mean + std of          (inference time)
         normal embeddings)
                │                    │
                └────────┬───────────┘
                         ▼
               Drift Analysis
               Raw metrics: Euclidean, Manhattan, Cosine
               Normalised metrics: z = (x − μ) / σ
                         │
                         ▼
                   Health Index
               score = 100 × (1 − ‖z‖ / 2·μ_healthy)
               Bounded [0, 100]
                         │
                         ▼
                  PipelineResult
          (dimensions, drift metrics, health score, state)
```

### Health State Bands

| Score   | State     | Meaning                                          |
|---------|-----------|--------------------------------------------------|
| 90–100  | EXCELLENT | Within normal healthy variation                  |
| 75–89   | GOOD      | Slightly outside typical healthy variation       |
| 50–74   | WARNING   | Statistically significant deviation              |
| 0–49    | CRITICAL  | Extreme outlier relative to healthy distribution |

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

## Dataset

This project uses the **MIMII (Malfunctioning Industrial Machine Investigation and Inspection) Dataset**.

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
│   │       └── ...
│   ├── id_02/
│   └── ...
├── pump/
├── valve/
└── slider/
```

Training uses only `normal/` recordings. `abnormal/` recordings are reserved exclusively for evaluation.

---

## BEATs Checkpoint

The BEATs encoder requires a pretrained checkpoint that is intentionally excluded from this repository due to its size.

Download `BEATs_iter3_plus_AS2M.pt` from the official Microsoft BEATs repository:

```
https://github.com/microsoft/unilm/tree/master/beats
```

Place the downloaded file at:

```
models/beats/BEATs_iter3_plus_AS2M.pt
```

All modules that use BEATs resolve this path automatically.

---

## Running Examples

All example scripts are run from the project root. Each script accepts `--help` for a full argument listing.

### Dataset

```bash
python examples/dataset_example.py --root data/raw/MIMII
```

### Preprocessing

```bash
python examples/preprocessing_example.py --root data/raw/MIMII
```

### Feature Extraction

```bash
python examples/feature_extraction_example.py --root data/raw/MIMII
```

### BEATs Embedding

```bash
python examples/beats_example.py --root data/raw/MIMII
# or pass a specific file
python examples/beats_example.py --file path/to/audio.wav
```

### Fusion

```bash
python examples/fusion_example.py --root data/raw/MIMII
```

### Fusion Cache

```bash
python examples/fusion_cache_example.py --root data/raw/MIMII
```

Demonstrates cache miss (first run, computes and saves) vs cache hit (second run, loads from disk) with elapsed time for both.

### Contrastive Dataset

```bash
python examples/contrastive_dataset_example.py --root data/raw/MIMII
```

### NT-Xent Loss

```bash
python examples/ntxent_loss_example.py --root data/raw/MIMII
```

### Projection Head

```bash
python examples/projection_head_example.py --root data/raw/MIMII
```

### Contrastive Training

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

The best checkpoint is saved to `models/contrastive/best_projection_head.pt`.

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

### Contrastive Inference

```bash
python examples/contrastive_inference_example.py \
    --root data/raw/MIMII \
    --checkpoint models/contrastive/best_projection_head.pt
```

### Contrastive Serializer

```bash
python examples/contrastive_serializer_example.py \
    --checkpoint models/contrastive/best_projection_head.pt
```

### Learned Profile

```bash
python examples/learned_profile_example.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --machine-id id_00 \
    --checkpoint models/contrastive/best_projection_head.pt
```

Builds the healthy learned fingerprint profile and saves it to `outputs/learned_profiles/`.

### Learned Drift

```bash
python examples/learned_drift_example.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --machine-id id_00 \
    --checkpoint models/contrastive/best_projection_head.pt
```

Analyzes one normal and one abnormal recording. Prints raw and normalised drift metrics for both.

### Learned Health Index

```bash
python examples/learned_health_example.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --machine-id id_00 \
    --checkpoint models/contrastive/best_projection_head.pt \
    --max-recordings 100
```

### End-to-End Pipeline

```bash
python examples/pipeline_example.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --machine-id id_00 \
    --checkpoint models/contrastive/best_projection_head.pt
```

Runs `InferencePipeline.analyze()` on held-out normal and abnormal recordings and prints a full health report.

### Pipeline Benchmark

```bash
python examples/benchmark_pipeline.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --machine-id id_00 \
    --checkpoint models/contrastive/best_projection_head.pt \
    --max-recordings 50
```

### Final Evaluation

```bash
python examples/final_evaluation.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --checkpoint models/contrastive/best_projection_head.pt \
    --max-recordings 100
```

Runs a full evaluation across every machine ID for the specified machine type. Reports normalised drift, health scores, separation ratio, and PASS/FAIL per machine ID.

---

## Unit Tests

Run the full test suite from the project root:

```bash
python -m pytest tests/
```

The suite covers:

| File | Module under test |
|---|---|
| `test_beats.py` | BEATsEncoder — loading, embedding shape, NaN/Inf, invalid checkpoint |
| `test_fusion.py` | FusionBuilder — output dimension, DSP/BEATs ordering, validation errors |
| `test_projection.py` | ProjectionHead — output dimension, L2 normalisation, weight persistence |
| `test_ntxent.py` | NTXentLoss — scalar output, finiteness, batch size and temperature validation |
| `test_profile.py` | LearnedProfileBuilder and serializer — shape, JSON/NPZ round-trip, invalid machine |
| `test_drift.py` | LearnedDriftAnalyzer and serializer — result type, finite metrics, machine mismatch |
| `test_health.py` | LearnedHealthAnalyzer and serializer — score bounds, valid state, JSON/NPZ round-trip |
| `test_pipeline.py` | InferencePipeline, PipelineResult, PipelineBenchmark — dimensions, bounds, serialisation |

Tests use session-scoped fixtures and the existing fusion cache on disk to avoid recomputing the full pipeline during the test run.

---

## Benchmark

`PipelineBenchmark` measures wall-clock time for each stage of the inference pipeline independently using `time.perf_counter()`:

| Stage | What is timed |
|---|---|
| Preprocessing | Audio load → resample → normalise |
| DSP Extraction | Feature extraction + vector construction |
| BEATs | Frozen encoder forward pass |
| Fusion | DSP + BEATs concatenation (or disk cache load) |
| Projection | ProjectionHead forward pass |
| Drift | Normalised drift metric computation |
| Health | Health score calculation |
| **Total** | **Full end-to-end wall-clock time** |

When a fused vector is already cached on disk, preprocessing, DSP, and BEATs times are reported as 0 ms and the fusion time reflects only the disk-load duration. The `cache_hit` field records which path was taken.

Run the benchmark example:

```bash
python examples/benchmark_pipeline.py \
    --root data/raw/MIMII \
    --machine-type pump \
    --machine-id id_00 \
    --checkpoint models/contrastive/best_projection_head.pt
```

---

## Technologies Used

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.1 |
| Audio Processing | librosa 0.11, torchaudio 2.1 |
| DSP Features | librosa (MFCC, spectral centroid/rolloff, RMS, HPSS) |
| Deep Audio Encoder | BEATs (Microsoft, pretrained frozen) |
| Similarity Search | FAISS |
| Numerical Computing | NumPy |
| Testing | pytest |
| Visualisation | matplotlib |

---

## References

- **BEATs** — Sanyuan Chen et al., *BEATs: Audio Pre-Training with Acoustic Tokenizers*, ICML 2023. [arXiv:2212.09058](https://arxiv.org/abs/2212.09058)
- **MIMII Dataset** — Harsh Purohit et al., *MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection*, DCASE 2019. [Zenodo](https://zenodo.org/record/3384388)
- **NT-Xent Loss** — Ting Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)*, ICML 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- **SimCLR** — Ting Chen et al., *Big Self-Supervised Models are Strong Semi-Supervised Learners*, NeurIPS 2020. [arXiv:2006.10029](https://arxiv.org/abs/2006.10029)

---

## License

This project is developed for academic research purposes. No license has been formally assigned. All rights reserved by the author.
