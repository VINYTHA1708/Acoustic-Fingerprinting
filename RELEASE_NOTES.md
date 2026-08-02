# Release Notes — Version 1.0.0

**Release date:** 2026
**License:** MIT
**Python:** 3.11+

---

## Overview

Version 1.0.0 is the initial production release of the Acoustic Fingerprinting system for industrial machine health monitoring. The system learns exclusively from healthy recordings and detects anomalies by measuring how much a new recording has drifted from a machine's healthy acoustic reference profile. No labelled fault examples are required during training.

The core pipeline combines classical DSP features with deep audio representations from the frozen BEATs encoder, fused into a 921-dimensional vector and projected by a trainable contrastive head into a 256-dimensional L2-normalised fingerprint. Drift from the healthy profile is quantified using Euclidean, Manhattan, and cosine metrics, normalised by the healthy distribution's statistics, and converted into a bounded health score with a qualitative state label.

---

## Major Features

- One-class learning — trains only on normal recordings; no fault labels required
- Dual-stream feature extraction — DSP (153-dim) fused with frozen BEATs embeddings (768-dim) into a 921-dim Fusion Vector
- Contrastive learning — NT-Xent loss with automatic positive/negative pair construction from MIMII directory structure
- Learned fingerprint — 256-dim L2-normalised embedding produced by a trainable ProjectionHead
- Drift analysis — raw and z-score normalised drift metrics (Euclidean, Manhattan, cosine) against a per-machine healthy profile
- Health index — bounded score [0, 100] with four qualitative states: EXCELLENT, GOOD, WARNING, CRITICAL
- Fusion cache — compressed NPZ disk cache keyed by recording path; eliminates redundant preprocessing on repeated runs
- End-to-end pipeline — single `InferencePipeline.analyze()` call returning a structured `PipelineResult`
- Per-stage benchmark — wall-clock timing for every pipeline stage with cache-hit detection
- Full serialisation — JSON and NPZ round-trip for profiles, drift results, health results, and pipeline results

---

## Implemented Modules

| Package | Module | Responsibility |
|---|---|---|
| `preprocessing` | `AudioLoader`, `AudioResampler`, `AudioNormalizer`, `SpectrogramGenerator` | Mono conversion, 16 kHz resampling, amplitude normalisation, log-Mel spectrogram |
| `feature_extraction` | `MFCCExtractor`, `SpectralExtractor`, `TemporalExtractor`, `HarmonicExtractor`, `FeatureVectorBuilder` | 153-dim DSP feature vector |
| `beats` | `BEATsEncoder` | Frozen pretrained BEATs encoder producing 768-dim mean-pooled embeddings |
| `fusion` | `FusionBuilder`, `FusedFeatureVector`, `FusionCache` | DSP ⊕ BEATs concatenation and NPZ disk cache |
| `contrastive_learning` | `ContrastiveDataset`, `ProjectionHead`, `NTXentLoss`, `ContrastiveTrainer`, `ContrastiveInference`, `ContrastiveSerializer` | Full contrastive training and inference pipeline |
| `learned_profile` | `LearnedProfileBuilder`, `LearnedFingerprintProfile`, serializer | Per-machine healthy profile (embeddings, mean, std) with JSON/NPZ persistence |
| `learned_drift` | `LearnedDriftAnalyzer`, `LearnedDriftResult`, `LearnedDriftMetrics`, serializer | Raw and normalised drift computation |
| `learned_health_index` | `LearnedHealthAnalyzer`, `LearnedHealthCalculator`, `LearnedHealthResult`, serializer | Health score and state derivation |
| `pipeline` | `InferencePipeline`, `PipelineResult` | End-to-end orchestration |
| `benchmark` | `PipelineBenchmark`, `BenchmarkResult` | Per-stage wall-clock timing |
| `dataset` | `DatasetLoader`, `AudioMetadata`, scanner | MIMII directory scanning and metadata |

---

## Testing

The pytest suite contains 68 passing tests across 8 test files. All tests are run automatically on every push and pull request via GitHub Actions (`ubuntu-latest`, Python 3.11).

| Test file | Coverage |
|---|---|
| `test_beats.py` | BEATsEncoder — loading, embedding shape, NaN/Inf checks, invalid checkpoint handling |
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

`PipelineBenchmark` measures wall-clock time for each stage independently using `time.perf_counter()`.

| Stage | What is timed |
|---|---|
| Preprocessing | Audio load → resample → normalise |
| DSP Extraction | Feature extraction and vector construction |
| BEATs | Frozen encoder forward pass |
| Fusion | DSP ⊕ BEATs concatenation or disk cache load |
| Projection | ProjectionHead forward pass |
| Drift | Normalised drift metric computation |
| Health | Health score calculation |
| Total | Full end-to-end wall-clock time |

When a fused vector is already cached on disk, preprocessing, DSP, and BEATs times are reported as 0 ms and fusion time reflects only the disk-load duration. The `cache_hit` field records which path was taken.

---

## Documentation

| File | Contents |
|---|---|
| `README.md` | Project overview, installation, dataset setup, example commands, test and benchmark instructions |
| `docs/system_architecture.md` | Mermaid flowchart of the full inference pipeline with dimension summary and health state bands |
| `docs/training_pipeline.md` | Mermaid flowchart of the contrastive training pipeline |
| `docs/repository_structure.md` | Annotated directory tree |
| `docs/module_dependencies.md` | Inter-module dependency diagram |
| `docs/workflow.md` | Mermaid sequence diagram of one complete inference request |
| `docs/API_REFERENCE.md` | Full public API reference (split across three parts) |
| `docs/sdd/` | System Design Documents v3 and v4 |
| `PROJECT_CONTEXT.md` | Design decisions and architectural rationale |
| `RELEASE_NOTES.md` | This file |

---

## Known Limitations

- Single machine type per training run — the ProjectionHead is trained on one machine type at a time; a separate checkpoint is required per machine type
- BEATs checkpoint not bundled — `BEATs_iter3_plus_AS2M.pt` must be downloaded separately from the Microsoft repository due to its size (~300 MB)
- MIMII dataset only — the pipeline and directory scanner are designed around the MIMII dataset structure; other datasets require a custom scanner
- No real-time streaming — the pipeline processes complete audio files; streaming or partial-buffer inference is not supported in this release
- CPU inference only tested — GPU acceleration is available through PyTorch but has not been benchmarked or validated end-to-end
- Health thresholds are fixed — the EXCELLENT / GOOD / WARNING / CRITICAL band boundaries are hardcoded and not calibrated per machine type or operating condition

---

## Future Work

- Multi-machine-type joint training — a single ProjectionHead trained across all machine types simultaneously
- Streaming inference — sliding-window buffer support for real-time anomaly detection
- Automatic threshold calibration — data-driven band boundaries derived from the healthy profile's empirical distribution
- GPU benchmark — validated end-to-end timing on CUDA hardware
- Expanded dataset support — adapters for DCASE 2020/2021/2022 challenge datasets
- Explainability — SHAP or attention-based attribution to identify which frequency bands drive high drift scores
- Dashboard — interactive visualisation of health scores, drift trends, and fingerprint similarity over time
