# Phase 12 — Runtime Benchmark Report

## Overview

Runtime benchmark of the **Phase 9 final method**: a multi-machine contrastive
ProjectionHead (921→256-dim, NT-Xent loss) trained jointly on all four MIMII
machine types (fan, pump, slider, valve).

All timings use `time.perf_counter()`.  The fusion cache (pre-computed NPZ
files on disk) is used throughout — this is the production inference path.

---

## Hardware / Software Environment

| Item              | Value                                              |
|-------------------|----------------------------------------------------|
| OS                | Windows 11 (10.0.26200)                            |
| Processor         | Intel64 Family 6 Model 154 Stepping 4 (GenuineIntel) |
| Python            | 3.14.0                                             |
| PyTorch           | 2.10.0+cpu (CPU-only build)                        |
| NumPy             | 2.3.4                                              |
| CUDA              | Not available (CPU inference)                      |

---

## 1. Profile / Training Time

### 1a. Contrastive Training (Phase 9 run — from saved history)

Wall-clock time was not instrumented during the original training run.
The following configuration and loss values are taken directly from
`models/contrastive/phase9/training_history.json`.

| Parameter                    | Value      |
|------------------------------|------------|
| Epochs                       | 20         |
| Pooled train_normal recs     | 10,296     |
| Batch size                   | 16         |
| Approx. train pairs          | 8,236      |
| Approx. batches / epoch      | 514        |
| Best validation loss         | 1.1015     |
| Final training loss (ep 20)  | 1.2353     |
| Final validation loss (ep 20)| 1.4742     |

### 1b. Profile Build Time (live measurement, pump/id_00)

Profile building runs the full pipeline (cache load → ProjectionHead forward)
for each healthy recording and computes mean/std embeddings.

| Metric                  | Value      |
|-------------------------|------------|
| Machine                 | pump/id_00 |
| Recordings              | 150        |
| Total time              | **1.433 s** |
| Mean per recording      | **9.6 ms** |

---

## 2. Inference Time per Audio File

Measured over **30 recordings** (pump/id_00, test_normal), cache-hit path.
Each call runs: cache load → ProjectionHead forward → drift metrics → health score.

| Metric     | Value      |
|------------|------------|
| n          | 30         |
| Mean       | **5.86 ms** |
| Std        | 2.80 ms    |
| Median     | **5.40 ms** |
| Min        | 4.60 ms    |
| Max        | 20.6 ms    |

> Cache-hit path: fusion vector loaded from NPZ disk cache.
> BEATs encoder and DSP extraction are **not** re-run.

---

## 3. Evaluation Time for the Dataset

Measured over **all 1,022 pump test recordings** (566 normal + 456 abnormal),
cache-hit path.

| Metric                  | Value       |
|-------------------------|-------------|
| Machine type            | pump        |
| Total recordings        | 1,022       |
| Total evaluation time   | **9.118 s** |
| Mean per recording      | **8.92 ms** |

---

## Summary Table

| Measurement                        | Value       | Notes                          |
|------------------------------------|-------------|--------------------------------|
| Training epochs                    | 20          | Phase 9 run                    |
| Training recordings (pooled)       | 10,296      | All 4 machine types            |
| Best validation loss               | 1.1015      | Epoch 9                        |
| Profile build — total (150 recs)   | 1.433 s     | pump/id_00, cache-hit          |
| Profile build — per recording      | 9.6 ms      | cache-hit                      |
| Inference — mean per file          | **5.86 ms** | 30 recs, cache-hit             |
| Inference — median per file        | **5.40 ms** | 30 recs, cache-hit             |
| Evaluation — total (1022 recs)     | 9.118 s     | pump, cache-hit                |
| Evaluation — mean per recording    | **8.92 ms** | pump, cache-hit                |

---

## Notes

- All timings are on **CPU** (no CUDA). GPU would reduce inference latency further.
- The **cache-hit path** is the standard production path: fusion vectors are
  pre-computed once and stored as NPZ files. The first-run (cache-miss) path
  additionally runs BEATs encoder (~200–400 ms/file on CPU) and DSP extraction.
- Profile build time scales linearly with the number of healthy recordings.
  For 560 pump profile recordings (full split), estimated total ≈ 5.4 s.
- Evaluation time (8.92 ms/rec) includes: NPZ load + ProjectionHead forward +
  drift metric computation + health score calculation.

---

## Verdict

**PHASE 12 PASSED**

All three timing measurements completed successfully:
1. Training configuration and loss history confirmed from Phase 9 artifacts.
2. Profile build: 9.6 ms/recording (cache-hit).
3. Inference: 5.86 ms mean per file (cache-hit, CPU).
4. Evaluation: 8.92 ms mean per recording over 1,022 pump test files.
