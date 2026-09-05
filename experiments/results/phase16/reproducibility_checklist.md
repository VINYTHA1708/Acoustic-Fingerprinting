# Phase 16 — Reproducibility Checklist

> **STATUS: FROZEN CONFIGURATION — Phase 13**
> This checklist documents every step required to reproduce the final
> evaluation results exactly. The method is frozen; no retraining is needed
> or permitted to reproduce the reported numbers.

---

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.10 or later |
| PyTorch | 2.1.0 |
| torchaudio | 2.1.0 |
| librosa | 0.11.0 |
| numpy | 1.26.4 |
| scikit-learn | 1.4.0 |
| Platform tested | Windows 11, CPU-only |

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Required Artefacts (not tracked by Git)

| Artefact | Location | Source |
|---|---|---|
| MIMII dataset | `data/raw/MIMII/` | https://zenodo.org/record/3384388 |
| BEATs checkpoint | `models/beats/BEATs_iter3_plus_AS2M.pt` | https://github.com/microsoft/unilm/tree/master/beats |
| Phase 9 ProjectionHead | `models/contrastive/phase9/best_projection_head.pt` | Produced by `experiments/phase9_train.py` |

The Phase 9 checkpoint is the **only trained artefact** required for evaluation.
It was produced with the exact configuration in `experiments/results/phase13/final_method_config.json`.

---

## 3. Dataset Directory Structure

```
data/raw/MIMII/
├── fan/
│   ├── id_00/normal/*.wav
│   ├── id_00/abnormal/*.wav
│   ├── id_02/  id_04/  id_06/  (same structure)
├── pump/   (same)
├── slider/ (same)
└── valve/  (same)
```

---

## 4. Random Seeds (Frozen — Phase 13)

All seeds are set to **42** before any data loading, splitting, or model
initialisation. The seed-setting call is:

```python
import random, numpy as np, torch

def _set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

| Context | Seed |
|---|---|
| `DatasetSplitter` | 42 |
| `ContrastiveDataset` internal val split | 42 |
| `ContrastiveTrainer` weight init + DataLoader | 42 |
| Python `random` | 42 |
| NumPy `np.random` | 42 |
| PyTorch `torch.manual_seed` | 42 |
| Phase 11 seed-stability runs | 42, 123, 2026 |
| Bootstrap CI (Phase 9 Step 10) | 42 (2 000 iterations) |

Seeds are applied in `experiments/phase9_train.py` and `experiments/phase9_evaluate.py`
via `_set_seeds(SEED)` at the top of `main()`.

---

## 5. Dataset Split Protocol (Frozen — Phase 13)

```python
DatasetSplitter(train_ratio=0.70, profile_ratio=0.15, seed=42)
```

| Partition | Ratio | Used for |
|---|---|---|
| `train_normal` | 70% of normal | Contrastive training |
| `profile_normal` | 15% of normal | Healthy profile construction |
| `test_normal` | 15% of normal | Evaluation (held-out) |
| `test_abnormal` | 100% of abnormal | Evaluation (held-out) |

Split is applied **independently per machine type**.
Verified train_normal counts (seed=42):

| Machine type | Total | id_00 | id_02 | id_04 | id_06 |
|---|---|---|---|---|---|
| fan | 2 851 | 707 | 711 | 723 | 710 |
| pump | 2 623 | 704 | 703 | 491 | 725 |
| slider | 2 240 | 747 | 747 | 373 | 373 |
| valve | 2 582 | 693 | 495 | 700 | 694 |
| **TOTAL** | **10 296** | | | | |

---

## 6. Frozen Model Configuration

Full specification: `experiments/results/phase13/final_method_config.json`

| Parameter | Value |
|---|---|
| BEATs checkpoint | `BEATs_iter3_plus_AS2M.pt` (frozen) |
| BEATs output dim | 768 |
| DSP output dim | 153 |
| Fusion dim | 921 |
| ProjectionHead | Linear(921→512) → ReLU → Linear(512→256) → L2-norm |
| Epochs | 20 |
| Batch size | 16 |
| Learning rate | 0.001 (Adam) |
| NT-Xent temperature | 0.07 |
| Best validation loss | 1.1015 (epoch 9) |
| Checkpoint | `models/contrastive/phase9/best_projection_head.pt` |

---

## 7. Reproducing the Final Evaluation (No Retraining)

The checkpoint is already trained. To reproduce evaluation results:

```bash
# Step 1: Build profiles and evaluate all machine types
python experiments/phase9_evaluate.py

# Results written to:
#   experiments/results/phase9/evaluation_results.csv
#   experiments/results/phase9/evaluation_summary.json
#   experiments/results/phase9/evaluation_{fan,pump,slider,valve}.csv
#   experiments/results/phase9/profiles/phase9_*_learned_profile.{json,npz}
```

Expected results (from `experiments/results/phase9/evaluation_summary.json`):

| Machine type | ROC-AUC | Cohen's d |
|---|---|---|
| fan | 0.6986 | 0.739 |
| pump | 0.8635 | 1.425 |
| slider | 0.8813 | 1.487 |
| valve | 0.8283 | 1.275 |
| **Overall** | **0.7875** | **1.061** |

---

## 8. Reproducing Training (Optional — Checkpoint Already Exists)

Only needed if the checkpoint file is missing:

```bash
python experiments/phase9_train.py
```

This will:
1. Set seeds (42)
2. Load all MIMII recordings
3. Split per machine type with `DatasetSplitter(train_ratio=0.70, profile_ratio=0.15, seed=42)`
4. Validate split counts against the frozen expected values
5. Train for 20 epochs, batch size 16, lr=0.001, temperature=0.07
6. Save best checkpoint to `models/contrastive/phase9/best_projection_head.pt`

---

## 9. Path Portability

All paths in experiment scripts are **relative to the repository root**:

```python
DATASET_ROOT    = Path("data/raw/MIMII")
CHECKPOINT_PATH = Path("models/contrastive/phase9/best_projection_head.pt")
RESULTS_DIR     = Path("experiments/results/phase9")
CACHE_ROOT      = Path("data/fusion_cache")
```

Scripts must be run from the repository root:

```bash
cd /path/to/Acoustic-Fingerprinting
python experiments/phase9_evaluate.py
```

---

## 10. Fusion Cache

Pre-computed fusion vectors are stored in `data/fusion_cache/` (not tracked by Git).
On first run, the cache is populated automatically. Subsequent runs load from disk,
reducing per-recording inference time from ~full pipeline to ~5.86 ms.

Cache files are keyed by recording path and stored as compressed NPZ files.

---

## 11. Checklist Summary

- [x] `requirements.txt` lists all direct project dependencies with pinned versions
- [x] Random seed 42 applied consistently in all experiment scripts via `_set_seeds()`
- [x] All paths in experiment scripts are relative to the repository root
- [x] Frozen configuration documented in `experiments/results/phase13/final_method_config.json`
- [x] Frozen specification documented in `experiments/results/phase13/final_method_specification.md`
- [x] Evaluation results archived in `experiments/results/phase9/evaluation_summary.json`
- [x] Phase 11 seed-stability results archived in `experiments/results/phase11/`
- [x] No threshold tuning performed on the test set
- [x] Abnormal recordings never used during training or profile construction
- [x] Split counts verified at runtime against frozen expected values
- [x] BEATs encoder and DSP extractors remain frozen (not trained)
- [x] Only `ProjectionHead` weights are trained
