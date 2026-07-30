# Acoustic Fingerprinting of Industrial Machines

An explainable acoustic fingerprinting system for industrial machine health monitoring using classical Digital Signal Processing (DSP) features and deep audio representations.

---

# Overview

Industrial machines naturally produce unique acoustic signatures during operation. As components wear, loosen, or become damaged, these acoustic characteristics gradually change.

This project proposes an explainable framework that models the healthy acoustic fingerprint of each machine and measures the drift of new recordings from that healthy reference profile.

Unlike traditional fault classification systems, this project does **not** require labeled fault types during training.

The system learns only from healthy recordings and detects deviations through acoustic fingerprint drift.

---

# Research Objective

The core research question is:

> How much has this specific recording drifted from this machine's healthy reference profile?

Rather than predicting predefined fault classes, the system estimates machine health by comparing new recordings against a learned healthy fingerprint profile.

---

# Key Features

- Acoustic Fingerprinting
- DSP Feature Extraction
- BEATs Audio Embeddings
- Fusion Fingerprint
- Healthy Fingerprint Profile
- Fingerprint Drift Analysis
- Statistical Health Index
- Confidence Score
- Explainable Predictions
- Spectrogram Difference Visualization
- Acoustic Signature Comparison
- Streamlit Dashboard

---

# Dataset

Dataset used:

**MIMII (Malfunctioning Industrial Machine Investigation and Inspection Dataset)**

Machine types:

- Fan
- Pump
- Valve
- Slider

Training uses only healthy recordings.

Abnormal recordings are reserved for evaluation.

---

# Project Architecture

```text
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
(DSP + BEATs)
      │
      ▼
Fusion Fingerprint
      │
      ▼
Healthy Fingerprint Profile
      │
      ▼
Fingerprint Drift Analysis
      │
      ▼
Health Index + Confidence
      │
      ▼
Explainability
      │
      ▼
Streamlit Dashboard
```

---

# Development Roadmap

## Version 1

- Dataset Loader
- Audio Preprocessing
- DSP Feature Extraction
- Simple Fingerprint
- Healthy Fingerprint Profile
- Fingerprint Drift
- Health Index
- Dashboard

---

## Version 2

- BEATs Integration
- Fusion Fingerprint

---

## Version 3

- Contrastive Learning
- Evaluation
- Explainability Refinement

---

# Technology Stack

| Component         | Technology              |
| ----------------- | ----------------------- |
| Language          | Python                  |
| Deep Learning     | PyTorch                 |
| Audio Processing  | librosa, torchaudio     |
| DSP               | librosa                 |
| Similarity Search | FAISS                   |
| Dashboard         | Streamlit               |
| Visualization     | matplotlib, UMAP, t-SNE |

---

# Folder Structure

```text
Acoustic-Fingerprinting/
│
├── docs/
├── src/
├── data/
├── models/
├── outputs/
├── configs/
├── notebooks/
├── tests/
├── requirements.txt
├── PROJECT_CONTEXT.md
├── README.md
└── main.py
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/VINYTHA1708/Acoustic-Fingerprinting.git
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.\.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Current Status

🚧 Active Development

Current implementation stage:

**Version 1 – DSP Baseline**

---

# License

This project is developed for academic research purposes.
