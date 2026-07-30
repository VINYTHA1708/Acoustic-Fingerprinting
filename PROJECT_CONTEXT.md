# Project Context

## Project Title

Acoustic Fingerprinting of Industrial Machines for Predictive Failure Detection Without Labeled Data

---

# Project Objective

The goal of this project is to develop an explainable acoustic fingerprinting system capable of monitoring industrial machines using only healthy recordings during training.

Instead of classifying known fault types, the system measures how much a new recording has drifted from the machine's healthy acoustic reference profile.

The primary research question is:

> "How much has this specific recording drifted from this machine's healthy reference profile?"

---

# Current Development Stage

Current Version:

**Version 1 (DSP Baseline)**

The project follows a staged implementation strategy.

Version 1

- Audio preprocessing
- DSP feature extraction
- Simple acoustic fingerprint
- Healthy Fingerprint Profile
- Fingerprint Drift
- Health Index
- Streamlit dashboard

Version 2

- BEATs encoder
- Fusion Fingerprint (DSP + BEATs)

Version 3

- Contrastive Learning
- Final evaluation
- Explainability refinement

Development must follow this order.

---

# Final Architecture

Machine Audio

↓

Audio Preprocessing

↓

Log-Mel Spectrogram

↓

Feature Extraction

• DSP Features
• BEATs Embedding (V2)

↓

Fusion Fingerprint

↓

Healthy Fingerprint Profile

↓

Fingerprint Drift Analysis

↓

Health Index + Confidence Score

↓

Explainability

↓

Streamlit Dashboard

---

# Dataset

Dataset:

MIMII Dataset

Machine Types

- Fan
- Pump
- Valve
- Slider

Training

Healthy recordings only.

Testing

Healthy and abnormal recordings.

Important

MIMII is NOT a longitudinal degradation dataset.

Fingerprint Drift is defined as:

Distance between

Healthy Fingerprint Profile

and

Current Recording Fingerprint

for the same machine.

---

# Technology Stack

Language

- Python

Machine Learning

- PyTorch

Audio Processing

- librosa
- torchaudio

DSP

- MFCC
- Spectral Centroid
- Spectral Rolloff
- RMS Energy
- Harmonic Features

Similarity Search

- FAISS

Dashboard

- Streamlit

Visualization

- matplotlib
- UMAP
- t-SNE

---

# Coding Principles

1. Keep modules independent.

2. Every folder has a single responsibility.

3. No duplicated logic.

4. Functions should be reusable.

5. Prefer readability over clever code.

6. Every module should be testable independently.

7. Add docstrings for public functions.

8. Use type hints whenever practical.

---

# Important Design Decisions

The architecture is frozen.

Do NOT redesign unless implementation requires it.

Identity Fingerprint and Health Fingerprint have been removed.

Use only

Reference Fingerprint

↓

Current Fingerprint

↓

Fingerprint Drift

DSP descriptors are computed, not learned.

Frequency explanations must always come from DSP features.

BEATs is used only for representation learning.

The dashboard must explain every prediction using measurable DSP quantities.

---

# Folder Responsibilities

dataset/
Dataset loading.

preprocessing/
Audio cleaning and normalization.

feature_extraction/
DSP extraction and BEATs encoder.

fingerprint/
Fusion fingerprint generation.

fingerprint_profile/
Healthy Fingerprint Profile creation.

drift_analysis/
Reference vs Current comparison.

health_index/
Health percentage and confidence score.

explainability/
Component attribution and spectrogram difference.

dashboard/
Streamlit application.

evaluation/
Metrics and comparison with baselines.

---

# Current Goal

Implement Version 1 completely before starting Version 2.

Do not implement BEATs until the DSP baseline is fully functional.

Do not implement Contrastive Learning until Version 2 is complete.

---

# AI Assistant Instructions

When generating code:

- Follow the folder structure.
- Keep functions modular.
- Prefer object-oriented design only where beneficial.
- Avoid unnecessary complexity.
- Follow the SDD exactly.
- If a requested implementation conflicts with the SDD, highlight the conflict instead of changing the architecture.
