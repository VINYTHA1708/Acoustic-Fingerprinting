"""Learned Drift Analysis module — Version 3.

Computes drift between a new recording's learned embedding (256-dim) and a
LearnedFingerprintProfile built from healthy recordings.

SDD v4 §7:
    Raw metrics compare the current embedding directly against the profile mean.
    Normalized metrics operate on the z-score vector:
        normalized_vector = (current - mean) / std
    Normalized metrics are the official input to the Health Index module.

Public API:
    LearnedDriftResult    — dataclass holding all drift metrics
    LearnedDriftMetrics   — computes raw and normalized metrics from an embedding
    LearnedDriftAnalyzer  — runs the full pipeline and returns a LearnedDriftResult
    LearnedDriftSerializer — save/load to JSON and NPZ
"""

from .learned_drift_result import LearnedDriftResult
from .metrics import LearnedDriftMetrics
from .analyzer import LearnedDriftAnalyzer
from .serializer import LearnedDriftSerializer

__all__ = [
    "LearnedDriftResult",
    "LearnedDriftMetrics",
    "LearnedDriftAnalyzer",
    "LearnedDriftSerializer",
]
