"""Learned Health Index module — Version 3.

Computes a bounded health score from normalized drift metrics produced by the
learned fingerprint pipeline.

SDD v4 §8:
    The normalized Euclidean distance is the primary anomaly score.
    Health score is bounded in [0, 100].
    Healthy recordings produce higher scores; larger drift reduces the score.

Public API:
    LearnedHealthResult      — dataclass holding health score, percentage, and state
    LearnedHealthCalculator  — converts normalized drift metrics into a health score
    LearnedHealthAnalyzer    — runs the full pipeline and returns a LearnedHealthResult
    LearnedHealthSerializer  — save/load to JSON and NPZ
"""

from .learned_health_result import LearnedHealthResult
from .calculator import LearnedHealthCalculator
from .analyzer import LearnedHealthAnalyzer
from .serializer import LearnedHealthSerializer

__all__ = [
    "LearnedHealthResult",
    "LearnedHealthCalculator",
    "LearnedHealthAnalyzer",
    "LearnedHealthSerializer",
]
