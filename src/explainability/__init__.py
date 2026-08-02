"""Explainability module — rule-based explanations for anomaly detection results.

Consumes outputs from InferencePipeline, LearnedDriftResult, and
LearnedHealthResult to produce human-readable ExplanationResult objects.

Public API:
    ExplanationResult    — dataclass holding the full explanation
    ExplainabilityEngine — rule-based engine that generates explanations
"""

from .explanation import ExplanationResult
from .explainer import ExplainabilityEngine

__all__ = [
    "ExplanationResult",
    "ExplainabilityEngine",
]
