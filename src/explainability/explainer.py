"""ExplainabilityEngine — rule-based explainer for anomaly detection results.

Maps health score bands to human-readable summaries, possible causes, and
operator recommendations.  Consumes existing result dataclasses without
duplicating any inference logic.

Health band rules (SDD v4 §8 state bands):
    >= 90  EXCELLENT — machine operating normally
    75–89  GOOD      — minor deviation detected
    50–74  WARNING   — moderate deviation from healthy profile
     < 50  CRITICAL  — significant deviation detected
"""

from __future__ import annotations

import logging

from ..learned_drift.learned_drift_result import LearnedDriftResult
from ..learned_health_index.learned_health_result import LearnedHealthResult
from .explanation import ExplanationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule tables — keyed by (summary, possible_causes, recommendation)
# ---------------------------------------------------------------------------

_RULES: list[tuple[float, float, str, list[str], str]] = [
    # (score_min, score_max, summary, possible_causes, recommendation)
    (
        90.0, 100.0,
        "Machine operating normally.",
        [],
        "No action required.",
    ),
    (
        75.0, 90.0,
        "Minor deviation detected.",
        [],
        "Continue monitoring.",
    ),
    (
        50.0, 75.0,
        "Moderate deviation from healthy profile.",
        ["bearing wear", "misalignment", "load variation"],
        "Schedule inspection.",
    ),
    (
        0.0, 50.0,
        "Significant deviation detected.",
        ["bearing failure", "shaft damage", "mechanical looseness"],
        "Immediate maintenance recommended.",
    ),
]


class ExplainabilityEngine:
    """Rule-based engine that generates human-readable anomaly explanations.

    Consumes a :class:`~learned_drift.learned_drift_result.LearnedDriftResult`
    and a :class:`~learned_health_index.learned_health_result.LearnedHealthResult`
    and applies a fixed set of health-band rules to produce an
    :class:`~explainability.explanation.ExplanationResult`.

    No inference logic is duplicated — all inputs are pre-computed results.
    """

    def explain(
        self,
        drift_result: LearnedDriftResult,
        health_result: LearnedHealthResult,
    ) -> ExplanationResult:
        """Generate a rule-based explanation for one recording.

        Selects the matching health band rule from the score in
        ``health_result.health_score`` and combines it with drift metrics
        from ``drift_result`` to produce a fully populated
        :class:`ExplanationResult`.

        Args:
            drift_result: :class:`~learned_drift.learned_drift_result.LearnedDriftResult`
                          for the recording being explained.
            health_result: :class:`~learned_health_index.learned_health_result.LearnedHealthResult`
                           for the same recording.

        Returns:
            :class:`ExplanationResult` with summary, possible causes, and
            recommendation populated according to the health band rules.
        """
        score = health_result.health_score
        summary, causes, recommendation = self._match_rule(score)

        result = ExplanationResult(
            machine_type=health_result.machine_type,
            machine_id=health_result.machine_id,
            filename=health_result.filename,
            health_score=score,
            health_state=health_result.health_state,
            raw_euclidean=drift_result.euclidean_distance,
            normalized_euclidean=drift_result.norm_euclidean_distance,
            summary=summary,
            possible_causes=causes,
            recommendation=recommendation,
        )

        logger.info(
            "ExplainabilityEngine — %s/%s '%s'  score=%.2f  state=%s  summary='%s'",
            result.machine_type, result.machine_id, result.filename,
            result.health_score, result.health_state, result.summary,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_rule(score: float) -> tuple[str, list[str], str]:
        """Return (summary, possible_causes, recommendation) for *score*.

        Iterates the rule table in descending priority order (highest band
        first).  The last rule acts as a catch-all for scores below 50.

        Args:
            score: Health score in [0, 100].

        Returns:
            Tuple of (summary, possible_causes, recommendation).
        """
        for score_min, score_max, summary, causes, recommendation in _RULES:
            if score_min <= score <= score_max:
                return summary, list(causes), recommendation
        # score < 0 or > 100 — fall back to the CRITICAL rule
        _, _, summary, causes, recommendation = _RULES[-1]
        return summary, list(causes), recommendation
