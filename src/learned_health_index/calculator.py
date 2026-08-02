"""LearnedHealthCalculator — converts normalized drift metrics into a health score.

SDD v4 §8:
    The normalized Euclidean distance is the primary anomaly score.
    Health score is bounded in [0, 100].
    Healthy recordings produce higher scores; larger drift reduces the score.

Normalization strategy:
    The scale used to map drift → score is derived from the profile itself:

        profile_healthy_norm = mean ‖z_i‖  over all healthy embeddings

    where z_i = (embedding_i − mean_vector) / std_vector.

    A recording at the healthy center scores ~100; a recording at
    2× the healthy norm scores ~0.  This is machine-specific and
    requires no hardcoded global constant.

Status bands (SDD v4 §8.2):
    90–100  EXCELLENT
    75–89   GOOD
    50–74   WARNING
    0–49    CRITICAL
"""

from __future__ import annotations

_THRESHOLDS = {
    "EXCELLENT": 90.0,
    "GOOD": 75.0,
    "WARNING": 50.0,
}


class LearnedHealthCalculator:
    """Computes a bounded health score from normalized drift metrics.

    Args:
        thresholds: Dict mapping state names to their lower-bound percentage.
                    Keys must include ``EXCELLENT``, ``GOOD``, and ``WARNING``.
                    Defaults to SDD v4 §8.2 values.
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self._thresholds = thresholds or _THRESHOLDS

    def calculate(
        self,
        normalized_euclidean: float,
        normalized_manhattan: float,
        normalized_cosine: float,
        profile_healthy_norm: float,
    ) -> tuple[float, str, str]:
        """Compute health score, percentage, and state from normalized drift metrics.

        The health score is primarily driven by normalized Euclidean distance,
        scaled by the machine-specific healthy norm derived from the profile.

        A recording whose drift equals the healthy norm scores ~100.
        A recording at 2× the healthy norm scores ~0.
        Values are clamped to [0, 100].

        Args:
            normalized_euclidean: Normalized Euclidean distance (primary input).
            normalized_manhattan: Normalized Manhattan distance.
            normalized_cosine: Normalized cosine similarity.
            profile_healthy_norm: Mean ‖z‖ of healthy embeddings from the profile.
                                  Derived by the caller from
                                  ``LearnedFingerprintProfile.embeddings``.

        Returns:
            Tuple of ``(health_score, health_percentage, health_state)`` where:
            - ``health_score`` is a float in [0, 100].
            - ``health_percentage`` is a formatted string e.g. ``"82.5%"``.
            - ``health_state`` is one of ``EXCELLENT``, ``GOOD``, ``WARNING``, ``CRITICAL``.
        """
        # A healthy recording sits at ~profile_healthy_norm.
        # Map [0, 2 * profile_healthy_norm] → [100, 0] linearly.
        scale = 2.0 * max(profile_healthy_norm, 1e-8)
        raw = 100.0 * (1.0 - normalized_euclidean / scale)
        health_score = max(0.0, min(100.0, raw))
        health_percentage = f"{health_score:.1f}%"
        health_state = self._classify(health_score)
        return health_score, health_percentage, health_state

    def _classify(self, score: float) -> str:
        if score >= self._thresholds["EXCELLENT"]:
            return "EXCELLENT"
        if score >= self._thresholds["GOOD"]:
            return "GOOD"
        if score >= self._thresholds["WARNING"]:
            return "WARNING"
        return "CRITICAL"
