"""LearnedHealthCalculator — converts normalized drift metrics into a health score.

SDD v4 §8:
    The normalized Euclidean distance is the primary anomaly score.
    Health score is bounded in [0, 100].
    Healthy recordings produce higher scores; larger drift reduces the score.

Calibration method — Gaussian survival function anchored to the healthy distribution
-------------------------------------------------------------------------------------
The z-score vector z_i = (e_i − μ) / σ has a norm ‖z_i‖ that follows a
chi distribution with 256 degrees of freedom.  For a 256-dimensional embedding
space the healthy norms cluster around √256 = 16, not around 0.  A linear
formula score = 100 × (1 − ‖z‖ / scale) is therefore miscalibrated by
construction: no choice of scale can simultaneously place healthy recordings
near 100 and anomalous recordings near 0, because the healthy distribution
does not sit near zero.

The correct approach is to measure how anomalous a recording is *relative to
the spread of the healthy distribution itself*, using a second-order z-score:

    t = (‖z_new‖ − μ_norm) / σ_norm

where μ_norm = mean(‖z_i‖) and σ_norm = std(‖z_i‖) over all healthy
embeddings in the profile.

The health score is then:

    score = 100 × Φ(c − t)

where Φ is the standard normal CDF and c = Φ⁻¹(0.95) ≈ 1.6449.

This anchoring means:
    t = 0  (recording at the healthy mean)  → score = 100 × Φ(1.6449) = 95
    t = −1 (1σ below healthy mean)          → score ≈ 99.6   EXCELLENT
    t = +1 (1σ above healthy mean)          → score ≈ 74      GOOD
    t = +2 (2σ above healthy mean)          → score ≈ 36      CRITICAL
    t = +3 (3σ above healthy mean)          → score ≈ 9       CRITICAL

Properties:
    - No hand-tuned constants.  c is derived from the requirement that the
      healthy mean maps to 95, which is a statistical statement, not a guess.
    - Generalises across machine types automatically: μ_norm and σ_norm are
      computed from the profile for each machine independently.
    - Smooth and monotone: larger drift always produces a lower score.
    - Asymptotically approaches 100 for very healthy recordings and 0 for
      extreme outliers without hard clipping distorting the gradient.
    - Score is still clamped to [0, 100] for display purposes.

Status bands (SDD v4 §8.2):
    90–100  EXCELLENT
    75–89   GOOD
    50–74   WARNING
    0–49    CRITICAL
"""

from __future__ import annotations

import math

_THRESHOLDS = {
    "EXCELLENT": 90.0,
    "GOOD": 75.0,
    "WARNING": 50.0,
}

# c = Φ⁻¹(0.95) — anchors the healthy mean to a score of 95.
# Derived from: score(t=0) = 100 × Φ(c) = 95  →  c = Φ⁻¹(0.95).
# scipy.special.ndtri(0.95) = 1.6448536269514729
_ANCHOR = 1.6448536269514729


def _standard_normal_cdf(x: float) -> float:
    """Standard normal CDF Φ(x) using math.erfc for zero-dependency computation."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


class LearnedHealthCalculator:
    """Computes a bounded health score from normalized drift metrics.

    Uses a Gaussian survival function anchored to the healthy distribution so
    that a recording at the healthy mean scores 95 and scores degrade smoothly
    as drift increases beyond the healthy spread.

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
        profile_healthy_norm_std: float,
    ) -> tuple[float, str, str]:
        """Compute health score, percentage, and state from normalized drift metrics.

        Computes the second-order z-score of the recording's normalized
        Euclidean distance relative to the healthy distribution, then maps it
        through a Gaussian survival function anchored so the healthy mean
        produces a score of 95.

        Args:
            normalized_euclidean: Normalized Euclidean distance ‖z_new‖ (primary input).
            normalized_manhattan: Normalized Manhattan distance (unused in score, kept
                                  for API compatibility and future use).
            normalized_cosine: Normalized cosine similarity (unused in score, kept
                               for API compatibility and future use).
            profile_healthy_norm: Mean ‖z_i‖ of healthy embeddings (μ_norm).
            profile_healthy_norm_std: Std of ‖z_i‖ of healthy embeddings (σ_norm).

        Returns:
            Tuple of ``(health_score, health_percentage, health_state)`` where:
            - ``health_score`` is a float in [0, 100].
            - ``health_percentage`` is a formatted string e.g. ``\"82.5%\"``.
            - ``health_state`` is one of ``EXCELLENT``, ``GOOD``, ``WARNING``, ``CRITICAL``.
        """
        safe_std = max(profile_healthy_norm_std, 1e-8)

        # Second-order z-score: how many healthy-distribution standard deviations
        # does this recording's drift exceed the healthy mean?
        t = (normalized_euclidean - profile_healthy_norm) / safe_std

        # Gaussian survival function anchored so t=0 → score=95.
        # score = 100 × Φ(c − t)  where c = Φ⁻¹(0.95) ≈ 1.6449
        raw = 100.0 * _standard_normal_cdf(_ANCHOR - t)
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
