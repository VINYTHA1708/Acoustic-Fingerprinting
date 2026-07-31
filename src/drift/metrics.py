"""DriftMetrics: computes per-feature and overall drift between a fingerprint and a profile.

Normalization note
------------------
Raw metrics compare the current feature vector directly against the profile mean.
Normalized metrics operate on the z-score vector:

    normalized_vector = (current - mean) / std

where std == 0 dimensions are clamped to avoid division by zero (deviation is
treated as zero for those dimensions). Validation experiments showed normalized
distances provide more consistent healthy/abnormal separation across machine IDs.
The normalized representation is the official input to the Health Index module.
"""

import logging

import numpy as np

from fingerprint.fingerprint import AcousticFingerprint
from fingerprint.similarity import FingerprintSimilarity
from profile.profile import HealthyFingerprintProfile

logger = logging.getLogger(__name__)

_STD_FLOOR = 1e-10
_similarity = FingerprintSimilarity()


def _profile_as_fingerprint(profile: HealthyFingerprintProfile) -> AcousticFingerprint:
    """Wrap a profile's mean vector as an AcousticFingerprint for similarity reuse.

    Args:
        profile: The healthy fingerprint profile.

    Returns:
        A minimal ``AcousticFingerprint`` whose ``feature_vector`` is the
        profile's mean vector. Only ``feature_vector`` and ``feature_names``
        are meaningful on the returned object.
    """
    return AcousticFingerprint(
        machine_type=profile.machine_type,
        machine_id=profile.machine_id,
        label="normal",
        filename="__profile_mean__",
        sample_rate=0,
        feature_names=profile.feature_names,
        feature_vector=profile.mean_vector.astype(np.float32),
    )


def _compute_normalized_vector(
    current: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Compute the z-score normalized feature vector.

    Dimensions where std < _STD_FLOOR are treated as zero deviation to avoid
    division by zero on constant features.

    Args:
        current: Current feature vector (float32).
        mean: Profile mean vector (float32).
        std: Profile std vector (float32).

    Returns:
        Z-score normalized vector (float32), same shape as ``current``.
    """
    safe_std = np.where(std < _STD_FLOOR, 1.0, std)
    return np.where(std < _STD_FLOOR, 0.0, (current - mean) / safe_std).astype(np.float32)


def _compute_normalized_metrics(z: np.ndarray) -> tuple[float, float, float]:
    """Compute Euclidean, Manhattan, and Cosine distances of a z-score vector from zero.

    Euclidean and Manhattan measure total deviation magnitude in normalized space.
    Cosine measures directional alignment with the all-ones vector (uniform deviation).

    Args:
        z: Z-score normalized feature vector.

    Returns:
        Tuple of (euclidean, manhattan, cosine) floats.
    """
    euclidean = float(np.linalg.norm(z))
    manhattan = float(np.sum(np.abs(z)))
    norm_z = np.linalg.norm(z)
    if norm_z > 0:
        ones = np.ones_like(z)
        cosine = float(np.dot(z, ones) / (norm_z * np.linalg.norm(ones)))
    else:
        cosine = 0.0
    return euclidean, manhattan, cosine


class DriftMetrics:
    """Computes drift metrics between a current fingerprint and a healthy profile.

    Reuses :class:`~fingerprint.similarity.FingerprintSimilarity` for the three
    raw distance metrics to avoid duplicating implementations.
    """

    def compute(
        self,
        fingerprint: AcousticFingerprint,
        profile: HealthyFingerprintProfile,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray, float, float, float, np.ndarray]:
        """Compute all drift metrics for one fingerprint against a profile.

        Args:
            fingerprint: The current recording's fingerprint.
            profile: The healthy reference profile for the same machine.

        Returns:
            A tuple of:
            - ``cosine_similarity`` (float) — raw
            - ``euclidean_distance`` (float) — raw
            - ``manhattan_distance`` (float) — raw
            - ``z_score_vector`` (float32 ndarray, shape ``(D,)``)
            - ``absolute_difference_vector`` (float32 ndarray, shape ``(D,)``)
            - ``norm_euclidean_distance`` (float) — normalized
            - ``norm_manhattan_distance`` (float) — normalized
            - ``norm_cosine_similarity`` (float) — normalized
            - ``normalized_vector`` (float32 ndarray, shape ``(D,)``)
        """
        current = fingerprint.feature_vector.astype(np.float32)
        mean    = profile.mean_vector.astype(np.float32)
        std     = profile.std_vector.astype(np.float32)

        abs_diff = np.abs(current - mean)

        safe_std = np.where(std < _STD_FLOOR, 1.0, std)
        z_scores = np.where(std < _STD_FLOOR, 0.0, (current - mean) / safe_std).astype(np.float32)

        # Raw metrics — current vector vs profile mean vector
        profile_fp = _profile_as_fingerprint(profile)
        cosine  = _similarity.cosine_similarity(fingerprint, profile_fp)
        euclid  = _similarity.euclidean_distance(fingerprint, profile_fp)
        manhat  = _similarity.manhattan_distance(fingerprint, profile_fp)

        # Normalized metrics — z-score vector vs zero vector
        norm_vec = _compute_normalized_vector(current, mean, std)
        norm_euclid, norm_manhat, norm_cosine = _compute_normalized_metrics(norm_vec)

        logger.debug(
            "Drift computed — raw: cosine=%.4f euclidean=%.4f manhattan=%.4f "
            "| normalized: euclidean=%.4f manhattan=%.4f cosine=%.4f",
            cosine, euclid, manhat, norm_euclid, norm_manhat, norm_cosine,
        )
        return cosine, euclid, manhat, z_scores, abs_diff, norm_euclid, norm_manhat, norm_cosine, norm_vec
