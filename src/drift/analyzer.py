"""DriftAnalyzer: validates inputs and produces a DriftResult."""

import logging

from fingerprint.fingerprint import AcousticFingerprint
from profile.profile import HealthyFingerprintProfile
from .drift_result import DriftResult
from .metrics import DriftMetrics

logger = logging.getLogger(__name__)


class DriftAnalyzer:
    """Compares a current :class:`~fingerprint.fingerprint.AcousticFingerprint`
    against a :class:`~profile.profile.HealthyFingerprintProfile` and returns
    a :class:`~drift.drift_result.DriftResult`.

    Validates that the fingerprint and profile belong to the same machine and
    share identical feature schemas before computing any metrics.
    """

    def __init__(self) -> None:
        self._metrics = DriftMetrics()

    def analyze(
        self,
        fingerprint: AcousticFingerprint,
        profile: HealthyFingerprintProfile,
    ) -> DriftResult:
        """Run drift analysis for one fingerprint against a healthy profile.

        Args:
            fingerprint: The current recording's fingerprint.
            profile: The healthy reference profile for the same machine.

        Returns:
            A :class:`~drift.drift_result.DriftResult` containing all drift metrics.

        Raises:
            ValueError: If machine type, machine ID, feature names, or vector
                lengths do not match between the fingerprint and the profile.
        """
        self._validate(fingerprint, profile)

        (
            cosine, euclid, manhat, z_scores, abs_diff,
            norm_euclid, norm_manhat, norm_cosine, norm_vec,
        ) = self._metrics.compute(fingerprint, profile)

        result = DriftResult(
            machine_type=fingerprint.machine_type,
            machine_id=fingerprint.machine_id,
            filename=fingerprint.filename,
            feature_names=fingerprint.feature_names,
            # Raw metrics
            cosine_similarity=cosine,
            euclidean_distance=euclid,
            manhattan_distance=manhat,
            # Per-feature arrays
            z_score_vector=z_scores,
            absolute_difference_vector=abs_diff,
            # Normalized metrics (official Health Index input)
            norm_euclidean_distance=norm_euclid,
            norm_manhattan_distance=norm_manhat,
            norm_cosine_similarity=norm_cosine,
            normalized_vector=norm_vec,
        )
        logger.info(
            "Drift analysis complete — %s/%s '%s' "
            "raw euclidean=%.4f | norm euclidean=%.4f",
            result.machine_type, result.machine_id, result.filename,
            result.euclidean_distance, result.norm_euclidean_distance,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(
        fingerprint: AcousticFingerprint,
        profile: HealthyFingerprintProfile,
    ) -> None:
        """Validate compatibility between fingerprint and profile.

        Args:
            fingerprint: Current recording fingerprint.
            profile: Healthy reference profile.

        Raises:
            ValueError: On any mismatch.
        """
        if fingerprint.machine_type != profile.machine_type:
            raise ValueError(
                f"machine_type mismatch: fingerprint='{fingerprint.machine_type}' "
                f"vs profile='{profile.machine_type}'."
            )
        if fingerprint.machine_id != profile.machine_id:
            raise ValueError(
                f"machine_id mismatch: fingerprint='{fingerprint.machine_id}' "
                f"vs profile='{profile.machine_id}'."
            )
        if fingerprint.feature_names != profile.feature_names:
            raise ValueError(
                "feature_names mismatch between fingerprint and profile."
            )
        if len(fingerprint.feature_vector) != len(profile.mean_vector):
            raise ValueError(
                f"Vector length mismatch: fingerprint={len(fingerprint.feature_vector)} "
                f"vs profile={len(profile.mean_vector)}."
            )
