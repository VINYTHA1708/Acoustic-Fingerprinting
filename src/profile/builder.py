"""HealthyProfileBuilder: builds a HealthyFingerprintProfile from AcousticFingerprints."""

import logging

from fingerprint.fingerprint import AcousticFingerprint
from .profile import HealthyFingerprintProfile
from .statistics import ProfileStatistics

logger = logging.getLogger(__name__)

_NORMAL_LABEL = "normal"


class HealthyProfileBuilder:
    """Builds a :class:`~profile.profile.HealthyFingerprintProfile` from a collection
    of healthy :class:`~fingerprint.fingerprint.AcousticFingerprint` objects.

    All fingerprints must:
    - Be labeled ``"normal"``.
    - Belong to the same ``machine_type`` and ``machine_id``.
    - Share identical ``feature_names``.
    """

    def __init__(self) -> None:
        self._stats = ProfileStatistics()

    def build(self, fingerprints: list[AcousticFingerprint]) -> HealthyFingerprintProfile:
        """Build a healthy fingerprint profile from a list of normal fingerprints.

        Args:
            fingerprints: Non-empty list of ``AcousticFingerprint`` objects.

        Returns:
            A :class:`~profile.profile.HealthyFingerprintProfile` summarising
            the healthy acoustic distribution for the machine.

        Raises:
            ValueError: If the list is empty, contains non-normal labels,
                mixes machine types or IDs, or has inconsistent feature names.
        """
        self._validate(fingerprints)

        vectors = [fp.feature_vector for fp in fingerprints]
        mean, std, minimum, maximum = self._stats.compute(vectors)

        ref = fingerprints[0]
        profile = HealthyFingerprintProfile(
            machine_type=ref.machine_type,
            machine_id=ref.machine_id,
            number_of_samples=len(fingerprints),
            feature_names=ref.feature_names,
            mean_vector=mean,
            std_vector=std,
            min_vector=minimum,
            max_vector=maximum,
        )
        logger.info(
            "Profile built — machine=%s/%s samples=%d features=%d",
            profile.machine_type, profile.machine_id,
            profile.number_of_samples, len(profile.feature_names),
        )
        return profile

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(fingerprints: list[AcousticFingerprint]) -> None:
        """Validate the fingerprint collection before building.

        Args:
            fingerprints: List of fingerprints to validate.

        Raises:
            ValueError: On any validation failure.
        """
        if not fingerprints:
            raise ValueError("At least one fingerprint is required to build a profile.")

        ref = fingerprints[0]

        for i, fp in enumerate(fingerprints):
            if fp.label != _NORMAL_LABEL:
                raise ValueError(
                    f"Fingerprint at index {i} ('{fp.filename}') has label "
                    f"'{fp.label}'; only '{_NORMAL_LABEL}' recordings are accepted."
                )
            if fp.machine_type != ref.machine_type:
                raise ValueError(
                    f"Fingerprint at index {i} has machine_type '{fp.machine_type}', "
                    f"expected '{ref.machine_type}'."
                )
            if fp.machine_id != ref.machine_id:
                raise ValueError(
                    f"Fingerprint at index {i} has machine_id '{fp.machine_id}', "
                    f"expected '{ref.machine_id}'."
                )
            if fp.feature_names != ref.feature_names:
                raise ValueError(
                    f"Fingerprint at index {i} ('{fp.filename}') has different "
                    f"feature_names than the first fingerprint."
                )
