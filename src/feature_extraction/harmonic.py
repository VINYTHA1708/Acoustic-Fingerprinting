"""Harmonic feature extraction via HPSS (Harmonic-Percussive Source Separation)."""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

_SILENCE_FLOOR = 1e-10


class HarmonicExtractor:
    """Extracts harmonic and percussive energy features using HPSS.

    Uses ``librosa.effects.hpss`` to separate the harmonic and percussive
    components of the waveform, then computes energy-based statistics.
    """

    def extract(self, waveform: np.ndarray) -> dict[str, float]:
        """Extract harmonic/percussive energy features from a waveform.

        Args:
            waveform: 1-D float32 audio waveform.

        Returns:
            Flat dict with:
            - ``harmonic_energy``: mean squared amplitude of the harmonic component.
            - ``percussive_energy``: mean squared amplitude of the percussive component.
            - ``harmonic_ratio``: harmonic_energy / (harmonic_energy + percussive_energy).
              Returns 0.0 for silent recordings.
        """
        harmonic, percussive = librosa.effects.hpss(waveform)

        h_energy = float(np.mean(harmonic ** 2))
        p_energy = float(np.mean(percussive ** 2))
        total = h_energy + p_energy

        harmonic_ratio = (h_energy / total) if total > _SILENCE_FLOOR else 0.0

        features: dict[str, float] = {
            "harmonic_energy": h_energy,
            "percussive_energy": p_energy,
            "harmonic_ratio": harmonic_ratio,
        }

        logger.debug("Harmonic features extracted: %d values.", len(features))
        return features
