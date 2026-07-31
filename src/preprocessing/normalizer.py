"""Amplitude normalization of audio waveforms."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_SILENCE_THRESHOLD = 1e-9


class AudioNormalizer:
    """Normalizes a waveform to the range [-1, 1] using peak normalization.

    Silent recordings (peak amplitude below ``_SILENCE_THRESHOLD``) are
    returned as-is to avoid division-by-zero or amplifying pure noise.
    """

    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Peak-normalize ``waveform`` to [-1, 1].

        Args:
            waveform: 1-D float32 audio waveform.

        Returns:
            Normalized waveform with the same shape and dtype as the input.
        """
        peak = np.max(np.abs(waveform))

        if peak < _SILENCE_THRESHOLD:
            logger.warning("Silent or near-silent recording detected — normalization skipped.")
            return waveform

        normalized = waveform / peak
        logger.debug("Normalized waveform — peak was %.6f.", peak)
        return normalized
