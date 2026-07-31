"""Audio resampling to a configurable target sample rate."""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_TARGET_SR = 16_000


class AudioResampler:
    """Resamples a waveform to a target sample rate.

    Args:
        target_sr: Desired output sample rate in Hz (default: 16 000).
    """

    def __init__(self, target_sr: int = _DEFAULT_TARGET_SR) -> None:
        self._target_sr = target_sr

    def resample(self, waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Resample ``waveform`` to the target sample rate.

        If ``sample_rate`` already equals the target rate the waveform is
        returned unchanged (no copy, no processing).

        Args:
            waveform: 1-D float32 audio waveform.
            sample_rate: Native sample rate of ``waveform`` in Hz.

        Returns:
            A tuple of ``(resampled_waveform, target_sr)``.
        """
        if sample_rate == self._target_sr:
            logger.debug("Resample skipped — already at %d Hz.", self._target_sr)
            return waveform, sample_rate

        resampled = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self._target_sr)
        logger.debug("Resampled %d Hz → %d Hz.", sample_rate, self._target_sr)
        return resampled, self._target_sr
