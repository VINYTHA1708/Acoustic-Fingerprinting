"""Temporal feature extraction: RMS energy, short-time energy, dynamic range."""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_FRAME_LENGTH = 1024
_DEFAULT_HOP_LENGTH = 512


class TemporalExtractor:
    """Extracts temporal energy features from a waveform.

    Args:
        frame_length: Frame length in samples (default: 1024).
        hop_length: Hop length in samples (default: 512).
    """

    def __init__(
        self,
        frame_length: int = _DEFAULT_FRAME_LENGTH,
        hop_length: int = _DEFAULT_HOP_LENGTH,
    ) -> None:
        self._frame_length = frame_length
        self._hop_length = hop_length

    def extract(self, waveform: np.ndarray) -> dict[str, float]:
        """Extract temporal energy statistics from a waveform.

        Args:
            waveform: 1-D float32 audio waveform.

        Returns:
            Flat dict with RMS energy (mean, std, max), short-time energy
            (mean, std), and dynamic range (dB).
        """
        rms = librosa.feature.rms(
            y=waveform,
            frame_length=self._frame_length,
            hop_length=self._hop_length,
        ).ravel()

        # Short-time energy: sum of squared samples per frame
        frames = librosa.util.frame(
            waveform,
            frame_length=self._frame_length,
            hop_length=self._hop_length,
        )
        ste = np.sum(frames ** 2, axis=0)

        # Dynamic range: difference between max and min RMS in dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        dynamic_range_db = float(rms_db.max() - rms_db.min())

        features: dict[str, float] = {
            "rms_mean": float(rms.mean()),
            "rms_std": float(rms.std()),
            "rms_max": float(rms.max()),
            "ste_mean": float(ste.mean()),
            "ste_std": float(ste.std()),
            "dynamic_range_db": dynamic_range_db,
        }

        logger.debug("Temporal features extracted: %d values.", len(features))
        return features
