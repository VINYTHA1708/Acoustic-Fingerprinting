"""MFCC, delta-MFCC, and delta-delta-MFCC feature extraction."""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_N_MFCC = 20


def _mean_std(matrix: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute per-coefficient mean and std and return a named flat dict.

    Args:
        matrix: 2-D array of shape ``(n_coeffs, time_frames)``.
        prefix: Key prefix, e.g. ``"mfcc"``.

    Returns:
        Dict with keys ``"<prefix>_<i>_mean"`` and ``"<prefix>_<i>_std"``.
    """
    return {
        **{f"{prefix}_{i}_mean": float(matrix[i].mean()) for i in range(matrix.shape[0])},
        **{f"{prefix}_{i}_std": float(matrix[i].std()) for i in range(matrix.shape[0])},
    }


class MFCCExtractor:
    """Extracts MFCC, delta-MFCC, and delta-delta-MFCC statistics.

    Args:
        n_mfcc: Number of MFCC coefficients (default: 20).
        sample_rate: Sample rate of the input waveform in Hz (default: 16 000).
    """

    def __init__(self, n_mfcc: int = _DEFAULT_N_MFCC, sample_rate: int = 16_000) -> None:
        self._n_mfcc = n_mfcc
        self._sr = sample_rate

    def extract(self, waveform: np.ndarray) -> dict[str, float]:
        """Extract MFCC-family statistics from a waveform.

        Args:
            waveform: 1-D float32 audio waveform.

        Returns:
            Flat dict of ``n_mfcc * 2 * 3`` named float values covering
            mean and std for MFCC, delta-MFCC, and delta-delta-MFCC.
        """
        mfcc = librosa.feature.mfcc(y=waveform, sr=self._sr, n_mfcc=self._n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features: dict[str, float] = {}
        features.update(_mean_std(mfcc, "mfcc"))
        features.update(_mean_std(delta, "mfcc_delta"))
        features.update(_mean_std(delta2, "mfcc_delta2"))

        logger.debug("MFCC features extracted: %d values.", len(features))
        return features
