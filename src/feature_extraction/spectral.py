"""Spectral feature extraction: centroid, bandwidth, rolloff, flatness, contrast, ZCR."""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def _frame_stats(frames: np.ndarray, name: str) -> dict[str, float]:
    """Return mean and std of a 1-D or 2-D (1, T) frame array under a named key.

    Args:
        frames: Array of shape ``(T,)`` or ``(1, T)``.
        name: Feature name used as key prefix.

    Returns:
        Dict with ``"<name>_mean"`` and ``"<name>_std"``.
    """
    flat = frames.ravel()
    return {f"{name}_mean": float(flat.mean()), f"{name}_std": float(flat.std())}


class SpectralExtractor:
    """Extracts spectral features from a waveform.

    Args:
        sample_rate: Sample rate of the input waveform in Hz (default: 16 000).
        n_fft: FFT window size (default: 1024).
        hop_length: Hop length in samples (default: 512).
        n_bands: Number of bands for spectral contrast (default: 6).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_bands: int = 6,
    ) -> None:
        self._sr = sample_rate
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._n_bands = n_bands

    def extract(self, waveform: np.ndarray) -> dict[str, float]:
        """Extract spectral statistics from a waveform.

        Args:
            waveform: 1-D float32 audio waveform.

        Returns:
            Flat dict of named float values for centroid, bandwidth, rolloff,
            flatness, zero-crossing rate (mean + std each), and per-band
            spectral contrast mean + std.
        """
        kw = dict(y=waveform, sr=self._sr, n_fft=self._n_fft, hop_length=self._hop_length)

        features: dict[str, float] = {}
        features.update(_frame_stats(librosa.feature.spectral_centroid(**kw), "spectral_centroid"))
        features.update(_frame_stats(librosa.feature.spectral_bandwidth(**kw), "spectral_bandwidth"))
        features.update(_frame_stats(librosa.feature.spectral_rolloff(**kw), "spectral_rolloff"))
        features.update(_frame_stats(librosa.feature.spectral_flatness(y=waveform, n_fft=self._n_fft, hop_length=self._hop_length), "spectral_flatness"))
        features.update(_frame_stats(librosa.feature.zero_crossing_rate(waveform, hop_length=self._hop_length), "zcr"))

        # Spectral contrast: shape (n_bands+1, T) — store per-band stats
        contrast = librosa.feature.spectral_contrast(
            y=waveform, sr=self._sr, n_fft=self._n_fft,
            hop_length=self._hop_length, n_bands=self._n_bands,
        )
        for band_idx in range(contrast.shape[0]):
            tag = f"spectral_contrast_band{band_idx}"
            features[f"{tag}_mean"] = float(contrast[band_idx].mean())
            features[f"{tag}_std"] = float(contrast[band_idx].std())

        logger.debug("Spectral features extracted: %d values.", len(features))
        return features
