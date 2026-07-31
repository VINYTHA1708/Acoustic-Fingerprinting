"""FeatureExtractor: orchestrates all DSP sub-extractors into a single feature dict."""

import logging

import numpy as np

from .harmonic import HarmonicExtractor
from .mfcc import MFCCExtractor
from .spectral import SpectralExtractor
from .temporal import TemporalExtractor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts the full DSP feature set from a waveform.

    Runs MFCC, spectral, temporal, and harmonic extractors and merges
    their outputs into a single flat dictionary. All keys are unique and
    deterministically ordered (insertion order, Python 3.7+).

    Args:
        sample_rate: Sample rate of the input waveform in Hz (default: 16 000).
        n_mfcc: Number of MFCC coefficients (default: 20).
        n_fft: FFT window size shared by spectral extractors (default: 1024).
        hop_length: Hop length in samples (default: 512).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mfcc: int = 20,
        n_fft: int = 1024,
        hop_length: int = 512,
    ) -> None:
        self._mfcc = MFCCExtractor(n_mfcc=n_mfcc, sample_rate=sample_rate)
        self._spectral = SpectralExtractor(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length
        )
        self._temporal = TemporalExtractor(frame_length=n_fft, hop_length=hop_length)
        self._harmonic = HarmonicExtractor()

    def extract(self, waveform: np.ndarray, sample_rate: int | None = None) -> dict[str, float]:
        """Extract all DSP features from a waveform.

        Args:
            waveform: 1-D float32 audio waveform.
            sample_rate: Ignored — present for API symmetry with future extractors.
                The sample rate is fixed at construction time.

        Returns:
            Flat dict mapping feature name → float value. Key order is
            deterministic: MFCC → spectral → temporal → harmonic.
        """
        features: dict[str, float] = {}
        features.update(self._mfcc.extract(waveform))
        features.update(self._spectral.extract(waveform))
        features.update(self._temporal.extract(waveform))
        features.update(self._harmonic.extract(waveform))

        logger.info("Total DSP features extracted: %d.", len(features))
        return features
