"""End-to-end audio preprocessing pipeline."""

import logging
from pathlib import Path
from typing import TypedDict

import numpy as np

from .audio_loader import AudioLoader
from .normalizer import AudioNormalizer
from .resampler import AudioResampler
from .spectrogram import SpectrogramGenerator

logger = logging.getLogger(__name__)


class PreprocessingResult(TypedDict):
    """Typed output of :class:`PreprocessingPipeline`."""

    waveform: np.ndarray
    sample_rate: int
    spectrogram: np.ndarray


class PreprocessingPipeline:
    """Runs the full preprocessing chain on a single audio file.

    Pipeline order:
        1. Load audio (mono conversion included).
        2. Resample to target sample rate.
        3. Normalize amplitude to [-1, 1].
        4. Generate log-Mel spectrogram.

    Args:
        target_sr: Target sample rate in Hz (default: 16 000).
        n_fft: FFT window size passed to :class:`SpectrogramGenerator`.
        hop_length: Hop length passed to :class:`SpectrogramGenerator`.
        win_length: Window length passed to :class:`SpectrogramGenerator`.
        n_mels: Number of Mel bands passed to :class:`SpectrogramGenerator`.
        fmin: Minimum frequency for Mel filter bank in Hz.
        fmax: Maximum frequency for Mel filter bank in Hz.
            Defaults to ``target_sr // 2``.
    """

    def __init__(
        self,
        target_sr: int = 16_000,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int = 1024,
        n_mels: int = 128,
        fmin: float = 20.0,
        fmax: float | None = None,
    ) -> None:
        self._loader = AudioLoader(mono=True)
        self._resampler = AudioResampler(target_sr=target_sr)
        self._normalizer = AudioNormalizer()
        self._spectrogram = SpectrogramGenerator(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax if fmax is not None else target_sr // 2,
        )

    def run(self, path: str | Path) -> PreprocessingResult:
        """Execute the full preprocessing pipeline on a single audio file.

        Args:
            path: Path to the ``.wav`` file to process.

        Returns:
            A :class:`PreprocessingResult` dict with keys:
            ``waveform``, ``sample_rate``, and ``spectrogram``.

        Raises:
            FileNotFoundError: Propagated from :class:`AudioLoader`.
            ValueError: Propagated from :class:`AudioLoader`.
        """
        path = Path(path)
        logger.info("Preprocessing: %s", path.name)

        waveform, original_sr = self._loader.load(path)
        logger.debug("Original sr=%d Hz, shape=%s", original_sr, waveform.shape)

        waveform, sample_rate = self._resampler.resample(waveform, original_sr)
        waveform = self._normalizer.normalize(waveform)
        spectrogram = self._spectrogram.generate(waveform)

        logger.info(
            "Done — waveform %s @ %d Hz | spectrogram %s",
            waveform.shape,
            sample_rate,
            spectrogram.shape,
        )

        return PreprocessingResult(
            waveform=waveform,
            sample_rate=sample_rate,
            spectrogram=spectrogram,
        )
