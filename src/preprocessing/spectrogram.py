"""Log-Mel spectrogram generation."""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_SR = 16_000
_DEFAULT_N_FFT = 1024
_DEFAULT_HOP_LENGTH = 512
_DEFAULT_WIN_LENGTH = 1024
_DEFAULT_N_MELS = 128
_DEFAULT_FMIN = 20


class SpectrogramGenerator:
    """Generates a log-Mel spectrogram from a waveform.

    Args:
        sample_rate: Expected sample rate of the input waveform (default: 16 000).
        n_fft: FFT window size (default: 1024).
        hop_length: Number of samples between frames (default: 512).
        win_length: Window length in samples (default: 1024).
        n_mels: Number of Mel filter banks (default: 128).
        fmin: Lowest frequency for the Mel filter bank in Hz (default: 20).
        fmax: Highest frequency for the Mel filter bank in Hz.
            Defaults to ``sample_rate // 2`` (Nyquist).
    """

    def __init__(
        self,
        sample_rate: int = _DEFAULT_SR,
        n_fft: int = _DEFAULT_N_FFT,
        hop_length: int = _DEFAULT_HOP_LENGTH,
        win_length: int = _DEFAULT_WIN_LENGTH,
        n_mels: int = _DEFAULT_N_MELS,
        fmin: float = _DEFAULT_FMIN,
        fmax: float | None = None,
    ) -> None:
        self._sr = sample_rate
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        self._n_mels = n_mels
        self._fmin = fmin
        self._fmax = fmax if fmax is not None else sample_rate // 2

    def generate(self, waveform: np.ndarray) -> np.ndarray:
        """Compute the log-Mel spectrogram of ``waveform``.

        Args:
            waveform: 1-D float32 audio waveform at ``self.sample_rate`` Hz.

        Returns:
            Log-Mel spectrogram as a 2-D float32 numpy array of shape
            ``(n_mels, time_frames)``.
        """
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self._sr,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._win_length,
            n_mels=self._n_mels,
            fmin=self._fmin,
            fmax=self._fmax,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        logger.debug("Spectrogram shape: %s", log_mel.shape)
        return log_mel
