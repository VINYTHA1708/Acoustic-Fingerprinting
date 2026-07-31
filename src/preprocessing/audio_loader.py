"""Audio file loading with mono conversion."""

import logging
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class AudioLoader:
    """Loads a .wav file and converts it to mono.

    Args:
        mono: If ``True`` (default), convert stereo to mono after loading.
    """

    def __init__(self, mono: bool = True) -> None:
        self._mono = mono

    def load(self, path: str | Path) -> tuple[np.ndarray, int]:
        """Load an audio file and return the waveform and its native sample rate.

        Args:
            path: Path to the ``.wav`` file.

        Returns:
            A tuple of ``(waveform, sample_rate)`` where ``waveform`` is a
            1-D float32 numpy array (mono) or 2-D array (stereo, channels-first)
            and ``sample_rate`` is the file's native rate in Hz.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or the format is unsupported.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.suffix.lower() not in {".wav", ".wave"}:
            raise ValueError(f"Unsupported audio format '{path.suffix}': {path}")

        try:
            waveform, sample_rate = librosa.load(path, sr=None, mono=self._mono)
        except Exception as exc:
            raise ValueError(f"Failed to load audio file '{path}': {exc}") from exc

        if waveform.size == 0:
            raise ValueError(f"Audio file is empty: {path}")

        logger.debug("Loaded '%s' — sr=%d Hz, shape=%s", path.name, sample_rate, waveform.shape)
        return waveform, int(sample_rate)
