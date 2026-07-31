"""Utility helpers for the BEATs encoder module.

Contains:
    - Waveform validation before passing to BEATs.
    - Mean-pooling of frame-level BEATs outputs to a single clip-level vector.
    - Checkpoint path resolution.

These are kept separate from encoder.py so they can be unit-tested independently
of the pretrained model weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_EXPECTED_SAMPLE_RATE = 16_000
_EMBEDDING_DIM = 768


def validate_waveform(waveform: np.ndarray, sample_rate: int) -> None:
    """Validate that a waveform is suitable for BEATs encoding.

    BEATs expects mono float32 audio at 16 kHz.

    Args:
        waveform: Audio waveform array. Must be 1-D and float32.
        sample_rate: Sample rate of the waveform.

    Raises:
        ValueError: If the waveform is not 1-D, not float32, or the sample
                    rate does not match the expected 16 000 Hz.
    """
    if waveform.ndim != 1:
        raise ValueError(
            f"BEATs requires a mono (1-D) waveform, got shape {waveform.shape}. "
            "Run PreprocessingPipeline first."
        )
    if waveform.dtype != np.float32:
        raise ValueError(
            f"BEATs requires a float32 waveform, got dtype {waveform.dtype}."
        )
    if sample_rate != _EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"BEATs requires a 16 000 Hz sample rate, got {sample_rate} Hz. "
            "Run PreprocessingPipeline with target_sr=16000 first."
        )


def mean_pool_frames(frame_embeddings: np.ndarray) -> np.ndarray:
    """Reduce frame-level BEATs outputs to a single clip-level embedding via mean pooling.

    BEATs produces one embedding per audio frame. Mean pooling over the time
    dimension is the standard approach for clip-level representation.

    Args:
        frame_embeddings: Frame-level embeddings, shape ``(T, 768)`` where T is
                          the number of frames.

    Returns:
        Clip-level embedding, shape ``(768,)``, dtype float32.

    Raises:
        ValueError: If ``frame_embeddings`` does not have shape ``(T, 768)``.

    TODO (V2): this function is called inside BEATsEncoder.encode() after the
    forward pass through the BEATs model. Verify that the pooling axis is
    correct for the specific BEATs variant being used.
    """
    if frame_embeddings.ndim != 2 or frame_embeddings.shape[1] != _EMBEDDING_DIM:
        raise ValueError(
            f"Expected frame_embeddings of shape (T, {_EMBEDDING_DIM}), "
            f"got {frame_embeddings.shape}."
        )
    return frame_embeddings.mean(axis=0).astype(np.float32)


def resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Resolve and validate the path to a BEATs checkpoint file.

    Args:
        checkpoint_path: Absolute or relative path to the ``.pt`` checkpoint.
                         Expected location: ``models/beats/<checkpoint>.pt``

    Returns:
        Resolved absolute :class:`~pathlib.Path`.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist at the given path.

    TODO (V2): call this inside BEATsEncoder._load_model() before torch.load().
    """
    path = Path(checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"BEATs checkpoint not found: {path}\n"
            "Download the pretrained checkpoint and place it under models/beats/.\n"
            "See third_party/beats/README.md for download instructions."
        )
    return path
