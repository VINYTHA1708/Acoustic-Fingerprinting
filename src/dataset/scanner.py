"""Recursive .wav file scanner, dataset-agnostic."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def scan_audio_files(root: str | Path, extension: str = ".wav") -> list[Path]:
    """Recursively scan a directory and return absolute paths of all audio files.

    Args:
        root: Root directory to scan.
        extension: Audio file extension to match (default: ``.wav``).

    Returns:
        Sorted list of absolute ``Path`` objects for every matched file.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        NotADirectoryError: If ``root`` is not a directory.
    """
    root = Path(root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {root}")

    files = sorted(root.rglob(f"*{extension}"))
    logger.info("Found %d %s files under %s", len(files), extension, root)
    return files
