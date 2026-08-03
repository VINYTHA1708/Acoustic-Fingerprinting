"""Audio file metadata extraction from MIMII-style directory paths.

Expected path convention (relative to dataset root):
    <machine_type>/<machine_id>/<label>/<filename>.wav

Example:
    fan/id_00/normal/00000000.wav
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_VALID_LABELS = {"normal", "abnormal"}


@dataclass(frozen=True)
class AudioMetadata:
    """Metadata extracted from a single audio file path.

    Attributes:
        machine_type: Type of machine (e.g. ``fan``, ``pump``).
        machine_id: Specific machine identifier (e.g. ``id_00``).
        label: Recording condition — ``normal`` or ``abnormal``.
        filename: Bare filename including extension.
        relative_path: Path relative to the dataset root.
        absolute_path: Fully resolved absolute path.
        is_uploaded: True for temporary uploads that should bypass the dataset cache.
    """

    machine_type: str
    machine_id: str
    label: str
    filename: str
    relative_path: Path
    absolute_path: Path
    is_uploaded: bool = False


def extract_metadata(file_path: Path, root: Path) -> AudioMetadata | None:
    """Extract metadata from a file path following the MIMII directory convention.

    Args:
        file_path: Absolute path to the audio file.
        root: Absolute dataset root used to compute the relative path.

    Returns:
        An ``AudioMetadata`` instance, or ``None`` if the path does not match
        the expected ``<machine_type>/<machine_id>/<label>/`` structure.
    """
    try:
        relative = file_path.relative_to(root)
        parts = relative.parts  # (machine_type, machine_id, label, filename)

        if len(parts) < 4:
            logger.warning("Skipping (too few path segments): %s", file_path)
            return None

        machine_type, machine_id, label = parts[0], parts[1], parts[2]

        if label not in _VALID_LABELS:
            logger.warning("Skipping (unknown label '%s'): %s", label, file_path)
            return None

        return AudioMetadata(
            machine_type=machine_type,
            machine_id=machine_id,
            label=label,
            filename=file_path.name,
            relative_path=relative,
            absolute_path=file_path,
        )

    except ValueError:
        logger.warning("Skipping (path not under root): %s", file_path)
        return None
