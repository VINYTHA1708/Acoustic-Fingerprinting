"""Dataset module: scanning, metadata extraction, and loading."""

from .loader import DatasetLoader
from .metadata import AudioMetadata, extract_metadata
from .scanner import scan_audio_files

__all__ = [
    "DatasetLoader",
    "AudioMetadata",
    "extract_metadata",
    "scan_audio_files",
]
