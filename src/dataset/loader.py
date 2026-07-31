"""DatasetLoader: scans a dataset root and provides filtered views over audio metadata."""

import logging
from pathlib import Path

from .metadata import AudioMetadata, extract_metadata
from .scanner import scan_audio_files

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads and indexes all audio files under a dataset root directory.

    Args:
        root: Path to the dataset root (e.g. ``data/MIMII``).

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        NotADirectoryError: If ``root`` is not a directory.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._records: list[AudioMetadata] = self._load()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> list[AudioMetadata]:
        """Scan root and build the internal metadata list."""
        files = scan_audio_files(self._root)
        records: list[AudioMetadata] = []

        for f in files:
            meta = extract_metadata(f, self._root)
            if meta is not None:
                records.append(meta)

        skipped = len(files) - len(records)
        if skipped:
            logger.warning("%d file(s) skipped due to unrecognised path structure.", skipped)

        logger.info("Loaded %d audio records from %s", len(records), self._root)
        return records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_files(self) -> list[AudioMetadata]:
        """Return all loaded audio metadata records.

        Returns:
            List of ``AudioMetadata`` for every valid audio file found.
        """
        return list(self._records)

    def get_machine_types(self) -> list[str]:
        """Return sorted unique machine types present in the dataset.

        Returns:
            Sorted list of machine type strings (e.g. ``['fan', 'pump']``).
        """
        return sorted({r.machine_type for r in self._records})

    def get_machine_ids(self, machine_type: str | None = None) -> list[str]:
        """Return sorted unique machine IDs, optionally filtered by machine type.

        Args:
            machine_type: If provided, restrict to IDs belonging to this type.

        Returns:
            Sorted list of machine ID strings (e.g. ``['id_00', 'id_02']``).
        """
        records = self._filter(machine_type=machine_type)
        return sorted({r.machine_id for r in records})

    def filter_by_machine(self, machine_type: str) -> list[AudioMetadata]:
        """Return all records for a given machine type.

        Args:
            machine_type: Machine type to filter on (e.g. ``'fan'``).

        Returns:
            List of matching ``AudioMetadata`` records.
        """
        return self._filter(machine_type=machine_type)

    def filter_by_machine_id(self, machine_id: str) -> list[AudioMetadata]:
        """Return all records for a given machine ID across all machine types.

        Args:
            machine_id: Machine ID to filter on (e.g. ``'id_00'``).

        Returns:
            List of matching ``AudioMetadata`` records.
        """
        return self._filter(machine_id=machine_id)

    def filter_by_label(self, label: str) -> list[AudioMetadata]:
        """Return all records matching a label (``'normal'`` or ``'abnormal'``).

        Args:
            label: Label to filter on.

        Returns:
            List of matching ``AudioMetadata`` records.
        """
        return self._filter(label=label)

    def summary(self) -> None:
        """Print a human-readable summary of the loaded dataset to stdout."""
        normal = sum(1 for r in self._records if r.label == "normal")
        abnormal = sum(1 for r in self._records if r.label == "abnormal")

        print("=" * 40)
        print("Dataset Summary")
        print("=" * 40)
        print(f"Root              : {self._root}")
        print(f"Machine types     : {self.get_machine_types()}")
        print(f"Number of machines: {len(self.get_machine_types())}")
        print(f"Machine IDs       : {self.get_machine_ids()}")
        print(f"Normal recordings : {normal}")
        print(f"Abnormal recordings: {abnormal}")
        print(f"Total recordings  : {len(self._records)}")
        print("=" * 40)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter(
        self,
        machine_type: str | None = None,
        machine_id: str | None = None,
        label: str | None = None,
    ) -> list[AudioMetadata]:
        """Generic filter over the internal records list.

        Args:
            machine_type: Optional machine type constraint.
            machine_id: Optional machine ID constraint.
            label: Optional label constraint.

        Returns:
            Filtered list of ``AudioMetadata`` records.
        """
        results = self._records
        if machine_type is not None:
            results = [r for r in results if r.machine_type == machine_type]
        if machine_id is not None:
            results = [r for r in results if r.machine_id == machine_id]
        if label is not None:
            results = [r for r in results if r.label == label]
        return results
