"""ContrastiveDataset — builds positive pairs for contrastive learning.

SDD v4 §2 (Version 4):
    Positive pairs: same machine (machine_type + machine_id), different recordings.
    Only normal recordings are used.
    NT-Xent treats all other batch samples as negatives implicitly — no explicit
    negative pairs are constructed.

Recording-level train/validation split
---------------------------------------
The split is performed on *recordings* (not pairs) before any pair is generated.
A recording assigned to the validation set never appears in any training pair,
preventing data leakage.

Each item returned is:
    (anchor_fused_vector, paired_fused_vector, label)
    label = 1  →  positive pair (same machine)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..beats.encoder import BEATsEncoder
from ..dataset.loader import DatasetLoader
from ..dataset.metadata import AudioMetadata
from ..feature_extraction.extractor import FeatureExtractor
from ..feature_extraction.feature_vector import FeatureVectorBuilder
from ..fusion.cache import FusionCache
from ..fusion.fused_vector import FusedFeatureVector
from ..fusion.fusion import FusionBuilder
from ..preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

_CHECKPOINT_REL = Path(__file__).resolve().parents[2] / "models" / "beats" / "BEATs_iter3_plus_AS2M.pt"


@dataclass(frozen=True)
class ContrastivePair:
    """A single contrastive pair.

    Attributes:
        anchor: Fused feature vector of the anchor recording.
        paired: Fused feature vector of the paired recording.
        label: Always 1 (positive — same machine, different recording).
    """

    anchor: FusedFeatureVector
    paired: FusedFeatureVector
    label: int  # 1 = positive (same machine)


class ContrastiveDataset:
    """Builds positive contrastive pairs from normal recordings in a MIMII-style dataset.

    Encodes every normal recording once (PreprocessingPipeline -> DSP -> BEATs ->
    FusionBuilder), then performs a recording-level train/validation split per
    machine before generating positive pairs.  This guarantees that no recording
    appears in both the training and validation pair sets.

    Accepts either an explicit list of ``AudioMetadata`` objects (preferred, no
    data leakage) or a ``dataset_root`` path for backward compatibility.
    When ``recordings`` is supplied, ``dataset_root`` must be ``None``.

    Args:
        dataset_root: Path to the dataset root (backward-compat; mutually
                      exclusive with ``recordings``).
        recordings: Explicit list of :class:`~src.dataset.metadata.AudioMetadata`
                    objects to use for training.  Must all have label ``'normal'``.
                    When provided, ``dataset_root`` must be ``None``.
        checkpoint_path: Path to the BEATs checkpoint. Defaults to the project
                         standard location ``models/beats/BEATs_iter3_plus_AS2M.pt``.
        cache_root: Root directory for the FusionCache.
        seed: Random seed used for pair sampling and shuffling.
        machine_type: If set, restrict to this machine type (only used with
                      ``dataset_root``; ignored when ``recordings`` is supplied).
        machine_id: If set, restrict to this machine ID (only used with
                    ``dataset_root``; ignored when ``recordings`` is supplied).
        max_recordings: Maximum total recordings to encode (only used with
                        ``dataset_root``; ignored when ``recordings`` is supplied).
        val_split: Fraction of recordings per machine reserved for validation.
                   Defaults to ``0.2``.

    Raises:
        ValueError: If both ``dataset_root`` and ``recordings`` are provided,
                    if neither is provided, if ``recordings`` is empty, if any
                    item in ``recordings`` is not an :class:`AudioMetadata`, or
                    if any recording has a label other than ``'normal'``.
    """

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        cache_root: str | Path | None = None,
        seed: int = 42,
        machine_type: str | None = None,
        machine_id: str | None = None,
        max_recordings: int | None = None,
        val_split: float = 0.2,
        recordings: list[AudioMetadata] | None = None,
    ) -> None:
        if not (0 < val_split < 1):
            raise ValueError(f"val_split must be in (0, 1), got {val_split}")

        if recordings is not None and dataset_root is not None:
            raise ValueError("Provide either 'recordings' or 'dataset_root', not both.")
        if recordings is None and dataset_root is None:
            raise ValueError("Either 'recordings' or 'dataset_root' must be provided.")

        # Validate explicit recordings before constructing any heavy objects.
        if recordings is not None:
            self._validate_explicit_recordings(recordings)

        self._rng = random.Random(seed)
        self._val_split = val_split
        checkpoint = Path(checkpoint_path) if checkpoint_path else _CHECKPOINT_REL
        _cache_root = Path(cache_root) if cache_root else _CHECKPOINT_REL.parents[2] / "data" / "fusion_cache"

        pipeline = PreprocessingPipeline(target_sr=16_000)
        extractor = FeatureExtractor(sample_rate=16_000)
        vec_builder = FeatureVectorBuilder()
        encoder = BEATsEncoder(checkpoint)
        fusion = FusionBuilder()

        self._cache = FusionCache(
            cache_root=_cache_root,
            pipeline=pipeline,
            extractor=extractor,
            vec_builder=vec_builder,
            encoder=encoder,
            fusion=fusion,
        )

        if recordings is not None:
            records = recordings
        else:
            loader = DatasetLoader(dataset_root)  # type: ignore[arg-type]
            records = loader.filter_by_label("normal")
            if machine_type is not None:
                records = [r for r in records if r.machine_type == machine_type]
            if machine_id is not None:
                records = [r for r in records if r.machine_id == machine_id]
            if max_recordings is not None:
                records = self._sample_evenly(records, max_recordings, pin_id=machine_id is not None)
            print(f"Machine type      : {machine_type or 'all'}")
            print(f"Machine ID        : {machine_id or 'all'}")
            print(f"Max recordings    : {max_recordings or 'all'}")

        print("Selected recordings")
        id_counts: dict[str, int] = {}
        for r in records:
            id_counts[r.machine_id] = id_counts.get(r.machine_id, 0) + 1
        for mid, cnt in sorted(id_counts.items()):
            print(f"  {mid} : {cnt}")
        logger.info("Recordings after filtering: %d", len(records))

        # Encode all normal recordings once and cache by (machine_type, machine_id)
        self._fused_by_machine: dict[tuple[str, str], list[FusedFeatureVector]] = {}
        self._all_fused: list[FusedFeatureVector] = []
        self._encode_all(records)

        # Recording-level split → pair generation
        self._train_positive_pairs: list[ContrastivePair] = []
        self._val_positive_pairs: list[ContrastivePair] = []
        self._build_split_pairs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def positive_pairs(self) -> list[ContrastivePair]:
        """All positive pairs across both train and validation sets."""
        return self._train_positive_pairs + self._val_positive_pairs

    @property
    def train_positive_pairs(self) -> list[ContrastivePair]:
        """Positive pairs built exclusively from training recordings."""
        return self._train_positive_pairs

    @property
    def val_positive_pairs(self) -> list[ContrastivePair]:
        """Positive pairs built exclusively from validation recordings."""
        return self._val_positive_pairs

    def machine_keys(self) -> list[tuple[str, str]]:
        """Sorted (machine_type, machine_id) keys present in the encoded recordings."""
        return sorted(self._fused_by_machine.keys())

    def machine_types(self) -> list[str]:
        """Sorted unique machine types present in the normal recordings."""
        return sorted({k[0] for k in self._fused_by_machine})

    def machine_ids(self) -> list[str]:
        """Sorted unique machine IDs present in the normal recordings."""
        return sorted({k[1] for k in self._fused_by_machine})

    def normal_recording_count(self) -> int:
        """Total number of normal recordings that were successfully encoded."""
        return len(self._all_fused)

    def __iter__(self) -> Iterator[ContrastivePair]:
        return iter(self.positive_pairs)

    def __len__(self) -> int:
        return len(self.positive_pairs)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_explicit_recordings(
        recordings: list[AudioMetadata],
    ) -> list[AudioMetadata]:
        """Validate an explicitly supplied recordings list.

        Args:
            recordings: Caller-supplied list of AudioMetadata objects.

        Returns:
            The validated list (unchanged).

        Raises:
            ValueError: If the list is empty, contains non-AudioMetadata items,
                        or contains any recording with a label other than
                        ``'normal'``.
        """
        if not recordings:
            raise ValueError("'recordings' must not be empty.")
        for i, rec in enumerate(recordings):
            if not hasattr(rec, "label") or not hasattr(rec, "machine_type") or not hasattr(rec, "machine_id"):
                raise ValueError(
                    f"recordings[{i}] is {type(rec).__name__!r}, expected AudioMetadata."
                )
            if rec.label != "normal":
                raise ValueError(
                    f"recordings[{i}] has label {rec.label!r}; "
                    "only 'normal' recordings may be used for contrastive training."
                )
        return recordings

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_evenly(
        records: list[AudioMetadata],
        max_recordings: int,
        pin_id: bool,
    ) -> list[AudioMetadata]:
        """Select up to max_recordings, distributed evenly across machine IDs.

        When pin_id is True (machine_id was explicitly set) there is only one
        group, so a simple truncation is used.  Otherwise the quota is spread
        across all distinct machine IDs using a round-robin fill that handles
        groups smaller than the per-ID quota by redistributing the remainder.
        """
        if pin_id:
            return records[:max_recordings]

        groups: dict[str, list[AudioMetadata]] = {}
        for r in records:
            groups.setdefault(r.machine_id, []).append(r)

        ids = sorted(groups.keys())
        if not ids:
            return []

        selected: list[AudioMetadata] = []
        remaining = max_recordings

        pending = list(ids)
        while remaining > 0 and pending:
            per_id = max(1, remaining // len(pending))
            next_pending = []
            for mid in pending:
                take = min(per_id, len(groups[mid]))
                selected.extend(groups[mid][:take])
                groups[mid] = groups[mid][take:]
                remaining -= take
                if groups[mid] and remaining > 0:
                    next_pending.append(mid)
            if len(next_pending) == len(pending):
                break
            pending = next_pending

        return selected

    def _encode_all(self, records: list[AudioMetadata]) -> None:
        """Load or compute fused vectors for every record via FusionCache."""
        total = len(records)
        print("Encoding recordings...")
        t_start = time.perf_counter()
        hits = 0
        misses = 0

        for idx, rec in enumerate(records, start=1):
            is_hit = self._cache.exists(rec)
            try:
                fused = self._cache.load_or_create(rec)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s — encoding failed: %s", rec.filename, exc)
                continue

            if is_hit:
                hits += 1
            else:
                misses += 1

            key = (rec.machine_type, rec.machine_id)
            self._fused_by_machine.setdefault(key, []).append(fused)
            self._all_fused.append(fused)

            if idx % 50 == 0:
                print(f"Processed {idx}/{total}")

        elapsed = time.perf_counter() - t_start
        encoded = len(self._all_fused)
        avg = elapsed / encoded if encoded else 0.0
        print(f"Cache hits               : {hits}")
        print(f"Cache misses             : {misses}")
        print(f"Total recordings encoded : {encoded}")
        print(f"Elapsed time             : {elapsed:.1f}s")
        print(f"Average time per recording: {avg:.3f}s")

        logger.info(
            "Loaded/encoded %d recordings across %d machines.",
            encoded,
            len(self._fused_by_machine),
        )

    # ------------------------------------------------------------------
    # Pair construction (recording-level split)
    # ------------------------------------------------------------------

    def _build_split_pairs(self) -> None:
        """Split recordings per machine, then build positive pairs within each split.

        For each (machine_type, machine_id) key:
          1. Shuffle the recordings with the shared RNG.
          2. Reserve the last ``val_split`` fraction as validation recordings.
          3. Build positive pairs only within the training recordings.
          4. Build positive pairs only within the validation recordings.

        This guarantees that no recording appears in both train and val pairs.
        """
        for key, fused_list in self._fused_by_machine.items():
            if len(fused_list) < 2:
                logger.warning(
                    "Machine %s has only %d recording(s) — skipping pair generation.",
                    key,
                    len(fused_list),
                )
                continue

            shuffled = list(fused_list)
            self._rng.shuffle(shuffled)

            n_val = max(1, int(len(shuffled) * self._val_split))
            val_recordings = shuffled[:n_val]
            train_recordings = shuffled[n_val:]

            self._train_positive_pairs.extend(
                self._pairs_from_recordings(train_recordings)
            )
            self._val_positive_pairs.extend(
                self._pairs_from_recordings(val_recordings)
            )

        logger.info(
            "Train positive pairs: %d  |  Val positive pairs: %d",
            len(self._train_positive_pairs),
            len(self._val_positive_pairs),
        )

    def _pairs_from_recordings(
        self,
        recordings: list[FusedFeatureVector],
    ) -> list[ContrastivePair]:
        """Build one positive pair per anchor from a list of recordings.

        Uses ``self._rng`` (seeded at construction) so that the same seed
        always produces the same pair assignments.

        Args:
            recordings: Fused vectors all belonging to the same machine.

        Returns:
            One ContrastivePair per recording.
        """
        if len(recordings) < 2:
            return []
        pairs: list[ContrastivePair] = []
        for anchor in recordings:
            pool = [f for f in recordings if f.filename != anchor.filename]
            paired = self._rng.choice(pool)
            pairs.append(ContrastivePair(anchor=anchor, paired=paired, label=1))
        return pairs
