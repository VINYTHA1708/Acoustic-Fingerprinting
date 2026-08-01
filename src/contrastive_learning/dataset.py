"""ContrastiveDataset — builds positive and negative pairs for contrastive learning.

SDD v4 §2 (Version 3):
    Positive pairs: same machine, different recordings.
    Negative pairs: different machine_id or different machine_type.
    Only normal recordings are used.

Each item returned is:
    (anchor_fused_vector, paired_fused_vector, label)
    label = 1  →  positive pair (same machine)
    label = 0  →  negative pair (different machine)
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
        label: 1 for positive (same machine), 0 for negative (different machine).
    """

    anchor: FusedFeatureVector
    paired: FusedFeatureVector
    label: int  # 1 = positive, 0 = negative


class ContrastiveDataset:
    """Builds contrastive pairs from normal recordings in a MIMII-style dataset.

    Encodes every normal recording once (PreprocessingPipeline → DSP → BEATs →
    FusionBuilder), then enumerates positive and negative pairs from the cached
    fused vectors.  Pairs are balanced: ``len(positive_pairs) == len(negative_pairs)``
    where possible.

    Args:
        dataset_root: Path to the dataset root (passed to :class:`DatasetLoader`).
        checkpoint_path: Path to the BEATs checkpoint. Defaults to the project
                         standard location ``models/beats/BEATs_iter3_plus_AS2M.pt``.
        seed: Random seed used for negative-pair sampling and shuffling.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        checkpoint_path: str | Path | None = None,
        cache_root: str | Path | None = None,
        seed: int = 42,
        machine_type: str | None = None,
        machine_id: str | None = None,
        max_recordings: int | None = None,
    ) -> None:
        self._rng = random.Random(seed)
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

        loader = DatasetLoader(dataset_root)
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

        self._positive_pairs: list[ContrastivePair] = self._build_positive_pairs()
        self._negative_pairs: list[ContrastivePair] = self._build_negative_pairs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def positive_pairs(self) -> list[ContrastivePair]:
        """All positive pairs (same machine, different recordings)."""
        return self._positive_pairs

    @property
    def negative_pairs(self) -> list[ContrastivePair]:
        """All negative pairs (different machine_id or machine_type)."""
        return self._negative_pairs

    @property
    def all_pairs(self) -> list[ContrastivePair]:
        """Balanced interleaving of positive and negative pairs."""
        n = min(len(self._positive_pairs), len(self._negative_pairs))
        pairs = self._positive_pairs[:n] + self._negative_pairs[:n]
        self._rng.shuffle(pairs)
        return pairs

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
        return iter(self.all_pairs)

    def __len__(self) -> int:
        return len(self.all_pairs)

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

        # Group by machine_id preserving original order within each group
        groups: dict[str, list[AudioMetadata]] = {}
        for r in records:
            groups.setdefault(r.machine_id, []).append(r)

        ids = sorted(groups.keys())
        n_ids = len(ids)
        if n_ids == 0:
            return []

        selected: list[AudioMetadata] = []
        remaining = max_recordings

        # Iteratively allocate quota; IDs with fewer recordings give back surplus
        pending = list(ids)
        while remaining > 0 and pending:
            per_id = max(1, remaining // len(pending))
            next_pending = []
            for mid in pending:
                take = min(per_id, len(groups[mid]))
                selected.extend(groups[mid][:take])
                groups[mid] = groups[mid][take:]  # consume taken entries
                remaining -= take
                if groups[mid] and remaining > 0:
                    next_pending.append(mid)
            # If no progress was made, stop to avoid infinite loop
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
    # Pair construction
    # ------------------------------------------------------------------

    def _build_positive_pairs(self) -> list[ContrastivePair]:
        """One positive pair per anchor: same machine, one randomly sampled partner."""
        pairs: list[ContrastivePair] = []

        for fused_list in self._fused_by_machine.values():
            if len(fused_list) < 2:
                continue
            for anchor in fused_list:
                pool = [f for f in fused_list if f.filename != anchor.filename]
                positive = self._rng.choice(pool)
                pairs.append(ContrastivePair(anchor=anchor, paired=positive, label=1))

        logger.info("Positive pairs built: %d", len(pairs))
        return pairs

    def _build_negative_pairs(self) -> list[ContrastivePair]:
        """Three negative pairs per anchor: different machine_id or machine_type."""
        pairs: list[ContrastivePair] = []

        for key, fused_list in self._fused_by_machine.items():
            negatives_pool = [
                f for k, fv in self._fused_by_machine.items() if k != key for f in fv
            ]
            if not negatives_pool:
                continue
            for anchor in fused_list:
                k = min(3, len(negatives_pool))
                for negative in self._rng.sample(negatives_pool, k):
                    pairs.append(ContrastivePair(anchor=anchor, paired=negative, label=0))

        logger.info("Negative pairs built: %d", len(pairs))
        return pairs
