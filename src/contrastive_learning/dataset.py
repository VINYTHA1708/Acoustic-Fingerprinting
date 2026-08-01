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
        seed: int = 42,
    ) -> None:
        self._rng = random.Random(seed)
        checkpoint = Path(checkpoint_path) if checkpoint_path else _CHECKPOINT_REL

        self._pipeline = PreprocessingPipeline(target_sr=16_000)
        self._extractor = FeatureExtractor(sample_rate=16_000)
        self._vec_builder = FeatureVectorBuilder()
        self._encoder = BEATsEncoder(checkpoint)
        self._fusion = FusionBuilder()

        loader = DatasetLoader(dataset_root)
        normal_records = loader.filter_by_label("normal")
        logger.info("Normal recordings found: %d", len(normal_records))

        # Encode all normal recordings once and cache by (machine_type, machine_id)
        self._fused_by_machine: dict[tuple[str, str], list[FusedFeatureVector]] = {}
        self._all_fused: list[FusedFeatureVector] = []
        self._encode_all(normal_records)

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

    def _encode_all(self, records: list[AudioMetadata]) -> None:
        """Encode every record and populate the internal cache."""
        total = len(records)
        print(f"Encoding recordings...")
        t_start = time.perf_counter()

        for idx, rec in enumerate(records, start=1):
            try:
                fused = self._encode_one(rec)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s — encoding failed: %s", rec.filename, exc)
                continue

            key = (rec.machine_type, rec.machine_id)
            self._fused_by_machine.setdefault(key, []).append(fused)
            self._all_fused.append(fused)

            if idx % 50 == 0:
                print(f"Processed {idx}/{total}")

        elapsed = time.perf_counter() - t_start
        encoded = len(self._all_fused)
        avg = elapsed / encoded if encoded else 0.0
        print(f"Total recordings encoded : {encoded}")
        print(f"Elapsed time             : {elapsed:.1f}s")
        print(f"Average time per recording: {avg:.3f}s")

        logger.info(
            "Encoded %d recordings across %d machines.",
            encoded,
            len(self._fused_by_machine),
        )

    def _encode_one(self, rec: AudioMetadata) -> FusedFeatureVector:
        result = self._pipeline.run(rec.absolute_path)
        features = self._extractor.extract(result["waveform"])
        dsp_vector, dsp_names = self._vec_builder.build(features)
        embedding = self._encoder.encode(
            waveform=result["waveform"],
            sample_rate=result["sample_rate"],
            filename=rec.filename,
        )
        return self._fusion.build(
            dsp_vector=dsp_vector,
            dsp_feature_names=dsp_names,
            beats_embedding=embedding,
            machine_type=rec.machine_type,
            machine_id=rec.machine_id,
            label=rec.label,
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
