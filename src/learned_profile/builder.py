"""LearnedProfileBuilder — builds a LearnedFingerprintProfile from a trained checkpoint.

SDD v4 §10 (Version 3):
    Pipeline per recording:
        Audio → Preprocessing → DSP → BEATs → Fusion → ProjectionHead → 256-dim embedding

    Only normal recordings are used.
    All embeddings for one machine are collected, then mean and std are computed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from ..beats.encoder import BEATsEncoder
from ..contrastive_learning.inference import ContrastiveInference
from ..contrastive_learning.model import ProjectionHead
from ..dataset.loader import DatasetLoader
from ..dataset.metadata import AudioMetadata
from ..feature_extraction.extractor import FeatureExtractor
from ..feature_extraction.feature_vector import FeatureVectorBuilder
from ..fusion.cache import FusionCache
from ..fusion.fusion import FusionBuilder
from ..preprocessing.pipeline import PreprocessingPipeline
from .learned_profile import LearnedFingerprintProfile

logger = logging.getLogger(__name__)

_BEATS_CHECKPOINT_REL = (
    Path(__file__).resolve().parents[2] / "models" / "beats" / "BEATs_iter3_plus_AS2M.pt"
)
_CACHE_ROOT_REL = Path(__file__).resolve().parents[2] / "data" / "fusion_cache"

_NORMAL_LABEL = "normal"
_EMBEDDING_DIM = 256


class LearnedProfileBuilder:
    """Builds a :class:`LearnedFingerprintProfile` for one machine.

    Runs every normal recording through the full pipeline:
    Preprocessing → DSP → BEATs → Fusion → ProjectionHead → 256-dim embedding.

    Args:
        checkpoint_path: Path to the trained ProjectionHead ``.pt`` checkpoint.
        beats_checkpoint: Path to the BEATs model checkpoint.
                          Defaults to the project standard location.
        cache_root: Directory for the FusionCache.
                    Defaults to ``data/fusion_cache``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        beats_checkpoint: str | Path | None = None,
        cache_root: str | Path | None = None,
    ) -> None:
        beats_ckpt = Path(beats_checkpoint) if beats_checkpoint else _BEATS_CHECKPOINT_REL
        _cache = Path(cache_root) if cache_root else _CACHE_ROOT_REL

        pipeline = PreprocessingPipeline(target_sr=16_000)
        extractor = FeatureExtractor(sample_rate=16_000)
        vec_builder = FeatureVectorBuilder()
        encoder = BEATsEncoder(beats_ckpt)
        fusion = FusionBuilder()

        self._cache = FusionCache(
            cache_root=_cache,
            pipeline=pipeline,
            extractor=extractor,
            vec_builder=vec_builder,
            encoder=encoder,
            fusion=fusion,
        )

        head = ProjectionHead()
        self._inference = ContrastiveInference(
            projection_head=head,
            checkpoint_path=checkpoint_path,
        )

    def build(
        self,
        machine_type: str,
        machine_id: str,
        loader: DatasetLoader | None = None,
        recordings: list[AudioMetadata] | None = None,
        max_recordings: int | None = None,
        exclude_filenames: set[str] | None = None,
    ) -> LearnedFingerprintProfile:
        """Build a healthy learned fingerprint profile for one machine.

        Exactly one of *loader* or *recordings* must be supplied.

        Args:
            machine_type: Machine type (e.g. ``"pump"``).
            machine_id: Machine ID (e.g. ``"id_00"``).
            loader: A :class:`~dataset.loader.DatasetLoader` pointing at the dataset root.
                    Mutually exclusive with *recordings*.
            recordings: Explicit list of :class:`~dataset.metadata.AudioMetadata` objects
                        to use as the healthy profile.  All items must be normal recordings
                        belonging to *machine_type* / *machine_id*.
                        Mutually exclusive with *loader*.
            max_recordings: If provided, limit the number of recordings processed.
            exclude_filenames: Filenames to exclude (loader path only).

        Returns:
            :class:`LearnedFingerprintProfile` with all embeddings and statistics.

        Raises:
            ValueError: If both or neither of *loader* / *recordings* are supplied,
                        or if validation of the supplied recordings fails.
        """
        if loader is not None and recordings is not None:
            raise ValueError(
                "Supply either 'loader' or 'recordings', not both."
            )
        if loader is None and recordings is None:
            raise ValueError(
                "One of 'loader' or 'recordings' must be supplied."
            )

        if recordings is not None:
            # --- explicit recordings path ---
            if not recordings:
                raise ValueError("'recordings' must not be empty.")
            for r in recordings:
                if not isinstance(r, AudioMetadata):
                    raise ValueError(
                        f"All items in 'recordings' must be AudioMetadata, got {type(r)}."
                    )
                if r.label != _NORMAL_LABEL:
                    raise ValueError(
                        f"Recording '{r.filename}' has label '{r.label}'; "
                        "only 'normal' recordings are allowed in the profile."
                    )
                if r.machine_type != machine_type:
                    raise ValueError(
                        f"Recording '{r.filename}' has machine_type '{r.machine_type}'; "
                        f"expected '{machine_type}'."
                    )
                if r.machine_id != machine_id:
                    raise ValueError(
                        f"Recording '{r.filename}' has machine_id '{r.machine_id}'; "
                        f"expected '{machine_id}'."
                    )
            records = list(recordings)
        else:
            # --- DatasetLoader path (original behaviour) ---
            records = [
                r for r in loader.get_all_files()  # type: ignore[union-attr]
                if r.machine_type == machine_type
                and r.machine_id == machine_id
                and r.label == _NORMAL_LABEL
                and (exclude_filenames is None or r.filename not in exclude_filenames)
            ]

            if not records:
                raise ValueError(
                    f"No normal recordings found for {machine_type}/{machine_id}."
                )

        if max_recordings is not None:
            records = records[:max_recordings]

        total = len(records)
        # ===== DEBUG START =====
        print(f"\n=== PROFILE BUILD DEBUG ===")
        print(f"  num_healthy_recordings_used : {total}")
        print(f"  exclude_filenames           : {exclude_filenames}")
        # ===== DEBUG END =====
        print(f"Healthy recordings : {total}")

        embeddings: list[np.ndarray] = []
        t_start = time.perf_counter()
        for idx, rec in enumerate(records, start=1):
            try:
                fused = self._cache.load_or_create(rec)
                emb = self._inference.generate_fingerprint(fused)
                embeddings.append(emb)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s — %s", rec.filename, exc)

            if idx % 50 == 0 or idx == total:
                print(f"Processed {idx}/{total}")

        elapsed = time.perf_counter() - t_start
        encoded = len(embeddings)
        avg = elapsed / encoded if encoded else 0.0
        print(f"Elapsed time             : {elapsed:.1f}s")
        print(f"Average time per recording: {avg:.3f}s")

        if not embeddings:
            raise ValueError(
                f"All recordings failed to encode for {machine_type}/{machine_id}."
            )

        matrix = np.stack(embeddings, axis=0).astype(np.float32)  # (N, 256)
        mean_vec = matrix.mean(axis=0)
        std_vec = matrix.std(axis=0)

        logger.info(
            "Learned profile built — %s/%s  recordings=%d  dim=%d",
            machine_type, machine_id, len(embeddings), _EMBEDDING_DIM,
        )

        return LearnedFingerprintProfile(
            machine_type=machine_type,
            machine_id=machine_id,
            embedding_dimension=_EMBEDDING_DIM,
            embeddings=matrix,
            mean_vector=mean_vec,
            std_vector=std_vec,
        )
