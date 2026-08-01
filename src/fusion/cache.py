"""FusionCache — disk cache for pre-computed FusedFeatureVector objects.

Avoids rerunning the full Preprocessing → DSP → BEATs → Fusion pipeline on
every ContrastiveDataset construction by persisting each recording's fused
vector as a compressed NPZ file.

Cache layout on disk::

    <cache_root>/
        <machine_type>/
            <machine_id>/
                <label>/
                    <filename>.npz
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..beats.encoder import BEATsEncoder
from ..dataset.metadata import AudioMetadata
from ..feature_extraction.extractor import FeatureExtractor
from ..feature_extraction.feature_vector import FeatureVectorBuilder
from ..fusion.fused_vector import FusedFeatureVector
from ..fusion.fusion import FusionBuilder
from ..fusion.serializer import FusedVectorSerializer
from ..preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)


class FusionCache:
    """Per-recording disk cache for :class:`~fusion.fused_vector.FusedFeatureVector`.

    Args:
        cache_root: Directory under which cached NPZ files are stored.
        pipeline: Shared :class:`~preprocessing.pipeline.PreprocessingPipeline`.
        extractor: Shared :class:`~feature_extraction.extractor.FeatureExtractor`.
        vec_builder: Shared :class:`~feature_extraction.feature_vector.FeatureVectorBuilder`.
        encoder: Shared :class:`~beats.encoder.BEATsEncoder`.
        fusion: Shared :class:`~fusion.fusion.FusionBuilder`.
    """

    def __init__(
        self,
        cache_root: str | Path,
        pipeline: PreprocessingPipeline,
        extractor: FeatureExtractor,
        vec_builder: FeatureVectorBuilder,
        encoder: BEATsEncoder,
        fusion: FusionBuilder,
    ) -> None:
        self._root = Path(cache_root)
        self._pipeline = pipeline
        self._extractor = extractor
        self._vec_builder = vec_builder
        self._encoder = encoder
        self._fusion = fusion
        self._serializer = FusedVectorSerializer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def exists(self, rec: AudioMetadata) -> bool:
        """Return True if a cached NPZ file exists for this recording.

        Args:
            rec: Audio metadata identifying the recording.
        """
        return self._cache_path(rec).exists()

    def save(self, fused: FusedFeatureVector, rec: AudioMetadata) -> None:
        """Persist a fused vector to disk.

        Args:
            fused: The fused vector to cache.
            rec: Audio metadata used to derive the cache file path.
        """
        path = self._cache_path(rec)
        self._serializer.save_npz(fused, path)
        logger.debug("Cache saved: %s", path)

    def load(self, rec: AudioMetadata) -> FusedFeatureVector:
        """Load a cached fused vector from disk.

        Args:
            rec: Audio metadata identifying the recording.

        Returns:
            The cached :class:`~fusion.fused_vector.FusedFeatureVector`.

        Raises:
            FileNotFoundError: If no cache file exists for this recording.
        """
        path = self._cache_path(rec)
        fused = self._serializer.load_npz(path)
        logger.debug("Cache loaded: %s", path)
        return fused

    def load_or_create(self, rec: AudioMetadata, *, verbose: bool = False) -> FusedFeatureVector:
        """Return the cached fused vector, computing and saving it if absent.

        Args:
            rec: Audio metadata for the recording to encode.
            verbose: If True, print cache hit/miss status to stdout.

        Returns:
            :class:`~fusion.fused_vector.FusedFeatureVector` for the recording.
        """
        if self.exists(rec):
            if verbose:
                print("Cache hit")
                print("Loaded cached fusion vector.")
            return self.load(rec)

        if verbose:
            print("Cache miss")
            print("Generating fusion vector...")

        fused = self._compute(rec)
        self.save(fused, rec)
        return fused

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cache_path(self, rec: AudioMetadata) -> Path:
        """Derive the NPZ cache path for a recording."""
        stem = Path(rec.filename).stem
        return self._root / rec.machine_type / rec.machine_id / rec.label / f"{stem}.npz"

    def _compute(self, rec: AudioMetadata) -> FusedFeatureVector:
        """Run the full pipeline and return a FusedFeatureVector."""
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
