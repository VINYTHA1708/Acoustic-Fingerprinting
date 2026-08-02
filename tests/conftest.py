"""Shared pytest fixtures for the acoustic fingerprinting test suite.

All heavy resources (BEATs encoder, ProjectionHead checkpoint, FusionCache,
LearnedFingerprintProfile) are constructed once per session and reused.
Cached fusion vectors on disk are used wherever possible so no audio
processing is repeated during the test run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the project root importable from tests/
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BEATS_CHECKPOINT = _ROOT / "models" / "beats" / "BEATs_iter3_plus_AS2M.pt"
CONTRASTIVE_CHECKPOINT = _ROOT / "models" / "contrastive" / "best_projection_head.pt"
CACHE_ROOT = _ROOT / "data" / "fusion_cache"
DATASET_ROOT = _ROOT / "data" / "raw" / "MIMII"

MACHINE_TYPE = "pump"
MACHINE_ID = "id_00"

# ---------------------------------------------------------------------------
# Lazy imports (deferred so collection is fast even if torch is slow)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def beats_encoder():
    # Direct submodule import — avoids beats/__init__.py eager chain
    import importlib
    mod = importlib.import_module("src.beats.encoder")
    return mod.BEATsEncoder(BEATS_CHECKPOINT)


@pytest.fixture(scope="session")
def fusion_cache(beats_encoder):
    import importlib
    FusionCache = importlib.import_module("src.fusion.cache").FusionCache
    FeatureExtractor = importlib.import_module("src.feature_extraction.extractor").FeatureExtractor
    FeatureVectorBuilder = importlib.import_module("src.feature_extraction.feature_vector").FeatureVectorBuilder
    FusionBuilder = importlib.import_module("src.fusion.fusion").FusionBuilder
    PreprocessingPipeline = importlib.import_module("src.preprocessing.pipeline").PreprocessingPipeline

    return FusionCache(
        cache_root=CACHE_ROOT,
        pipeline=PreprocessingPipeline(target_sr=16_000),
        extractor=FeatureExtractor(sample_rate=16_000),
        vec_builder=FeatureVectorBuilder(),
        encoder=beats_encoder,
        fusion=FusionBuilder(),
    )


@pytest.fixture(scope="session")
def dataset_loader():
    import importlib
    DatasetLoader = importlib.import_module("src.dataset.loader").DatasetLoader
    return DatasetLoader(DATASET_ROOT)


@pytest.fixture(scope="session")
def normal_records(dataset_loader):
    return [
        r for r in dataset_loader.get_all_files()
        if r.machine_type == MACHINE_TYPE
        and r.machine_id == MACHINE_ID
        and r.label == "normal"
    ]


@pytest.fixture(scope="session")
def first_normal_record(normal_records):
    return normal_records[0]


@pytest.fixture(scope="session")
def cached_fused_vector(fusion_cache, first_normal_record):
    """Load the fused vector from disk cache (never recomputes)."""
    return fusion_cache.load_or_create(first_normal_record)


@pytest.fixture(scope="session")
def learned_profile(dataset_loader):
    import importlib
    LearnedProfileBuilder = importlib.import_module("src.learned_profile.builder").LearnedProfileBuilder

    builder = LearnedProfileBuilder(checkpoint_path=CONTRASTIVE_CHECKPOINT)
    return builder.build(
        loader=dataset_loader,
        machine_type=MACHINE_TYPE,
        machine_id=MACHINE_ID,
        max_recordings=10,
    )


@pytest.fixture(scope="session")
def contrastive_inference():
    import importlib
    ContrastiveInference = importlib.import_module("src.contrastive_learning.inference").ContrastiveInference
    ProjectionHead = importlib.import_module("src.contrastive_learning.model").ProjectionHead

    head = ProjectionHead()
    return ContrastiveInference(
        projection_head=head,
        checkpoint_path=CONTRASTIVE_CHECKPOINT,
    )


@pytest.fixture(scope="session")
def sample_embedding(contrastive_inference, cached_fused_vector):
    """A single 256-dim learned embedding produced from the cached fused vector."""
    return contrastive_inference.generate_fingerprint(cached_fused_vector)


@pytest.fixture(scope="session")
def drift_result(first_normal_record, learned_profile):
    import importlib
    LearnedDriftAnalyzer = importlib.import_module("src.learned_drift.analyzer").LearnedDriftAnalyzer

    analyzer = LearnedDriftAnalyzer(checkpoint_path=CONTRASTIVE_CHECKPOINT)
    return analyzer.analyze(first_normal_record, learned_profile)


@pytest.fixture(scope="session")
def health_result(first_normal_record, learned_profile):
    import importlib
    LearnedHealthAnalyzer = importlib.import_module("src.learned_health_index.analyzer").LearnedHealthAnalyzer

    analyzer = LearnedHealthAnalyzer(checkpoint_path=CONTRASTIVE_CHECKPOINT)
    return analyzer.analyze(first_normal_record, learned_profile)
