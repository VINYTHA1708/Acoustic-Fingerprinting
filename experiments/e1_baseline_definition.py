"""Experiment E1 — Baseline Method Definitions.

Defines three baseline methods for comparison against the proposed contrastive
acoustic fingerprinting system. No baselines are implemented or run here;
this module documents the selected methods and their evaluation setup.

Baselines
---------
B1 — Raw MFCC Distance
    Feature : 153-dim DSP vector (MFCC mean+std, spectral centroid, spectral
              rolloff, RMS energy, harmonic salience) extracted by the project's
              existing FeatureExtractor + FeatureVectorBuilder.
    Profile : Per-machine mean vector computed over profile_normal recordings.
    Scoring : Euclidean distance between a test recording's DSP vector and the
              machine's mean profile vector.  Higher distance → more anomalous.
    Rationale: Establishes the ceiling for classical hand-crafted features
               without any learned representation.

B2 — Statistical Audio Feature Distance
    Feature : 3-dim vector [rms_mean, zcr_mean, spectral_centroid_mean] drawn
              directly from the FeatureExtractor output dict — no additional
              computation required.
    Profile : Per-machine mean vector over profile_normal recordings.
    Scoring : Euclidean distance between a test recording's 3-dim vector and
              the machine's mean profile vector.
    Rationale: Minimal-feature baseline; tests whether a handful of simple
               statistics are sufficient for anomaly detection.

B3 — Non-Contrastive Embedding Distance
    Feature : 256-dim ProjectionHead output produced by a randomly initialised
              (untrained) ProjectionHead applied to the same 921-dim Fusion
              Vector used by the proposed method.
    Profile : Per-machine mean vector over profile_normal embeddings.
    Scoring : Euclidean distance between a test recording's embedding and the
              machine's mean profile vector.
    Rationale: Isolates the contribution of contrastive training by holding the
               architecture constant and removing the learned weights.

Evaluation Protocol (shared across all baselines and the proposed method)
--------------------------------------------------------------------------
- Dataset      : MIMII pump, machine IDs id_00 / id_02 / id_04 / id_06.
- Split        : DatasetSplitter(train_ratio=0.70, profile_ratio=0.15, seed=42).
- Profile set  : split.profile_normal recordings (per machine ID).
- Test set     : split.test_normal ∪ split.test_abnormal (per machine ID).
- Anomaly score: distance from the test embedding/vector to the machine profile mean.
- Metrics      : AUROC, separation ratio (mean_abnormal / mean_normal distance).
- Threshold    : not fixed; AUROC is threshold-free.

Usage
-----
    python experiments/e1_baseline_definition.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Baseline descriptor
# ---------------------------------------------------------------------------

BaselineID = Literal["B1_mfcc_distance", "B2_stat_distance", "B3_random_projection"]


@dataclass(frozen=True)
class BaselineDefinition:
    """Immutable descriptor for one baseline method.

    Attributes:
        baseline_id: Short unique identifier.
        name: Human-readable name.
        feature_dim: Dimensionality of the feature vector used.
        feature_source: Which project component supplies the features.
        profile_strategy: How the per-machine profile is built.
        scoring_metric: Distance / similarity metric used for anomaly scoring.
        description: Free-text description of the method.
    """

    baseline_id: BaselineID
    name: str
    feature_dim: int
    feature_source: str
    profile_strategy: str
    scoring_metric: str
    description: str


# ---------------------------------------------------------------------------
# Evaluation protocol descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationProtocol:
    """Shared evaluation protocol applied to all baselines and the proposed method.

    Attributes:
        machine_type: MIMII machine type used in the experiment.
        machine_ids: Machine IDs included in the evaluation.
        train_ratio: Fraction of normal recordings used for training.
        profile_ratio: Fraction of normal recordings used to build the profile.
        seed: Random seed for the DatasetSplitter.
        metrics: Evaluation metrics reported for each method.
    """

    machine_type: str
    machine_ids: tuple[str, ...]
    train_ratio: float
    profile_ratio: float
    seed: int
    metrics: tuple[str, ...]


# ---------------------------------------------------------------------------
# Registered baselines
# ---------------------------------------------------------------------------

BASELINES: dict[BaselineID, BaselineDefinition] = {
    "B1_mfcc_distance": BaselineDefinition(
        baseline_id="B1_mfcc_distance",
        name="Raw MFCC Distance",
        feature_dim=153,
        feature_source="FeatureExtractor + FeatureVectorBuilder (DSP vector)",
        profile_strategy="Mean of profile_normal DSP vectors per machine ID",
        scoring_metric="Euclidean distance to profile mean",
        description=(
            "Uses the full 153-dim DSP feature vector produced by the existing "
            "FeatureExtractor. A per-machine mean profile vector is computed from "
            "profile_normal recordings. Anomaly score is the Euclidean distance "
            "between the test DSP vector and the profile mean."
        ),
    ),
    "B2_stat_distance": BaselineDefinition(
        baseline_id="B2_stat_distance",
        name="Statistical Audio Feature Distance",
        feature_dim=3,
        feature_source="FeatureExtractor output dict keys: rms_mean, zcr_mean, spectral_centroid_mean",
        profile_strategy="Mean of profile_normal 3-dim stat vectors per machine ID",
        scoring_metric="Euclidean distance to profile mean",
        description=(
            "Uses only three scalar statistics — RMS energy mean, zero-crossing "
            "rate mean, and spectral centroid mean — drawn directly from the "
            "FeatureExtractor output dict. Tests whether minimal hand-crafted "
            "features are sufficient for anomaly detection."
        ),
    ),
    "B3_random_projection": BaselineDefinition(
        baseline_id="B3_random_projection",
        name="Non-Contrastive (Random) Projection Embedding",
        feature_dim=256,
        feature_source="ProjectionHead (randomly initialised, no checkpoint loaded) on 921-dim FusedFeatureVector",
        profile_strategy="Mean of profile_normal 256-dim random embeddings per machine ID",
        scoring_metric="Euclidean distance to profile mean",
        description=(
            "Applies the same ProjectionHead architecture used by the proposed "
            "method, but with randomly initialised weights (no contrastive "
            "training). Isolates the contribution of contrastive training by "
            "holding the architecture and fusion pipeline constant."
        ),
    ),
}

PROTOCOL = EvaluationProtocol(
    machine_type="pump",
    machine_ids=("id_00", "id_02", "id_04", "id_06"),
    train_ratio=0.70,
    profile_ratio=0.15,
    seed=42,
    metrics=("auroc", "separation_ratio"),
)

# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_baseline(baseline_id: BaselineID) -> BaselineDefinition:
    """Return the BaselineDefinition for *baseline_id*.

    Raises:
        KeyError: If *baseline_id* is not registered.
    """
    if baseline_id not in BASELINES:
        raise KeyError(f"Unknown baseline_id '{baseline_id}'. Valid: {list(BASELINES)}")
    return BASELINES[baseline_id]


def get_all_baselines() -> list[BaselineDefinition]:
    """Return all registered BaselineDefinition objects in insertion order."""
    return list(BASELINES.values())


def get_protocol() -> EvaluationProtocol:
    """Return the shared EvaluationProtocol."""
    return PROTOCOL


# ---------------------------------------------------------------------------
# CLI summary
# ---------------------------------------------------------------------------

def _print_summary() -> None:
    proto = get_protocol()
    print("=" * 60)
    print("Experiment E1 — Baseline Method Definitions")
    print("=" * 60)
    print(f"Machine type : {proto.machine_type}")
    print(f"Machine IDs  : {proto.machine_ids}")
    print(f"Split        : train={proto.train_ratio}  profile={proto.profile_ratio}  seed={proto.seed}")
    print(f"Metrics      : {proto.metrics}")
    print("-" * 60)
    for b in get_all_baselines():
        print(f"\n[{b.baseline_id}]  {b.name}")
        print(f"  Feature dim    : {b.feature_dim}")
        print(f"  Feature source : {b.feature_source}")
        print(f"  Profile        : {b.profile_strategy}")
        print(f"  Scoring        : {b.scoring_metric}")
        print(f"  Description    : {b.description}")
    print("=" * 60)


if __name__ == "__main__":
    _print_summary()
