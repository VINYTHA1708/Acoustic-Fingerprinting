"""Experiment E1 — Ablation Study Definitions.

Defines the full proposed method and four ablation configurations for the
contrastive acoustic fingerprinting system.  Each ablation removes or
modifies exactly one component while keeping the dataset, split protocol,
machine IDs, random seed, and evaluation metrics identical.

Full Method (FM)
----------------
The complete proposed system as evaluated in Phase 7.3:
    Feature    : 921-dim Fusion Vector (DSP 153 + BEATs 768).
    Head       : ProjectionHead trained with NT-Xent contrastive loss.
    Profile    : Per-machine mean of 256-dim L2-normalised embeddings.
    Scoring    : Euclidean distance to profile mean.

Ablations
---------
A1 — No BEATs (DSP-only fusion)
    Remove the BEATs encoder entirely.  The fusion vector is the 153-dim DSP
    vector alone.  A new ProjectionHead (153→256) is trained with NT-Xent.
    Isolates the contribution of the deep audio representation.

A2 — No DSP (BEATs-only fusion)
    Remove all DSP features.  The fusion vector is the 768-dim BEATs embedding
    alone.  A new ProjectionHead (768→256) is trained with NT-Xent.
    Isolates the contribution of the hand-crafted DSP features.

A3 — No Contrastive Training (random ProjectionHead)
    Keep the full 921-dim Fusion Vector but replace the trained ProjectionHead
    with a randomly initialised one (no checkpoint loaded).  Profile and
    scoring are otherwise identical.
    Isolates the contribution of contrastive training vs. random projection.

A4 — No ProjectionHead (raw Fusion Vector scoring)
    Skip the ProjectionHead entirely.  Score recordings by Euclidean distance
    between the raw 921-dim Fusion Vector and the per-machine mean Fusion
    Vector computed from profile_normal recordings.
    Isolates the contribution of the learned dimensionality reduction.

Evaluation Protocol (shared across full method and all ablations)
-----------------------------------------------------------------
- Dataset      : MIMII pump, machine IDs id_00 / id_02 / id_04 / id_06.
- Split        : DatasetSplitter(train_ratio=0.70, profile_ratio=0.15, seed=42).
- Profile set  : split.profile_normal recordings (per machine ID).
- Test set     : split.test_normal ∪ split.test_abnormal (per machine ID).
- Anomaly score: Euclidean distance from the test embedding/vector to the
                 per-machine profile mean.
- Metrics      : AUROC, separation ratio (mean_abnormal / mean_normal distance).
- Threshold    : not fixed; AUROC is threshold-free.

Usage
-----
    python experiments/e1_ablation_definition.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AblationID = Literal[
    "FM_full_method",
    "A1_no_beats",
    "A2_no_dsp",
    "A3_no_contrastive",
    "A4_no_projection",
]

# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationDefinition:
    """Immutable descriptor for one ablation (or the full method) configuration.

    Attributes:
        ablation_id: Short unique identifier.
        name: Human-readable name.
        fusion_dim: Dimensionality of the vector fed into the scoring step.
        feature_source: Which components supply the input features.
        projection_trained: Whether a contrastively trained ProjectionHead is used.
        scoring_dim: Dimensionality of the vector used for distance scoring.
        component_removed: The component that is ablated (empty string for FM).
        description: Free-text rationale for this configuration.
    """

    ablation_id: AblationID
    name: str
    fusion_dim: int
    feature_source: str
    projection_trained: bool
    scoring_dim: int
    component_removed: str
    description: str


@dataclass(frozen=True)
class AblationProtocol:
    """Shared evaluation protocol applied to all ablation configurations.

    Attributes:
        machine_type: MIMII machine type used in the experiment.
        machine_ids: Machine IDs included in the evaluation.
        train_ratio: Fraction of normal recordings used for training.
        profile_ratio: Fraction of normal recordings used to build the profile.
        seed: Random seed for the DatasetSplitter.
        metrics: Evaluation metrics reported for each configuration.
    """

    machine_type: str
    machine_ids: tuple[str, ...]
    train_ratio: float
    profile_ratio: float
    seed: int
    metrics: tuple[str, ...]


# ---------------------------------------------------------------------------
# Registered configurations
# ---------------------------------------------------------------------------

ABLATIONS: dict[AblationID, AblationDefinition] = {
    "FM_full_method": AblationDefinition(
        ablation_id="FM_full_method",
        name="Full Method (DSP + BEATs + Contrastive ProjectionHead)",
        fusion_dim=921,
        feature_source="FeatureExtractor (153-dim DSP) + BEATsEncoder (768-dim)",
        projection_trained=True,
        scoring_dim=256,
        component_removed="",
        description=(
            "Complete proposed system. The 921-dim Fusion Vector (DSP 153 + BEATs 768) "
            "is projected to a 256-dim L2-normalised embedding by a ProjectionHead "
            "trained with NT-Xent contrastive loss. Anomaly score is Euclidean distance "
            "to the per-machine profile mean embedding."
        ),
    ),
    "A1_no_beats": AblationDefinition(
        ablation_id="A1_no_beats",
        name="Ablation A1 — No BEATs (DSP-only)",
        fusion_dim=153,
        feature_source="FeatureExtractor (153-dim DSP only; BEATsEncoder removed)",
        projection_trained=True,
        scoring_dim=256,
        component_removed="BEATsEncoder",
        description=(
            "Removes the BEATs encoder. The fusion vector is the 153-dim DSP vector "
            "alone. A ProjectionHead with input_dim=153 is trained from scratch with "
            "NT-Xent loss. Isolates the contribution of the deep audio representation."
        ),
    ),
    "A2_no_dsp": AblationDefinition(
        ablation_id="A2_no_dsp",
        name="Ablation A2 — No DSP (BEATs-only)",
        fusion_dim=768,
        feature_source="BEATsEncoder (768-dim only; FeatureExtractor removed)",
        projection_trained=True,
        scoring_dim=256,
        component_removed="FeatureExtractor (DSP)",
        description=(
            "Removes all DSP features. The fusion vector is the 768-dim BEATs embedding "
            "alone. A ProjectionHead with input_dim=768 is trained from scratch with "
            "NT-Xent loss. Isolates the contribution of the hand-crafted DSP features."
        ),
    ),
    "A3_no_contrastive": AblationDefinition(
        ablation_id="A3_no_contrastive",
        name="Ablation A3 — No Contrastive Training (random ProjectionHead)",
        fusion_dim=921,
        feature_source="FeatureExtractor (153-dim DSP) + BEATsEncoder (768-dim)",
        projection_trained=False,
        scoring_dim=256,
        component_removed="NT-Xent contrastive training",
        description=(
            "Keeps the full 921-dim Fusion Vector but replaces the trained "
            "ProjectionHead with a randomly initialised one (no checkpoint loaded, "
            "no contrastive training). Profile and scoring are otherwise identical. "
            "Isolates the contribution of contrastive training vs. random projection."
        ),
    ),
    "A4_no_projection": AblationDefinition(
        ablation_id="A4_no_projection",
        name="Ablation A4 — No ProjectionHead (raw Fusion Vector scoring)",
        fusion_dim=921,
        feature_source="FeatureExtractor (153-dim DSP) + BEATsEncoder (768-dim)",
        projection_trained=False,
        scoring_dim=921,
        component_removed="ProjectionHead",
        description=(
            "Skips the ProjectionHead entirely. Anomaly score is the Euclidean "
            "distance between the raw 921-dim Fusion Vector and the per-machine "
            "mean Fusion Vector computed from profile_normal recordings. "
            "Isolates the contribution of the learned dimensionality reduction."
        ),
    ),
}

PROTOCOL = AblationProtocol(
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


def get_ablation(ablation_id: AblationID) -> AblationDefinition:
    """Return the AblationDefinition for *ablation_id*.

    Raises:
        KeyError: If *ablation_id* is not registered.
    """
    if ablation_id not in ABLATIONS:
        raise KeyError(f"Unknown ablation_id '{ablation_id}'. Valid: {list(ABLATIONS)}")
    return ABLATIONS[ablation_id]


def get_all_ablations() -> list[AblationDefinition]:
    """Return all registered AblationDefinition objects in insertion order."""
    return list(ABLATIONS.values())


def get_protocol() -> AblationProtocol:
    """Return the shared AblationProtocol."""
    return PROTOCOL


# ---------------------------------------------------------------------------
# CLI summary
# ---------------------------------------------------------------------------


def _print_summary() -> None:
    proto = get_protocol()
    print("=" * 65)
    print("Experiment E1 — Ablation Study Definitions")
    print("=" * 65)
    print(f"Machine type : {proto.machine_type}")
    print(f"Machine IDs  : {proto.machine_ids}")
    print(f"Split        : train={proto.train_ratio}  profile={proto.profile_ratio}  seed={proto.seed}")
    print(f"Metrics      : {proto.metrics}")
    print("-" * 65)
    for a in get_all_ablations():
        tag = "(full method)" if not a.component_removed else f"(removes: {a.component_removed})"
        print(f"\n[{a.ablation_id}]  {a.name}  {tag}")
        print(f"  Fusion dim       : {a.fusion_dim}")
        print(f"  Feature source   : {a.feature_source}")
        print(f"  Projection trained: {a.projection_trained}")
        print(f"  Scoring dim      : {a.scoring_dim}")
        print(f"  Description      : {a.description}")
    print("=" * 65)


if __name__ == "__main__":
    _print_summary()
