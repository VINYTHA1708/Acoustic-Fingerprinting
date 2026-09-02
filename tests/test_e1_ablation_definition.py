"""Tests for experiments/e1_ablation_definition.py.

Validates the ablation registry, protocol descriptor, and accessor functions.
No audio processing, BEATs, or dataset access required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from experiments.e1_ablation_definition import (
    ABLATIONS,
    PROTOCOL,
    AblationDefinition,
    AblationProtocol,
    get_ablation,
    get_all_ablations,
    get_protocol,
)

# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


def test_five_configurations_registered():
    assert len(ABLATIONS) == 5


def test_all_expected_ids_present():
    expected = {
        "FM_full_method",
        "A1_no_beats",
        "A2_no_dsp",
        "A3_no_contrastive",
        "A4_no_projection",
    }
    assert set(ABLATIONS.keys()) == expected


# ---------------------------------------------------------------------------
# AblationDefinition field contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ablation_id,expected_fusion_dim", [
    ("FM_full_method", 921),
    ("A1_no_beats", 153),
    ("A2_no_dsp", 768),
    ("A3_no_contrastive", 921),
    ("A4_no_projection", 921),
])
def test_fusion_dim(ablation_id, expected_fusion_dim):
    assert get_ablation(ablation_id).fusion_dim == expected_fusion_dim


@pytest.mark.parametrize("ablation_id,expected_scoring_dim", [
    ("FM_full_method", 256),
    ("A1_no_beats", 256),
    ("A2_no_dsp", 256),
    ("A3_no_contrastive", 256),
    ("A4_no_projection", 921),
])
def test_scoring_dim(ablation_id, expected_scoring_dim):
    assert get_ablation(ablation_id).scoring_dim == expected_scoring_dim


@pytest.mark.parametrize("ablation_id,expected_trained", [
    ("FM_full_method", True),
    ("A1_no_beats", True),
    ("A2_no_dsp", True),
    ("A3_no_contrastive", False),
    ("A4_no_projection", False),
])
def test_projection_trained_flag(ablation_id, expected_trained):
    assert get_ablation(ablation_id).projection_trained == expected_trained


@pytest.mark.parametrize("ablation_id", list({
    "FM_full_method", "A1_no_beats", "A2_no_dsp", "A3_no_contrastive", "A4_no_projection",
}))
def test_required_fields_non_empty(ablation_id):
    a = get_ablation(ablation_id)
    assert a.name
    assert a.feature_source
    assert a.description


def test_full_method_has_no_component_removed():
    assert get_ablation("FM_full_method").component_removed == ""


@pytest.mark.parametrize("ablation_id", ["A1_no_beats", "A2_no_dsp", "A3_no_contrastive", "A4_no_projection"])
def test_ablations_have_component_removed_set(ablation_id):
    assert get_ablation(ablation_id).component_removed != ""


def test_all_ablations_return_ablation_definition_instances():
    for a in get_all_ablations():
        assert isinstance(a, AblationDefinition)


# ---------------------------------------------------------------------------
# Component-removal semantics
# ---------------------------------------------------------------------------


def test_a1_no_beats_removes_beats_encoder():
    a = get_ablation("A1_no_beats")
    assert "BEATs" in a.component_removed or "beats" in a.component_removed.lower()


def test_a2_no_dsp_removes_dsp():
    a = get_ablation("A2_no_dsp")
    assert "DSP" in a.component_removed or "dsp" in a.component_removed.lower()


def test_a3_no_contrastive_removes_training():
    a = get_ablation("A3_no_contrastive")
    assert "contrastive" in a.component_removed.lower() or "NT-Xent" in a.component_removed


def test_a4_no_projection_removes_head():
    a = get_ablation("A4_no_projection")
    assert "ProjectionHead" in a.component_removed or "projection" in a.component_removed.lower()


def test_a4_scoring_dim_equals_fusion_dim():
    """A4 scores in raw fusion space — scoring_dim must equal fusion_dim."""
    a = get_ablation("A4_no_projection")
    assert a.scoring_dim == a.fusion_dim


# ---------------------------------------------------------------------------
# Accessor behaviour
# ---------------------------------------------------------------------------


def test_get_ablation_returns_correct_object():
    a = get_ablation("FM_full_method")
    assert a.ablation_id == "FM_full_method"


def test_get_ablation_raises_for_unknown_id():
    with pytest.raises(KeyError):
        get_ablation("A99_nonexistent")  # type: ignore[arg-type]


def test_get_all_ablations_length():
    assert len(get_all_ablations()) == 5


def test_get_all_ablations_order():
    ids = [a.ablation_id for a in get_all_ablations()]
    assert ids == [
        "FM_full_method",
        "A1_no_beats",
        "A2_no_dsp",
        "A3_no_contrastive",
        "A4_no_projection",
    ]


# ---------------------------------------------------------------------------
# AblationProtocol
# ---------------------------------------------------------------------------


def test_protocol_is_ablation_protocol_instance():
    assert isinstance(get_protocol(), AblationProtocol)


def test_protocol_machine_type():
    assert get_protocol().machine_type == "pump"


def test_protocol_machine_ids():
    assert set(get_protocol().machine_ids) == {"id_00", "id_02", "id_04", "id_06"}


def test_protocol_split_ratios_sum_to_one_or_less():
    p = get_protocol()
    assert p.train_ratio + p.profile_ratio <= 1.0


def test_protocol_seed():
    assert get_protocol().seed == 42


def test_protocol_metrics_non_empty():
    assert len(get_protocol().metrics) > 0


def test_protocol_contains_auroc_metric():
    assert "auroc" in get_protocol().metrics


def test_protocol_contains_separation_ratio_metric():
    assert "separation_ratio" in get_protocol().metrics


# ---------------------------------------------------------------------------
# Protocol alignment with baseline definition
# ---------------------------------------------------------------------------


def test_ablation_protocol_matches_baseline_protocol():
    """Ablation protocol must be identical to the baseline evaluation protocol."""
    from experiments.e1_baseline_definition import PROTOCOL as baseline_proto

    p = get_protocol()
    assert p.machine_type == baseline_proto.machine_type
    assert set(p.machine_ids) == set(baseline_proto.machine_ids)
    assert p.train_ratio == baseline_proto.train_ratio
    assert p.profile_ratio == baseline_proto.profile_ratio
    assert p.seed == baseline_proto.seed


# ---------------------------------------------------------------------------
# Immutability (frozen dataclasses)
# ---------------------------------------------------------------------------


def test_ablation_definition_is_frozen():
    a = get_ablation("FM_full_method")
    with pytest.raises((AttributeError, TypeError)):
        a.fusion_dim = 999  # type: ignore[misc]


def test_protocol_is_frozen():
    p = get_protocol()
    with pytest.raises((AttributeError, TypeError)):
        p.seed = 0  # type: ignore[misc]
