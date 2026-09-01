"""Tests for experiments/e1_baseline_definition.py.

Validates the baseline registry, protocol descriptor, and accessor functions.
No audio processing, BEATs, or dataset access required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from experiments.e1_baseline_definition import (
    BASELINES,
    PROTOCOL,
    BaselineDefinition,
    EvaluationProtocol,
    get_all_baselines,
    get_baseline,
    get_protocol,
)

# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

def test_three_baselines_registered():
    assert len(BASELINES) == 3


def test_all_expected_ids_present():
    expected = {"B1_mfcc_distance", "B2_stat_distance", "B3_random_projection"}
    assert set(BASELINES.keys()) == expected


# ---------------------------------------------------------------------------
# BaselineDefinition field contracts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("baseline_id,expected_dim", [
    ("B1_mfcc_distance", 153),
    ("B2_stat_distance", 3),
    ("B3_random_projection", 256),
])
def test_feature_dim(baseline_id, expected_dim):
    b = get_baseline(baseline_id)
    assert b.feature_dim == expected_dim


@pytest.mark.parametrize("baseline_id", ["B1_mfcc_distance", "B2_stat_distance", "B3_random_projection"])
def test_required_fields_non_empty(baseline_id):
    b = get_baseline(baseline_id)
    assert b.name
    assert b.feature_source
    assert b.profile_strategy
    assert b.scoring_metric
    assert b.description


def test_all_baselines_return_baseline_definition_instances():
    for b in get_all_baselines():
        assert isinstance(b, BaselineDefinition)


# ---------------------------------------------------------------------------
# Accessor behaviour
# ---------------------------------------------------------------------------

def test_get_baseline_returns_correct_object():
    b = get_baseline("B1_mfcc_distance")
    assert b.baseline_id == "B1_mfcc_distance"


def test_get_baseline_raises_for_unknown_id():
    with pytest.raises(KeyError):
        get_baseline("B99_nonexistent")  # type: ignore[arg-type]


def test_get_all_baselines_length():
    assert len(get_all_baselines()) == 3


def test_get_all_baselines_order():
    ids = [b.baseline_id for b in get_all_baselines()]
    assert ids == ["B1_mfcc_distance", "B2_stat_distance", "B3_random_projection"]


# ---------------------------------------------------------------------------
# EvaluationProtocol
# ---------------------------------------------------------------------------

def test_protocol_is_evaluation_protocol_instance():
    assert isinstance(get_protocol(), EvaluationProtocol)


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


# ---------------------------------------------------------------------------
# Immutability (frozen dataclasses)
# ---------------------------------------------------------------------------

def test_baseline_definition_is_frozen():
    b = get_baseline("B1_mfcc_distance")
    with pytest.raises((AttributeError, TypeError)):
        b.feature_dim = 999  # type: ignore[misc]


def test_protocol_is_frozen():
    p = get_protocol()
    with pytest.raises((AttributeError, TypeError)):
        p.seed = 0  # type: ignore[misc]
