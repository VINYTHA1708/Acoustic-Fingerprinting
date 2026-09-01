"""Unit tests for LearnedProfileBuilder explicit recordings interface.

No BEATs checkpoint, audio files, or MIMII dataset required.
LearnedProfileBuilder.__init__ is bypassed via mock injection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.dataset.metadata import AudioMetadata
from src.learned_profile.builder import LearnedProfileBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MACHINE_TYPE = "pump"
_MACHINE_ID = "id_00"
_DIM = 256


def _make_record(
    label: str = "normal",
    machine_type: str = _MACHINE_TYPE,
    machine_id: str = _MACHINE_ID,
    filename: str = "rec_00.wav",
) -> AudioMetadata:
    p = Path(f"data/raw/MIMII/{machine_type}/{machine_id}/{label}/{filename}")
    return AudioMetadata(
        machine_type=machine_type,
        machine_id=machine_id,
        label=label,
        filename=filename,
        relative_path=p,
        absolute_path=p.resolve(),
    )


def _make_builder() -> LearnedProfileBuilder:
    """Return a LearnedProfileBuilder with all heavy components mocked out."""
    with (
        patch("src.learned_profile.builder.BEATsEncoder"),
        patch("src.learned_profile.builder.FusionCache"),
        patch("src.learned_profile.builder.ContrastiveInference"),
        patch("src.learned_profile.builder.ProjectionHead"),
    ):
        builder = LearnedProfileBuilder(checkpoint_path="fake.pt")

    # Inject a mock cache and inference that produce valid 256-dim embeddings
    mock_cache = MagicMock()
    mock_cache.load_or_create.return_value = MagicMock()

    mock_inference = MagicMock()
    rng = np.random.default_rng(0)
    mock_inference.generate_fingerprint.side_effect = lambda _: rng.standard_normal(_DIM).astype(np.float32)

    builder._cache = mock_cache
    builder._inference = mock_inference
    return builder


# ---------------------------------------------------------------------------
# A. Explicit recordings interface accepts valid normal recordings
# ---------------------------------------------------------------------------

class TestExplicitRecordingsAccepted:

    def test_builds_profile_from_valid_recordings(self):
        builder = _make_builder()
        recs = [_make_record(filename=f"rec_{i:02d}.wav") for i in range(3)]
        profile = builder.build(
            machine_type=_MACHINE_TYPE,
            machine_id=_MACHINE_ID,
            recordings=recs,
        )
        assert profile is not None
        assert profile.machine_type == _MACHINE_TYPE
        assert profile.machine_id == _MACHINE_ID
        assert profile.embeddings.shape == (3, _DIM)
        assert profile.mean_vector.shape == (_DIM,)
        assert profile.std_vector.shape == (_DIM,)

    def test_only_supplied_recordings_are_used(self):
        """Cache.load_or_create is called exactly once per supplied recording."""
        builder = _make_builder()
        recs = [_make_record(filename=f"rec_{i:02d}.wav") for i in range(4)]
        builder.build(machine_type=_MACHINE_TYPE, machine_id=_MACHINE_ID, recordings=recs)
        assert builder._cache.load_or_create.call_count == 4


# ---------------------------------------------------------------------------
# B. Explicit recordings rejects an empty list
# ---------------------------------------------------------------------------

class TestExplicitRecordingsRejectsEmpty:

    def test_empty_list_raises_value_error(self):
        builder = _make_builder()
        with pytest.raises(ValueError, match="empty"):
            builder.build(
                machine_type=_MACHINE_TYPE,
                machine_id=_MACHINE_ID,
                recordings=[],
            )


# ---------------------------------------------------------------------------
# C. Explicit recordings rejects abnormal recordings
# ---------------------------------------------------------------------------

class TestExplicitRecordingsRejectsAbnormal:

    def test_abnormal_label_raises_value_error(self):
        builder = _make_builder()
        recs = [_make_record(label="abnormal")]
        with pytest.raises(ValueError, match="label"):
            builder.build(
                machine_type=_MACHINE_TYPE,
                machine_id=_MACHINE_ID,
                recordings=recs,
            )

    def test_mixed_labels_raises_value_error(self):
        builder = _make_builder()
        recs = [
            _make_record(label="normal", filename="n.wav"),
            _make_record(label="abnormal", filename="ab.wav"),
        ]
        with pytest.raises(ValueError, match="label"):
            builder.build(
                machine_type=_MACHINE_TYPE,
                machine_id=_MACHINE_ID,
                recordings=recs,
            )


# ---------------------------------------------------------------------------
# D. Explicit recordings rejects machine_type mismatch
# ---------------------------------------------------------------------------

class TestExplicitRecordingsRejectsMachineTypeMismatch:

    def test_wrong_machine_type_raises_value_error(self):
        builder = _make_builder()
        recs = [_make_record(machine_type="valve")]
        with pytest.raises(ValueError, match="machine_type"):
            builder.build(
                machine_type=_MACHINE_TYPE,
                machine_id=_MACHINE_ID,
                recordings=recs,
            )


# ---------------------------------------------------------------------------
# E. Explicit recordings rejects machine_id mismatch
# ---------------------------------------------------------------------------

class TestExplicitRecordingsRejectsMachineIdMismatch:

    def test_wrong_machine_id_raises_value_error(self):
        builder = _make_builder()
        recs = [_make_record(machine_id="id_99")]
        with pytest.raises(ValueError, match="machine_id"):
            builder.build(
                machine_type=_MACHINE_TYPE,
                machine_id=_MACHINE_ID,
                recordings=recs,
            )


# ---------------------------------------------------------------------------
# F. Supplying both loader and recordings raises ValueError
# ---------------------------------------------------------------------------

class TestBothLoaderAndRecordingsRaises:

    def test_both_supplied_raises_value_error(self):
        builder = _make_builder()
        mock_loader = MagicMock()
        recs = [_make_record()]
        with pytest.raises(ValueError, match="not both"):
            builder.build(
                machine_type=_MACHINE_TYPE,
                machine_id=_MACHINE_ID,
                loader=mock_loader,
                recordings=recs,
            )


# ---------------------------------------------------------------------------
# G. Supplying neither loader nor recordings raises ValueError
# ---------------------------------------------------------------------------

class TestNeitherLoaderNorRecordingsRaises:

    def test_neither_supplied_raises_value_error(self):
        builder = _make_builder()
        with pytest.raises(ValueError, match="must be supplied"):
            builder.build(
                machine_type=_MACHINE_TYPE,
                machine_id=_MACHINE_ID,
            )


# ---------------------------------------------------------------------------
# H. Existing DatasetLoader-based usage still works
# ---------------------------------------------------------------------------

class TestLoaderBackwardCompatibility:

    def test_loader_path_still_works(self):
        builder = _make_builder()
        mock_loader = MagicMock()
        mock_loader.get_all_files.return_value = [
            _make_record(filename=f"rec_{i:02d}.wav") for i in range(3)
        ]
        profile = builder.build(
            machine_type=_MACHINE_TYPE,
            machine_id=_MACHINE_ID,
            loader=mock_loader,
        )
        assert profile is not None
        assert profile.embeddings.shape == (3, _DIM)
        mock_loader.get_all_files.assert_called_once()

    def test_loader_filters_by_machine_type_and_id(self):
        """Records from other machines in the loader are excluded."""
        builder = _make_builder()
        mock_loader = MagicMock()
        mock_loader.get_all_files.return_value = [
            _make_record(filename="good.wav"),
            _make_record(machine_type="valve", filename="other_type.wav"),
            _make_record(machine_id="id_99", filename="other_id.wav"),
        ]
        profile = builder.build(
            machine_type=_MACHINE_TYPE,
            machine_id=_MACHINE_ID,
            loader=mock_loader,
        )
        # Only the one matching record should be processed
        assert profile.embeddings.shape == (1, _DIM)

    def test_loader_no_matching_records_raises(self):
        builder = _make_builder()
        mock_loader = MagicMock()
        mock_loader.get_all_files.return_value = []
        with pytest.raises(ValueError):
            builder.build(
                machine_type=_MACHINE_TYPE,
                machine_id=_MACHINE_ID,
                loader=mock_loader,
            )
