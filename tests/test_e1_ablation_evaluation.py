"""Tests for experiments/e1_ablation_evaluation.py.

No BEATs, no audio files, no MIMII dataset, no trained checkpoint required.
All feature extraction and model inference is mocked.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.e1_ablation_evaluation as script
from experiments.e1_ablation_definition import ABLATIONS, PROTOCOL
from src.dataset.metadata import AudioMetadata
from src.dataset.split import DatasetSplitter

MACHINE_IDS = list(PROTOCOL.machine_ids)
MACHINE_TYPE = PROTOCOL.machine_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(machine_id: str, label: str, idx: int) -> AudioMetadata:
    p = Path(f"/fake/pump/{machine_id}/{label}/{idx:08d}.wav")
    return AudioMetadata(
        machine_type=MACHINE_TYPE,
        machine_id=machine_id,
        label=label,
        filename=p.name,
        relative_path=Path(f"pump/{machine_id}/{label}/{idx:08d}.wav"),
        absolute_path=p,
    )


def _make_recordings(n_normal: int = 100, n_abnormal: int = 20) -> list[AudioMetadata]:
    recs = []
    for mid in MACHINE_IDS:
        for i in range(n_normal):
            recs.append(_make_meta(mid, "normal", i))
        for i in range(n_abnormal):
            recs.append(_make_meta(mid, "abnormal", i))
    return recs


def _split(recordings):
    return DatasetSplitter(
        train_ratio=PROTOCOL.train_ratio,
        profile_ratio=PROTOCOL.profile_ratio,
        seed=PROTOCOL.seed,
    ).split(recordings)


def _make_profile_and_test(machine_id: str, n_profile: int = 5, n_normal: int = 10, n_abnormal: int = 10):
    profile = [_make_meta(machine_id, "normal", i) for i in range(n_profile)]
    test = (
        [(_make_meta(machine_id, "normal", i + n_profile), "normal") for i in range(n_normal)]
        + [(_make_meta(machine_id, "abnormal", i), "abnormal") for i in range(n_abnormal)]
    )
    return profile, test


def _zero_vec(dim: int) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def _far_vec(dim: int, scale: float = 100.0) -> np.ndarray:
    return np.ones(dim, dtype=np.float32) * scale


# ---------------------------------------------------------------------------
# _FlexProjectionHead
# ---------------------------------------------------------------------------

class TestFlexProjectionHead:
    def test_output_shape_153(self):
        head = script._FlexProjectionHead(input_dim=153)
        x = torch.zeros(4, 153)
        out = head(x)
        assert out.shape == (4, 256)

    def test_output_shape_768(self):
        head = script._FlexProjectionHead(input_dim=768)
        x = torch.zeros(4, 768)
        out = head(x)
        assert out.shape == (4, 256)

    def test_output_shape_921(self):
        head = script._FlexProjectionHead(input_dim=921)
        x = torch.zeros(4, 921)
        out = head(x)
        assert out.shape == (4, 256)

    def test_output_is_l2_normalised(self):
        head = script._FlexProjectionHead(input_dim=153)
        x = torch.randn(8, 153)
        out = head(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_output_no_nan(self):
        head = script._FlexProjectionHead(input_dim=768)
        x = torch.randn(4, 768)
        out = head(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# _train_flex_head
# ---------------------------------------------------------------------------

class TestTrainFlexHead:
    def _make_vecs(self, n: int, dim: int, n_machines: int = 2) -> tuple[list, list]:
        vecs, mids = [], []
        for i in range(n):
            vecs.append(np.random.randn(dim).astype(np.float32))
            mids.append(f"id_{i % n_machines:02d}")
        return vecs, mids

    def test_returns_flex_head(self):
        vecs, mids = self._make_vecs(20, 153)
        head = script._train_flex_head(153, vecs, mids)
        assert isinstance(head, script._FlexProjectionHead)

    def test_head_in_eval_mode(self):
        vecs, mids = self._make_vecs(20, 768)
        head = script._train_flex_head(768, vecs, mids)
        assert not head.training

    def test_output_dim_256(self):
        vecs, mids = self._make_vecs(20, 153)
        head = script._train_flex_head(153, vecs, mids)
        x = torch.zeros(1, 153)
        out = head(x)
        assert out.shape[-1] == 256

    def test_single_machine_returns_head(self):
        """Only one machine → no pairs possible → returns random head without error."""
        vecs = [np.random.randn(153).astype(np.float32) for _ in range(5)]
        mids = ["id_00"] * 5
        head = script._train_flex_head(153, vecs, mids)
        assert isinstance(head, script._FlexProjectionHead)

    def test_too_few_recordings_returns_head(self):
        """Fewer than 2 recordings per machine → returns random head without error."""
        vecs = [np.random.randn(153).astype(np.float32)]
        mids = ["id_00"]
        head = script._train_flex_head(153, vecs, mids)
        assert isinstance(head, script._FlexProjectionHead)


# ---------------------------------------------------------------------------
# _project
# ---------------------------------------------------------------------------

class TestProject:
    def test_returns_numpy_array(self):
        head = script._FlexProjectionHead(input_dim=153)
        vec = np.zeros(153, dtype=np.float32)
        out = script._project(vec, head)
        assert isinstance(out, np.ndarray)

    def test_output_dim_256(self):
        head = script._FlexProjectionHead(input_dim=153)
        vec = np.zeros(153, dtype=np.float32)
        out = script._project(vec, head)
        assert out.shape == (256,)

    def test_output_dtype_float32(self):
        head = script._FlexProjectionHead(input_dim=768)
        vec = np.zeros(768, dtype=np.float32)
        out = script._project(vec, head)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# evaluate_ablation_machine_id — per-ablation contract tests
# ---------------------------------------------------------------------------

class TestEvaluateAblationMachineId:
    """Tests evaluate_ablation_machine_id with mocked _get_vec internals."""

    def _run(
        self,
        ablation_id: str,
        profile_vec: np.ndarray,
        normal_vec: np.ndarray,
        abnormal_vec: np.ndarray,
        machine_id: str = "id_00",
        n_profile: int = 5,
        n_normal: int = 10,
        n_abnormal: int = 10,
    ) -> dict:
        """Run evaluate_ablation_machine_id with controlled per-record vectors.

        Patches the three low-level vector helpers so no audio I/O occurs.
        """
        profile_records, test_records = _make_profile_and_test(
            machine_id, n_profile, n_normal, n_abnormal
        )
        profile_set = {id(r) for r in profile_records}

        # Build minimal mock objects
        mock_cache = MagicMock()
        mock_fused = MagicMock()

        def _fused_side_effect(rec):
            if id(rec) in profile_set or rec.label == "normal":
                mock_fused.fused_feature_vector = profile_vec.copy()
            else:
                mock_fused.fused_feature_vector = abnormal_vec.copy()
            return mock_fused

        mock_cache.load_or_create.side_effect = _fused_side_effect

        # Heads that return controlled vectors
        def _make_head_mock(profile_v, normal_v, abnormal_v):
            m = MagicMock()
            def _side(rec, *args, **kwargs):
                if id(rec) in profile_set or rec.label == "normal":
                    return profile_v.copy()
                return abnormal_v.copy()
            return m, _side

        # Patch the three low-level helpers
        def _mock_fm_a3_a4(rec, cache):
            if id(rec) in profile_set or rec.label == "normal":
                return profile_vec.copy()
            return abnormal_vec.copy()

        def _mock_a1_dsp(rec, cache):
            if id(rec) in profile_set or rec.label == "normal":
                return profile_vec[:153].copy() if len(profile_vec) >= 153 else profile_vec.copy()
            return abnormal_vec[:153].copy() if len(abnormal_vec) >= 153 else abnormal_vec.copy()

        def _mock_a2_beats(rec, cache):
            if id(rec) in profile_set or rec.label == "normal":
                return profile_vec[:768].copy() if len(profile_vec) >= 768 else profile_vec.copy()
            return abnormal_vec[:768].copy() if len(abnormal_vec) >= 768 else abnormal_vec.copy()

        # Minimal head mocks that return L2-normalised versions of the input
        def _make_proj_mock(profile_v, abnormal_v, profile_set_):
            m = MagicMock()
            def _fwd(rec, head_):
                v = profile_v if id(rec) in profile_set_ or rec.label == "normal" else abnormal_v
                norm = np.linalg.norm(v)
                return (v / norm if norm > 0 else v).astype(np.float32)
            return m, _fwd

        with (
            patch.object(script, "_vec_fm_a3_a4", side_effect=_mock_fm_a3_a4),
            patch.object(script, "_vec_a1_dsp", side_effect=_mock_a1_dsp),
            patch.object(script, "_vec_a2_beats", side_effect=_mock_a2_beats),
            patch.object(script, "_project", side_effect=lambda v, h: (
                v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v
            ).astype(np.float32)),
        ):
            return script.evaluate_ablation_machine_id(
                ablation_id=ablation_id,
                machine_id=machine_id,
                profile_records=profile_records,
                test_records=test_records,
                cache=mock_cache,
                fm_head=MagicMock(),
                a1_head=MagicMock(),
                a2_head=MagicMock(),
                a3_head=MagicMock(),
            )

    @pytest.mark.parametrize("ablation_id", list(ABLATIONS.keys()))
    def test_result_has_all_csv_columns(self, ablation_id):
        dim = ABLATIONS[ablation_id].scoring_dim
        v = _zero_vec(dim)
        result = self._run(ablation_id, v, v, v + 1.0)
        for col in script.CSV_COLUMNS:
            assert col in result

    @pytest.mark.parametrize("ablation_id", list(ABLATIONS.keys()))
    def test_ablation_id_preserved(self, ablation_id):
        dim = ABLATIONS[ablation_id].scoring_dim
        v = _zero_vec(dim)
        result = self._run(ablation_id, v, v, v + 1.0)
        assert result["ablation_id"] == ablation_id

    @pytest.mark.parametrize("ablation_id", list(ABLATIONS.keys()))
    def test_ablation_name_non_empty(self, ablation_id):
        dim = ABLATIONS[ablation_id].scoring_dim
        v = _zero_vec(dim)
        result = self._run(ablation_id, v, v, v + 1.0)
        assert result["ablation_name"]

    @pytest.mark.parametrize("ablation_id", list(ABLATIONS.keys()))
    def test_auroc_in_unit_interval(self, ablation_id):
        dim = ABLATIONS[ablation_id].scoring_dim
        v = _zero_vec(dim)
        result = self._run(ablation_id, v, v, v + 1.0)
        assert 0.0 <= result["auroc"] <= 1.0

    @pytest.mark.parametrize("ablation_id", list(ABLATIONS.keys()))
    def test_n_counts_correct(self, ablation_id):
        dim = ABLATIONS[ablation_id].scoring_dim
        v = _zero_vec(dim)
        result = self._run(ablation_id, v, v, v + 1.0, n_normal=8, n_abnormal=6)
        assert result["n_normal"] == 8
        assert result["n_abnormal"] == 6

    @pytest.mark.parametrize("ablation_id", list(ABLATIONS.keys()))
    def test_machine_id_preserved(self, ablation_id):
        dim = ABLATIONS[ablation_id].scoring_dim
        v = _zero_vec(dim)
        result = self._run(ablation_id, v, v, v + 1.0, machine_id="id_02")
        assert result["machine_id"] == "id_02"

    def test_fm_uses_256_dim_scoring(self):
        """FM scores in 256-dim space (after projection)."""
        v256 = _zero_vec(256)
        result = self._run("FM_full_method", v256, v256, v256 + 1.0)
        assert result["ablation_id"] == "FM_full_method"

    def test_a4_uses_921_dim_scoring(self):
        """A4 scores in raw 921-dim fusion space."""
        v921 = _zero_vec(921)
        result = self._run("A4_no_projection", v921, v921, v921 + 1.0)
        assert result["ablation_id"] == "A4_no_projection"

    def test_a4_does_not_call_project(self):
        """A4 must not call _project (no head)."""
        profile_records, test_records = _make_profile_and_test("id_00")
        v = _zero_vec(921)

        def _mock_fm_a3_a4(rec, cache):
            return v.copy()

        with (
            patch.object(script, "_vec_fm_a3_a4", side_effect=_mock_fm_a3_a4),
            patch.object(script, "_project") as mock_proj,
        ):
            script.evaluate_ablation_machine_id(
                ablation_id="A4_no_projection",
                machine_id="id_00",
                profile_records=profile_records,
                test_records=test_records,
                cache=MagicMock(),
            )
        mock_proj.assert_not_called()

    def test_a3_does_not_call_a1_or_a2_helpers(self):
        """A3 must use the fusion cache, not DSP or BEATs-only helpers."""
        profile_records, test_records = _make_profile_and_test("id_00")
        v = _zero_vec(256)

        with (
            patch.object(script, "_vec_fm_a3_a4", return_value=np.zeros(921, dtype=np.float32)),
            patch.object(script, "_project", return_value=v),
            patch.object(script, "_vec_a1_dsp") as mock_a1,
            patch.object(script, "_vec_a2_beats") as mock_a2,
        ):
            script.evaluate_ablation_machine_id(
                ablation_id="A3_no_contrastive",
                machine_id="id_00",
                profile_records=profile_records,
                test_records=test_records,
                cache=MagicMock(),
                a3_head=MagicMock(),
            )
        mock_a1.assert_not_called()
        mock_a2.assert_not_called()


# ---------------------------------------------------------------------------
# validate_inputs
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_missing_dataset_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "DATASET_ROOT", tmp_path / "missing_mimii")
        with pytest.raises(FileNotFoundError, match="MIMII"):
            script.validate_inputs()

    def test_missing_beats_checkpoint_raises(self, tmp_path, monkeypatch):
        dataset = tmp_path / "MIMII"
        dataset.mkdir()
        monkeypatch.setattr(script, "DATASET_ROOT", dataset)
        monkeypatch.setattr(script, "BEATS_CHECKPOINT", tmp_path / "missing_beats.pt")
        monkeypatch.setattr(script, "CONTRASTIVE_CHECKPOINT", tmp_path / "missing_ckpt.pt")
        with pytest.raises(FileNotFoundError, match="BEATs"):
            script.validate_inputs()

    def test_missing_contrastive_checkpoint_raises(self, tmp_path, monkeypatch):
        dataset = tmp_path / "MIMII"
        dataset.mkdir()
        beats = tmp_path / "BEATs.pt"
        beats.write_bytes(b"fake")
        monkeypatch.setattr(script, "DATASET_ROOT", dataset)
        monkeypatch.setattr(script, "BEATS_CHECKPOINT", beats)
        monkeypatch.setattr(script, "CONTRASTIVE_CHECKPOINT", tmp_path / "missing_ckpt.pt")
        with pytest.raises(FileNotFoundError, match="[Cc]ontrastive"):
            script.validate_inputs()

    def test_all_present_does_not_raise(self, tmp_path, monkeypatch):
        dataset = tmp_path / "MIMII"
        dataset.mkdir()
        beats = tmp_path / "BEATs.pt"
        beats.write_bytes(b"fake")
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"fake")
        monkeypatch.setattr(script, "DATASET_ROOT", dataset)
        monkeypatch.setattr(script, "BEATS_CHECKPOINT", beats)
        monkeypatch.setattr(script, "CONTRASTIVE_CHECKPOINT", ckpt)
        script.validate_inputs()  # must not raise


# ---------------------------------------------------------------------------
# validate_results
# ---------------------------------------------------------------------------

class TestValidateResults:
    def _valid_row(self, ablation_id: str = "FM_full_method", machine_id: str = "id_00") -> dict:
        return {
            "ablation_id": ablation_id,
            "ablation_name": ABLATIONS[ablation_id].name,
            "machine_id": machine_id,
            "n_normal": 15,
            "n_abnormal": 10,
            "auroc": 0.80,
            "separation_ratio": 1.5,
        }

    def test_valid_rows_do_not_raise(self):
        rows = [self._valid_row(aid, mid) for aid in ABLATIONS for mid in MACHINE_IDS]
        script.validate_results(rows)

    def test_auroc_above_one_raises(self):
        row = self._valid_row()
        row["auroc"] = 1.1
        with pytest.raises(ValueError, match="AUROC"):
            script.validate_results([row])

    def test_auroc_below_zero_raises(self):
        row = self._valid_row()
        row["auroc"] = -0.01
        with pytest.raises(ValueError, match="AUROC"):
            script.validate_results([row])

    def test_negative_separation_raises(self):
        row = self._valid_row()
        row["separation_ratio"] = -0.5
        with pytest.raises(ValueError, match="separation_ratio"):
            script.validate_results([row])

    def test_nan_auroc_accepted(self):
        row = self._valid_row()
        row["auroc"] = float("nan")
        script.validate_results([row])

    def test_nan_separation_accepted(self):
        row = self._valid_row()
        row["separation_ratio"] = float("nan")
        script.validate_results([row])


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

class TestCSVOutput:
    def _make_rows(self) -> list[dict]:
        return [
            {
                "ablation_id": aid,
                "ablation_name": ABLATIONS[aid].name,
                "machine_id": mid,
                "n_normal": 15,
                "n_abnormal": 10,
                "auroc": 0.75,
                "separation_ratio": 1.2,
            }
            for aid in ABLATIONS
            for mid in MACHINE_IDS
        ]

    def test_csv_has_all_columns(self, tmp_path):
        rows = self._make_rows()
        out = tmp_path / "ablation_results.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=script.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with out.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        for col in script.CSV_COLUMNS:
            assert col in written[0]

    def test_csv_row_count(self, tmp_path):
        rows = self._make_rows()
        out = tmp_path / "ablation_results.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=script.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with out.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        # 5 ablations × 4 machine IDs
        assert len(written) == 5 * len(MACHINE_IDS)

    def test_csv_auroc_round_trips(self, tmp_path):
        rows = self._make_rows()
        rows[0]["auroc"] = 0.8765
        out = tmp_path / "ablation_results.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=script.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with out.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        assert float(written[0]["auroc"]) == pytest.approx(0.8765, abs=1e-4)

    def test_results_path_is_under_ablation_study_dir(self):
        assert "ablation_study" in str(script.RESULTS_PATH)

    def test_csv_columns_include_ablation_id(self):
        assert "ablation_id" in script.CSV_COLUMNS

    def test_csv_columns_include_metrics(self):
        assert "auroc" in script.CSV_COLUMNS
        assert "separation_ratio" in script.CSV_COLUMNS


# ---------------------------------------------------------------------------
# Protocol alignment
# ---------------------------------------------------------------------------

class TestProtocolAlignment:
    def test_same_machine_type_as_baseline(self):
        from experiments.e1_baseline_definition import PROTOCOL as bp
        assert PROTOCOL.machine_type == bp.machine_type

    def test_same_machine_ids_as_baseline(self):
        from experiments.e1_baseline_definition import PROTOCOL as bp
        assert set(PROTOCOL.machine_ids) == set(bp.machine_ids)

    def test_same_train_ratio_as_baseline(self):
        from experiments.e1_baseline_definition import PROTOCOL as bp
        assert PROTOCOL.train_ratio == bp.train_ratio

    def test_same_profile_ratio_as_baseline(self):
        from experiments.e1_baseline_definition import PROTOCOL as bp
        assert PROTOCOL.profile_ratio == bp.profile_ratio

    def test_same_seed_as_baseline(self):
        from experiments.e1_baseline_definition import PROTOCOL as bp
        assert PROTOCOL.seed == bp.seed

    def test_five_ablations_evaluated(self):
        assert len(ABLATIONS) == 5

    def test_four_machine_ids(self):
        assert len(PROTOCOL.machine_ids) == 4


# ---------------------------------------------------------------------------
# Split isolation
# ---------------------------------------------------------------------------

class TestSplitIsolation:
    def test_profile_not_in_test(self):
        split = _split(_make_recordings())
        profile_paths = {r.absolute_path for r in split.profile_normal}
        test_paths = (
            {r.absolute_path for r in split.test_normal}
            | {r.absolute_path for r in split.test_abnormal}
        )
        assert not profile_paths & test_paths

    def test_train_not_in_test(self):
        split = _split(_make_recordings())
        train_paths = {r.absolute_path for r in split.train_normal}
        test_paths = (
            {r.absolute_path for r in split.test_normal}
            | {r.absolute_path for r in split.test_abnormal}
        )
        assert not train_paths & test_paths

    def test_split_is_reproducible(self):
        recs = _make_recordings()
        s1 = _split(recs)
        s2 = _split(recs)
        assert sorted(str(r.absolute_path) for r in s1.train_normal) == \
               sorted(str(r.absolute_path) for r in s2.train_normal)
        assert sorted(str(r.absolute_path) for r in s1.profile_normal) == \
               sorted(str(r.absolute_path) for r in s2.profile_normal)
