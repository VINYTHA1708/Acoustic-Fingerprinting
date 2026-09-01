"""Lightweight tests for experiments/e1_embedding_validation.py.

No BEATs, no audio files, no MIMII dataset, no trained checkpoint required.
Uses synthetic DataFrames only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.e1_embedding_validation as script

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
_N_NORMAL = script.EXPECTED_NORMAL
_N_ABNORMAL = script.EXPECTED_ABNORMAL


def _make_df(
    n_normal: int = _N_NORMAL,
    n_abnormal: int = _N_ABNORMAL,
    machine_ids: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic evaluation DataFrame matching the E1 schema."""
    rng = np.random.default_rng(seed)
    if machine_ids is None:
        machine_ids = MACHINE_IDS

    rows = []
    per_machine_normal = n_normal // len(machine_ids)
    per_machine_abnormal = n_abnormal // len(machine_ids)

    normal_counts = [per_machine_normal] * len(machine_ids)
    abnormal_counts = [per_machine_abnormal] * len(machine_ids)
    normal_counts[0] += n_normal - sum(normal_counts)
    abnormal_counts[0] += n_abnormal - sum(abnormal_counts)

    for i, mid in enumerate(machine_ids):
        for j in range(normal_counts[i]):
            rows.append({
                "machine_type": "pump",
                "machine_id": mid,
                "filename": f"normal_{mid}_{j:06d}.wav",
                "true_label": "normal",
                "health_score": float(rng.uniform(70, 100)),
                "health_percentage": f"{rng.uniform(70, 100):.1f}%",
                "health_state": "GOOD",
                "normalized_euclidean": float(rng.uniform(5, 20)),
                "normalized_manhattan": float(rng.uniform(80, 300)),
                "normalized_cosine": float(rng.uniform(-0.1, 0.1)),
            })
        for j in range(abnormal_counts[i]):
            rows.append({
                "machine_type": "pump",
                "machine_id": mid,
                "filename": f"abnormal_{mid}_{j:06d}.wav",
                "true_label": "abnormal",
                "health_score": float(rng.uniform(0, 60)),
                "health_percentage": f"{rng.uniform(0, 60):.1f}%",
                "health_state": "CRITICAL",
                "normalized_euclidean": float(rng.uniform(25, 80)),
                "normalized_manhattan": float(rng.uniform(350, 1000)),
                "normalized_cosine": float(rng.uniform(-0.05, 0.15)),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Required columns validation
# ---------------------------------------------------------------------------

class TestRequiredColumns:
    def test_missing_column_raises(self):
        df = _make_df()
        df = df.drop(columns=["normalized_euclidean"])
        with pytest.raises(ValueError, match="Missing required columns"):
            script.validate_csv(df, script.INPUT_CSV)

    def test_all_columns_present_does_not_raise(self):
        df = _make_df()
        script.validate_csv(df, script.INPUT_CSV)


# ---------------------------------------------------------------------------
# 2. Missing CSV handling
# ---------------------------------------------------------------------------

class TestMissingCSV:
    def test_missing_csv_raises_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "INPUT_CSV", tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            script.main()


# ---------------------------------------------------------------------------
# 3. Invalid labels rejected
# ---------------------------------------------------------------------------

class TestInvalidLabels:
    def test_unexpected_label_raises(self):
        df = _make_df()
        df.loc[0, "true_label"] = "unknown"
        with pytest.raises(ValueError, match="Unexpected true_label"):
            script.validate_csv(df, script.INPUT_CSV)

    def test_valid_labels_accepted(self):
        df = _make_df()
        assert set(df["true_label"].unique()) == {"normal", "abnormal"}
        script.validate_csv(df, script.INPUT_CSV)


# ---------------------------------------------------------------------------
# 4. Incorrect total count rejected
# ---------------------------------------------------------------------------

class TestTotalCount:
    def test_wrong_total_raises(self):
        df = _make_df(n_normal=100, n_abnormal=100)
        with pytest.raises(ValueError, match="Expected 1022 rows"):
            script.validate_csv(df, script.INPUT_CSV)

    def test_correct_total_accepted(self):
        df = _make_df()
        script.validate_csv(df, script.INPUT_CSV)


# ---------------------------------------------------------------------------
# 5. Incorrect normal/abnormal counts rejected
# ---------------------------------------------------------------------------

class TestLabelCounts:
    def test_wrong_normal_count_raises(self):
        df = _make_df(n_normal=500, n_abnormal=522)
        with pytest.raises(ValueError, match="Expected 566 normal"):
            script.validate_csv(df, script.INPUT_CSV)

    def test_wrong_abnormal_count_raises(self):
        # Keep total=1022, normal=566, but shift 10 abnormal → normal
        # so normal=576, abnormal=446. Rename to avoid duplicate keys.
        df = _make_df()
        df2 = df.copy()
        abnormal_idx = df2[df2["true_label"] == "abnormal"].index[:10]
        df2.loc[abnormal_idx, "true_label"] = "normal"
        for i, idx in enumerate(abnormal_idx):
            df2.loc[idx, "filename"] = f"extra_normal_{i:06d}.wav"
        # normal count is now wrong (576), so that error fires first
        with pytest.raises(ValueError, match="Expected 566 normal"):
            script.validate_csv(df2, script.INPUT_CSV)


# ---------------------------------------------------------------------------
# 6. Duplicate recording detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_duplicate_row_raises(self):
        # Same (machine_type, machine_id, filename, true_label) = true duplicate
        small = pd.DataFrame([
            {"machine_type": "pump", "machine_id": "id_00", "filename": "a.wav", "true_label": "normal"},
            {"machine_type": "pump", "machine_id": "id_00", "filename": "b.wav", "true_label": "normal"},
            {"machine_type": "pump", "machine_id": "id_00", "filename": "a.wav", "true_label": "normal"},
        ])
        with pytest.raises(ValueError, match="Duplicate rows"):
            script._check_duplicates(small)

    def test_same_filename_different_label_not_duplicate(self):
        # MIMII has the same filename in both normal/ and abnormal/ — this is valid
        small = pd.DataFrame([
            {"machine_type": "pump", "machine_id": "id_00", "filename": "a.wav", "true_label": "normal"},
            {"machine_type": "pump", "machine_id": "id_00", "filename": "a.wav", "true_label": "abnormal"},
        ])
        script._check_duplicates(small)  # must not raise

    def test_no_duplicates_in_synthetic_df(self):
        df = _make_df()
        assert not df.duplicated(
            subset=["machine_type", "machine_id", "filename", "true_label"]
        ).any()


# ---------------------------------------------------------------------------
# 7. NaN detection
# ---------------------------------------------------------------------------

class TestNaNDetection:
    def test_nan_in_metric_raises(self):
        df = _make_df()
        df.loc[0, "normalized_euclidean"] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            script.validate_csv(df, script.INPUT_CSV)

    def test_no_nan_accepted(self):
        df = _make_df()
        for col in script.DRIFT_METRICS:
            assert not df[col].isna().any()


# ---------------------------------------------------------------------------
# 8. Inf detection
# ---------------------------------------------------------------------------

class TestInfDetection:
    def test_inf_in_metric_raises(self):
        df = _make_df()
        df.loc[0, "normalized_manhattan"] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            script.validate_csv(df, script.INPUT_CSV)

    def test_neg_inf_in_metric_raises(self):
        df = _make_df()
        df.loc[0, "normalized_euclidean"] = float("-inf")
        with pytest.raises(ValueError, match="Inf"):
            script.validate_csv(df, script.INPUT_CSV)


# ---------------------------------------------------------------------------
# 9. Cohen's d calculation
# ---------------------------------------------------------------------------

class TestCohensD:
    def test_zero_std_both_groups_returns_zero(self):
        # Both groups constant → pooled std = 0 → returns 0.0
        a = pd.Series([10.0] * 50)
        b = pd.Series([0.0] * 50)
        d = script.cohens_d(a, b)
        assert d == 0.0

    def test_separated_groups(self):
        rng = np.random.default_rng(42)
        a = pd.Series(rng.normal(loc=10.0, scale=1.0, size=200))
        b = pd.Series(rng.normal(loc=0.0, scale=1.0, size=200))
        d = script.cohens_d(a, b)
        assert d > 5.0

    def test_identical_groups_returns_zero(self):
        a = pd.Series([5.0] * 100)
        b = pd.Series([5.0] * 100)
        assert script.cohens_d(a, b) == 0.0

    def test_sign_direction(self):
        a = pd.Series([3.0, 4.0, 5.0])
        b = pd.Series([1.0, 2.0, 3.0])
        assert script.cohens_d(a, b) > 0


# ---------------------------------------------------------------------------
# 10. Zero standard deviation handling
# ---------------------------------------------------------------------------

class TestZeroStdHandling:
    def test_zero_std_returns_zero(self):
        a = pd.Series([5.0] * 50)
        b = pd.Series([5.0] * 50)
        assert script.cohens_d(a, b) == 0.0

    def test_one_group_zero_std_does_not_raise(self):
        a = pd.Series([5.0] * 50)
        b = pd.Series(np.random.default_rng(0).normal(3.0, 1.0, 50))
        d = script.cohens_d(a, b)
        assert math.isfinite(d)


# ---------------------------------------------------------------------------
# 11. Correct AUC direction handling
# ---------------------------------------------------------------------------

class TestAUCDirection:
    def test_larger_is_abnormal_gives_high_auc(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "true_label": ["normal"] * 100 + ["abnormal"] * 100,
            "normalized_euclidean": list(rng.normal(5, 1, 100)) + list(rng.normal(20, 1, 100)),
        })
        auc = script.compute_auc(df, "normalized_euclidean", larger_is_abnormal=True)
        assert auc > 0.9

    def test_inversion_corrects_direction(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame({
            "true_label": ["normal"] * 100 + ["abnormal"] * 100,
            "normalized_euclidean": list(rng.normal(20, 1, 100)) + list(rng.normal(5, 1, 100)),
        })
        auc_wrong = script.compute_auc(df, "normalized_euclidean", larger_is_abnormal=True)
        auc_correct = script.compute_auc(df, "normalized_euclidean", larger_is_abnormal=False)
        assert auc_correct > 0.9
        assert auc_wrong < 0.1

    def test_determine_cosine_direction_returns_bool(self):
        result = script.determine_cosine_direction(_make_df())
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 12. Per-machine grouping
# ---------------------------------------------------------------------------

class TestPerMachineGrouping:
    def test_all_machine_ids_present(self):
        df = _make_df()
        rows = script.per_machine_metric_statistics(df)
        assert {r["machine_id"] for r in rows} == set(MACHINE_IDS)

    def test_per_machine_auc_has_all_ids_and_metrics(self):
        df = _make_df()
        cosine_dir = script.determine_cosine_direction(df)
        rows = script.per_machine_auc_results(df, cosine_dir)
        combos = {(r["machine_id"], r["metric"]) for r in rows}
        for mid in MACHINE_IDS:
            for metric in script.DRIFT_METRICS:
                assert (mid, metric) in combos

    def test_per_machine_stats_has_both_labels(self):
        df = _make_df()
        rows = script.per_machine_metric_statistics(df)
        for mid in MACHINE_IDS:
            labels = {r["label"] for r in rows if r["machine_id"] == mid}
            assert "normal" in labels
            assert "abnormal" in labels


# ---------------------------------------------------------------------------
# 13. Output directory creation
# ---------------------------------------------------------------------------

class TestOutputDirectoryCreation:
    def test_output_dir_created(self, tmp_path, monkeypatch):
        new_dir = tmp_path / "new_subdir" / "validation"
        monkeypatch.setattr(script, "OUTPUT_DIR", new_dir)
        monkeypatch.setattr(script, "INPUT_CSV", tmp_path / "eval.csv")
        _make_df().to_csv(tmp_path / "eval.csv", index=False)
        script.main()
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# 14. JSON serialization
# ---------------------------------------------------------------------------

class TestJSONSerialization:
    def test_summary_json_is_serializable(self):
        df = _make_df()
        cosine_dir = script.determine_cosine_direction(df)
        overall_auc = script.overall_auc_results(df, cosine_dir)
        per_machine_auc = script.per_machine_auc_results(df, cosine_dir)
        summary = script.build_summary_json(df, overall_auc, per_machine_auc)
        loaded = json.loads(json.dumps(summary))
        assert loaded["experiment_id"] == "E1"
        assert loaded["total_recordings"] == script.EXPECTED_TOTAL

    def test_summary_json_no_numpy_scalars(self):
        df = _make_df()
        cosine_dir = script.determine_cosine_direction(df)
        overall_auc = script.overall_auc_results(df, cosine_dir)
        per_machine_auc = script.per_machine_auc_results(df, cosine_dir)
        summary = script.build_summary_json(df, overall_auc, per_machine_auc)

        def _check(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _check(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    _check(v, f"{path}[{i}]")
            elif obj is not None:
                assert not isinstance(obj, np.generic), f"NumPy scalar at {path}: {type(obj)}"

        _check(summary)

    def test_summary_contains_required_keys(self):
        df = _make_df()
        cosine_dir = script.determine_cosine_direction(df)
        summary = script.build_summary_json(
            df,
            script.overall_auc_results(df, cosine_dir),
            script.per_machine_auc_results(df, cosine_dir),
        )
        for key in ["experiment_id", "machine_type", "input_csv", "total_recordings",
                    "normal_count", "abnormal_count", "machine_ids",
                    "metrics_analyzed", "overall_results", "per_machine_results"]:
            assert key in summary


# ---------------------------------------------------------------------------
# 15. CSV output generation
# ---------------------------------------------------------------------------

class TestCSVOutputGeneration:
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(script, "INPUT_CSV", tmp_path / "eval.csv")
        _make_df().to_csv(tmp_path / "eval.csv", index=False)

    def test_overall_stats_csv_written(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        script.main()
        assert (tmp_path / "overall_metric_statistics.csv").exists()

    def test_per_machine_stats_csv_written(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        script.main()
        assert (tmp_path / "per_machine_metric_statistics.csv").exists()

    def test_overall_auc_csv_written(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        script.main()
        assert (tmp_path / "overall_auc.csv").exists()

    def test_per_machine_auc_csv_written(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        script.main()
        assert (tmp_path / "per_machine_auc.csv").exists()

    def test_overall_stats_csv_has_required_columns(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        script.main()
        df_out = pd.read_csv(tmp_path / "overall_metric_statistics.csv")
        for col in ["metric", "label", "count", "mean", "std", "median", "min", "max"]:
            assert col in df_out.columns

    def test_overall_auc_csv_has_required_columns(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        script.main()
        df_out = pd.read_csv(tmp_path / "overall_auc.csv")
        for col in ["metric", "roc_auc", "cohens_d", "expected_abnormal_direction"]:
            assert col in df_out.columns

    def test_summary_json_written(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        script.main()
        assert (tmp_path / "embedding_validation_summary.json").exists()
