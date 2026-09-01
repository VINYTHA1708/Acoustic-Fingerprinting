"""Tests for experiments/e1_health_calibration.py (Phase 6.1).

No BEATs, no audio files, no MIMII dataset required.
Uses synthetic DataFrames only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.e1_health_calibration as script


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n_normal: int = 10,
    n_abnormal: int = 8,
    normal_euclidean: float = 10.0,
    normal_manhattan: float = 150.0,
    normal_cosine: float = -0.05,
) -> pd.DataFrame:
    """Minimal valid evaluation DataFrame."""
    rows = (
        [{"machine_type": "pump", "machine_id": "id_00", "filename": f"n_{i}.wav",
          "true_label": "normal", "health_score": 90.0,
          "normalized_euclidean": normal_euclidean,
          "normalized_manhattan": normal_manhattan,
          "normalized_cosine": normal_cosine}
         for i in range(n_normal)]
        +
        [{"machine_type": "pump", "machine_id": "id_00", "filename": f"a_{i}.wav",
          "true_label": "abnormal", "health_score": 40.0,
          "normalized_euclidean": 40.0,
          "normalized_manhattan": 500.0,
          "normalized_cosine": 0.05}
         for i in range(n_abnormal)]
    )
    return pd.DataFrame(rows)


def _make_varied_normal_df(n: int = 20) -> pd.DataFrame:
    """DataFrame with varied normal drift values for IQR tests."""
    import numpy as np
    rng = np.random.default_rng(0)
    rows = [
        {"machine_type": "pump", "machine_id": "id_00", "filename": f"n_{i}.wav",
         "true_label": "normal", "health_score": 90.0,
         "normalized_euclidean": float(rng.uniform(5, 20)),
         "normalized_manhattan": float(rng.uniform(80, 250)),
         "normalized_cosine": float(rng.uniform(-0.15, 0.05))}
        for i in range(n)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# validate_csv
# ---------------------------------------------------------------------------

class TestValidateCSV:
    def test_missing_column_raises(self):
        df = _make_df().drop(columns=["normalized_euclidean"])
        with pytest.raises(ValueError, match="Missing required columns"):
            script.validate_csv(df)

    def test_empty_df_raises(self):
        empty = _make_df().iloc[0:0]
        with pytest.raises(ValueError, match="CSV is empty"):
            script.validate_csv(empty)

    def test_non_numeric_column_raises(self):
        df = _make_df()
        df["normalized_euclidean"] = "bad"
        with pytest.raises(ValueError, match="not numeric"):
            script.validate_csv(df)

    def test_valid_df_does_not_raise(self):
        script.validate_csv(_make_df())


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------

class TestLoadCSV:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            script.load_csv(tmp_path / "nonexistent.csv")

    def test_valid_file_returns_dataframe(self, tmp_path):
        path = tmp_path / "eval.csv"
        _make_df().to_csv(path, index=False)
        df = script.load_csv(path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_invalid_csv_raises_value_error(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"col_a": [1, 2]}).to_csv(path, index=False)
        with pytest.raises(ValueError):
            script.load_csv(path)


# ---------------------------------------------------------------------------
# compute_healthy_reference — row count
# ---------------------------------------------------------------------------

class TestComputeHealthyReferenceRowCount:
    def test_returns_three_rows(self):
        ref = script.compute_healthy_reference(_make_df())
        assert len(ref) == 3

    def test_one_row_per_metric(self):
        ref = script.compute_healthy_reference(_make_df())
        assert set(ref["metric"].unique()) == set(script.DRIFT_METRICS)


# ---------------------------------------------------------------------------
# compute_healthy_reference — expected columns
# ---------------------------------------------------------------------------

class TestComputeHealthyReferenceColumns:
    def test_all_expected_columns_present(self):
        ref = script.compute_healthy_reference(_make_df())
        for col in script.REFERENCE_COLUMNS:
            assert col in ref.columns, f"Missing column: {col}"

    def test_no_extra_columns(self):
        ref = script.compute_healthy_reference(_make_df())
        assert list(ref.columns) == script.REFERENCE_COLUMNS


# ---------------------------------------------------------------------------
# compute_healthy_reference — only normal recordings used
# ---------------------------------------------------------------------------

class TestOnlyNormalRecordingsUsed:
    def test_count_equals_normal_count(self):
        ref = script.compute_healthy_reference(_make_df(n_normal=12, n_abnormal=8))
        for _, row in ref.iterrows():
            assert int(row["count"]) == 12

    def test_mean_reflects_only_normal_values(self):
        # normal euclidean = 10.0, abnormal = 40.0 — mean must be 10.0
        ref = script.compute_healthy_reference(_make_df(n_normal=10, n_abnormal=10))
        mean_val = ref.loc[ref["metric"] == "normalized_euclidean", "mean"].iloc[0]
        assert abs(mean_val - 10.0) < 1e-9

    def test_abnormal_only_df_raises(self):
        df = _make_df(n_normal=0, n_abnormal=5)
        with pytest.raises(ValueError, match="No normal recordings"):
            script.compute_healthy_reference(df)

    def test_mixed_df_ignores_abnormal(self):
        ref = script.compute_healthy_reference(_make_df(n_normal=7, n_abnormal=100))
        count = int(ref.loc[ref["metric"] == "normalized_euclidean", "count"].iloc[0])
        assert count == 7


# ---------------------------------------------------------------------------
# compute_healthy_reference — correct count
# ---------------------------------------------------------------------------

class TestCorrectCount:
    def test_count_matches_n_normal(self):
        for n in [5, 15, 30]:
            ref = script.compute_healthy_reference(_make_df(n_normal=n))
            for _, row in ref.iterrows():
                assert int(row["count"]) == n


# ---------------------------------------------------------------------------
# compute_healthy_reference — IQR threshold calculations
# ---------------------------------------------------------------------------

class TestIQRThresholds:
    def test_lower_threshold_formula(self):
        ref = script.compute_healthy_reference(_make_varied_normal_df())
        for _, row in ref.iterrows():
            iqr = row["q3"] - row["q1"]
            expected_lower = row["q1"] - 1.5 * iqr
            assert abs(row["lower_threshold"] - expected_lower) < 1e-9

    def test_upper_threshold_formula(self):
        ref = script.compute_healthy_reference(_make_varied_normal_df())
        for _, row in ref.iterrows():
            iqr = row["q3"] - row["q1"]
            expected_upper = row["q3"] + 1.5 * iqr
            assert abs(row["upper_threshold"] - expected_upper) < 1e-9

    def test_upper_threshold_greater_than_lower(self):
        ref = script.compute_healthy_reference(_make_varied_normal_df())
        for _, row in ref.iterrows():
            assert row["upper_threshold"] > row["lower_threshold"]

    def test_constant_values_zero_iqr(self):
        # All normal values identical → IQR = 0 → lower == upper == q1 == q3
        ref = script.compute_healthy_reference(_make_df(n_normal=10))
        row = ref.loc[ref["metric"] == "normalized_euclidean"].iloc[0]
        assert row["q1"] == row["q3"]
        assert row["lower_threshold"] == row["upper_threshold"]

    def test_q1_leq_median_leq_q3(self):
        ref = script.compute_healthy_reference(_make_varied_normal_df())
        for _, row in ref.iterrows():
            assert row["q1"] <= row["median"] <= row["q3"]


# ---------------------------------------------------------------------------
# save_healthy_reference — CSV saving
# ---------------------------------------------------------------------------

class TestSaveHealthyReference:
    def test_creates_output_directory(self, tmp_path):
        out = tmp_path / "subdir" / "healthy_reference.csv"
        ref = script.compute_healthy_reference(_make_df())
        script.save_healthy_reference(ref, out)
        assert out.parent.exists()

    def test_csv_file_created(self, tmp_path):
        out = tmp_path / "health_calibration" / "healthy_reference.csv"
        ref = script.compute_healthy_reference(_make_df())
        script.save_healthy_reference(ref, out)
        assert out.exists()

    def test_csv_round_trip_row_count(self, tmp_path):
        out = tmp_path / "healthy_reference.csv"
        ref = script.compute_healthy_reference(_make_df())
        script.save_healthy_reference(ref, out)
        loaded = pd.read_csv(out)
        assert len(loaded) == 3

    def test_csv_round_trip_columns(self, tmp_path):
        out = tmp_path / "healthy_reference.csv"
        ref = script.compute_healthy_reference(_make_df())
        script.save_healthy_reference(ref, out)
        loaded = pd.read_csv(out)
        for col in script.REFERENCE_COLUMNS:
            assert col in loaded.columns

    def test_csv_values_preserved(self, tmp_path):
        out = tmp_path / "healthy_reference.csv"
        ref = script.compute_healthy_reference(_make_df(n_normal=10))
        script.save_healthy_reference(ref, out)
        loaded = pd.read_csv(out)
        mean_val = loaded.loc[
            loaded["metric"] == "normalized_euclidean", "mean"
        ].iloc[0]
        assert abs(mean_val - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# main — integration
# ---------------------------------------------------------------------------

class TestMain:
    def _patch(self, monkeypatch, tmp_path):
        path = tmp_path / "eval.csv"
        _make_varied_normal_df().to_csv(path, index=False)
        out = tmp_path / "health_calibration" / "healthy_reference.csv"
        monkeypatch.setattr(script, "INPUT_CSV", path)
        monkeypatch.setattr(script, "HEALTHY_REFERENCE_CSV", out)
        return path, out

    def test_main_runs_without_error(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        script.main()

    def test_main_raises_on_missing_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "INPUT_CSV", tmp_path / "missing.csv")
        with pytest.raises(FileNotFoundError):
            script.main()

    def test_main_saves_healthy_reference_csv(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        assert out.exists()

    def test_main_csv_has_three_rows(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        loaded = pd.read_csv(out)
        assert len(loaded) == 3

    def test_main_csv_has_expected_columns(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        loaded = pd.read_csv(out)
        for col in script.REFERENCE_COLUMNS:
            assert col in loaded.columns

    def test_main_prints_output(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch, tmp_path)
        script.main()
        out = capsys.readouterr().out
        assert "normalized_euclidean" in out
        assert "count" in out


# ---------------------------------------------------------------------------
# Phase 6.2 — Per-Metric Health Score Mapping
# ---------------------------------------------------------------------------

def _make_ref(euclidean_upper: float = 30.0,
              manhattan_upper: float = 450.0,
              cosine_upper: float = 0.30) -> pd.DataFrame:
    """Minimal healthy_reference DataFrame with controllable upper_thresholds."""
    rows = [
        {"metric": "normalized_euclidean", "count": 10, "mean": 10.0, "std": 2.0,
         "median": 10.0, "q1": 8.0, "q3": 12.0,
         "lower_threshold": 5.0, "upper_threshold": euclidean_upper},
        {"metric": "normalized_manhattan", "count": 10, "mean": 150.0, "std": 30.0,
         "median": 150.0, "q1": 120.0, "q3": 180.0,
         "lower_threshold": 75.0, "upper_threshold": manhattan_upper},
        {"metric": "normalized_cosine", "count": 10, "mean": -0.05, "std": 0.05,
         "median": -0.05, "q1": -0.09, "q3": 0.07,
         "lower_threshold": -0.33, "upper_threshold": cosine_upper},
    ]
    return pd.DataFrame(rows, columns=script.REFERENCE_COLUMNS)


class TestComputeHealthScoresColumns:
    def test_health_score_columns_exist(self):
        df = _make_df()
        ref = _make_ref()
        scored = script.compute_health_scores(df, ref)
        for col in script.HEALTH_SCORE_COLUMNS:
            assert col in scored.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self):
        df = _make_df()
        ref = _make_ref()
        scored = script.compute_health_scores(df, ref)
        for col in df.columns:
            assert col in scored.columns

    def test_row_count_unchanged(self):
        df = _make_df(n_normal=10, n_abnormal=8)
        ref = _make_ref()
        scored = script.compute_health_scores(df, ref)
        assert len(scored) == len(df)

    def test_missing_drift_column_raises(self):
        df = _make_df().drop(columns=["normalized_euclidean"])
        ref = _make_ref()
        with pytest.raises(ValueError, match="Missing drift columns"):
            script.compute_health_scores(df, ref)

    def test_missing_ref_metric_raises(self):
        df = _make_df()
        ref = _make_ref().iloc[1:]  # drop euclidean row
        with pytest.raises(ValueError, match="Missing metrics in healthy_reference"):
            script.compute_health_scores(df, ref)


class TestComputeHealthScoresBounds:
    def test_scores_between_0_and_100(self):
        df = _make_df(n_normal=10, n_abnormal=10)
        ref = _make_ref()
        scored = script.compute_health_scores(df, ref)
        for col in script.HEALTH_SCORE_COLUMNS:
            assert (scored[col] >= 0).all(), f"{col} has values below 0"
            assert (scored[col] <= 100).all(), f"{col} has values above 100"

    def test_very_high_drift_clipped_to_zero(self):
        # drift >> upper_threshold → raw score negative → clipped to 0
        df = _make_df(n_normal=5, n_abnormal=0, normal_euclidean=9999.0)
        ref = _make_ref(euclidean_upper=30.0)
        scored = script.compute_health_scores(df, ref)
        assert (scored["euclidean_health_score"] == 0.0).all()

    def test_zero_drift_gives_100(self):
        df = _make_df(n_normal=5, n_abnormal=0, normal_euclidean=0.0)
        ref = _make_ref(euclidean_upper=30.0)
        scored = script.compute_health_scores(df, ref)
        assert (scored["euclidean_health_score"] == 100.0).all()

    def test_drift_at_upper_threshold_gives_zero(self):
        upper = 30.0
        df = _make_df(n_normal=5, n_abnormal=0, normal_euclidean=upper)
        ref = _make_ref(euclidean_upper=upper)
        scored = script.compute_health_scores(df, ref)
        assert (scored["euclidean_health_score"] == 0.0).all()

    def test_drift_above_upper_threshold_clipped_to_zero(self):
        upper = 30.0
        df = _make_df(n_normal=5, n_abnormal=0, normal_euclidean=upper * 2)
        ref = _make_ref(euclidean_upper=upper)
        scored = script.compute_health_scores(df, ref)
        assert (scored["euclidean_health_score"] == 0.0).all()


class TestComputeHealthScoresMonotonicity:
    def test_higher_drift_lower_score(self):
        import numpy as np
        # Two recordings: low drift and high drift
        rows = [
            {"machine_type": "pump", "machine_id": "id_00", "filename": "low.wav",
             "true_label": "normal",
             "normalized_euclidean": 5.0,
             "normalized_manhattan": 100.0,
             "normalized_cosine": -0.05},
            {"machine_type": "pump", "machine_id": "id_00", "filename": "high.wav",
             "true_label": "abnormal",
             "normalized_euclidean": 25.0,
             "normalized_manhattan": 400.0,
             "normalized_cosine": 0.25},
        ]
        df = pd.DataFrame(rows)
        ref = _make_ref()
        scored = script.compute_health_scores(df, ref)
        for col in script.HEALTH_SCORE_COLUMNS:
            assert scored.loc[0, col] > scored.loc[1, col], (
                f"{col}: low-drift score should exceed high-drift score"
            )

    def test_normal_recordings_score_higher_than_abnormal(self):
        df = _make_df(n_normal=10, n_abnormal=10)
        ref = _make_ref()
        scored = script.compute_health_scores(df, ref)
        normal_mean = scored.loc[scored["true_label"] == "normal", "euclidean_health_score"].mean()
        abnormal_mean = scored.loc[scored["true_label"] == "abnormal", "euclidean_health_score"].mean()
        assert normal_mean > abnormal_mean


class TestComputeHealthScoresFormula:
    def test_score_formula_euclidean(self):
        upper = 30.0
        drift = 15.0
        expected = 100.0 * (1.0 - drift / upper)  # 50.0
        df = _make_df(n_normal=1, n_abnormal=0, normal_euclidean=drift)
        ref = _make_ref(euclidean_upper=upper)
        scored = script.compute_health_scores(df, ref)
        assert abs(scored["euclidean_health_score"].iloc[0] - expected) < 1e-9

    def test_score_formula_manhattan(self):
        upper = 450.0
        drift = 225.0
        expected = 100.0 * (1.0 - drift / upper)  # 50.0
        df = _make_df(n_normal=1, n_abnormal=0, normal_manhattan=drift)
        ref = _make_ref(manhattan_upper=upper)
        scored = script.compute_health_scores(df, ref)
        assert abs(scored["manhattan_health_score"].iloc[0] - expected) < 1e-9

    def test_score_formula_cosine(self):
        upper = 0.30
        drift = 0.15
        expected = 100.0 * (1.0 - drift / upper)  # 50.0
        df = _make_df(n_normal=1, n_abnormal=0, normal_cosine=drift)
        ref = _make_ref(cosine_upper=upper)
        scored = script.compute_health_scores(df, ref)
        assert abs(scored["cosine_health_score"].iloc[0] - expected) < 1e-9


class TestSaveHealthScores:
    def test_creates_output_directory(self, tmp_path):
        out = tmp_path / "subdir" / "health_scores.csv"
        scored = script.compute_health_scores(_make_df(), _make_ref())
        script.save_health_scores(scored, out)
        assert out.parent.exists()

    def test_csv_file_created(self, tmp_path):
        out = tmp_path / "health_calibration" / "health_scores.csv"
        scored = script.compute_health_scores(_make_df(), _make_ref())
        script.save_health_scores(scored, out)
        assert out.exists()

    def test_csv_round_trip_row_count(self, tmp_path):
        out = tmp_path / "health_scores.csv"
        df = _make_df(n_normal=10, n_abnormal=8)
        scored = script.compute_health_scores(df, _make_ref())
        script.save_health_scores(scored, out)
        loaded = pd.read_csv(out)
        assert len(loaded) == len(df)

    def test_csv_round_trip_health_columns(self, tmp_path):
        out = tmp_path / "health_scores.csv"
        scored = script.compute_health_scores(_make_df(), _make_ref())
        script.save_health_scores(scored, out)
        loaded = pd.read_csv(out)
        for col in script.HEALTH_SCORE_COLUMNS:
            assert col in loaded.columns

    def test_csv_preserves_original_columns(self, tmp_path):
        out = tmp_path / "health_scores.csv"
        df = _make_df()
        scored = script.compute_health_scores(df, _make_ref())
        script.save_health_scores(scored, out)
        loaded = pd.read_csv(out)
        for col in df.columns:
            assert col in loaded.columns


class TestMainPhase62:
    def _patch(self, monkeypatch, tmp_path):
        path = tmp_path / "eval.csv"
        _make_df(n_normal=15, n_abnormal=10).to_csv(path, index=False)
        hc = tmp_path / "health_calibration"
        monkeypatch.setattr(script, "INPUT_CSV", path)
        monkeypatch.setattr(script, "HEALTHY_REFERENCE_CSV", hc / "healthy_reference.csv")
        monkeypatch.setattr(script, "HEALTH_SCORES_CSV", hc / "health_scores.csv")
        return path, hc

    def test_main_saves_health_scores_csv(self, tmp_path, monkeypatch):
        _, hc = self._patch(monkeypatch, tmp_path)
        script.main()
        assert (hc / "health_scores.csv").exists()

    def test_main_health_scores_csv_has_correct_rows(self, tmp_path, monkeypatch):
        _, hc = self._patch(monkeypatch, tmp_path)
        script.main()
        loaded = pd.read_csv(hc / "health_scores.csv")
        assert len(loaded) == 25  # 15 normal + 10 abnormal

    def test_main_health_scores_csv_has_score_columns(self, tmp_path, monkeypatch):
        _, hc = self._patch(monkeypatch, tmp_path)
        script.main()
        loaded = pd.read_csv(hc / "health_scores.csv")
        for col in script.HEALTH_SCORE_COLUMNS:
            assert col in loaded.columns

    def test_main_prints_health_score_summary(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch, tmp_path)
        script.main()
        out = capsys.readouterr().out
        assert "euclidean_health_score" in out
        assert "manhattan_health_score" in out
        assert "cosine_health_score" in out

    def test_main_scores_bounded(self, tmp_path, monkeypatch):
        _, hc = self._patch(monkeypatch, tmp_path)
        script.main()
        loaded = pd.read_csv(hc / "health_scores.csv")
        for col in script.HEALTH_SCORE_COLUMNS:
            assert (loaded[col] >= 0).all()
            assert (loaded[col] <= 100).all()
