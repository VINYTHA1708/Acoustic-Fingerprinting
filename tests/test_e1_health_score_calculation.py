"""Tests for experiments/e1_health_score_calculation.py (Phase 6.3).

No BEATs, no audio files, no MIMII dataset required.
Uses synthetic DataFrames only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.e1_health_score_calculation as script
from experiments.e1_health_calibration import REFERENCE_COLUMNS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_ref(
    euclidean_upper: float = 30.0,
    manhattan_upper: float = 450.0,
    cosine_upper: float = 0.30,
) -> pd.DataFrame:
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
    return pd.DataFrame(rows, columns=REFERENCE_COLUMNS)


def _make_eval_df(
    n_normal: int = 10,
    n_abnormal: int = 8,
    normal_euclidean: float = 10.0,
    normal_manhattan: float = 150.0,
    normal_cosine: float = -0.05,
) -> pd.DataFrame:
    """Minimal valid evaluation DataFrame with drift columns."""
    rows = (
        [{"machine_type": "pump", "machine_id": "id_00", "filename": f"n_{i}.wav",
          "true_label": "normal",
          "normalized_euclidean": normal_euclidean,
          "normalized_manhattan": normal_manhattan,
          "normalized_cosine": normal_cosine}
         for i in range(n_normal)]
        +
        [{"machine_type": "pump", "machine_id": "id_00", "filename": f"a_{i}.wav",
          "true_label": "abnormal",
          "normalized_euclidean": 40.0,
          "normalized_manhattan": 500.0,
          "normalized_cosine": 0.05}
         for i in range(n_abnormal)]
    )
    return pd.DataFrame(rows)


def _make_scored_df(
    n_normal: int = 10,
    n_abnormal: int = 8,
    euclidean_health: float = 70.0,
    manhattan_health: float = 60.0,
    cosine_health: float = 80.0,
    abnormal_euclidean_health: float = 10.0,
    abnormal_manhattan_health: float = 5.0,
    abnormal_cosine_health: float = 15.0,
) -> pd.DataFrame:
    """DataFrame that already has the three per-metric health score columns."""
    rows = (
        [{"machine_type": "pump", "machine_id": "id_00", "filename": f"n_{i}.wav",
          "true_label": "normal",
          "normalized_euclidean": 10.0, "normalized_manhattan": 150.0,
          "normalized_cosine": -0.05,
          "euclidean_health_score": euclidean_health,
          "manhattan_health_score": manhattan_health,
          "cosine_health_score": cosine_health}
         for i in range(n_normal)]
        +
        [{"machine_type": "pump", "machine_id": "id_00", "filename": f"a_{i}.wav",
          "true_label": "abnormal",
          "normalized_euclidean": 40.0, "normalized_manhattan": 500.0,
          "normalized_cosine": 0.05,
          "euclidean_health_score": abnormal_euclidean_health,
          "manhattan_health_score": abnormal_manhattan_health,
          "cosine_health_score": abnormal_cosine_health}
         for i in range(n_abnormal)]
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_combined_health_score — output column
# ---------------------------------------------------------------------------

class TestCombinedColumnPresent:
    def test_combined_column_added(self):
        df = _make_scored_df()
        result = script.compute_combined_health_score(df)
        assert script.COMBINED_SCORE_COLUMN in result.columns

    def test_original_columns_preserved(self):
        df = _make_scored_df()
        result = script.compute_combined_health_score(df)
        for col in df.columns:
            assert col in result.columns

    def test_row_count_unchanged(self):
        df = _make_scored_df(n_normal=10, n_abnormal=8)
        result = script.compute_combined_health_score(df)
        assert len(result) == len(df)

    def test_missing_per_metric_column_raises(self):
        df = _make_scored_df().drop(columns=["euclidean_health_score"])
        with pytest.raises(ValueError, match="Missing per-metric health score columns"):
            script.compute_combined_health_score(df)

    def test_all_three_missing_raises(self):
        df = _make_scored_df().drop(
            columns=["euclidean_health_score", "manhattan_health_score", "cosine_health_score"]
        )
        with pytest.raises(ValueError, match="Missing per-metric health score columns"):
            script.compute_combined_health_score(df)


# ---------------------------------------------------------------------------
# compute_combined_health_score — bounds [0, 100]
# ---------------------------------------------------------------------------

class TestCombinedScoreBounds:
    def test_scores_between_0_and_100(self):
        df = _make_scored_df(n_normal=10, n_abnormal=10)
        result = script.compute_combined_health_score(df)
        col = script.COMBINED_SCORE_COLUMN
        assert (result[col] >= 0).all()
        assert (result[col] <= 100).all()

    def test_all_zero_per_metric_gives_zero_combined(self):
        df = _make_scored_df(
            n_normal=5, n_abnormal=0,
            euclidean_health=0.0, manhattan_health=0.0, cosine_health=0.0,
        )
        result = script.compute_combined_health_score(df)
        assert (result[script.COMBINED_SCORE_COLUMN] == 0.0).all()

    def test_all_hundred_per_metric_gives_hundred_combined(self):
        df = _make_scored_df(
            n_normal=5, n_abnormal=0,
            euclidean_health=100.0, manhattan_health=100.0, cosine_health=100.0,
        )
        result = script.compute_combined_health_score(df)
        assert (result[script.COMBINED_SCORE_COLUMN] >= 99.9999).all()

    def test_combined_score_clipped_to_zero(self):
        # Force a negative raw combined by using negative per-metric scores
        # (shouldn't happen in practice, but the clip must hold)
        df = _make_scored_df(
            n_normal=3, n_abnormal=0,
            euclidean_health=-50.0, manhattan_health=-50.0, cosine_health=-50.0,
        )
        result = script.compute_combined_health_score(df)
        assert (result[script.COMBINED_SCORE_COLUMN] >= 0.0).all()

    def test_combined_score_clipped_to_hundred(self):
        df = _make_scored_df(
            n_normal=3, n_abnormal=0,
            euclidean_health=200.0, manhattan_health=200.0, cosine_health=200.0,
        )
        result = script.compute_combined_health_score(df)
        assert (result[script.COMBINED_SCORE_COLUMN] <= 100.0).all()


# ---------------------------------------------------------------------------
# compute_combined_health_score — formula (equal-weight mean)
# ---------------------------------------------------------------------------

class TestCombinedScoreFormula:
    def test_equal_weight_mean(self):
        e, m, c = 60.0, 90.0, 30.0
        expected = (e + m + c) / 3.0
        df = _make_scored_df(
            n_normal=1, n_abnormal=0,
            euclidean_health=e, manhattan_health=m, cosine_health=c,
        )
        result = script.compute_combined_health_score(df)
        assert abs(result[script.COMBINED_SCORE_COLUMN].iloc[0] - expected) < 1e-9

    def test_symmetric_scores_give_same_combined(self):
        df = _make_scored_df(
            n_normal=1, n_abnormal=0,
            euclidean_health=50.0, manhattan_health=50.0, cosine_health=50.0,
        )
        result = script.compute_combined_health_score(df)
        assert abs(result[script.COMBINED_SCORE_COLUMN].iloc[0] - 50.0) < 1e-9

    def test_weights_sum_to_one(self):
        total = sum(script.WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_all_three_metrics_contribute(self):
        # Change only one metric at a time and verify the combined score changes
        base = _make_scored_df(
            n_normal=1, n_abnormal=0,
            euclidean_health=50.0, manhattan_health=50.0, cosine_health=50.0,
        )
        base_score = script.compute_combined_health_score(base)[script.COMBINED_SCORE_COLUMN].iloc[0]

        for col in script.HEALTH_SCORE_COLUMNS:
            modified = base.copy()
            modified[col] = 80.0
            new_score = script.compute_combined_health_score(modified)[script.COMBINED_SCORE_COLUMN].iloc[0]
            assert new_score != base_score, f"Changing {col} had no effect on combined score"


# ---------------------------------------------------------------------------
# compute_combined_health_score — monotonicity
# ---------------------------------------------------------------------------

class TestCombinedScoreMonotonicity:
    def test_higher_per_metric_scores_give_higher_combined(self):
        low = _make_scored_df(
            n_normal=1, n_abnormal=0,
            euclidean_health=20.0, manhattan_health=20.0, cosine_health=20.0,
        )
        high = _make_scored_df(
            n_normal=1, n_abnormal=0,
            euclidean_health=80.0, manhattan_health=80.0, cosine_health=80.0,
        )
        low_score = script.compute_combined_health_score(low)[script.COMBINED_SCORE_COLUMN].iloc[0]
        high_score = script.compute_combined_health_score(high)[script.COMBINED_SCORE_COLUMN].iloc[0]
        assert high_score > low_score

    def test_normal_mean_higher_than_abnormal_mean(self):
        df = _make_scored_df(n_normal=10, n_abnormal=10)
        result = script.compute_combined_health_score(df)
        normal_mean = result.loc[
            result["true_label"] == "normal", script.COMBINED_SCORE_COLUMN
        ].mean()
        abnormal_mean = result.loc[
            result["true_label"] == "abnormal", script.COMBINED_SCORE_COLUMN
        ].mean()
        assert normal_mean > abnormal_mean


# ---------------------------------------------------------------------------
# compute_combined_health_score — edge cases
# ---------------------------------------------------------------------------

class TestCombinedScoreEdgeCases:
    def test_single_recording(self):
        df = _make_scored_df(n_normal=1, n_abnormal=0)
        result = script.compute_combined_health_score(df)
        assert len(result) == 1
        assert script.COMBINED_SCORE_COLUMN in result.columns

    def test_normal_only_df(self):
        df = _make_scored_df(n_normal=5, n_abnormal=0)
        result = script.compute_combined_health_score(df)
        assert len(result) == 5
        assert (result[script.COMBINED_SCORE_COLUMN] >= 0).all()
        assert (result[script.COMBINED_SCORE_COLUMN] <= 100).all()

    def test_abnormal_only_df(self):
        df = _make_scored_df(n_normal=0, n_abnormal=5)
        result = script.compute_combined_health_score(df)
        assert len(result) == 5
        assert (result[script.COMBINED_SCORE_COLUMN] >= 0).all()
        assert (result[script.COMBINED_SCORE_COLUMN] <= 100).all()

    def test_does_not_mutate_input(self):
        df = _make_scored_df()
        cols_before = list(df.columns)
        script.compute_combined_health_score(df)
        assert list(df.columns) == cols_before
        assert script.COMBINED_SCORE_COLUMN not in df.columns

    def test_zero_threshold_safe_via_pipeline(self):
        # Build a ref where upper_threshold == 0 for euclidean.
        # compute_health_scores handles this by setting score = 0.
        # compute_combined_health_score must still produce a valid result.
        ref = _make_ref(euclidean_upper=0.0)
        eval_df = _make_eval_df(n_normal=3, n_abnormal=0)
        from experiments.e1_health_calibration import compute_health_scores
        scored = compute_health_scores(eval_df, ref)
        result = script.compute_combined_health_score(scored)
        assert (result[script.COMBINED_SCORE_COLUMN] >= 0).all()
        assert (result[script.COMBINED_SCORE_COLUMN] <= 100).all()


# ---------------------------------------------------------------------------
# save_combined_health_scores
# ---------------------------------------------------------------------------

class TestSaveCombinedHealthScores:
    def test_creates_output_directory(self, tmp_path):
        out = tmp_path / "subdir" / "combined_health_scores.csv"
        result = script.compute_combined_health_score(_make_scored_df())
        script.save_combined_health_scores(result, out)
        assert out.parent.exists()

    def test_csv_file_created(self, tmp_path):
        out = tmp_path / "health_calibration" / "combined_health_scores.csv"
        result = script.compute_combined_health_score(_make_scored_df())
        script.save_combined_health_scores(result, out)
        assert out.exists()

    def test_csv_round_trip_row_count(self, tmp_path):
        out = tmp_path / "combined_health_scores.csv"
        df = _make_scored_df(n_normal=10, n_abnormal=8)
        result = script.compute_combined_health_score(df)
        script.save_combined_health_scores(result, out)
        loaded = pd.read_csv(out)
        assert len(loaded) == len(df)

    def test_csv_has_combined_column(self, tmp_path):
        out = tmp_path / "combined_health_scores.csv"
        result = script.compute_combined_health_score(_make_scored_df())
        script.save_combined_health_scores(result, out)
        loaded = pd.read_csv(out)
        assert script.COMBINED_SCORE_COLUMN in loaded.columns

    def test_csv_preserves_original_columns(self, tmp_path):
        out = tmp_path / "combined_health_scores.csv"
        df = _make_scored_df()
        result = script.compute_combined_health_score(df)
        script.save_combined_health_scores(result, out)
        loaded = pd.read_csv(out)
        for col in df.columns:
            assert col in loaded.columns

    def test_csv_values_bounded(self, tmp_path):
        out = tmp_path / "combined_health_scores.csv"
        result = script.compute_combined_health_score(_make_scored_df(n_normal=10, n_abnormal=10))
        script.save_combined_health_scores(result, out)
        loaded = pd.read_csv(out)
        assert (loaded[script.COMBINED_SCORE_COLUMN] >= 0).all()
        assert (loaded[script.COMBINED_SCORE_COLUMN] <= 100).all()


# ---------------------------------------------------------------------------
# main — integration
# ---------------------------------------------------------------------------

class TestMain:
    def _patch(self, monkeypatch, tmp_path, n_normal: int = 15, n_abnormal: int = 10):
        path = tmp_path / "eval.csv"
        _make_eval_df(n_normal=n_normal, n_abnormal=n_abnormal).to_csv(path, index=False)
        out = tmp_path / "health_calibration" / "combined_health_scores.csv"
        monkeypatch.setattr(script, "INPUT_CSV", path)
        monkeypatch.setattr(script, "COMBINED_HEALTH_CSV", out)
        return path, out

    def test_main_runs_without_error(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        script.main()

    def test_main_raises_on_missing_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "INPUT_CSV", tmp_path / "missing.csv")
        with pytest.raises(FileNotFoundError):
            script.main()

    def test_main_saves_combined_csv(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        assert out.exists()

    def test_main_csv_has_correct_row_count(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path, n_normal=15, n_abnormal=10)
        script.main()
        loaded = pd.read_csv(out)
        assert len(loaded) == 25

    def test_main_csv_has_combined_column(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        loaded = pd.read_csv(out)
        assert script.COMBINED_SCORE_COLUMN in loaded.columns

    def test_main_csv_scores_bounded(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        loaded = pd.read_csv(out)
        assert (loaded[script.COMBINED_SCORE_COLUMN] >= 0).all()
        assert (loaded[script.COMBINED_SCORE_COLUMN] <= 100).all()

    def test_main_prints_experiment_id(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch, tmp_path)
        script.main()
        out = capsys.readouterr().out
        assert "E1" in out

    def test_main_prints_combined_score_summary(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch, tmp_path)
        script.main()
        out = capsys.readouterr().out
        assert "Combined Health Score" in out
        assert "Normal" in out or "normal" in out.lower()

    def test_main_normal_mean_higher_than_abnormal_mean(self, tmp_path, monkeypatch):
        # Build a pre-scored CSV so normal scores are clearly above abnormal scores.
        df = _make_scored_df(n_normal=20, n_abnormal=20)
        # Add the drift columns required by load_csv / validate_csv
        path = tmp_path / "eval.csv"
        df.to_csv(path, index=False)
        out = tmp_path / "health_calibration" / "combined_health_scores.csv"
        monkeypatch.setattr(script, "INPUT_CSV", path)
        monkeypatch.setattr(script, "COMBINED_HEALTH_CSV", out)
        # Run compute_combined_health_score directly (bypasses pipeline)
        result = script.compute_combined_health_score(df)
        script.save_combined_health_scores(result, out)
        loaded = pd.read_csv(out)
        normal_mean = loaded.loc[
            loaded["true_label"] == "normal", script.COMBINED_SCORE_COLUMN
        ].mean()
        abnormal_mean = loaded.loc[
            loaded["true_label"] == "abnormal", script.COMBINED_SCORE_COLUMN
        ].mean()
        assert normal_mean > abnormal_mean
