"""Tests for experiments/e1_health_score_evaluation.py (Phase 6.4).

No BEATs, no audio files, no MIMII dataset required.
Uses synthetic DataFrames only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.e1_health_score_evaluation as script


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_combined_df(
    n_normal: int = 20,
    n_abnormal: int = 15,
    normal_score: float = 75.0,
    abnormal_score: float = 30.0,
) -> pd.DataFrame:
    """DataFrame with combined_health_score already present."""
    rows = (
        [{"machine_type": "pump", "machine_id": "id_00", "filename": f"n_{i}.wav",
          "true_label": "normal",
          script.COMBINED_SCORE_COLUMN: normal_score}
         for i in range(n_normal)]
        +
        [{"machine_type": "pump", "machine_id": "id_00", "filename": f"a_{i}.wav",
          "true_label": "abnormal",
          script.COMBINED_SCORE_COLUMN: abnormal_score}
         for i in range(n_abnormal)]
    )
    return pd.DataFrame(rows)


def _make_varied_combined_df(seed: int = 42) -> pd.DataFrame:
    """DataFrame with varied scores so statistics are non-trivial."""
    rng = np.random.default_rng(seed)
    normal_scores = rng.uniform(55, 95, size=30).tolist()
    abnormal_scores = rng.uniform(10, 50, size=20).tolist()
    rows = (
        [{"true_label": "normal",   script.COMBINED_SCORE_COLUMN: s} for s in normal_scores]
        + [{"true_label": "abnormal", script.COMBINED_SCORE_COLUMN: s} for s in abnormal_scores]
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# validate_combined_df
# ---------------------------------------------------------------------------

class TestValidateCombinedDF:
    def test_empty_df_raises(self):
        empty = _make_combined_df().iloc[0:0]
        with pytest.raises(ValueError, match="empty"):
            script.validate_combined_df(empty)

    def test_missing_true_label_raises(self):
        df = _make_combined_df().drop(columns=["true_label"])
        with pytest.raises(ValueError, match="Missing required columns"):
            script.validate_combined_df(df)

    def test_missing_combined_score_raises(self):
        df = _make_combined_df().drop(columns=[script.COMBINED_SCORE_COLUMN])
        with pytest.raises(ValueError, match="Missing required columns"):
            script.validate_combined_df(df)

    def test_non_numeric_score_raises(self):
        df = _make_combined_df()
        df[script.COMBINED_SCORE_COLUMN] = "bad"
        with pytest.raises(ValueError, match="not numeric"):
            script.validate_combined_df(df)

    def test_valid_df_does_not_raise(self):
        script.validate_combined_df(_make_combined_df())


# ---------------------------------------------------------------------------
# compute_evaluation — output shape and columns
# ---------------------------------------------------------------------------

class TestComputeEvaluationShape:
    def test_returns_single_row(self):
        ev = script.compute_evaluation(_make_combined_df())
        assert len(ev) == 1

    def test_all_expected_columns_present(self):
        ev = script.compute_evaluation(_make_combined_df())
        for col in script.EVALUATION_COLUMNS:
            assert col in ev.columns, f"Missing column: {col}"

    def test_no_extra_columns(self):
        ev = script.compute_evaluation(_make_combined_df())
        assert list(ev.columns) == script.EVALUATION_COLUMNS

    def test_missing_normal_label_raises(self):
        df = _make_combined_df(n_normal=0, n_abnormal=10)
        with pytest.raises(ValueError, match="normal"):
            script.compute_evaluation(df)

    def test_missing_abnormal_label_raises(self):
        df = _make_combined_df(n_normal=10, n_abnormal=0)
        with pytest.raises(ValueError, match="abnormal"):
            script.compute_evaluation(df)


# ---------------------------------------------------------------------------
# compute_evaluation — descriptive statistics
# ---------------------------------------------------------------------------

class TestComputeEvaluationDescriptiveStats:
    def test_normal_count(self):
        ev = script.compute_evaluation(_make_combined_df(n_normal=20, n_abnormal=15))
        assert int(ev["normal_count"].iloc[0]) == 20

    def test_abnormal_count(self):
        ev = script.compute_evaluation(_make_combined_df(n_normal=20, n_abnormal=15))
        assert int(ev["abnormal_count"].iloc[0]) == 15

    def test_normal_mean_constant(self):
        ev = script.compute_evaluation(_make_combined_df(normal_score=75.0))
        assert abs(ev["normal_mean"].iloc[0] - 75.0) < 1e-9

    def test_abnormal_mean_constant(self):
        ev = script.compute_evaluation(_make_combined_df(abnormal_score=30.0))
        assert abs(ev["abnormal_mean"].iloc[0] - 30.0) < 1e-9

    def test_normal_median_constant(self):
        ev = script.compute_evaluation(_make_combined_df(normal_score=75.0))
        assert abs(ev["normal_median"].iloc[0] - 75.0) < 1e-9

    def test_abnormal_median_constant(self):
        ev = script.compute_evaluation(_make_combined_df(abnormal_score=30.0))
        assert abs(ev["abnormal_median"].iloc[0] - 30.0) < 1e-9

    def test_normal_std_zero_for_constant(self):
        ev = script.compute_evaluation(_make_combined_df(normal_score=75.0))
        assert abs(ev["normal_std"].iloc[0]) < 1e-9

    def test_normal_min_max_constant(self):
        ev = script.compute_evaluation(_make_combined_df(normal_score=75.0))
        assert abs(ev["normal_min"].iloc[0] - 75.0) < 1e-9
        assert abs(ev["normal_max"].iloc[0] - 75.0) < 1e-9

    def test_normal_mean_higher_than_abnormal_mean(self):
        ev = script.compute_evaluation(
            _make_combined_df(normal_score=75.0, abnormal_score=30.0)
        )
        assert ev["normal_mean"].iloc[0] > ev["abnormal_mean"].iloc[0]

    def test_varied_stats_are_finite(self):
        ev = script.compute_evaluation(_make_varied_combined_df())
        for col in script.EVALUATION_COLUMNS:
            assert pd.notna(ev[col].iloc[0]), f"NaN in column: {col}"


# ---------------------------------------------------------------------------
# compute_evaluation — Mann-Whitney U
# ---------------------------------------------------------------------------

class TestComputeEvaluationMannWhitney:
    def test_u_statistic_non_negative(self):
        ev = script.compute_evaluation(_make_varied_combined_df())
        assert ev["u_statistic"].iloc[0] >= 0

    def test_p_value_in_range(self):
        ev = script.compute_evaluation(_make_varied_combined_df())
        p = ev["p_value"].iloc[0]
        assert 0.0 <= p <= 1.0

    def test_well_separated_data_is_significant(self):
        ev = script.compute_evaluation(
            _make_combined_df(n_normal=30, n_abnormal=30,
                              normal_score=80.0, abnormal_score=20.0)
        )
        assert ev["p_value"].iloc[0] < 0.05

    def test_identical_distributions_not_significant(self):
        # Both groups same score → p should be large (not significant)
        ev = script.compute_evaluation(
            _make_combined_df(n_normal=20, n_abnormal=20,
                              normal_score=50.0, abnormal_score=50.0)
        )
        assert ev["p_value"].iloc[0] > 0.05

    def test_correct_sample_counts_in_u(self):
        ev = script.compute_evaluation(_make_varied_combined_df())
        # U ≤ n_normal * n_abnormal
        assert ev["u_statistic"].iloc[0] <= 30 * 20


# ---------------------------------------------------------------------------
# compute_evaluation — rank-biserial correlation
# ---------------------------------------------------------------------------

class TestComputeEvaluationEffectSize:
    def test_rbc_in_range(self):
        ev = script.compute_evaluation(_make_varied_combined_df())
        r = ev["rank_biserial_correlation"].iloc[0]
        assert -1.0 <= r <= 1.0

    def test_well_separated_data_has_large_effect(self):
        ev = script.compute_evaluation(
            _make_combined_df(n_normal=30, n_abnormal=30,
                              normal_score=80.0, abnormal_score=20.0)
        )
        assert abs(ev["rank_biserial_correlation"].iloc[0]) > 0.5

    def test_rbc_formula_matches_manual(self):
        df = _make_combined_df(n_normal=20, n_abnormal=15,
                               normal_score=75.0, abnormal_score=30.0)
        from scipy.stats import mannwhitneyu
        normal = df.loc[df["true_label"] == "normal", script.COMBINED_SCORE_COLUMN]
        abnormal = df.loc[df["true_label"] == "abnormal", script.COMBINED_SCORE_COLUMN]
        u, _ = mannwhitneyu(normal, abnormal, alternative="two-sided")
        expected_rbc = 1.0 - (2.0 * float(u)) / (len(normal) * len(abnormal))
        ev = script.compute_evaluation(df)
        assert abs(ev["rank_biserial_correlation"].iloc[0] - expected_rbc) < 1e-9

    def test_identical_distributions_rbc_near_zero(self):
        ev = script.compute_evaluation(
            _make_combined_df(n_normal=20, n_abnormal=20,
                              normal_score=50.0, abnormal_score=50.0)
        )
        assert abs(ev["rank_biserial_correlation"].iloc[0]) < 0.1


# ---------------------------------------------------------------------------
# compute_evaluation — ROC-AUC
# ---------------------------------------------------------------------------

class TestComputeEvaluationROCAUC:
    def test_roc_auc_in_range(self):
        ev = script.compute_evaluation(_make_varied_combined_df())
        auc = ev["roc_auc"].iloc[0]
        assert 0.0 <= auc <= 1.0

    def test_perfect_separation_gives_auc_one(self):
        # Normal scores all above abnormal → anomaly score inverts → AUC = 1.0
        ev = script.compute_evaluation(
            _make_combined_df(n_normal=20, n_abnormal=20,
                              normal_score=100.0, abnormal_score=0.0)
        )
        assert abs(ev["roc_auc"].iloc[0] - 1.0) < 1e-9

    def test_identical_distributions_auc_near_half(self):
        ev = script.compute_evaluation(
            _make_combined_df(n_normal=20, n_abnormal=20,
                              normal_score=50.0, abnormal_score=50.0)
        )
        assert abs(ev["roc_auc"].iloc[0] - 0.5) < 0.05

    def test_well_separated_data_has_high_auc(self):
        ev = script.compute_evaluation(_make_varied_combined_df())
        assert ev["roc_auc"].iloc[0] > 0.7

    def test_anomaly_score_direction(self):
        # Higher combined_health_score → lower anomaly → should be normal
        # Lower combined_health_score → higher anomaly → should be abnormal
        # AUC must be > 0.5 for well-separated data
        ev = script.compute_evaluation(
            _make_combined_df(n_normal=30, n_abnormal=30,
                              normal_score=80.0, abnormal_score=20.0)
        )
        assert ev["roc_auc"].iloc[0] > 0.5


# ---------------------------------------------------------------------------
# save_evaluation
# ---------------------------------------------------------------------------

class TestSaveEvaluation:
    def test_creates_output_directory(self, tmp_path):
        out = tmp_path / "subdir" / "health_score_evaluation.csv"
        ev = script.compute_evaluation(_make_combined_df())
        script.save_evaluation(ev, out)
        assert out.parent.exists()

    def test_csv_file_created(self, tmp_path):
        out = tmp_path / "health_calibration" / "health_score_evaluation.csv"
        ev = script.compute_evaluation(_make_combined_df())
        script.save_evaluation(ev, out)
        assert out.exists()

    def test_csv_has_one_row(self, tmp_path):
        out = tmp_path / "health_score_evaluation.csv"
        ev = script.compute_evaluation(_make_combined_df())
        script.save_evaluation(ev, out)
        loaded = pd.read_csv(out)
        assert len(loaded) == 1

    def test_csv_has_all_columns(self, tmp_path):
        out = tmp_path / "health_score_evaluation.csv"
        ev = script.compute_evaluation(_make_combined_df())
        script.save_evaluation(ev, out)
        loaded = pd.read_csv(out)
        for col in script.EVALUATION_COLUMNS:
            assert col in loaded.columns

    def test_csv_values_preserved(self, tmp_path):
        out = tmp_path / "health_score_evaluation.csv"
        ev = script.compute_evaluation(
            _make_combined_df(normal_score=75.0, abnormal_score=30.0)
        )
        script.save_evaluation(ev, out)
        loaded = pd.read_csv(out)
        assert abs(loaded["normal_mean"].iloc[0] - 75.0) < 1e-6
        assert abs(loaded["abnormal_mean"].iloc[0] - 30.0) < 1e-6

    def test_roc_auc_preserved_in_csv(self, tmp_path):
        out = tmp_path / "health_score_evaluation.csv"
        ev = script.compute_evaluation(_make_varied_combined_df())
        script.save_evaluation(ev, out)
        loaded = pd.read_csv(out)
        assert 0.0 <= loaded["roc_auc"].iloc[0] <= 1.0


# ---------------------------------------------------------------------------
# main — integration
# ---------------------------------------------------------------------------

class TestMain:
    def _make_full_eval_df(self, n_normal: int = 20, n_abnormal: int = 15) -> pd.DataFrame:
        """Evaluation CSV with all columns required by load_csv."""
        rng = np.random.default_rng(7)
        rows = (
            [{"machine_type": "pump", "machine_id": "id_00",
              "filename": f"n_{i}.wav", "true_label": "normal",
              "normalized_euclidean": float(rng.uniform(5, 20)),
              "normalized_manhattan": float(rng.uniform(80, 250)),
              "normalized_cosine": float(rng.uniform(-0.15, 0.05))}
             for i in range(n_normal)]
            +
            [{"machine_type": "pump", "machine_id": "id_00",
              "filename": f"a_{i}.wav", "true_label": "abnormal",
              "normalized_euclidean": float(rng.uniform(40, 100)),
              "normalized_manhattan": float(rng.uniform(500, 1200)),
              "normalized_cosine": float(rng.uniform(-0.05, 0.15))}
             for i in range(n_abnormal)]
        )
        return pd.DataFrame(rows)

    def _patch(self, monkeypatch, tmp_path, n_normal=20, n_abnormal=15):
        path = tmp_path / "eval.csv"
        self._make_full_eval_df(n_normal, n_abnormal).to_csv(path, index=False)
        out = tmp_path / "health_calibration" / "health_score_evaluation.csv"
        monkeypatch.setattr(script, "INPUT_CSV", path)
        monkeypatch.setattr(script, "EVALUATION_CSV", out)
        return path, out

    def test_main_runs_without_error(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        script.main()

    def test_main_raises_on_missing_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "INPUT_CSV", tmp_path / "missing.csv")
        with pytest.raises(FileNotFoundError):
            script.main()

    def test_main_saves_evaluation_csv(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        assert out.exists()

    def test_main_csv_has_one_row(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        loaded = pd.read_csv(out)
        assert len(loaded) == 1

    def test_main_csv_has_all_columns(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        loaded = pd.read_csv(out)
        for col in script.EVALUATION_COLUMNS:
            assert col in loaded.columns

    def test_main_csv_roc_auc_in_range(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        loaded = pd.read_csv(out)
        assert 0.0 <= loaded["roc_auc"].iloc[0] <= 1.0

    def test_main_csv_p_value_in_range(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        loaded = pd.read_csv(out)
        assert 0.0 <= loaded["p_value"].iloc[0] <= 1.0

    def test_main_csv_rbc_in_range(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        loaded = pd.read_csv(out)
        assert -1.0 <= loaded["rank_biserial_correlation"].iloc[0] <= 1.0

    def test_main_prints_experiment_id(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch, tmp_path)
        script.main()
        assert "E1" in capsys.readouterr().out

    def test_main_prints_roc_auc(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch, tmp_path)
        script.main()
        assert "ROC-AUC" in capsys.readouterr().out

    def test_main_prints_p_value(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch, tmp_path)
        script.main()
        assert "p-value" in capsys.readouterr().out

    def test_main_prints_effect_size(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch, tmp_path)
        script.main()
        assert "Effect size" in capsys.readouterr().out
