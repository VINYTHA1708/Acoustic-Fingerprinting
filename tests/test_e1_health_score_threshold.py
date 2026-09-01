"""Tests for experiments/e1_health_score_threshold.py (Phase 6.5).

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

import experiments.e1_health_score_threshold as script


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
        [{"true_label": "normal",   script.COMBINED_SCORE_COLUMN: normal_score}
         for _ in range(n_normal)]
        + [{"true_label": "abnormal", script.COMBINED_SCORE_COLUMN: abnormal_score}
           for _ in range(n_abnormal)]
    )
    return pd.DataFrame(rows)


def _make_varied_df(seed: int = 0) -> pd.DataFrame:
    """Varied scores with clear separation for non-trivial metric tests."""
    rng = np.random.default_rng(seed)
    normal_scores   = rng.uniform(55, 95, size=30).tolist()
    abnormal_scores = rng.uniform(10, 50, size=20).tolist()
    rows = (
        [{"true_label": "normal",   script.COMBINED_SCORE_COLUMN: s} for s in normal_scores]
        + [{"true_label": "abnormal", script.COMBINED_SCORE_COLUMN: s} for s in abnormal_scores]
    )
    return pd.DataFrame(rows)


def _make_full_eval_df(n_normal: int = 20, n_abnormal: int = 15, seed: int = 7) -> pd.DataFrame:
    """Full evaluation CSV with drift columns required by load_csv."""
    rng = np.random.default_rng(seed)
    rows = (
        [{"machine_type": "pump", "machine_id": "id_00",
          "filename": f"n_{i}.wav", "true_label": "normal",
          "normalized_euclidean": float(rng.uniform(5, 20)),
          "normalized_manhattan": float(rng.uniform(80, 250)),
          "normalized_cosine":    float(rng.uniform(-0.15, 0.05))}
         for i in range(n_normal)]
        + [{"machine_type": "pump", "machine_id": "id_00",
            "filename": f"a_{i}.wav", "true_label": "abnormal",
            "normalized_euclidean": float(rng.uniform(40, 100)),
            "normalized_manhattan": float(rng.uniform(500, 1200)),
            "normalized_cosine":    float(rng.uniform(-0.05, 0.15))}
           for i in range(n_abnormal)]
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# validate_df
# ---------------------------------------------------------------------------

class TestValidateDF:
    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            script.validate_df(_make_combined_df().iloc[0:0])

    def test_missing_true_label_raises(self):
        df = _make_combined_df().drop(columns=["true_label"])
        with pytest.raises(ValueError, match="Missing required columns"):
            script.validate_df(df)

    def test_missing_score_column_raises(self):
        df = _make_combined_df().drop(columns=[script.COMBINED_SCORE_COLUMN])
        with pytest.raises(ValueError, match="Missing required columns"):
            script.validate_df(df)

    def test_non_numeric_score_raises(self):
        df = _make_combined_df()
        df[script.COMBINED_SCORE_COLUMN] = "bad"
        with pytest.raises(ValueError, match="not numeric"):
            script.validate_df(df)

    def test_valid_df_does_not_raise(self):
        script.validate_df(_make_combined_df())


# ---------------------------------------------------------------------------
# find_optimal_threshold — basic properties
# ---------------------------------------------------------------------------

class TestFindOptimalThreshold:
    def test_returns_float(self):
        t = script.find_optimal_threshold(_make_varied_df())
        assert isinstance(t, float)

    def test_threshold_in_score_range(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        lo = float(df[script.COMBINED_SCORE_COLUMN].min())
        hi = float(df[script.COMBINED_SCORE_COLUMN].max())
        # Allow a small margin beyond the observed range (ROC thresholds can
        # extend slightly outside the data range)
        assert lo - 1.0 <= t <= hi + 1.0

    def test_threshold_between_group_means(self):
        # With perfectly separated groups the threshold must lie between them
        df = _make_combined_df(normal_score=80.0, abnormal_score=20.0)
        t = script.find_optimal_threshold(df)
        assert 20.0 <= t <= 80.0

    def test_missing_normal_raises(self):
        df = _make_combined_df(n_normal=0, n_abnormal=10)
        with pytest.raises(ValueError, match="normal"):
            script.find_optimal_threshold(df)

    def test_missing_abnormal_raises(self):
        df = _make_combined_df(n_normal=10, n_abnormal=0)
        with pytest.raises(ValueError, match="abnormal"):
            script.find_optimal_threshold(df)

    def test_deterministic_for_same_input(self):
        df = _make_varied_df()
        assert script.find_optimal_threshold(df) == script.find_optimal_threshold(df)


# ---------------------------------------------------------------------------
# compute_threshold_metrics — output shape and columns
# ---------------------------------------------------------------------------

class TestComputeThresholdMetricsShape:
    def test_returns_single_row(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        result = script.compute_threshold_metrics(df, t)
        assert len(result) == 1

    def test_all_expected_columns_present(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        result = script.compute_threshold_metrics(df, t)
        for col in script.THRESHOLD_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_extra_columns(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        result = script.compute_threshold_metrics(df, t)
        assert list(result.columns) == script.THRESHOLD_COLUMNS

    def test_threshold_stored_correctly(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        result = script.compute_threshold_metrics(df, t)
        assert abs(result["threshold"].iloc[0] - t) < 1e-9


# ---------------------------------------------------------------------------
# compute_threshold_metrics — metric bounds
# ---------------------------------------------------------------------------

class TestComputeThresholdMetricsBounds:
    def test_accuracy_in_range(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        r = script.compute_threshold_metrics(df, t)
        assert 0.0 <= r["accuracy"].iloc[0] <= 1.0

    def test_precision_in_range(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        r = script.compute_threshold_metrics(df, t)
        assert 0.0 <= r["precision"].iloc[0] <= 1.0

    def test_recall_in_range(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        r = script.compute_threshold_metrics(df, t)
        assert 0.0 <= r["recall"].iloc[0] <= 1.0

    def test_f1_in_range(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        r = script.compute_threshold_metrics(df, t)
        assert 0.0 <= r["f1_score"].iloc[0] <= 1.0

    def test_sensitivity_in_range(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        r = script.compute_threshold_metrics(df, t)
        assert 0.0 <= r["sensitivity"].iloc[0] <= 1.0

    def test_specificity_in_range(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        r = script.compute_threshold_metrics(df, t)
        assert 0.0 <= r["specificity"].iloc[0] <= 1.0


# ---------------------------------------------------------------------------
# compute_threshold_metrics — confusion matrix consistency
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_cm_values_non_negative(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        r = script.compute_threshold_metrics(df, t)
        for col in ("tp", "fp", "tn", "fn"):
            assert r[col].iloc[0] >= 0

    def test_cm_sums_to_total(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        r = script.compute_threshold_metrics(df, t)
        total = int(r["tp"].iloc[0]) + int(r["fp"].iloc[0]) + \
                int(r["tn"].iloc[0]) + int(r["fn"].iloc[0])
        assert total == len(df)

    def test_perfect_separation_cm(self):
        # Use an explicit midpoint threshold so separation is guaranteed
        df = _make_combined_df(n_normal=20, n_abnormal=20,
                               normal_score=80.0, abnormal_score=20.0)
        r = script.compute_threshold_metrics(df, threshold=50.0)
        assert int(r["fp"].iloc[0]) == 0
        assert int(r["fn"].iloc[0]) == 0

    def test_sensitivity_equals_recall(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        r = script.compute_threshold_metrics(df, t)
        assert abs(r["sensitivity"].iloc[0] - r["recall"].iloc[0]) < 1e-9

    def test_specificity_formula(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        r = script.compute_threshold_metrics(df, t)
        tn, fp = int(r["tn"].iloc[0]), int(r["fp"].iloc[0])
        expected = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        assert abs(r["specificity"].iloc[0] - expected) < 1e-9

    def test_accuracy_formula(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        r = script.compute_threshold_metrics(df, t)
        tp, fp = int(r["tp"].iloc[0]), int(r["fp"].iloc[0])
        tn, fn = int(r["tn"].iloc[0]), int(r["fn"].iloc[0])
        expected = (tp + tn) / (tp + fp + tn + fn)
        assert abs(r["accuracy"].iloc[0] - expected) < 1e-9


# ---------------------------------------------------------------------------
# compute_threshold_metrics — well-separated data quality
# ---------------------------------------------------------------------------

class TestThresholdQuality:
    def test_well_separated_accuracy_above_half(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        r = script.compute_threshold_metrics(df, t)
        assert r["accuracy"].iloc[0] > 0.5

    def test_well_separated_f1_above_half(self):
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        r = script.compute_threshold_metrics(df, t)
        assert r["f1_score"].iloc[0] > 0.5

    def test_youden_threshold_maximises_j_statistic(self):
        # Youden's J must be >= J at every other ROC-derived health-score threshold
        from sklearn.metrics import roc_curve
        df = _make_varied_df()
        t_youden = script.find_optimal_threshold(df)
        r_youden = script.compute_threshold_metrics(df, t_youden)
        j_youden = r_youden["sensitivity"].iloc[0] + r_youden["specificity"].iloc[0] - 1.0

        binary_labels = (df["true_label"] == "abnormal").astype(int)
        anomaly_scores = 100.0 - df[script.COMBINED_SCORE_COLUMN]
        fpr, tpr, roc_thresholds = roc_curve(binary_labels, anomaly_scores)
        for anomaly_t in roc_thresholds:
            health_t = 100.0 - float(anomaly_t)
            r = script.compute_threshold_metrics(df, health_t)
            j = r["sensitivity"].iloc[0] + r["specificity"].iloc[0] - 1.0
            assert j_youden >= j - 1e-6, (
                f"Youden J={j_youden:.6f} < J={j:.6f} at threshold={health_t:.4f}"
            )

    def test_custom_threshold_all_predicted_normal(self):
        # Threshold below all scores → everything predicted normal → TP=0, FP=0
        df = _make_combined_df(normal_score=80.0, abnormal_score=20.0)
        r = script.compute_threshold_metrics(df, threshold=0.0)
        assert int(r["tp"].iloc[0]) == 0
        assert int(r["fp"].iloc[0]) == 0

    def test_custom_threshold_all_predicted_abnormal(self):
        # Threshold above all scores → everything predicted abnormal → TN=0, FN=0
        df = _make_combined_df(normal_score=80.0, abnormal_score=20.0)
        r = script.compute_threshold_metrics(df, threshold=100.1)
        assert int(r["tn"].iloc[0]) == 0
        assert int(r["fn"].iloc[0]) == 0


# ---------------------------------------------------------------------------
# save_threshold_results
# ---------------------------------------------------------------------------

class TestSaveThresholdResults:
    def test_creates_output_directory(self, tmp_path):
        out = tmp_path / "subdir" / "health_score_threshold.csv"
        df = _make_varied_df()
        r = script.compute_threshold_metrics(df, script.find_optimal_threshold(df))
        script.save_threshold_results(r, out)
        assert out.parent.exists()

    def test_csv_file_created(self, tmp_path):
        out = tmp_path / "health_calibration" / "health_score_threshold.csv"
        df = _make_varied_df()
        r = script.compute_threshold_metrics(df, script.find_optimal_threshold(df))
        script.save_threshold_results(r, out)
        assert out.exists()

    def test_csv_has_one_row(self, tmp_path):
        out = tmp_path / "health_score_threshold.csv"
        df = _make_varied_df()
        r = script.compute_threshold_metrics(df, script.find_optimal_threshold(df))
        script.save_threshold_results(r, out)
        assert len(pd.read_csv(out)) == 1

    def test_csv_has_all_columns(self, tmp_path):
        out = tmp_path / "health_score_threshold.csv"
        df = _make_varied_df()
        r = script.compute_threshold_metrics(df, script.find_optimal_threshold(df))
        script.save_threshold_results(r, out)
        loaded = pd.read_csv(out)
        for col in script.THRESHOLD_COLUMNS:
            assert col in loaded.columns

    def test_csv_values_preserved(self, tmp_path):
        out = tmp_path / "health_score_threshold.csv"
        df = _make_varied_df()
        t = script.find_optimal_threshold(df)
        r = script.compute_threshold_metrics(df, t)
        script.save_threshold_results(r, out)
        loaded = pd.read_csv(out)
        assert abs(loaded["threshold"].iloc[0] - t) < 1e-6
        assert 0.0 <= loaded["accuracy"].iloc[0] <= 1.0


# ---------------------------------------------------------------------------
# main — integration
# ---------------------------------------------------------------------------

class TestMain:
    def _patch(self, monkeypatch, tmp_path, n_normal=20, n_abnormal=15):
        path = tmp_path / "eval.csv"
        _make_full_eval_df(n_normal, n_abnormal).to_csv(path, index=False)
        out = tmp_path / "health_calibration" / "health_score_threshold.csv"
        monkeypatch.setattr(script, "INPUT_CSV", path)
        monkeypatch.setattr(script, "THRESHOLD_CSV", out)
        return path, out

    def test_main_runs_without_error(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        script.main()

    def test_main_raises_on_missing_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "INPUT_CSV", tmp_path / "missing.csv")
        with pytest.raises(FileNotFoundError):
            script.main()

    def test_main_saves_threshold_csv(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        assert out.exists()

    def test_main_csv_has_one_row(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        assert len(pd.read_csv(out)) == 1

    def test_main_csv_has_all_columns(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        loaded = pd.read_csv(out)
        for col in script.THRESHOLD_COLUMNS:
            assert col in loaded.columns

    def test_main_csv_metrics_in_range(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path)
        script.main()
        loaded = pd.read_csv(out)
        for col in ("accuracy", "precision", "recall", "f1_score",
                    "sensitivity", "specificity"):
            assert 0.0 <= loaded[col].iloc[0] <= 1.0, f"{col} out of range"

    def test_main_cm_sums_to_total(self, tmp_path, monkeypatch):
        _, out = self._patch(monkeypatch, tmp_path, n_normal=20, n_abnormal=15)
        script.main()
        loaded = pd.read_csv(out)
        total = sum(int(loaded[c].iloc[0]) for c in ("tp", "fp", "tn", "fn"))
        assert total == 35

    def test_main_prints_threshold(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch, tmp_path)
        script.main()
        assert "Threshold" in capsys.readouterr().out

    def test_main_prints_accuracy(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch, tmp_path)
        script.main()
        assert "Accuracy" in capsys.readouterr().out

    def test_main_prints_confusion_matrix(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch, tmp_path)
        script.main()
        assert "Confusion Matrix" in capsys.readouterr().out

    def test_main_prints_sensitivity_and_specificity(self, tmp_path, monkeypatch, capsys):
        self._patch(monkeypatch, tmp_path)
        script.main()
        out = capsys.readouterr().out
        assert "Sensitivity" in out
        assert "Specificity" in out
