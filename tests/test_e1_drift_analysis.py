"""Lightweight tests for experiments/e1_drift_analysis.py.

No BEATs, no audio files, no MIMII dataset required.
Uses synthetic DataFrames only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.e1_drift_analysis as script


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n_normal: int = 10,
    n_abnormal: int = 8,
    include_cols: list[str] | None = None,
    machine_id: str = "id_00",
) -> pd.DataFrame:
    """Minimal valid evaluation DataFrame."""
    rows = (
        [{"machine_type": "pump", "machine_id": machine_id, "filename": f"n_{i}.wav",
          "true_label": "normal", "health_score": 90.0, "health_percentage": "90.0%",
          "health_state": "EXCELLENT", "normalized_euclidean": 10.0,
          "normalized_manhattan": 150.0, "normalized_cosine": -0.05}
         for i in range(n_normal)]
        +
        [{"machine_type": "pump", "machine_id": machine_id, "filename": f"a_{i}.wav",
          "true_label": "abnormal", "health_score": 40.0, "health_percentage": "40.0%",
          "health_state": "CRITICAL", "normalized_euclidean": 40.0,
          "normalized_manhattan": 500.0, "normalized_cosine": 0.05}
         for i in range(n_abnormal)]
    )
    df = pd.DataFrame(rows)
    if include_cols is not None:
        df = df[include_cols]
    return df


def _make_varied_df() -> pd.DataFrame:
    """DataFrame with varied drift values so Mann-Whitney is non-trivial."""
    import numpy as np
    rng = np.random.default_rng(42)
    normal_rows = [
        {"machine_type": "pump", "machine_id": "id_00", "filename": f"n_{i}.wav",
         "true_label": "normal", "health_score": 90.0, "health_percentage": "90.0%",
         "health_state": "EXCELLENT",
         "normalized_euclidean": float(rng.uniform(5, 20)),
         "normalized_manhattan": float(rng.uniform(80, 250)),
         "normalized_cosine": float(rng.uniform(-0.15, 0.05))}
        for i in range(30)
    ]
    abnormal_rows = [
        {"machine_type": "pump", "machine_id": "id_00", "filename": f"a_{i}.wav",
         "true_label": "abnormal", "health_score": 35.0, "health_percentage": "35.0%",
         "health_state": "CRITICAL",
         "normalized_euclidean": float(rng.uniform(40, 100)),
         "normalized_manhattan": float(rng.uniform(500, 1200)),
         "normalized_cosine": float(rng.uniform(-0.05, 0.15))}
        for i in range(20)
    ]
    return pd.DataFrame(normal_rows + abnormal_rows)


def _make_multi_machine_df() -> pd.DataFrame:
    """Two machine IDs with distinct drift values for per-machine tests."""
    rows = [
        # id_00 normal: euclidean=10, id_02 normal: euclidean=20
        *[{"machine_type": "pump", "machine_id": "id_00", "filename": f"n_{i}.wav",
           "true_label": "normal", "health_score": 90.0, "health_percentage": "90.0%",
           "health_state": "EXCELLENT", "normalized_euclidean": 10.0,
           "normalized_manhattan": 150.0, "normalized_cosine": -0.05}
          for i in range(6)],
        *[{"machine_type": "pump", "machine_id": "id_00", "filename": f"a_{i}.wav",
           "true_label": "abnormal", "health_score": 40.0, "health_percentage": "40.0%",
           "health_state": "CRITICAL", "normalized_euclidean": 40.0,
           "normalized_manhattan": 500.0, "normalized_cosine": 0.05}
          for i in range(4)],
        *[{"machine_type": "pump", "machine_id": "id_02", "filename": f"n_{i}.wav",
           "true_label": "normal", "health_score": 85.0, "health_percentage": "85.0%",
           "health_state": "GOOD", "normalized_euclidean": 20.0,
           "normalized_manhattan": 250.0, "normalized_cosine": -0.03}
          for i in range(5)],
        *[{"machine_type": "pump", "machine_id": "id_02", "filename": f"a_{i}.wav",
           "true_label": "abnormal", "health_score": 35.0, "health_percentage": "35.0%",
           "health_state": "CRITICAL", "normalized_euclidean": 60.0,
           "normalized_manhattan": 700.0, "normalized_cosine": 0.08}
          for i in range(3)],
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# validate_csv — required columns
# ---------------------------------------------------------------------------

class TestRequiredColumns:
    def test_missing_column_raises(self):
        df = _make_df().drop(columns=["normalized_euclidean"])
        with pytest.raises(ValueError, match="Missing required columns"):
            script.validate_csv(df)

    def test_multiple_missing_columns_raises(self):
        df = _make_df().drop(columns=["normalized_euclidean", "normalized_manhattan"])
        with pytest.raises(ValueError, match="Missing required columns"):
            script.validate_csv(df)

    def test_all_columns_present_does_not_raise(self):
        script.validate_csv(_make_df())


# ---------------------------------------------------------------------------
# validate_csv — empty DataFrame
# ---------------------------------------------------------------------------

class TestEmptyDataFrame:
    def test_empty_df_raises(self):
        df = _make_df()
        empty = df.iloc[0:0]
        with pytest.raises(ValueError, match="CSV is empty"):
            script.validate_csv(empty)


# ---------------------------------------------------------------------------
# validate_csv — true_label values
# ---------------------------------------------------------------------------

class TestLabelValidation:
    def test_unexpected_label_raises(self):
        df = _make_df()
        df.loc[0, "true_label"] = "unknown"
        with pytest.raises(ValueError, match="Unexpected true_label"):
            script.validate_csv(df)

    def test_only_normal_is_valid(self):
        df = _make_df(n_abnormal=0)
        script.validate_csv(df)

    def test_only_abnormal_is_valid(self):
        df = _make_df(n_normal=0)
        script.validate_csv(df)

    def test_both_labels_valid(self):
        script.validate_csv(_make_df())


# ---------------------------------------------------------------------------
# validate_csv — numeric drift columns
# ---------------------------------------------------------------------------

class TestNumericColumns:
    def test_non_numeric_euclidean_raises(self):
        df = _make_df()
        df["normalized_euclidean"] = "not_a_number"
        with pytest.raises(ValueError, match="not numeric"):
            script.validate_csv(df)

    def test_non_numeric_manhattan_raises(self):
        df = _make_df()
        df["normalized_manhattan"] = "bad"
        with pytest.raises(ValueError, match="not numeric"):
            script.validate_csv(df)

    def test_non_numeric_cosine_raises(self):
        df = _make_df()
        df["normalized_cosine"] = "bad"
        with pytest.raises(ValueError, match="not numeric"):
            script.validate_csv(df)

    def test_numeric_columns_accepted(self):
        df = _make_df()
        for col in script.DRIFT_METRICS:
            assert pd.api.types.is_numeric_dtype(df[col])
        script.validate_csv(df)


# ---------------------------------------------------------------------------
# load_csv — missing file
# ---------------------------------------------------------------------------

class TestLoadCSV:
    def test_missing_file_raises_file_not_found(self, tmp_path):
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
        # Write a CSV missing required columns
        pd.DataFrame({"col_a": [1, 2]}).to_csv(path, index=False)
        with pytest.raises(ValueError):
            script.load_csv(path)


# ---------------------------------------------------------------------------
# print_dataset_summary — counts and machine IDs
# ---------------------------------------------------------------------------

class TestPrintDatasetSummary:
    def test_runs_without_error(self, capsys):
        df = _make_df()
        script.print_dataset_summary(df)
        out = capsys.readouterr().out
        assert "10" in out   # normal count
        assert "8" in out    # abnormal count
        assert "id_00" in out

    def test_shows_correct_total(self, capsys):
        df = _make_df(n_normal=5, n_abnormal=3)
        script.print_dataset_summary(df)
        out = capsys.readouterr().out
        assert "8" in out    # total

    def test_shows_all_machine_ids(self, capsys):
        rows = []
        for mid in ["id_00", "id_02", "id_04"]:
            rows.append({"machine_type": "pump", "machine_id": mid,
                         "filename": "x.wav", "true_label": "normal",
                         "health_score": 90.0, "health_percentage": "90.0%",
                         "health_state": "GOOD", "normalized_euclidean": 10.0,
                         "normalized_manhattan": 150.0, "normalized_cosine": -0.05})
        df = pd.DataFrame(rows)
        script.print_dataset_summary(df)
        out = capsys.readouterr().out
        assert "id_00" in out
        assert "id_02" in out
        assert "id_04" in out


# ---------------------------------------------------------------------------
# print_header — content
# ---------------------------------------------------------------------------

class TestPrintHeader:
    def test_experiment_id_in_output(self, capsys):
        script.print_header()
        out = capsys.readouterr().out
        assert "E1" in out

    def test_stage_in_output(self, capsys):
        script.print_header()
        out = capsys.readouterr().out
        assert "Drift" in out


# ---------------------------------------------------------------------------
# compute_overall_drift_statistics
# ---------------------------------------------------------------------------

class TestComputeOverallDriftStatistics:
    def test_returns_six_rows(self):
        # 3 metrics × 2 labels = 6 rows
        stats = script.compute_overall_drift_statistics(_make_df())
        assert len(stats) == 6

    def test_columns_present(self):
        stats = script.compute_overall_drift_statistics(_make_df())
        for col in ("metric", "label", "count", "mean", "std", "median", "min", "max", "q1", "q3"):
            assert col in stats.columns

    def test_normal_abnormal_separation(self):
        stats = script.compute_overall_drift_statistics(_make_df())
        for metric in script.DRIFT_METRICS:
            normal_mean = stats.loc[
                (stats["metric"] == metric) & (stats["label"] == "normal"), "mean"
            ].iloc[0]
            abnormal_mean = stats.loc[
                (stats["metric"] == metric) & (stats["label"] == "abnormal"), "mean"
            ].iloc[0]
            # abnormal values are higher in _make_df (40 vs 10, 500 vs 150, 0.05 vs -0.05)
            assert abnormal_mean > normal_mean

    def test_correct_count(self):
        stats = script.compute_overall_drift_statistics(_make_df(n_normal=10, n_abnormal=8))
        normal_count = stats.loc[
            (stats["metric"] == "normalized_euclidean") & (stats["label"] == "normal"), "count"
        ].iloc[0]
        assert int(normal_count) == 10

    def test_correct_mean(self):
        stats = script.compute_overall_drift_statistics(_make_df())
        # All normal euclidean values are 10.0
        mean_val = stats.loc[
            (stats["metric"] == "normalized_euclidean") & (stats["label"] == "normal"), "mean"
        ].iloc[0]
        assert abs(mean_val - 10.0) < 1e-9

    def test_std_zero_for_constant_values(self):
        stats = script.compute_overall_drift_statistics(_make_df())
        std_val = stats.loc[
            (stats["metric"] == "normalized_euclidean") & (stats["label"] == "normal"), "std"
        ].iloc[0]
        assert abs(std_val) < 1e-9

    def test_all_three_metrics_present(self):
        stats = script.compute_overall_drift_statistics(_make_df())
        assert set(stats["metric"].unique()) == set(script.DRIFT_METRICS)

    def test_both_labels_present(self):
        stats = script.compute_overall_drift_statistics(_make_df())
        assert set(stats["label"].unique()) == {"normal", "abnormal"}


# ---------------------------------------------------------------------------
# save_overall_drift_statistics
# ---------------------------------------------------------------------------

class TestSaveOverallDriftStatistics:
    def test_creates_output_directory(self, tmp_path):
        out = tmp_path / "subdir" / "stats.csv"
        stats = script.compute_overall_drift_statistics(_make_df())
        script.save_overall_drift_statistics(stats, out)
        assert out.parent.exists()

    def test_csv_file_created(self, tmp_path):
        out = tmp_path / "drift_analysis" / "overall_drift_statistics.csv"
        stats = script.compute_overall_drift_statistics(_make_df())
        script.save_overall_drift_statistics(stats, out)
        assert out.exists()

    def test_csv_round_trip(self, tmp_path):
        out = tmp_path / "stats.csv"
        stats = script.compute_overall_drift_statistics(_make_df())
        script.save_overall_drift_statistics(stats, out)
        loaded = pd.read_csv(out)
        assert len(loaded) == 6
        for col in ("metric", "label", "count", "mean", "std", "median", "min", "max", "q1", "q3"):
            assert col in loaded.columns

    def test_csv_has_correct_row_count(self, tmp_path):
        out = tmp_path / "stats.csv"
        stats = script.compute_overall_drift_statistics(_make_df())
        script.save_overall_drift_statistics(stats, out)
        loaded = pd.read_csv(out)
        assert len(loaded) == 6  # 3 metrics × 2 labels


# ---------------------------------------------------------------------------
# compute_per_machine_drift_statistics
# ---------------------------------------------------------------------------

class TestComputePerMachineDriftStatistics:
    def test_row_count_two_machines(self):
        # 2 machines × 3 metrics × 2 labels = 12 rows
        stats = script.compute_per_machine_drift_statistics(_make_multi_machine_df())
        assert len(stats) == 12

    def test_row_count_one_machine(self):
        # 1 machine × 3 metrics × 2 labels = 6 rows
        stats = script.compute_per_machine_drift_statistics(_make_df())
        assert len(stats) == 6

    def test_columns_present(self):
        stats = script.compute_per_machine_drift_statistics(_make_multi_machine_df())
        for col in ("machine_id", "metric", "label", "count", "mean", "std",
                    "median", "min", "max", "q1", "q3"):
            assert col in stats.columns

    def test_all_machine_ids_present(self):
        stats = script.compute_per_machine_drift_statistics(_make_multi_machine_df())
        assert set(stats["machine_id"].unique()) == {"id_00", "id_02"}

    def test_all_metrics_present(self):
        stats = script.compute_per_machine_drift_statistics(_make_multi_machine_df())
        assert set(stats["metric"].unique()) == set(script.DRIFT_METRICS)

    def test_both_labels_present(self):
        stats = script.compute_per_machine_drift_statistics(_make_multi_machine_df())
        assert set(stats["label"].unique()) == {"normal", "abnormal"}

    def test_correct_mean_per_machine(self):
        stats = script.compute_per_machine_drift_statistics(_make_multi_machine_df())
        mean_id00 = stats.loc[
            (stats["machine_id"] == "id_00") &
            (stats["metric"] == "normalized_euclidean") &
            (stats["label"] == "normal"), "mean"
        ].iloc[0]
        mean_id02 = stats.loc[
            (stats["machine_id"] == "id_02") &
            (stats["metric"] == "normalized_euclidean") &
            (stats["label"] == "normal"), "mean"
        ].iloc[0]
        assert abs(mean_id00 - 10.0) < 1e-9
        assert abs(mean_id02 - 20.0) < 1e-9

    def test_correct_count_per_machine(self):
        stats = script.compute_per_machine_drift_statistics(_make_multi_machine_df())
        count_id00_normal = stats.loc[
            (stats["machine_id"] == "id_00") &
            (stats["metric"] == "normalized_euclidean") &
            (stats["label"] == "normal"), "count"
        ].iloc[0]
        assert int(count_id00_normal) == 6

    def test_machines_are_independent(self):
        # id_00 normal mean != id_02 normal mean
        stats = script.compute_per_machine_drift_statistics(_make_multi_machine_df())
        m00 = stats.loc[
            (stats["machine_id"] == "id_00") &
            (stats["metric"] == "normalized_euclidean") &
            (stats["label"] == "normal"), "mean"
        ].iloc[0]
        m02 = stats.loc[
            (stats["machine_id"] == "id_02") &
            (stats["metric"] == "normalized_euclidean") &
            (stats["label"] == "normal"), "mean"
        ].iloc[0]
        assert m00 != m02

    def test_skips_missing_label_for_machine(self):
        # A machine with only normal recordings should produce 3 rows (no abnormal)
        df = _make_df(n_normal=5, n_abnormal=0)
        stats = script.compute_per_machine_drift_statistics(df)
        assert len(stats) == 3
        assert set(stats["label"].unique()) == {"normal"}


# ---------------------------------------------------------------------------
# save_per_machine_drift_statistics
# ---------------------------------------------------------------------------

class TestSavePerMachineDriftStatistics:
    def test_creates_output_directory(self, tmp_path):
        out = tmp_path / "subdir" / "per_machine.csv"
        stats = script.compute_per_machine_drift_statistics(_make_multi_machine_df())
        script.save_per_machine_drift_statistics(stats, out)
        assert out.parent.exists()

    def test_csv_file_created(self, tmp_path):
        out = tmp_path / "drift_analysis" / "per_machine_drift_statistics.csv"
        stats = script.compute_per_machine_drift_statistics(_make_multi_machine_df())
        script.save_per_machine_drift_statistics(stats, out)
        assert out.exists()

    def test_csv_round_trip_columns(self, tmp_path):
        out = tmp_path / "per_machine.csv"
        stats = script.compute_per_machine_drift_statistics(_make_multi_machine_df())
        script.save_per_machine_drift_statistics(stats, out)
        loaded = pd.read_csv(out)
        for col in ("machine_id", "metric", "label", "count", "mean", "std",
                    "median", "min", "max", "q1", "q3"):
            assert col in loaded.columns

    def test_csv_row_count(self, tmp_path):
        out = tmp_path / "per_machine.csv"
        stats = script.compute_per_machine_drift_statistics(_make_multi_machine_df())
        script.save_per_machine_drift_statistics(stats, out)
        loaded = pd.read_csv(out)
        assert len(loaded) == 12  # 2 machines × 3 metrics × 2 labels


# ---------------------------------------------------------------------------
# compute_overall_significance
# ---------------------------------------------------------------------------

class TestComputeOverallSignificance:
    def test_returns_three_rows(self):
        sig = script.compute_overall_significance(_make_varied_df())
        assert len(sig) == 3

    def test_columns_present(self):
        sig = script.compute_overall_significance(_make_varied_df())
        for col in ("metric", "n_normal", "n_abnormal", "u_statistic", "p_value"):
            assert col in sig.columns

    def test_all_metrics_present(self):
        sig = script.compute_overall_significance(_make_varied_df())
        assert set(sig["metric"].unique()) == set(script.DRIFT_METRICS)

    def test_p_value_in_range(self):
        sig = script.compute_overall_significance(_make_varied_df())
        assert (sig["p_value"] >= 0).all()
        assert (sig["p_value"] <= 1).all()

    def test_u_statistic_non_negative(self):
        sig = script.compute_overall_significance(_make_varied_df())
        assert (sig["u_statistic"] >= 0).all()

    def test_correct_sample_counts(self):
        sig = script.compute_overall_significance(_make_varied_df())
        row = sig[sig["metric"] == "normalized_euclidean"].iloc[0]
        assert int(row["n_normal"]) == 30
        assert int(row["n_abnormal"]) == 20

    def test_well_separated_data_is_significant(self):
        # Clearly separated distributions must yield p < 0.05
        sig = script.compute_overall_significance(_make_varied_df())
        euclid_p = sig.loc[sig["metric"] == "normalized_euclidean", "p_value"].iloc[0]
        assert euclid_p < 0.05

    def test_constant_values_produce_finite_result(self):
        # All-same values: Mann-Whitney should still return without error
        sig = script.compute_overall_significance(_make_df())
        assert len(sig) == 3
        assert sig["p_value"].notna().all()


# ---------------------------------------------------------------------------
# compute_per_machine_significance
# ---------------------------------------------------------------------------

class TestComputePerMachineSignificance:
    def test_row_count_two_machines(self):
        # 2 machines × 3 metrics = 6 rows
        sig = script.compute_per_machine_significance(_make_multi_machine_df())
        assert len(sig) == 6

    def test_row_count_one_machine(self):
        sig = script.compute_per_machine_significance(_make_varied_df())
        assert len(sig) == 3

    def test_columns_present(self):
        sig = script.compute_per_machine_significance(_make_multi_machine_df())
        for col in ("machine_id", "metric", "n_normal", "n_abnormal", "u_statistic", "p_value"):
            assert col in sig.columns

    def test_all_machine_ids_present(self):
        sig = script.compute_per_machine_significance(_make_multi_machine_df())
        assert set(sig["machine_id"].unique()) == {"id_00", "id_02"}

    def test_all_metrics_present(self):
        sig = script.compute_per_machine_significance(_make_multi_machine_df())
        assert set(sig["metric"].unique()) == set(script.DRIFT_METRICS)

    def test_p_value_in_range(self):
        sig = script.compute_per_machine_significance(_make_multi_machine_df())
        assert (sig["p_value"] >= 0).all()
        assert (sig["p_value"] <= 1).all()

    def test_u_statistic_non_negative(self):
        sig = script.compute_per_machine_significance(_make_multi_machine_df())
        assert (sig["u_statistic"] >= 0).all()

    def test_skips_machine_missing_a_label(self):
        # Machine with only normal recordings: no row produced
        df = _make_df(n_normal=5, n_abnormal=0)
        sig = script.compute_per_machine_significance(df)
        assert len(sig) == 0

    def test_well_separated_data_is_significant(self):
        sig = script.compute_per_machine_significance(_make_varied_df())
        euclid_p = sig.loc[sig["metric"] == "normalized_euclidean", "p_value"].iloc[0]
        assert euclid_p < 0.05


# ---------------------------------------------------------------------------
# save_significance
# ---------------------------------------------------------------------------

class TestSaveSignificance:
    def test_creates_output_directory(self, tmp_path):
        out = tmp_path / "subdir" / "sig.csv"
        sig = script.compute_overall_significance(_make_varied_df())
        script.save_significance(sig, out)
        assert out.parent.exists()

    def test_csv_file_created(self, tmp_path):
        out = tmp_path / "overall_significance.csv"
        sig = script.compute_overall_significance(_make_varied_df())
        script.save_significance(sig, out)
        assert out.exists()

    def test_overall_csv_round_trip(self, tmp_path):
        out = tmp_path / "overall_significance.csv"
        sig = script.compute_overall_significance(_make_varied_df())
        script.save_significance(sig, out)
        loaded = pd.read_csv(out)
        assert len(loaded) == 3
        for col in ("metric", "n_normal", "n_abnormal", "u_statistic", "p_value"):
            assert col in loaded.columns

    def test_per_machine_csv_round_trip(self, tmp_path):
        out = tmp_path / "per_machine_significance.csv"
        sig = script.compute_per_machine_significance(_make_multi_machine_df())
        script.save_significance(sig, out)
        loaded = pd.read_csv(out)
        assert len(loaded) == 6
        for col in ("machine_id", "metric", "n_normal", "n_abnormal", "u_statistic", "p_value"):
            assert col in loaded.columns


# ---------------------------------------------------------------------------
# main — integration
# ---------------------------------------------------------------------------

class TestMain:
    def _patch(self, monkeypatch, tmp_path):
        path = tmp_path / "eval.csv"
        _make_varied_df().to_csv(path, index=False)
        da = tmp_path / "drift_analysis"
        monkeypatch.setattr(script, "INPUT_CSV", path)
        monkeypatch.setattr(script, "OVERALL_STATS_CSV", da / "overall_drift_statistics.csv")
        monkeypatch.setattr(script, "PER_MACHINE_STATS_CSV", da / "per_machine_drift_statistics.csv")
        monkeypatch.setattr(script, "OVERALL_SIG_CSV", da / "overall_significance.csv")
        monkeypatch.setattr(script, "PER_MACHINE_SIG_CSV", da / "per_machine_significance.csv")
        monkeypatch.setattr(script, "OVERALL_EFFECT_CSV", da / "overall_effect_sizes.csv")
        monkeypatch.setattr(script, "PER_MACHINE_EFFECT_CSV", da / "per_machine_effect_sizes.csv")
        monkeypatch.setattr(script, "PLOTS_DIR", da / "plots")
        monkeypatch.setattr(script, "OVERALL_SUMMARY_CSV", da / "overall_results_summary.csv")
        monkeypatch.setattr(script, "PER_MACHINE_SUMMARY_CSV", da / "per_machine_results_summary.csv")
        return path, da

    def test_main_runs_with_valid_csv(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, tmp_path)
        script.main()  # must not raise

    def test_main_raises_on_missing_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "INPUT_CSV", tmp_path / "missing.csv")
        with pytest.raises(FileNotFoundError):
            script.main()

    def test_main_saves_overall_stats_csv(self, tmp_path, monkeypatch):
        _, da = self._patch(monkeypatch, tmp_path)
        script.main()
        assert (da / "overall_drift_statistics.csv").exists()

    def test_main_saves_per_machine_stats_csv(self, tmp_path, monkeypatch):
        _, da = self._patch(monkeypatch, tmp_path)
        script.main()
        assert (da / "per_machine_drift_statistics.csv").exists()

    def test_main_saves_overall_significance_csv(self, tmp_path, monkeypatch):
        _, da = self._patch(monkeypatch, tmp_path)
        script.main()
        assert (da / "overall_significance.csv").exists()

    def test_main_saves_per_machine_significance_csv(self, tmp_path, monkeypatch):
        _, da = self._patch(monkeypatch, tmp_path)
        script.main()
        assert (da / "per_machine_significance.csv").exists()

    def test_main_saves_overall_effect_csv(self, tmp_path, monkeypatch):
        _, da = self._patch(monkeypatch, tmp_path)
        script.main()
        assert (da / "overall_effect_sizes.csv").exists()

    def test_main_saves_per_machine_effect_csv(self, tmp_path, monkeypatch):
        _, da = self._patch(monkeypatch, tmp_path)
        script.main()
        assert (da / "per_machine_effect_sizes.csv").exists()

    def test_main_generates_all_six_plots(self, tmp_path, monkeypatch):
        _, da = self._patch(monkeypatch, tmp_path)
        script.main()
        plots_dir = da / "plots"
        expected = (
            [f"overall_{m}.png" for m in script.DRIFT_METRICS]
            + [f"per_machine_{m}.png" for m in script.DRIFT_METRICS]
        )
        for name in expected:
            assert (plots_dir / name).exists(), f"Missing plot: {name}"

    def test_main_saves_overall_summary_csv(self, tmp_path, monkeypatch):
        _, da = self._patch(monkeypatch, tmp_path)
        script.main()
        assert (da / "overall_results_summary.csv").exists()

    def test_main_saves_per_machine_summary_csv(self, tmp_path, monkeypatch):
        _, da = self._patch(monkeypatch, tmp_path)
        script.main()
        assert (da / "per_machine_results_summary.csv").exists()


# ---------------------------------------------------------------------------
# Phase 5.5 — Effect Size Analysis (Rank-Biserial Correlation)
# ---------------------------------------------------------------------------

class TestComputeOverallEffectSizes:
    def test_returns_three_rows(self):
        effect = script.compute_overall_effect_sizes(_make_varied_df())
        assert len(effect) == 3

    def test_columns_present(self):
        effect = script.compute_overall_effect_sizes(_make_varied_df())
        for col in ("metric", "n_normal", "n_abnormal", "rank_biserial_correlation"):
            assert col in effect.columns

    def test_all_metrics_present(self):
        effect = script.compute_overall_effect_sizes(_make_varied_df())
        assert set(effect["metric"].unique()) == set(script.DRIFT_METRICS)

    def test_rbc_in_range(self):
        effect = script.compute_overall_effect_sizes(_make_varied_df())
        assert (effect["rank_biserial_correlation"] >= -1).all()
        assert (effect["rank_biserial_correlation"] <= 1).all()

    def test_well_separated_data_has_large_effect(self):
        effect = script.compute_overall_effect_sizes(_make_varied_df())
        r = effect.loc[effect["metric"] == "normalized_euclidean", "rank_biserial_correlation"].iloc[0]
        assert abs(r) > 0.5

    def test_correct_sample_counts(self):
        effect = script.compute_overall_effect_sizes(_make_varied_df())
        row = effect[effect["metric"] == "normalized_euclidean"].iloc[0]
        assert int(row["n_normal"]) == 30
        assert int(row["n_abnormal"]) == 20

    def test_constant_values_produce_finite_result(self):
        effect = script.compute_overall_effect_sizes(_make_df())
        assert effect["rank_biserial_correlation"].notna().all()


class TestComputePerMachineEffectSizes:
    def test_row_count_two_machines(self):
        effect = script.compute_per_machine_effect_sizes(_make_multi_machine_df())
        assert len(effect) == 6

    def test_row_count_one_machine(self):
        effect = script.compute_per_machine_effect_sizes(_make_varied_df())
        assert len(effect) == 3

    def test_columns_present(self):
        effect = script.compute_per_machine_effect_sizes(_make_multi_machine_df())
        for col in ("machine_id", "metric", "n_normal", "n_abnormal", "rank_biserial_correlation"):
            assert col in effect.columns

    def test_all_machine_ids_present(self):
        effect = script.compute_per_machine_effect_sizes(_make_multi_machine_df())
        assert set(effect["machine_id"].unique()) == {"id_00", "id_02"}

    def test_all_metrics_present(self):
        effect = script.compute_per_machine_effect_sizes(_make_multi_machine_df())
        assert set(effect["metric"].unique()) == set(script.DRIFT_METRICS)

    def test_rbc_in_range(self):
        effect = script.compute_per_machine_effect_sizes(_make_multi_machine_df())
        assert (effect["rank_biserial_correlation"] >= -1).all()
        assert (effect["rank_biserial_correlation"] <= 1).all()

    def test_skips_machine_missing_a_label(self):
        df = _make_df(n_normal=5, n_abnormal=0)
        effect = script.compute_per_machine_effect_sizes(df)
        assert len(effect) == 0


class TestSaveEffectSizes:
    def test_creates_output_directory(self, tmp_path):
        out = tmp_path / "subdir" / "effect.csv"
        effect = script.compute_overall_effect_sizes(_make_varied_df())
        script.save_effect_sizes(effect, out)
        assert out.parent.exists()

    def test_csv_file_created(self, tmp_path):
        out = tmp_path / "overall_effect_sizes.csv"
        effect = script.compute_overall_effect_sizes(_make_varied_df())
        script.save_effect_sizes(effect, out)
        assert out.exists()

    def test_overall_csv_round_trip(self, tmp_path):
        out = tmp_path / "overall_effect_sizes.csv"
        effect = script.compute_overall_effect_sizes(_make_varied_df())
        script.save_effect_sizes(effect, out)
        loaded = pd.read_csv(out)
        assert len(loaded) == 3
        for col in ("metric", "n_normal", "n_abnormal", "rank_biserial_correlation"):
            assert col in loaded.columns

    def test_per_machine_csv_round_trip(self, tmp_path):
        out = tmp_path / "per_machine_effect_sizes.csv"
        effect = script.compute_per_machine_effect_sizes(_make_multi_machine_df())
        script.save_effect_sizes(effect, out)
        loaded = pd.read_csv(out)
        assert len(loaded) == 6
        for col in ("machine_id", "metric", "n_normal", "n_abnormal", "rank_biserial_correlation"):
            assert col in loaded.columns


# ---------------------------------------------------------------------------
# Phase 5.6 — Visualization of Healthy vs Abnormal Drift
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestPlotOverallDistribution:
    def test_returns_figure(self):
        fig = script.plot_overall_distribution(_make_df(), "normalized_euclidean")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_works_for_all_metrics(self):
        for metric in script.DRIFT_METRICS:
            fig = script.plot_overall_distribution(_make_df(), metric)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)


class TestSaveOverallDistributionPlot:
    def test_creates_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "PLOTS_DIR", tmp_path / "plots")
        script.save_overall_distribution_plot(_make_df(), "normalized_euclidean")
        assert (tmp_path / "plots").exists()

    def test_creates_png(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "PLOTS_DIR", tmp_path / "plots")
        script.save_overall_distribution_plot(_make_df(), "normalized_euclidean")
        assert (tmp_path / "plots" / "overall_normalized_euclidean.png").exists()

    def test_png_is_non_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "PLOTS_DIR", tmp_path / "plots")
        script.save_overall_distribution_plot(_make_df(), "normalized_euclidean")
        assert (tmp_path / "plots" / "overall_normalized_euclidean.png").stat().st_size > 0


class TestPlotPerMachineDistribution:
    def test_returns_figure(self):
        fig = script.plot_per_machine_distribution(_make_df(), "normalized_euclidean")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_works_for_multiple_machines(self):
        fig = script.plot_per_machine_distribution(_make_multi_machine_df(), "normalized_euclidean")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_works_for_one_machine(self):
        fig = script.plot_per_machine_distribution(_make_df(), "normalized_euclidean")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_works_for_all_metrics(self):
        for metric in script.DRIFT_METRICS:
            fig = script.plot_per_machine_distribution(_make_multi_machine_df(), metric)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_machine_ids_are_dynamic(self):
        df = _make_multi_machine_df()
        machine_ids = sorted(df["machine_id"].unique())
        fig = script.plot_per_machine_distribution(df, "normalized_euclidean")
        ax = fig.axes[0]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        for mid in machine_ids:
            assert mid in tick_labels
        plt.close(fig)


class TestSavePerMachineDistributionPlot:
    def test_creates_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "PLOTS_DIR", tmp_path / "plots")
        script.save_per_machine_distribution_plot(_make_df(), "normalized_euclidean")
        assert (tmp_path / "plots").exists()

    def test_creates_png(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "PLOTS_DIR", tmp_path / "plots")
        script.save_per_machine_distribution_plot(_make_df(), "normalized_euclidean")
        assert (tmp_path / "plots" / "per_machine_normalized_euclidean.png").exists()

    def test_png_is_non_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "PLOTS_DIR", tmp_path / "plots")
        script.save_per_machine_distribution_plot(_make_df(), "normalized_euclidean")
        assert (tmp_path / "plots" / "per_machine_normalized_euclidean.png").stat().st_size > 0


class TestGenerateAllPlots:
    def test_generates_all_six_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "PLOTS_DIR", tmp_path / "plots")
        script.generate_all_plots(_make_multi_machine_df())
        expected = [
            f"overall_{m}.png" for m in script.DRIFT_METRICS
        ] + [
            f"per_machine_{m}.png" for m in script.DRIFT_METRICS
        ]
        for name in expected:
            assert (tmp_path / "plots" / name).exists(), f"Missing: {name}"

    def test_all_files_non_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "PLOTS_DIR", tmp_path / "plots")
        script.generate_all_plots(_make_multi_machine_df())
        for f in (tmp_path / "plots").iterdir():
            assert f.stat().st_size > 0, f"Empty file: {f.name}"


# ---------------------------------------------------------------------------
# Phase 5.7 — Consolidated Results Summary
# ---------------------------------------------------------------------------

def _make_summary_inputs(df=None):
    """Return (overall_stats, overall_sig, overall_effect) from a DataFrame."""
    if df is None:
        df = _make_multi_machine_df()
    stats = script.compute_overall_drift_statistics(df)
    sig = script.compute_overall_significance(df)
    effect = script.compute_overall_effect_sizes(df)
    return stats, sig, effect


def _make_pm_summary_inputs(df=None):
    """Return (per_machine_stats, per_machine_sig, per_machine_effect)."""
    if df is None:
        df = _make_multi_machine_df()
    stats = script.compute_per_machine_drift_statistics(df)
    sig = script.compute_per_machine_significance(df)
    effect = script.compute_per_machine_effect_sizes(df)
    return stats, sig, effect


_OVERALL_SUMMARY_COLS = [
    "metric",
    "normal_count", "normal_mean", "normal_std", "normal_median",
    "abnormal_count", "abnormal_mean", "abnormal_std", "abnormal_median",
    "u_statistic", "p_value", "rank_biserial",
]

_PM_SUMMARY_COLS = [
    "machine_id", "metric",
    "normal_count", "normal_mean", "normal_std", "normal_median",
    "abnormal_count", "abnormal_mean", "abnormal_std", "abnormal_median",
    "u_statistic", "p_value", "rank_biserial",
]


class TestComputeOverallResultsSummary:
    def test_returns_three_rows(self):
        summary = script.compute_overall_results_summary(*_make_summary_inputs())
        assert len(summary) == 3

    def test_expected_columns_present(self):
        summary = script.compute_overall_results_summary(*_make_summary_inputs())
        for col in _OVERALL_SUMMARY_COLS:
            assert col in summary.columns, f"Missing column: {col}"

    def test_all_three_metrics_present(self):
        summary = script.compute_overall_results_summary(*_make_summary_inputs())
        assert set(summary["metric"].unique()) == set(script.DRIFT_METRICS)

    def test_correct_normal_values_merged(self):
        stats, sig, effect = _make_summary_inputs(_make_df())
        summary = script.compute_overall_results_summary(stats, sig, effect)
        row = summary[summary["metric"] == "normalized_euclidean"].iloc[0]
        assert abs(row["normal_mean"] - 10.0) < 1e-9
        assert int(row["normal_count"]) == 10

    def test_correct_abnormal_values_merged(self):
        stats, sig, effect = _make_summary_inputs(_make_df())
        summary = script.compute_overall_results_summary(stats, sig, effect)
        row = summary[summary["metric"] == "normalized_euclidean"].iloc[0]
        assert abs(row["abnormal_mean"] - 40.0) < 1e-9
        assert int(row["abnormal_count"]) == 8

    def test_p_values_merged_correctly(self):
        stats, sig, effect = _make_summary_inputs(_make_varied_df())
        summary = script.compute_overall_results_summary(stats, sig, effect)
        # p-values must be in [0, 1]
        assert (summary["p_value"] >= 0).all()
        assert (summary["p_value"] <= 1).all()

    def test_effect_sizes_merged_correctly(self):
        stats, sig, effect = _make_summary_inputs(_make_varied_df())
        summary = script.compute_overall_results_summary(stats, sig, effect)
        assert (summary["rank_biserial"] >= -1).all()
        assert (summary["rank_biserial"] <= 1).all()


class TestComputePerMachineResultsSummary:
    def test_correct_row_count_two_machines(self):
        # 2 machines × 3 metrics = 6 rows
        summary = script.compute_per_machine_results_summary(*_make_pm_summary_inputs())
        assert len(summary) == 6

    def test_correct_row_count_one_machine(self):
        # 1 machine × 3 metrics = 3 rows
        summary = script.compute_per_machine_results_summary(
            *_make_pm_summary_inputs(_make_df())
        )
        assert len(summary) == 3

    def test_expected_columns_present(self):
        summary = script.compute_per_machine_results_summary(*_make_pm_summary_inputs())
        for col in _PM_SUMMARY_COLS:
            assert col in summary.columns, f"Missing column: {col}"

    def test_all_machine_ids_present(self):
        summary = script.compute_per_machine_results_summary(*_make_pm_summary_inputs())
        assert set(summary["machine_id"].unique()) == {"id_00", "id_02"}

    def test_all_metrics_present(self):
        summary = script.compute_per_machine_results_summary(*_make_pm_summary_inputs())
        assert set(summary["metric"].unique()) == set(script.DRIFT_METRICS)

    def test_correct_normal_values_merged(self):
        summary = script.compute_per_machine_results_summary(*_make_pm_summary_inputs())
        row = summary[
            (summary["machine_id"] == "id_00") &
            (summary["metric"] == "normalized_euclidean")
        ].iloc[0]
        assert abs(row["normal_mean"] - 10.0) < 1e-9
        assert int(row["normal_count"]) == 6

    def test_correct_abnormal_values_merged(self):
        summary = script.compute_per_machine_results_summary(*_make_pm_summary_inputs())
        row = summary[
            (summary["machine_id"] == "id_00") &
            (summary["metric"] == "normalized_euclidean")
        ].iloc[0]
        assert abs(row["abnormal_mean"] - 40.0) < 1e-9
        assert int(row["abnormal_count"]) == 4

    def test_p_values_merged_correctly(self):
        summary = script.compute_per_machine_results_summary(*_make_pm_summary_inputs())
        assert (summary["p_value"] >= 0).all()
        assert (summary["p_value"] <= 1).all()

    def test_effect_sizes_merged_correctly(self):
        summary = script.compute_per_machine_results_summary(*_make_pm_summary_inputs())
        assert (summary["rank_biserial"] >= -1).all()
        assert (summary["rank_biserial"] <= 1).all()


class TestSaveResultsSummary:
    def test_creates_output_directory(self, tmp_path):
        out = tmp_path / "subdir" / "summary.csv"
        summary = script.compute_overall_results_summary(*_make_summary_inputs())
        script.save_results_summary(summary, out)
        assert out.parent.exists()

    def test_creates_csv(self, tmp_path):
        out = tmp_path / "overall_results_summary.csv"
        summary = script.compute_overall_results_summary(*_make_summary_inputs())
        script.save_results_summary(summary, out)
        assert out.exists()

    def test_csv_is_non_empty(self, tmp_path):
        out = tmp_path / "overall_results_summary.csv"
        summary = script.compute_overall_results_summary(*_make_summary_inputs())
        script.save_results_summary(summary, out)
        assert out.stat().st_size > 0

    def test_csv_round_trip_preserves_columns(self, tmp_path):
        out = tmp_path / "overall_results_summary.csv"
        summary = script.compute_overall_results_summary(*_make_summary_inputs())
        script.save_results_summary(summary, out)
        loaded = pd.read_csv(out)
        for col in _OVERALL_SUMMARY_COLS:
            assert col in loaded.columns
