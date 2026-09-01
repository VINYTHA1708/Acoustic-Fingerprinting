"""Tests for experiments/e1_consolidated_baseline_comparison.py.

No audio files, no BEATs, no MIMII dataset required.
All tests operate on synthetic in-memory row dicts or temporary CSV files.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.e1_consolidated_baseline_comparison as script

MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
MAIN_ID     = script.MAIN_METHOD_ID   # "contrastive_main"


# ---------------------------------------------------------------------------
# Fixtures — synthetic rows that mirror the real CSV data
# ---------------------------------------------------------------------------

def _baseline_rows() -> list[dict]:
    data = {
        "B1_mfcc_distance": {
            "id_00": (0.5068, 1.0155), "id_02": (0.6670, 1.6148),
            "id_04": (0.5816, 1.2087), "id_06": (0.5416, 0.9942),
        },
        "B2_stat_distance": {
            "id_00": (0.4581, 0.9210), "id_02": (0.6316, 1.5113),
            "id_04": (0.5441, 1.1763), "id_06": (0.5265, 0.9312),
        },
        "B3_random_projection": {
            "id_00": (0.5198, 1.0002), "id_02": (0.5145, 1.0003),
            "id_04": (0.5228, 1.0005), "id_06": (0.5442, 1.0009),
        },
    }
    names = {
        "B1_mfcc_distance":    "Raw MFCC Distance",
        "B2_stat_distance":    "Statistical Audio Feature Distance",
        "B3_random_projection": "Non-Contrastive (Random) Projection Embedding",
    }
    rows = []
    for bid, machines in data.items():
        for mid, (auroc, sep) in machines.items():
            rows.append({
                "baseline_id": bid, "baseline_name": names[bid],
                "machine_id": mid, "n_normal": 152, "n_abnormal": 143,
                "auroc": auroc, "separation_ratio": sep,
            })
    return rows


def _main_rows() -> list[dict]:
    data = {
        "id_00": (0.7836, 2.3389), "id_02": (0.8046, 1.7205),
        "id_04": (0.9578, 4.0659), "id_06": (0.6851, 2.1632),
    }
    return [
        {
            "baseline_id": MAIN_ID,
            "baseline_name": "Contrastive Acoustic Fingerprinting",
            "machine_id": mid, "n_normal": 152, "n_abnormal": 143,
            "auroc": auroc, "separation_ratio": sep,
        }
        for mid, (auroc, sep) in data.items()
    ]


def _all_rows() -> list[dict]:
    return _baseline_rows() + _main_rows()


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------

class TestLoadCSV:
    def _write(self, tmp_path: Path, rows: list[dict]) -> Path:
        p = tmp_path / "test.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return p

    def test_returns_correct_row_count(self, tmp_path):
        rows = _baseline_rows()
        p = self._write(tmp_path, rows)
        loaded = script.load_csv(p)
        assert len(loaded) == len(rows)

    def test_auroc_is_float(self, tmp_path):
        rows = _baseline_rows()
        p = self._write(tmp_path, rows)
        loaded = script.load_csv(p)
        assert all(isinstance(r["auroc"], float) for r in loaded)

    def test_n_normal_is_int(self, tmp_path):
        rows = _baseline_rows()
        p = self._write(tmp_path, rows)
        loaded = script.load_csv(p)
        assert all(isinstance(r["n_normal"], int) for r in loaded)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            script.load_csv(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# build_consolidated
# ---------------------------------------------------------------------------

class TestBuildConsolidated:
    def test_row_count_equals_input(self):
        rows = _all_rows()
        result = script.build_consolidated(rows)
        assert len(result) == len(rows)

    def test_baseline_rows_have_empty_delta(self):
        result = script.build_consolidated(_all_rows())
        for row in result:
            if row["baseline_id"] != MAIN_ID:
                assert row["auroc_delta_vs_best_baseline"] == ""

    def test_main_rows_have_numeric_delta(self):
        result = script.build_consolidated(_all_rows())
        for row in result:
            if row["baseline_id"] == MAIN_ID:
                assert isinstance(row["auroc_delta_vs_best_baseline"], float)

    def test_delta_is_main_minus_best_baseline(self):
        result = script.build_consolidated(_all_rows())
        # For id_00: best baseline = max(0.5068, 0.4581, 0.5198) = 0.5198
        # main = 0.7836  →  delta = 0.7836 - 0.5198 = 0.2638
        main_id00 = next(
            r for r in result
            if r["baseline_id"] == MAIN_ID and r["machine_id"] == "id_00"
        )
        assert main_id00["auroc_delta_vs_best_baseline"] == pytest.approx(
            0.7836 - 0.5198, abs=1e-3
        )

    def test_delta_positive_when_main_outperforms(self):
        result = script.build_consolidated(_all_rows())
        for row in result:
            if row["baseline_id"] == MAIN_ID:
                assert row["auroc_delta_vs_best_baseline"] > 0

    def test_all_consolidated_columns_present(self):
        result = script.build_consolidated(_all_rows())
        for row in result:
            for col in script.CONSOLIDATED_COLUMNS:
                assert col in row

    def test_original_fields_preserved(self):
        rows = _all_rows()
        result = script.build_consolidated(rows)
        for orig, cons in zip(rows, result):
            assert cons["auroc"] == orig["auroc"]
            assert cons["machine_id"] == orig["machine_id"]


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:
    def test_one_row_per_method(self):
        summary = script.build_summary(_all_rows())
        method_ids = [r["baseline_id"] for r in summary]
        assert len(method_ids) == len(set(method_ids))
        assert len(summary) == 4   # B1, B2, B3, main

    def test_mean_auroc_is_float(self):
        summary = script.build_summary(_all_rows())
        for row in summary:
            assert isinstance(row["mean_auroc"], float)

    def test_mean_auroc_in_unit_interval(self):
        summary = script.build_summary(_all_rows())
        for row in summary:
            assert 0.0 <= row["mean_auroc"] <= 1.0

    def test_mean_sep_is_positive(self):
        summary = script.build_summary(_all_rows())
        for row in summary:
            assert row["mean_separation_ratio"] > 0.0

    def test_baseline_rows_have_empty_improvement(self):
        summary = script.build_summary(_all_rows())
        for row in summary:
            if row["baseline_id"] != MAIN_ID:
                assert row["auroc_improvement_over_best_baseline"] == ""

    def test_main_method_has_numeric_improvement(self):
        summary = script.build_summary(_all_rows())
        main_row = next(r for r in summary if r["baseline_id"] == MAIN_ID)
        assert isinstance(main_row["auroc_improvement_over_best_baseline"], float)

    def test_main_improvement_positive(self):
        summary = script.build_summary(_all_rows())
        main_row = next(r for r in summary if r["baseline_id"] == MAIN_ID)
        assert main_row["auroc_improvement_over_best_baseline"] > 0.0

    def test_main_mean_auroc_highest(self):
        """Main method must have the highest mean AUROC across all methods."""
        summary = script.build_summary(_all_rows())
        main_auroc = next(r["mean_auroc"] for r in summary if r["baseline_id"] == MAIN_ID)
        for row in summary:
            if row["baseline_id"] != MAIN_ID:
                assert main_auroc > row["mean_auroc"]

    def test_mean_auroc_computed_correctly_for_b1(self):
        summary = script.build_summary(_all_rows())
        b1 = next(r for r in summary if r["baseline_id"] == "B1_mfcc_distance")
        expected = round(np.mean([0.5068, 0.6670, 0.5816, 0.5416]), 4)
        assert b1["mean_auroc"] == pytest.approx(expected, abs=1e-4)

    def test_all_summary_columns_present(self):
        summary = script.build_summary(_all_rows())
        for row in summary:
            for col in script.SUMMARY_COLUMNS:
                assert col in row


# ---------------------------------------------------------------------------
# validate_inputs
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_consistent_inputs_do_not_raise(self):
        script.validate_inputs(_baseline_rows(), _main_rows())

    def test_machine_id_mismatch_raises(self):
        bad_main = [
            {**r, "machine_id": "id_99"}
            for r in _main_rows()
        ]
        with pytest.raises(ValueError, match="Machine ID mismatch"):
            script.validate_inputs(_baseline_rows(), bad_main)

    def test_main_id_in_baseline_file_raises(self):
        bad_baseline = _baseline_rows() + [
            {**_main_rows()[0], "baseline_id": MAIN_ID}
        ]
        with pytest.raises(ValueError, match=MAIN_ID):
            script.validate_inputs(bad_baseline, _main_rows())

    def test_auroc_out_of_range_raises(self):
        bad = _baseline_rows()
        bad[0] = {**bad[0], "auroc": 1.5}
        with pytest.raises(ValueError, match="AUROC"):
            script.validate_inputs(bad, _main_rows())

    def test_auroc_negative_raises(self):
        bad = _main_rows()
        bad[0] = {**bad[0], "auroc": -0.1}
        with pytest.raises(ValueError, match="AUROC"):
            script.validate_inputs(_baseline_rows(), bad)


# ---------------------------------------------------------------------------
# save_csv / round-trip
# ---------------------------------------------------------------------------

class TestSaveCSV:
    def test_consolidated_round_trip(self, tmp_path):
        rows = script.build_consolidated(_all_rows())
        out = tmp_path / "consolidated.csv"
        script.save_csv(rows, out, script.CONSOLIDATED_COLUMNS)
        assert out.exists()
        with out.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        assert len(written) == len(rows)
        for col in script.CONSOLIDATED_COLUMNS:
            assert col in written[0]

    def test_summary_round_trip(self, tmp_path):
        rows = script.build_summary(_all_rows())
        out = tmp_path / "summary.csv"
        script.save_csv(rows, out, script.SUMMARY_COLUMNS)
        assert out.exists()
        with out.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        assert len(written) == len(rows)
        for col in script.SUMMARY_COLUMNS:
            assert col in written[0]

    def test_creates_parent_directory(self, tmp_path):
        rows = script.build_summary(_all_rows())
        out = tmp_path / "nested" / "dir" / "summary.csv"
        script.save_csv(rows, out, script.SUMMARY_COLUMNS)
        assert out.exists()


# ---------------------------------------------------------------------------
# End-to-end: main() with real on-disk CSVs
# ---------------------------------------------------------------------------

class TestMainEndToEnd:
    def test_main_produces_both_output_files(self, tmp_path, monkeypatch):
        # Write synthetic input CSVs to tmp_path
        b_path = tmp_path / "baseline_results.csv"
        m_path = tmp_path / "main_method_results.csv"
        c_path = tmp_path / "consolidated_comparison.csv"
        s_path = tmp_path / "method_summary.csv"

        b_rows = _baseline_rows()
        m_rows = _main_rows()

        cols = ["baseline_id", "baseline_name", "machine_id",
                "n_normal", "n_abnormal", "auroc", "separation_ratio"]
        for path, rows in [(b_path, b_rows), (m_path, m_rows)]:
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=cols)
                writer.writeheader()
                writer.writerows(rows)

        monkeypatch.setattr(script, "BASELINE_CSV",    b_path)
        monkeypatch.setattr(script, "MAIN_CSV",        m_path)
        monkeypatch.setattr(script, "CONSOLIDATED_CSV", c_path)
        monkeypatch.setattr(script, "SUMMARY_CSV",     s_path)

        script.main()

        assert c_path.exists()
        assert s_path.exists()

    def test_main_consolidated_has_correct_row_count(self, tmp_path, monkeypatch):
        b_path = tmp_path / "baseline_results.csv"
        m_path = tmp_path / "main_method_results.csv"
        c_path = tmp_path / "consolidated_comparison.csv"
        s_path = tmp_path / "method_summary.csv"

        cols = ["baseline_id", "baseline_name", "machine_id",
                "n_normal", "n_abnormal", "auroc", "separation_ratio"]
        for path, rows in [(b_path, _baseline_rows()), (m_path, _main_rows())]:
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=cols)
                writer.writeheader()
                writer.writerows(rows)

        monkeypatch.setattr(script, "BASELINE_CSV",    b_path)
        monkeypatch.setattr(script, "MAIN_CSV",        m_path)
        monkeypatch.setattr(script, "CONSOLIDATED_CSV", c_path)
        monkeypatch.setattr(script, "SUMMARY_CSV",     s_path)

        script.main()

        with c_path.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        # 3 baselines × 4 machine IDs + 1 main × 4 machine IDs = 16
        assert len(written) == 16

    def test_main_summary_has_four_rows(self, tmp_path, monkeypatch):
        b_path = tmp_path / "baseline_results.csv"
        m_path = tmp_path / "main_method_results.csv"
        c_path = tmp_path / "consolidated_comparison.csv"
        s_path = tmp_path / "method_summary.csv"

        cols = ["baseline_id", "baseline_name", "machine_id",
                "n_normal", "n_abnormal", "auroc", "separation_ratio"]
        for path, rows in [(b_path, _baseline_rows()), (m_path, _main_rows())]:
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=cols)
                writer.writeheader()
                writer.writerows(rows)

        monkeypatch.setattr(script, "BASELINE_CSV",    b_path)
        monkeypatch.setattr(script, "MAIN_CSV",        m_path)
        monkeypatch.setattr(script, "CONSOLIDATED_CSV", c_path)
        monkeypatch.setattr(script, "SUMMARY_CSV",     s_path)

        script.main()

        with s_path.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        assert len(written) == 4   # B1, B2, B3, main
