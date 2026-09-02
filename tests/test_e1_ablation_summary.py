"""Tests for experiments/e1_ablation_summary.py.

No audio files, no BEATs, no MIMII dataset required.
All tests operate on synthetic in-memory row dicts or temporary CSV files.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.e1_ablation_summary as script
from experiments.e1_ablation_definition import ABLATIONS

MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
FM_ID = script.FM_ID


# ---------------------------------------------------------------------------
# Synthetic rows mirroring the real ablation_results.csv
# ---------------------------------------------------------------------------

_RAW = {
    "FM_full_method": {
        "id_00": (0.7836, 2.3389),
        "id_02": (0.8046, 1.7205),
        "id_04": (0.9578, 4.0659),
        "id_06": (0.6851, 2.1632),
    },
    "A1_no_beats": {
        "id_00": (0.6862, 1.5276),
        "id_02": (0.6255, 1.2478),
        "id_04": (0.9592, 4.4392),
        "id_06": (0.7191, 1.6807),
    },
    "A2_no_dsp": {
        "id_00": (0.8683, 1.9803),
        "id_02": (0.8699, 1.8483),
        "id_04": (0.9502, 2.8513),
        "id_06": (0.7318, 1.6514),
    },
    "A3_no_contrastive": {
        "id_00": (0.5586, 1.1443),
        "id_02": (0.6550, 1.2437),
        "id_04": (0.5852, 1.2166),
        "id_06": (0.5764, 0.8916),
    },
    "A4_no_projection": {
        "id_00": (0.5068, 1.0155),
        "id_02": (0.6670, 1.6148),
        "id_04": (0.5816, 1.2087),
        "id_06": (0.5416, 0.9942),
    },
}

_NAMES = {aid: ABLATIONS[aid].name for aid in _RAW}

_COUNTS = {
    "id_00": (152, 143),
    "id_02": (152, 111),
    "id_04": (106, 100),
    "id_06": (156, 102),
}


def _make_rows(
    raw: dict | None = None,
    counts: dict | None = None,
) -> list[dict]:
    """Build synthetic ablation result rows."""
    raw    = raw    or _RAW
    counts = counts or _COUNTS
    rows = []
    for aid, machines in raw.items():
        for mid, (auroc, sep) in machines.items():
            n_norm, n_abn = counts.get(mid, (152, 143))
            rows.append({
                "ablation_id":      aid,
                "ablation_name":    _NAMES.get(aid, aid),
                "machine_id":       mid,
                "n_normal":         n_norm,
                "n_abnormal":       n_abn,
                "auroc":            auroc,
                "separation_ratio": sep,
            })
    return rows


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "ablation_results.csv"
    cols = ["ablation_id", "ablation_name", "machine_id",
            "n_normal", "n_abnormal", "auroc", "separation_ratio"]
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    return p


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------

class TestLoadCSV:
    def test_returns_correct_row_count(self, tmp_path):
        rows = _make_rows()
        p = _write_csv(tmp_path, rows)
        loaded = script.load_csv(p)
        assert len(loaded) == 20

    def test_auroc_is_float(self, tmp_path):
        p = _write_csv(tmp_path, _make_rows())
        loaded = script.load_csv(p)
        assert all(isinstance(r["auroc"], float) for r in loaded)

    def test_separation_ratio_is_float(self, tmp_path):
        p = _write_csv(tmp_path, _make_rows())
        loaded = script.load_csv(p)
        assert all(isinstance(r["separation_ratio"], float) for r in loaded)

    def test_n_normal_is_int(self, tmp_path):
        p = _write_csv(tmp_path, _make_rows())
        loaded = script.load_csv(p)
        assert all(isinstance(r["n_normal"], int) for r in loaded)

    def test_n_abnormal_is_int(self, tmp_path):
        p = _write_csv(tmp_path, _make_rows())
        loaded = script.load_csv(p)
        assert all(isinstance(r["n_abnormal"], int) for r in loaded)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            script.load_csv(tmp_path / "nonexistent.csv")

    def test_ablation_ids_preserved(self, tmp_path):
        p = _write_csv(tmp_path, _make_rows())
        loaded = script.load_csv(p)
        ids = {r["ablation_id"] for r in loaded}
        assert ids == set(_RAW.keys())


# ---------------------------------------------------------------------------
# validate_rows
# ---------------------------------------------------------------------------

class TestValidateRows:
    def test_valid_rows_do_not_raise(self):
        script.validate_rows(_make_rows())

    def test_wrong_row_count_raises(self):
        rows = _make_rows()[:19]
        with pytest.raises(ValueError, match="rows"):
            script.validate_rows(rows)

    def test_wrong_ablation_count_raises(self):
        rows = [r for r in _make_rows() if r["ablation_id"] != "A4_no_projection"]
        # 4 ablations × 4 machines = 16 rows — wrong count triggers first
        with pytest.raises(ValueError):
            script.validate_rows(rows)

    def test_auroc_above_one_raises(self):
        rows = _make_rows()
        rows[0] = {**rows[0], "auroc": 1.1}
        with pytest.raises(ValueError, match="AUROC"):
            script.validate_rows(rows)

    def test_auroc_below_zero_raises(self):
        rows = _make_rows()
        rows[0] = {**rows[0], "auroc": -0.01}
        with pytest.raises(ValueError, match="AUROC"):
            script.validate_rows(rows)

    def test_negative_separation_raises(self):
        rows = _make_rows()
        rows[0] = {**rows[0], "separation_ratio": -0.5}
        with pytest.raises(ValueError, match="separation_ratio"):
            script.validate_rows(rows)

    def test_inconsistent_counts_raises(self):
        rows = _make_rows()
        # Change n_normal for id_00 in one ablation only
        for i, r in enumerate(rows):
            if r["ablation_id"] == "A1_no_beats" and r["machine_id"] == "id_00":
                rows[i] = {**r, "n_normal": 999}
                break
        with pytest.raises(ValueError, match="Inconsistent"):
            script.validate_rows(rows)

    def test_each_ablation_has_four_machines(self):
        rows = _make_rows()
        script.validate_rows(rows)  # must not raise
        groups: dict[str, list] = {}
        for r in rows:
            groups.setdefault(r["ablation_id"], []).append(r)
        for aid, arows in groups.items():
            assert len(arows) == 4, f"{aid} has {len(arows)} rows"


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:
    def test_returns_five_rows(self):
        summary = script.build_summary(_make_rows())
        assert len(summary) == 5

    def test_one_row_per_ablation(self):
        summary = script.build_summary(_make_rows())
        ids = [s["ablation_id"] for s in summary]
        assert len(ids) == len(set(ids))

    def test_all_summary_columns_present(self):
        summary = script.build_summary(_make_rows())
        for row in summary:
            for col in script.SUMMARY_COLUMNS:
                assert col in row

    def test_mean_auroc_is_float(self):
        summary = script.build_summary(_make_rows())
        for row in summary:
            assert isinstance(row["mean_auroc"], float)

    def test_mean_auroc_in_unit_interval(self):
        summary = script.build_summary(_make_rows())
        for row in summary:
            assert 0.0 <= row["mean_auroc"] <= 1.0

    def test_mean_sep_is_positive(self):
        summary = script.build_summary(_make_rows())
        for row in summary:
            assert row["mean_separation_ratio"] > 0.0

    def test_fm_delta_is_zero(self):
        summary = script.build_summary(_make_rows())
        fm = next(s for s in summary if s["ablation_id"] == FM_ID)
        assert fm["auroc_delta_vs_fm"] == pytest.approx(0.0, abs=1e-6)

    def test_fm_mean_auroc_correct(self):
        summary = script.build_summary(_make_rows())
        fm = next(s for s in summary if s["ablation_id"] == FM_ID)
        expected = round(np.mean([0.7836, 0.8046, 0.9578, 0.6851]), 4)
        assert fm["mean_auroc"] == pytest.approx(expected, abs=1e-4)

    def test_a1_mean_auroc_correct(self):
        summary = script.build_summary(_make_rows())
        a1 = next(s for s in summary if s["ablation_id"] == "A1_no_beats")
        expected = round(np.mean([0.6862, 0.6255, 0.9592, 0.7191]), 4)
        assert a1["mean_auroc"] == pytest.approx(expected, abs=1e-4)

    def test_a2_mean_auroc_correct(self):
        summary = script.build_summary(_make_rows())
        a2 = next(s for s in summary if s["ablation_id"] == "A2_no_dsp")
        expected = round(np.mean([0.8683, 0.8699, 0.9502, 0.7318]), 4)
        assert a2["mean_auroc"] == pytest.approx(expected, abs=1e-4)

    def test_a3_mean_auroc_correct(self):
        summary = script.build_summary(_make_rows())
        a3 = next(s for s in summary if s["ablation_id"] == "A3_no_contrastive")
        expected = round(np.mean([0.5586, 0.6550, 0.5852, 0.5764]), 4)
        assert a3["mean_auroc"] == pytest.approx(expected, abs=1e-4)

    def test_a4_mean_auroc_correct(self):
        summary = script.build_summary(_make_rows())
        a4 = next(s for s in summary if s["ablation_id"] == "A4_no_projection")
        expected = round(np.mean([0.5068, 0.6670, 0.5816, 0.5416]), 4)
        assert a4["mean_auroc"] == pytest.approx(expected, abs=1e-4)

    def test_a2_delta_vs_fm_positive(self):
        """A2 (BEATs-only) outperforms FM on mean AUROC — delta must be > 0."""
        summary = script.build_summary(_make_rows())
        a2 = next(s for s in summary if s["ablation_id"] == "A2_no_dsp")
        assert a2["auroc_delta_vs_fm"] > 0.0

    def test_a3_delta_vs_fm_negative(self):
        """A3 (random head) must underperform FM."""
        summary = script.build_summary(_make_rows())
        a3 = next(s for s in summary if s["ablation_id"] == "A3_no_contrastive")
        assert a3["auroc_delta_vs_fm"] < 0.0

    def test_a4_delta_vs_fm_negative(self):
        """A4 (no projection) must underperform FM."""
        summary = script.build_summary(_make_rows())
        a4 = next(s for s in summary if s["ablation_id"] == "A4_no_projection")
        assert a4["auroc_delta_vs_fm"] < 0.0

    def test_delta_equals_mean_auroc_minus_fm(self):
        summary = script.build_summary(_make_rows())
        fm_mean = next(s["mean_auroc"] for s in summary if s["ablation_id"] == FM_ID)
        for s in summary:
            expected_delta = round(s["mean_auroc"] - fm_mean, 4)
            assert s["auroc_delta_vs_fm"] == pytest.approx(expected_delta, abs=1e-4)


# ---------------------------------------------------------------------------
# build_per_machine_delta
# ---------------------------------------------------------------------------

class TestBuildPerMachineDelta:
    def test_returns_all_ablation_ids(self):
        result = script.build_per_machine_delta(_make_rows())
        assert set(result.keys()) == set(_RAW.keys())

    def test_fm_deltas_are_zero(self):
        result = script.build_per_machine_delta(_make_rows())
        for mid, delta in result[FM_ID].items():
            assert delta == pytest.approx(0.0, abs=1e-4)

    def test_a2_id00_delta_correct(self):
        result = script.build_per_machine_delta(_make_rows())
        # A2 id_00: 0.8683 − 0.7836 = 0.0847
        assert result["A2_no_dsp"]["id_00"] == pytest.approx(0.8683 - 0.7836, abs=1e-4)

    def test_a1_id02_delta_correct(self):
        result = script.build_per_machine_delta(_make_rows())
        # A1 id_02: 0.6255 − 0.8046 = −0.1791
        assert result["A1_no_beats"]["id_02"] == pytest.approx(0.6255 - 0.8046, abs=1e-4)

    def test_each_ablation_has_four_machine_entries(self):
        result = script.build_per_machine_delta(_make_rows())
        for aid, deltas in result.items():
            assert len(deltas) == 4, f"{aid} has {len(deltas)} machine entries"


# ---------------------------------------------------------------------------
# save_csv / round-trip
# ---------------------------------------------------------------------------

class TestSaveCSV:
    def test_summary_round_trip(self, tmp_path):
        rows = script.build_summary(_make_rows())
        out = tmp_path / "ablation_summary.csv"
        script.save_csv(rows, out)
        assert out.exists()
        with out.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        assert len(written) == 5
        for col in script.SUMMARY_COLUMNS:
            assert col in written[0]

    def test_creates_parent_directory(self, tmp_path):
        rows = script.build_summary(_make_rows())
        out = tmp_path / "nested" / "dir" / "ablation_summary.csv"
        script.save_csv(rows, out)
        assert out.exists()

    def test_mean_auroc_round_trips(self, tmp_path):
        rows = script.build_summary(_make_rows())
        out = tmp_path / "ablation_summary.csv"
        script.save_csv(rows, out)
        with out.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        fm_written = next(r for r in written if r["ablation_id"] == FM_ID)
        assert float(fm_written["mean_auroc"]) == pytest.approx(
            next(s["mean_auroc"] for s in rows if s["ablation_id"] == FM_ID),
            abs=1e-4,
        )


# ---------------------------------------------------------------------------
# End-to-end: main() with real on-disk CSV
# ---------------------------------------------------------------------------

class TestMainEndToEnd:
    def test_main_produces_summary_csv(self, tmp_path, monkeypatch):
        in_path  = _write_csv(tmp_path, _make_rows())
        out_path = tmp_path / "ablation_summary.csv"
        monkeypatch.setattr(script, "ABLATION_CSV", in_path)
        monkeypatch.setattr(script, "SUMMARY_CSV",  out_path)
        script.main()
        assert out_path.exists()

    def test_main_summary_has_five_rows(self, tmp_path, monkeypatch):
        in_path  = _write_csv(tmp_path, _make_rows())
        out_path = tmp_path / "ablation_summary.csv"
        monkeypatch.setattr(script, "ABLATION_CSV", in_path)
        monkeypatch.setattr(script, "SUMMARY_CSV",  out_path)
        script.main()
        with out_path.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        assert len(written) == 5

    def test_main_summary_has_all_columns(self, tmp_path, monkeypatch):
        in_path  = _write_csv(tmp_path, _make_rows())
        out_path = tmp_path / "ablation_summary.csv"
        monkeypatch.setattr(script, "ABLATION_CSV", in_path)
        monkeypatch.setattr(script, "SUMMARY_CSV",  out_path)
        script.main()
        with out_path.open("r", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        for col in script.SUMMARY_COLUMNS:
            assert col in written[0]

    def test_main_missing_csv_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "ABLATION_CSV", tmp_path / "missing.csv")
        with pytest.raises(FileNotFoundError):
            script.main()
