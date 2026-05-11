"""Tests for CSV I/O helpers."""
import csv
import tempfile
from pathlib import Path

import pytest
from srt.io import load_from_csv, save_results_to_csv
from srt.analysis import run_all_methods


@pytest.fixture
def tmp_csv(tmp_path) -> Path:
    p = tmp_path / "test_srt.csv"
    rows = [
        ["step", "flow_rate_m3h", "duration_min", "end_drawdown_m",
         "well_id", "test_date", "swl_m", "thickness_m", "radius_m"],
        [1, 5.0, 60, 2.15, "BH-TEST", "2024-01-01", 12.5, 50.0, 0.15],
        [2, 10.0, 60, 5.40, "", "", "", "", ""],
        [3, 15.0, 60, 10.25, "", "", "", "", ""],
        [4, 20.0, 60, 17.60, "", "", "", "", ""],
        [5, 25.0, 60, 27.35, "", "", "", "", ""],
    ]
    with open(p, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return p


class TestLoadFromCSV:
    def test_loads_steps(self, tmp_csv):
        data = load_from_csv(tmp_csv)
        assert data.n_steps == 5

    def test_well_id_from_csv(self, tmp_csv):
        data = load_from_csv(tmp_csv)
        assert data.well_id == "BH-TEST"

    def test_flow_rates(self, tmp_csv):
        data = load_from_csv(tmp_csv)
        assert list(data.flow_rates) == [5.0, 10.0, 15.0, 20.0, 25.0]

    def test_drawdowns(self, tmp_csv):
        data = load_from_csv(tmp_csv)
        assert data.drawdowns[0] == pytest.approx(2.15)

    def test_static_water_level(self, tmp_csv):
        data = load_from_csv(tmp_csv)
        assert data.static_water_level == pytest.approx(12.5)


class TestSaveResultsToCSV:
    def test_creates_file(self, tmp_csv, tmp_path):
        data = load_from_csv(tmp_csv)
        results = run_all_methods(data)
        out = tmp_path / "results.csv"
        save_results_to_csv(data, results, out)
        assert out.exists()

    def test_has_header_and_rows(self, tmp_csv, tmp_path):
        data = load_from_csv(tmp_csv)
        results = run_all_methods(data)
        out = tmp_path / "results.csv"
        save_results_to_csv(data, results, out)

        with open(out) as fh:
            lines = fh.readlines()
        assert len(lines) == 6  # header + 5 steps

    def test_predicted_columns_present(self, tmp_csv, tmp_path):
        data = load_from_csv(tmp_csv)
        results = run_all_methods(data)
        out = tmp_path / "results.csv"
        save_results_to_csv(data, results, out)

        with open(out) as fh:
            header = fh.readline()
        assert "predicted_jacob" in header
        assert "efficiency_pct_rorabaugh" in header
