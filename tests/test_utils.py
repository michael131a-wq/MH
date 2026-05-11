"""Tests for unit-conversion and reporting utilities."""
import pytest
from srt.utils import (
    lps_to_m3h, m3h_to_lps,
    gpm_to_m3h, m3h_to_gpm,
    ft_to_m, m_to_ft,
    print_report,
)
from srt.analysis import run_all_methods


class TestUnitConversions:
    def test_lps_to_m3h_round_trip(self):
        q = 5.5
        assert m3h_to_lps(lps_to_m3h(q)) == pytest.approx(q, rel=1e-9)

    def test_gpm_to_m3h_round_trip(self):
        q = 100.0
        assert m3h_to_gpm(gpm_to_m3h(q)) == pytest.approx(q, rel=1e-6)

    def test_ft_to_m_round_trip(self):
        v = 25.0
        assert m_to_ft(ft_to_m(v)) == pytest.approx(v, rel=1e-9)

    def test_lps_to_m3h_known_value(self):
        # 1 L/s = 3.6 m³/h
        assert lps_to_m3h(1.0) == pytest.approx(3.6)

    def test_ft_to_m_known_value(self):
        # 1 ft = 0.3048 m
        assert ft_to_m(1.0) == pytest.approx(0.3048)


class TestPrintReport:
    def test_runs_without_error(self, sample_data, capsys):
        results = run_all_methods(sample_data)
        print_report(sample_data, results)
        captured = capsys.readouterr()
        assert "BH-01" not in captured.out or "TEST-01" in captured.out

    def test_contains_well_id(self, sample_data, capsys):
        results = run_all_methods(sample_data)
        print_report(sample_data, results)
        out = capsys.readouterr().out
        assert sample_data.well_id in out

    def test_contains_jacob(self, sample_data, capsys):
        results = run_all_methods(sample_data)
        print_report(sample_data, results)
        out = capsys.readouterr().out
        assert "Jacob" in out
