"""Tests for SRT analysis methods."""
import numpy as np
import pytest
from srt.analysis import JacobAnalysis, RorabaughAnalysis, EdenHazelAnalysis, run_all_methods
from srt.models import PumpingStep, SRTTestData


# ──────────────────────────────────────────────────────────────────────────────
class TestJacobAnalysis:
    def test_returns_results(self, sample_data):
        j = JacobAnalysis(sample_data)
        assert j.results is not None

    def test_coefficients_non_negative(self, sample_data):
        j = JacobAnalysis(sample_data)
        assert j.results.B >= 0
        assert j.results.C >= 0

    def test_r_squared_high_for_smooth_data(self, sample_data):
        j = JacobAnalysis(sample_data)
        assert j.results.r_squared >= 0.90

    def test_n_is_two(self, sample_data):
        j = JacobAnalysis(sample_data)
        assert j.results.n == pytest.approx(2.0)

    def test_method_label(self, sample_data):
        j = JacobAnalysis(sample_data)
        assert "Jacob" in j.results.method

    def test_predicts_reasonable_drawdown(self, sample_data):
        j = JacobAnalysis(sample_data)
        sw = j.results.predict_drawdown(10.0)
        # Observed drawdown at Q=10 is 5.40 m; prediction should be in ±20%
        assert 4.0 < sw < 7.0

    def test_minimal_data(self, minimal_data):
        j = JacobAnalysis(minimal_data)
        assert j.results.r_squared > 0.80

    def test_perfect_data(self):
        """Known B=0.10, C=0.04, n=2 → recovery of coefficients."""
        B, C = 0.10, 0.04
        steps = [
            PumpingStep(i, float(q), 60, B * q + C * q ** 2)
            for i, q in enumerate([5, 10, 15, 20, 25], start=1)
        ]
        data = SRTTestData("P", "2024", 10.0, 30.0, 0.15, steps)
        j = JacobAnalysis(data)
        assert j.results.B == pytest.approx(B, rel=1e-6)
        assert j.results.C == pytest.approx(C, rel=1e-6)
        assert j.results.r_squared == pytest.approx(1.0, abs=1e-10)


# ──────────────────────────────────────────────────────────────────────────────
class TestRorabaughAnalysis:
    def test_returns_results(self, sample_data):
        r = RorabaughAnalysis(sample_data)
        assert r.results is not None

    def test_coefficients_non_negative(self, sample_data):
        r = RorabaughAnalysis(sample_data)
        assert r.results.B >= 0
        assert r.results.C >= 0

    def test_n_in_valid_range(self, sample_data):
        r = RorabaughAnalysis(sample_data)
        assert 1.5 <= r.results.n <= 3.5

    def test_r_squared_reasonable(self, sample_data):
        r = RorabaughAnalysis(sample_data)
        assert r.results.r_squared >= 0.85

    def test_method_label(self, sample_data):
        r = RorabaughAnalysis(sample_data)
        assert "Rorabaugh" in r.results.method

    def test_best_n_attribute(self, sample_data):
        r = RorabaughAnalysis(sample_data)
        assert r.best_n == pytest.approx(r.results.n)

    def test_n2_data_recovered(self):
        """When data obeys n=2 exactly, Rorabaugh should converge near n=2."""
        B, C = 0.08, 0.035
        steps = [
            PumpingStep(i, float(q), 60, B * q + C * q ** 2)
            for i, q in enumerate([5, 10, 15, 20, 25], start=1)
        ]
        data = SRTTestData("P", "2024", 10.0, 30.0, 0.15, steps)
        r = RorabaughAnalysis(data)
        assert r.results.n == pytest.approx(2.0, abs=0.1)


# ──────────────────────────────────────────────────────────────────────────────
class TestEdenHazelAnalysis:
    def test_returns_results(self, sample_data):
        e = EdenHazelAnalysis(sample_data)
        assert e.results is not None

    def test_coefficients_non_negative(self, sample_data):
        e = EdenHazelAnalysis(sample_data)
        assert e.results.B >= 0
        assert e.results.C >= 0

    def test_r_squared_reasonable(self, sample_data):
        e = EdenHazelAnalysis(sample_data)
        assert e.results.r_squared >= 0.80

    def test_method_label(self, sample_data):
        e = EdenHazelAnalysis(sample_data)
        assert "Eden" in e.results.method

    def test_custom_n(self, sample_data):
        e = EdenHazelAnalysis(sample_data, n=2.5)
        assert e.results.n == pytest.approx(2.5)

    def test_minimal_data(self, minimal_data):
        e = EdenHazelAnalysis(minimal_data)
        assert e.results is not None


# ──────────────────────────────────────────────────────────────────────────────
class TestRunAllMethods:
    def test_returns_three_results(self, sample_data):
        results = run_all_methods(sample_data)
        assert set(results.keys()) == {"jacob", "rorabaugh", "eden_hazel"}

    def test_all_r_squared_positive(self, sample_data):
        results = run_all_methods(sample_data)
        for res in results.values():
            assert res.r_squared >= 0

    def test_all_predict_positive_drawdown(self, sample_data):
        results = run_all_methods(sample_data)
        for res in results.values():
            assert res.predict_drawdown(15.0) > 0


# ──────────────────────────────────────────────────────────────────────────────
class TestWellEfficiencyBehaviour:
    """Behaviour tests that apply to all methods."""

    @pytest.mark.parametrize("method_cls", [JacobAnalysis, RorabaughAnalysis, EdenHazelAnalysis])
    def test_efficiency_decreases_with_Q(self, method_cls, sample_data):
        analysis = method_cls(sample_data)
        e_low = analysis.results.well_efficiency(5.0)
        e_high = analysis.results.well_efficiency(25.0)
        assert e_low > e_high

    @pytest.mark.parametrize("method_cls", [JacobAnalysis, RorabaughAnalysis, EdenHazelAnalysis])
    def test_efficiency_bounded_0_100(self, method_cls, sample_data):
        analysis = method_cls(sample_data)
        for Q in [1, 5, 15, 30]:
            eff = analysis.results.well_efficiency(Q)
            assert 0 <= eff <= 100
