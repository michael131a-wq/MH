"""Tests for data models."""
import numpy as np
import pytest
from srt.models import PumpingStep, SRTTestData, AnalysisResults


class TestPumpingStep:
    def test_specific_drawdown(self):
        s = PumpingStep(1, flow_rate=10.0, duration=60, end_drawdown=5.0)
        assert s.specific_drawdown == pytest.approx(0.5)

    def test_specific_drawdown_at_low_flow(self):
        s = PumpingStep(1, flow_rate=2.0, duration=60, end_drawdown=1.0)
        assert s.specific_drawdown == pytest.approx(0.5)


class TestSRTTestData:
    def test_flow_rates_array(self, sample_data):
        Q = sample_data.flow_rates
        assert list(Q) == [5.0, 10.0, 15.0, 20.0, 25.0]

    def test_drawdowns_array(self, sample_data):
        sw = sample_data.drawdowns
        assert len(sw) == 5

    def test_specific_drawdowns_shape(self, sample_data):
        sd = sample_data.specific_drawdowns
        assert sd.shape == (5,)

    def test_cumulative_times(self, sample_data):
        ct = sample_data.cumulative_times
        # each step is 60 min → cumulative should be 60, 120, 180, 240, 300
        np.testing.assert_array_equal(ct, [60, 120, 180, 240, 300])

    def test_specific_capacity_positive(self, sample_data):
        sc = sample_data.specific_capacity
        assert np.all(sc > 0)

    def test_n_steps(self, sample_data):
        assert sample_data.n_steps == 5


class TestAnalysisResults:
    @pytest.fixture
    def results(self):
        return AnalysisResults(method="Test", B=0.05, C=0.03, n=2.0, r_squared=0.99)

    def test_well_efficiency_high_at_low_Q(self, results):
        eff = results.well_efficiency(1.0)
        assert eff > 60

    def test_well_efficiency_decreases_with_Q(self, results):
        eff_low = results.well_efficiency(5.0)
        eff_high = results.well_efficiency(25.0)
        assert eff_low > eff_high

    def test_well_efficiency_bounded(self, results):
        for Q in [1, 10, 50, 100]:
            eff = results.well_efficiency(Q)
            assert 0 <= eff <= 100

    def test_predict_drawdown_positive(self, results):
        assert results.predict_drawdown(10.0) > 0

    def test_predict_drawdown_increases_with_Q(self, results):
        assert results.predict_drawdown(10.0) < results.predict_drawdown(20.0)

    def test_optimal_rate_n2(self, results):
        allowable = 5.0
        Q_opt = results.optimal_rate(allowable)
        sw_at_Q = results.predict_drawdown(Q_opt)
        assert sw_at_Q == pytest.approx(allowable, rel=1e-4)

    def test_optimal_rate_general_n(self):
        res = AnalysisResults(method="Test", B=0.05, C=0.02, n=2.5, r_squared=0.98)
        Q_opt = res.optimal_rate(8.0)
        assert not np.isnan(Q_opt)
        assert res.predict_drawdown(Q_opt) == pytest.approx(8.0, rel=1e-3)

    def test_summary_contains_method(self, results):
        assert "Test" in results.summary()
