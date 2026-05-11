"""Shared pytest fixtures for SRT tests."""
import pytest
from srt.models import PumpingStep, SRTTestData


@pytest.fixture
def sample_data() -> SRTTestData:
    """Five-step synthetic dataset with known B≈0.05, C≈0.03 (n=2)."""
    steps = [
        PumpingStep(1,  5.0, 60,  2.15),
        PumpingStep(2, 10.0, 60,  5.40),
        PumpingStep(3, 15.0, 60, 10.25),
        PumpingStep(4, 20.0, 60, 17.60),
        PumpingStep(5, 25.0, 60, 27.35),
    ]
    return SRTTestData(
        well_id="TEST-01",
        test_date="2024-01-01",
        static_water_level=15.0,
        aquifer_thickness=40.0,
        well_radius=0.15,
        steps=steps,
    )


@pytest.fixture
def minimal_data() -> SRTTestData:
    """Three-step minimal dataset (minimum for all methods)."""
    steps = [
        PumpingStep(1,  5.0, 60,  1.8),
        PumpingStep(2, 10.0, 60,  4.9),
        PumpingStep(3, 15.0, 60,  9.6),
    ]
    return SRTTestData(
        well_id="MIN-01",
        test_date="2024-06-01",
        static_water_level=8.0,
        aquifer_thickness=20.0,
        well_radius=0.10,
        steps=steps,
    )
