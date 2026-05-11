"""Data models for Step Rate Test (SRT) hydrogeological analysis."""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class PumpingStep:
    """A single constant-rate pumping step in an SRT."""
    step_number: int
    flow_rate: float        # m³/h
    duration: float         # minutes
    end_drawdown: float     # m — stabilised drawdown at end of step
    time_series: Optional[List[float]] = field(default=None)
    drawdown_series: Optional[List[float]] = field(default=None)

    @property
    def specific_drawdown(self) -> float:
        """sw / Q  [m / (m³/h)]"""
        return self.end_drawdown / self.flow_rate


@dataclass
class SRTTestData:
    """Complete dataset for a single Step Rate Test."""
    well_id: str
    test_date: str
    static_water_level: float   # m below ground surface
    aquifer_thickness: float    # m (saturated)
    well_radius: float          # m
    steps: List[PumpingStep]

    # ------------------------------------------------------------------ helpers
    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def flow_rates(self) -> np.ndarray:
        return np.array([s.flow_rate for s in self.steps])

    @property
    def drawdowns(self) -> np.ndarray:
        return np.array([s.end_drawdown for s in self.steps])

    @property
    def specific_drawdowns(self) -> np.ndarray:
        return np.array([s.specific_drawdown for s in self.steps])

    @property
    def cumulative_times(self) -> np.ndarray:
        """Cumulative elapsed time (minutes) at the end of each step."""
        return np.cumsum([s.duration for s in self.steps])

    @property
    def specific_capacity(self) -> np.ndarray:
        """Q / sw  [m³/h / m] — inverse of specific drawdown."""
        return self.flow_rates / self.drawdowns


@dataclass
class AnalysisResults:
    """Output from any SRT analysis method."""
    method: str
    B: float          # Formation (aquifer) loss coefficient  m/(m³/h)
    C: float          # Well loss coefficient                 m/(m³/h)^n
    n: float          # Well loss exponent (2 for Jacob)
    r_squared: float  # Coefficient of determination

    # ------------------------------------------------------------------ public
    def well_efficiency(self, Q: float) -> float:
        """Well efficiency at pumping rate Q [%]."""
        formation_loss = self.B * Q
        total = formation_loss + self.C * Q ** self.n
        if total <= 0:
            return 100.0
        return min((formation_loss / total) * 100.0, 100.0)

    def predict_drawdown(self, Q: float) -> float:
        """Predicted total drawdown at rate Q [m]."""
        return self.B * Q + self.C * Q ** self.n

    def optimal_rate(self, allowable_drawdown: float) -> float:
        """Maximum pumping rate that keeps drawdown ≤ allowable_drawdown [m³/h]."""
        if self.n == 2.0:
            discriminant = self.B ** 2 + 4.0 * self.C * allowable_drawdown
            if discriminant < 0:
                return np.nan
            return (-self.B + np.sqrt(discriminant)) / (2.0 * self.C)
        # General n — numerical solution
        from scipy.optimize import brentq
        try:
            return brentq(lambda Q: self.predict_drawdown(Q) - allowable_drawdown,
                          1e-6, 1e6, xtol=1e-4)
        except Exception:
            return np.nan

    def summary(self) -> str:
        lines = [
            f"Method : {self.method}",
            f"  B    = {self.B:.6f}  m/(m³/h)",
            f"  C    = {self.C:.8f}  m/(m³/h)^{self.n:.1f}",
            f"  n    = {self.n:.2f}",
            f"  R²   = {self.r_squared:.4f}",
        ]
        return "\n".join(lines)
