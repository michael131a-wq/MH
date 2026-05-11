"""
SRT analysis methods:
  - Jacob (1947)       : sw = BQ + CQ²  (n fixed at 2, linear regression)
  - Rorabaugh (1953)   : sw = BQ + CQ^n (n determined iteratively)
  - Eden-Hazel (1973)  : accounts for time-varying aquifer component via superposition
"""
from __future__ import annotations

import numpy as np
from scipy import stats, optimize
from typing import Tuple

from .models import SRTTestData, AnalysisResults


# ──────────────────────────────────────────────────────────────────────────────
class JacobAnalysis:
    """
    Jacob's step-drawdown method (1947).

    Linearises sw = BQ + CQ² → sw/Q = B + C·Q and fits by OLS regression.
    B is the formation (aquifer) loss coefficient; C is the well loss coefficient.
    """

    def __init__(self, data: SRTTestData) -> None:
        self.data = data
        self.results = self._fit()

    def _fit(self) -> AnalysisResults:
        Q = self.data.flow_rates
        sw_Q = self.data.specific_drawdowns

        slope, intercept, r, *_ = stats.linregress(Q, sw_Q)

        return AnalysisResults(
            method="Jacob (1947)",
            B=max(float(intercept), 0.0),
            C=max(float(slope), 0.0),
            n=2.0,
            r_squared=float(r ** 2),
        )


# ──────────────────────────────────────────────────────────────────────────────
class RorabaughAnalysis:
    """
    Rorabaugh's generalised method (1953).

    Fits sw = BQ + CQ^n by iterating over n in [1.5, 3.5] and selecting the
    value that maximises R².
    """

    N_RANGE = np.arange(1.5, 3.55, 0.05)

    def __init__(self, data: SRTTestData) -> None:
        self.data = data
        self.results, self.best_n = self._fit()

    def _fit(self) -> Tuple[AnalysisResults, float]:
        Q = self.data.flow_rates
        sw = self.data.drawdowns

        best: Tuple[float, float, float, float] | None = None  # (B, C, n, r2)

        for n in self.N_RANGE:
            try:
                B, C, r2 = self._ols_for_n(Q, sw, n)
            except Exception:
                continue
            if B >= 0 and C >= 0 and (best is None or r2 > best[3]):
                best = (B, C, float(n), r2)

        if best is None:
            raise RuntimeError("Rorabaugh analysis failed to converge for any n.")

        B, C, n, r2 = best
        return (
            AnalysisResults(
                method=f"Rorabaugh (1953)  n={n:.2f}",
                B=B, C=C, n=n, r_squared=r2,
            ),
            n,
        )

    @staticmethod
    def _ols_for_n(Q: np.ndarray, sw: np.ndarray, n: float
                   ) -> Tuple[float, float, float]:
        # sw/Q = B + C·Q^(n-1)  →  OLS in X = Q^(n-1)
        X = Q ** (n - 1)
        sw_Q = sw / Q
        slope, intercept, r, *_ = stats.linregress(X, sw_Q)
        return max(float(intercept), 0.0), max(float(slope), 0.0), float(r ** 2)


# ──────────────────────────────────────────────────────────────────────────────
class EdenHazelAnalysis:
    """
    Eden-Hazel step-drawdown method (1973).

    Uses Cooper-Jacob log approximation with superposition to separate the
    time-dependent aquifer loss from the instantaneous well loss:

        s_wp = A · Hp · Qp  +  C · Qp^n

    where  Hp = (1/Qp) · Σᵢ ΔQᵢ · log₁₀(tp − t_{i−1})

    Parameters solved by constrained least-squares (B, C ≥ 0).
    n defaults to 2 but can be supplied externally.
    """

    def __init__(self, data: SRTTestData, n: float = 2.0) -> None:
        self.data = data
        self.n = n
        self.results = self._fit()

    # ------------------------------------------------------------------ public
    def _fit(self) -> AnalysisResults:
        Q = self.data.flow_rates
        sw = self.data.drawdowns
        H = self._compute_H()

        jacob = JacobAnalysis(self.data)
        x0 = [jacob.results.B, jacob.results.C]

        def residuals(params: list) -> np.ndarray:
            A, C = params
            return sw - (A * H * Q + C * Q ** self.n)

        result = optimize.least_squares(
            residuals, x0,
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            method="trf",
        )

        A, C = result.x
        sw_pred = A * H * Q + C * Q ** self.n
        ss_res = np.sum((sw - sw_pred) ** 2)
        ss_tot = np.sum((sw - sw.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return AnalysisResults(
            method=f"Eden-Hazel (1973)  n={self.n:.1f}",
            B=float(A),
            C=float(C),
            n=self.n,
            r_squared=float(r2),
        )

    def _compute_H(self) -> np.ndarray:
        """Time-factor vector H for each step using superposition."""
        steps = self.data.steps
        cum_t = self.data.cumulative_times
        H = np.zeros(len(steps))

        for p, step_p in enumerate(steps):
            Q_p = step_p.flow_rate
            t_p = cum_t[p]
            h_sum = 0.0

            for i, step_i in enumerate(steps[: p + 1]):
                delta_Q = step_i.flow_rate - (steps[i - 1].flow_rate if i > 0 else 0.0)
                t_prev = cum_t[i - 1] if i > 0 else 0.0
                dt = t_p - t_prev
                if dt > 0:
                    h_sum += delta_Q * np.log10(dt)

            H[p] = h_sum / Q_p if Q_p > 0 else 0.0

        return H


# ──────────────────────────────────────────────────────────────────────────────
def run_all_methods(data: SRTTestData) -> dict[str, AnalysisResults]:
    """Convenience wrapper that runs all three methods and returns a dict."""
    jacob = JacobAnalysis(data)
    rorabaugh = RorabaughAnalysis(data)
    eden_hazel = EdenHazelAnalysis(data)

    return {
        "jacob": jacob.results,
        "rorabaugh": rorabaugh.results,
        "eden_hazel": eden_hazel.results,
    }
