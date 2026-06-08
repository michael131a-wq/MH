#!/usr/bin/env python3
"""
Theis (1935) Constant Rate Test (CRT) Analysis

Confined aquifer model.  Fits transmissivity (T) and storativity (S)
to observed drawdown, applies the Bourdet (1989) logarithmic derivative
with smoothing L = 0.8, and reports model-fit statistics.

Usage:
  python theis_crt.py --data crt_data.csv --Q 0.001 --r 50 [options]

CSV format (header optional, "#" lines ignored):
  time(s), drawdown(m)   e.g.   60, 0.12
                                  120, 0.21
                                  ...

Options:
  --Q       Pumping rate (m³/s)            [required]
  --r       Obs-well distance from pump (m)[required]
  --T_init  Initial T guess (m²/s)         [default: 1e-3]
  --S_init  Initial S guess (-)            [default: 1e-4]
  --L       Bourdet smoothing factor       [default: 0.8]
  --plot    Save diagnostic plot PNG
  --out     Output PNG filename            [default: theis_crt_analysis.png]
"""

import argparse
import sys
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    from scipy.special import exp1
    from scipy.optimize import curve_fit
except ImportError:
    sys.exit("ERROR: scipy is required.  Install with: pip install scipy")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load time(s) and drawdown(m) columns from a CSV file."""
    data = np.genfromtxt(path, delimiter=",", comments="#", dtype=float)
    if data.ndim == 1:
        raise ValueError("CSV must have at least two columns (time, drawdown).")
    # Strip rows where either column is NaN or time <= 0
    mask = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1]) & (data[:, 0] > 0)
    if mask.sum() < 3:
        raise ValueError("Need at least 3 valid (time > 0, finite drawdown) rows.")
    t = data[mask, 0]
    s = data[mask, 1]
    order = np.argsort(t)
    return t[order], s[order]


# ─────────────────────────────────────────────────────────────────────────────
# Theis well function and drawdown model
# ─────────────────────────────────────────────────────────────────────────────

def theis_W(u: np.ndarray) -> np.ndarray:
    """Theis well function W(u) = E₁(u) (exponential integral)."""
    return exp1(np.asarray(u, dtype=float))


def theis_drawdown(t: np.ndarray, T: float, S: float,
                   Q: float, r: float) -> np.ndarray:
    """
    Theis (1935) confined aquifer drawdown.

    s(r, t) = Q / (4πT) · W(u),   u = r²S / (4Tt)

    Parameters
    ----------
    t : array  Time (s)
    T : float  Transmissivity (m²/s)
    S : float  Storativity (-)
    Q : float  Pumping rate (m³/s)
    r : float  Radial distance (m)
    """
    u = (r ** 2 * S) / (4.0 * T * np.asarray(t, dtype=float))
    return (Q / (4.0 * np.pi * T)) * theis_W(u)


# ─────────────────────────────────────────────────────────────────────────────
# Bourdet derivative
# ─────────────────────────────────────────────────────────────────────────────

def bourdet_derivative(t: np.ndarray, s: np.ndarray,
                       L: float = 0.8) -> np.ndarray:
    """
    Bourdet (1989) logarithmic pressure derivative with smoothing L.

    For each point i the algorithm finds:
      A — last point to the left  with  ln(t_i) − ln(t_A) ≥ L
      B — first point to the right with  ln(t_B) − ln(t_i) ≥ L

    Then computes the weighted two-point central difference:
      ds'_i = [ (ds_A/dln_A)·dln_B  +  (ds_B/dln_B)·dln_A ] / (dln_A + dln_B)

    Falls back to a one-sided difference near the ends.

    Parameters
    ----------
    t : array  Time (s), must be strictly increasing
    s : array  Drawdown (m)
    L : float  Smoothing in natural-log cycles (default 0.8)
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    n = len(t)
    ln_t = np.log(t)
    deriv = np.full(n, np.nan)

    for i in range(n):
        # Leftmost eligible point A
        a_idx = None
        for j in range(i - 1, -1, -1):
            if (ln_t[i] - ln_t[j]) >= L:
                a_idx = j
                break

        # Rightmost eligible point B
        b_idx = None
        for j in range(i + 1, n):
            if (ln_t[j] - ln_t[i]) >= L:
                b_idx = j
                break

        if a_idx is not None and b_idx is not None:
            dln_A = ln_t[i] - ln_t[a_idx]
            dln_B = ln_t[b_idx] - ln_t[i]
            deriv[i] = (
                (s[i] - s[a_idx]) / dln_A * dln_B
                + (s[b_idx] - s[i]) / dln_B * dln_A
            ) / (dln_A + dln_B)
        elif a_idx is not None:
            deriv[i] = (s[i] - s[a_idx]) / (ln_t[i] - ln_t[a_idx])
        elif b_idx is not None:
            deriv[i] = (s[b_idx] - s[i]) / (ln_t[b_idx] - ln_t[i])

    return deriv


# ─────────────────────────────────────────────────────────────────────────────
# Fit statistics
# ─────────────────────────────────────────────────────────────────────────────

def fit_statistics(obs: np.ndarray, model: np.ndarray) -> dict:
    """Return RMSE, MAE, R², and Nash–Sutcliffe Efficiency (NSE)."""
    res = obs - model
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae  = float(np.mean(np.abs(res)))
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2  = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    nse = r2  # identical formula for drawdown vs model
    return {"RMSE (m)": rmse, "MAE  (m)": mae, "R²": r2, "NSE": nse}


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(t: np.ndarray, s_obs: np.ndarray,
                 Q: float, r: float,
                 T_init: float = 1e-3, S_init: float = 1e-4,
                 L: float = 0.8) -> dict:
    """
    Fit Theis model to CRT data, compute Bourdet derivative, report fit.

    Returns a dict with keys: T, S, s_model, stats, deriv_obs, deriv_model.
    """
    print("=" * 62)
    print("  THEIS SOLUTION — Constant Rate Test Analysis")
    print("=" * 62)
    print(f"  Pumping rate   Q = {Q:.4e} m³/s")
    print(f"  Observation    r = {r:.2f} m")
    print(f"  Data points      = {len(t)}")
    print(f"  Time range       = {t[0]:.1f} – {t[-1]:.1f} s")
    print(f"  Drawdown range   = {s_obs.min():.4f} – {s_obs.max():.4f} m")
    print("-" * 62)

    # ── Curve fit ────────────────────────────────────────────────────────────
    def _model(t_arr, T, S):
        return theis_drawdown(t_arr, T, S, Q, r)

    T_fit, S_fit, perr = T_init, S_init, np.array([np.nan, np.nan])
    try:
        popt, pcov = curve_fit(
            _model, t, s_obs,
            p0=[T_init, S_init],
            bounds=([1e-10, 1e-12], [10.0, 1.0]),
            maxfev=20000,
        )
        T_fit, S_fit = popt
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError as exc:
        print(f"  WARNING: curve_fit did not converge — {exc}")
        print(f"  Falling back to initial guesses.")

    print(f"  Fitted T  = {T_fit:.4e} ± {perr[0]:.2e} m²/s")
    print(f"  Fitted S  = {S_fit:.4e} ± {perr[1]:.2e}")

    # Derived aquifer parameter
    print(f"  Hydraulic diffusivity T/S = {T_fit/S_fit:.4e} m²/s")

    # ── Model drawdown ────────────────────────────────────────────────────────
    s_model = _model(t, T_fit, S_fit)

    # ── Drawdown fit statistics ───────────────────────────────────────────────
    stats = fit_statistics(s_obs, s_model)
    print("-" * 62)
    print("  Drawdown Model Fit:")
    for k, v in stats.items():
        print(f"    {k:12s} = {v:.6f}")

    # ── Bourdet derivative ────────────────────────────────────────────────────
    deriv_obs   = bourdet_derivative(t, s_obs,   L=L)
    deriv_model = bourdet_derivative(t, s_model, L=L)
    valid = ~(np.isnan(deriv_obs) | np.isnan(deriv_model) |
              (deriv_obs <= 0) | (deriv_model <= 0))

    print(f"\n  Bourdet Derivative (L = {L}):")
    print(f"    Valid points = {valid.sum()} / {len(t)}")
    if valid.sum() >= 2:
        dstats = fit_statistics(deriv_obs[valid], deriv_model[valid])
        for k, v in dstats.items():
            print(f"    {k:12s} = {v:.6f}")
    else:
        print("    (insufficient valid derivative points for statistics)")
        dstats = {}

    print("=" * 62)

    return {
        "T": T_fit, "S": S_fit,
        "s_model": s_model,
        "stats": stats, "deriv_stats": dstats,
        "deriv_obs": deriv_obs, "deriv_model": deriv_model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(t: np.ndarray, s_obs: np.ndarray, result: dict,
                 Q: float, r: float, L: float,
                 out_path: str = "theis_crt_analysis.png") -> None:
    if not HAS_PLOT:
        print("  matplotlib not available — skipping plot.")
        return

    T_fit    = result["T"]
    S_fit    = result["S"]
    s_model  = result["s_model"]
    stats    = result["stats"]
    d_obs    = result["deriv_obs"]
    d_model  = result["deriv_model"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Theis (1935) CRT Analysis — Confined Aquifer\n"
        f"T = {T_fit:.3e} m²/s   S = {S_fit:.3e}   "
        f"R² = {stats['R²']:.4f}   RMSE = {stats['RMSE (m)']:.4f} m   "
        f"NSE = {stats['NSE']:.4f}",
        fontsize=10, fontweight="bold",
    )

    # ── (1) Log–log: drawdown + Bourdet derivative ────────────────────────────
    ax = axes[0, 0]
    ax.loglog(t, s_obs,   "o",  color="steelblue",      ms=5,  label="Observed Δs",      zorder=4)
    ax.loglog(t, s_model, "-",  color="firebrick",       lw=2,  label="Theis model Δs")
    m1 = ~np.isnan(d_obs)   & (d_obs   > 0)
    m2 = ~np.isnan(d_model) & (d_model > 0)
    if m1.any():
        ax.loglog(t[m1], d_obs[m1],   "s",  color="cornflowerblue",
                  ms=5, mfc="none",   label=f"Obs Δs' (L={L})")
    if m2.any():
        ax.loglog(t[m2], d_model[m2], "--", color="salmon",
                  lw=1.5,             label="Model Δs'")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Δs  /  Δs'  (m)")
    ax.set_title("Log–Log Diagnostic (Bourdet derivative)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # ── (2) Semi-log drawdown ─────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.semilogx(t, s_obs,   "o", color="steelblue",  ms=5, label="Observed")
    ax.semilogx(t, s_model, "-", color="firebrick",   lw=2, label="Theis model")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Drawdown (m)")
    ax.set_title("Semi-Log Drawdown")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # ── (3) 1:1 observed vs modelled ─────────────────────────────────────────
    ax = axes[1, 0]
    ax.scatter(s_model, s_obs, c="steelblue", s=25, alpha=0.75, zorder=3)
    lo = min(s_obs.min(), s_model.min()) * 0.85
    hi = max(s_obs.max(), s_model.max()) * 1.10
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1 : 1")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Modelled Drawdown (m)")
    ax.set_ylabel("Observed Drawdown (m)")
    ax.set_title(f"Observed vs Modelled  (R² = {stats['R²']:.4f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (4) Residuals vs time ─────────────────────────────────────────────────
    ax = axes[1, 1]
    residuals = s_obs - s_model
    ax.semilogx(t, residuals, "o", color="darkorange", ms=5)
    ax.axhline(0, color="k", lw=1, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual  obs − model  (m)")
    ax.set_title(f"Residuals  (RMSE = {stats['RMSE (m)']:.4f} m)")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (for testing without a CSV)
# ─────────────────────────────────────────────────────────────────────────────

def generate_test_data(T: float = 1e-3, S: float = 1e-4,
                       Q: float = 1e-3, r: float = 50.0,
                       t_min: float = 60, t_max: float = 86400,
                       n_pts: int = 40, noise_std: float = 0.005,
                       seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, s_noisy) arrays for a synthetic Theis CRT."""
    t = np.logspace(np.log10(t_min), np.log10(t_max), n_pts)
    s = theis_drawdown(t, T, S, Q, r)
    rng = np.random.default_rng(seed)
    s_noisy = np.maximum(s + rng.normal(0, noise_std, size=n_pts), 0.0)
    return t, s_noisy


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Theis CRT analysis with Bourdet derivative and model-fit reporting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",   help="CSV file: time(s), drawdown(m)")
    p.add_argument("--Q",      type=float, required=True,
                   help="Pumping rate (m³/s)")
    p.add_argument("--r",      type=float, required=True,
                   help="Distance from pumping well to observation well (m)")
    p.add_argument("--T_init", type=float, default=1e-3,
                   help="Initial T guess (m²/s)")
    p.add_argument("--S_init", type=float, default=1e-4,
                   help="Initial S guess (-)")
    p.add_argument("--L",      type=float, default=0.8,
                   help="Bourdet smoothing factor (natural-log cycles)")
    p.add_argument("--plot",   action="store_true",
                   help="Save diagnostic plot PNG")
    p.add_argument("--out",    default="theis_crt_analysis.png",
                   help="Output PNG filename (requires --plot)")
    p.add_argument("--demo",   action="store_true",
                   help="Run with built-in synthetic data (no CSV needed)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.demo or args.data is None:
        if args.data is None and not args.demo:
            print("No --data file given; running in demo mode with synthetic data.")
        print("  Generating synthetic Theis CRT data …")
        t, s_obs = generate_test_data(
            T=args.T_init, S=args.S_init, Q=args.Q, r=args.r
        )
    else:
        path = Path(args.data)
        if not path.exists():
            sys.exit(f"ERROR: data file not found: {path}")
        t, s_obs = load_data(str(path))

    result = run_analysis(
        t, s_obs,
        Q=args.Q, r=args.r,
        T_init=args.T_init, S_init=args.S_init,
        L=args.L,
    )

    if args.plot:
        plot_results(t, s_obs, result, Q=args.Q, r=args.r,
                     L=args.L, out_path=args.out)

    return result


if __name__ == "__main__":
    main()
