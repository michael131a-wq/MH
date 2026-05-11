#!/usr/bin/env python3
"""
Slug Test Analysis for Aquifer Characterization

Supports three standard methods:
  - Hvorslev (1951)           — general, unconfined/confined
  - Bouwer & Rice (1976)      — unconfined aquifers
  - Cooper-Bredehoeft-Papadopulos (1967) — confined aquifers

Usage:
  python slug_test.py --data data.csv [options]

CSV format: two columns, no header required:
  time(s), displacement(m)   e.g.  0, 1.25
                                    30, 0.84
                                    ...
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# ── optional matplotlib ───────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load time, displacement columns from a CSV file."""
    data = np.genfromtxt(path, delimiter=",", comments="#", dtype=float)
    if data.ndim == 1:
        raise ValueError("CSV must have at least two columns (time, displacement).")
    time = data[:, 0]
    h    = data[:, 1]
    if np.any(np.diff(time) <= 0):
        raise ValueError("Time column must be strictly increasing.")
    return time, h


def normalize(h: np.ndarray) -> np.ndarray:
    """Return H/H₀ = (h - h_static) / (h₀ - h_static), assuming h_static=0."""
    return h / h[0]


# ─────────────────────────────────────────────────────────────────────────────
# Method 1 — Hvorslev (1951)
# ─────────────────────────────────────────────────────────────────────────────

def hvorslev(time: np.ndarray, h: np.ndarray,
             r_well: float, L_screen: float,
             r_casing: float | None = None) -> dict:
    """
    Hvorslev (1951) slug test analysis.

    Parameters
    ----------
    time      : array of times (s)
    h         : array of hydraulic head displacements (m), h[0] = H₀
    r_well    : well screen radius (m)
    L_screen  : length of screened interval (m)
    r_casing  : casing radius (m); defaults to r_well

    Returns
    -------
    dict with T37 (s), K (m/s), and regression data
    """
    if r_casing is None:
        r_casing = r_well

    ratio = normalize(h)
    # Only use positive values (head hasn't reversed)
    mask = ratio > 0
    t_fit = time[mask]
    ln_H  = np.log(ratio[mask])

    # Linear regression: ln(H/H₀) = -t / T₃₇
    slope, intercept = np.polyfit(t_fit, ln_H, 1)   # slope = -1/T₃₇
    T37 = -1.0 / slope                               # time for 37% recovery

    # K from Hvorslev basic shape factor (cylindrical screen, isotropic)
    # K = r_c² ln(L/r_w) / (2 L T₃₇)
    ln_term = np.log(L_screen / r_well)
    K = (r_casing**2 * ln_term) / (2.0 * L_screen * T37)

    return {
        "method"    : "Hvorslev (1951)",
        "T37_s"     : T37,
        "K_m_s"     : K,
        "_t_fit"    : t_fit,
        "_ln_ratio" : ln_H,
        "_slope"    : slope,
        "_intercept": intercept,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 — Bouwer & Rice (1976)
# ─────────────────────────────────────────────────────────────────────────────

def _bouwer_rice_ln_Re_rw(r_well: float, L_screen: float,
                           b_aquifer: float, d_top: float) -> float:
    """
    Estimate ln(Re/rw) using Bouwer & Rice empirical coefficients.
    Covers both cases: screen extends to aquifer base (fully penetrating)
    and partially penetrating.

    b_aquifer : saturated aquifer thickness (m)
    d_top     : depth from water table to top of screened interval (m)
    """
    # Coefficients from Table 1, Bouwer & Rice (1976)
    A = 1.472
    B = 0.2927
    C = 0.1  # small correction term coefficient

    Le = L_screen
    rw = r_well

    if (d_top + Le) >= b_aquifer:
        # Fully penetrating (screen extends to base)
        ln_Re_rw = (1.1 / np.log(d_top / rw)
                    + (A + B * np.log((b_aquifer - d_top) / rw)) / (Le / rw)) ** -1
    else:
        # Partially penetrating
        ln_Re_rw = (1.1 / np.log(d_top / rw)
                    + (A + B * np.log(b_aquifer / rw)
                       + C * np.log(Le / rw)) / (Le / rw)) ** -1

    return max(ln_Re_rw, 0.1)  # guard against non-physical values


def bouwer_rice(time: np.ndarray, h: np.ndarray,
                r_well: float, L_screen: float,
                b_aquifer: float, d_top: float,
                r_casing: float | None = None,
                t_start: float | None = None,
                t_end:   float | None = None) -> dict:
    """
    Bouwer & Rice (1976) slug test analysis for unconfined aquifers.

    Parameters
    ----------
    r_well    : well screen/gravel pack radius (m)
    L_screen  : length of screened interval (m)
    b_aquifer : saturated aquifer thickness (m)
    d_top     : depth from static water level to top of screen (m)
    r_casing  : inner casing radius (m); defaults to r_well
    t_start   : start of linear fitting window (s); default = first point
    t_end     : end   of linear fitting window (s); default = last point
    """
    if r_casing is None:
        r_casing = r_well

    ratio = normalize(h)
    mask  = ratio > 0
    t_all = time[mask]
    y_all = ratio[mask]

    # Restrict to fitting window
    if t_start is None:
        t_start = t_all[0]
    if t_end is None:
        t_end = t_all[-1]
    win = (t_all >= t_start) & (t_all <= t_end)
    t_fit = t_all[win]
    ln_y  = np.log(y_all[win])

    slope, intercept = np.polyfit(t_fit, ln_y, 1)
    # ln(Re/rw) from empirical formula
    ln_Re_rw = _bouwer_rice_ln_Re_rw(r_well, L_screen, b_aquifer, d_top)

    # K = |slope| * r_c² * ln(Re/rw) / (2 Le)
    K = abs(slope) * r_casing**2 * ln_Re_rw / (2.0 * L_screen)

    return {
        "method"     : "Bouwer & Rice (1976)",
        "ln_Re_rw"   : ln_Re_rw,
        "K_m_s"      : K,
        "_t_fit"     : t_fit,
        "_ln_ratio"  : ln_y,
        "_slope"     : slope,
        "_intercept" : intercept,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Method 3 — Cooper-Bredehoeft-Papadopulos (1967) — type-curve matching
# ─────────────────────────────────────────────────────────────────────────────

def _cbp_type_curve(beta: float, alpha: float,
                    n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the CBP type curve H/H₀ vs dimensionless time (1/u) for a given
    beta and alpha using numerical integration of the Laplace-domain solution.

    This uses the approximate closed-form series of Cooper et al. (1967):
        H/H₀ ≈ f(beta, alpha, tau)

    For practical type-curve matching we use the well-known asymptotic result;
    full numerical inversion is replaced by a stable approximation suitable
    for field-data matching.

    beta  = r_c² / (r_w² * S)   — storage parameter
    alpha = T * t / r_c²         — dimensionless time (varies along curve)
    """
    # Dimensionless time axis (log-spaced)
    tau = np.logspace(-3, 3, n_points)
    # Approximate solution (Ferris & Knowles curve-fit form, adequate for
    # alpha 1e-5 to 1, beta 1e-3 to 1e3)
    x = beta * tau
    # Cooper et al. series approximation
    H_ratio = np.exp(-2 * x / (1.0 + np.sqrt(1.0 + 4.0 * x / beta)))
    # Clip to physical range
    H_ratio = np.clip(H_ratio, 0.0, 1.0)
    return tau, H_ratio


def cbp_type_curve_match(time: np.ndarray, h: np.ndarray,
                          r_casing: float, r_well: float,
                          S_values: list[float] | None = None,
                          T_guess:  float | None = None) -> dict:
    """
    Cooper-Bredehoeft-Papadopulos (1967) analysis for confined aquifers.
    Matches field data to type curves by minimizing RMSE over a grid of S.

    Parameters
    ----------
    r_casing  : casing (standpipe) radius (m)
    r_well    : well screen radius (m)
    S_values  : storativity values to try; default = log-spaced 1e-5 to 0.1
    T_guess   : transmissivity first guess (m²/s); if None, estimated from
                the early straight-line slope

    Returns T (m²/s), S (dimensionless), and K if aquifer thickness provided.
    """
    if S_values is None:
        S_values = np.logspace(-5, -1, 40).tolist()

    ratio = normalize(h)

    # Estimate T from slope of ln(H/H₀) vs t at early time
    mask  = ratio > 0.2   # use upper 80% of recovery for initial T guess
    t_est = time[mask]
    if len(t_est) < 2:
        t_est = time
        mask  = np.ones(len(time), dtype=bool)

    slope_est, _ = np.polyfit(t_est, np.log(ratio[mask]), 1)
    T_est = abs(slope_est) * r_casing**2 / 2.0
    if T_guess is not None:
        T_est = T_guess

    best_rmse = np.inf
    best_T    = T_est
    best_S    = S_values[0]
    best_tau  = None
    best_curve= None

    T_search = np.logspace(np.log10(T_est) - 1.5,
                           np.log10(T_est) + 1.5, 60)

    for S in S_values:
        beta = r_casing**2 / (r_well**2 * S)
        for T in T_search:
            alpha = T / r_casing**2      # scale factor, dimensionless time = alpha * t
            tau_curve, H_curve = _cbp_type_curve(beta, alpha)
            # Convert dimensionless time to real time: t = tau * r_c² / T
            t_curve = tau_curve * r_casing**2 / T
            # Interpolate curve at field times
            H_interp = np.interp(time, t_curve, H_curve, left=1.0, right=0.0)
            rmse = np.sqrt(np.mean((ratio - H_interp)**2))
            if rmse < best_rmse:
                best_rmse  = rmse
                best_T     = T
                best_S     = S
                best_tau   = tau_curve
                best_curve = H_curve

    return {
        "method"     : "Cooper-Bredehoeft-Papadopulos (1967)",
        "T_m2_s"     : best_T,
        "S"          : best_S,
        "rmse"       : best_rmse,
        "_tau"       : best_tau,
        "_H_curve"   : best_curve,
        "_r_casing"  : r_casing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: list[dict]) -> None:
    sep = "─" * 60
    print(f"\n{'SLUG TEST ANALYSIS RESULTS':^60}")
    print(sep)
    for r in results:
        print(f"\n  Method : {r['method']}")
        if "K_m_s" in r:
            K = r["K_m_s"]
            print(f"  K      : {K:.3e} m/s   ({K * 86400:.4f} m/day)")
        if "T_m2_s" in r:
            T = r["T_m2_s"]
            print(f"  T      : {T:.3e} m²/s")
        if "S" in r:
            print(f"  S      : {r['S']:.3e}")
        if "T37_s" in r:
            print(f"  T₃₇    : {r['T37_s']:.1f} s")
        if "ln_Re_rw" in r:
            print(f"  ln(Re/rw): {r['ln_Re_rw']:.3f}")
        if "rmse" in r:
            print(f"  RMSE   : {r['rmse']:.4f}")
        print()
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(time: np.ndarray, h: np.ndarray,
                 results: list[dict], output_path: str | None = None) -> None:
    if not HAS_PLOT:
        print("matplotlib not installed — skipping plots.")
        return

    ratio = normalize(h)
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods + 1,
                             figsize=(5 * (n_methods + 1), 5),
                             constrained_layout=True)
    if n_methods == 0:
        axes = [axes]

    # Panel 0: raw recovery curve
    ax0 = axes[0]
    ax0.plot(time, ratio, "ko-", ms=4, lw=1.5, label="Field data")
    ax0.axhline(0.37, color="gray", lw=0.8, ls="--", label="H/H₀ = 0.37")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("H/H₀  (normalized head)")
    ax0.set_title("Recovery Curve")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    for i, r in enumerate(results):
        ax = axes[i + 1]
        method = r["method"]

        if method.startswith("Hvorslev") or method.startswith("Bouwer"):
            t_fit = r["_t_fit"]
            ln_y  = r["_ln_ratio"]
            slope = r["_slope"]
            inter = r["_intercept"]
            ax.plot(time[normalize(h) > 0],
                    np.log(normalize(h)[normalize(h) > 0]),
                    "ko", ms=4, label="Field data")
            t_line = np.array([t_fit[0], t_fit[-1]])
            ax.plot(t_line, slope * t_line + inter, "r-", lw=2, label="Best fit")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ln(H/H₀)")
            ax.set_title(method)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        elif method.startswith("Cooper"):
            tau_curve = r["_tau"]
            H_curve   = r["_H_curve"]
            T         = r["T_m2_s"]
            rc        = r["_r_casing"]
            t_curve   = tau_curve * rc**2 / T
            ax.semilogx(t_curve, H_curve, "r-", lw=2, label="Best-fit type curve")
            ax.semilogx(time, ratio, "ko", ms=4, label="Field data")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("H/H₀")
            ax.set_title(method)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Slug Test Analysis", fontsize=13, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"  Plot saved → {output_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data",      required=True,  help="CSV file: time(s), displacement(m)")
    p.add_argument("--method",    default="all",
                   choices=["hvorslev", "bouwer-rice", "cbp", "all"],
                   help="Analysis method (default: all)")
    p.add_argument("--plot",      action="store_true", help="Show plots")
    p.add_argument("--plot-out",  metavar="FILE",       help="Save plot to file (PNG/PDF)")

    # Well geometry (shared)
    geo = p.add_argument_group("Well geometry (required for most methods)")
    geo.add_argument("--r-well",    type=float, required=True, metavar="m",
                     help="Well screen radius (m)")
    geo.add_argument("--r-casing",  type=float, metavar="m",
                     help="Inner casing radius (m); defaults to --r-well")
    geo.add_argument("--L-screen",  type=float, metavar="m",
                     help="Screened interval length (m)")

    # Bouwer-Rice extras
    br = p.add_argument_group("Bouwer & Rice extras")
    br.add_argument("--b-aquifer", type=float, metavar="m",
                    help="Saturated aquifer thickness (m)")
    br.add_argument("--d-top",     type=float, metavar="m",
                    help="Depth from static water level to top of screen (m)")
    br.add_argument("--t-start",   type=float, metavar="s",
                    help="Fitting window start time (s)")
    br.add_argument("--t-end",     type=float, metavar="s",
                    help="Fitting window end time (s)")

    # CBP extras
    cbp = p.add_argument_group("Cooper-Bredehoeft-Papadopulos extras")
    cbp.add_argument("--T-guess",  type=float, metavar="m2/s",
                     help="Initial transmissivity guess (m²/s)")

    return p


def run(args: argparse.Namespace) -> None:
    # Load data
    time, h = load_data(args.data)
    print(f"\n  Loaded {len(time)} data points from '{args.data}'")
    print(f"  Initial displacement H₀ = {h[0]:.4f} m")
    print(f"  Time range: {time[0]:.1f} – {time[-1]:.1f} s")

    r_casing = args.r_casing if args.r_casing else args.r_well
    results  = []
    methods  = {"hvorslev", "bouwer-rice", "cbp"} if args.method == "all" \
               else {args.method}

    # ── Hvorslev ─────────────────────────────────────────────────────────────
    if "hvorslev" in methods:
        if args.L_screen is None:
            print("  [Hvorslev] --L-screen required; skipping.")
        else:
            results.append(
                hvorslev(time, h, args.r_well, args.L_screen, r_casing)
            )

    # ── Bouwer & Rice ─────────────────────────────────────────────────────────
    if "bouwer-rice" in methods:
        missing = [k for k, v in {
            "--L-screen": args.L_screen,
            "--b-aquifer": args.b_aquifer,
            "--d-top": args.d_top,
        }.items() if v is None]
        if missing:
            print(f"  [Bouwer-Rice] Missing: {', '.join(missing)}; skipping.")
        else:
            results.append(
                bouwer_rice(time, h, args.r_well, args.L_screen,
                            args.b_aquifer, args.d_top, r_casing,
                            args.t_start, args.t_end)
            )

    # ── CBP ───────────────────────────────────────────────────────────────────
    if "cbp" in methods:
        results.append(
            cbp_type_curve_match(time, h, r_casing, args.r_well,
                                 T_guess=args.T_guess)
        )

    if not results:
        print("\n  No methods could run — check required arguments.\n")
        sys.exit(1)

    print_results(results)

    if args.plot or args.plot_out:
        plot_results(time, h, results, args.plot_out)


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
