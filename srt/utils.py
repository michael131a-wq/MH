"""Unit-conversion helpers and reporting utilities."""
from __future__ import annotations

from typing import List

import numpy as np

from .models import SRTTestData, AnalysisResults


# ── unit conversions ──────────────────────────────────────────────────────────
def lps_to_m3h(q_lps: float) -> float:
    return q_lps * 3.6

def m3h_to_lps(q_m3h: float) -> float:
    return q_m3h / 3.6

def gpm_to_m3h(q_gpm: float) -> float:
    return q_gpm * 0.22712

def m3h_to_gpm(q_m3h: float) -> float:
    return q_m3h / 0.22712

def ft_to_m(v: float) -> float:
    return v * 0.3048

def m_to_ft(v: float) -> float:
    return v / 0.3048


# ── print report ─────────────────────────────────────────────────────────────
def print_report(data: SRTTestData, results_dict: dict[str, AnalysisResults]) -> None:
    sep = "=" * 64
    print(sep)
    print(f"  Step Rate Test — Well: {data.well_id}   ({data.test_date})")
    print(sep)

    print(f"\n  Static Water Level : {data.static_water_level:.2f} m bgl")
    print(f"  Aquifer Thickness  : {data.aquifer_thickness:.1f} m")
    print(f"  Well Radius        : {data.well_radius:.3f} m")
    print(f"  Number of Steps    : {data.n_steps}")

    print("\n  Per-step summary:")
    print(f"  {'Step':>4}  {'Q (m³/h)':>10}  {'Duration':>10}  "
          f"{'sw (m)':>8}  {'sw/Q':>10}  {'Sp.Cap.':>10}")
    print("  " + "-" * 60)
    for s in data.steps:
        print(f"  {s.step_number:>4}  {s.flow_rate:>10.2f}  "
              f"{s.duration:>9.0f}m  {s.end_drawdown:>8.3f}  "
              f"{s.specific_drawdown:>10.5f}  {1/s.specific_drawdown:>10.4f}")

    print()
    for name, res in results_dict.items():
        print(f"\n{res.summary()}")
        print("  Well efficiency at each step:")
        for s in data.steps:
            eff = res.well_efficiency(s.flow_rate)
            print(f"    Step {s.step_number}  Q={s.flow_rate:.1f}  E={eff:.1f}%")

    # Optimal rates
    print(f"\n  Optimal pumping rates (Jacob method, if available):")
    if "jacob" in results_dict:
        res = results_dict["jacob"]
        for frac in [0.25, 0.50, 0.75]:
            max_sw = frac * data.static_water_level
            Q_opt = res.optimal_rate(max_sw)
            print(f"    Allowable drawdown {max_sw:.1f} m  →  Q_opt = {Q_opt:.2f} m³/h")

    print("\n" + sep)
