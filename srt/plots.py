"""Matplotlib visualisations for SRT analysis results."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .models import SRTTestData, AnalysisResults

_COLORS = ["steelblue", "darkorange", "seagreen", "crimson"]


# ──────────────────────────────────────────────────────────────────────────────
def plot_jacob_analysis(
    data: SRTTestData,
    results: AnalysisResults,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Jacob's linear regression plot + drawdown component breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Jacob's Step-Drawdown Analysis — Well: {data.well_id}",
        fontsize=13, fontweight="bold",
    )

    Q = data.flow_rates
    sw = data.drawdowns
    sw_Q = data.specific_drawdowns
    Q_fit = np.linspace(0, max(Q) * 1.25, 200)

    # Left: sw/Q vs Q
    ax = axes[0]
    ax.scatter(Q, sw_Q, color="steelblue", s=80, zorder=5, label="Observed")
    ax.plot(
        Q_fit,
        results.B + results.C * Q_fit,
        "r-", linewidth=2,
        label=f"B={results.B:.5f}\nC={results.C:.7f}\nR²={results.r_squared:.4f}",
    )
    for i, (q, sq) in enumerate(zip(Q, sw_Q)):
        ax.annotate(f"Step {i+1}", (q, sq), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Pumping Rate Q (m³/h)")
    ax.set_ylabel("Specific Drawdown  sw/Q  (m/(m³/h))")
    ax.set_title("Jacob's Specific Drawdown Plot")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: component breakdown
    ax = axes[1]
    formation = results.B * Q_fit
    well_loss = results.C * Q_fit ** results.n
    total = formation + well_loss
    ax.fill_between(Q_fit, 0, formation, alpha=0.40, color="royalblue", label="Formation loss (BQ)")
    ax.fill_between(Q_fit, formation, total, alpha=0.40, color="tomato", label="Well loss (CQ²)")
    ax.plot(Q_fit, total, "k-", linewidth=2, label="Total drawdown")
    ax.scatter(Q, sw, color="black", s=60, zorder=5, label="Observed")
    ax.set_xlabel("Pumping Rate Q (m³/h)")
    ax.set_ylabel("Drawdown sw (m)")
    ax.set_title("Drawdown Component Breakdown")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_well_efficiency(
    data: SRTTestData,
    results_list: List[AnalysisResults],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Well efficiency curves for one or more analysis results."""
    fig, ax = plt.subplots(figsize=(8, 5))
    Q_range = np.linspace(0.1, max(data.flow_rates) * 1.5, 300)

    for i, res in enumerate(results_list):
        eff = [res.well_efficiency(q) for q in Q_range]
        ax.plot(Q_range, eff, linewidth=2, color=_COLORS[i % len(_COLORS)],
                label=res.method)

    ax.axhline(80, color="green", linestyle="--", alpha=0.7, label="80 % threshold")
    ax.axhline(70, color="orange", linestyle="--", alpha=0.7, label="70 % threshold")
    ax.set_xlabel("Pumping Rate Q (m³/h)")
    ax.set_ylabel("Well Efficiency (%)")
    ax.set_title(f"Well Efficiency — {data.well_id}")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_comprehensive_report(
    data: SRTTestData,
    results_list: List[AnalysisResults],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Four-panel report: time series · Jacob plot · components · efficiency."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    fig.suptitle(
        f"Step Rate Test — Comprehensive Analysis\nWell: {data.well_id}   Date: {data.test_date}",
        fontsize=13, fontweight="bold",
    )

    Q = data.flow_rates
    sw = data.drawdowns
    sw_Q = data.specific_drawdowns
    cum_t = data.cumulative_times
    Q_fit = np.linspace(0.01, max(Q) * 1.3, 200)
    best = max(results_list, key=lambda r: r.r_squared)

    # ── Panel 1: pumping steps time series ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    t_vals, s_vals, q_vals = [0.0], [0.0], [0.0]
    for step in data.steps:
        t_end = float(cum_t[step.step_number - 1])
        t_start = t_end - step.duration
        t_vals += [t_start, t_end]
        s_vals += [step.end_drawdown, step.end_drawdown]
        q_vals += [step.flow_rate, step.flow_rate]

    ax1_r = ax1.twinx()
    ax1.plot(t_vals, s_vals, "b-", linewidth=1.8, label="Drawdown")
    ax1_r.plot(t_vals, q_vals, "r--", linewidth=1.4, alpha=0.7, label="Q")
    ax1.set_xlabel("Elapsed Time (min)")
    ax1.set_ylabel("Drawdown (m)", color="steelblue")
    ax1_r.set_ylabel("Q (m³/h)", color="tomato")
    ax1.set_title("Test Time Series")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Jacob specific-drawdown plot ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(Q, sw_Q, color="steelblue", s=80, zorder=5)
    for res in results_list:
        if "Jacob" in res.method:
            ax2.plot(Q_fit, res.B + res.C * Q_fit, linewidth=2,
                     label=f"B={res.B:.5f}  C={res.C:.7f}\nR²={res.r_squared:.4f}")
    ax2.set_xlabel("Q (m³/h)")
    ax2.set_ylabel("sw/Q  (m/(m³/h))")
    ax2.set_title("Jacob's Specific Drawdown Plot")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: drawdown components (best method) ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    formation = best.B * Q_fit
    well_l = best.C * Q_fit ** best.n
    total = formation + well_l
    ax3.fill_between(Q_fit, 0, formation, alpha=0.40, color="royalblue", label="Formation loss")
    ax3.fill_between(Q_fit, formation, total, alpha=0.40, color="tomato", label="Well loss")
    ax3.plot(Q_fit, total, "k-", linewidth=2, label="Total (predicted)")
    ax3.scatter(Q, sw, color="black", s=60, zorder=5, label="Observed")
    ax3.set_xlabel("Q (m³/h)")
    ax3.set_ylabel("Drawdown (m)")
    ax3.set_title(f"Drawdown Components\n({best.method})")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: well efficiency ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for i, res in enumerate(results_list):
        eff = [res.well_efficiency(q) for q in Q_fit]
        ax4.plot(Q_fit, eff, linewidth=2, color=_COLORS[i % len(_COLORS)],
                 label=res.method.split("(")[0].strip())
    obs_eff = [best.well_efficiency(q) for q in Q]
    ax4.scatter(Q, obs_eff, color="black", s=60, zorder=5)
    ax4.axhline(80, color="green", linestyle="--", alpha=0.6, label="80 %")
    ax4.axhline(70, color="orange", linestyle="--", alpha=0.6, label="70 %")
    ax4.set_xlabel("Q (m³/h)")
    ax4.set_ylabel("Well Efficiency (%)")
    ax4.set_title("Well Efficiency Curve")
    ax4.set_ylim(0, 105)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
