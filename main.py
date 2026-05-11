"""
Step Rate Test (SRT) Analysis — Hydrogeology
============================================
Thesis solution demonstrating all three analysis methods with the
example well dataset included in data/example_srt.csv.

Usage
-----
    python main.py                    # uses built-in example data
    python main.py data/my_well.csv   # loads your own CSV file
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless rendering; remove for interactive display

from srt import (
    PumpingStep, SRTTestData,
    JacobAnalysis, RorabaughAnalysis, EdenHazelAnalysis,
    load_from_csv, save_results_to_csv, print_report,
)
from srt.plots import plot_comprehensive_report, plot_jacob_analysis, plot_well_efficiency


# ── build-in example dataset ──────────────────────────────────────────────────
def _example_data() -> SRTTestData:
    steps = [
        PumpingStep(1,  5.0, 60,  2.15),
        PumpingStep(2, 10.0, 60,  5.40),
        PumpingStep(3, 15.0, 60, 10.25),
        PumpingStep(4, 20.0, 60, 17.60),
        PumpingStep(5, 25.0, 60, 27.35),
    ]
    return SRTTestData(
        well_id="BH-01",
        test_date="2024-03-15",
        static_water_level=12.5,
        aquifer_thickness=50.0,
        well_radius=0.15,
        steps=steps,
    )


def main() -> None:
    # ── load data ─────────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
        print(f"Loading data from: {csv_path}")
        data = load_from_csv(csv_path)
    else:
        print("Using built-in example dataset (BH-01).")
        data = _example_data()

    # ── run analyses ──────────────────────────────────────────────────────────
    jacob      = JacobAnalysis(data)
    rorabaugh  = RorabaughAnalysis(data)
    eden_hazel = EdenHazelAnalysis(data)

    results = {
        "jacob":      jacob.results,
        "rorabaugh":  rorabaugh.results,
        "eden_hazel": eden_hazel.results,
    }

    # ── print console report ──────────────────────────────────────────────────
    print_report(data, results)

    # ── save plots ────────────────────────────────────────────────────────────
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    fig1 = plot_comprehensive_report(
        data, list(results.values()),
        save_path=out_dir / f"{data.well_id}_comprehensive.png",
    )
    fig2 = plot_jacob_analysis(
        data, jacob.results,
        save_path=out_dir / f"{data.well_id}_jacob.png",
    )
    fig3 = plot_well_efficiency(
        data, list(results.values()),
        save_path=out_dir / f"{data.well_id}_efficiency.png",
    )

    # ── save CSV results ──────────────────────────────────────────────────────
    csv_out = out_dir / f"{data.well_id}_results.csv"
    save_results_to_csv(data, results, csv_out)

    print(f"\nOutputs written to '{out_dir}/':")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
