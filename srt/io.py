"""CSV I/O helpers for SRT test data."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from .models import PumpingStep, SRTTestData


def load_from_csv(path: str | Path) -> SRTTestData:
    """
    Load SRT data from a CSV file.

    Expected columns (header row required):
        step,flow_rate_m3h,duration_min,end_drawdown_m

    Metadata columns (optional, read from first data row):
        well_id, test_date, swl_m, thickness_m, radius_m
    """
    path = Path(path)
    steps: List[PumpingStep] = []

    meta = {
        "well_id": path.stem,
        "test_date": "unknown",
        "swl_m": 0.0,
        "thickness_m": 1.0,
        "radius_m": 0.15,
    }

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # optional metadata columns
            for key in meta:
                if key in row and row[key]:
                    try:
                        meta[key] = float(row[key]) if key not in ("well_id", "test_date") else row[key]
                    except ValueError:
                        meta[key] = row[key]

            steps.append(
                PumpingStep(
                    step_number=int(row["step"]),
                    flow_rate=float(row["flow_rate_m3h"]),
                    duration=float(row["duration_min"]),
                    end_drawdown=float(row["end_drawdown_m"]),
                )
            )

    return SRTTestData(
        well_id=str(meta["well_id"]),
        test_date=str(meta["test_date"]),
        static_water_level=float(meta["swl_m"]),
        aquifer_thickness=float(meta["thickness_m"]),
        well_radius=float(meta["radius_m"]),
        steps=steps,
    )


def save_results_to_csv(
    data: SRTTestData,
    results_dict: dict,
    path: str | Path,
) -> None:
    """Write per-step results (observed + predicted drawdowns, efficiency) to CSV."""
    path = Path(path)
    method_names = list(results_dict.keys())

    with open(path, "w", newline="") as fh:
        fieldnames = (
            ["step", "flow_rate_m3h", "duration_min",
             "observed_drawdown_m", "specific_drawdown"]
            + [f"predicted_{m}" for m in method_names]
            + [f"efficiency_pct_{m}" for m in method_names]
        )
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for step in data.steps:
            row: dict = {
                "step": step.step_number,
                "flow_rate_m3h": step.flow_rate,
                "duration_min": step.duration,
                "observed_drawdown_m": step.end_drawdown,
                "specific_drawdown": f"{step.specific_drawdown:.5f}",
            }
            for name, res in results_dict.items():
                row[f"predicted_{name}"] = f"{res.predict_drawdown(step.flow_rate):.4f}"
                row[f"efficiency_pct_{name}"] = f"{res.well_efficiency(step.flow_rate):.2f}"
            writer.writerow(row)
