#!/usr/bin/env python3
"""
survey_processor.py

Process raw SURVEY-IQ downhole geophysical survey exports into Acquire-ready
format and generate a gamma log figure.

Usage:
    python survey_processor.py <survey.csv> <log.las>
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    print(f"ERROR: {exc}\nInstall: pip install pandas numpy", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend; must precede pyplot import
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("ERROR: matplotlib is required.\nInstall: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def strip_trailing_zeros(s: str) -> str:
    """Strip trailing zeros after a decimal point; also remove a bare trailing '.'.

    >>> strip_trailing_zeros('-89.920')
    '-89.92'
    >>> strip_trailing_zeros('318.500')
    '318.5'
    >>> strip_trailing_zeros('0.123000000')
    '0.123'
    >>> strip_trailing_zeros('100')
    '100'
    """
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def fmt_earth_rate_delta(raw: str) -> str:
    """Round to 9 decimal places then strip trailing zeros.

    SURVEY-IQ Earth Rate Delta values carry 16–18 decimal places that are pure
    floating-point artefacts.  Only 9 dp are meaningful.

    Non-numeric tokens (e.g. 'NA') are returned unchanged.

    >>> fmt_earth_rate_delta('0.000123456789012345678')
    '0.000123457'
    >>> fmt_earth_rate_delta('0.123000000012345')
    '0.123'
    >>> fmt_earth_rate_delta('NA')
    'NA'
    """
    s = raw.strip()
    if not s or s in ("nan", "NaN", "NA", "N/A"):
        return s
    try:
        rounded = Decimal(s).quantize(Decimal("0.000000001"), rounding=ROUND_HALF_UP)
        return strip_trailing_zeros(str(rounded))
    except InvalidOperation:
        return s


def fmt_strip_zeros_3dp(raw: str) -> str:
    """Round to 3 dp then strip trailing zeros.

    SURVEY-IQ exports Dip, Azimuth, Gravity TF, and Vertical TF zero-padded
    to 3 decimal places.  Acquire rejects the padded form.

    Non-numeric tokens are returned unchanged.

    >>> fmt_strip_zeros_3dp('-89.920')
    '-89.92'
    >>> fmt_strip_zeros_3dp('318.500')
    '318.5'
    >>> fmt_strip_zeros_3dp('45.000')
    '45'
    >>> fmt_strip_zeros_3dp('NA')
    'NA'
    """
    s = raw.strip()
    if not s or s in ("nan", "NaN", "NA", "N/A"):
        return s
    try:
        rounded = Decimal(s).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        return strip_trailing_zeros(str(rounded))
    except InvalidOperation:
        return s


# Values treated as "no active flag" inside a QC cell.
_QC_EMPTY = frozenset({"", "0", "NA", "N/A", "nan", "NaN", "NONE", "None", "none"})


def count_qc_entries(value: str) -> int:
    """Count the number of flag tokens in a single QC cell.

    Tokens may be separated by commas, pipes, or semicolons.  Empty, 'NA',
    'NONE', or '0' cells (and any split token that matches those) count as
    zero flags.

    >>> count_qc_entries('ACCEL_WARN,TEMP_HIGH')
    2
    >>> count_qc_entries('ACCEL_WARN')
    1
    >>> count_qc_entries('NA')
    0
    >>> count_qc_entries('')
    0
    """
    s = str(value).strip()
    if s in _QC_EMPTY:
        return 0
    parts = re.split(r"[,|;]", s)
    return sum(1 for p in parts if p.strip() and p.strip() not in _QC_EMPTY)


def find_qc_column(df: pd.DataFrame) -> str | None:
    """Return the name of the QC column, or None if no match is found.

    Checks for an exact match on 'QC' first, then common full names, then any
    column whose name starts with 'QC'.
    """
    exact = {"QC", "QC FLAGS", "QC FLAG", "QUALITY FLAGS", "QUALITY FLAG"}
    for col in df.columns:
        if col.strip().upper() in exact:
            return col
    for col in df.columns:
        if col.strip().upper().startswith("QC"):
            return col
    return None


def date_from_filename(name: str) -> str:
    """Extract a DD-MM-YYYY segment from *name* and return it as DDMMYYYY.

    >>> date_from_filename('BH001_15-03-2024_multishot.csv')
    '15032024'
    """
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", name)
    if not m:
        raise ValueError(
            f"Cannot find a DD-MM-YYYY date in filename {name!r}. "
            "Expected something like '15-03-2024'."
        )
    dd, mm, yyyy = m.groups()
    return f"{dd}{mm}{yyyy}"


# ---------------------------------------------------------------------------
# LAS text helpers
# ---------------------------------------------------------------------------

_LOGU_BLANK_RE = re.compile(r"^(\s*LOGU\.\S*)(\s+)(:.*)")
_EXPORT_VERSION_RE = re.compile(r"^\s*EXPORTED\s+FROM\s+APP\s+VERSION\s*:", re.IGNORECASE)
_NOTES_RE = re.compile(r"^\s*NOTES\s*:", re.IGNORECASE)


def parse_las_field(line: str) -> tuple | None:
    """Parse one LAS parameter / curve line.

    LAS 2.0 format:  MNEM.UNIT  VALUE : DESCRIPTION

    The unit is present only when a non-whitespace character appears
    immediately after the dot.  If the first character after '.' is a space,
    the unit is empty and everything before ':' is the value.

    Returns (mnemonic, unit, value, description) or None for comments/blanks.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    colon = line.find(":")
    dot = line.find(".")
    if colon == -1 or dot == -1 or dot > colon:
        return None
    mnem = line[:dot].strip()
    after_dot = line[dot + 1 : colon]
    desc = line[colon + 1 :].strip()
    # If the first character after the dot is non-whitespace, it is the unit.
    if after_dot and after_dot[0] not in (" ", "\t"):
        parts = after_dot.split(None, 1)
        unit = parts[0]
        value = parts[1].strip() if len(parts) > 1 else ""
    else:
        unit = ""
        value = after_dot.strip()
    return mnem, unit, value, desc


def _patch_logu(line: str) -> str:
    """Insert 'WREGAM081' into a LOGU. line whose value field is blank.

    Leaves the line unchanged when a value is already present.
    """
    m = _LOGU_BLANK_RE.match(line)
    if m:
        return m.group(1) + "              WREGAM081 " + m.group(3)
    return line


def parse_las(text: str) -> dict:
    """Parse a LAS 2.0 file into a structured dict.

    Returns::

        {
          'metadata': {MNEM: value, ...},  # from ~V, ~W, ~P sections
          'curves':   ['DEPT', 'GR', ...], # from ~C, in column order
          'null':     float,               # null/absent value (default -999.25)
          'data':     {MNEM: np.ndarray},  # from ~A, nulls replaced with NaN
        }

    Only LAS files with WRAP. NO are supported.
    """
    lines = text.splitlines()
    metadata: dict = {}
    curves: list = []
    data_lines: list = []

    in_meta = False
    in_curve = False
    in_data = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("~"):
            head = stripped.upper()
            in_meta  = head.startswith("~V") or head.startswith("~W") or head.startswith("~P")
            in_curve = head.startswith("~C")
            in_data  = head.startswith("~A")
            continue

        if stripped.startswith("#") or not stripped:
            continue

        if in_meta:
            parsed = parse_las_field(line)
            if parsed:
                mnem, _unit, value, _desc = parsed
                metadata[mnem.strip().upper()] = value

        elif in_curve:
            parsed = parse_las_field(line)
            if parsed:
                curves.append(parsed[0].strip().upper())

        elif in_data:
            data_lines.append(stripped)

    null_val = float(metadata.get("NULL", -999.25))
    n = len(curves)
    arrays: list = [[] for _ in range(n)]

    for dl in data_lines:
        parts = dl.split()
        for i in range(min(n, len(parts))):
            try:
                v = float(parts[i])
            except ValueError:
                v = float("nan")
            arrays[i].append(v)

    data: dict = {}
    for i, mnem in enumerate(curves):
        arr = np.array(arrays[i], dtype=float)
        # Mask null values (compare within a small tolerance)
        tol = max(abs(null_val) * 1e-4, 0.01)
        arr[np.abs(arr - null_val) < tol] = np.nan
        data[mnem] = arr

    return {"metadata": metadata, "curves": curves, "null": null_val, "data": data}


def find_gamma_column(curves: list) -> str:
    """Return the mnemonic of the gamma ray curve.

    Checks a priority list of common mnemonics, then falls back to any curve
    whose name contains 'GR' or 'GAMMA'.
    """
    priority = ["GR", "SGR", "CGR", "GAMMA", "NGAM", "GAPI", "GR_CORR", "GRC"]
    for cand in priority:
        if cand in curves:
            return cand
    for curve in curves:
        if "GR" in curve or "GAMMA" in curve:
            return curve
    raise ValueError(
        f"No gamma ray curve found in LAS file.  Available curves: {curves}"
    )


# ---------------------------------------------------------------------------
# Main processing functions
# ---------------------------------------------------------------------------

def process_csv(csv_path: Path, output_dir: Path) -> tuple:
    """Process a SURVEY-IQ multishot CSV export.

    Returns ``(output_path, hole_id)``.
    """
    date_str = date_from_filename(csv_path.name)

    # Read all columns as plain strings to control every aspect of formatting.
    # na_values=[] + keep_default_na=False prevents pandas from silently
    # converting literal 'NA' (used in Grid Azimuth) to float NaN.
    df = pd.read_csv(
        csv_path,
        dtype=str,
        na_values=[],
        keep_default_na=False,
    )

    # 1. Rename TN Azimuth → Azimuth
    if "TN Azimuth" in df.columns:
        df = df.rename(columns={"TN Azimuth": "Azimuth"})

    # 2. Strip leading RIG prefix from Rig column — only the literal prefix
    #    "RIG" (e.g. RIG276 → 276).  Values like BDC276 are left untouched
    #    because they do not start with the RIG token.
    if "Rig" in df.columns:
        df["Rig"] = df["Rig"].str.replace(r"^RIG", "", regex=True)

    # 3. Remove duplicate Measured Depth rows.
    #    Keep the row with the fewest entries in the QC column; break ties
    #    with the lowest (numerically smallest) Earth Rate Delta value.
    depth_col = "Measured Depth"
    erd_col   = "Earth Rate Delta"

    if depth_col in df.columns:
        qc_col = find_qc_column(df)

        # Temporary sort keys — never written to output.
        df["__depth_n"] = pd.to_numeric(df[depth_col], errors="coerce")
        df["__erd_n"] = (
            pd.to_numeric(df[erd_col], errors="coerce").fillna(0.0)
            if erd_col in df.columns
            else pd.Series(0.0, index=df.index)
        )
        df["__qc_n"] = (
            df[qc_col].apply(count_qc_entries)
            if qc_col is not None
            else 0
        )

        df = (
            df.sort_values(
                ["__depth_n", "__qc_n", "__erd_n"],
                ascending=True,
                kind="stable",
            )
            .drop_duplicates(subset=[depth_col], keep="first")
        )

        # Drop helpers and restore depth-ascending order.
        df = (
            df.drop(columns=["__depth_n", "__erd_n", "__qc_n"])
            .assign(__depth_n=lambda d: pd.to_numeric(d[depth_col], errors="coerce"))
            .sort_values("__depth_n", kind="stable")
            .drop(columns=["__depth_n"])
            .reset_index(drop=True)
        )

    # 4. Format Earth Rate Delta: 9 dp, strip trailing zeros.
    if erd_col in df.columns:
        df[erd_col] = df[erd_col].apply(fmt_earth_rate_delta)

    # 5. Strip trailing zeros from angular / toolface columns.
    for col in ("Dip", "Azimuth", "Gravity TF", "Vertical TF"):
        if col in df.columns:
            df[col] = df[col].apply(fmt_strip_zeros_3dp)

    # 6. Extract hole ID from Drillhole Name.
    hole_col = "Drillhole Name"
    if hole_col not in df.columns:
        raise ValueError(
            f"Required column {hole_col!r} not found.  "
            f"Columns present: {list(df.columns)}"
        )
    hole_id = df[hole_col].iloc[0].strip()

    # 7. Write with CRLF line endings (required by Acquire importer).
    out_name = f"{hole_id}_{date_str}_REFLEX.csv"
    out_path = output_dir / out_name
    df.to_csv(out_path, index=False, lineterminator="\r\n")

    return out_path, hole_id


def process_las(las_path: Path, hole_id: str, output_dir: Path) -> Path:
    """Process a SURVEY-IQ LAS gamma / geophysical log export."""
    text = las_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    in_param = False
    in_other = False
    out_lines: list = []

    for line in lines:
        head = line.strip().upper()

        if head.startswith("~"):
            in_param = head.startswith("~P")
            in_other = head.startswith("~O")
            out_lines.append(line)
            continue

        if in_param and re.match(r"\s*LOGU\.", line):
            line = _patch_logu(line)

        if in_other:
            if _EXPORT_VERSION_RE.match(line):
                continue   # drop "EXPORTED FROM APP VERSION: ..." line
            if _NOTES_RE.match(line):
                continue   # drop "NOTES: ..." line

        out_lines.append(line)

    out_text = "\n".join(out_lines)
    if text.endswith("\n"):
        out_text += "\n"

    out_name = f"{hole_id}_UP.las"
    out_path = output_dir / out_name
    out_path.write_text(out_text, encoding="utf-8")

    return out_path


def generate_gamma_figure(las_path: Path, hole_id: str, output_dir: Path) -> Path:
    """Generate a gamma log PNG from a processed LAS file.

    Layout
    ------
    - X-axis (Gamma Ray, API) on *top* of the plot.
    - X-axis upper limit is the max gamma value rounded up to the nearest 10.
    - Dashed vertical gridlines every 10 API units.
    - Y-axis is depth, 0 at the top, increasing downward, via an explicit
      ``set_ylim(max_depth, 0)`` — NOT ``invert_yaxis()``.
    - Major depth ticks every 10 m; minor ticks every 5 m.
    - Title: "{HOLEID} — Gamma Ray" with tool name and log date as subtitle.
    """
    text = las_path.read_text(encoding="utf-8", errors="replace")
    las = parse_las(text)

    curves   = las["curves"]
    data     = las["data"]
    metadata = las["metadata"]

    if not curves:
        raise ValueError("No curve information found in LAS file.")

    # Depth is always the first curve (LAS convention).
    depth_col = curves[0]
    gamma_col = find_gamma_column(curves)

    depth = data[depth_col]
    gamma = data[gamma_col]

    # Keep only rows where both depth and gamma are finite (non-null, non-NaN).
    valid = np.isfinite(depth) & np.isfinite(gamma)
    depth = depth[valid]
    gamma = gamma[valid]

    if depth.size == 0:
        raise ValueError(
            "No valid depth/gamma data found in LAS file after null-value removal."
        )

    # ── X-axis scaling ────────────────────────────────────────────────────
    # Round max gamma UP to the nearest 10 so the curve is never clipped.
    x_max = math.ceil(float(gamma.max()) / 10) * 10
    x_max = max(x_max, 10)   # guard against zero/near-zero gamma data

    # ── Y-axis range ──────────────────────────────────────────────────────
    y_top    = 0.0              # 0 is always at the top
    y_bottom = float(depth.max())

    # ── Figure size: narrow width, height scales with depth range ─────────
    depth_range = y_bottom - float(depth.min())
    fig_height  = max(6.0, min(24.0, depth_range / 8.0))
    fig, ax = plt.subplots(figsize=(4.5, fig_height))

    # ── Curve: line + light fill ──────────────────────────────────────────
    ax.plot(gamma, depth, color="steelblue", linewidth=0.8)
    ax.fill_betweenx(depth, 0.0, gamma, alpha=0.12, color="steelblue")

    # ── X-axis: ticks and label on TOP ────────────────────────────────────
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Gamma Ray (API)", labelpad=8)
    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    # Dashed vertical gridlines every 10 API units
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, color="gray", alpha=0.65)

    # ── Y-axis: depth increasing downward ─────────────────────────────────
    # Use explicit ylim — NOT invert_yaxis() — so 0 is always at the top.
    ax.set_ylim(y_bottom, y_top)
    ax.set_ylabel("Depth (m)")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.tick_params(axis="y", which="minor", length=3)

    # ── Title: hole ID + subtitle (tool name, log date) ───────────────────
    # Look for tool name and date in any order of likely LAS mnemonics.
    tool_name = (
        metadata.get("TOOL")
        or metadata.get("DEVI")
        or metadata.get("GDEV")
        or metadata.get("BSEL")
        or metadata.get("SRVC")
        or ""
    )
    log_date = (
        metadata.get("DATE")
        or metadata.get("LDAT")
        or metadata.get("DDAT")
        or ""
    )
    subtitle_parts = [p.strip() for p in (tool_name, log_date) if p.strip()]
    subtitle = "  |  ".join(subtitle_parts)

    title_text = f"{hole_id} — Gamma Ray"
    if subtitle:
        title_text = f"{title_text}\n{subtitle}"

    # Extra pad accommodates the x-axis ticks and label sitting above the plot.
    ax.set_title(title_text, fontsize=10, pad=50)

    # ── Save ──────────────────────────────────────────────────────────────
    out_name = f"{hole_id}_gamma.png"
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="survey_processor",
        description=(
            "Process raw SURVEY-IQ downhole geophysical exports into "
            "Acquire-ready format and generate a gamma log figure."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python survey_processor.py survey_15-03-2024.csv gammalog.las\n"
            "  python survey_processor.py data/BH001_15-03-2024.csv data/BH001.las\n"
        ),
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="SURVEY-IQ multishot CSV export (filename must contain DD-MM-YYYY)",
    )
    parser.add_argument(
        "las_file",
        type=Path,
        help="SURVEY-IQ LAS gamma / geophysical log export",
    )
    args = parser.parse_args()

    for path, label in ((args.csv_file, "CSV"), (args.las_file, "LAS")):
        if not path.is_file():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    output_dir = args.csv_file.resolve().parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    try:
        csv_out, hole_id = process_csv(args.csv_file, output_dir)
        print(f"  CSV → {csv_out}")
    except Exception as exc:
        print(f"ERROR processing CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        las_out = process_las(args.las_file, hole_id, output_dir)
        print(f"  LAS → {las_out}")
    except Exception as exc:
        print(f"ERROR processing LAS: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        png_out = generate_gamma_figure(las_out, hole_id, output_dir)
        print(f"  PNG → {png_out}")
    except Exception as exc:
        print(f"ERROR generating gamma figure: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
