#!/usr/bin/env python3
"""
reflex_processor.py

Process raw SURVEY-IQ downhole geophysical survey exports into the format
accepted by the Acquire database.

Usage:
    python reflex_processor.py <survey.csv> <log.las>

See README.md for full documentation.
"""

from __future__ import annotations

import argparse
import re
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print(
        "ERROR: pandas is required.  Install it with:  pip install pandas",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def strip_trailing_zeros(s: str) -> str:
    """Strip trailing zeros after a decimal point.

    Also removes a bare trailing decimal point ('1.' → '1').
    Non-decimal strings are returned unchanged.

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


# Values considered "not flagged / OK" when counting active QC flags.
_QC_OK_VALUES: frozenset = frozenset(
    {"", "0", "NA", "N/A", "nan", "NaN", "PASS", "Pass", "pass", "OK", "ok", "Ok"}
)


# ---------------------------------------------------------------------------
# CSV-specific helpers
# ---------------------------------------------------------------------------

def fmt_earth_rate_delta(raw: str) -> str:
    """Round *raw* to 9 decimal places then strip trailing zeros.

    SURVEY-IQ emits Earth Rate Delta values with 16–18 significant decimal
    places that are pure floating-point artefacts.  Rounding to 9 dp removes
    the noise while preserving the meaningful precision.

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
        rounded = Decimal(s).quantize(
            Decimal("0.000000001"), rounding=ROUND_HALF_UP
        )
        return strip_trailing_zeros(str(rounded))
    except InvalidOperation:
        return s


def fmt_strip_zeros_3dp(raw: str) -> str:
    """Round to 3 decimal places then strip trailing zeros.

    SURVEY-IQ exports Dip, Azimuth, Gravity TF, and Vertical TF with exactly
    3 decimal places, always zero-padded (e.g. '-89.920', '318.500').
    Acquire rejects the padded form, so the trailing zeros are removed.

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
        # Parse via Decimal to avoid float-precision noise before rounding.
        rounded = Decimal(s).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        return strip_trailing_zeros(str(rounded))
    except InvalidOperation:
        return s


def flag_columns(df: pd.DataFrame) -> list:
    """Return column names that look like QC flag fields.

    Matches columns whose names contain 'flag', 'qc', 'pass', 'fail', or
    'error' (case-insensitive).
    """
    keywords = ("flag", "qc", "pass", "fail", "error")
    return [c for c in df.columns if any(k in c.lower() for k in keywords)]


def active_flag_count(row: pd.Series, cols: list) -> int:
    """Count how many columns in *cols* hold an active / failed QC value."""
    return sum(1 for c in cols if str(row[c]).strip() not in _QC_OK_VALUES)


def date_from_filename(name: str) -> str:
    """Extract a DD-MM-YYYY segment from *name* and return it as DDMMYYYY.

    Raises ValueError if the pattern is not present.

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
# LAS-specific helpers
# ---------------------------------------------------------------------------

# Matches a LOGU. line whose value field is blank (only whitespace before ':').
# Group 1: LOGU. + optional unit characters (non-whitespace after dot)
# Group 2: the blank value field (whitespace)
# Group 3: colon + description
_LOGU_BLANK_RE = re.compile(r"^(\s*LOGU\.\S*)(\s+)(:.*)")

_EXPORT_VERSION_RE = re.compile(
    r"^\s*EXPORTED\s+FROM\s+APP\s+VERSION\s*:", re.IGNORECASE
)
_NOTES_RE = re.compile(r"^\s*NOTES\s*:", re.IGNORECASE)


def _patch_logu(line: str) -> str:
    """Insert 'WREGAM081' into a LOGU. line whose value field is blank.

    LAS 2.0 field format:   MNEM.UNIT  VALUE : DESCRIPTION

    If a value is already present the line is returned unchanged.
    """
    m = _LOGU_BLANK_RE.match(line)
    if m:
        # Group 2 is the gap between the unit and the colon — the empty value.
        return m.group(1) + "              WREGAM081 " + m.group(3)
    return line


# ---------------------------------------------------------------------------
# Main processing functions
# ---------------------------------------------------------------------------

def process_csv(csv_path: Path, output_dir: Path) -> tuple:
    """Process a SURVEY-IQ multishot CSV export.

    Returns ``(output_path, hole_id)``.
    """
    date_str = date_from_filename(csv_path.name)

    # Read all columns as plain strings so we control every aspect of numeric
    # formatting.  na_values=[] + keep_default_na=False prevents pandas from
    # silently converting the literal string 'NA' (used in Grid Azimuth) to
    # float NaN.
    df = pd.read_csv(
        csv_path,
        dtype=str,
        na_values=[],
        keep_default_na=False,
    )

    # ── Step 1: rename TN Azimuth → Azimuth ──────────────────────────────
    if "TN Azimuth" in df.columns:
        df = df.rename(columns={"TN Azimuth": "Azimuth"})

    # ── Step 2: strip RIG prefix from Rig column (RIG276 → 276) ──────────
    if "Rig" in df.columns:
        df["Rig"] = df["Rig"].str.replace(r"^RIG", "", regex=True)

    # ── Step 3: remove duplicate Measured Depth rows ──────────────────────
    #   Priority: fewest active QC flags → lowest Earth Rate Delta.
    depth_col = "Measured Depth"
    erd_col = "Earth Rate Delta"

    if depth_col in df.columns:
        fc = flag_columns(df)

        # Temporary numeric columns used only for sort/dedup; never written.
        df["__depth_n"] = pd.to_numeric(df[depth_col], errors="coerce")
        df["__erd_n"] = (
            pd.to_numeric(df[erd_col], errors="coerce").fillna(0.0)
            if erd_col in df.columns
            else pd.Series(0.0, index=df.index)
        )
        df["__flags"] = df.apply(lambda r: active_flag_count(r, fc), axis=1)

        # Stable sort ensures the original ordering is the final tiebreaker.
        df = (
            df.sort_values(
                ["__depth_n", "__flags", "__erd_n"],
                ascending=True,
                kind="stable",
            )
            .drop_duplicates(subset=[depth_col], keep="first")
        )

        # Drop helpers and restore depth-ascending order.
        df = (
            df.drop(columns=["__depth_n", "__erd_n", "__flags"])
            .assign(__depth_n=lambda d: pd.to_numeric(d[depth_col], errors="coerce"))
            .sort_values("__depth_n", kind="stable")
            .drop(columns=["__depth_n"])
            .reset_index(drop=True)
        )

    # ── Step 4: format Earth Rate Delta (round to 9 dp, strip zeros) ──────
    if erd_col in df.columns:
        df[erd_col] = df[erd_col].apply(fmt_earth_rate_delta)

    # ── Step 5: strip trailing zeros from angular / toolface columns ───────
    for col in ("Dip", "Azimuth", "Gravity TF", "Vertical TF"):
        if col in df.columns:
            df[col] = df[col].apply(fmt_strip_zeros_3dp)

    # ── Step 6: extract hole ID from Drillhole Name ────────────────────────
    hole_col = "Drillhole Name"
    if hole_col not in df.columns:
        raise ValueError(
            f"Required column {hole_col!r} not found. "
            f"Columns present: {list(df.columns)}"
        )
    hole_id = df[hole_col].iloc[0].strip()

    # ── Step 7: write output with Windows-style (CRLF) line endings ────────
    out_name = f"{hole_id}_{date_str}_REFLEX.csv"
    out_path = output_dir / out_name
    df.to_csv(out_path, index=False, lineterminator="\r\n")

    return out_path, hole_id


def process_las(las_path: Path, hole_id: str, output_dir: Path) -> Path:
    """Process a SURVEY-IQ LAS gamma / geophysical log export."""
    text = las_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    in_param = False   # inside ~PARAMETER INFORMATION
    in_other = False   # inside ~OTHER INFORMATION
    out_lines: list = []

    for line in lines:
        head = line.strip().upper()

        # ── Section detection ──────────────────────────────────────────
        if head.startswith("~"):
            in_param = head.startswith("~P")   # ~P / ~PARAMETER INFORMATION
            in_other = head.startswith("~O")   # ~O / ~OTHER INFORMATION
            # ~A resets both flags; data rows thereafter pass through verbatim.
            out_lines.append(line)
            continue

        # ── ~PARAMETER INFORMATION: populate blank LOGU. value ────────
        if in_param and re.match(r"\s*LOGU\.", line):
            line = _patch_logu(line)

        # ── ~OTHER INFORMATION: drop unwanted metadata lines ──────────
        if in_other:
            if _EXPORT_VERSION_RE.match(line):
                continue   # drop "EXPORTED FROM APP VERSION: ..." line
            if _NOTES_RE.match(line):
                continue   # drop "NOTES: ..." line

        out_lines.append(line)

    # Reconstruct with LF line endings; preserve trailing newline if present.
    out_text = "\n".join(out_lines)
    if text.endswith("\n"):
        out_text += "\n"

    out_name = f"{hole_id}_UP.las"
    out_path = output_dir / out_name
    out_path.write_text(out_text, encoding="utf-8")

    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="reflex_processor",
        description=(
            "Process raw SURVEY-IQ downhole geophysical exports into the "
            "format expected by the Acquire database."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python reflex_processor.py survey_15-03-2024.csv gammalog.las\n"
            "  python reflex_processor.py data/BH001_15-03-2024_raw.csv data/BH001.las\n"
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

    # processed/ lives next to the CSV file.
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

    print("Done.")


if __name__ == "__main__":
    main()
