# reflex_processor.py

Command-line tool that transforms raw **SURVEY-IQ** downhole geophysical exports
into the format accepted by the **Acquire** database.

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.8 or later |
| [pandas](https://pandas.pydata.org/) | any recent release |

```bash
pip install pandas
```

---

## Usage

```bash
python reflex_processor.py <survey.csv> <log.las>
```

| Argument | Description |
|----------|-------------|
| `survey.csv` | SURVEY-IQ multishot survey export (CSV) — filename **must** contain a `DD-MM-YYYY` date segment |
| `log.las` | SURVEY-IQ gamma / geophysical log export (LAS 2.0) |

Processed files are written to a `processed/` subdirectory created alongside the
CSV input.  The directory is created automatically if it does not exist.

### Example

```
$ python reflex_processor.py BH001_15-03-2024_multishot.csv BH001_gamma.las

Output directory: /data/BH001/processed
  CSV → /data/BH001/processed/BH001_15032024_REFLEX.csv
  LAS → /data/BH001/processed/BH001_UP.las
Done.
```

---

## Output naming

| File | Pattern | Example |
|------|---------|---------|
| Processed survey | `{HOLEID}_{DDMMYYYY}_REFLEX.csv` | `BH001_15032024_REFLEX.csv` |
| Processed log | `{HOLEID}_UP.las` | `BH001_UP.las` |

- **`HOLEID`** — first value found in the `Drillhole Name` column of the CSV.
- **`DDMMYYYY`** — extracted from the `DD-MM-YYYY` segment in the raw CSV filename
  (e.g. `BH001_15-03-2024_multishot.csv` → `15032024`).

---

## Input file formats

### CSV — SURVEY-IQ multishot export

A comma-separated file with a single header row and one row per measured-depth
station.  Columns consumed or modified by the processor:

| Column | Notes |
|--------|-------|
| `TN Azimuth` | Renamed to `Azimuth` in output |
| `Rig` | Contains values like `RIG276` |
| `Earth Rate Delta` | Floating-point with 16–18 significant decimal places |
| `Dip` | Three decimal places, zero-padded (e.g. `-89.920`) |
| `Gravity TF` | Three decimal places, zero-padded |
| `Vertical TF` | Three decimal places, zero-padded |
| `Measured Depth` | Primary key; used for duplicate detection |
| `Drillhole Name` | Source of the hole ID used in output filenames |
| `Grid Azimuth` | May contain the literal string `NA` as a valid sentinel |

Any QC flag columns (names containing *flag*, *qc*, *pass*, *fail*, or *error*)
are used during duplicate-depth resolution.

The raw CSV filename **must** contain a date segment in `DD-MM-YYYY` format
(e.g. `BH001_15-03-2024_multishot.csv`).

### LAS — SURVEY-IQ gamma / geophysical log

A standard **LAS 2.0** file.  Sections consumed or modified:

| Section | What is modified |
|---------|-----------------|
| `~PARAMETER INFORMATION` | `LOGU.` value populated |
| `~OTHER INFORMATION` | Two specific lines removed |
| `~A` (data) | **Untouched** |

---

## Processing steps

### CSV processing

#### 1 — Rename `TN Azimuth` → `Azimuth`

SURVEY-IQ exports the true-north azimuth under the header `TN Azimuth`.
Acquire's import schema expects it as `Azimuth`.

---

#### 2 — Strip `RIG` prefix from `Rig` column

Raw value: `RIG276`  →  output: `276`

SURVEY-IQ prepends the string `RIG` to every rig identifier.  Acquire stores
only the numeric portion, so the prefix is removed with a regex substitution on
the `Rig` column.

---

#### 3 — Round `Earth Rate Delta` to 9 dp, strip trailing zeros

Raw value: `0.000123456789012345678`  →  output: `0.000123457`

SURVEY-IQ accumulates floating-point precision artefacts during calculation,
producing values with 16–18 decimal places where only the first 9 are
meaningful.  Rounding via Python's `Decimal` type (using `ROUND_HALF_UP`)
eliminates the noise.  Trailing zeros are stripped so that, for example,
`0.123000000` becomes `0.123`.

---

#### 4 — Strip trailing zeros from `Dip`, `Azimuth`, `Gravity TF`, `Vertical TF`

Raw value: `-89.920`  →  output: `-89.92`  
Raw value: `318.500`  →  output: `318.5`  
Raw value: `45.000`   →  output: `45`

SURVEY-IQ zero-pads all of these columns to exactly 3 decimal places.
Acquire's validator rejects the padded form, so trailing zeros (and any
resulting bare decimal point) are removed.  The rounding step uses `Decimal`
arithmetic to avoid introducing new float-precision artefacts.

---

#### 5 — Preserve `NA` strings literally

The `Grid Azimuth` column (and potentially others) legitimately contains the
string `NA` as a sentinel meaning "not available / not calculated".

By default, pandas converts many tokens — including `NA`, `N/A`, `nan` — to
`float NaN`, which would later be written back to the CSV as an empty cell,
silently destroying the sentinel.  The processor disables all default NA
conversions (`na_values=[], keep_default_na=False`) so that `NA` is read and
written back as the literal string `NA`.

---

#### 6 — Remove duplicate `Measured Depth` rows

SURVEY-IQ can write two survey passes at the same depth (e.g. a static and a
dynamic reading).  Acquire requires a unique depth key.

Where two rows share the same `Measured Depth` value:

1. **Keep** the row with the **fewest active QC flags** (columns whose names
   contain *flag*, *qc*, *pass*, *fail*, or *error* and whose values are not
   `PASS`, `OK`, `0`, or blank).
2. If the flag count is equal, keep the row with the **lowest `Earth Rate Delta`**
   (numerically smallest value).
3. If both are equal, the row that appeared first in the source file is kept
   (stable sort preserves original order as the final tiebreaker).

---

#### 7 — CRLF line endings

Acquire's CSV importer expects Windows-style line endings (`\r\n`).  The output
file is written with `lineterminator="\r\n"` to ensure compatibility regardless
of the host OS.

---

### LAS processing

#### 1 — Populate blank `LOGU.` field with `WREGAM081`

In `~PARAMETER INFORMATION`, SURVEY-IQ exports a blank value for the `LOGU.`
(log-unit) mnemonic:

```
 LOGU.                       : LOG UNITS
```

Acquire requires this field to be populated with the unit identifier
`WREGAM081`.  The processor inserts the value while leaving the mnemonic,
unit, and description intact:

```
 LOGU.              WREGAM081 : LOG UNITS
```

Detection is conservative: the value is only inserted when the field is blank
(i.e. only whitespace between the unit and the `:` separator); if a value is
already present it is left unchanged.

---

#### 2 — Remove `EXPORTED FROM APP VERSION:` line from `~OTHER INFORMATION`

SURVEY-IQ appends its own application version to the `~OTHER INFORMATION`
section:

```
 EXPORTED FROM APP VERSION: 3.14.2
```

Acquire's LAS parser does not expect this line and rejects the file when it is
present.  The line is dropped entirely.

---

#### 3 — Remove `NOTES:` line from `~OTHER INFORMATION`

Similarly, SURVEY-IQ may append a `NOTES:` line:

```
 NOTES: Exported by operator on rig
```

This is also internal metadata that Acquire does not accept.  The line is
dropped entirely.

---

#### 4 — Data rows left untouched

All lines in the `~A` (ASCII data) section are passed through verbatim.
No reformatting, rounding, or line-ending changes are applied to the raw sensor
data to ensure zero precision loss.

---

## Error handling

The script exits with a non-zero status and a descriptive message if:

- Either input file does not exist.
- The CSV filename does not contain a `DD-MM-YYYY` date segment.
- The `Drillhole Name` column is absent from the CSV.
- Any unexpected parsing error occurs.
