# Flexible Timestamp Column Resolution

## Problem

The ERA5 processor (`process_era5land_gee_opt_fix.py`) saves DataFrames with
`to_csv(index_label='timestamp')`, creating a duplicate `timestamp` column.
When pandas re-reads this CSV, the duplicate becomes `timestamp.1`.  Both the
prediction and visualization scripts had hardcoded references to `timestamp.1`,
creating a brittle dependency on this naming accident.

## Solution

Both scripts now use a shared `_resolve_timestamp_column()` helper function
that auto-detects the timestamp column from a priority-ordered candidate list:

```
timestamp → timestamp.1 → date → datetime → time → Date → Timestamp → TIMESTAMP
→ (any column containing 'time' or 'date', case-insensitive)
```

### Prediction Script (`predict_sap_velocity_sequential.py`)

- `load_preprocessed_data()` calls `_resolve_timestamp_column(df.columns)`
- Once detected, the column is **renamed** to canonical `"timestamp"`
- All downstream output (Parquet/CSV) now uses clean `"timestamp"` column name

### Visualization Script (`prediction_visualization_hpc.py`)

- `discover_timestamps()` peeks at file headers (CSV first line or Parquet schema)
  and resolves the timestamp column before reading data
- `rasterize_all_timestamps()` does the same auto-resolution
- `--timestamp-col` CLI default changed from `"timestamp.1"` to `"timestamp"` with
  help text explaining the fallback chain
- **Backward compatible**: old files with `timestamp.1` still work because
  `_resolve_timestamp_column(cols, preferred="timestamp")` falls back to
  `"timestamp.1"` when `"timestamp"` is absent

### Test Impact

- `conftest.py` and new test files updated to use canonical `"timestamp"`
- Existing 44 tests (using `timestamp.1` data) still pass because the auto-resolver
  finds `timestamp.1` in their test DataFrames
- **101/101 tests pass**

## Candidate List

| Priority | Column Name | Source |
|----------|-------------|--------|
| 1 | `timestamp` | Canonical (prediction output after fix) |
| 2 | `timestamp.1` | Legacy ERA5 CSV with duplicate columns |
| 3 | `date` | Common alternative |
| 4 | `datetime` | Common alternative |
| 5 | `time` | Less common |
| 6 | `Date` / `Timestamp` / `TIMESTAMP` | Case variants |
| 7 | Any column with 'time'/'date' | Fuzzy fallback |

## Commit

`ff3ddaa` on `refactor/prediction-pipeline-cleanup`
