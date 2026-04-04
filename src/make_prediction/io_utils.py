"""I/O utilities for the prediction pipeline.

Provides format-agnostic save/read helpers that auto-detect Parquet vs CSV
from the file extension. Parquet is the primary format; CSV is kept as a
fallback for debugging and backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_PARQUET_EXTS = {".parquet", ".pq"}
_CSV_EXTS = {".csv", ".tsv", ".txt"}


def _resolve_fmt(path: Path, fmt: str | None) -> str:
    """Return 'parquet' or 'csv' based on *fmt* or the file extension."""
    if fmt is not None:
        return fmt
    ext = path.suffix.lower()
    if ext in _PARQUET_EXTS:
        return "parquet"
    if ext in _CSV_EXTS:
        return "csv"
    return "parquet"  # default


def _check_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def _check_extension(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _PARQUET_EXTS:
        return "parquet"
    if ext in _CSV_EXTS:
        return "csv"
    raise ValueError(f"Unsupported file extension '{ext}' for {path}")


# ---------------------------------------------------------------------------
# save_df
# ---------------------------------------------------------------------------


def save_df(
    df: pd.DataFrame,
    path: str | Path,
    *,
    fmt: str | None = None,
    compression: str = "gzip",
    index: bool = False,
) -> Path:
    """Save a DataFrame as Parquet or CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    path : str | Path
        Output file path. Extension determines format when *fmt* is None.
    fmt : str, optional
        Explicit format: ``"parquet"`` or ``"csv"``. When ``None``, inferred
        from the file extension (default behaviour).
    compression : str
        Parquet compression codec (ignored for CSV). Default ``"gzip"``.
    index : bool
        Whether to write the DataFrame index. Default ``False``.

    Returns
    -------
    Path
        The path the file was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_fmt(path, fmt)

    if resolved == "parquet":
        df.to_parquet(path, compression=compression, engine="pyarrow", index=index)
    else:
        df.to_csv(path, index=index)

    return path


# ---------------------------------------------------------------------------
# read_df  (pandas)
# ---------------------------------------------------------------------------


def read_df(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a DataFrame from Parquet or CSV, auto-detected from extension.

    Parameters
    ----------
    path : str | Path
        Input file path.
    **kwargs
        Forwarded to ``pd.read_parquet`` or ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    _check_exists(path)
    resolved = _check_extension(path)

    if resolved == "parquet":
        return pd.read_parquet(path, engine="pyarrow", **kwargs)
    return pd.read_csv(path, low_memory=False, **kwargs)


# ---------------------------------------------------------------------------
# read_df_dask
# ---------------------------------------------------------------------------


def read_df_dask(path: str | Path, **kwargs: Any):
    """Read a Dask DataFrame from Parquet or CSV, auto-detected.

    Parameters
    ----------
    path : str | Path
        Input file path.
    **kwargs
        Forwarded to ``dd.read_parquet`` or ``dd.read_csv``.

    Returns
    -------
    dask.dataframe.DataFrame
    """
    import dask.dataframe as dd

    path = Path(path)
    _check_exists(path)
    resolved = _check_extension(path)

    if resolved == "parquet":
        return dd.read_parquet(path, engine="pyarrow", **kwargs)
    # CSV-specific defaults
    kwargs.setdefault("blocksize", "128MB")
    kwargs.setdefault("assume_missing", True)
    return dd.read_csv(str(path), **kwargs)


# ---------------------------------------------------------------------------
# discover_input_files
# ---------------------------------------------------------------------------


def discover_input_files(directory: str | Path) -> list[Path]:
    """Find prediction data files in *directory*, preferring Parquet.

    When both ``foo.parquet`` and ``foo.csv`` exist, only ``foo.parquet`` is
    returned.  Files are sorted by name for deterministic ordering.

    Parameters
    ----------
    directory : str | Path
        Directory to scan.

    Returns
    -------
    list[Path]
        Sorted list of discovered file paths.
    """
    directory = Path(directory)
    parquet_files = sorted(directory.glob("*.parquet"))
    csv_files = sorted(directory.glob("*.csv"))

    # Deduplicate: if a stem exists in both formats, keep only Parquet
    parquet_stems = {f.stem for f in parquet_files}
    csv_unique = [f for f in csv_files if f.stem not in parquet_stems]

    return sorted(parquet_files + csv_unique, key=lambda p: p.name)
