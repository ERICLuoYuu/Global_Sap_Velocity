"""Temporal aggregation of AOA DI results."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _aggregate_group(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate multiple monthly summary DataFrames into one summary per pixel.

    Uses median of monthly medians, mean of monthly means, mean of stds,
    mean of fracs, sum of n_timestamps.
    """
    combined = pd.concat(dfs, ignore_index=True)
    return (
        combined.groupby(["latitude", "longitude"])
        .agg(
            median_DI=("median_DI", "median"),
            mean_DI=("mean_DI", "mean"),
            std_DI=("std_DI", "mean"),
            frac_inside_aoa=("frac_inside_aoa", "mean"),
            n_timestamps=("n_timestamps", "sum"),
        )
        .reset_index()
    )


def aggregate_yearly(monthly_dir: Path, output_dir: Path, years: list[int]) -> list[Path]:
    """Read monthly summaries -> per-year summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for year in years:
        files = sorted(monthly_dir.glob(f"di_monthly_{year}_*.parquet"))
        if not files:
            logger.warning(f"No monthly files for {year}")
            continue
        dfs = [pd.read_parquet(f) for f in files]
        result = _aggregate_group(dfs)
        out = output_dir / f"di_yearly_{year}.parquet"
        result.to_parquet(out, index=False, compression="gzip")
        paths.append(out)
    return paths


def aggregate_climatological(monthly_dir: Path, output_dir: Path) -> list[Path]:
    """Read all monthly summaries -> 12 climatological summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for month in range(1, 13):
        files = sorted(monthly_dir.glob(f"di_monthly_*_{month:02d}.parquet"))
        if not files:
            continue
        dfs = [pd.read_parquet(f) for f in files]
        result = _aggregate_group(dfs)
        out = output_dir / f"di_clim_{month:02d}.parquet"
        result.to_parquet(out, index=False, compression="gzip")
        paths.append(out)
    return paths


def aggregate_overall(monthly_dir: Path, output_path: Path) -> Path:
    """Read ALL monthly summaries -> single overall summary."""
    files = sorted(monthly_dir.glob("di_monthly_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No monthly files in {monthly_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    result = _aggregate_group(dfs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False, compression="gzip")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate AOA DI results")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--time-scale", default="daily")
    args = parser.parse_args()

    base = args.input_dir / args.time_scale
    monthly_dir = base / "monthly"

    years = sorted({int(f.stem.split("_")[2]) for f in monthly_dir.glob("di_monthly_*.parquet")})
    aggregate_yearly(monthly_dir, base / "yearly", years)
    aggregate_climatological(monthly_dir, base / "climatological")
    aggregate_overall(monthly_dir, base / "di_overall.parquet")


if __name__ == "__main__":
    main()
