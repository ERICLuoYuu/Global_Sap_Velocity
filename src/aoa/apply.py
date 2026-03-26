"""Compute DI for prediction grids — the apply step."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.aoa import core
from src.aoa.prepare import load_aoa_reference

logger = logging.getLogger(__name__)

_WINDOWED_RE = re.compile(r"^(.+)_t-(\d+)$")


def load_model_config(path: Path) -> dict:
    """Load model config JSON and extract fields needed by AOA."""
    with open(path) as f:
        raw = json.load(f)
    return {
        "model_type": raw["model_type"],
        "run_id": raw["run_id"],
        "feature_names": raw["feature_names"],
        "is_windowing": raw.get("data_info", {}).get("IS_WINDOWING", False),
        "input_width": raw.get("data_info", {}).get("input_width"),
        "shift": raw.get("data_info", {}).get("shift"),
    }


def parse_windowed_feature_names(
    feature_names: list[str],
) -> tuple[dict[str, list[int]], list[str]]:
    """Parse windowed feature names into dynamic {base: [lags]} and static names.

    Example:
        ["ta_t-0", "ta_t-1", "vpd_t-0", "elevation"]
        -> ({"ta": [0, 1], "vpd": [0]}, ["elevation"])
    """
    dynamic: dict[str, list[int]] = {}
    static: list[str] = []
    for name in feature_names:
        m = _WINDOWED_RE.match(name)
        if m:
            base, lag = m.group(1), int(m.group(2))
            dynamic.setdefault(base, []).append(lag)
        else:
            static.append(name)
    return dynamic, static


def create_windowed_features(
    df: pd.DataFrame,
    dynamic_features: dict[str, list[int]],
    static_features: list[str],
) -> pd.DataFrame:
    """Create lagged columns from raw data grouped by pixel.

    Groups by (latitude, longitude), sorts by timestamp, creates
    {feat}_t-{lag} columns via shift. Rows with incomplete history
    (NaN from shift) are kept — handled by downstream NaN dropping.
    """
    df = df.sort_values(["latitude", "longitude", "timestamp"]).reset_index(drop=True)
    result = df[["latitude", "longitude", "timestamp"]].copy()

    for feat in static_features:
        result[feat] = df[feat].values

    for base, lags in dynamic_features.items():
        if base not in df.columns:
            raise ValueError(f"Base feature '{base}' not found in input columns")
        for lag in sorted(lags):
            col_name = f"{base}_t-{lag}"
            if lag == 0:
                result[col_name] = df[base].values
            else:
                result[col_name] = df.groupby(["latitude", "longitude"])[base].shift(lag).values

    return result


def parse_range(s: str) -> list[int]:
    """Parse '1995-2018' -> [1995,...,2018] or '2015,2016' -> [2015,2016]."""
    if "-" in s and "," not in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in s.split(",")]


def load_input_file(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Load CSV or parquet; deduplicate column names.

    Args:
        usecols: If provided, only load these columns (significant speedup for CSV).
    """
    if path.suffix == ".csv":
        df = pd.read_csv(path, usecols=usecols)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=usecols)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def convert_csv_to_parquet(csv_path: Path, parquet_path: Path | None = None) -> Path:
    """Convert ERA5 CSV to parquet for faster I/O.

    Parquet is typically 5-10x smaller and 10-50x faster to load than CSV.
    """
    if parquet_path is None:
        parquet_path = csv_path.with_suffix(".parquet")
    logger.info(f"Converting {csv_path.name} -> {parquet_path.name}")
    df = pd.read_csv(csv_path)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    df.to_parquet(parquet_path, index=False, compression="snappy")
    logger.info(f"Saved {parquet_path} ({parquet_path.stat().st_size / 1e6:.0f} MB)")
    return parquet_path


def select_and_validate_features(
    df: pd.DataFrame,
    feature_names: list[str],
    reference_feature_names: list[str],
) -> pd.DataFrame:
    """Select features in reference order, validate completeness."""
    if set(feature_names) != set(reference_feature_names):
        raise ValueError(
            f"Feature mismatch: config has {len(feature_names)} features, "
            f"reference has {len(reference_feature_names)}. "
            f"Missing from config: {set(reference_feature_names) - set(feature_names)}, "
            f"Extra in config: {set(feature_names) - set(reference_feature_names)}"
        )
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        col_lower = {c.lower(): c for c in df.columns}
        still_missing = []
        for f in missing:
            if f.lower() in col_lower:
                df = df.rename(columns={col_lower[f.lower()]: f})
            else:
                still_missing.append(f)
        if still_missing:
            raise ValueError(f"Missing features in input: {still_missing}")
    return df[reference_feature_names]


def compute_di_for_dataframe(
    df: pd.DataFrame,
    reference: dict,
    tree: core.cKDTree,
    feature_names: list[str],
    batch_size: int = 500_000,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Compute DI and AOA mask for a DataFrame.

    Args:
        n_jobs: Number of threads for KDTree queries. -1 uses all cores.

    Returns:
        (di_values, aoa_mask, valid_mask) — di/aoa aligned to valid rows only.
        valid_mask is a boolean Series aligned to original df index.
    """
    X_df = select_and_validate_features(df, feature_names, reference["feature_names"])
    valid_mask = ~X_df.isna().any(axis=1)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} rows with NaN features")
    X = X_df[valid_mask].values
    if len(X) == 0:
        return np.array([]), np.array([], dtype=bool), valid_mask

    X_s = core.standardize_features(X, reference["feature_means"], reference["feature_stds"])
    X_sw = core.apply_importance_weights(X_s, reference["feature_weights"])

    di = np.empty(len(X_sw))
    for start in range(0, len(X_sw), batch_size):
        end = min(start + batch_size, len(X_sw))
        di[start:end] = core.compute_prediction_di(X_sw[start:end], tree, reference["d_bar"], workers=n_jobs)

    aoa_mask = di <= reference["threshold"]
    return di, aoa_mask, valid_mask


def discover_era5_files(
    input_dir: Path,
    time_scale: str,
    years: tuple[int, ...],
    months: tuple[int, ...],
) -> list[Path]:
    """Find ERA5 preprocessed files matching year/month filters.

    Pattern: {input_dir}/{YYYY}_{time_scale}/prediction_{YYYY}_{MM}_{time_scale}.*
    Prefers parquet over CSV when both exist (parquet is 10-50x faster to load).
    """
    files = []
    for year in years:
        year_dir = input_dir / f"{year}_{time_scale}"
        if not year_dir.exists():
            logger.warning(f"Year dir not found: {year_dir}")
            continue
        for month in months:
            pattern = f"prediction_{year}_{month:02d}_{time_scale}.*"
            matches = list(year_dir.glob(pattern))
            # Prefer parquet over CSV when both exist
            parquet = [m for m in matches if m.suffix == ".parquet"]
            if parquet:
                files.extend(parquet)
            else:
                files.extend(matches)
    return sorted(files)


def compute_monthly_summary(per_ts_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-timestamp DI to monthly per-pixel summary."""
    summary = (
        per_ts_df.groupby(["latitude", "longitude"])
        .agg(
            median_DI=("DI", "median"),
            mean_DI=("DI", "mean"),
            std_DI=("DI", "std"),
            frac_inside_aoa=("aoa_mask", "mean"),
            n_timestamps=("DI", "count"),
        )
        .reset_index()
    )
    # fillna for std_DI: single-timestamp pixels get NaN from ddof=1
    summary["std_DI"] = summary["std_DI"].fillna(0.0)
    return summary.astype(
        {
            "latitude": np.float32,
            "longitude": np.float32,
            "median_DI": np.float32,
            "mean_DI": np.float32,
            "std_DI": np.float32,
            "frac_inside_aoa": np.float32,
            "n_timestamps": np.int32,
        }
    )


def write_aoa_meta(config, reference: dict, output_dir: Path) -> None:
    """Write metadata JSON."""
    meta = {
        "run_id": config.run_id,
        "model_type": config.model_type,
        "time_scale": config.time_scale,
        "threshold": reference["threshold"],
        "d_bar": reference["d_bar"],
        "d_bar_method": reference["d_bar_method"],
        "n_training_samples": int(reference["reference_cloud_weighted"].shape[0]),
        "n_features": int(reference["reference_cloud_weighted"].shape[1]),
        "feature_names": reference["feature_names"],
        "weighting_method": "mean_abs_shap",
        "iqr_multiplier": config.iqr_multiplier,
        "cv_strategy": "StratifiedGroupKFold",
        "cv_n_splits": int(np.unique(reference["fold_assignments"]).size),
        "random_seed": 42,
        "prediction_years": list(config.years),
        "save_per_timestamp": config.save_per_timestamp,
        "aoa_reference_path": str(config.aoa_reference_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "aoa_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def process_files(config, reference: dict, model_config: dict) -> None:
    """Main processing loop: iterate files, compute DI, write outputs."""

    feature_names = model_config["feature_names"]
    is_windowing = model_config.get("is_windowing", False)
    tree = core.build_kdtree(reference["reference_cloud_weighted"])
    logger.info(f"Using n_jobs={config.n_jobs} for KDTree queries")

    # Pre-parse windowed feature names once
    windowing_info = None
    if is_windowing:
        dynamic, static = parse_windowed_feature_names(feature_names)
        windowing_info = (dynamic, static)
        # For usecols: need base dynamic features + static + meta
        base_features = list(dynamic.keys()) + static
        logger.info(f"Windowed model: {len(dynamic)} dynamic features, {len(static)} static features")
    else:
        base_features = list(feature_names)

    # Only load columns we need (significant CSV speedup)
    needed_cols = ["latitude", "longitude", "timestamp"] + base_features

    ts_dir = config.output_dir / config.time_scale / "per_timestamp"
    monthly_dir = config.output_dir / config.time_scale / "monthly"
    ts_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir.mkdir(parents=True, exist_ok=True)

    files = discover_era5_files(config.input_dir, config.time_scale, config.years, config.months)
    if not files:
        logger.warning("No input files found")
        return

    for file_path in files:
        try:
            t0 = time.perf_counter()
            logger.info(f"Processing {file_path.name}")
            df = load_input_file(file_path, usecols=needed_cols)
            t_load = time.perf_counter()
            logger.info(f"  Load: {t_load - t0:.1f}s, {len(df)} rows")

            # Apply windowing if model uses it
            if windowing_info is not None:
                dynamic, static = windowing_info
                df = create_windowed_features(df, dynamic, static)

            meta_cols = ["latitude", "longitude", "timestamp"]
            meta = df[meta_cols].copy()

            di_values, aoa_mask, valid_mask = compute_di_for_dataframe(
                df, reference, tree, feature_names, config.batch_size, n_jobs=config.n_jobs
            )
            t_di = time.perf_counter()
            logger.info(f"  DI compute: {t_di - t_load:.1f}s, {len(di_values)} valid rows")

            if len(di_values) == 0:
                logger.warning(f"No valid rows in {file_path.name}, skipping")
                continue

            meta_valid = meta[valid_mask].reset_index(drop=True)
            per_ts_df = pd.DataFrame(
                {
                    "latitude": meta_valid["latitude"].values.astype(np.float32),
                    "longitude": meta_valid["longitude"].values.astype(np.float32),
                    "timestamp": meta_valid["timestamp"].values,
                    "DI": di_values.astype(np.float32),
                    "aoa_mask": aoa_mask,
                }
            )

            stem = file_path.stem
            parts = stem.split("_")
            year, month = parts[1], parts[2]

            if config.save_per_timestamp:
                ts_path = ts_dir / f"di_{year}_{month}_{config.time_scale}.parquet"
                per_ts_df.to_parquet(ts_path, index=False, compression="gzip")

            summary = compute_monthly_summary(per_ts_df)
            summary_path = monthly_dir / f"di_monthly_{year}_{month}.parquet"
            summary.to_parquet(summary_path, index=False, compression="gzip")
            t_write = time.perf_counter()
            logger.info(f"  Write: {t_write - t_di:.1f}s | Total: {t_write - t0:.1f}s")

        except (OSError, ValueError, KeyError, pd.errors.EmptyDataError) as e:
            logger.error(f"Failed processing {file_path.name}: {e}")
            continue

    meta_dir = config.output_dir / config.time_scale
    write_aoa_meta(config, reference, meta_dir)


def _sanitize_run_id(run_id: str) -> str:
    """Strip shell-unsafe characters from run_id for use in filenames/SLURM."""
    return re.sub(r"[^a-zA-Z0-9_\-.]", "_", run_id)


def generate_slurm_script(config, args) -> None:
    """Write SLURM array job script."""
    safe_run_id = _sanitize_run_id(config.run_id)
    n_months = len(config.years) * len(config.months)
    years_str = " ".join(str(y) for y in config.years)
    months_str = " ".join(str(m) for m in config.months)
    save_flag = "--save-per-timestamp" if config.save_per_timestamp else ""

    script = f"""#!/bin/bash
#SBATCH --partition=zen4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=36
#SBATCH --mem=200G --time=01:00:00
#SBATCH --array=0-{n_months - 1}
#SBATCH --job-name=aoa_di_{safe_run_id}

YEARS=({years_str})
MONTHS=({months_str})
N_MONTHS={len(config.months)}
YEAR=${{YEARS[$((SLURM_ARRAY_TASK_ID / N_MONTHS))]}}
MONTH=${{MONTHS[$((SLURM_ARRAY_TASK_ID % N_MONTHS))]}}

python -m src.aoa.apply \\
    --aoa-reference "{config.aoa_reference_path}" \\
    --input-dir "{config.input_dir}" \\
    --model-config "{config.model_config_path}" \\
    --output-dir "{config.output_dir}" \\
    --time-scale {config.time_scale} \\
    {save_flag} \\
    --years $YEAR --months $MONTH \\
    --n-jobs 36
"""
    script_path = Path(f"job_aoa_apply_{safe_run_id}.sh")
    script_path.write_text(script)
    logger.info(f"SLURM script written to {script_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute AOA DI on grids")
    parser.add_argument("--aoa-reference", type=Path, required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="ERA5 parent dir (e.g. data/dataset_for_prediction/)",
    )
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--time-scale",
        default="daily",
        choices=["daily", "hourly"],
    )
    parser.add_argument(
        "--save-per-timestamp",
        action="store_true",
        default=None,
    )
    parser.add_argument("--years", required=True, help="e.g. 1995-2018")
    parser.add_argument("--months", default="1-12", help="e.g. 1-12")
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--generate-slurm",
        action="store_true",
        help="Write SLURM script instead of running",
    )
    parser.add_argument(
        "--convert-parquet",
        action="store_true",
        help="Convert discovered CSV files to parquet and exit",
    )
    return parser.parse_args()


def main() -> None:
    import time as _time

    from src.aoa.config import AOAConfig

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    t_start = _time.perf_counter()
    args = parse_args()
    model_config = load_model_config(args.model_config)
    reference = load_aoa_reference(args.aoa_reference)
    config = AOAConfig(
        model_type=model_config["model_type"],
        run_id=model_config["run_id"],
        time_scale=args.time_scale,
        aoa_reference_path=args.aoa_reference,
        input_dir=args.input_dir,
        model_config_path=args.model_config,
        output_dir=args.output_dir,
        save_per_timestamp=args.save_per_timestamp,
        years=tuple(parse_range(args.years)),
        months=tuple(parse_range(args.months)),
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
    )
    if args.convert_parquet:
        files = discover_era5_files(config.input_dir, config.time_scale, config.years, config.months)
        for f in files:
            if f.suffix == ".csv":
                convert_csv_to_parquet(f)
        return
    if args.generate_slurm:
        generate_slurm_script(config, args)
    else:
        process_files(config, reference, model_config)


if __name__ == "__main__":
    main()
