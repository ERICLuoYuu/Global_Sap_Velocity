"""Serve the training sites: rebuild each site's features from ERA5-Land and compare with the training tree.

For every site of a servable daily tree, two frames go through :mod:`src.serving.engineer`:

* ``tree``    the trainer's input, the merged daily CSV as it is;
* ``served``  the same days rebuilt the way a prediction cell is: every ERA5-Land and sensor column from
              the site's hourly ERA5-Land extract (:mod:`src.serving.era5_daily`), and ``ext_rad``,
              ``day_length`` and ``solar_TIMESTAMP`` computed. LAI, the static layers and the PFT are copied
              from the tree, so the comparison isolates the dynamic drivers and the time index.

Writes to ``--out_dir``: ``servability.csv``; ``base_column_parity.csv`` and ``feature_parity.csv`` (per site,
plus ``site == "ALL"`` pooled); with ``--run_id`` also ``site_predictions.parquet`` and ``skill.json``:
the model on the tree's inputs and on the served ones, against the observed sap velocity. The tree-input
prediction is IN-SAMPLE for a FINAL model; the difference between the two is the serving effect.

Usage (repo root): python -m src.serving.site_parity --tree_dir ... --era5_csv ... --cube ... --out_dir ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_contract.definitions import SERVABLE_CONTRACT, assert_dataset_contract
from src.feature_contract.rebuild_daily import site_coordinates
from src.feature_contract.swvl_climatology import DEFAULT_MAX_RADIUS_CELLS, load_climatology, lookup_cell
from src.forward_selection.feature_registry import CANDIDATE_FEATURES, MANDATORY_FEATURES
from src.hyperparameter_optimization.shap_constants import PFT_COLUMNS
from src.serving.base_columns import (
    DAY_COLUMN,
    ERA5_COLUMNS,
    LAI_COLUMNS,
    PFT_COLUMN,
    SENSOR_SUBSTITUTED,
    SOLAR_COLUMNS,
    STATIC_COLUMNS,
    TIME_COLUMN,
)
from src.serving.dependencies import is_hourly_feature, servability_report, synthetic_cell
from src.serving.engineer import engineer_cell, model_matrix
from src.serving.era5_daily import HOURLY_BANDS, daily_base_columns, day_length_daily, ext_rad_daily, solar_timestamp
from src.serving.model_bundle import load_bundle

LOGGER = logging.getLogger("site_parity")

TARGET = "sap_velocity"
ERA5_SITE_COLUMN = "site_name"
ERA5_TIME_COLUMN = "datetime"
DAILY_SUFFIX = "_daily.csv"
AGGREGATE_MARKER = "all_biomes_merged"
CHUNK_ROWS = 1_000_000
RELATIVE_EQUAL = 1e-6
SOLAR_OFFSET = "solar_offset_minutes"
POOLED = "ALL"


def default_candidates(contract: str) -> list:
    """Every daily FFS candidate that a synthetic cell produces under ``contract``."""
    names = list(MANDATORY_FEATURES) + [name for group in CANDIDATE_FEATURES.values() for name in group]
    produced = engineer_cell(synthetic_cell(), contract).columns
    return [
        name
        for name in dict.fromkeys(names)
        if not is_hourly_feature(name) and (name in produced or name in PFT_COLUMNS)
    ]


def read_era5(path, sites) -> dict:
    """Site -> its hourly ERA5-Land frame, index UTC hours, first of any duplicated hour kept (as the merge)."""
    usecols = [ERA5_SITE_COLUMN, ERA5_TIME_COLUMN, *HOURLY_BANDS]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_ROWS):
        chunk = chunk[chunk[ERA5_SITE_COLUMN].isin(sites)]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        raise LookupError(f"{path} holds none of the {len(sites)} requested sites")
    frame = pd.concat(parts, ignore_index=True)
    frame[ERA5_TIME_COLUMN] = pd.to_datetime(frame[ERA5_TIME_COLUMN], utc=True)
    by_site = {}
    for site, group in frame.groupby(ERA5_SITE_COLUMN):
        hourly = group.drop(columns=ERA5_SITE_COLUMN).set_index(ERA5_TIME_COLUMN).sort_index()
        duplicated = hourly.index.duplicated(keep="first")
        if duplicated.any():
            LOGGER.info("%s: %d duplicated ERA5 hours, first kept", site, int(duplicated.sum()))
        by_site[site] = hourly[~duplicated]
    return by_site


def _utc_days(values) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return index.normalize()


def _first_valid(series: pd.Series):
    valid = series.dropna()
    return valid.iloc[0] if len(valid) else np.nan


def _solar_offset_minutes(frame: pd.DataFrame) -> pd.Series:
    solar = pd.to_datetime(frame[TIME_COLUMN], utc=True, format="mixed")
    return (solar - pd.to_datetime(frame[DAY_COLUMN], utc=True)).dt.total_seconds() / 60.0


def served_frame(tree: pd.DataFrame, hourly: pd.DataFrame, cells: dict, latitude: float, longitude: float) -> pd.DataFrame:
    """The tree's days rebuilt as a prediction cell builds them; see the module docstring."""
    tree_days = _utc_days(tree[DAY_COLUMN])
    if tree_days.has_duplicates:
        raise ValueError("daily tree holds more than one row per UTC day")
    daily = daily_base_columns(hourly, cells)
    daily = daily.loc[(daily.index >= tree_days.min()) & (daily.index <= tree_days.max())]
    if daily.empty:
        raise ValueError("no ERA5-Land day inside the tree's record")
    days = daily.index
    by_day = tree.set_index(tree_days)
    elevation = float(_first_valid(tree["elevation"])) if "elevation" in tree.columns else 0.0

    served = daily.copy()
    served["ext_rad"] = ext_rad_daily(days, latitude, longitude).to_numpy()
    served["day_length"] = day_length_daily(days, latitude, longitude, elevation).to_numpy()
    for column in LAI_COLUMNS:
        served[column] = by_day[column].reindex(days).to_numpy() if column in by_day.columns else np.nan
    for column in STATIC_COLUMNS:
        served[column] = _first_valid(tree[column]) if column in tree.columns else np.nan
    served[PFT_COLUMN] = tree[PFT_COLUMN].dropna().mode().iloc[0]
    served["latitude"] = latitude
    served["longitude"] = longitude
    served[TARGET] = by_day[TARGET].reindex(days).to_numpy()
    served[DAY_COLUMN] = days
    served[TIME_COLUMN] = solar_timestamp(days, longitude)
    return served.reset_index(drop=True)


def agreement(tree_values: np.ndarray, served_values: np.ndarray) -> dict:
    mask = np.isfinite(tree_values) & np.isfinite(served_values)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    x, y = tree_values[mask], served_values[mask]
    d = y - x
    corr = float(np.corrcoef(x, y)[0, 1]) if n > 2 and x.std() > 0 and y.std() > 0 else np.nan
    equal = np.abs(d) <= RELATIVE_EQUAL * np.maximum(1.0, np.abs(x))
    return {
        "n": n,
        "tree_mean": float(x.mean()),
        "served_mean": float(y.mean()),
        "bias": float(d.mean()),
        "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d**2).mean())),
        "corr": corr,
        "max_abs": float(np.abs(d).max()),
        "frac_equal": float(equal.mean()),
    }


def _by_day(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index(_utc_days(frame[DAY_COLUMN]))


def paired_columns(tree_frame: pd.DataFrame, served: pd.DataFrame, columns) -> dict:
    """Column -> (tree values, served values) on the days both frames hold."""
    left, right = _by_day(tree_frame), _by_day(served)
    common = left.index.intersection(right.index)
    return {
        column: (left.loc[common, column].to_numpy(dtype=float), right.loc[common, column].to_numpy(dtype=float))
        for column in columns
        if column in left.columns and column in right.columns
    }


def predictions(bundle, engineered: pd.DataFrame, site: str) -> pd.DataFrame:
    matrix = model_matrix(engineered, bundle.feature_names, site)
    rows = engineered.loc[matrix.index]
    return pd.DataFrame(
        {
            DAY_COLUMN: _utc_days(rows[DAY_COLUMN]),
            "observed": rows[TARGET].to_numpy(dtype=float),
            "predicted": bundle.predict(matrix),
        }
    )


def process_site(path: Path, era5: dict, clim, features, contract: str, bundle) -> dict:
    tree = pd.read_csv(path, parse_dates=[DAY_COLUMN])
    site = str(tree["site_name"].iloc[0])
    latitude, longitude = site_coordinates(tree, site)
    if site not in era5:
        raise LookupError(f"{site}: no hourly ERA5-Land rows in the extract")
    cells = {layer: lookup_cell(clim, latitude, longitude, layer, DEFAULT_MAX_RADIUS_CELLS) for layer in (1, 2, 3, 4)}
    served = served_frame(tree, era5[site], cells, latitude, longitude)
    tree_input = tree.assign(latitude=latitude, longitude=longitude)

    base = paired_columns(
        tree_input.assign(**{SOLAR_OFFSET: _solar_offset_minutes(tree_input)}),
        served.assign(**{SOLAR_OFFSET: _solar_offset_minutes(served)}),
        [*ERA5_COLUMNS, *SOLAR_COLUMNS, SOLAR_OFFSET],
    )
    tree_engineered = engineer_cell(tree_input, contract)
    served_engineered = engineer_cell(served, contract)
    feature_columns = [name for name in features if name not in PFT_COLUMNS]
    feature = paired_columns(tree_engineered, served_engineered, feature_columns)

    result = {"site": site, "base": base, "feature": feature, "predictions": None}
    if bundle is not None:
        tree_pred = predictions(bundle, tree_engineered, site)
        served_pred = predictions(bundle, served_engineered, site).drop(columns="observed")
        joined = tree_pred.merge(served_pred, on=DAY_COLUMN, how="outer", suffixes=("_tree", "_served"))
        result["predictions"] = joined.assign(site=site)
    return result


def parity_table(per_site: list, key: str, scales: dict) -> pd.DataFrame:
    rows, pooled = [], {}
    for result in per_site:
        for column, (a, b) in result[key].items():
            rows.append({"site": result["site"], "column": column, **agreement(a, b)})
            pooled.setdefault(column, ([], []))
            pooled[column][0].append(a)
            pooled[column][1].append(b)
    for column, (a_parts, b_parts) in pooled.items():
        rows.append({"site": POOLED, "column": column, **agreement(np.concatenate(a_parts), np.concatenate(b_parts))})
    table = pd.DataFrame(rows)
    table["sensor_substituted"] = table["column"].isin(SENSOR_SUBSTITUTED)
    if scales:
        table["bias_in_scaler_sd"] = table["bias"] / table["column"].map(scales)
    return table


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = ((observed - predicted) ** 2).sum()
    total = ((observed - observed.mean()) ** 2).sum()
    return float(1.0 - residual / total) if total > 0 else np.nan


def skill_summary(frame: pd.DataFrame) -> dict:
    """Pooled R2 and per-site NSE of both predictions, on the days where observed and both predictions exist."""
    usable = frame.dropna(subset=["observed", "predicted_tree", "predicted_served"])
    summary = {"n_days": int(len(usable)), "n_sites": int(usable["site"].nunique())}
    if usable.empty:
        return summary
    for which in ("tree", "served"):
        column = f"predicted_{which}"
        per_site = usable.groupby("site").apply(
            lambda g, c=column: _r2(g["observed"].to_numpy(), g[c].to_numpy()) if len(g) > 2 else np.nan
        )
        summary[which] = {
            "pooled_r2": _r2(usable["observed"].to_numpy(), usable[column].to_numpy()),
            "median_site_nse": float(np.nanmedian(per_site)),
            "mean_prediction": float(usable[column].mean()),
        }
    summary["mean_observed"] = float(usable["observed"].mean())
    difference = usable["predicted_served"] - usable["predicted_tree"]
    summary["served_minus_tree"] = {"mean": float(difference.mean()), "mae": float(difference.abs().mean())}
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--tree_dir", required=True, help="merged config dir (the parent of daily/)")
    parser.add_argument("--era5_csv", required=True)
    parser.add_argument("--cube", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--models_root", default="models")
    parser.add_argument("--sites", default=None, help="comma-separated site names (default: every site)")
    parser.add_argument("--contract", default=SERVABLE_CONTRACT)
    return parser.parse_args(argv)


def site_files(tree_dir: Path, sites) -> list:
    files = sorted(p for p in (tree_dir / "daily").glob(f"*{DAILY_SUFFIX}") if AGGREGATE_MARKER not in p.name)
    if sites:
        files = [p for p in files if p.name[: -len(DAILY_SUFFIX)] in sites]
    if not files:
        raise FileNotFoundError(f"no site files under {tree_dir / 'daily'} (sites={sites})")
    return files


def write_outputs(out_dir: Path, per_site: list, bundle, failures: dict) -> None:
    scales = dict(zip(bundle.numeric_names, bundle.scaler.scale_)) if bundle is not None else {}
    parity_table(per_site, "base", {}).to_csv(out_dir / "base_column_parity.csv", index=False)
    parity_table(per_site, "feature", scales).to_csv(out_dir / "feature_parity.csv", index=False)
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    if bundle is None:
        return
    frame = pd.concat([r["predictions"] for r in per_site if r["predictions"] is not None], ignore_index=True)
    frame.to_parquet(out_dir / "site_predictions.parquet", index=False)
    summary = skill_summary(frame)
    (out_dir / "skill.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tree_dir, out_dir = Path(args.tree_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_dataset_contract(tree_dir, args.contract)

    bundle = load_bundle(args.models_root, args.run_id) if args.run_id else None
    features = list(bundle.feature_names) if bundle is not None else default_candidates(args.contract)
    report = servability_report(features, args.contract)
    report.to_csv(out_dir / "servability.csv", index=False)
    LOGGER.info("servability of %d features written; never servable: %s", len(features),
                report.loc[report["never_servable"], "feature"].tolist())

    files = site_files(tree_dir, set(args.sites.split(",")) if args.sites else None)
    era5 = read_era5(args.era5_csv, {p.name[: -len(DAILY_SUFFIX)] for p in files})
    clim = load_climatology(args.cube)
    per_site, failures = [], {}
    for path in files:
        try:
            per_site.append(process_site(path, era5, clim, features, args.contract, bundle))
        except Exception as exc:  # noqa: BLE001 - recorded per site and reported
            LOGGER.exception("%s failed", path.name)
            failures[path.name] = repr(exc)
    LOGGER.info("%d sites compared, %d failed", len(per_site), len(failures))
    if not per_site:
        return 1
    write_outputs(out_dir, per_site, bundle, failures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
