"""Load & prepare hourly sap-flow data for the afternoon-depression analysis.

Resolves the growing-season-only, daytime-only, raw-SWC hourly dataset, standardises
columns to the analysis schema, and applies the Liu SI Text S1 filter chain.

Climate pairing (``climate_source``):
  * ``"site"`` (default): in-situ measured VPD + Tair — the true microclimate the tree
    transpires into, and cleaner than reanalysis for a ground-based study. (Deliberate
    improvement over Liu's all-ERA5 choice, which suited a global gridded analysis.)
  * ``"era5"``: all-ERA5 (VPD from temperature_2m + dewpoint_2m) for a Liu-exact run.
Soil moisture is ALWAYS the depth-weighted 0–100 cm mean of the raw ERA5-Land layers
(no in-situ SWC exists at sap-flow sites), regardless of ``climate_source``.

EC Text S1 filters implemented here:
  * negative sap flow removed (set to NaN so it cannot enter AM/PM means)
  * local-solar daytime window 06:00–18:00
  * frost-free season: daily-mean Tair > 5 °C (Knauer et al., 2018)
  * low-activity days removed: daily-mean sap flow < ``min_daily_sf``
The ≥-record-length rule (Text S1 #1) is applied downstream as a min-valid-days
threshold in ``decoupling.decouple_all_sites``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.afternoon_depression.diurnal_metrics import (
    build_site_day_table,
    depth_weight_sm,
    solar_decimal_hour,
    vpd_from_era5,
)
from src.sm_vpd_decoupling.conductance import canopy_conductance

Response = Literal["sf", "gc"]

logger = logging.getLogger(__name__)

ClimateSource = Literal["era5", "site"]

# Raw ERA5-Land volumetric soil water columns for the 0–100 cm weighting.
_RAW_SWC_COLS = (
    "volumetric_soil_water_layer_1_raw",
    "volumetric_soil_water_layer_2_raw",
    "volumetric_soil_water_layer_3_raw",
)
_DAYTIME_WINDOW = (6.0, 18.0)


def resolve_hourly_dir(scale: str, data_dir: str | None, processed_root: Path) -> Path:
    """Return the hourly GS+raw data directory, honouring an explicit override."""
    if data_dir is not None:
        d = Path(data_dir)
        if not (d.exists() and any(d.glob("*_hourly.csv"))):
            raise FileNotFoundError(f"--data-dir has no *_hourly.csv files: {d}")
        return d
    candidates = [
        processed_root / scale / "merged" / "daytime_only" / "growing_season" / "hourly",
        processed_root / scale / "merged_daytime_only" / "growing_season" / "hourly",
        processed_root / scale / "merged_daytime_only" / "hourly",
        processed_root / scale / "merged" / "hourly",
    ]
    for c in candidates:
        try:
            if c.exists() and any(c.glob("*_hourly.csv")):
                logger.info("Resolved hourly data dir: %s", c)
                return c
        except (OSError, PermissionError):
            continue
    raise FileNotFoundError("No hourly data dir found. Tried:\n  " + "\n  ".join(str(c) for c in candidates))


def _kelvin_to_celsius(series: pd.Series) -> pd.Series:
    """Convert to °C if the series looks like Kelvin (median > 100); else pass through."""
    s = pd.to_numeric(series, errors="coerce")
    return s - 273.15 if s.median(skipna=True) > 100 else s


def _validate_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: required columns missing: {missing}")
    # Plain notna() (not numeric coercion) so string columns like solar_TIMESTAMP
    # are not falsely flagged as all-NaN.
    empty = [c for c in required if not df[c].notna().any()]
    if empty:
        hint = " (try --climate-source site)" if "dewpoint_2m" in empty else ""
        raise ValueError(f"{source}: required columns present but ALL-NaN: {empty}{hint}")


def standardise_site_frame(df: pd.DataFrame, climate_source: ClimateSource, source: str) -> pd.DataFrame:
    """Map one site's raw hourly CSV to the analysis schema (sf, vpd, tair, sm, rad, lai…)."""
    _validate_columns(df, ["sap_velocity", "solar_TIMESTAMP", *_RAW_SWC_COLS], source)
    out = pd.DataFrame(index=df.index)
    out["site_name"] = df["site_name"] if "site_name" in df.columns else source
    ts = pd.to_datetime(df["solar_TIMESTAMP"])
    out["solar_date"] = ts.dt.date
    out["solar_hour"] = solar_decimal_hour(df["solar_TIMESTAMP"])
    out["sf"] = pd.to_numeric(df["sap_velocity"], errors="coerce")
    out["sm"] = depth_weight_sm(*(pd.to_numeric(df[c], errors="coerce") for c in _RAW_SWC_COLS))
    if climate_source == "era5":
        _validate_columns(df, ["temperature_2m", "dewpoint_2m"], source)
        out["tair"] = _kelvin_to_celsius(df["temperature_2m"])
        out["vpd"] = vpd_from_era5(out["tair"], _kelvin_to_celsius(df["dewpoint_2m"]))
        rad = pd.to_numeric(df.get("surface_solar_radiation_downwards_hourly"), errors="coerce")
        out["rad"] = rad / 3600.0  # accumulated J/m² → W/m²
    else:
        _validate_columns(df, ["vpd", "ta"], source)
        out["tair"] = pd.to_numeric(df["ta"], errors="coerce")
        out["vpd"] = pd.to_numeric(df["vpd"], errors="coerce")
        out["rad"] = pd.to_numeric(df.get("sw_in"), errors="coerce")
    out["lai"] = pd.to_numeric(df.get("LAI"), errors="coerce")
    for col, std in [
        ("pft", "pft"),
        ("biome", "biome"),
        ("prcip/PET", "aridity"),
        ("elevation", "elevation"),  # altitude h for Flo 2021 Gc (exp(0.00012·h))
        ("latitude_x", "lat"),
        ("longitude_x", "lon"),
    ]:
        if col in df.columns:
            out[std] = df[col]
    return out


def prepare_hourly(hourly_dir: Path, climate_source: ClimateSource = "era5", response: Response = "sf") -> pd.DataFrame:
    """Load + standardise every site CSV, then apply hourly-level EC filters.

    With ``response="gc"`` a per-hour whole-tree canopy conductance column ``gc``
    (Flo et al. 2021, Eqn 2) is added *after* negative sap flow is masked to NaN, so
    reverse/invalid flow yields NaN Gc. ``canopy_conductance`` already returns NaN where
    VPD ≤ 0, so low-VPD dawn hours drop out of the AM mean rather than blowing up.
    """
    files = [f for f in sorted(hourly_dir.glob("*_hourly.csv")) if f.name != "all_biomes_merged_hourly.csv"]
    if not files:
        raise FileNotFoundError(f"No per-site *_hourly.csv in {hourly_dir}")
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            raw = pd.read_csv(f)
            if raw.empty:
                continue
            frames.append(standardise_site_frame(raw, climate_source, f.name))
        except (ValueError, KeyError) as e:
            logger.warning("Skipping %s: %s", f.name, e)
    if not frames:
        raise ValueError("No site files could be standardised (check columns / --climate-source).")
    df = pd.concat(frames, ignore_index=True)
    # Hourly EC filters: negative sap → NaN; restrict to solar 06–18.
    df.loc[df["sf"] < 0, "sf"] = np.nan
    df = df[(df["solar_hour"] >= _DAYTIME_WINDOW[0]) & (df["solar_hour"] < _DAYTIME_WINDOW[1])].copy()
    if response == "gc":
        # Altitude h for Flo Eqn 2; absent/NaN → 0 m (sea level, exp(0)=1), as in
        # src/sm_vpd_decoupling/loader.py. Negative sf is already NaN → NaN Gc.
        elev = df["elevation"] if "elevation" in df.columns else pd.Series(0.0, index=df.index)
        elev = pd.to_numeric(elev, errors="coerce").fillna(0.0)
        df["gc"] = canopy_conductance(df["sf"], df["tair"], df["vpd"], elev)
    logger.info("Prepared hourly frame: %d rows, %d sites", len(df), df["site_name"].nunique())
    return df


def apply_day_filters(table: pd.DataFrame, tair_min: float = 5.0, min_daily_sf: float = 0.0) -> pd.DataFrame:
    """Frost-free (daily Tair > tair_min) + low-activity-day removal on the site-day table."""
    before = len(table)
    out = table[(table["tair"] > tair_min) & (table["sf_day_mean"] >= min_daily_sf)].copy()
    logger.info("Day filters (Tair>%.1f°C, dailySF≥%.3g): %d → %d site-days", tair_min, min_daily_sf, before, len(out))
    return out


def apply_morning_flow_filter(table: pd.DataFrame, min_am_pm_ratio: float = 0.10) -> pd.DataFrame:
    """Drop site-days with negligible morning flow relative to the afternoon.

    ΔSF = (SF_AM − SF_PM)/SF_AM is unstable when SF_AM → 0 (e.g. boreal dawn),
    producing |ΔSF| up to millions of %. Requiring ``SF_AM ≥ ratio·SF_PM``
    analytically bounds ΔSF ≥ (1 − 1/ratio)·100 % (at ratio=0.10 → ≥ −900 %) and
    removes the divide-by-near-zero tail. This is the sap-flow analog of Liu SI
    Text S1 step 3 ("remove low-flux records"). ``ratio ≤ 0`` disables the filter.
    """
    if min_am_pm_ratio <= 0:
        return table
    before = len(table)
    out = table[table["sf_am"] >= min_am_pm_ratio * table["sf_pm"]].copy()
    logger.info("Morning-flow filter (SF_AM≥%.2g·SF_PM): %d → %d site-days", min_am_pm_ratio, before, len(out))
    return out


def load_site_day_table(
    hourly_dir: Path,
    climate_source: ClimateSource = "era5",
    response: Response = "sf",
    tair_min: float = 5.0,
    min_daily_sf: float = 0.0,
    min_window_hours: int = 2,
    min_am_pm_ratio: float = 0.10,
) -> pd.DataFrame:
    """Full pipeline: hourly CSVs → filtered per-site-day ΔSF/C_SF table.

    ``response`` selects the variable that flows through the abstract response slot:
    ``"sf"`` (sap velocity, default — unchanged) or ``"gc"`` (Flo 2021 canopy
    conductance). For ``"gc"`` the AM/PM means, ``delta_sf`` (= ΔGc) and ``centroid``
    are all computed on Gc; the column names stay ``sf_*``/``delta_sf`` so every
    downstream consumer (decoupling, RF, plotting) works unchanged.
    """
    hourly = prepare_hourly(hourly_dir, climate_source, response=response)
    table = build_site_day_table(
        hourly,
        sf_col="gc" if response == "gc" else "sf",
        hour_col="solar_hour",
        date_col="solar_date",
        site_col="site_name",
        driver_cols=("vpd", "tair", "sm", "rad", "lai"),
        carry_cols=("pft", "biome", "aridity", "lat", "lon"),
        min_window_hours=min_window_hours,
    )
    table = apply_day_filters(table, tair_min=tair_min, min_daily_sf=min_daily_sf)
    return apply_morning_flow_filter(table, min_am_pm_ratio=min_am_pm_ratio)
