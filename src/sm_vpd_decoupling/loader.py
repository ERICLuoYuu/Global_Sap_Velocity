# src/sm_vpd_decoupling/loader.py
"""Load and prepare the daily merged dataset for SM-VPD decoupling.

Reads the daytime, treatment-filtered, all-season daily CSVs produced by the
data-production merge run, resolves a PPFD source, computes root-zone SM and Gc,
applies Liu's day filter, and per-site-normalizes both responses (E and Gc).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.conductance import SW_TO_PPFD, canopy_conductance

logger = logging.getLogger(__name__)

ClimateSource = Literal["site", "era5"]

ROOT_ZONE_WEIGHTS = (0.07, 0.21, 0.72)  # ERA5-Land layer thickness fractions to 100 cm
VPD_MIN_KPA = 0.5
PPFD_MIN = 500.0
NORM_QUANTILE = 0.90


def root_zone_sm(swvl1: pd.Series, swvl2: pd.Series, swvl3: pd.Series) -> pd.Series:
    """Fixed 0-100 cm thickness-weighted root-zone SM (Liu's 0-1 m analog)."""
    w1, w2, w3 = ROOT_ZONE_WEIGHTS
    return w1 * swvl1 + w2 * swvl2 + w3 * swvl3


def resolve_ppfd(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """PPFD source resolution: ppfd_in -> sw_in*2.04 -> ERA5 ssrd*2.04.

    Returns (ppfd_series, source_label). Picks the first source that has any
    non-null value; conversions assume daily-mean W m-2 -> umol m-2 s-1 PAR.
    """
    if "ppfd_in" in df and pd.to_numeric(df["ppfd_in"], errors="coerce").notna().any():
        return pd.to_numeric(df["ppfd_in"], errors="coerce"), "ppfd_in"
    if "sw_in" in df and pd.to_numeric(df["sw_in"], errors="coerce").notna().any():
        return pd.to_numeric(df["sw_in"], errors="coerce") * SW_TO_PPFD, "sw_in"
    col = "surface_solar_radiation_downwards_hourly"
    if col in df and pd.to_numeric(df[col], errors="coerce").notna().any():
        return pd.to_numeric(df[col], errors="coerce") * SW_TO_PPFD, "era5_ssrd"
    return pd.Series(np.nan, index=df.index), "none"


def normalize_per_site(series: pd.Series) -> pd.Series:
    """Divide by the mean of values at/above the 90th percentile (Liu).

    Uses >= the quantile so the anchor is never empty; a constant series
    normalizes to 1.0 everywhere (no div-by-zero).
    """
    s = pd.to_numeric(series, errors="coerce")
    anchor = s[s >= s.quantile(NORM_QUANTILE)].mean()
    if not np.isfinite(anchor) or anchor == 0:
        return pd.Series(np.nan, index=s.index)
    return s / anchor


# --- part 2: day filter + site-frame standardisation ---

_CARRY_COLS = {
    "pft": "pft",
    "biome": "biome",
    "prcip/PET": "aridity",
    "canopy_height": "canopy_height",
    "elevation": "elevation",
    "latitude_x": "lat",
    "longitude_x": "lon",
}


def _kelvin_to_celsius(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s - 273.15 if s.median(skipna=True) > 100 else s


def _vpd_from_era5(tair_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    """VPD (kPa) from temperature and dewpoint via Tetens (es - ea)."""
    es = 0.6108 * np.exp(17.27 * tair_c / (tair_c + 237.3))
    ea = 0.6108 * np.exp(17.27 * dewpoint_c / (dewpoint_c + 237.3))
    return (es - ea).clip(lower=0.0)


def standardise_site_frame(df: pd.DataFrame, climate_source: ClimateSource) -> pd.DataFrame:
    """Map one site's raw daily CSV to the analysis schema (E, Gc, drivers, SM, carry)."""
    out = pd.DataFrame(index=df.index)
    out["site_name"] = df["site_name"] if "site_name" in df.columns else "unknown"
    out["date"] = pd.to_datetime(df["TIMESTAMP"]).dt.date

    e = pd.to_numeric(df["sap_velocity"], errors="coerce")
    e = e.mask(e < 0)  # reverse/invalid flow
    out["E"] = e

    if climate_source == "era5":
        tair = _kelvin_to_celsius(df["temperature_2m"])
        out["tair"] = tair
        out["vpd"] = _vpd_from_era5(tair, _kelvin_to_celsius(df["dewpoint_2m"]))
    else:
        out["tair"] = pd.to_numeric(df["ta"], errors="coerce")
        out["vpd"] = pd.to_numeric(df["vpd"], errors="coerce")

    out["ppfd"], out["ppfd_source"] = resolve_ppfd(df)

    out["swvl1"] = pd.to_numeric(df["volumetric_soil_water_layer_1"], errors="coerce")
    out["swvl2"] = pd.to_numeric(df["volumetric_soil_water_layer_2"], errors="coerce")
    out["swvl3"] = pd.to_numeric(df["volumetric_soil_water_layer_3"], errors="coerce")
    out["swvl4"] = pd.to_numeric(df["volumetric_soil_water_layer_4"], errors="coerce")
    out["root_zone_sm"] = root_zone_sm(out["swvl1"], out["swvl2"], out["swvl3"])

    elevation = pd.to_numeric(df.get("elevation", pd.Series(np.nan, index=df.index)), errors="coerce")
    out["Gc"] = canopy_conductance(out["E"], out["tair"], out["vpd"], elevation.fillna(0.0))

    for raw_col, std_col in _CARRY_COLS.items():
        if raw_col in df.columns:
            out[std_col] = df[raw_col]
    return out


def apply_day_filter(table: pd.DataFrame, tair_min: float) -> pd.DataFrame:
    """Liu day filter (AND): Tair > tair_min, VPD > 0.5 kPa, PPFD > 500 umol m-2 s-1."""
    before = len(table)
    mask = (table["tair"] > tair_min) & (table["vpd"] > VPD_MIN_KPA) & (table["ppfd"] > PPFD_MIN)
    out = table[mask].copy()
    logger.info(
        "Day filter (Tair>%.0f,VPD>%.1f,PPFD>%.0f): %d -> %d rows",
        tair_min,
        VPD_MIN_KPA,
        PPFD_MIN,
        before,
        len(out),
    )
    return out


# --- part 3: directory resolution + full table build ---

_DEFAULT_CANDIDATES = (
    "merged_decoupling/daily",
    "merged_decoupling/daytime_only/daily",
    "merged/daytime_only/daily",
)


def resolve_daily_dir(data_dir: str | None, processed_root: Path | None = None) -> Path:
    """Return the daily CSV directory. An explicit ``data_dir`` wins; otherwise
    search known candidates under ``processed_root/sapwood``."""
    if data_dir is not None:
        d = Path(data_dir)
        if not (d.exists() and any(d.glob("*.csv"))):
            raise FileNotFoundError(f"--data-dir has no *.csv files: {d}")
        return d
    root = (processed_root or Path("outputs/processed_data")) / "sapwood"
    for cand in _DEFAULT_CANDIDATES:
        d = root / cand
        if d.exists() and any(d.glob("*.csv")):
            logger.info("Resolved daily dir: %s", d)
            return d
    raise FileNotFoundError(f"No daily dir found under {root}")


def load_table(
    data_dir: str | None,
    climate_source: ClimateSource = "site",
    tair_min: float = 15.0,
    processed_root: Path | None = None,
) -> pd.DataFrame:
    """Full pipeline: per-site daily CSVs -> filtered, normalized site-day table.

    Normalization (E_norm, Gc_norm) is computed PER SITE on the day-filtered rows.
    """
    daily_dir = resolve_daily_dir(data_dir, processed_root)
    files = [f for f in sorted(daily_dir.glob("*.csv")) if "all_biomes" not in f.name]
    if not files:
        raise FileNotFoundError(f"No per-site daily CSVs in {daily_dir}")

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            raw = pd.read_csv(f)
            if raw.empty or "sap_velocity" not in raw.columns:
                continue
            std = standardise_site_frame(raw, climate_source)
            std = apply_day_filter(std, tair_min=tair_min)
            if std.empty:
                continue
            std["E_norm"] = normalize_per_site(std["E"])
            std["Gc_norm"] = normalize_per_site(std["Gc"])
            frames.append(std)
        except (ValueError, KeyError, AttributeError, UnicodeDecodeError, OSError) as exc:
            logger.warning("Skipping %s - %s: %s", f.name, type(exc).__name__, exc)
    if not frames:
        raise ValueError("No site files survived loading/filtering.")
    table = pd.concat(frames, ignore_index=True)
    logger.info("Loaded table: %d rows, %d sites", len(table), table["site_name"].nunique())
    return table
