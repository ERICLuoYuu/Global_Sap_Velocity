import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

# Assuming path_config is in your local environment
from path_config import get_default_paths
from src.feature_contract.definitions import SERVABLE_CONTRACT, validate_contract
from src.feature_contract.merge_contract import (
    DEFAULT_MAX_RADIUS_CELLS,
    DEFAULT_SWVL_CUBE,
    FeatureContractError,
    apply_servable_precip,
    cell_rows,
    clear_sidecar,
    contract_block,
    load_cube,
    preflight_cells,
    require_era5_precip,
    rename_gauge_columns,
    swc_climatology_moments,
    write_sidecar,
)
from src.site_info.pft_resolution import load_site_landcover_modes, resolve_pft, summarise
from src.site_info.derive_igbp import derive_for_scale, derived_only

paths = get_default_paths()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        # On Palma compute nodes git may not be on PATH; print so the
        # MANIFEST author can see why it says 'unknown'.
        print(f"  [MANIFEST] could not read git SHA: {exc}")
        return "unknown"


def calculate_soil_hydraulics(sand, clay, organic_matter, coarse_fragments_vol_percent):
    """
    Saxton & Rawls (2006) two-step soil hydraulic property estimation.

    Parameters
    ----------
    sand, clay : float
        Fractions (0-1), NOT percentages.
    organic_matter : float
        Percentage (0-5 typical).
    coarse_fragments_vol_percent : float
        Volumetric coarse fragment percentage (0-100).

    Returns
    -------
    (theta_wp, theta_fc, theta_sat) : tuple of float
        Volumetric water content (m³/m³) at wilting point, field capacity,
        and saturation, adjusted for coarse fragments.
    """
    S, C, OM = sand, clay, organic_matter

    # 1. Wilting point (1500 kPa) — first estimate then adjustment
    t1500t = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * (S * OM) - 0.013 * (C * OM) + 0.068 * (S * C) + 0.031
    theta_wp = t1500t + (0.14 * t1500t - 0.02)

    # 2. Field capacity (33 kPa) — first estimate then adjustment
    #    Correction uses t33t itself, NOT theta_wp
    t33t = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * (S * OM) - 0.027 * (C * OM) + 0.452 * (S * C) + 0.299
    theta_fc = t33t + (1.283 * t33t**2 - 0.374 * t33t - 0.015)

    # 3. Saturation (porosity) — three steps
    ts33t = 0.278 * S + 0.034 * C + 0.022 * OM - 0.018 * (S * OM) - 0.027 * (C * OM) - 0.584 * (S * C) + 0.078
    ts33 = ts33t + (0.636 * ts33t - 0.107)
    theta_sat = theta_fc + ts33 - 0.097 * S + 0.043

    # Adjust for coarse fragments (rocks reduce pore volume)
    cf_frac = coarse_fragments_vol_percent / 100.0
    return (
        theta_wp * (1 - cf_frac),
        theta_fc * (1 - cf_frac),
        theta_sat * (1 - cf_frac),
    )


def get_weighted_soil_props(soil_row, depths=["0_5", "5_15", "15_30", "30_60", "60_100"]):
    """
    Calculate thickness-weighted average for soil properties down to 100cm.
    """
    thickness_map = {"0_5": 5, "5_15": 10, "15_30": 15, "30_60": 30, "60_100": 40}
    properties = ["sand", "clay", "soc", "bdod", "cfvo"]
    weighted_sums = {p: 0.0 for p in properties}
    total_thickness = 0.0

    for depth in depths:
        thick = thickness_map.get(depth, 0)
        if f"sand_{depth}" not in soil_row.index:
            continue
        if pd.isna(soil_row.get(f"sand_{depth}")):
            continue

        total_thickness += thick
        for p in properties:
            val = soil_row.get(f"{p}_{depth}", 0)
            if pd.isna(val):
                val = 0
            weighted_sums[p] += val * thick

    if total_thickness == 0:
        return {p: np.nan for p in properties}

    return {p: weighted_sums[p] / total_thickness for p in properties}


def filter_growing_season_by_sapflow(
    df_daily: pd.DataFrame,
    sap_col: str = "sap_velocity",
    smooth_window: int = 14,
    amplitude_threshold: float = 0.3,
    sap_pct_threshold: float = 0.20,
) -> tuple:
    """
    Filter daily data to growing-season rows using sap velocity amplitude.

    Algorithm (operates on full time series, no per-year loops):
      1. Smooth sap_velocity with a centered rolling mean (14-day default),
         computed only on non-NaN observations to avoid bias from multi-year gaps.
      2. Compute robust relative amplitude = (q95 - q05) / median of smoothed
         series (insensitive to single-year anomalies).
      3. If amplitude < 0.3 → site is evergreen-like → keep ALL rows.
      4. Else → threshold = 20% of smoothed 95th percentile
               → keep rows where smoothed > threshold.

    Assumes daily-frequency rows in calendar order. Hourly DataFrames must be
    aggregated to daily before calling this function.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily-aggregated data with a sap_velocity column.
    sap_col : str
        Column name for sap velocity (default ``'sap_velocity'``).
    smooth_window : int
        Rolling-mean window in days (default 14).
    amplitude_threshold : float
        Robust relative amplitude below which the site is treated as evergreen (default 0.3).
    sap_pct_threshold : float
        Fraction of the smoothed 95th percentile used as the activity threshold (default 0.20).

    Returns
    -------
    (filtered_df, stats) : tuple[pd.DataFrame, dict]
    """
    stats = {}
    stats["n_before"] = len(df_daily)

    if sap_col not in df_daily.columns or df_daily[sap_col].dropna().empty:
        stats["rel_amplitude"] = np.nan
        stats["is_evergreen"] = True
        stats["threshold_value"] = np.nan
        stats["n_after"] = len(df_daily)
        stats["pct_retained"] = 100.0
        return df_daily, stats

    # Smooth only on non-NaN observations to avoid bias from multi-year data gaps
    # (resample("D") pads gaps with NaN rows; rolling across those distorts edge values)
    sap_valid = df_daily[sap_col].dropna()
    smoothed_valid = sap_valid.rolling(smooth_window, center=True, min_periods=1).mean()
    smoothed = smoothed_valid.reindex(df_daily.index)
    smoothed = smoothed.clip(lower=0)  # guard against negative artifacts from edge effects

    # Robust amplitude: (q95 - q05) / median — insensitive to single-year anomalies
    smooth_median = smoothed.median()
    q95 = smoothed.quantile(0.95)
    q05 = smoothed.quantile(0.05)
    if smooth_median > 0:
        rel_amplitude = (q95 - q05) / smooth_median
    elif q95 > 0:
        # Zero median but non-zero peaks → clearly seasonal (e.g. deciduous with long winter)
        rel_amplitude = float("inf")
    else:
        rel_amplitude = 0.0  # truly no signal
    stats["rel_amplitude"] = rel_amplitude

    if rel_amplitude < amplitude_threshold:
        stats["is_evergreen"] = True
        stats["threshold_value"] = np.nan
        stats["n_after"] = len(df_daily)
        stats["pct_retained"] = 100.0
        return df_daily, stats

    threshold = sap_pct_threshold * q95
    mask = smoothed > threshold
    filtered = df_daily.loc[mask]

    stats["is_evergreen"] = False
    stats["threshold_value"] = threshold
    stats["n_after"] = len(filtered)
    stats["pct_retained"] = 100.0 * len(filtered) / len(df_daily) if len(df_daily) > 0 else 0.0
    return filtered, stats


def calculate_daytime_mask(timestamps, lat, lon, elevation=0, elevation_threshold=0):
    """
    Calculate a boolean mask indicating daytime based on solar elevation.

    Args:
        timestamps: DatetimeIndex or datetime column
        lat: Latitude
        lon: Longitude
        elevation: Site elevation in meters
        elevation_threshold: Solar elevation angle threshold (0 = horizon)

    Returns:
        Boolean Series (True = daytime, False = nighttime)
    """
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(timestamps)

    if timestamps.tzinfo is None:
        timestamps = timestamps.tz_localize("UTC")

    solar_position = pvlib.solarposition.get_solarposition(timestamps, lat, lon, altitude=elevation)
    return solar_position["elevation"] > elevation_threshold


def load_stand_age_map(base_path):
    """
    Load the global stand age CSV and create a dictionary mapping site_name -> mean_age.
    """
    age_file_path = base_path / "extracted_data" / "terrain_site_data" / paths.scale / "stand_age_data.csv"

    if not age_file_path.exists():
        age_file_path = (
            paths.base_data_dir / "raw" / "extracted_data" / "terrain_site_data" / paths.scale / "stand_age_data.csv"
        )

    if age_file_path.exists():
        print(f"Loading Stand Age data from: {age_file_path}")
        df = pd.read_csv(age_file_path)
        tc_cols = [c for c in df.columns if "stand_age_TC" in c]
        if tc_cols:
            df["mean_age"] = df[tc_cols].mean(axis=1)
            return df.set_index("site_name")["mean_age"].to_dict()

    print(f"WARNING: Stand Age file not found at {age_file_path}")
    return {}


#: Written into ``biome`` when a site ships no ``si_biome``. A NAMED placeholder,
#: never NaN: the by-biome filename is built by ITERATING the label (NaN is not
#: iterable), and NaN never compares equal to itself, so a NaN key gives every
#: unlabelled site its own dict entry -- which then all collapse onto a single
#: filename and silently keep only the last. 24 internal sites have no si_biome.
UNCLASSIFIED_BIOME = "Unclassified"

#: Env variables the merge carries forward. Anything else an env file ships --
#: ``pa``, ``swc_shallow`` -- is dropped, which is what the old post-join
#: aggregation did by omitting them from its rules; naming them makes it explicit.
ENV_MEAN_COLS = ("vpd", "ta", "ws", "sw_in", "rh", "netrad", "ppfd_in", "ext_rad")

#: Env variables that are TOTALS accumulated over the reporting interval rather
#: than instantaneous states, so they are summed when collapsing to the hour.
ENV_SUM_COLS = ("precip",)

#: Sap-side columns that are static per tree and carried through unchanged.
SAP_META_COLS = ("pl_code", "pl_dbh", "pl_sens_meth", "pl_species")

#: Sap-side measurement columns that ``--daytime-sap-only`` rescopes to daylight.
#: Env names are deliberately absent. The global prediction is served from the CDS
#: product ``derived-era5-land-daily-statistics``, whose daily_mean / daily_minimum /
#: daily_maximum are 24-HOUR statistics, so narrowing an env column to daylight here
#: makes the trained feature mean something the served feature does not. Only the
#: target moves; adding a name to this tuple reintroduces that skew.
SAP_DAYTIME_COLS = ("sap_velocity",)


def apply_daytime_sap_scope(df_daily, df_hourly, daytime_mask, sap_cols=SAP_DAYTIME_COLS, plant_level=False):
    """Return ``df_daily`` with its sap statistics recomputed over daylight only.

    ``--daytime-only`` filters the hourly frame BEFORE the daily aggregation, so it
    rescopes every column at once. This is the narrow version: the 24-hour frame
    supplies the row index and every env column, and a second pass over the daylight
    hours alone overwrites the sap columns in place. Rows are never dropped -- a day
    keeps its env even when it holds no daylight at all.

    Days with no daylight hours come back NaN, not 0.0. ``resample("D").sum()``
    reports 0.0 for a day holding no rows, which reads downstream as a genuine
    "the tree moved no water" measurement rather than an absent one. ``min_count=1``
    is what stops that, and it is load-bearing at the polar sites (FIN_HYY, SWE_NOR)
    where whole days sit below the horizon.
    """
    present = [c for c in sap_cols if c in df_hourly.columns and c in df_daily.columns]
    if not present:
        return df_daily.copy()

    by_plant = plant_level and "pl_code" in df_hourly.columns and "pl_code" in df_daily.columns
    keys = ["TIMESTAMP", "pl_code"] if by_plant else ["TIMESTAMP"]

    wanted = ["TIMESTAMP"] + present + (["pl_code"] if by_plant else [])
    lit = df_hourly.loc[np.asarray(daytime_mask, dtype=bool), wanted].set_index("TIMESTAMP")

    if by_plant:
        # pl_code as an EXTERNAL grouping key, mirroring aggregate_sap_to_hourly:
        # resampling a frame that still carries its own grouping column is
        # deprecated, and selecting columns instead returns MultiIndex columns.
        codes = lit["pl_code"]
        resampler = lit.drop(columns=["pl_code"]).groupby(codes).resample("D")
    else:
        resampler = lit.resample("D")

    stats = {
        "": resampler.mean(),
        "_min": resampler.min(),
        "_max": resampler.max(),
        # min_count=1 keeps a day with no daylight rows at NaN instead of 0.0.
        "_sum": resampler.sum(min_count=1),
    }
    daytime = pd.concat(
        [frame.rename(columns={c: f"{c}{suffix}" for c in present}) for suffix, frame in stats.items()],
        axis=1,
    ).reset_index()

    # Align on the keys rather than merging, so the 24-hour frame keeps both its
    # row order and its column order; days absent from the daylight pass resolve
    # to NaN, which is the honest answer for a day that was never lit.
    out = df_daily.copy()
    target = daytime.set_index(keys)
    lookup = pd.MultiIndex.from_frame(out[keys]) if len(keys) > 1 else pd.Index(out[keys[0]])
    for col in target.columns:
        out[col] = target[col].reindex(lookup).to_numpy()
    return out


def _drop_manufactured_hours(hourly, rows_per_hour):
    """Remove hours ``resample`` invented across a gap.

    ``resample("h")`` returns a CONTIGUOUS index, so every hour inside a gap
    comes back as a row. Emptiness is judged by the number of SOURCE rows, not
    by the aggregated values, because a summed column reports 0.0 for an
    all-NaN hour and would make a genuinely empty hour look populated.
    """
    return hourly[rows_per_hour > 0]


def aggregate_env_to_hourly(env_data, mean_cols=ENV_MEAN_COLS, sum_cols=ENV_SUM_COLS):
    """Collapse env to one row per hour, keeping only hours that hold observations.

    Precipitation is a total over the interval and is summed HERE, once. Because
    the join downstream is hour-to-hour, this value is never duplicated, so no
    later step has to know not to re-sum it.

    An hour whose readings are all missing stays missing (``min_count=1``). The plain
    sum wrote 0.0 there: 541,019 hours at 155 sites, and 20,849 sap-days on which the
    gauge had no reading at all reached the model as dry days (measured 2026-09-14).
    """
    totals = [c for c in sum_cols if c in env_data.columns]
    means = [c for c in mean_cols if c in env_data.columns]
    binned = env_data[means + totals].resample("h")
    parts = []
    if totals:
        parts.append(binned[totals].sum(min_count=1))
    if means:
        parts.append(binned[means].mean())
    rows_per_hour = binned.size()
    hourly = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=rows_per_hour.index)
    return _drop_manufactured_hours(hourly, rows_per_hour)


def aggregate_sap_to_hourly(sap_data, plant_level=False):
    """Collapse sap to one row per hour -- per tree when ``plant_level``.

    The mirror of aggregate_env_to_hourly. Reducing each side on its own terms
    before joining is what lets two deliveries with different clocks meet at all:
    FRA_HES shares no instant with its env, but it shares every hour.
    """
    rules = {"sap_velocity": "mean"}
    if "solar_TIMESTAMP" in sap_data.columns:
        rules["solar_TIMESTAMP"] = "first"
    for col in SAP_META_COLS:
        if col != "pl_code" and col in sap_data.columns:
            rules[col] = "first"

    if plant_level and "pl_code" in sap_data.columns:
        # Group by pl_code as an EXTERNAL key, with the column dropped from the
        # frame being resampled. DataFrameGroupBy.resample otherwise operates on
        # the grouping column, which pandas deprecates; selecting columns instead
        # (``groupby(...)[[cols]]``) silences it but makes agg() return MultiIndex
        # columns, which then cannot be joined against the single-level env frame.
        # pl_code comes back as an index level and is restored below.
        codes = sap_data["pl_code"]
        grouped = sap_data.drop(columns=["pl_code"]).groupby(codes)
        hourly = _drop_manufactured_hours(grouped.resample("h").agg(rules), grouped.resample("h").size())
        return hourly.reset_index().set_index("TIMESTAMP")

    return _drop_manufactured_hours(sap_data.resample("h").agg(rules), sap_data.resample("h").size())


#: Instants are rounded to the minute before any hourly binning. 79 of 225 env files store some instants
#: one second early ("01:59:59"); floored to the hour, that reading lands in the previous hour, and at
#: SEN_SOU_* every midnight reading lands in the previous UTC day (measured 2026-09-14).
TIMESTAMP_RESOLUTION = "min"

HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 1440

#: ERA5-Land extract columns that identify the site rather than describe its weather.
ERA5_ID_COLS = ("site_id", "site_name", "latitude", "longitude")

#: Daily statistics written per dynamic column; the mean keeps the bare name.
DAILY_STAT_SUFFIXES = ("", "_min", "_max", "_sum")

#: ERA5-Land soil-moisture layers, and the per-site standard deviation below which they are not normalised.
SWC_LAYERS = (1, 2, 3, 4)
SWC_MIN_STD = 1e-10


def round_instants(frame):
    """Return a copy of ``frame`` with its DatetimeIndex rounded to :data:`TIMESTAMP_RESOLUTION`."""
    out = frame.copy()
    out.index = pd.DatetimeIndex(out.index.round(TIMESTAMP_RESOLUTION), name=frame.index.name)
    return out


def normalise_timestamps(frame):
    """Round the index to the minute and collapse the rows rounding made one instant.

    Each column takes its first non-missing value. A duplicated instant would be summed twice by the hourly
    precipitation total; keeping the whole first row instead dropped the second of two logger streams stored
    seconds apart (PRT_LEZ_ARN 16:00:00 env empty, 16:00:09 env valid): 285,967 values at 15 sites
    (measured 2026-09-14).
    """
    out = round_instants(frame)
    if not out.index.has_duplicates:
        return out
    return out.groupby(level=0).first()


def _utc(index):
    index = pd.DatetimeIndex(index)
    return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")


def daylight_hours(lat, lon, elevation):
    """A ``keep_hours`` callable for ``--daytime-only``: True where the sun is above the horizon."""
    elevation = 0 if pd.isna(elevation) else elevation

    def keep(stamps):
        return np.asarray(calculate_daytime_mask(stamps, lat=lat, lon=lon, elevation=elevation), dtype=bool)

    return keep


def _expected_hours(days, keep_hours):
    """Hours a complete day needs: 24, or the hours of that day ``keep_hours`` retains."""
    if keep_hours is None or len(days) == 0:
        return pd.Series(HOURS_PER_DAY, index=days)
    grid = pd.date_range(days.min(), days.max() + pd.Timedelta(hours=HOURS_PER_DAY - 1), freq="h")
    kept = pd.Series(keep_hours(grid), index=grid)
    return kept.groupby(grid.floor("D")).sum().reindex(days).fillna(0).astype(int)


def _stat_columns(columns):
    return [col + suffix for col in columns for suffix in DAILY_STAT_SUFFIXES]


def complete_day_statistics(hourly, columns, keep_hours=None):
    """Daily mean (bare name), ``_min``, ``_max`` and ``_sum`` of hourly ``columns``, one row per UTC day.

    ``hourly`` holds at most one row per hour. A column's statistics for a day are NaN unless every hour
    the day needs holds a value -- 24, or the hours ``keep_hours`` retains -- so a daily total is never
    the total of part of a day, and a daily maximum never misses the afternoon.
    """
    columns = list(columns)
    frame = hourly[columns].set_axis(_utc(hourly.index), axis=0)
    if frame.index.has_duplicates:
        raise ValueError("complete_day_statistics needs one row per hour; the index has duplicates")
    if keep_hours is not None and len(frame):
        frame = frame[keep_hours(frame.index)]
    if frame.empty:
        return pd.DataFrame(columns=_stat_columns(columns), index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
    day = frame.resample("D")
    count = day.count()
    complete = count.ge(_expected_hours(count.index, keep_hours), axis=0) & count.gt(0)
    stats = {"": day.mean(), "_min": day.min(), "_max": day.max(), "_sum": day.sum(min_count=1)}
    parts = [values.where(complete).add_suffix(suffix) for suffix, values in stats.items()]
    return pd.concat(parts, axis=1)[_stat_columns(columns)]


def swc_site_moments(hourly):
    """Mean and standard deviation of each ERA5-Land soil-moisture layer over the site's merged hours."""
    moments = {}
    for layer in SWC_LAYERS:
        col = f"volumetric_soil_water_layer_{layer}"
        if col in hourly.columns:
            moments[col] = (hourly[col].mean(), hourly[col].std())
    return moments


def add_swc_representations(frame, moments, site, warn=True):
    """Return a copy of ``frame`` with three representations of each soil-moisture layer in ``moments``.

    ``*_raw`` is the level in m3/m3; the bare column becomes raw / sigma and ``*_zscore`` (raw - mu) / sigma,
    mu and sigma taken from ``moments``. A sigma at or below :data:`SWC_MIN_STD` (or NaN) writes 0.0 to both.
    """
    out = frame.copy()
    for col, (mean, std) in moments.items():
        if col not in out.columns:
            continue
        raw = out[col]
        out[f"{col}_raw"] = raw
        if std > SWC_MIN_STD:
            out[col] = raw / std
            out[f"{col}_zscore"] = (raw - mean) / std
        else:
            out[col] = 0.0
            out[f"{col}_zscore"] = 0.0
            if warn:
                print(f"  Warning: {site} has near-zero std for {col}")
    return out


def era5_daily_for_site(era5_site, swc_moments, site, keep_hours=None):
    """Daily statistics of every ERA5-Land column over ERA5's own hours of each UTC day.

    ``era5_site`` is this site's slice of the extract, TIMESTAMP as a column. The merged hourly frame holds
    only the hours in which sap AND env observed, so statistics taken there covered part of a day wherever
    either side had a gap: 2.2% of sap-days, where total precipitation read 19% low and the 2 m
    temperature maximum more than 1 K low on 267 days (measured 2026-09-14). ERA5-Land has every hour.
    Soil temperature is aggregated like every other column; the old ``soil_`` static prefix took its
    first hour. The soil-moisture representations use the moments of the merged hours, as hourly.
    """
    frame = era5_site.drop(columns=[c for c in ERA5_ID_COLS if c in era5_site.columns])
    frame = frame.set_index("TIMESTAMP").sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    frame = add_swc_representations(frame, swc_moments, site, warn=False)
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    return complete_day_statistics(frame, numeric, keep_hours)


def gauge_reporting_minutes(values):
    """The usual interval between a gauge's valid readings, in minutes; NaN with fewer than two."""
    stamps = pd.DatetimeIndex(values.dropna().index)
    if len(stamps) < 2:
        return np.nan
    steps = np.diff(stamps.as_unit("s").asi8) / 60.0
    steps = steps[steps > 0]
    return float(pd.Series(steps).mode().iloc[0]) if len(steps) else np.nan


def gauge_daily_totals(readings, col, keep_hours=None):
    """Daily totals of gauge column ``col`` from the env record's own readings, one row per UTC day.

    Taken from the env file, not the merged hourly frame, which drops env hours that have no sap. A day is
    complete when every slot of it holds a valid reading. A slot is one hour, or the gauge's reporting
    interval where that is longer (CRI_TAM_TOW reports once a day, RUS_CHE_Y4 every 12 h). A reading
    missing inside an hour of a sub-hourly gauge is not detected.

    Columns: ``{col}`` and ``{col}_sum`` the total, ``{col}_min`` / ``{col}_max`` the smallest and largest
    slot total -- all NaN on an incomplete day -- and ``{col}_coverage``, the fraction of slots holding a
    reading. ``keep_hours`` scopes an hourly-slot gauge to the retained hours; a coarser gauge cannot be
    split and is not scoped.
    """
    names = [col, f"{col}_min", f"{col}_max", f"{col}_sum", f"{col}_coverage"]
    empty = pd.DataFrame(columns=names, index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
    values = pd.to_numeric(readings[col], errors="coerce").set_axis(_utc(readings.index)).dropna()
    if values.empty:
        return empty
    step = gauge_reporting_minutes(values)
    slot = int(round(step)) if step > MINUTES_PER_HOUR else MINUTES_PER_HOUR
    totals = values.resample(f"{slot}min", origin="start_day").sum(min_count=1).dropna()
    hourly_slots = slot == MINUTES_PER_HOUR
    if hourly_slots and keep_hours is not None:
        totals = totals[keep_hours(totals.index)]
    if totals.empty:
        return empty
    day = totals.resample("D")
    count = day.count()
    if hourly_slots:
        expected = _expected_hours(count.index, keep_hours)
    else:
        expected = pd.Series(max(1, int(round(MINUTES_PER_DAY / slot))), index=count.index)
    complete = (count >= expected) & (count > 0)
    total = day.sum(min_count=1).where(complete)
    out = pd.DataFrame(
        {
            col: total,
            f"{col}_min": day.min().where(complete),
            f"{col}_max": day.max().where(complete),
            f"{col}_sum": total,
            f"{col}_coverage": (count / expected.where(expected > 0)).clip(upper=1.0),
        }
    )
    return out[names]


def overwrite_daily_columns(df_daily, daily, day_col="TIMESTAMP"):
    """Return a copy of ``df_daily`` with the columns of ``daily`` set by UTC day.

    A day ``daily`` does not hold becomes NaN. Existing columns keep their position and new ones are
    appended; rows are never added or dropped, and every tree of a plant-level frame gets its day's value.
    """
    out = df_daily.copy()
    lookup = _utc(out[day_col])
    for col in daily.columns:
        out[col] = daily[col].reindex(lookup).to_numpy()
    return out


def _preflight_sites(sap_input_dir, raw_csv_dir, suffix_words=4):
    """(site, lat, lon) for every sap file whose site metadata carries a coordinate.

    The climatology lookup runs over this before the main loop, so a site that cannot be placed is
    known before any work is done. Three kinds of site are left out here, and they do NOT share a
    fate. A site whose ``*_site_md.csv`` is missing or unreadable is skipped by the main loop too --
    the loop reads the same file inside a try whose except continues -- so it is absent either way.
    A site whose coordinate is NaN is NOT skipped by the loop: under the servable contract it raises
    there instead, naming the site, rather than quietly shrinking the corpus. Each is reported.
    """
    sites = []
    for sap_file in sorted(sap_input_dir.glob("*.csv")):
        site = "_".join(sap_file.stem.split("_")[:-suffix_words])
        md_file = raw_csv_dir / f"{site}_site_md.csv"
        if not md_file.is_file():
            print(f"  [contract] {site}: no {md_file.name}; the main loop skips it as well")
            continue
        try:
            md = pd.read_csv(md_file, usecols=["si_lat", "si_long"], nrows=1)
        except Exception as exc:  # noqa: BLE001 - an unreadable coordinate stays the loop's business
            print(f"  [contract] {site}: cannot read si_lat/si_long ({exc})")
            continue
        if md.empty or pd.isna(md["si_lat"][0]) or pd.isna(md["si_long"][0]):
            # The loop does NOT skip this one: it will raise under the servable contract. Say so
            # here, where the reason is known, so the later failure is not the first anyone hears.
            print(f"  [contract] {site}: si_lat/si_long is empty, so it cannot be placed")
            continue
        sites.append((site, float(md["si_lat"][0]), float(md["si_long"][0])))
    return sites


def merge_sap_env_data_site(
    output_base_dir,
    daytime_only=False,
    daytime_sap_only=False,
    plant_level=False,
    growing_season_only=False,
    apply_flo2019_correction_flag=False,
    apply_treatment_filter_flag=False,
    feature_contract=SERVABLE_CONTRACT,
    swvl_cube=DEFAULT_SWVL_CUBE,
):
    """
    Merge sap flow and environmental data for all sites.

    Args:
        output_base_dir: Base directory for output files
        daytime_only: If True, filter to daytime data only before daily aggregation.
                     Rescopes EVERY column, env variables included.
        daytime_sap_only: If True, scope only the sap target to daylight and leave
                     env variables at their 24-hour daily statistics, matching the
                     CDS product the global prediction is served from. Mutually
                     exclusive with daytime_only.
        plant_level: If True, output one row per tree per timestamp (long format)
                     instead of site-averaged sap velocity. Joins plant_md.csv
                     to add tree metadata (pl_dbh, pl_species, pl_sens_meth, etc.).
        growing_season_only: If True, apply sap-flow-based growing season filter
                             after daily aggregation to remove dormant periods.
        apply_flo2019_correction_flag: If True, multiply non-calibrated HD/CHD
                             plants' SFD by Flo 2019 multipliers (HD ≈1.6804,
                             CHD ≈1.6373) before melt/aggregation.
        apply_treatment_filter_flag: If True, drop plant columns whose
                             pl_treatment / st_treatment indicates predictor-
                             target decoupling (irrigation, drought, root
                             trenching, elevated CO2, shade).
        feature_contract: servable (the default) normalises soil moisture against the
                             ERA5-Land climatology cube and writes precip as the ERA5
                             daily total in mm/day, keeping the gauge own totals under
                             precip_gauge*. The tree is then ready to train and to serve
                             with no further pass. site is the legacy behaviour:
                             per-site soil-moisture moments and gauge precip.
        swvl_cube: The soil-moisture climatology the servable contract normalises
                             against. Ignored under the site contract.
    """

    # Imports kept inside the function so they are only resolved when the
    # merge is actually invoked (avoids module-import-time cost for tools
    # that load this file just to scrape its argparse). Hoisted out of the
    # per-site loop per code review.
    from src.Analyzers.calibration_corrector import (
        FLO2019_CHD_MULTIPLIER,
        FLO2019_HD_MULTIPLIER,
        apply_flo2019_correction,
    )
    from src.Analyzers.treatment_filter import filter_by_treatment

    # Collect per-site growing season stats
    gs_stats_list = []
    # Per-plant audit records for calibration + treatment filter actions
    correction_audit_records: list[dict] = []
    # Sites that survived the empty-plants guard (used for MANIFEST.json)
    sites_retained = 0

    if daytime_only:
        print("\n" + "=" * 60)
        print("DAYTIME-ONLY MODE ENABLED")
        print("Nighttime data will be filtered out before daily aggregation")
        print("=" * 60 + "\n")

    if daytime_sap_only:
        print("\n" + "=" * 60)
        print("DAYTIME-SAP-ONLY MODE ENABLED")
        print("Sap velocity is aggregated over daylight hours only;")
        print("env variables keep their 24-hour daily statistics")
        print("=" * 60 + "\n")

    if growing_season_only:
        print("\n" + "=" * 60)
        print("GROWING-SEASON-ONLY MODE ENABLED")
        print("Dormant-period rows will be removed after daily aggregation")
        print("using sap-velocity amplitude gate + 20% of q95 threshold")
        print("=" * 60 + "\n")

    if plant_level:
        print("\n" + "=" * 60)
        print("PLANT-LEVEL MODE ENABLED")
        print("Output: one row per tree per timestamp (long format)")
        print("Tree metadata from plant_md.csv will be joined")
        print("=" * 60 + "\n")

    # --- SETUP DIRECTORIES ---
    if not output_base_dir.exists():
        output_base_dir.mkdir(parents=True, exist_ok=True)

    hourly_out_dir = output_base_dir / "hourly"
    hourly_out_dir.mkdir(exist_ok=True)

    daily_out_dir = output_base_dir / "daily"
    daily_out_dir.mkdir(exist_ok=True)

    # --- FEATURE CONTRACT ---
    # Read before anything else: a tree labelled servable but normalised against something else
    # cannot be told from a correct one once it is written.
    contract = validate_contract(feature_contract)
    clim = load_cube(swvl_cube) if contract == SERVABLE_CONTRACT else None
    contract_cell_rows = []

    # Collections for final merge
    biome_merged_hourly = {}
    biome_merged_daily = {}

    all_pft_types = set()
    site_biome_mapping = {}

    # --- LOAD STATIC DATASETS ---
    site_info = pd.read_csv(paths.env_extracted_data_path)

    soil_file_path = paths.terrain_attributes_data_path.parent / "soilgrids_data.csv"
    if soil_file_path.exists():
        soil_data = pd.read_csv(soil_file_path)
        print("Loaded SoilGrids data.")
    else:
        print("WARNING: SoilGrids data not found.")
        soil_data = pd.DataFrame()

    stand_age_map = load_stand_age_map(paths.raw_csv_dir.parent)

    # --- LOAD FAN 2017 ROOT DEPTH + HYDROLOGIC REGIME FEATURES ---
    # Left-joined into site_info so the per-site `.values[0]` lookup below
    # returns them alongside canopy_height / bio1 / etc. Missing file → NaN
    # columns (same soft-fail pattern as soil_data / stand_age).
    _FAN2017_FEATURE_COLS = (
        "root_depth",
        "infiltration_depth",
        "deep_drainage",
        "drainage_frequency",
        "regolith_flush_rate",
        "gw_residence_time",
    )
    if paths.root_depth_fan2017_data_path.exists():
        print(f"Loading Fan 2017 root depth from: {paths.root_depth_fan2017_data_path}")
        _rd_data = pd.read_csv(paths.root_depth_fan2017_data_path)
        _rd_keep = ["site_name"] + [c for c in _FAN2017_FEATURE_COLS if c in _rd_data.columns]
        site_info = site_info.merge(_rd_data[_rd_keep], on="site_name", how="left")
    else:
        print(f"WARNING: Fan 2017 root depth file not found at {paths.root_depth_fan2017_data_path}")
        for _col in _FAN2017_FEATURE_COLS:
            site_info[_col] = np.nan

    era5_data = pd.read_csv(paths.era5_discrete_data_path)
    # Avoid division by zero
    pet = np.abs(era5_data["potential_evaporation_hourly"].values)
    precip = era5_data["total_precipitation_hourly"].values
    # Avoid division by zero
    pet = np.where(pet < 1e-10, 1e-10, pet)
    precip_PET = np.clip(precip / pet, 0, 10)
    era5_data["prcip/PET"] = precip_PET
    lai_data = pd.read_csv(paths.globmap_lai_data_path)

    # MODIS land cover, reduced to ONE short IGBP code per site (mode across
    # years).  This product used to be read here in full -- 5.4 M hourly rows,
    # 303 MB -- mapped to long class names, timestamp-normalised, sorted, and
    # then never joined to anything.  `pft` has always come from the provider's
    # si_igbp a few lines below, and si_igbp is ABSENT at 23 of the 226 corpus
    # sites (the entire internal/ICOS set), which therefore carried pft = NaN.
    # The reduction is the fallback for exactly those sites.
    #
    # The vocabularies have to agree: si_igbp holds short IGBP codes
    # (ENF/DBF/EBF/MF/SAV/DNF/WSA/WET), so the old long-name mapping could
    # never have been substituted into the same column.
    modis_pft_by_site = {}
    if Path(paths.pft_data_path).exists():
        modis_pft_by_site = load_site_landcover_modes(paths.pft_data_path)
        print(f"MODIS land cover reduced to one class at {len(modis_pft_by_site)} site(s)")
    else:
        print(f"WARNING: land cover product not found at {paths.pft_data_path}")

    # si_igbp is absent at the 23 internal sites, which used to mean they fell
    # straight through to MODIS.  Derive it from the stand and species metadata
    # those sites ship instead: st_forest_type is the description the provider
    # wrote, and it beats a 463 m pixel that called a holm-oak dehesa Croplands.
    # Recomputed every run, so a Stage-1 regeneration has nothing to revert.
    derived_pft_by_site = derived_only(derive_for_scale(str(paths.raw_csv_dir)))
    print(f"si_igbp derived from site metadata at {len(derived_pft_by_site)} site(s)")
    pft_sources = {}

    for df_temp in [lai_data, era5_data]:
        df_temp.rename(columns={"timestamp": "TIMESTAMP", "datetime": "TIMESTAMP"}, inplace=True)
        df_temp["TIMESTAMP"] = pd.to_datetime(df_temp["TIMESTAMP"], utc=True)
        df_temp.sort_values("TIMESTAMP", inplace=True)

    count = 0

    # --- Always use outliers_removed (gap-filling is NOT part of the main pipeline) ---
    sap_input_dir = paths.sap_outliers_removed_dir
    env_input_dir = paths.env_outliers_removed_dir
    print(f"Sap input: {sap_input_dir}")
    print(f"Env input: {env_input_dir}")

    # Every climatology lookup happens here, outside the loop: the loop ends in a bare except, so a
    # lookup that failed inside it would leave the site out of the tree with only a printed line.
    _placed = _preflight_sites(sap_input_dir, paths.raw_csv_dir) if clim else []
    cells_by_site = preflight_cells(clim, _placed) if clim else {}
    print(f"Feature contract: {contract}")
    if contract != SERVABLE_CONTRACT:
        # Nothing may be left claiming this tree is servable: the gate believes the badge.
        for _stale in clear_sidecar(output_base_dir):
            print(f"  removed a stale contract badge from an earlier run: {_stale.name}")
    if clim is not None:
        # Whether the extract can supply servable precipitation is a property of the extract, not of
        # any one site, so it is settled here rather than on the first site to reach the daily pass.
        require_era5_precip(era5_data, paths.era5_discrete_data_path)
        print(f"  climatology {clim.path}: {len(cells_by_site)} sites placed")
        # servable is the DEFAULT, so a script written before this existed now builds a servable tree
        # without asking. Two columns change meaning without changing name; a reader of this log is
        # the last person who can notice before the numbers are downstream.
        print("  precip is now the ERA5-Land daily TOTAL in mm/day, not the gauge daily mean of")
        print("  hourly mm -- about 24x larger. Soil moisture is normalised by the climatology")
        print("  cell, not by the record of each site. Pass --feature-contract site for the")
        print("  previous behaviour; the tree records which contract built it either way.")
        # Written BEFORE the loop, because the loop writes each site's CSV as it goes (:1268, :1438)
        # and a run killed part way through would otherwise leave servable numbers with no badge --
        # which the site contract accepts. This one names every placed site; the loop's own badge
        # below replaces it with the sites actually kept.
        _provisional = [row for _s, _la, _lo in _placed for row in cell_rows(_s, _la, _lo, cells_by_site[_s])]
        write_sidecar(
            output_base_dir,
            contract_block(clim, SWC_LAYERS, DEFAULT_MAX_RADIUS_CELLS, _provisional, _git_short_sha(), complete=False),
            _provisional,
        )

    # --- MAIN LOOP ---
    for sap_data_file in sap_input_dir.glob("*.csv"):
        try:
            parts = sap_data_file.stem.split("_")
            # Handle both "_sapf_data_gap_filled" and "_sapf_data_outliers_removed" suffixes (4 words each)
            suffix_words = 4
            location_type = "_".join(parts[:-suffix_words])

            # --- LOAD SITE METADATA ---
            env_data_file = env_input_dir / f"{location_type}_env_data_outliers_removed.csv"
            biome_data_file = paths.raw_csv_dir / f"{location_type}_site_md.csv"

            try:
                biome_info = pd.read_csv(biome_data_file)
                # Normalised HERE, at the single point of read, so the dict key,
                # site_biome_mapping and the biome column can never disagree, and
                # so a NaN can never reach the by-biome filename builder.
                _raw_biome = biome_info["si_biome"][0]
                biome_type = (
                    str(_raw_biome).strip()
                    if pd.notna(_raw_biome) and str(_raw_biome).strip()
                    else UNCLASSIFIED_BIOME
                )
                # Provider value first, metadata-derived second, MODIS third.
                # resolve_pft never
                # raises: this sits inside a try whose except skips the site
                # entirely, so an exception here would silently shrink the
                # training corpus.
                igbp_type, _pft_source = resolve_pft(
                    biome_info["si_igbp"][0],
                    modis_pft_by_site.get(location_type),
                    derived_pft_by_site.get(location_type),
                )
                pft_sources[location_type] = _pft_source
                lat = biome_info["si_lat"][0]
                lon = biome_info["si_long"][0]

                curr_site_info = site_info[site_info["site_name"] == location_type]
                if curr_site_info.empty:
                    continue

                ele = curr_site_info["elevation_m"].values[0]
                slope = curr_site_info["slope_deg"].values[0]
                aspect_sin = curr_site_info["aspect_sin"].values[0]
                aspect_cos = curr_site_info["aspect_cos"].values[0]
                mean_annual_temp = curr_site_info["bio1"].values[0]
                mean_annual_precip = curr_site_info["bio12"].values[0]
                canopy_height = curr_site_info["canopy_height_m"].values[0]
                # Fan 2017 static features (left-joined into site_info above).
                # Missing sites → NaN, consistent with stand_age/soil fallback.
                fan2017_features = {
                    _col: (curr_site_info[_col].values[0] if _col in curr_site_info.columns else np.nan)
                    for _col in _FAN2017_FEATURE_COLS
                }
                temp_seasonality = curr_site_info["bio4"].values[0]
                precip_seasonality = curr_site_info["bio15"].values[0]

                # --- 1. EXTRACT STAND AGE ---
                stand_age = np.nan
                if location_type in stand_age_map:
                    stand_age = stand_age_map[location_type]

                if pd.isna(stand_age):
                    stand_md_file = paths.raw_csv_dir / f"{location_type}_stand_md.csv"
                    if stand_md_file.exists():
                        try:
                            s_md = pd.read_csv(stand_md_file)
                            if "st_age" in s_md.columns:
                                val = s_md["st_age"].iloc[0]
                                stand_age = pd.to_numeric(val, errors="coerce")
                        except Exception:
                            pass

                # --- 2. EXTRACT SOIL PROPS ---
                avg_props = {"sand": np.nan, "clay": np.nan, "soc": np.nan, "bdod": np.nan, "cfvo": np.nan}
                raw_soil_props = {}
                theta_wp, theta_fc, theta_sat = np.nan, np.nan, np.nan

                if not soil_data.empty and location_type in soil_data["site_name"].values:
                    soil_row = soil_data[soil_data["site_name"] == location_type].iloc[0]
                    try:
                        avg_props = get_weighted_soil_props(soil_row)

                        raw_props = ["sand", "clay", "soc", "bdod", "cfvo"]
                        raw_depths = ["0_5", "5_15", "15_30", "30_60", "60_100"]

                        for p in raw_props:
                            for d in raw_depths:
                                col_key = f"{p}_{d}"
                                val = soil_row.get(col_key, np.nan)
                                raw_soil_props[f"soil_{col_key}"] = val

                        if not pd.isna(avg_props["sand"]) and not pd.isna(avg_props["clay"]):
                            om_pct = (avg_props["soc"] / 10.0) * 1.72
                            theta_wp, theta_fc, theta_sat = calculate_soil_hydraulics(
                                sand=avg_props["sand"] / 100.0,
                                clay=avg_props["clay"] / 100.0,
                                organic_matter=om_pct,
                                coarse_fragments_vol_percent=avg_props["cfvo"],
                            )
                    except Exception as soil_e:
                        print(f"Error calculating soil hydraulics for {location_type}: {soil_e}")

                # --- 3. AGGREGATE PLANT METADATA TO SITE LEVEL ---
                site_plant_meta = {}
                plant_md_file = paths.raw_csv_dir / f"{location_type}_plant_md.csv"
                if plant_md_file.exists():
                    try:
                        pmd = pd.read_csv(plant_md_file)
                        pmd = pmd.replace("NA", np.nan)

                        site_plant_meta["site_n_trees"] = len(pmd)

                        # DBH statistics
                        dbh = pd.to_numeric(pmd.get("pl_dbh"), errors="coerce").dropna()
                        site_plant_meta["site_mean_dbh"] = dbh.mean() if len(dbh) > 0 else np.nan
                        site_plant_meta["site_std_dbh"] = dbh.std() if len(dbh) > 1 else np.nan

                        # Species / genus diversity
                        species = pmd.get("pl_species", pd.Series(dtype=str)).dropna()
                        site_plant_meta["site_n_species"] = species.nunique() if len(species) > 0 else np.nan
                        genera = species.str.split().str[0]
                        site_plant_meta["site_n_genera"] = genera.nunique() if len(genera) > 0 else np.nan
                        if len(genera) > 0:
                            dominant_genus = genera.value_counts().index[0]
                            site_plant_meta["site_frac_dominant_genus"] = genera.value_counts().iloc[0] / len(genera)
                            site_plant_meta["site_dominant_genus"] = dominant_genus
                        else:
                            site_plant_meta["site_frac_dominant_genus"] = np.nan
                            site_plant_meta["site_dominant_genus"] = np.nan

                        # Dominant sensing method
                        sens = pmd.get("pl_sens_meth", pd.Series(dtype=str)).dropna()
                        if len(sens) > 0:
                            site_plant_meta["site_dominant_sens_meth"] = sens.value_counts().index[0]
                        else:
                            site_plant_meta["site_dominant_sens_meth"] = np.nan
                    except Exception as pmd_e:
                        print(f"  Warning: plant_md aggregation failed for {location_type}: {pmd_e}")

                site_biome_mapping[location_type] = biome_type
                all_pft_types.add(igbp_type)

            except Exception:
                continue

            # --- LOAD TIME SERIES ---
            env_data = pd.read_csv(env_data_file, parse_dates=["TIMESTAMP"])
            sap_data = pd.read_csv(sap_data_file, parse_dates=["TIMESTAMP"])
            sap_data.drop(columns=[col for col in sap_data.columns if "Unnamed" in col], inplace=True)

            env_data.set_index("TIMESTAMP", inplace=True)
            # solar_TIMESTAMP comes from sap side (canonical); drop env duplicate to avoid merge conflict
            env_data.drop(columns=["solar_TIMESTAMP"], errors="ignore", inplace=True)
            env_data = normalise_timestamps(env_data)

            # --- TREATMENT FILTER + FLO 2019 CALIBRATION (predictor-target alignment) ---
            # Order: filter FIRST (drop decoupled plants), calibrate SECOND (apply
            # ×1.6804 / ×1.6373 to surviving HD/CHD non-calibrated plants).
            # Both are no-ops when plant_md / stand_md are missing (ICOS pass-through).
            if apply_treatment_filter_flag or apply_flo2019_correction_flag:
                _plant_md_path = paths.raw_csv_dir / f"{location_type}_plant_md.csv"
                _stand_md_path = paths.raw_csv_dir / f"{location_type}_stand_md.csv"
                _plant_md = pd.read_csv(_plant_md_path).replace("NA", np.nan) if _plant_md_path.exists() else None
                _stand_md = pd.read_csv(_stand_md_path).replace("NA", np.nan) if _stand_md_path.exists() else None

                if apply_treatment_filter_flag:
                    sap_data, _tf_report = filter_by_treatment(sap_data, _plant_md, _stand_md, site_code=location_type)
                    for _pc in _tf_report["dropped"]:
                        correction_audit_records.append(
                            {
                                "site_name": location_type,
                                "pl_code": _pc,
                                "action": "dropped",
                                "reason": _tf_report["reason_by_plant"].get(_pc, ""),
                                "detail": "",
                                "timestamp_utc": _now_iso(),
                            }
                        )

                if apply_flo2019_correction_flag:
                    sap_data, _calib_records = apply_flo2019_correction(sap_data, _plant_md)
                    for _r in _calib_records:
                        correction_audit_records.append(
                            {
                                "site_name": location_type,
                                "pl_code": _r["pl_code"],
                                "action": "calibrated",
                                "reason": f"{_r['method']}_not_calibrated",
                                "detail": f"multiplier={_r['multiplier']:.4f}",
                                "timestamp_utc": _now_iso(),
                            }
                        )

            col_names = [
                col for col in sap_data.columns if col not in ["solar_TIMESTAMP", "TIMESTAMP", "TIMESTAMP_LOCAL"]
            ]

            # Empty-site guard: skip the site if no plant columns remain. Two
            # possible causes — distinguish them in the log so post-run debug
            # on Palma is unambiguous.
            if not col_names:
                cause = "TREATMENT FILTER" if apply_treatment_filter_flag else "EMPTY SOURCE"
                print(f"  [{cause}] {location_type}: no plant columns remain, skipping site")
                continue
            sites_retained += 1

            if plant_level:
                # --- PLANT-LEVEL: melt wide → long (one row per tree per timestamp) ---
                id_vars = ["TIMESTAMP"]
                if "solar_TIMESTAMP" in sap_data.columns:
                    id_vars.append("solar_TIMESTAMP")
                sap_long = sap_data.melt(
                    id_vars=id_vars,
                    value_vars=col_names,
                    var_name="pl_code",
                    value_name="sap_velocity",
                )
                # Drop rows where sap_velocity is NaN (tree not measured at this timestamp)
                sap_long = sap_long.dropna(subset=["sap_velocity"])

                # Join plant metadata from plant_md.csv
                plant_md_file = paths.raw_csv_dir / f"{location_type}_plant_md.csv"
                if plant_md_file.exists():
                    plant_md = pd.read_csv(plant_md_file)
                    # Keep only useful columns for driver analysis
                    md_cols = [
                        "pl_code",
                        "pl_dbh",
                        "pl_sens_meth",
                        "pl_species",
                    ]
                    md_cols = [c for c in md_cols if c in plant_md.columns]
                    sap_long = sap_long.merge(plant_md[md_cols], on="pl_code", how="left")

                plant_sap_data = sap_long.set_index("TIMESTAMP")
            else:
                # --- SITE-LEVEL: average across trees (original behaviour) ---
                sap_data["site"] = sap_data[col_names].mean(axis=1)
                sap_dict = {
                    "TIMESTAMP": sap_data["TIMESTAMP"],
                    "sap_velocity": sap_data["site"],
                    "TIMESTAMP_LOCAL": sap_data.get("TIMESTAMP_LOCAL"),
                }
                if "solar_TIMESTAMP" in sap_data.columns:
                    sap_dict["solar_TIMESTAMP"] = sap_data["solar_TIMESTAMP"]
                plant_sap_data = pd.DataFrame(sap_dict).set_index("TIMESTAMP")

            # --- PROCESS HOURLY DATA ---
            # Each side is collapsed to the hour on its OWN terms and only then
            # joined, so neither delivery's clock has to land on the other's. The
            # previous code joined on the exact instant BEFORE resampling, which
            # discarded every sap reading absent from the env index -- 10.05% of
            # the corpus (5,959,149 of 59,305,852 readings), and 100% at FRA_HES,
            # whose sap sits on :15/:45 against env on :00/:30.
            #
            # Inner, so an hour is kept only where BOTH sides observed -- the
            # behaviour the old exact-instant join was reaching for, expressed at
            # a granularity the two clocks can actually meet on.
            sap_hourly = aggregate_sap_to_hourly(round_instants(plant_sap_data), plant_level=plant_level)
            env_hourly = aggregate_env_to_hourly(env_data)
            df_hourly = sap_hourly.join(env_hourly, how="inner").reset_index()

            # --- ADD STATIC VARIABLES (Common function for DF population) ---
            def add_static_vars(target_df):
                target_df["biome"] = biome_type
                target_df["pft"] = igbp_type
                target_df["site_name"] = location_type
                target_df["latitude"] = lat
                target_df["longitude"] = lon
                target_df["elevation"] = ele
                target_df["slope"] = slope
                target_df["aspect_sin"] = aspect_sin
                target_df["aspect_cos"] = aspect_cos
                target_df["mean_annual_temp"] = mean_annual_temp
                target_df["mean_annual_precip"] = mean_annual_precip
                target_df["temp_seasonality"] = temp_seasonality
                target_df["precip_seasonality"] = precip_seasonality
                target_df["canopy_height"] = canopy_height
                for _fan2017_col, _fan2017_val in fan2017_features.items():
                    target_df[_fan2017_col] = _fan2017_val
                target_df["stand_age"] = stand_age
                target_df["soil_sand"] = avg_props["sand"]
                target_df["soil_clay"] = avg_props["clay"]
                target_df["soil_bdod"] = avg_props["bdod"]
                target_df["soil_soc"] = avg_props["soc"]
                target_df["soil_cfvo"] = avg_props["cfvo"]
                for k, v in raw_soil_props.items():
                    target_df[k] = v
                target_df["soil_theta_wp"] = theta_wp
                target_df["soil_theta_fc"] = theta_fc
                target_df["soil_theta_sat"] = theta_sat
                # Site-aggregated plant metadata
                for pm_key, pm_val in site_plant_meta.items():
                    target_df[pm_key] = pm_val
                return target_df

            df_hourly = add_static_vars(df_hourly)
            df_hourly = df_hourly.sort_values("TIMESTAMP")
            # 1. Ensure timezone consistency
            if df_hourly["TIMESTAMP"].dt.tz is None:
                df_hourly["TIMESTAMP"] = df_hourly["TIMESTAMP"].dt.tz_localize("UTC")
            # --- MERGE EXTERNAL DATA (LAI, ERA5) TO HOURLY ---
            # LAI: bounded nearest match.  merge_asof has no notion of a
            # series ENDING, so an unbounded "nearest" keeps matching the last
            # composite forever -- 271,917 h were carrying values up to 1812 d
            # old and the row still looked complete.  GlobMap V3 ends 2023-12-27.
            #
            # 8 d is the exact separator, not a chosen threshold.  Measured
            # 2026-08-17 over the 226-site corpus (5,389,733 site-hours): the
            # furthest any site-hour INTERIOR to the record sits from a composite
            # is 8.00 d, and everything strictly past 8.00 d is extrapolation
            # beyond a record's end.  GlobMap is 8-day from 2001 and half-month
            # (15-16 d) for 1991-2000, and "nearest" halves the gap.  So no
            # smaller bound keeps every real match, and no larger one excludes
            # extrapolation.
            #
            # The comparison must therefore be INCLUSIVE, and pandas 2.3.3 is
            # (verified: gap 8 d / tolerance 8 d matches; 8.01 d does not).
            # pandas is unpinned in requirements.txt, so if that ever flips, every
            # half-month-era match sitting at exactly 8.00 d is dropped silently.
            # The symptom to look for is pre-2001 LAI coverage collapsing -- it
            # should be 0 missing out of 288,072 pre-2001 site-hours.
            #
            # Result: 47,262 h without LAI (0.88%), none of them pre-2001.  A 16 d
            # bound would instead admit 960 h of 9-16 d-old values over
            # 2024-01-04..01-12 at the five sites that outrun the record
            # (AUT_MMG, CZE_LNZ, DEU_HAR, DEU_HOH, POL_MEZ, 192 h each).
            df_hourly = pd.merge_asof(df_hourly, lai_data, on="TIMESTAMP", by="site_name",
                                      direction="nearest", tolerance=pd.Timedelta(days=8))
            # Bounded: ERA5-Land and every merged hour sit on the hour, so a real match is exact, and the
            # bound only stops an hour outside the extract from borrowing the nearest one.
            df_hourly = pd.merge_asof(df_hourly, era5_data, on="TIMESTAMP", by="site_name",
                                      direction="nearest", tolerance=pd.Timedelta(minutes=30))

            # --- VOLUMETRIC SOIL WATER NORMALIZATION ---
            # Per layer: *_raw m3/m3, the bare column raw / sigma and *_zscore (raw - mu) / sigma, mu and
            # sigma from the climatology cell under the servable contract, so a 0.1 degree prediction
            # cell is normalised the same way, and from the site own merged hours under site. Applied
            # to the HOURLY values, so every daily statistic below follows from this one affine
            # transform, the daily sum included: it comes out as (sum(raw) - n*mu)/sigma with the true
            # n. The daily ERA5 pass below reuses the same moments.
            if contract == SERVABLE_CONTRACT:
                _cells = cells_by_site.get(location_type)
                if _cells is None:
                    raise FeatureContractError(
                        f"{location_type}: no climatology cell was resolved; its site metadata has no "
                        "usable si_lat/si_long, so the servable contract cannot normalise it"
                    )
                swc_moments = swc_climatology_moments(_cells)
            else:
                swc_moments = swc_site_moments(df_hourly)
            df_hourly = add_swc_representations(df_hourly, swc_moments, location_type)

            # --- CALCULATE DAY LENGTH ---
            # Day length in hours based on latitude, elevation, and date (astronomical calculation)
            # Note: Use UTC TIMESTAMP for pvlib calculations - it needs actual moment in time
            try:
                timestamps = pd.DatetimeIndex(df_hourly["TIMESTAMP"])
                if timestamps.tzinfo is None:
                    timestamps = timestamps.tz_localize("UTC")

                # Get unique dates for efficiency
                unique_dates = timestamps.normalize().unique()
                site_elevation = ele if not pd.isna(ele) else 0

                # Calculate day length for each unique date
                # Use sun_rise_set_transit_geometric (doesn't require ephem package)
                day_length_map = {}
                for date in unique_dates:
                    try:
                        # Create a time range for the day to find sunrise/sunset
                        times_of_day = pd.date_range(date, date + pd.Timedelta(hours=24), freq="1min", tz="UTC")
                        solar_pos = pvlib.solarposition.get_solarposition(
                            times_of_day, lat, lon, altitude=site_elevation
                        )

                        # Find when sun crosses horizon (elevation = 0)
                        elevations = solar_pos["elevation"].values
                        above_horizon = elevations > 0

                        if above_horizon.all():
                            # Polar day - sun never sets
                            day_length_hours = 24.0
                        elif not above_horizon.any():
                            # Polar night - sun never rises
                            day_length_hours = 0.0
                        else:
                            # Count minutes with sun above horizon
                            day_length_hours = above_horizon.sum() / 60.0

                        day_length_map[date] = day_length_hours
                    except Exception:
                        day_length_map[date] = np.nan

                # Map day length to each row based on date
                df_hourly["day_length"] = timestamps.normalize().map(day_length_map)
            except Exception as e:
                print(f"  Warning: Could not calculate day_length for {location_type}: {e}")
                df_hourly["day_length"] = np.nan

            # --- SAVE HOURLY ---
            df_hourly.to_csv(hourly_out_dir / f"{location_type}_hourly.csv", index=False)

            # --- PROCESS DAILY DATA (Aggregation from Hourly) ---

            # Both daylight modes need the same solar position; compute it once.
            # Use UTC TIMESTAMP for solar position calculation (pvlib needs actual moment in time)
            daytime_mask = None
            if daytime_only or daytime_sap_only:
                daytime_mask = calculate_daytime_mask(
                    df_hourly["TIMESTAMP"], lat=lat, lon=lon, elevation=ele if not pd.isna(ele) else 0
                )

            # Optional: Filter to daytime only before daily aggregation
            if daytime_only:
                n_before = len(df_hourly)
                df_hourly_for_daily = df_hourly[daytime_mask.values].copy()
                n_after = len(df_hourly_for_daily)
                print(
                    f"  {location_type}: Filtered {n_before - n_after}/{n_before} nighttime hours ({(n_before - n_after) / n_before * 100:.1f}%)"
                )
            else:
                df_hourly_for_daily = df_hourly

            # Define Static columns that should NOT be summed/min/maxed (just take first)
            static_prefixes = ["soil_", "biome", "pft", "site_name", "site_", "solar_TIMESTAMP"]
            static_exact = [
                "latitude",
                "longitude",
                "elevation",
                "slope",
                "aspect_sin",
                "aspect_cos",
                "mean_annual_temp",
                "mean_annual_precip",
                "temp_seasonality",
                "precip_seasonality",
                "canopy_height",
                "stand_age",
                "root_depth",
                "infiltration_depth",
                "deep_drainage",
                "drainage_frequency",
                "regolith_flush_rate",
                "gw_residence_time",
            ]
            if plant_level:
                # Tree metadata columns are static per tree — aggregate as "first"
                static_prefixes.append("pl_")
                static_exact.append("pl_code")

            daily_agg_rules = {}

            for col in df_hourly_for_daily.columns:
                # Skip TIMESTAMP - it's handled by resample's on= parameter
                if col == "TIMESTAMP":
                    continue

                # Check if column is static metadata
                is_static = (col in static_exact) or any(col.startswith(p) for p in static_prefixes)

                if is_static:
                    daily_agg_rules[col] = "first"
                elif pd.api.types.is_numeric_dtype(df_hourly_for_daily[col]):
                    # For all dynamic numeric variables (Env, Sap, ERA5, LAI), save all stats
                    daily_agg_rules[col] = ["mean", "min", "max", "sum"]
                else:
                    # Fallback for other non-numeric types
                    daily_agg_rules[col] = "first"

            # 2. Resample using UTC TIMESTAMP
            # This creates a MultiIndex column structure (e.g. ('vpd', 'mean'), ('vpd', 'sum'))
            df_hourly_for_daily = df_hourly_for_daily.set_index("TIMESTAMP")
            if plant_level and "pl_code" in df_hourly_for_daily.columns:
                # Group by tree, then resample each tree to daily
                agg_no_plcode = {k: v for k, v in daily_agg_rules.items() if k != "pl_code"}
                df_daily = df_hourly_for_daily.groupby("pl_code").resample("D").agg(agg_no_plcode)
                # pl_code is now in the index; move it back to a column
                df_daily = df_daily.reset_index(level="pl_code")
            else:
                df_daily = df_hourly_for_daily.resample("D").agg(daily_agg_rules)

            # 3. Flatten MultiIndex columns
            new_cols = []
            for col_info in df_daily.columns:
                if isinstance(col_info, tuple) and len(col_info) == 2:
                    col_name, agg_method = col_info
                    if not agg_method or agg_method == "first":
                        new_cols.append(col_name)
                    elif agg_method == "mean":
                        new_cols.append(f"{col_name}")
                    else:
                        new_cols.append(f"{col_name}_{agg_method}")
                else:
                    # Regular column (e.g. pl_code after groupby reset_index)
                    new_cols.append(col_info)

            df_daily.columns = new_cols
            df_daily = df_daily.reset_index()

            # --- DAILY STATISTICS FROM EACH SOURCE'S OWN HOURS ---
            # The aggregation above ran over the hours in which sap AND env observed. ERA5-Land and the rain
            # gauge are aggregated again over their own records, so neither loses part of a day to the other
            # side's gaps: see era5_daily_for_site and gauge_daily_totals. The gauge's bare ``precip`` is
            # therefore the daily TOTAL in mm/day, no longer the mean of its hourly values.
            # --daytime-only keeps its daylight scope.
            keep_hours = daylight_hours(lat, lon, ele) if daytime_only else None
            era5_site = era5_data[era5_data["site_name"] == location_type]
            df_daily = overwrite_daily_columns(
                df_daily, era5_daily_for_site(era5_site, swc_moments, location_type, keep_hours)
            )
            for _total_col in [c for c in ENV_SUM_COLS if c in env_data.columns]:
                _gauge = gauge_daily_totals(env_data, _total_col, keep_hours)
                if contract == SERVABLE_CONTRACT:
                    # The gauge keeps its own daily totals; the servable column belongs to ERA5-Land.
                    _gauge = rename_gauge_columns(_gauge, _total_col)
                df_daily = overwrite_daily_columns(df_daily, _gauge)
            if contract == SERVABLE_CONTRACT:
                df_daily = apply_servable_precip(df_daily)

            # --- DAYTIME-SAP-ONLY: rescope the target, leave env at 24 h ---
            # Placed BEFORE the growing-season filter so that filter reads the
            # same sap series the model will train on.
            if daytime_sap_only:
                n_all = len(df_hourly)
                n_lit = int(np.asarray(daytime_mask, dtype=bool).sum())
                df_daily = apply_daytime_sap_scope(df_daily, df_hourly, daytime_mask, plant_level=plant_level)
                print(
                    f"  {location_type}: sap scoped to {n_lit}/{n_all} daylight hours "
                    f"({n_lit / max(n_all, 1) * 100:.1f}%); env kept at 24 h"
                )

            # 4. Derived Calculations (using the new suffixed column names)
            # Use 'sum' for precipitation related ratio, or adjust to 'mean' if preferred.
            # Here we use sum to stay consistent with typical daily totals.
            if (
                "total_precipitation_hourly_sum" in df_daily.columns
                and "potential_evaporation_hourly_sum" in df_daily.columns
            ):
                pet = np.abs(df_daily["potential_evaporation_hourly_sum"].values)
                precip = df_daily["total_precipitation_hourly_sum"].values
                # Avoid division by zero
                pet = np.where(pet < 1e-10, 1e-10, pet)
                precip_PET = np.clip(precip / pet, 0, 10)
                df_daily["prcip/PET"] = precip_PET

            # --- GROWING SEASON FILTER (daily level) ---
            if growing_season_only:
                if plant_level and "pl_code" in df_daily.columns:
                    # Filter per-tree to avoid mixing sap signals across trees
                    filtered_parts = []
                    for pl_code, grp in df_daily.groupby("pl_code"):
                        filt, st = filter_growing_season_by_sapflow(grp)
                        filtered_parts.append(filt)
                        st["site"] = f"{location_type}_{pl_code}"
                        gs_stats_list.append(st)
                        print(
                            f"  GS filter {location_type}/{pl_code}: kept {st['pct_retained']:.1f}% "
                            f"({'evergreen' if st['is_evergreen'] else 'seasonal'})"
                        )
                    df_daily = pd.concat(filtered_parts, ignore_index=True)
                else:
                    df_daily, gs_stats = filter_growing_season_by_sapflow(df_daily)
                    gs_stats["site"] = location_type
                    gs_stats_list.append(gs_stats)
                    print(
                        f"  GS filter {location_type}: kept {gs_stats['pct_retained']:.1f}% "
                        f"({'evergreen' if gs_stats['is_evergreen'] else 'seasonal'})"
                    )

            # --- SAVE DAILY ---
            df_daily.to_csv(daily_out_dir / f"{location_type}_daily.csv", index=False)

            # --- COLLECT FOR BIOME MERGING ---
            # Note: hourly output retains full-year data even with --growing-season-only.
            # Only daily output (the model-training input) is filtered.
            if biome_type not in biome_merged_hourly:
                biome_merged_hourly[biome_type] = []
                biome_merged_daily[biome_type] = []

            biome_merged_hourly[biome_type].append(df_hourly)
            biome_merged_daily[biome_type].append(df_daily)

            if contract == SERVABLE_CONTRACT:
                contract_cell_rows.extend(cell_rows(location_type, lat, lon, cells_by_site[location_type]))

            count += 1
            if count % 10 == 0:
                print(f"Processed {count} sites...")

        except FeatureContractError:
            # Not a messy site: a contract failure ends the run rather than the site, so the tree is
            # never quietly short of what it claims to contain.
            raise
        except Exception as e:
            print(f"Error processing {sap_data_file}: {str(e)}")
            continue

    print(summarise(pft_sources))

    if contract == SERVABLE_CONTRACT:
        _block = contract_block(
            clim, SWC_LAYERS, DEFAULT_MAX_RADIUS_CELLS, contract_cell_rows, _git_short_sha()
        )
        _sidecar, _cells_file = write_sidecar(output_base_dir, _block, contract_cell_rows)
        print(f"Feature contract: {len(contract_cell_rows)} site-layers -> {_sidecar.name}, {_cells_file.name}")

    # --- FINAL MERGING (HOURLY & DAILY) ---
    def save_biome_collections(collection_dict, output_dir, file_suffix, skip_all_biomes_merge=False):
        biome_dir = output_dir / "by_biome"
        biome_dir.mkdir(exist_ok=True)

        last_biome_df = None
        for biome, dfs in collection_dict.items():
            if dfs:
                biome_df = pd.concat(dfs)
                safe_biome_name = "".join(c if c.isalnum() or c in [" ", "_", "-"] else "_" for c in biome).replace(
                    " ", "_"
                )
                biome_df.to_csv(biome_dir / f"{safe_biome_name}_{file_suffix}.csv", index=False)
                last_biome_df = biome_df
                # Free memory after saving each biome
                del biome_df

        # For hourly data, skip all_biomes merge to avoid memory issues
        if skip_all_biomes_merge:
            print(f"  Skipping all_biomes merge for {file_suffix} to save memory")
            return last_biome_df

        # For daily data, merge all biomes (smaller files)
        all_biomes_path = output_dir / f"all_biomes_{file_suffix}.csv"
        print(f"  Merging all biomes into {all_biomes_path.name}...")

        # Write header from first file, then append others
        biome_files = list(biome_dir.glob(f"*_{file_suffix}.csv"))
        if biome_files:
            first_file = True
            for bf in biome_files:
                chunk_iter = pd.read_csv(bf, chunksize=50000)
                for chunk in chunk_iter:
                    chunk.to_csv(all_biomes_path, mode="a" if not first_file else "w", header=first_file, index=False)
                    first_file = False
            return pd.read_csv(all_biomes_path, nrows=1)  # Return just header for reference
        return None

    if biome_merged_hourly:
        print("Saving merged hourly data...")
        save_biome_collections(biome_merged_hourly, hourly_out_dir, "merged_hourly", skip_all_biomes_merge=True)

        print("Saving merged daily data...")
        daily_all = save_biome_collections(
            biome_merged_daily, daily_out_dir, "merged_daily", skip_all_biomes_merge=False
        )

        # Save mapping
        pd.DataFrame({"site": list(site_biome_mapping.keys()), "biome": list(site_biome_mapping.values())}).to_csv(
            output_base_dir / "site_biome_mapping.csv", index=False
        )

        # Save growing season filter statistics
        if growing_season_only and gs_stats_list:
            gs_df = pd.DataFrame(gs_stats_list)
            gs_df.to_csv(output_base_dir / "growing_season_stats.csv", index=False)
            print(f"\nGrowing season stats saved ({len(gs_df)} sites)")
            n_evergreen = gs_df["is_evergreen"].sum()
            print(f"  Evergreen/low-variation: {n_evergreen}, Seasonal: {len(gs_df) - n_evergreen}")
            print(f"  Mean retention: {gs_df['pct_retained'].mean():.1f}%")

        # --- AUDIT CSV + MANIFEST.json (calibration + treatment filter actions) ---
        # Mirrors removal_log_report.csv convention. MANIFEST records the flags
        # and multipliers used so downstream training can detect drift.
        if apply_flo2019_correction_flag or apply_treatment_filter_flag:
            import json

            audit_dir = output_base_dir / "_audit"
            audit_dir.mkdir(exist_ok=True)
            audit_path = audit_dir / "correction_audit.csv"
            pd.DataFrame(
                correction_audit_records,
                columns=["site_name", "pl_code", "action", "reason", "detail", "timestamp_utc"],
            ).to_csv(audit_path, index=False)
            print(f"\nCorrection audit saved: {audit_path} ({len(correction_audit_records)} rows)")

            n_dropped = sum(1 for r in correction_audit_records if r["action"] == "dropped")
            n_calibrated = sum(1 for r in correction_audit_records if r["action"] == "calibrated")

            manifest = {
                "build_timestamp_utc": _now_iso(),
                "git_sha": _git_short_sha(),
                "flags": {
                    "apply_flo2019_correction": apply_flo2019_correction_flag,
                    "apply_treatment_filter": apply_treatment_filter_flag,
                    "daytime_only": daytime_only,
                    "daytime_sap_only": daytime_sap_only,
                    "plant_level": plant_level,
                    "growing_season_only": growing_season_only,
                },
                "calibration_multipliers": {
                    "HD": FLO2019_HD_MULTIPLIER,
                    "CHD": FLO2019_CHD_MULTIPLIER,
                },
                "n_sites_processed": count,
                "n_sites_retained": sites_retained,
                "n_actions_dropped": n_dropped,
                "n_actions_calibrated": n_calibrated,
            }
            manifest_path = output_base_dir / "MANIFEST.json"
            with open(manifest_path, "w", encoding="utf-8") as _f:
                json.dump(manifest, _f, indent=2)
            print(f"Manifest saved: {manifest_path}")

        print(f"Processing complete. Data saved to {output_base_dir}")
        return daily_all

    raise ValueError("No data was successfully processed")


def main():
    parser = argparse.ArgumentParser(description="Merge sap flow and environmental data")
    # The two daylight modes disagree about env, so at most one can be in effect.
    daylight = parser.add_mutually_exclusive_group()
    daylight.add_argument(
        "--daytime-only",
        action="store_true",
        help="Filter to daytime data only before daily aggregation (uses solar elevation > 0). "
        "Rescopes EVERY column, env variables included.",
    )
    daylight.add_argument(
        "--daytime-sap-only",
        action="store_true",
        help="Scope only the sap target to daylight; env variables keep their 24-hour "
        "daily statistics, matching the CDS derived-era5-land-daily-statistics product "
        "the global prediction is served from. Mutually exclusive with --daytime-only.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory (default: paths.merged_data_root)"
    )
    parser.add_argument(
        "--plant-level",
        action="store_true",
        help="Output one row per tree per timestamp (long format) instead of site-averaged. "
        "Joins plant_md.csv to add tree metadata (pl_dbh, pl_species, pl_sens_meth, etc.).",
    )
    parser.add_argument(
        "--growing-season-only",
        action="store_true",
        help="Apply sap-flow-based growing season filter after daily aggregation. "
        "Removes dormant-period rows using amplitude gate + 20%% of q95 threshold.",
    )
    parser.add_argument(
        "--apply-flo2019-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Multiply non-calibrated HD/CHD plant SFD by Flo 2019 multipliers "
        "(HD ×1.6804, CHD ×1.6373) before melt/aggregation. Default: off.",
    )
    parser.add_argument(
        "--apply-treatment-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop plant columns whose pl_treatment / st_treatment indicates "
        "predictor-target decoupling (irrigation, drought, root trenching, "
        "elevated CO2, shade). Default: off.",
    )
    parser.add_argument(
        "--feature-contract",
        choices=("site", "servable"),
        default=SERVABLE_CONTRACT,
        help="servable (the default) normalises soil moisture against the ERA5-Land climatology "
        "and writes precip as the ERA5 daily total in mm/day, so the tree needs no further pass; "
        "the gauge keeps its own totals under precip_gauge*. site is the legacy behaviour.",
    )
    parser.add_argument(
        "--swvl-cube",
        type=str,
        default=str(DEFAULT_SWVL_CUBE),
        help="Soil-moisture climatology for the servable contract (default: the v2 1995-2019 cube).",
    )
    args = parser.parse_args()
    # Defines the base directory for output
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        output_base_dir = paths.merged_data_root

    # Auto-append growing_season/ subdirectory when filter is active
    if args.growing_season_only:
        output_base_dir = output_base_dir / "growing_season"

    merge_sap_env_data_site(
        output_base_dir,
        daytime_only=args.daytime_only,
        daytime_sap_only=args.daytime_sap_only,
        plant_level=args.plant_level,
        growing_season_only=args.growing_season_only,
        apply_flo2019_correction_flag=args.apply_flo2019_correction,
        apply_treatment_filter_flag=args.apply_treatment_filter,
        feature_contract=args.feature_contract,
        swvl_cube=args.swvl_cube,
    )


if __name__ == "__main__":
    main()
