"""Diurnal-asymmetry metrics for transpiration (Liu et al. 2024, GRL — EC method).

Pure, side-effect-free functions:
- ``delta_sf``      : ΔSF = (SF_AM − SF_PM)/SF_AM × 100  (Eq. 1; positive ⇒ afternoon depression)
- ``centroid``      : C_SF = Σ(SF·h)/Σ(SF)               (Eq. 2; <12 ⇒ morning-shifted peak)
- ``depth_weight_sm``: raw swvl layers 1–3 → 0–100 cm weighted mean (7/21/72 cm)
- ``vpd_from_era5`` : VPD (kPa) from ERA5 Tair + Tdew (Hersbach 2020 / Tetens)
- ``solar_decimal_hour``: decimal local-solar hour from ``solar_TIMESTAMP`` (cosmetic +00:00)
- ``build_site_day_table``: hourly rows → one (ΔSF, C_SF, daily drivers) record per site-day
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Union

import numpy as np
import pandas as pd

Numeric = Union[float, "pd.Series", "np.ndarray"]

# Tetens saturation-vapour-pressure constants (kPa, °C); consistent with the
# ERA5/IFS formulation cited by Liu et al. (Hersbach et al., 2020).
_ES_A = 0.6108
_ES_B = 17.27
_ES_C = 237.3

# ERA5-Land soil layer thicknesses (cm) for the 0–100 cm weighted mean:
# layer 1 = 0–7, layer 2 = 7–28, layer 3 = 28–100.
_LAYER_THICKNESS_CM = (7.0, 21.0, 72.0)


def _saturation_vapour_pressure(temp_c: Numeric) -> Numeric:
    """Saturation vapour pressure (kPa) at temperature ``temp_c`` (°C)."""
    return _ES_A * np.exp(_ES_B * temp_c / (temp_c + _ES_C))


def vpd_from_era5(tair_c: Numeric, tdew_c: Numeric) -> Numeric:
    """Vapour pressure deficit (kPa) from air and dewpoint temperature (°C).

    VPD = e_s(Tair) − e_s(Tdew); physically non-negative, so negatives (numerical
    noise when Tdew ≳ Tair) are clipped to 0.
    """
    vpd = _saturation_vapour_pressure(tair_c) - _saturation_vapour_pressure(tdew_c)
    return np.clip(vpd, 0.0, None)


def delta_sf(sf_am: float, sf_pm: float) -> float:
    """ΔSF = (SF_AM − SF_PM)/SF_AM × 100 %. NaN when SF_AM ≤ 0 (unstable ratio)."""
    if not (sf_am > 0):  # also catches NaN and negative morning means
        return float("nan")
    return (sf_am - sf_pm) / sf_am * 100.0


def centroid(hours: Sequence[float], values: Sequence[float]) -> float:
    """Diurnal centroid Σ(v·h)/Σ(v) over finite pairs; NaN when Σv ≤ 0."""
    h = np.asarray(hours, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(h) & np.isfinite(v)
    h, v = h[mask], v[mask]
    total = v.sum()
    if not (total > 0):
        return float("nan")
    return float((v * h).sum() / total)


def depth_weight_sm(
    layer1: Numeric,
    layer2: Numeric,
    layer3: Numeric,
    thicknesses: tuple[float, float, float] = _LAYER_THICKNESS_CM,
) -> Numeric:
    """Depth-weighted 0–100 cm soil moisture from raw swvl layers 1–3 (m³/m³)."""
    t1, t2, t3 = thicknesses
    return (t1 * layer1 + t2 * layer2 + t3 * layer3) / (t1 + t2 + t3)


def solar_decimal_hour(timestamps: pd.Series) -> pd.Series:
    """Decimal local-solar hour from ``solar_TIMESTAMP``.

    The stored ``+00:00`` offset is cosmetic — the clock reading itself is local
    solar time — so we read the wall-clock h:m:s directly.
    """
    ts = pd.to_datetime(timestamps)
    return ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0


def build_site_day_table(
    df: pd.DataFrame,
    *,
    sf_col: str,
    hour_col: str,
    date_col: str,
    site_col: str,
    driver_cols: Iterable[str] = ("vpd", "tair", "sm"),
    carry_cols: Iterable[str] = (),
    am_window: tuple[float, float] = (6.0, 12.0),
    pm_window: tuple[float, float] = (12.0, 18.0),
    min_window_hours: int = 2,
) -> pd.DataFrame:
    """Collapse daytime hourly rows into one record per site-day.

    Mirrors Liu SI Text S1: half-hourly → "two daily values, one each for the
    morning and afternoon", ΔSF per day. Days lacking ``min_window_hours`` valid
    observations in either the AM or PM window are dropped.
    """
    driver_cols = tuple(driver_cols)
    carry_cols = tuple(carry_cols)
    records: list[dict] = []
    for (site, date), g in df.groupby([site_col, date_col], sort=True):
        am = g.loc[(g[hour_col] >= am_window[0]) & (g[hour_col] < am_window[1]), sf_col].dropna()
        pm = g.loc[(g[hour_col] >= pm_window[0]) & (g[hour_col] < pm_window[1]), sf_col].dropna()
        if len(am) < min_window_hours or len(pm) < min_window_hours:
            continue
        day = g[(g[hour_col] >= am_window[0]) & (g[hour_col] < pm_window[1])]
        rec: dict = {
            site_col: site,
            date_col: date,
            "sf_am": float(am.mean()),
            "sf_pm": float(pm.mean()),
            "sf_day_mean": float(day[sf_col].mean()),
            "delta_sf": delta_sf(float(am.mean()), float(pm.mean())),
            "centroid": centroid(day[hour_col].to_numpy(), day[sf_col].to_numpy()),
            "n_hours": int(day[sf_col].notna().sum()),
        }
        valid = day.dropna(subset=[sf_col])
        rec["peak_hour"] = float(valid.loc[valid[sf_col].idxmax(), hour_col]) if len(valid) else float("nan")
        for col in driver_cols:
            if col in g.columns:
                rec[col] = float(pd.to_numeric(g[col], errors="coerce").mean())
        for col in carry_cols:
            if col in g.columns:
                nonnull = g[col].dropna()
                rec[col] = nonnull.iloc[0] if len(nonnull) else np.nan
        records.append(rec)
    return pd.DataFrame.from_records(records)


def aggregate_monthly(
    table: pd.DataFrame,
    *,
    site_col: str = "site_name",
    date_col: str = "solar_date",
    metric_cols: Iterable[str] = ("delta_sf", "centroid"),
    driver_cols: Iterable[str] = ("vpd", "tair", "sm", "rad", "lai"),
    carry_cols: Iterable[str] = ("pft", "biome", "aridity", "lat", "lon"),
) -> pd.DataFrame:
    """Collapse the per-site-day table into one record per site-calendar-month.

    Liu et al. (2024) bin at *monthly* pixel scale (Fig 1/Fig 2): the daily ΔGPP
    and the daily drivers are averaged within each calendar month before binning.
    We mirror that literally — each metric (``delta_sf``, ``centroid``) and driver
    (VPD/Tair/SM…) is the simple mean of that month's qualifying site-days, so the
    output carries the SAME column schema as the site-day table and feeds the
    existing figure functions unchanged.

    Monthly averaging is the mechanism behind the paper's clean response curves and
    sharply empty Fig 2 corners: it removes day-to-day weather scatter, lifting the
    within-site VPD–Tair correlation toward Liu's monthly ~0.95 (vs daily ~0.78).

    Returns columns ``[site_col, "month", <metrics>, <drivers>, "n_days", <carry>]``
    where ``month`` is the ``YYYY-MM`` period string and ``n_days`` counts the
    site-days pooled into that month. Carry columns (site-constant: PFT, aridity…)
    take the first non-null value per site. An empty input returns an empty frame.
    """
    if table.empty or date_col not in table.columns:
        return table.copy()
    d = table.copy()
    d["month"] = pd.to_datetime(d[date_col], errors="coerce").dt.to_period("M").astype(str)
    d = d[d["month"] != "NaT"]
    if d.empty:
        return pd.DataFrame(columns=[site_col, "month", *metric_cols, *driver_cols, "n_days"])
    mean_cols = [c for c in (*metric_cols, *driver_cols) if c in d.columns]
    grouped = d.groupby([site_col, "month"], sort=True)
    out = grouped[mean_cols].mean()
    out["n_days"] = grouped.size()
    out = out.reset_index()
    for col in carry_cols:
        if col in d.columns:
            # site-constant metadata → first non-null per site (GroupBy.first skips NaN)
            out[col] = out[site_col].map(d.groupby(site_col)[col].first())
    return out
