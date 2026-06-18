"""Unit tests for diurnal_metrics — ΔSF, centroid, SM depth-weighting, ERA5 VPD, solar hour.

Correspondence to Liu et al. (2024) SI Text S1 (EC pipeline) and Eq. 1–2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.afternoon_depression.diurnal_metrics import (
    aggregate_monthly,
    build_site_day_table,
    centroid,
    delta_sf,
    depth_weight_sm,
    solar_decimal_hour,
    vpd_from_era5,
)


# ── ΔSF = (SF_AM − SF_PM)/SF_AM × 100 ───────────────────────────────────────
@pytest.mark.unit
def test_delta_sf_afternoon_depression_positive() -> None:
    # morning 10 > afternoon 6 → depression = 40%
    assert delta_sf(10.0, 6.0) == pytest.approx(40.0)


@pytest.mark.unit
def test_delta_sf_afternoon_enhancement_negative() -> None:
    assert delta_sf(6.0, 9.0) == pytest.approx(-50.0)


@pytest.mark.unit
def test_delta_sf_zero_or_negative_morning_is_nan() -> None:
    # SF_AM <= 0 would divide-by-(non-positive) and blow the metric up → NaN guard
    assert np.isnan(delta_sf(0.0, 5.0))
    assert np.isnan(delta_sf(-1.0, 5.0))


# ── Diurnal centroid C_SF = Σ(SF·h)/Σ(SF), h decimal solar hour 6–18 ────────
@pytest.mark.unit
def test_centroid_symmetric_cycle_is_noon() -> None:
    hours = np.arange(6, 19, dtype=float)
    # symmetric triangle peaking at 12
    vals = 6.0 - np.abs(hours - 12.0)
    assert centroid(hours, vals) == pytest.approx(12.0, abs=1e-9)


@pytest.mark.unit
def test_centroid_morning_shifted_below_noon() -> None:
    hours = np.array([8.0, 10.0, 12.0, 14.0, 16.0])
    vals = np.array([10.0, 8.0, 4.0, 2.0, 1.0])  # weight toward morning
    assert centroid(hours, vals) < 12.0


@pytest.mark.unit
def test_centroid_afternoon_shifted_above_noon() -> None:
    hours = np.array([8.0, 10.0, 12.0, 14.0, 16.0])
    vals = np.array([1.0, 2.0, 4.0, 8.0, 10.0])
    assert centroid(hours, vals) > 12.0


@pytest.mark.unit
def test_centroid_nonpositive_sum_is_nan() -> None:
    assert np.isnan(centroid(np.array([6.0, 12.0]), np.array([0.0, 0.0])))


# ── Depth-weighted 0–100 cm SM from raw layers 1–3 (7, 21, 72 cm) ───────────
@pytest.mark.unit
def test_depth_weight_sm_constant_profile_returns_same_value() -> None:
    # uniform 0.30 m³/m³ through the profile → weighted mean 0.30
    assert depth_weight_sm(0.30, 0.30, 0.30) == pytest.approx(0.30)


@pytest.mark.unit
def test_depth_weight_sm_known_weights() -> None:
    # (7*0.1 + 21*0.2 + 72*0.3) / 100 = (0.7 + 4.2 + 21.6)/100 = 0.265
    assert depth_weight_sm(0.1, 0.2, 0.3) == pytest.approx(0.265)


@pytest.mark.unit
def test_depth_weight_sm_vectorized() -> None:
    out = depth_weight_sm(pd.Series([0.30, 0.10]), pd.Series([0.30, 0.20]), pd.Series([0.30, 0.30]))
    np.testing.assert_allclose(np.asarray(out, dtype=float), [0.30, 0.265])


# ── VPD from ERA5 Tair + Tdew (Hersbach 2020 / Tetens), kPa ─────────────────
@pytest.mark.unit
def test_vpd_zero_when_saturated() -> None:
    # Tdew == Tair → air saturated → VPD = 0
    assert vpd_from_era5(20.0, 20.0) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_vpd_positive_and_increases_with_dryness() -> None:
    moist = vpd_from_era5(25.0, 20.0)
    dry = vpd_from_era5(25.0, 5.0)
    assert 0.0 < moist < dry


@pytest.mark.unit
def test_vpd_known_value_kpa() -> None:
    # es(25)=0.6108*exp(17.27*25/(25+237.3))=3.168 kPa; es(15)=1.705 → VPD≈1.463
    assert vpd_from_era5(25.0, 15.0) == pytest.approx(1.463, abs=2e-2)


# ── Solar decimal hour from cosmetic-tz solar_TIMESTAMP ─────────────────────
@pytest.mark.unit
def test_solar_decimal_hour_uses_clock_reading() -> None:
    # +00:00 label is cosmetic; the clock reading IS local solar time
    s = pd.Series(["2009-11-18 22:24:18+00:00", "2009-11-19 06:30:00+00:00"])
    out = solar_decimal_hour(s)
    np.testing.assert_allclose(out.to_numpy(), [22.405, 6.5], atol=1e-2)


# ── Per-site-day table builder (EC "two daily values" → ΔSF per day) ────────
def _two_day_hourly() -> pd.DataFrame:
    """One site, two solar-days, hours 6..17. Day1 morning-heavy, Day2 flat."""
    rows = []
    for day, ampm in [("2010-06-01", "am_heavy"), ("2010-06-02", "flat")]:
        for h in range(6, 18):
            if ampm == "am_heavy":
                sf = 10.0 if h < 12 else 5.0  # AM 10, PM 5 → ΔSF = 50%
            else:
                sf = 8.0  # AM 8, PM 8 → ΔSF = 0
            rows.append(
                {
                    "site_name": "S1",
                    "solar_date": pd.Timestamp(day).date(),
                    "solar_hour": float(h),
                    "sap_velocity": sf,
                    "vpd": 1.0 + h * 0.1,
                    "tair": 20.0,
                    "sm": 0.3,
                    "pft": "DBF",
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_build_site_day_table_computes_delta_and_centroid() -> None:
    table = build_site_day_table(
        _two_day_hourly(),
        sf_col="sap_velocity",
        hour_col="solar_hour",
        date_col="solar_date",
        site_col="site_name",
        driver_cols=("vpd", "tair", "sm"),
        carry_cols=("pft",),
        min_window_hours=2,
    )
    assert len(table) == 2
    d1 = table[table["solar_date"] == pd.Timestamp("2010-06-01").date()].iloc[0]
    d2 = table[table["solar_date"] == pd.Timestamp("2010-06-02").date()].iloc[0]
    assert d1["delta_sf"] == pytest.approx(50.0)
    assert d2["delta_sf"] == pytest.approx(0.0)
    # daily driver means carried through
    assert d1["tair"] == pytest.approx(20.0)
    assert d1["pft"] == "DBF"


# ── Monthly aggregation (Liu Fig 1/Fig 2 pixel scale) ───────────────────────
def _site_day_rows() -> pd.DataFrame:
    """Two sites, spanning two calendar months, with known per-day metrics/drivers."""
    rows = []
    # S1: June (2 days) + July (1 day); S2: June (1 day) — distinct values to test means.
    spec = [
        ("S1", "2010-06-10", 40.0, 11.0, 1.0, 20.0, 0.30, "DBF"),
        ("S1", "2010-06-20", 60.0, 13.0, 2.0, 22.0, 0.20, "DBF"),
        ("S1", "2010-07-05", 50.0, 12.0, 3.0, 24.0, 0.10, "DBF"),
        ("S2", "2010-06-15", 10.0, 12.0, 0.5, 18.0, 0.40, "ENF"),
    ]
    for site, day, dsf, cen, vpd, tair, sm, pft in spec:
        rows.append(
            {
                "site_name": site,
                "solar_date": pd.Timestamp(day).date(),
                "delta_sf": dsf,
                "centroid": cen,
                "vpd": vpd,
                "tair": tair,
                "sm": sm,
                "pft": pft,
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_aggregate_monthly_means_per_site_month() -> None:
    monthly = aggregate_monthly(_site_day_rows(), carry_cols=("pft",))
    # 3 site-months: S1-2010-06, S1-2010-07, S2-2010-06
    assert len(monthly) == 3
    s1_jun = monthly[(monthly["site_name"] == "S1") & (monthly["month"] == "2010-06")].iloc[0]
    assert s1_jun["delta_sf"] == pytest.approx(50.0)  # mean(40, 60)
    assert s1_jun["vpd"] == pytest.approx(1.5)  # mean(1.0, 2.0)
    assert s1_jun["n_days"] == 2
    assert s1_jun["pft"] == "DBF"


@pytest.mark.unit
def test_aggregate_monthly_single_day_month_passes_through() -> None:
    monthly = aggregate_monthly(_site_day_rows(), carry_cols=("pft",))
    s1_jul = monthly[(monthly["site_name"] == "S1") & (monthly["month"] == "2010-07")].iloc[0]
    assert s1_jul["delta_sf"] == pytest.approx(50.0)
    assert s1_jul["n_days"] == 1


@pytest.mark.unit
def test_aggregate_monthly_preserves_schema_for_plotting() -> None:
    monthly = aggregate_monthly(_site_day_rows(), carry_cols=("pft",))
    for col in ("site_name", "delta_sf", "centroid", "vpd", "tair", "sm"):
        assert col in monthly.columns


@pytest.mark.unit
def test_aggregate_monthly_empty_input_returns_empty() -> None:
    out = aggregate_monthly(pd.DataFrame(columns=["site_name", "solar_date", "delta_sf"]))
    assert out.empty


@pytest.mark.unit
def test_build_site_day_table_drops_days_below_min_coverage() -> None:
    df = _two_day_hourly()
    # gut day 2 down to a single afternoon hour → insufficient AM/PM coverage
    df = df[~((df["solar_date"] == pd.Timestamp("2010-06-02").date()) & (df["solar_hour"] != 13.0))]
    table = build_site_day_table(
        df,
        sf_col="sap_velocity",
        hour_col="solar_hour",
        date_col="solar_date",
        site_col="site_name",
        driver_cols=("vpd", "tair", "sm"),
        carry_cols=("pft",),
        min_window_hours=2,
    )
    assert (table["solar_date"] == pd.Timestamp("2010-06-02").date()).sum() == 0
