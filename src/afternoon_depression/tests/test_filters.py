"""Tests for the EC Text S1 filter chain and ERA5 driver standardisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.afternoon_depression.data_loader import (
    apply_day_filters,
    apply_morning_flow_filter,
    prepare_hourly,
    standardise_site_frame,
)


def _raw_hourly_one_day(date: str = "2010-06-01", tair_k: float = 290.0, tdew_k: float = 285.0) -> pd.DataFrame:
    rows = []
    for h in range(6, 18):
        rows.append(
            {
                "site_name": "S1",
                "solar_TIMESTAMP": f"{date} {h:02d}:00:00+00:00",
                "sap_velocity": 10.0 if h < 12 else 6.0,
                "temperature_2m": tair_k,
                "dewpoint_2m": tdew_k,
                "volumetric_soil_water_layer_1_raw": 0.1,
                "volumetric_soil_water_layer_2_raw": 0.2,
                "volumetric_soil_water_layer_3_raw": 0.3,
                "surface_solar_radiation_downwards_hourly": 1_800_000.0,
                "LAI": 5.0,
                "pft": "DBF",
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_standardise_converts_kelvin_and_depth_weights_sm() -> None:
    out = standardise_site_frame(_raw_hourly_one_day(), "era5", "S1.csv")
    assert out["tair"].iloc[0] == pytest.approx(290.0 - 273.15)
    assert out["sm"].iloc[0] == pytest.approx(0.265)  # (7*.1+21*.2+72*.3)/100
    assert (out["vpd"] >= 0).all() and out["vpd"].iloc[0] > 0  # tair > tdew → VPD > 0
    assert out["rad"].iloc[0] == pytest.approx(1_800_000.0 / 3600.0)


@pytest.mark.unit
def test_standardise_fails_fast_on_all_nan_dewpoint() -> None:
    df = _raw_hourly_one_day()
    df["dewpoint_2m"] = np.nan
    with pytest.raises(ValueError, match="climate-source site"):
        standardise_site_frame(df, "era5", "S1.csv")


@pytest.mark.unit
def test_standardise_site_source_uses_measured_vpd_ta() -> None:
    df = _raw_hourly_one_day()
    df["vpd"] = 1.5
    df["ta"] = 18.0
    out = standardise_site_frame(df, "site", "S1.csv")
    assert out["tair"].iloc[0] == pytest.approx(18.0)
    assert out["vpd"].iloc[0] == pytest.approx(1.5)


@pytest.mark.unit
def test_prepare_hourly_nans_negative_sf_and_restricts_window(tmp_path) -> None:
    df = _raw_hourly_one_day()
    # inject a pre-dawn hour (out of window) and a negative-flux hour
    extra = df.iloc[[0]].copy()
    extra["solar_TIMESTAMP"] = "2010-06-01 03:00:00+00:00"
    df = pd.concat([df, extra], ignore_index=True)
    df.loc[df["solar_TIMESTAMP"].str.contains("06:00"), "sap_velocity"] = -5.0
    df.to_csv(tmp_path / "S1_hourly.csv", index=False)

    out = prepare_hourly(tmp_path, "era5")
    assert (out["solar_hour"] >= 6).all() and (out["solar_hour"] < 18).all()  # 03:00 dropped
    assert not (out["sf"] < 0).any()  # negative flux → NaN
    assert out["sf"].isna().sum() == 1


@pytest.mark.unit
def test_apply_day_filters_frostfree_and_lowflux() -> None:
    table = pd.DataFrame(
        {
            "site_name": ["S1", "S1", "S1"],
            "tair": [20.0, 2.0, 20.0],  # row 2 is frosty (≤5 °C)
            "sf_day_mean": [5.0, 5.0, 0.1],  # row 3 is low-activity
            "delta_sf": [40.0, 40.0, 40.0],
        }
    )
    out = apply_day_filters(table, tair_min=5.0, min_daily_sf=0.5)
    assert len(out) == 1
    assert out.iloc[0]["tair"] == 20.0 and out.iloc[0]["sf_day_mean"] == 5.0


@pytest.mark.unit
def test_morning_flow_filter_drops_negligible_morning_and_bounds_delta_sf() -> None:
    # Liu SI Text S1 step-3 analog: a near-zero morning makes ΔSF=(AM-PM)/AM blow up.
    table = pd.DataFrame(
        {
            "site_name": ["S1", "S1", "S1"],
            "sf_am": [5.0, 0.001, 2.0],  # row 2: negligible morning vs strong afternoon
            "sf_pm": [4.0, 8.0, 6.0],
            "delta_sf": [20.0, -799_900.0, -200.0],
        }
    )
    out = apply_morning_flow_filter(table, min_am_pm_ratio=0.10)
    assert len(out) == 2  # row 2 dropped (0.001 < 0.10*8); rows 1,3 kept
    assert (out["sf_am"] >= 0.10 * out["sf_pm"]).all()
    # surviving ΔSF is bounded below by (1 - 1/ratio)*100 = -900 %
    assert out["delta_sf"].min() >= -900.0
    # immutability: input untouched
    assert len(table) == 3


@pytest.mark.unit
def test_morning_flow_filter_ratio_zero_disables() -> None:
    table = pd.DataFrame({"sf_am": [0.0001], "sf_pm": [8.0], "delta_sf": [-1e6]})
    out = apply_morning_flow_filter(table, min_am_pm_ratio=0.0)
    assert len(out) == 1  # disabled → nothing dropped
