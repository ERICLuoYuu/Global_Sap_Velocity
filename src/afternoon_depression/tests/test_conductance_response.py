"""Tests for the canopy-conductance (Gc) response path (--response gc).

The afternoon-depression pipeline is response-agnostic: Gc (Flo et al. 2021, Eqn 2)
flows through the SAME abstract response slot as sap velocity, so the only Gc-specific
logic is computing Gc per hour in the loader. These tests pin that wiring and the
end-to-end runner, and assert the sap-velocity (sf) default path is untouched.

The synthetic data varies VPD WITHIN the day (cool/humid morning, hot/dry afternoon)
so Gc∝1/VPD makes ΔGc genuinely differ from ΔSF — exercising the confound the verdict warns about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.afternoon_depression.data_loader import load_site_day_table, prepare_hourly
from src.afternoon_depression.diurnal_metrics import vpd_from_era5
from src.sm_vpd_decoupling.conductance import canopy_conductance


def _write_site_csv(path, site: str, n_days: int, seed: int, *, elevation: float | None = 1200.0) -> None:
    """One site of era5-schema hourly rows with a diurnal VPD swing + (optional) elevation."""
    rng = np.random.RandomState(seed)
    rows = []
    start = pd.Timestamp("2011-05-01")
    for d in range(n_days):
        date = (start + pd.Timedelta(days=d)).date()
        tdew_c = rng.uniform(4.0, 9.0)  # ~constant within day
        sm_raw = rng.uniform(0.18, 0.36)
        for h in range(6, 18):
            # diurnal warming → afternoon VPD higher than morning VPD
            tair_c = 12.0 + 0.9 * (h - 6) + rng.uniform(-0.5, 0.5)
            vpd = float(vpd_from_era5(tair_c, tdew_c))
            sf = 10.0 if h < 12 else 10.0 * (1.0 - min(0.6, 0.2 * vpd))  # afternoon sap suppression
            row = {
                "site_name": site,
                "solar_TIMESTAMP": f"{date} {h:02d}:00:00+00:00",
                "sap_velocity": sf,
                "ta": tair_c,
                "vpd": vpd,
                "temperature_2m": tair_c + 273.15,
                "dewpoint_2m": tdew_c + 273.15,
                "volumetric_soil_water_layer_1_raw": sm_raw,
                "volumetric_soil_water_layer_2_raw": sm_raw,
                "volumetric_soil_water_layer_3_raw": sm_raw,
                "surface_solar_radiation_downwards_hourly": 1_800_000.0,
                "LAI": 5.0,
                "pft": "DBF",
            }
            if elevation is not None:
                row["elevation"] = elevation
            rows.append(row)
    pd.DataFrame(rows).to_csv(path / f"{site}_hourly.csv", index=False)


@pytest.mark.unit
def test_prepare_hourly_adds_gc_column(tmp_path) -> None:
    _write_site_csv(tmp_path, "SITE_A", n_days=3, seed=1, elevation=1200.0)
    df = prepare_hourly(tmp_path, climate_source="era5", response="gc")
    assert "gc" in df.columns
    # gc equals Flo Eqn 2 recomputed on the same standardised inputs (wiring check).
    expected = canopy_conductance(df["sf"], df["tair"], df["vpd"], pd.Series(1200.0, index=df.index))
    valid = df["gc"].notna()
    assert valid.any()
    np.testing.assert_allclose(df.loc[valid, "gc"], expected[valid], rtol=1e-9)
    assert (df.loc[valid, "gc"] > 0).all()


@pytest.mark.unit
def test_negative_sap_yields_nan_gc(tmp_path) -> None:
    _write_site_csv(tmp_path, "SITE_A", n_days=2, seed=2)
    # Inject a reverse-flow row; the hourly EC filter NaNs negative sf BEFORE Gc is computed.
    csv = tmp_path / "SITE_A_hourly.csv"
    raw = pd.read_csv(csv)
    raw.loc[0, "sap_velocity"] = -5.0
    raw.to_csv(csv, index=False)
    df = prepare_hourly(tmp_path, climate_source="era5", response="gc").reset_index(drop=True)
    neg = df["sf"].isna()
    assert neg.any()
    assert df.loc[neg, "gc"].isna().all()  # negative sap → NaN sf → NaN Gc


@pytest.mark.unit
def test_missing_elevation_falls_back_to_sea_level(tmp_path) -> None:
    # No elevation column → h=0 m (exp(0)=1), so Gc matches altitude_m=0.
    _write_site_csv(tmp_path, "SITE_A", n_days=2, seed=3, elevation=None)
    df = prepare_hourly(tmp_path, climate_source="era5", response="gc")
    assert "elevation" not in df.columns
    expected = canopy_conductance(df["sf"], df["tair"], df["vpd"], pd.Series(0.0, index=df.index))
    valid = df["gc"].notna()
    np.testing.assert_allclose(df.loc[valid, "gc"], expected[valid], rtol=1e-9)


@pytest.mark.unit
def test_load_site_day_table_response_slot_holds_gc(tmp_path) -> None:
    _write_site_csv(tmp_path, "SITE_A", n_days=20, seed=4)
    sf_tab = load_site_day_table(tmp_path, climate_source="era5", response="sf", tair_min=5.0)
    gc_tab = load_site_day_table(tmp_path, climate_source="era5", response="gc", tair_min=5.0)
    # Same schema; the response slot (delta_sf / sf_am) now carries Gc-derived values.
    assert {"delta_sf", "sf_am", "sf_pm", "vpd", "tair", "sm"}.issubset(gc_tab.columns)
    assert gc_tab["delta_sf"].notna().any()
    # Diurnal VPD swing makes ΔGc differ from ΔSF (the 1/VPD confound), so the two
    # response slots are NOT identical even on the same sap-flow series.
    merged = sf_tab.merge(gc_tab, on=["site_name", "solar_date"], suffixes=("_sf", "_gc"))
    assert not np.allclose(merged["delta_sf_sf"], merged["delta_sf_gc"], rtol=1e-3)


@pytest.mark.unit
def test_plot_decoupling_lines_accepts_resp_label(tmp_path) -> None:
    from src.afternoon_depression.plotting import plot_decoupling_lines

    _write_site_csv(tmp_path, "SITE_A", n_days=60, seed=5)
    table = load_site_day_table(tmp_path, climate_source="era5", response="gc", tair_min=5.0)
    out = tmp_path / "fig.png"
    plot_decoupling_lines(table, out, n_bins=5, min_valid_days=10, resp_label="ΔGc")
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.integration
def test_runner_response_gc_writes_gc_subdir_and_caveat(tmp_path) -> None:
    from src.afternoon_depression.run_afternoon_depression import build_parser, run

    _write_site_csv(tmp_path, "SITE_A", n_days=160, seed=1)
    _write_site_csv(tmp_path, "SITE_B", n_days=160, seed=2)
    out = tmp_path / "gc_out"
    args = build_parser().parse_args(
        [
            "--data-dir",
            str(tmp_path),
            "--climate-source",
            "era5",
            "--response",
            "gc",
            "--min-valid-days",
            "50",
            "--no-rf",
            "--output-dir",
            str(out),
        ]
    )
    run(args)
    table = pd.read_csv(out / "site_day_table.csv")
    assert table["delta_sf"].notna().any()  # ΔGc lives in the response slot
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "canopy conductance (Gc)" in report
    assert "1/VPD" in report and "Oren" in report  # the confound caveat is surfaced
    assert "ΔGc" in report
    assert (out / "figures" / "fig2abc_decoupling_lines.png").exists()
