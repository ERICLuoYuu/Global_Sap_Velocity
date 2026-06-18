"""End-to-end: fabricated hourly CSVs → loader → site-day table → decoupling.

Constructs data where the afternoon depression is driven by ERA5 VPD (via dewpoint),
with Tair and SM varying independently, then asserts the decoupling recovers VPD as the
dominant driver — the pipeline-level analog of the unit orthogonality test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.afternoon_depression.data_loader import load_site_day_table
from src.afternoon_depression.decoupling import decouple_all_sites
from src.afternoon_depression.diurnal_metrics import vpd_from_era5


def _write_site_csv(path, site: str, n_days: int, seed: int) -> None:
    rng = np.random.RandomState(seed)
    rows = []
    start = pd.Timestamp("2011-05-01")
    for d in range(n_days):
        date = (start + pd.Timedelta(days=d)).date()
        tair_c = rng.uniform(12.0, 24.0)  # frost-free, varies independently
        tdew_c = tair_c - rng.uniform(2.0, 16.0)  # wide dewpoint depression → drives VPD
        vpd = float(vpd_from_era5(tair_c, tdew_c))
        supp = min(0.85, 0.18 * vpd)  # afternoon suppression ∝ VPD
        sm_raw = rng.uniform(0.18, 0.36)  # SM independent of the depression
        for h in range(6, 18):
            sf = 10.0 if h < 12 else 10.0 * (1.0 - supp)
            rows.append(
                {
                    "site_name": site,
                    "solar_TIMESTAMP": f"{date} {h:02d}:00:00+00:00",
                    "sap_velocity": sf,
                    "temperature_2m": tair_c + 273.15,
                    "dewpoint_2m": tdew_c + 273.15,
                    "volumetric_soil_water_layer_1_raw": sm_raw,
                    "volumetric_soil_water_layer_2_raw": sm_raw,
                    "volumetric_soil_water_layer_3_raw": sm_raw,
                    "surface_solar_radiation_downwards_hourly": 1_800_000.0,
                    "LAI": 5.0,
                    "pft": "DBF",
                }
            )
    pd.DataFrame(rows).to_csv(path / f"{site}_hourly.csv", index=False)


@pytest.mark.integration
def test_pipeline_recovers_vpd_dominance(tmp_path) -> None:
    _write_site_csv(tmp_path, "SITE_A", n_days=160, seed=1)
    _write_site_csv(tmp_path, "SITE_B", n_days=160, seed=2)

    table = load_site_day_table(tmp_path, climate_source="era5", tair_min=5.0, min_daily_sf=0.0)
    # one record per site-day, drivers present, depression positive (AM > PM)
    assert len(table) == pytest.approx(320, abs=5)
    assert {"delta_sf", "vpd", "tair", "sm"}.issubset(table.columns)
    assert table["delta_sf"].median() > 0

    effects = decouple_all_sites(table, min_valid_days=50, n_bins=10)
    assert len(effects) == 2  # both sites qualify
    for _, row in effects.iterrows():
        assert np.isfinite(row["vpd_given_tair"])
        assert row["vpd_given_tair"] > abs(row["tair_given_vpd"])  # VPD dominates Tair
        assert row["vpd_given_tair"] > abs(row["sm_given_vpd"])  # VPD dominates SM


@pytest.mark.integration
def test_restrict_to_qualifying_drops_short_record_sites(tmp_path) -> None:
    # Two dense sites (≥120 valid days) + one short site (40 days). --restrict-to-qualifying
    # must drop the short site from the analysed table (so Fig1/RF/decoupling share one set).
    from src.afternoon_depression.run_afternoon_depression import build_parser, run

    _write_site_csv(tmp_path, "LONG_A", n_days=160, seed=1)
    _write_site_csv(tmp_path, "LONG_B", n_days=160, seed=2)
    _write_site_csv(tmp_path, "SHORT_C", n_days=40, seed=3)
    out = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "--data-dir",
            str(tmp_path),
            "--climate-source",
            "era5",
            "--min-valid-days",
            "120",
            "--restrict-to-qualifying",
            "--no-rf",
            "--output-dir",
            str(out),
        ]
    )
    run(args)
    saved = pd.read_csv(out / "site_day_table.csv")
    assert set(saved["site_name"]) == {"LONG_A", "LONG_B"}  # SHORT_C (40 days) excluded
