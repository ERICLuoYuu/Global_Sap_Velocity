# src/sm_vpd_decoupling/tests/test_loader_table.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.loader import apply_day_filter, standardise_site_frame


def _raw_site_frame() -> pd.DataFrame:
    n = 10
    return pd.DataFrame(
        {
            "site_name": ["S1"] * n,
            "TIMESTAMP": pd.date_range("2010-06-01", periods=n, freq="D"),
            "sap_velocity": np.linspace(1.0, 10.0, n),
            "ta": np.linspace(10.0, 25.0, n),
            "vpd": np.linspace(0.2, 2.0, n),
            "ppfd_in": np.linspace(300.0, 900.0, n),
            "sw_in": np.linspace(100.0, 400.0, n),
            "surface_solar_radiation_downwards_hourly": np.linspace(100.0, 400.0, n),
            "volumetric_soil_water_layer_1": np.linspace(0.1, 0.3, n),
            "volumetric_soil_water_layer_2": np.linspace(0.1, 0.3, n),
            "volumetric_soil_water_layer_3": np.linspace(0.1, 0.3, n),
            "volumetric_soil_water_layer_4": np.linspace(0.1, 0.3, n),
            "temperature_2m": np.linspace(10.0, 25.0, n) + 273.15,
            "dewpoint_2m": np.linspace(5.0, 15.0, n) + 273.15,
            "elevation": [100.0] * n,
            "pft": ["ENF"] * n,
            "biome": ["temperate"] * n,
            "prcip/PET": [0.6] * n,
            "canopy_height": [20.0] * n,
            "latitude_x": [45.0] * n,
            "longitude_x": [7.0] * n,
        }
    )


def test_standardise_site_frame_schema():
    out = standardise_site_frame(_raw_site_frame(), climate_source="site")
    for col in [
        "site_name",
        "date",
        "E",
        "Gc",
        "vpd",
        "tair",
        "ppfd",
        "swvl1",
        "swvl2",
        "swvl3",
        "swvl4",
        "root_zone_sm",
        "pft",
        "biome",
        "aridity",
        "canopy_height",
        "elevation",
    ]:
        assert col in out.columns, col
    # negative sap velocity dropped -> here none negative, all finite E.
    assert out["E"].notna().all()


def test_standardise_drops_negative_sap_velocity():
    raw = _raw_site_frame()
    raw.loc[0, "sap_velocity"] = -5.0
    out = standardise_site_frame(raw, climate_source="site")
    assert np.isnan(out.loc[out.index[0], "E"])


def test_apply_day_filter_thresholds():
    out = standardise_site_frame(_raw_site_frame(), climate_source="site")
    filt15 = apply_day_filter(out, tair_min=15.0)
    filt5 = apply_day_filter(out, tair_min=5.0)
    # All retained rows satisfy the three thresholds.
    assert (filt15["tair"] > 15.0).all()
    assert (filt15["vpd"] > 0.5).all()
    assert (filt15["ppfd"] > 500.0).all()
    # Relaxing Tair keeps at least as many rows.
    assert len(filt5) >= len(filt15)
