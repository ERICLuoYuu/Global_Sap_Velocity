"""A site without a rain gauge enters the servable FFS cache, with ``precip`` from ERA5-Land in mm/day.

Uses the real feature engineering: the replacement happens there, and the loader's skip of a site
missing any column happens after it.
"""

import json

import numpy as np
import pandas as pd

from src.forward_selection import data_loader
from src.forward_selection.data_loader import load_and_cache_features
from src.hyperparameter_optimization.feature_engineering import apply_all_feature_engineering

BASE = [
    "sw_in", "ws", "ta", "ta_max", "ta_min", "vpd", "vpd_max", "vpd_min", "ext_rad",
    "ppfd_in", "canopy_height", "elevation", "LAI", "prcip/PET", "soil_temperature_level_1", "day_length",
]  # fmt: skip
TP_MEAN, TP_SUM = "total_precipitation_hourly", "total_precipitation_hourly_sum"
TIME = ["Day sin", "Day cos", "Week sin", "Week cos", "Month sin", "Month cos", "Year sin", "Year cos"]
N_DAYS = 40  # the 30-day running anomalies need more than a week before a row is complete


def _tree(tmp_path, built_as):
    """Site 0 has no gauge; site 1 has one. Site i's ERA5 precipitation is a constant (i + 1) mm/day."""
    config_dir = tmp_path / "fullyear_fullday_tx"
    daily = config_dir / "daily"
    daily.mkdir(parents=True)
    if built_as is not None:
        (config_dir / "RUN_PROVENANCE.json").write_text(json.dumps({"feature_contract": {"name": built_as}}))
    rng = np.random.RandomState(0)
    days = pd.date_range("2020-06-01", periods=N_DAYS, freq="D")
    for i in range(2):
        frame = pd.DataFrame({name: rng.uniform(0.1, 1.0, N_DAYS) for name in BASE + ["sap_velocity"]})
        if i == 1:
            frame["precip"] = rng.uniform(0.1, 1.0, N_DAYS)
        frame[TP_MEAN] = 0.001 * (i + 1) / 24  # m/h -> (i + 1) mm/day
        frame[TP_SUM] = 0.0005  # a short day's undercounted sum: must not be read
        frame.insert(0, "TIMESTAMP", days)
        frame["solar_TIMESTAMP"] = days
        frame["site_name"] = "SITE_%d" % i
        frame["lat"], frame["lon"], frame["pft"] = 50.0 + i, 8.0 + i, "ENF"
        frame.to_csv(daily / ("SITE_%d_daily.csv" % i), index=False)
    return daily


def _helpers(monkeypatch):
    def add_time_features(df, datetime_column=None):
        return df.assign(**{name: 0.5 for name in TIME})

    def create_spatial_groups(lat, lon, method="default", lat_grid_size=0.05, lon_grid_size=0.05, **kwargs):
        return np.arange(len(lat)), {}

    monkeypatch.setattr(
        data_loader,
        "_import_training_helpers",
        lambda: (apply_all_feature_engineering, add_time_features, create_spatial_groups),
    )
    monkeypatch.setattr(data_loader, "ADDITIONAL_FEATURES", [])


def test_servable_keeps_the_gauge_less_site_with_era5_precip(tmp_path, monkeypatch):
    _helpers(monkeypatch)
    cache = load_and_cache_features(_tree(tmp_path, "servable"), tmp_path / "c.npz", contract="servable")
    assert len(set(cache["groups"].tolist())) == 2
    precip = cache["X"][:, cache["feature_names"].index("precip")]
    np.testing.assert_allclose(sorted(set(precip.tolist())), [1.0, 2.0], rtol=1e-6)


def test_site_still_skips_the_gauge_less_site(tmp_path, monkeypatch):
    # POSITIVE CONTROL: under the legacy contract the gauge is required, as before.
    _helpers(monkeypatch)
    cache = load_and_cache_features(_tree(tmp_path, None), tmp_path / "c.npz", contract="site")
    assert len(set(cache["groups"].tolist())) == 1
