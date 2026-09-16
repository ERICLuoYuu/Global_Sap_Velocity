"""The FFS cache is built under a feature contract: checked against the tree, passed to feature
engineering for every site, and recorded in the cache so selection and training can assert it."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from src.forward_selection import data_loader, run_selection
from src.forward_selection.data_loader import load_and_cache_features, load_cache

#: The loader's base column list (local to load_and_cache_features, minus the target and ``pft``).
#: ADDITIONAL_FEATURES is emptied by the fixture, so a site file with these columns is kept. If the
#: loader's list changes, the build tests fail with "No valid site data", not silently.
BASE = [
    "sw_in", "ws", "precip", "ta", "ta_max", "ta_min", "vpd", "vpd_max", "vpd_min", "ext_rad",
    "ppfd_in", "canopy_height", "elevation", "LAI", "prcip/PET", "soil_temperature_level_1", "day_length",
]  # fmt: skip
TIME = ["Day sin", "Day cos", "Week sin", "Week cos", "Month sin", "Month cos", "Year sin", "Year cos"]
N_SITES, N_DAYS = 2, 6


def _tree(tmp_path, built_as=None):
    """A merged config directory with ``daily/`` site files; ``built_as`` writes the contract provenance."""
    config_dir = tmp_path / "fullyear_fullday_tx"
    daily = config_dir / "daily"
    daily.mkdir(parents=True)
    if built_as is not None:
        (config_dir / "RUN_PROVENANCE.json").write_text(json.dumps({"feature_contract": {"name": built_as}}))
    rng = np.random.RandomState(0)
    days = pd.date_range("2020-06-01", periods=N_DAYS, freq="D")
    for i in range(N_SITES):
        frame = pd.DataFrame({name: rng.uniform(0.1, 1.0, N_DAYS) for name in BASE + ["sap_velocity"]})
        frame.insert(0, "TIMESTAMP", days)
        frame["solar_TIMESTAMP"] = days
        frame["site_name"] = "SITE_%d" % i
        frame["lat"] = 50.0 + i
        frame["lon"] = 8.0 + i
        frame["pft"] = "ENF"
        frame.to_csv(daily / ("SITE_%d_daily.csv" % i), index=False)
    return daily


class _Helpers:
    """Stand-ins for the training helpers; feature engineering records the contract it is given."""

    def __init__(self):
        self.contracts = []

    def feature_engineering(self, df, time_scale="daily", verbose=False, contract="site"):
        self.contracts.append(contract)
        return df, []

    @staticmethod
    def add_time_features(df, datetime_column=None):
        return df.assign(**{name: 0.5 for name in TIME})

    @staticmethod
    def create_spatial_groups(lat, lon, method="default", lat_grid_size=0.05, lon_grid_size=0.05, **kwargs):
        return np.arange(len(lat)), {}


@pytest.fixture
def helpers(monkeypatch):
    fake = _Helpers()
    monkeypatch.setattr(
        data_loader,
        "_import_training_helpers",
        lambda: (fake.feature_engineering, fake.add_time_features, fake.create_spatial_groups),
    )
    monkeypatch.setattr(data_loader, "ADDITIONAL_FEATURES", [])
    return fake


def _must_not_run():
    raise AssertionError("the contract check must come before the training helpers are imported")


class TestTheTreeMustMatchTheContract:
    def test_servable_refuses_a_legacy_tree_before_loading_anything(self, tmp_path, monkeypatch):
        daily = _tree(tmp_path)
        monkeypatch.setattr(data_loader, "_import_training_helpers", _must_not_run)
        with pytest.raises(ValueError, match="servable"):
            load_and_cache_features(daily, tmp_path / "cache.npz", contract="servable")
        assert not (tmp_path / "cache.npz").exists()

    def test_site_refuses_a_servable_tree(self, tmp_path, monkeypatch):
        daily = _tree(tmp_path, built_as="servable")
        monkeypatch.setattr(data_loader, "_import_training_helpers", _must_not_run)
        with pytest.raises(ValueError, match="built as 'servable'"):
            load_and_cache_features(daily, tmp_path / "cache.npz", contract="site")

    def test_an_unknown_contract_is_refused(self, tmp_path, monkeypatch):
        daily = _tree(tmp_path)
        monkeypatch.setattr(data_loader, "_import_training_helpers", _must_not_run)
        with pytest.raises(ValueError, match="unknown feature contract"):
            load_and_cache_features(daily, tmp_path / "cache.npz", contract="grid")


class TestTheContractReachesFeatureEngineering:
    @pytest.mark.parametrize("contract,built_as", [("site", None), ("servable", "servable")])
    def test_every_site_is_engineered_under_the_contract(self, tmp_path, helpers, contract, built_as):
        daily = _tree(tmp_path, built_as=built_as)
        load_and_cache_features(daily, tmp_path / "cache.npz", contract=contract)
        assert helpers.contracts == [contract] * N_SITES

    def test_the_default_is_site(self, tmp_path, helpers):
        daily = _tree(tmp_path)
        load_and_cache_features(daily, tmp_path / "cache.npz")
        assert helpers.contracts == ["site"] * N_SITES


class TestTheCacheRecordsTheContract:
    def test_a_servable_cache_round_trips_its_contract(self, tmp_path, helpers):
        daily = _tree(tmp_path, built_as="servable")
        built = load_and_cache_features(daily, tmp_path / "cache.npz", contract="servable")
        loaded = load_cache(tmp_path / "cache.npz")
        assert built["feature_contract"] == "servable"
        assert loaded["feature_contract"] == "servable"
        # POSITIVE CONTROL: the build really assembled every site's rows.
        assert loaded["X"].shape[0] == N_SITES * N_DAYS

    def test_a_cache_without_the_key_loads_as_unrecorded(self, tmp_path):
        path = tmp_path / "old_cache.npz"
        np.savez_compressed(
            path,
            X=np.zeros((2, 1)),
            y=np.zeros(2),
            groups=np.zeros(2, dtype=int),
            pfts_encoded=np.zeros(2, dtype=int),
            feature_names=np.array(["vpd"]),
            pft_categories=np.array(["ENF"]),
        )
        assert load_cache(path)["feature_contract"] is None


class TestRunSelectionFlag:
    @staticmethod
    def _parse(monkeypatch, *argv):
        monkeypatch.setattr(sys, "argv", ["run_selection", *argv])
        return run_selection._parse_args()

    def test_the_default_is_servable(self, monkeypatch):
        """Matches the merge, which builds servable unless told otherwise."""
        assert self._parse(monkeypatch, "--build_cache").feature_contract == "servable"

    def test_site_is_still_accepted(self, monkeypatch):
        assert (
            self._parse(monkeypatch, "--build_cache", "--feature_contract", "site").feature_contract
            == "site"
        )

    def test_servable_is_accepted(self, monkeypatch):
        assert self._parse(monkeypatch, "--build_cache", "--feature_contract", "servable").feature_contract == "servable"

    def test_an_unknown_contract_is_rejected_by_the_parser(self, monkeypatch):
        with pytest.raises(SystemExit):
            self._parse(monkeypatch, "--build_cache", "--feature_contract", "grid")

    def test_build_cache_forwards_the_contract(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(data_loader, "load_and_cache_features", lambda **kwargs: seen.update(kwargs))
        args = self._parse(
            monkeypatch,
            "--build_cache",
            "--data_dir", str(tmp_path),
            "--output_dir", str(tmp_path / "out"),
            "--feature_contract", "servable",
        )  # fmt: skip
        run_selection._cmd_build_cache(args)
        assert seen["contract"] == "servable"
        assert seen["data_dir"] == tmp_path / "daily"
