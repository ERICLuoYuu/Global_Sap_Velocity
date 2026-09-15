"""The Stage-4 merge emits servable-contract features in one pass.

Soil moisture is normalised on the HOURLY frame and aggregated afterwards, so one affine transform
carries every daily statistic: the mean, ``_min`` and ``_max`` because it is monotone, and ``_sum``
because it distributes -- giving ``(sum(raw) - n*mu)/sigma`` with the true hour count, where the
retired rebuild had to recover ``n`` from ``raw_sum / raw_mean``. These tests compose the merge's own
helpers with the contract module exactly as the merge does, so no cube and no corpus is needed.
"""

import inspect

import numpy as np
import pandas as pd

from notebooks.merge_gap_filled_hourly_orginal import (
    _preflight_sites,
    add_swc_representations,
    era5_daily_for_site,
    swc_site_moments,
)
from notebooks.merge_gap_filled_hourly_orginal import merge_sap_env_data_site
from src.feature_contract.definitions import ERA5_HOURLY_MEAN_PRECIP_COL, SERVABLE_CONTRACT
from src.feature_contract.merge_contract import apply_servable_precip, swc_climatology_moments
from src.feature_contract.swvl_climatology import CellStats
from src.hyperparameter_optimization.feature_engineering import _servable_precip

SWC1 = "volumetric_soil_water_layer_1"
MU, SIGMA = 0.30, 0.05
DAY = pd.Timestamp("2015-06-01", tz="UTC")
HOURLY_PRECIP_M = 1e-4


def _cells(sigma=SIGMA, mean=MU):
    return {1: CellStats(1, sigma, mean, 50.9, 8.0, 0.0, False)}


def _era5_site(n_days=2, constant=None):
    """ERA5 hours for one site: soil moisture 0.30 and 0.31 on alternate hours unless pinned."""
    index = pd.date_range(DAY, periods=24 * n_days, freq="h")
    swc = np.full(len(index), constant) if constant is not None else MU + 0.01 * (index.hour.to_numpy() % 2)
    return pd.DataFrame(
        {
            "site_id": 7,
            "site_name": "S",
            "latitude": 50.9,
            "longitude": 8.0,
            "TIMESTAMP": index,
            ERA5_HOURLY_MEAN_PRECIP_COL: HOURLY_PRECIP_M,
            SWC1: swc,
        }
    )


def _daylight(stamps):
    return np.isin(pd.DatetimeIndex(stamps).hour, range(6, 18))


class TestSoilMoistureFamilies:
    """12 hours at 0.30 and 12 at 0.31: mean 0.305, min 0.30, max 0.31, sum 7.32."""

    def test_every_daily_statistic_is_the_climatology_normalisation(self):
        out = era5_daily_for_site(_era5_site(), swc_climatology_moments(_cells()), "S")

        np.testing.assert_allclose(out[SWC1 + "_raw"].to_numpy(), [0.305, 0.305])
        np.testing.assert_allclose(out[SWC1 + "_raw_min"].to_numpy(), [0.30, 0.30])
        np.testing.assert_allclose(out[SWC1 + "_raw_max"].to_numpy(), [0.31, 0.31])
        np.testing.assert_allclose(out[SWC1 + "_raw_sum"].to_numpy(), [7.32, 7.32])

        np.testing.assert_allclose(out[SWC1].to_numpy(), [0.305 / SIGMA] * 2)
        np.testing.assert_allclose(out[SWC1 + "_min"].to_numpy(), [0.30 / SIGMA] * 2)
        np.testing.assert_allclose(out[SWC1 + "_max"].to_numpy(), [0.31 / SIGMA] * 2)
        np.testing.assert_allclose(out[SWC1 + "_sum"].to_numpy(), [7.32 / SIGMA] * 2)

        np.testing.assert_allclose(out[SWC1 + "_zscore"].to_numpy(), [(0.305 - MU) / SIGMA] * 2)
        np.testing.assert_allclose(out[SWC1 + "_zscore_min"].to_numpy(), [(0.30 - MU) / SIGMA] * 2)
        np.testing.assert_allclose(out[SWC1 + "_zscore_max"].to_numpy(), [(0.31 - MU) / SIGMA] * 2)

    def test_the_zscore_sum_is_the_day_total_less_n_times_mu(self):
        out = era5_daily_for_site(_era5_site(), swc_climatology_moments(_cells()), "S")
        expected = (7.32 - 24 * MU) / SIGMA
        np.testing.assert_allclose(out[SWC1 + "_zscore_sum"].to_numpy(), [expected, expected])

    def test_the_zscore_sum_follows_a_shorter_day(self):
        """Under --daytime-only n is 12, not 24; the sum must use the hours the day actually needed."""
        out = era5_daily_for_site(_era5_site(), swc_climatology_moments(_cells()), "S", _daylight)
        expected = (3.66 - 12 * MU) / SIGMA
        np.testing.assert_allclose(out[SWC1 + "_raw_sum"].to_numpy(), [3.66, 3.66])
        np.testing.assert_allclose(out[SWC1 + "_zscore_sum"].to_numpy(), [expected, expected])

    def test_a_missing_hour_leaves_every_family_missing(self):
        out = era5_daily_for_site(_era5_site().iloc[1:], swc_climatology_moments(_cells()), "S")
        for suffix in ("", "_min", "_max", "_sum"):
            assert np.isnan(out[SWC1 + suffix].iloc[0])
            assert np.isnan(out[SWC1 + "_zscore" + suffix].iloc[0])
            assert np.isnan(out[SWC1 + "_raw" + suffix].iloc[0])
    def test_a_flat_record_normalises_instead_of_collapsing_to_zero(self):
        """A site whose level never moves has sigma 0 of its own, and the legacy branch writes 0.0.

        The climatology sigma is always >= 1e-6, so the servable path cannot reach that branch: a flat
        record stays a finite constant. This is a property of the transform, not a claim about any site.
        """
        flat = pd.DataFrame({SWC1: [MU, MU, MU, MU]})

        legacy = add_swc_representations(flat, swc_site_moments(flat), "S", warn=False)
        assert (legacy[SWC1] == 0.0).all()

        servable = add_swc_representations(flat, swc_climatology_moments(_cells()), "S")
        np.testing.assert_allclose(servable[SWC1].to_numpy(), MU / SIGMA)
        np.testing.assert_allclose(servable[SWC1 + "_zscore"].to_numpy(), 0.0)

    def test_site_mode_still_normalises_by_the_site(self):
        frame = pd.DataFrame({SWC1: [0.2, 0.3, 0.4, 0.5]})
        out = add_swc_representations(frame, swc_site_moments(frame), "S")
        np.testing.assert_allclose(out[SWC1].to_numpy(), frame[SWC1] / frame[SWC1].std())


class TestPrecipitation:
    def test_precip_is_the_era5_daily_total_in_mm(self):
        daily = era5_daily_for_site(_era5_site(), {}, "S")
        out = apply_servable_precip(daily)
        np.testing.assert_allclose(out["precip"].to_numpy(), [2.4, 2.4])

    def test_precip_equals_the_era5_daily_sum_on_a_complete_day(self):
        """On an own-hours tree the daily mean times 24 and the daily sum are the same day."""
        out = apply_servable_precip(era5_daily_for_site(_era5_site(), {}, "S"))
        np.testing.assert_allclose(out["precip"].to_numpy(), out["total_precipitation_hourly_sum"].to_numpy() * 1000)

    def test_a_partial_day_carries_no_precipitation_at_all(self):
        out = apply_servable_precip(era5_daily_for_site(_era5_site().iloc[1:], {}, "S"))
        assert np.isnan(out["precip"].iloc[0])
    def test_feature_engineering_recomputes_the_identical_value(self):
        """_servable_precip stays the producer for the grid; on this tree it must be a no-op."""
        out = apply_servable_precip(era5_daily_for_site(_era5_site(), {}, "S"))
        again = _servable_precip(out, "servable", "daily")
        np.testing.assert_allclose(again["precip"].to_numpy(), out["precip"].to_numpy())


class TestMergeWiring:
    """The merge takes the contract, and places every site before the loop that swallows failures."""

    def test_a_tree_is_servable_with_no_extra_argument(self):
        default = inspect.signature(merge_sap_env_data_site).parameters["feature_contract"].default
        assert default == SERVABLE_CONTRACT

    def test_the_cube_has_a_default_so_the_stage_4_command_still_runs(self):
        default = inspect.signature(merge_sap_env_data_site).parameters["swvl_cube"].default
        assert str(default).endswith("swvl_sigma_climatology_v2_1995_2019.nc")

    @staticmethod
    def _corpus(tmp_path, coordinates):
        sap, raw = tmp_path / "sap", tmp_path / "raw"
        sap.mkdir()
        raw.mkdir()
        for site, row in coordinates.items():
            (sap / f"{site}_sapf_data_outliers_removed.csv").write_text("")
            if row is not None:
                pd.DataFrame([row]).to_csv(raw / f"{site}_site_md.csv", index=False)
        return sap, raw

    def test_it_places_a_site_that_has_a_coordinate(self, tmp_path):
        sap, raw = self._corpus(tmp_path, {"AAA": {"si_lat": 50.9, "si_long": 8.0}})
        assert _preflight_sites(sap, raw) == [("AAA", 50.9, 8.0)]

    def test_a_site_without_metadata_is_left_to_the_loop(self, tmp_path):
        sap, raw = self._corpus(tmp_path, {"AAA": {"si_lat": 50.9, "si_long": 8.0}, "BBB": None})
        assert [site for site, _lat, _lon in _preflight_sites(sap, raw)] == ["AAA"]

    def test_a_site_without_a_coordinate_is_left_to_the_loop(self, tmp_path):
        sap, raw = self._corpus(
            tmp_path,
            {"AAA": {"si_lat": 50.9, "si_long": 8.0}, "BBB": {"si_lat": np.nan, "si_long": 8.0}},
        )
        assert [site for site, _lat, _lon in _preflight_sites(sap, raw)] == ["AAA"]
