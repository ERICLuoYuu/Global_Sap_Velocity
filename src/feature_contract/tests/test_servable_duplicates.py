"""Two names for one quantity, under the servable contract.

Under ``site``, ``precip`` is the rain GAUGE and :data:`ERA5_DAILY_PRECIP_COL` is the ERA5-Land
reanalysis -- different measurements, both worth offering to selection. Under ``servable`` the merge
writes ``precip`` as the ERA5-Land daily total itself, and then the two are one variable a thousand
apart. That failure is quiet by construction: both columns are present and numeric, so no loader
reports anything missing.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_contract.definitions import (
    ERA5_DAILY_PRECIP_COL,
    HOURS_PER_DAY,
    M_TO_MM,
    SERVABLE_DUPLICATE_FEATURES,
    assert_no_servable_duplicate_features,
    era5_hourly_mean_precip_mm_per_day,
    servable_duplicate_features,
)


class TestTheArithmeticTheClaimRestsOn:
    """If this stops holding, the two columns are no longer one variable and the rest is wrong."""

    def test_precip_is_the_era5_daily_total_a_thousand_apart(self):
        mean_m_per_hour = pd.Series([0.0, 1e-5, 2.5e-4, 3e-3])
        # complete_day_statistics takes the mean and the sum over the SAME complete day, and emits
        # neither unless every hour of it is present -- so on any row that exists at all, the sum is
        # exactly HOURS_PER_DAY x the mean.
        daily_sum_m = HOURS_PER_DAY * mean_m_per_hour
        precip_mm = era5_hourly_mean_precip_mm_per_day(mean_m_per_hour)
        assert np.array_equal(precip_mm.to_numpy(), (M_TO_MM * daily_sum_m).to_numpy())

    def test_the_column_named_as_the_duplicate_is_the_one_holding_that_total(self):
        assert ERA5_DAILY_PRECIP_COL in SERVABLE_DUPLICATE_FEATURES


class TestWhichContractHasDuplicates:
    def test_servable_has_exactly_this_one(self):
        assert set(servable_duplicate_features("servable")) == {ERA5_DAILY_PRECIP_COL}

    def test_site_has_none(self):
        """Gauge and reanalysis are different measurements; under site both belong."""
        assert servable_duplicate_features("site") == {}

    def test_an_unknown_contract_is_rejected(self):
        with pytest.raises(ValueError):
            servable_duplicate_features("clim")

    def test_the_mapping_handed_out_is_a_copy(self):
        handed_out = servable_duplicate_features("servable")
        handed_out.clear()
        assert set(servable_duplicate_features("servable")) == {ERA5_DAILY_PRECIP_COL}


class TestTheGuard:
    def test_it_refuses_a_servable_list_naming_both(self):
        with pytest.raises(ValueError, match=ERA5_DAILY_PRECIP_COL):
            assert_no_servable_duplicate_features(
                ["precip", ERA5_DAILY_PRECIP_COL, "ta"], "a list", "servable"
            )

    def test_the_message_says_what_the_name_duplicates(self):
        with pytest.raises(ValueError, match="precip"):
            assert_no_servable_duplicate_features([ERA5_DAILY_PRECIP_COL], "a list", "servable")

    def test_the_message_names_the_source(self):
        with pytest.raises(ValueError, match="features.json"):
            assert_no_servable_duplicate_features([ERA5_DAILY_PRECIP_COL], "features.json", "servable")

    def test_a_servable_list_without_it_passes(self):
        assert_no_servable_duplicate_features(["precip", "ta", "vpd"], "a list", "servable")

    def test_the_very_same_list_passes_under_site(self):
        assert_no_servable_duplicate_features(["precip", ERA5_DAILY_PRECIP_COL], "a list", "site")

    def test_an_empty_list_passes(self):
        assert_no_servable_duplicate_features([], "a list", "servable")
