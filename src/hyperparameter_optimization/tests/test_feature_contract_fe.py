"""Feature contract inside feature engineering.

``site``      the legacy definitions: precipitation memory from the site gauge, and
              soil-moisture features that are statistics of each site's own record.
``servable``  definitions a 0.1 deg grid cell can reproduce at prediction time:
              ERA5-Land daily precipitation, and soil moisture normalised by the
              per-cell 1995-2018 climatology instead of the site record.
"""

import numpy as np
import pandas as pd
import pytest
from src.feature_contract.definitions import SERVABLE_RAIN_EVENT_MM_PER_DAY

from src.hyperparameter_optimization.feature_engineering import (
    apply_all_feature_engineering,
    apply_feature_engineering,
)

#: Daily mean of the hourly ERA5-Land values, metres per hour: mm/day = TP * 24 * 1000.
TP = "total_precipitation_hourly"
PRECIP_FEATURES = {"precip_sum_3d", "precip_sum_7d", "days_since_rain"}


class TestContractValidation:
    def test_unknown_contract_raises(self):
        df = pd.DataFrame({"precip": [1.0, 0.0]})
        with pytest.raises(ValueError, match="contract"):
            apply_feature_engineering(df, ["precip_memory"], contract="bogus")

    def test_unknown_contract_raises_in_apply_all(self):
        df = pd.DataFrame({"vpd": [1.0, 2.0]})
        with pytest.raises(ValueError, match="contract"):
            apply_all_feature_engineering(df, contract="bogus")

    def test_default_is_site(self):
        df = pd.DataFrame({"precip": [5.0, 0.0, 0.0, 3.0, 0.0], TP: [0.0] * 5})
        default, _ = apply_feature_engineering(df, ["precip_memory"])
        site, _ = apply_feature_engineering(df, ["precip_memory"], contract="site")
        pd.testing.assert_frame_equal(default, site)


class TestPrecipMemoryContract:
    def test_site_reads_site_precip_and_ignores_era5(self):
        df = pd.DataFrame({"precip": [5.0, 0.0, 0.0, 3.0, 0.0], TP: [0.0, 0.009, 0.009, 0.0, 0.0]})
        out, _ = apply_feature_engineering(df, ["precip_memory"], contract="site")
        assert out["days_since_rain"].tolist() == [0, 1, 2, 0, 1]
        assert out["precip_sum_3d"].tolist() == [5.0, 5.0, 5.0, 3.0, 3.0]

    def test_servable_reads_era5_in_mm_and_ignores_site_precip(self):
        # The site gauge says the opposite of ERA5 on purpose: reading it would show.
        df = pd.DataFrame({"precip": [0.0, 9.0, 9.0, 0.0, 0.0], TP: [v / 24 for v in (0.005, 0.0, 0.0, 0.002, 0.0)]})
        out, feats = apply_feature_engineering(df, ["precip_memory"], contract="servable")
        assert set(feats) >= PRECIP_FEATURES
        np.testing.assert_allclose(out["precip_sum_3d"].to_numpy(), [5.0, 5.0, 5.0, 2.0, 2.0])
        np.testing.assert_allclose(out["precip_sum_7d"].to_numpy(), [5.0, 5.0, 5.0, 7.0, 7.0])
        assert out["days_since_rain"].tolist() == [0, 1, 2, 0, 1]

    def test_servable_rain_event_threshold_is_one_mm_per_day(self):
        assert SERVABLE_RAIN_EVENT_MM_PER_DAY == 1.0
        df = pd.DataFrame({TP: [v / 24 for v in (0.0012, 0.0008, 0.0008, 0.0012)]})  # 1.2, 0.8, 0.8, 1.2 mm
        out, _ = apply_feature_engineering(df, ["precip_memory"], contract="servable")
        assert out["days_since_rain"].tolist() == [0, 1, 2, 0]

    def test_servable_without_era5_column_emits_no_precip_features(self):
        df = pd.DataFrame({"precip": [5.0, 0.0, 3.0]})
        out, feats = apply_feature_engineering(df, ["precip_memory"], contract="servable")
        assert not PRECIP_FEATURES & set(feats)
        assert not PRECIP_FEATURES & set(out.columns)

    def test_servable_hourly_is_not_defined(self):
        df = pd.DataFrame({TP: [0.001, 0.0]})
        with pytest.raises(NotImplementedError):
            apply_feature_engineering(df, ["precip_memory"], time_scale="hourly", contract="servable")


class TestIndicatorsContract:
    def test_site_quantile_features_are_not_emitted_under_servable(self):
        df = pd.DataFrame({"vpd": [1.0, 3.0, 2.0], "volumetric_soil_water_layer_1_raw": [0.2, 0.3, 0.25]})
        _, site_feats = apply_feature_engineering(df, ["indicators"], contract="site")
        out, serv_feats = apply_feature_engineering(df, ["indicators"], contract="servable")
        assert {"soil_moisture_rel", "soil_dry"} <= set(site_feats)
        assert not {"soil_moisture_rel", "soil_dry"} & set(serv_feats)
        assert "soil_moisture_rel" not in out.columns
        assert "vpd_high" in serv_feats


class TestApplyAllContract:
    @staticmethod
    def _frame():
        rng = np.random.RandomState(1)
        n = 20
        return pd.DataFrame(
            {
                "vpd": rng.uniform(0.5, 3.0, n),
                "ta": rng.uniform(5.0, 30.0, n),
                "sw_in": rng.uniform(50.0, 400.0, n),
                "precip": np.zeros(n),
                TP: np.full(n, 0.004 / 24),
                "volumetric_soil_water_layer_1_raw": rng.uniform(0.1, 0.4, n),
                "volumetric_soil_water_layer_3_zscore": rng.uniform(-2.0, 2.0, n),
                "latitude": np.full(n, 51.0),
            }
        )

    def test_contract_reaches_the_groups(self):
        df = self._frame()
        site, _ = apply_all_feature_engineering(df, contract="site")
        serv, _ = apply_all_feature_engineering(df, contract="servable")
        assert site["precip_sum_7d"].max() == 0.0
        np.testing.assert_allclose(serv["precip_sum_7d"].iloc[-1], 28.0)
        z3 = "volumetric_soil_water_layer_3_zscore"
        np.testing.assert_array_equal(serv[z3].to_numpy(), df[z3].to_numpy())

    def test_features_outside_the_contract_are_identical(self):
        df = self._frame()
        site, _ = apply_all_feature_engineering(df, contract="site")
        serv, _ = apply_all_feature_engineering(df, contract="servable")
        for col in ["ta_lag1d", "vpd_roll14d_std", "vpd_x_swc", "ta_x_swc", "vpd_log", "ta_change_1d"]:
            pd.testing.assert_series_equal(site[col], serv[col])
