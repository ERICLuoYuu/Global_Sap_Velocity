"""Write-time invariants of the daily env statistics (plan WP-B3, WP-B4; acceptance criteria AC6, AC9, AC12)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.env_daily.invariants import (
    KT_BASELINE_TOTAL,
    KT_RESIDUAL_SITES,
    SupportInvariantError,
    check_support,
    env_daily_rule,
    kt_exceedance_days,
    radiation_gate,
)
from src.env_daily.support import MIN_ENV_HOURS, MIN_TOA_COVERAGE, env_daily_statistics

DAY = pd.Timestamp("2015-06-21", tz="UTC")


def built():
    """Statistics as the module writes them: one complete day and one with gaps."""
    index = pd.date_range(DAY, periods=48, freq="h", name="TIMESTAMP")
    ta = pd.Series(np.arange(48, dtype=float), index=index)
    ta.iloc[30:40] = np.nan
    sw = pd.Series(np.where((index.hour >= 6) & (index.hour < 18), 400.0, np.nan), index=index)
    sw.iloc[34] = np.nan
    return env_daily_statistics(pd.DataFrame({"ta": ta, "sw_in": sw}), 45.0, 0.0)


class TestCheckSupport:
    def test_statistics_built_by_the_module_pass(self):
        check_support(built())

    def test_a_forged_continuous_value_without_support_raises(self):
        stats = built()
        stats.loc[DAY, "ta_hours"] = MIN_ENV_HOURS - 1
        with pytest.raises(SupportInvariantError) as caught:
            check_support(stats)
        assert "ta" in str(caught.value) and "2015-06-21" in str(caught.value)

    def test_a_forged_radiation_value_below_coverage_raises(self):
        stats = built()
        stats.loc[DAY, "sw_in_toa_coverage"] = MIN_TOA_COVERAGE - 0.01
        with pytest.raises(SupportInvariantError):
            check_support(stats)

    def test_a_forged_radiation_extreme_without_twenty_filled_hours_raises(self):
        stats = built()
        assert np.isnan(stats.loc[DAY, "sw_in_max"])  # 19 filled hours: the module wrote no extreme
        stats.loc[DAY, "sw_in_max"] = 400.0
        stats.loc[DAY, "sw_in_hours"] = MIN_ENV_HOURS - 1
        with pytest.raises(SupportInvariantError):
            check_support(stats)

    def test_a_missing_value_needs_no_support(self):
        stats = built()
        stats.loc[DAY, ["ta", "ta_min", "ta_max", "ta_sum"]] = np.nan
        stats.loc[DAY, "ta_hours"] = 3
        check_support(stats)

    def test_coverage_exactly_at_the_threshold_passes(self):
        stats = built()
        stats.loc[DAY, "sw_in_toa_coverage"] = MIN_TOA_COVERAGE
        check_support(stats)

    def test_a_statistic_without_its_hours_column_raises(self):
        with pytest.raises(SupportInvariantError):
            check_support(built().drop(columns=["ta_hours"]))

    def test_a_plant_level_frame_names_the_day_from_its_timestamp_column(self):
        # The merge checks its daily frame: one row per tree per day, TIMESTAMP a column, a RangeIndex.
        one = built().rename_axis("TIMESTAMP").reset_index()
        trees = pd.concat([one, one], ignore_index=True)
        day_rows = trees.index[trees["TIMESTAMP"] == DAY]
        trees.loc[day_rows[1], "ta_hours"] = MIN_ENV_HOURS - 1
        with pytest.raises(SupportInvariantError) as caught:
            check_support(trees)
        message = str(caught.value)
        assert message.startswith("1 daily env value")
        assert "2015-06-21" in message

    def test_a_repeated_day_index_reports_each_violation_once(self):
        stats = pd.concat([built(), built()])
        stats.iloc[0, stats.columns.get_loc("ta_hours")] = MIN_ENV_HOURS - 1
        with pytest.raises(SupportInvariantError) as caught:
            check_support(stats)
        assert str(caught.value).startswith("1 daily env value")
        assert "2015-06-21: 19 hours" in str(caught.value)

    @pytest.mark.parametrize("hours, ok", [(10, True), (9, False)])
    def test_the_threshold_scales_with_the_expected_hours(self, hours, ok):
        stats = built()
        stats.loc[DAY, "env_expected_hours"] = 12
        stats.loc[DAY, "ta_hours"] = hours
        if ok:
            check_support(stats)
        else:
            with pytest.raises(SupportInvariantError):
                check_support(stats)


class TestKtExceedance:
    def frame(self, days, sw_in, ext_rad):
        return pd.DataFrame({"TIMESTAMP": pd.DatetimeIndex(days, tz="UTC"), "sw_in": sw_in, "ext_rad": ext_rad})

    def test_counts_days_on_which_sw_in_exceeds_ext_rad(self):
        frame = self.frame(["2015-06-01", "2015-06-02", "2015-06-03"], [100.0, 350.0, 50.0], [300.0, 300.0, 300.0])
        assert kt_exceedance_days(frame) == 1

    def test_equal_values_do_not_count(self):
        assert kt_exceedance_days(self.frame(["2015-06-01"], [300.0], [300.0])) == 0

    def test_a_plant_level_frame_counts_each_day_once(self):
        frame = self.frame(["2015-06-02", "2015-06-02", "2015-06-02"], [350.0] * 3, [300.0] * 3)
        assert kt_exceedance_days(frame) == 1

    def test_missing_and_non_positive_ext_rad_rows_are_ignored(self):
        frame = self.frame(["2015-06-01", "2015-06-02", "2015-06-03"], [np.nan, 10.0, 10.0], [300.0, 0.0, np.nan])
        assert kt_exceedance_days(frame) == 0

    def test_a_frame_without_the_columns_counts_zero(self):
        assert kt_exceedance_days(pd.DataFrame({"TIMESTAMP": [DAY], "ta": [1.0]})) == 0


class TestRadiationGate:
    def test_the_documented_residual_passes(self):
        assert sum(KT_RESIDUAL_SITES.values()) == KT_BASELINE_TOTAL
        assert radiation_gate(dict(KT_RESIDUAL_SITES)) == []

    def test_a_total_above_the_baseline_fails(self):
        counts = dict(KT_RESIDUAL_SITES)
        counts["FIN_HYY_SME"] += 1
        assert radiation_gate(counts)

    def test_a_new_site_with_any_exceedance_fails_even_below_the_baseline(self):
        errors = radiation_gate({"USA_TNO": 1})
        assert any("USA_TNO" in error for error in errors)

    def test_a_residual_site_that_worsens_fails_even_when_the_total_holds(self):
        # The total is a sum, and a sum lets one site pay for another: FIN_HYY_SME can climb from 155 to
        # 173 while USA_BNZ_BLA falls to 0, and the corpus total does not move at all. Each site answers
        # for itself.
        counts = dict(KT_RESIDUAL_SITES)
        counts["FIN_HYY_SME"] = KT_RESIDUAL_SITES["FIN_HYY_SME"] + KT_RESIDUAL_SITES["USA_BNZ_BLA"]
        counts["USA_BNZ_BLA"] = 0
        assert sum(counts.values()) <= KT_BASELINE_TOTAL
        errors = radiation_gate(counts)
        assert any("FIN_HYY_SME" in error for error in errors)

    def test_zero_counts_at_other_sites_pass(self):
        assert radiation_gate({"USA_TNO": 0, "FIN_HYY_SME": 10}) == []


class TestRule:
    def test_the_rule_records_every_threshold(self):
        rule = env_daily_rule()
        assert rule["version"] == 1
        assert rule["min_hours"] == MIN_ENV_HOURS
        assert rule["min_toa_coverage"] == MIN_TOA_COVERAGE
        assert rule["ext_rad"] == "geometry"
        assert "netrad" in rule["not_covered"]

    def test_the_rule_is_json_serialisable(self):
        assert json.loads(json.dumps(env_daily_rule())) == env_daily_rule()
