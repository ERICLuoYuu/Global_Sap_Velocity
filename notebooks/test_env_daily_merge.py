"""The Stage-4 merge takes env daily statistics from env's own hours and ext_rad from geometry (plan WP-B).

The merge's daily aggregation runs over the hours in which sap AND env both observed, with no completeness rule.
``env_daily_overwrite`` replaces the env sensor statistics with :mod:`src.env_daily`'s, by UTC day, after the
ERA5 and gauge overwrites; the site loop checks the support invariant before every daily write and records the
rule, the clock registry and the kt > 1 count in the feature-contract block.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from notebooks.merge_gap_filled_hourly_orginal import env_daily_overwrite, radiation_provenance
from src.clock_offsets.registry import registry_sha256
from src.env_daily.invariants import KT_BASELINE_TOTAL, KT_RESIDUAL_SITES, check_support, env_daily_rule
from src.env_daily.support import MIN_ENV_HOURS, hourly_toa
from src.serving.era5_daily import ext_rad_daily

LAT, LON = 45.0, 0.0
DAYS = pd.DatetimeIndex(["2015-06-20", "2015-06-21"], tz="UTC")
KT = 0.7
SENTINEL = -999.0
MERGE = Path(__file__).with_name("merge_gap_filled_hourly_orginal.py")
SUPPORT_COLUMNS = ("ta_hours", "sw_in_hours", "sw_in_toa_coverage", "env_expected_hours")


def env_hourly(sw_night_na=True, ta_hours_day2=24, shipped_ext_rad=True):
    """What ``aggregate_env_to_hourly`` hands the merge: one row per hour, index TIMESTAMP."""
    index = pd.date_range(DAYS[0], periods=48, freq="h", name="TIMESTAMP")
    toa, dark = hourly_toa(DAYS, LAT, LON)
    sw_in = KT * toa.ravel()
    if sw_night_na:
        sw_in = np.where(dark.ravel(), np.nan, sw_in)
    ta = 15.0 + 10.0 * np.sin(np.arange(48) / 24.0 * 2.0 * np.pi)
    ta[24 + ta_hours_day2 :] = np.nan
    columns = {"ta": ta, "sw_in": sw_in, "netrad": np.full(48, 123.0)}
    if shipped_ext_rad:
        columns["ext_rad"] = np.roll(toa.ravel(), 5)  # the archive's column, displaced by five hours
    return pd.DataFrame(columns, index=index)


def joined_daily(days=DAYS, trees=None):
    """The merge's daily frame before the overwrite: sap-and-env statistics, sentinels here."""
    frame = pd.DataFrame({"TIMESTAMP": days})
    for col in ("ta", "sw_in", "ext_rad", "netrad"):
        for suffix in ("", "_min", "_max", "_sum"):
            frame[col + suffix] = SENTINEL
    frame["sap_velocity"] = np.arange(1.0, len(days) + 1.0)
    frame["site_name"] = "SITE"
    if trees:
        frame = pd.concat([frame.assign(pl_code=tree) for tree in trees], ignore_index=True)
    return frame


def lit(stamps):
    return np.isin(pd.DatetimeIndex(stamps).hour, range(6, 18))


class TestEnvDailyOverwrite:
    def test_env_columns_come_from_envs_own_hours(self):
        out = env_daily_overwrite(joined_daily(), env_hourly(), LAT, LON)
        ta = env_hourly()["ta"].to_numpy().reshape(2, 24)
        np.testing.assert_allclose(out["ta"].to_numpy(), ta.mean(axis=1), rtol=0, atol=1e-12)
        np.testing.assert_allclose(out["ta_max"].to_numpy(), ta.max(axis=1), rtol=0, atol=1e-12)

    def test_sap_gaps_cannot_change_an_env_value(self):
        # POL_TUC: the joined statistics differ with sap's gaps; the written env values must not.
        other = joined_daily()
        other.loc[:, ["ta", "sw_in", "ext_rad"]] = 1.0
        first = env_daily_overwrite(joined_daily(), env_hourly(), LAT, LON)
        second = env_daily_overwrite(other, env_hourly(), LAT, LON)
        for col in ("ta", "ta_min", "ta_max", "sw_in", "sw_in_sum", "ext_rad"):
            np.testing.assert_array_equal(first[col].to_numpy(), second[col].to_numpy())

    def test_night_na_sw_in_becomes_the_24_hour_mean_not_the_daylight_mean(self):
        out = env_daily_overwrite(joined_daily(), env_hourly(sw_night_na=True), LAT, LON)
        toa, _dark = hourly_toa(DAYS, LAT, LON)
        np.testing.assert_allclose(out["sw_in"].to_numpy(), KT * toa.mean(axis=1), rtol=1e-12)
        daylight_mean = np.nanmean(env_hourly(sw_night_na=True)["sw_in"].to_numpy().reshape(2, 24), axis=1)
        assert (out["sw_in"].to_numpy() < daylight_mean).all()

    @pytest.mark.parametrize("shipped", [True, False])
    def test_ext_rad_is_the_served_value_on_every_row(self, shipped):
        out = env_daily_overwrite(joined_daily(), env_hourly(shipped_ext_rad=shipped), LAT, LON)
        served = ext_rad_daily(DAYS, LAT, LON).to_numpy()
        assert np.max(np.abs(out["ext_rad"].to_numpy() - served)) <= 1e-9

    def test_netrad_and_the_sap_columns_keep_the_joined_statistics(self):
        daily = joined_daily()
        out = env_daily_overwrite(daily, env_hourly(), LAT, LON)
        for col in ("netrad", "netrad_min", "netrad_max", "netrad_sum"):
            assert (out[col] == SENTINEL).all(), col
        pd.testing.assert_series_equal(out["sap_velocity"], daily["sap_velocity"])

    def test_rows_and_their_order_are_unchanged_and_the_input_is_not_modified(self):
        daily = joined_daily()
        before = daily.copy(deep=True)
        out = env_daily_overwrite(daily, env_hourly(), LAT, LON)
        pd.testing.assert_frame_equal(daily, before)
        assert list(out["TIMESTAMP"]) == list(daily["TIMESTAMP"])
        assert list(out.columns[: len(daily.columns)]) == list(daily.columns)

    def test_a_variable_env_does_not_carry_is_not_invented(self):
        out = env_daily_overwrite(joined_daily(), env_hourly(), LAT, LON)
        assert not {"vpd", "rh", "ws", "ppfd_in", "vpd_hours"} & set(out.columns)

    def test_a_day_env_covered_too_few_hours_is_missing_not_partial(self):
        out = env_daily_overwrite(joined_daily(), env_hourly(ta_hours_day2=MIN_ENV_HOURS - 1), LAT, LON)
        assert np.isfinite(out.loc[0, "ta"])
        assert out.loc[1, ["ta", "ta_min", "ta_max", "ta_sum"]].isna().all()
        assert out.loc[1, "ta_hours"] == MIN_ENV_HOURS - 1

    def test_the_support_is_written_and_satisfies_the_invariant(self):
        out = env_daily_overwrite(joined_daily(), env_hourly(), LAT, LON)
        assert set(SUPPORT_COLUMNS) <= set(out.columns)
        check_support(out)

    def test_a_plant_level_frame_gives_every_tree_its_days_values(self):
        out = env_daily_overwrite(joined_daily(trees=["T1", "T2"]), env_hourly(), LAT, LON)
        by_tree = out.set_index(["pl_code", "TIMESTAMP"])
        for col in ("ta", "sw_in", "ext_rad", "ta_hours"):
            np.testing.assert_array_equal(by_tree.loc["T1", col].to_numpy(), by_tree.loc["T2", col].to_numpy())

    def test_naive_day_stamps_are_utc(self):
        naive = env_daily_overwrite(joined_daily(days=DAYS.tz_localize(None)), env_hourly(), LAT, LON)
        aware = env_daily_overwrite(joined_daily(), env_hourly(), LAT, LON)
        for col in ("ta", "sw_in", "ext_rad"):
            np.testing.assert_array_equal(naive[col].to_numpy(), aware[col].to_numpy())

    def test_under_daytime_only_the_daily_clearness_is_exact(self):
        # --daytime-only scopes every env statistic to the kept hours, ext_rad included, so sw_in / ext_rad
        # is the clearness of the kept hours rather than a daylight mean over a 24-hour one.
        out = env_daily_overwrite(joined_daily(), env_hourly(sw_night_na=False), LAT, LON, keep_hours=lit)
        np.testing.assert_allclose((out["sw_in"] / out["ext_rad"]).to_numpy(), KT, rtol=1e-12)
        assert (out["env_expected_hours"] == 12).all()


class TestRadiationProvenance:
    def test_the_badge_before_the_loop_records_the_rule_and_the_registry_but_no_count(self):
        block = radiation_provenance()
        assert block["env_daily_rule"] == env_daily_rule()
        assert block["clock_registry_sha256"] == registry_sha256()
        assert block["kt_exceedance"] is None

    def test_the_final_block_counts_sites_with_exceedances_and_carries_the_gate(self):
        counts = {"SITE_A": 0, "FIN_HYY_SME": 3, "USA_TNO": 2}
        kt = radiation_provenance(counts)["kt_exceedance"]
        assert kt["by_site"] == {"FIN_HYY_SME": 3, "USA_TNO": 2}
        assert kt["total"] == 5
        assert kt["n_sites_counted"] == 3
        assert kt["baseline_total"] == KT_BASELINE_TOTAL
        assert any("USA_TNO" in reason for reason in kt["gate"])

    def test_the_documented_residual_passes_its_own_gate(self):
        assert radiation_provenance(dict(KT_RESIDUAL_SITES))["kt_exceedance"]["gate"] == []

    def test_it_is_json_serialisable(self):
        block = radiation_provenance({"FIN_HYY_SME": np.int64(3)})
        assert json.loads(json.dumps(block)) == block


def _merge_function():
    tree = ast.parse(MERGE.read_text())
    return next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "merge_sap_env_data_site"
    )


def _calls(node, name):
    """Line numbers of calls to ``name`` (a bare function or a method) under ``node``."""
    found = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if called == name:
            found.append((call.lineno, call))
    return sorted(found, key=lambda item: item[0])


def _daily_write_line(function):
    writes = [
        line
        for line, call in _calls(function, "to_csv")
        if isinstance(call.func.value, ast.Name) and call.func.value.id == "df_daily"
    ]
    assert len(writes) == 1, writes
    return writes[0]


class TestSiteLoopWiring:
    def test_the_overwrite_runs_once_after_era5_and_the_gauge_and_before_the_sap_scope(self):
        function = _merge_function()
        overwrite = _calls(function, "env_daily_overwrite")
        assert len(overwrite) == 1
        line = overwrite[0][0]
        assert line > max(found for found, _call in _calls(function, "apply_servable_precip"))
        assert line > max(found for found, _call in _calls(function, "gauge_daily_totals"))
        assert line < min(found for found, _call in _calls(function, "apply_daytime_sap_scope"))

    def test_support_is_checked_after_the_overwrite_and_before_the_daily_write(self):
        function = _merge_function()
        checks = _calls(function, "check_support")
        assert len(checks) == 1
        assert _calls(function, "env_daily_overwrite")[0][0] < checks[0][0] < _daily_write_line(function)

    def test_kt_is_counted_for_every_written_site(self):
        function = _merge_function()
        counted = _calls(function, "kt_exceedance_days")
        assert len(counted) == 1
        assert counted[0][0] > _daily_write_line(function)

    def test_a_support_violation_ends_the_run_instead_of_skipping_the_site(self):
        function = _merge_function()
        handlers = next(
            node.handlers
            for node in ast.walk(function)
            if isinstance(node, ast.Try)
            and any(getattr(handler.type, "id", None) == "FeatureContractError" for handler in node.handlers)
        )
        names = [getattr(handler.type, "id", None) for handler in handlers]
        assert names.index("SupportInvariantError") < names.index("Exception")
        support = handlers[names.index("SupportInvariantError")]
        assert len(support.body) == 1 and isinstance(support.body[0], ast.Raise) and support.body[0].exc is None

    def test_both_contract_badges_carry_the_radiation_provenance(self):
        calls = _calls(_merge_function(), "radiation_provenance")
        assert sorted(len(call.args) for _line, call in calls) == [0, 1]

    def test_the_merge_never_applies_a_clock_offset(self):
        # F1: offsets are applied once, at the Stage-3 load; the merge reads the corrected output.
        tree = ast.parse(MERGE.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert "src.clock_offsets.apply" not in imported
        assert not {"correct_clock", "correct_loaded_frames"} & imported


def test_no_support_column_is_an_ffs_candidate():
    # F16: the support columns are written to the tree, and must never become features.
    from src.forward_selection.feature_registry import get_all_expected_columns

    written = {f"{col}_hours" for col in ("ta", "vpd", "rh", "ws", "sw_in", "ppfd_in")}
    written |= {"sw_in_toa_coverage", "ppfd_in_toa_coverage", "env_expected_hours"}
    assert not written & set(get_all_expected_columns())
