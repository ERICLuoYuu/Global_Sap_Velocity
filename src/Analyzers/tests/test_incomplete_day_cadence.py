"""Stage 3's incomplete-day rule judges each tree against the step it logs at, not the step of the file's grid.

HUN_SIK's file sits on a 10-minute grid with values only on the hour, so the rule saw 16 of 92 daytime points and
removed every day; CRI_TAM_TOW, MEX_VER_BSM, MEX_VER_BSJ, UZB_YAN_DIS and ESP_LAH_COM lost 22-61% of their
day-columns the same way.
"""

import numpy as np
import pandas as pd

from src.Analyzers.sap_analyzer import SapFlowAnalyzer


def _grid(start, periods, minutes):
    return pd.date_range(start, periods=periods, freq=f"{minutes}min", tz="UTC")


def _daylight(index):
    return pd.Series((index.hour >= 6) & (index.hour < 18), index=index)


def _remove(df):
    analyzer = object.__new__(SapFlowAnalyzer)
    return analyzer._remove_incomplete_days(df, _daylight(df.index), completeness_threshold=0.5, data_columns=["tree"])


def test_an_hourly_tree_on_a_ten_minute_grid_keeps_its_complete_days():
    index = _grid("2014-06-01", 6 * 24 * 3, 10)
    df = pd.DataFrame({"tree": np.where(index.minute == 0, 1.0, np.nan)}, index=index)

    out, report = _remove(df)

    assert out["tree"].notna().sum() == df["tree"].notna().sum()  # nosec B101
    daytime = report[report["expected_daytime_points"] > 0]
    assert not daytime["removed"].any()  # nosec B101
    assert (daytime["completeness"] == 1.0).all()  # nosec B101
    assert (daytime["column_step_minutes"] == 60.0).all()  # nosec B101
    assert np.allclose(daytime["expected_samples"], 12.0)  # nosec B101


def test_an_hourly_tree_missing_most_of_a_day_still_loses_that_day():
    index = _grid("2014-06-01", 6 * 24 * 2, 10)
    values = np.where(index.minute == 0, 1.0, np.nan)
    values[(index.day == 2) & (index.hour >= 9)] = np.nan  # 3 of 12 daylight hours left on day 2
    df = pd.DataFrame({"tree": values}, index=index)

    out, _ = _remove(df)

    assert out.loc[index.day == 2, "tree"].isna().all()  # nosec B101
    assert out.loc[index.day == 1, "tree"].notna().sum() == 24  # nosec B101


def test_a_tree_logging_at_the_grid_step_is_judged_exactly_as_before():
    index = _grid("2014-06-01", 48 * 6, 30)
    present = np.random.default_rng(1).random(len(index)) < 0.6
    df = pd.DataFrame({"tree": np.where(present, 1.0, np.nan)}, index=index)

    out, report = _remove(df)

    daytime = report[report["expected_daytime_points"] > 0]
    rows_rule = daytime["valid_daytime_points"] / daytime["expected_daytime_points"]
    assert (daytime["completeness"] == rows_rule).all()  # nosec B101
    assert (daytime["removed"] == (rows_rule < 0.5)).all()  # nosec B101
    assert (daytime["expected_samples"] == daytime["expected_daytime_points"]).all()  # nosec B101
    assert (daytime["valid_samples"] == daytime["valid_daytime_points"]).all()  # nosec B101


def test_ten_minute_values_in_one_hour_do_not_make_an_hourly_tree_s_day():
    # The tree logs hourly for three days, then every 10 minutes for one hour of day four only.
    index = _grid("2014-06-01", 6 * 24 * 4, 10)
    values = np.where(index.minute == 0, 1.0, np.nan)
    day_four = index.day == 4
    values[day_four] = np.nan
    values[day_four & (index.hour == 10)] = 1.0
    df = pd.DataFrame({"tree": values}, index=index)

    out, _ = _remove(df)

    assert out.loc[day_four, "tree"].isna().all()  # nosec B101
    assert out.loc[~day_four, "tree"].notna().sum() == 72  # nosec B101


def test_near_duplicate_rows_that_both_hold_a_value_count_once():
    # ZAF_FRA_FRA: rows seconds apart. Five of twelve daylight hours present, each twice.
    hourly = _grid("2015-02-02", 24, 60)
    index = hourly.append(hourly + pd.Timedelta(seconds=5)).sort_values()
    present = (index.hour >= 6) & (index.hour < 11)
    df = pd.DataFrame({"tree": np.where(present, 1.0, np.nan)}, index=index)

    out, report = _remove(df)

    assert out["tree"].isna().all()  # nosec B101
    assert report.loc[report["expected_daytime_points"] > 0, "completeness"].iloc[0] < 0.5  # nosec B101


def test_a_tree_left_with_one_value_a_day_is_judged_by_rows():
    index = _grid("2014-06-01", 6 * 24 * 3, 10)
    df = pd.DataFrame({"tree": np.where((index.hour == 12) & (index.minute == 0), 1.0, np.nan)}, index=index)

    out, report = _remove(df)

    assert out["tree"].isna().all()  # nosec B101
    assert report["column_step_minutes"].isna().all()  # nosec B101


def test_an_hourly_tree_with_exactly_half_its_daylight_hours_keeps_the_day():
    # 72 daylight rows of weight 1/6 must sum to 12, not 1 ulp above it.
    index = _grid("2014-06-01", 6 * 24 * 3, 10)
    values = np.where(index.minute == 0, 1.0, np.nan)
    values[(index.day == 2) & (index.hour >= 12)] = np.nan  # day 2: hours 6-11, 6 of 12
    df = pd.DataFrame({"tree": values}, index=index)

    out, report = _remove(df)

    day_two = report[(report["expected_daytime_points"] > 0) & (pd.to_datetime(report["date"]).dt.day == 2)]
    assert (day_two["completeness"] == 0.5).all()  # nosec B101
    assert out.loc[index.day == 2, "tree"].notna().sum() == 12  # nosec B101


def test_the_new_report_columns_come_after_the_old_ones():
    index = _grid("2014-06-01", 48, 30)
    _, report = _remove(pd.DataFrame({"tree": 1.0}, index=index))

    assert list(report.columns) == [  # nosec B101
        "date",
        "column",
        "expected_daytime_points",
        "valid_daytime_points",
        "completeness",
        "removed",
        "column_step_minutes",
        "expected_samples",
        "valid_samples",
    ]
