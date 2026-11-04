"""How often a column logs, and how much of one of its samples each row of the shared grid stands for."""

import numpy as np
import pandas as pd
import pytest
from src.Analyzers.sampling_cadence import column_step_minutes, sample_weights, valid_sample_weights


def _grid(start, periods, minutes):
    return pd.date_range(start, periods=periods, freq=f"{minutes}min", tz="UTC")


def test_hourly_values_on_a_ten_minute_grid_log_every_sixty_minutes():
    index = _grid("2014-06-01", 6 * 24 * 3, 10)
    values = pd.Series(np.where(index.minute == 0, 1.0, np.nan), index=index)
    assert column_step_minutes(index, values) == 60.0  # nosec B101


def test_a_column_missing_most_values_at_random_still_logs_at_the_grid_step():
    index = _grid("2014-06-01", 6 * 24 * 10, 10)
    present = np.random.default_rng(0).random(len(index)) < 0.35
    values = pd.Series(np.where(present, 1.0, np.nan), index=index)
    assert column_step_minutes(index, values) == 10.0  # nosec B101


def test_clock_jitter_rounds_to_the_whole_minute():
    start = pd.Timestamp("2015-02-02", tz="UTC")
    index = pd.DatetimeIndex([start + pd.Timedelta(minutes=60.02 * i) for i in range(50)])
    assert column_step_minutes(index, pd.Series(1.0, index=index)) == 60.0  # nosec B101


def test_fewer_than_two_values_have_no_step():
    index = _grid("2014-06-01", 10, 10)
    assert np.isnan(column_step_minutes(index, pd.Series([np.nan] * 9 + [1.0], index=index)))  # nosec B101


def test_rows_on_a_grid_at_the_column_step_each_weigh_exactly_one():
    index = _grid("2014-06-01", 48, 30)
    assert (sample_weights(index, 30.0) == 1.0).all()  # nosec B101


def test_a_gap_in_the_grid_never_makes_the_row_before_it_weigh_more_than_one():
    index = _grid("2014-06-01", 4, 30).append(_grid("2014-06-05", 4, 30))
    assert (sample_weights(index, 30.0) == 1.0).all()  # nosec B101


def test_six_ten_minute_rows_make_one_hourly_sample():
    index = _grid("2014-06-01", 6 * 24, 10)
    weights = sample_weights(index, 60.0)
    assert weights.to_numpy() == pytest.approx(np.full(len(index), 1 / 6))  # nosec B101
    assert weights.sum() == pytest.approx(24.0)  # nosec B101


def test_a_fifteen_minute_column_on_an_interleaved_grid_expects_four_samples_an_hour():
    # MEX_VER_BSJ and ESP_LAH_COM: a 10-minute and a 15-minute grid in one file.
    start = pd.Timestamp("2012-06-01", tz="UTC")
    minutes = [0, 10, 15, 20, 30, 40, 45, 50]
    index = pd.DatetimeIndex([start + pd.Timedelta(hours=h, minutes=m) for h in range(24) for m in minutes])
    weights = sample_weights(index, 15.0)
    assert weights[index.hour < 23].sum() == pytest.approx(23 * 4)  # nosec B101


def test_a_near_duplicate_row_adds_nothing():
    # ZAF_FRA_FRA holds rows seconds apart.
    hourly = _grid("2015-02-02", 24, 60)
    index = hourly.append(pd.DatetimeIndex([hourly[5] + pd.Timedelta(seconds=5)])).sort_values()
    weights = sample_weights(index, 60.0)
    assert weights.iloc[:-1].sum() == pytest.approx(23.0)  # nosec B101


def test_no_step_weighs_every_row_one():
    index = _grid("2014-06-01", 12, 10)
    assert (sample_weights(index, float("nan")) == 1.0).all()  # nosec B101
    assert (sample_weights(index[:1], 60.0) == 1.0).all()  # nosec B101


def test_a_step_above_two_hours_is_not_a_cadence():
    # A tree left with one value a day would otherwise expect half a sample per daylight.
    index = _grid("2014-06-01", 6 * 24 * 5, 10)
    values = pd.Series(np.where((index.hour == 12) & (index.minute == 0), 1.0, np.nan), index=index)
    assert np.isnan(column_step_minutes(index, values))  # nosec B101


def test_two_hourly_values_are_still_a_cadence():
    index = _grid("2014-06-01", 6 * 24 * 5, 10)
    values = pd.Series(np.where((index.hour % 2 == 0) & (index.minute == 0), 1.0, np.nan), index=index)
    assert column_step_minutes(index, values) == 120.0  # nosec B101


def test_valid_values_weigh_one_sample_each_at_their_own_step():
    index = _grid("2014-06-01", 6 * 24, 10)
    values = pd.Series(np.where(index.minute == 0, 1.0, np.nan), index=index)
    weights = valid_sample_weights(index, values, 60.0)
    assert (weights[values.notna()] == 1.0).all() and (weights[values.isna()] == 0.0).all()  # nosec B101


def test_values_finer_than_the_step_share_one_sample():
    index = _grid("2014-06-01", 6, 10)
    weights = valid_sample_weights(index, pd.Series(1.0, index=index), 60.0)
    assert weights.sum() == pytest.approx(1.0)  # nosec B101


def test_a_near_duplicate_pair_of_values_counts_once():
    hourly = _grid("2015-02-02", 24, 60)
    index = hourly.append(hourly + pd.Timedelta(seconds=5)).sort_values()
    weights = valid_sample_weights(index, pd.Series(1.0, index=index), 60.0)
    assert weights.iloc[:-2].sum() == pytest.approx(23.0)  # nosec B101


def test_the_index_unit_does_not_change_the_steps():
    index = _grid("2014-06-01", 6 * 24 * 3, 10).as_unit("s")
    values = pd.Series(np.where(index.minute == 0, 1.0, np.nan), index=index)
    assert column_step_minutes(index, values) == 60.0  # nosec B101
    assert sample_weights(index, 60.0).to_numpy() == pytest.approx(np.full(len(index), 1 / 6))  # nosec B101
