from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.static_features.earth_engine import ShortReply
from src.static_features.products.evi import HARMONIC_COLUMNS, composites_from_reply, seasonal_fit, summarise
from src.static_features.results import NO_DATA, OK, validate

COLUMNS = ["site_name", "date", "qa_class", "mean_evi", "pixel_count", "land_pixels", "all_pixels"]
LAND = 16
ALL = 20


def _rows(site, year, values, qa=0, count=16, land=LAND, all_pixels=ALL):
    """One row per composite of a year, one class each. Months stand in for the 16-day dates."""
    return [
        (site, f"{year}-{month:02d}-01", qa, value, count, land, all_pixels)
        for month, value in enumerate(values, start=1)
    ]


def _composites(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def _feature(site, date, groups):
    return {"properties": {"site_name": site, "date": date, "groups": groups}}


def _cell_pixels():
    return pd.DataFrame({"land_pixels": [16.0], "all_pixels": [20.0]}, index=pd.Index(["A"], name="site_name"))


def test_composite_mean_weights_classes_by_pixels():
    rows = _rows("A", 2010, [0.5] * 6, qa=0, count=6, land=8) + _rows("A", 2010, [0.3] * 6, qa=1, count=2, land=8)
    assert summarise(_composites(rows)).values.loc["A", "evi_mean"] == pytest.approx(0.45)


def test_snow_and_cloud_are_excluded():
    rows = (
        _rows("A", 2010, [0.5] * 6, qa=0, count=8)
        + _rows("A", 2010, [0.9] * 6, qa=2, count=4)
        + _rows("A", 2010, [0.1] * 6, qa=3, count=4)
    )
    assert summarise(_composites(rows)).values.loc["A", "evi_mean"] == pytest.approx(0.5)


def test_composite_needs_half_its_land_pixels():
    assert summarise(_composites(_rows("A", 2010, [0.5] * 6, count=7))).values.loc["A", "evi_years_used"] == 0
    assert summarise(_composites(_rows("A", 2010, [0.5] * 6, count=8))).values.loc["A", "evi_years_used"] == 1


def test_year_needs_six_composites():
    assert summarise(_composites(_rows("A", 2010, [0.5] * 5))).values.loc["A", "evi_years_used"] == 0
    assert summarise(_composites(_rows("A", 2010, [0.5] * 6))).values.loc["A", "evi_years_used"] == 1


def test_amplitude_is_p90_minus_p10():
    rows = _rows("A", 2010, [float(value) for value in range(11)])
    assert summarise(_composites(rows)).values.loc["A", "evi_amplitude"] == pytest.approx(8.0)


def test_amplitude_is_taken_within_each_year_then_averaged():
    # Pooled, these 22 composites would span 0 to 20; each year alone has an amplitude of 8.
    rows = _rows("A", 2010, [float(value) for value in range(11)]) + _rows(
        "A", 2011, [float(value) + 10 for value in range(11)]
    )
    assert summarise(_composites(rows)).values.loc["A", "evi_amplitude"] == pytest.approx(8.0)


def test_years_used_and_no_qualifying_year_is_no_data():
    rows = (
        _rows("A", 2010, [0.5] * 6)
        + _rows("A", 2011, [0.5] * 6)
        + _rows("A", 2012, [0.5] * 5)
        + _rows("B", 2010, [0.5] * 5)
    )
    result = summarise(_composites(rows))

    assert result.values.loc["A", "evi_years_used"] == 2
    assert result.values.loc["B", "evi_years_used"] == 0
    assert result.status.loc["B", "evi_years_used"] == OK
    assert np.isnan(result.values.loc["B", "evi_mean"])
    assert np.isnan(result.values.loc["B", "evi_amplitude"])
    assert result.status.loc["B", "evi_mean"] == NO_DATA
    assert result.status.loc["B", "evi_amplitude"] == NO_DATA
    assert result.coverage.loc["B", "evi_mean"] == 0.0
    validate(result, ["A", "B"])


def test_a_site_with_only_snow_and_cloud_is_no_data():
    rows = _rows("A", 2010, [0.9] * 6, qa=2) + _rows("A", 2010, [0.1] * 6, qa=3)
    result = summarise(_composites(rows))

    assert result.status.loc["A", "evi_mean"] == NO_DATA
    assert result.values.loc["A", "evi_years_used"] == 0
    validate(result, ["A"])


def test_summaries_average_years_not_composites():
    rows = _rows("A", 2010, [0.2] * 6) + _rows("A", 2011, [0.6] * 12)
    assert summarise(_composites(rows)).values.loc["A", "evi_mean"] == pytest.approx(0.4)


def test_summaries_ignore_years_outside_the_window():
    rows = _rows("A", 2000, [0.9] * 6) + _rows("A", 2021, [0.9] * 6) + _rows("A", 2010, [0.2] * 6)
    result = summarise(_composites(rows))

    assert result.values.loc["A", "evi_mean"] == pytest.approx(0.2)
    assert result.values.loc["A", "evi_years_used"] == 1


def test_coverage_is_valid_pixels_over_all_pixels_in_used_years():
    rows = _rows("A", 2010, [0.5] * 6, count=16, all_pixels=20) + _rows("A", 2011, [0.5] * 5, count=8, all_pixels=20)
    assert summarise(_composites(rows)).coverage.loc["A", "evi_mean"] == pytest.approx(0.8)


def test_by_year_holds_each_year_in_the_window():
    rows = _rows("A", 2010, [0.2] * 6) + _rows("A", 2011, [0.6] * 5)
    result = summarise(_composites(rows), first_year=2010, last_year=2011)
    table = result.by_year.set_index(["column", "year"])

    assert table.loc[("evi_mean", 2010), "value"] == pytest.approx(0.2)
    assert table.loc[("evi_amplitude", 2010), "status"] == OK
    assert np.isnan(table.loc[("evi_mean", 2011), "value"])
    assert table.loc[("evi_mean", 2011), "status"] == NO_DATA
    validate(result, ["A"])


def test_extras_keep_every_class_so_another_filter_can_be_recomputed():
    table = _composites(_rows("A", 2010, [0.9] * 6, qa=2))
    extras = summarise(table).extras
    assert extras["evi_composites"]["qa_class"].tolist() == [2] * 6


def test_composites_reply_scales_and_joins_the_cell_pixels():
    reply = {
        "features": [
            _feature("A", "2010-01-01", [{"qa": 0, "mean": 5000.0, "count": 6}, {"qa": 1, "mean": 3000.0, "count": 2}])
        ]
    }
    table = composites_from_reply(reply, ["A"], ["2010-01-01"], _cell_pixels())

    assert list(table.columns) == COLUMNS
    assert table["qa_class"].tolist() == [0, 1]
    assert table["mean_evi"].tolist() == pytest.approx([0.5, 0.3])
    assert table["pixel_count"].tolist() == [6, 2]
    assert table["land_pixels"].tolist() == [16, 16]
    assert table["all_pixels"].tolist() == [20, 20]


def test_composites_reply_keeps_a_composite_with_no_pixels():
    reply = {"features": [_feature("A", "2010-01-01", [])]}
    table = composites_from_reply(reply, ["A"], ["2010-01-01"], _cell_pixels())

    assert len(table) == 1
    assert np.isnan(table["qa_class"].iloc[0])
    assert np.isnan(table["mean_evi"].iloc[0])
    assert table["pixel_count"].iloc[0] == 0
    assert summarise(table).status.loc["A", "evi_mean"] == NO_DATA


def test_composites_reply_missing_a_date_is_short():
    reply = {"features": [_feature("A", "2010-01-01", [])]}
    with pytest.raises(ShortReply, match="2010-01-17"):
        composites_from_reply(reply, ["A"], ["2010-01-01", "2010-01-17"], _cell_pixels())


def test_composites_reply_with_an_unasked_date_is_short():
    reply = {"features": [_feature("A", "2010-01-01", []), _feature("A", "1999-01-01", [])]}
    with pytest.raises(ShortReply, match="1999-01-01"):
        composites_from_reply(reply, ["A"], ["2010-01-01"], _cell_pixels())


def test_composites_reply_repeating_a_composite_is_short():
    reply = {"features": [_feature("A", "2010-01-01", []), _feature("A", "2010-01-01", [])]}
    with pytest.raises(ShortReply, match="repeats"):
        composites_from_reply(reply, ["A"], ["2010-01-01"], _cell_pixels())


def test_composites_reply_without_groups_is_short():
    reply = {"features": [{"properties": {"site_name": "A", "date": "2010-01-01"}}]}
    with pytest.raises(ShortReply, match="groups"):
        composites_from_reply(reply, ["A"], ["2010-01-01"], _cell_pixels())


def test_composites_reply_with_a_class_without_a_mean_is_short():
    reply = {"features": [_feature("A", "2010-01-01", [{"qa": 0, "count": 6}])]}
    with pytest.raises(ShortReply, match="mean"):
        composites_from_reply(reply, ["A"], ["2010-01-01"], _cell_pixels())


W = 2 * np.pi / 365.25


def _seasonal_rows(site, years, level, amplitude, peak_day):
    """23 composites a year, dated by their first day as MOD13Q1 is, following one annual cosine."""
    rows = []
    for year in years:
        for date in pd.date_range(f"{year}-01-01", periods=23, freq="16D"):
            centre = date.dayofyear + 8
            rows.append((site, date.strftime("%Y-%m-%d"), 0, level + amplitude * np.cos(W * (centre - peak_day)), 16, LAND, ALL))
    return rows


def test_the_harmonic_columns():
    assert HARMONIC_COLUMNS == ("evi_season_amplitude", "evi_season_r2", "evi_peak_cos_w", "evi_peak_sin_w")


def test_seasonal_fit_recovers_a_known_cycle():
    result = summarise(_composites(_seasonal_rows("A", [2010, 2011, 2012], 0.4, 0.1, 200.0)))
    values = result.values.loc["A"]

    assert values["evi_season_amplitude"] == pytest.approx(0.1, abs=1e-9)
    assert values["evi_season_r2"] == pytest.approx(1.0, abs=1e-9)
    assert values["evi_peak_cos_w"] == pytest.approx(0.1 * np.cos(W * 200.0), abs=1e-9)
    assert values["evi_peak_sin_w"] == pytest.approx(0.1 * np.sin(W * 200.0), abs=1e-9)
    for column in HARMONIC_COLUMNS:
        assert result.status.loc["A", column] == OK
        assert result.coverage.loc["A", column] == result.coverage.loc["A", "evi_mean"]
    validate(result, ["A"])


def test_a_flat_series_has_no_season():
    fit = seasonal_fit(np.arange(1.0, 365.0, 16.0), np.full(23, 0.3))

    assert fit["evi_season_amplitude"] == pytest.approx(0.0, abs=1e-12)
    assert fit["evi_season_r2"] == 0.0
    assert fit["evi_peak_cos_w"] == pytest.approx(0.0, abs=1e-12)
    assert fit["evi_peak_sin_w"] == pytest.approx(0.0, abs=1e-12)


def test_the_peak_vector_shrinks_with_the_season():
    rows = _seasonal_rows("A", [2010], 0.4, 0.2, 100.0) + _seasonal_rows("B", [2010], 0.4, 0.02, 100.0)
    values = summarise(_composites(rows)).values

    assert values.loc["A", "evi_peak_cos_w"] == pytest.approx(10 * values.loc["B", "evi_peak_cos_w"])
    assert values.loc["A", "evi_peak_sin_w"] == pytest.approx(10 * values.loc["B", "evi_peak_sin_w"])


def test_noise_lowers_r2_but_not_the_amplitude_much():
    rows = _seasonal_rows("A", range(2001, 2021), 0.4, 0.1, 200.0)
    noise = np.random.default_rng(42).normal(scale=0.05, size=len(rows))
    rows = [(*row[:3], row[3] + noise[position], *row[4:]) for position, row in enumerate(rows)]
    values = summarise(_composites(rows)).values.loc["A"]

    assert values["evi_season_amplitude"] == pytest.approx(0.1, abs=0.01)
    assert 0.5 < values["evi_season_r2"] < 0.8


def test_only_the_composites_of_counted_years_enter_the_fit():
    sparse = _rows("A", 2011, [5.0] * 5)  # five composites: the year does not count, and must not bend the fit
    result = summarise(_composites(_seasonal_rows("A", [2010], 0.4, 0.1, 200.0) + sparse))
    assert result.values.loc["A", "evi_season_amplitude"] == pytest.approx(0.1, abs=1e-9)


def test_no_counted_year_leaves_the_harmonic_empty():
    result = summarise(_composites(_seasonal_rows("A", [2010], 0.4, 0.1, 200.0) + _rows("B", 2010, [0.5] * 5)))

    for column in HARMONIC_COLUMNS:
        assert np.isnan(result.values.loc["B", column])
        assert result.status.loc["B", column] == NO_DATA
    validate(result, ["A", "B"])


def _snowy_rows(site, years, snow_evi=0.05, seed=1):
    """Clear only from day 153 to 281, as at a boreal stand; snow covers every land pixel the rest of the year.

    Greenness is a summer bump over a flat winter, not a cosine, and swings 0.05 inside the clear window. Fitted on
    that window alone, five terms extrapolate across winter and the amplitude comes out several times too large.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for date in pd.date_range(f"{year}-01-01", periods=23, freq="16D"):
            centre, day = date.dayofyear + 8, date.strftime("%Y-%m-%d")
            if 153 <= centre <= 281:
                value = 0.30 + 0.05 * np.exp(-(((centre - 217) / 40.0) ** 2)) + rng.normal(scale=0.01)
                rows.append((site, day, 0, value, LAND, LAND, ALL))
            else:
                rows.append((site, day, 2, snow_evi, LAND, LAND, ALL))
    return rows


def _peak_day(values):
    return float(np.degrees(np.arctan2(values["evi_peak_sin_w"], values["evi_peak_cos_w"])) % 360 / 360 * 365.25)


def test_snow_composites_hold_the_fit_to_the_season_seen():
    values = summarise(_composites(_snowy_rows("A", range(2010, 2015)))).values.loc["A"]

    assert 0.0 < values["evi_season_amplitude"] <= 0.05
    assert 153 <= _peak_day(values) <= 281


def test_snow_composites_enter_at_the_dormant_level_not_their_own_reading():
    low = summarise(_composites(_snowy_rows("A", range(2010, 2015), snow_evi=0.05))).values.loc["A"]
    high = summarise(_composites(_snowy_rows("A", range(2010, 2015), snow_evi=0.9))).values.loc["A"]

    for column in HARMONIC_COLUMNS:
        assert low[column] == pytest.approx(high[column])


def test_snow_leaves_the_mean_and_amplitude_of_the_snow_free_season_alone():
    rows = _snowy_rows("A", range(2010, 2015))
    clear_only = [row for row in rows if row[2] == 0]
    with_snow, without = summarise(_composites(rows)).values.loc["A"], summarise(_composites(clear_only)).values.loc["A"]

    assert with_snow["evi_mean"] == pytest.approx(without["evi_mean"])
    assert with_snow["evi_amplitude"] == pytest.approx(without["evi_amplitude"])
