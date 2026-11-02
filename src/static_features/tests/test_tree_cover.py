from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.static_features.products.tree_cover import window_result
from src.static_features.results import NO_DATA, OK, validate

COLUMNS = ["site_name", "year", "Percent_Tree_Cover_mean", "Percent_Tree_Cover_count", "all_count"]


def _yearly(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_tree_cover_zero_counts():
    result = window_result(_yearly([("A", year, 0.0, 16, 16) for year in (2010, 2011, 2012)]))

    assert result.values.loc["A", "tree_cover_pct"] == 0.0
    assert result.status.loc["A", "tree_cover_pct"] == OK
    assert result.coverage.loc["A", "tree_cover_pct"] == 1.0


def test_tree_cover_window_mean_and_by_year():
    rows = [
        ("A", year, value, 16, 16)
        for year, value in ((2009, 90.0), (2010, 10.0), (2011, 20.0), (2012, 60.0), (2013, 90.0))
    ]
    result = window_result(_yearly(rows))

    assert result.values.loc["A", "tree_cover_pct"] == pytest.approx(30.0)
    table = result.by_year.set_index("year")
    assert sorted(table.index) == [2009, 2010, 2011, 2012, 2013]
    assert table.loc[2013, "value"] == 90.0
    assert table.loc[2009, "status"] == OK
    assert set(table["column"]) == {"tree_cover_pct"}
    validate(result, ["A"])


def test_tree_cover_year_without_pixels_is_left_out_of_the_mean_but_lowers_coverage():
    rows = [("A", 2010, np.nan, 0, 16), ("A", 2011, 20.0, 16, 16), ("A", 2012, 40.0, 8, 16)]
    result = window_result(_yearly(rows))

    assert result.values.loc["A", "tree_cover_pct"] == pytest.approx(30.0)
    assert result.coverage.loc["A", "tree_cover_pct"] == pytest.approx(0.5)
    assert result.by_year.set_index("year").loc[2010, "status"] == NO_DATA
    validate(result, ["A"])


def test_tree_cover_no_pixels_in_the_window_is_no_data_with_zero_coverage():
    rows = [("A", year, np.nan, 0, 16) for year in (2010, 2011, 2012)] + [("A", 2015, 40.0, 16, 16)]
    result = window_result(_yearly(rows))

    assert np.isnan(result.values.loc["A", "tree_cover_pct"])
    assert result.status.loc["A", "tree_cover_pct"] == NO_DATA
    assert result.coverage.loc["A", "tree_cover_pct"] == 0.0
    assert result.by_year.set_index("year").loc[2015, "value"] == 40.0
    validate(result, ["A"])


def test_tree_cover_keeps_each_site_apart_and_in_order():
    rows = [("B", year, 80.0, 16, 16) for year in (2010, 2011, 2012)] + [
        ("A", year, 20.0, 16, 16) for year in (2010, 2011, 2012)
    ]
    result = window_result(_yearly(rows))

    assert list(result.values.index) == ["B", "A"]
    assert result.values.loc["B", "tree_cover_pct"] == 80.0
    assert result.values.loc["A", "tree_cover_pct"] == 20.0
    validate(result, ["A", "B"])
