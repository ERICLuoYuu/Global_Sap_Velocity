"""The merge's stand age: BGI at the site's own node, then the ground st_age, then BGI borrowed from the nearest node."""

import numpy as np
import pandas as pd
import pytest

from notebooks.merge_gap_filled_hourly_orginal import (
    ground_stand_age,
    load_stand_age_map,
    load_stand_age_neighbours,
    site_stand_age,
)


def _stand_md(tmp_path, age):
    path = tmp_path / "A_stand_md.csv"
    pd.DataFrame({"st_age": [age], "st_height": [20.0]}).to_csv(path, index=False)
    return path


def test_the_own_node_comes_first(tmp_path):
    assert site_stand_age("A", {"A": 60.0}, {"A": 55.0}, _stand_md(tmp_path, 41.0)) == 60.0  # nosec B101


def test_the_ground_st_age_comes_before_a_borrowed_age(tmp_path):
    assert site_stand_age("A", {"A": np.nan}, {"A": 55.0}, _stand_md(tmp_path, 41.0)) == 41.0  # nosec B101


def test_a_borrowed_age_fills_only_where_both_are_missing(tmp_path):
    assert site_stand_age("A", {"A": np.nan}, {"A": 55.0}, tmp_path / "absent_stand_md.csv") == 55.0  # nosec B101
    assert site_stand_age("A", {}, {"A": 55.0}, _stand_md(tmp_path, np.nan)) == 55.0  # nosec B101


def test_no_age_anywhere_stays_empty(tmp_path):
    assert np.isnan(site_stand_age("A", {}, {"A": np.nan}, tmp_path / "absent_stand_md.csv"))  # nosec B101
    assert np.isnan(site_stand_age("A", {}, {}, tmp_path / "absent_stand_md.csv"))  # nosec B101


def test_ground_stand_age_reads_st_age_and_is_empty_without_it(tmp_path):
    assert ground_stand_age(_stand_md(tmp_path, "35")) == 35.0  # nosec B101
    assert np.isnan(ground_stand_age(tmp_path / "absent_stand_md.csv"))  # nosec B101
    path = tmp_path / "B_stand_md.csv"
    pd.DataFrame({"st_height": [20.0]}).to_csv(path, index=False)
    assert np.isnan(ground_stand_age(path))  # nosec B101


def test_borrowed_ages_come_from_their_own_column_and_never_enter_the_own_mean(tmp_path):
    path = tmp_path / "stand_age_data.csv"
    pd.DataFrame(
        {
            "site_name": ["A", "B"],
            "stand_age_TC000": [np.nan, 30.0],
            "stand_age_TC010": [np.nan, 50.0],
            "stand_age_neighbour": [55.0, np.nan],
        }
    ).to_csv(path, index=False)
    neighbours = load_stand_age_neighbours(path)
    own = load_stand_age_map(path)

    assert neighbours["A"] == 55.0 and np.isnan(neighbours["B"])  # nosec B101
    assert np.isnan(own["A"]) and own["B"] == 40.0  # nosec B101


def test_a_stand_age_file_without_the_borrowed_column_stops_naming_both(tmp_path):
    path = tmp_path / "stand_age_data.csv"
    pd.DataFrame({"site_name": ["A"], "stand_age_TC000": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"stand_age_data\.csv.*stand_age_neighbour"):
        load_stand_age_neighbours(path)
