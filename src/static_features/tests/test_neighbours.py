from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.static_features.grid import CELLS_PER_DEGREE, N_COLS, cell_of
from src.static_features.neighbours import (
    BORROWED_COLUMNS,
    KM_PER_DEGREE,
    SEARCH_KM,
    Borrowed,
    Node,
    apply_borrowed,
    borrowed_table,
    cells_by_distance,
    fill_gaps,
    nearest_values,
    nodes_by_distance,
)
from src.static_features.results import NEIGHBOUR, NO_DATA, OK

CELL_KM = KM_PER_DEGREE / CELLS_PER_DEGREE


def _node(distance, row, col):
    return Node(distance_km=distance, row=row, col=col, lat=0.0, lon=0.0)


def test_search_reaches_25_km():
    assert SEARCH_KM == 25.0


def test_nodes_come_nearest_first_with_their_distance():
    lats, lons = np.array([2.0, 1.0, 0.0]), np.array([0.0, 0.1, 0.2])
    nodes = nodes_by_distance(0.0, 0.0, lats, lons, max_km=1000.0)

    assert (nodes[0].row, nodes[0].col) == (2, 0)
    assert nodes[0].distance_km == 0.0
    assert (nodes[1].row, nodes[1].col) == (2, 1)
    assert nodes[1].distance_km == pytest.approx(0.1 * KM_PER_DEGREE)
    assert (nodes[1].lat, nodes[1].lon) == (0.0, 0.1)
    assert [node.distance_km for node in nodes] == sorted(node.distance_km for node in nodes)


def test_nodes_beyond_the_reach_are_left_out():
    lons = np.array([0.0, 0.2, 0.3])  # 22.2 km and 33.4 km east of the site
    nodes = nodes_by_distance(0.0, 0.0, np.array([0.0]), lons)
    assert [node.col for node in nodes] == [0, 1]


def test_a_tie_goes_to_the_earlier_row_then_column():
    lats, lons = np.array([0.1, 0.0, -0.1]), np.array([-0.1, 0.0, 0.1])
    nodes = nodes_by_distance(0.0, 0.0, lats, lons, max_km=12.0)
    assert [(node.row, node.col) for node in nodes[1:]] == [(0, 1), (1, 0), (1, 2), (2, 1)]


def test_east_west_distance_shrinks_with_latitude():
    nodes = nodes_by_distance(60.0, 0.0, np.array([60.0]), np.array([0.4]))
    assert nodes[0].distance_km == pytest.approx(0.4 * KM_PER_DEGREE * 0.5)


def test_the_search_wraps_the_dateline():
    lons = np.array([-179.95, 0.0, 179.95])
    nodes = nodes_by_distance(0.0, 179.99, np.array([0.0]), lons)
    assert [node.col for node in nodes] == [2, 0]
    assert nodes[1].distance_km == pytest.approx(0.06 * KM_PER_DEGREE)


def test_cells_start_at_the_own_cell_and_stay_within_the_reach():
    lat, lon = 50.0525, 10.0525
    cells = cells_by_distance(lat, lon)

    assert (cells[0].row, cells[0].col) == cell_of(lat, lon)
    assert max(cell.distance_km for cell in cells) <= SEARCH_KM
    assert len(cells) > 2000  # about pi 25^2 / (0.93 x 0.60) cells
    assert cells[1].distance_km < CELL_KM


def test_cells_wrap_the_dateline():
    cells = cells_by_distance(0.0, 179.999, max_km=2.0)
    assert {cell.col for cell in cells} >= {0, N_COLS - 1}


def _evaluate_from(table, calls):
    def evaluate(chunk):
        calls.append([node.row for node in chunk])
        return table.iloc[[node.row for node in chunk]].reset_index(drop=True)

    return evaluate


def test_nearest_values_takes_the_first_node_holding_each_column():
    table = pd.DataFrame({"x": [np.nan, np.nan, 3.0, 4.0], "y": [np.nan, 2.0, np.nan, 5.0]})
    nodes = [_node(float(row), row, 0) for row in range(4)]
    found = nearest_values(nodes, _evaluate_from(table, []), ["x", "y"], chunk_size=10)

    assert found["x"] == Borrowed(3.0, nodes[2])
    assert found["y"] == Borrowed(2.0, nodes[1])


def test_nearest_values_stops_reading_once_every_column_has_a_value():
    table = pd.DataFrame({"x": [np.nan, 1.0, np.nan, np.nan, 9.0, 9.0]})
    nodes = [_node(float(row), row, 0) for row in range(6)]
    calls = []
    nearest_values(nodes, _evaluate_from(table, calls), ["x"], chunk_size=2)
    assert calls == [[0, 1]]


def test_nearest_values_leaves_out_a_column_no_node_holds():
    table = pd.DataFrame({"x": [np.nan, np.nan], "y": [1.0, np.nan]})
    nodes = [_node(0.0, 0, 0), _node(1.0, 1, 0)]
    assert set(nearest_values(nodes, _evaluate_from(table, []), ["x", "y"], chunk_size=1)) == {"y"}


def test_nearest_values_refuses_a_reply_of_the_wrong_length():
    nodes = [_node(0.0, 0, 0), _node(1.0, 1, 0)]
    with pytest.raises(ValueError, match="2 nodes"):
        nearest_values(nodes, lambda chunk: pd.DataFrame({"x": [1.0]}), ["x"], chunk_size=2)


def test_fill_gaps_searches_only_the_empty_columns_of_each_site():
    sites = pd.DataFrame({"site_name": ["A", "B"], "lat": [50.0525, 10.0], "lon": [10.0525, 20.0]})
    empty = pd.DataFrame({"x": [True, False], "y": [False, False]}, index=pd.Index(["A", "B"], name="site_name"))
    asked = []

    def evaluate(chunk, columns):
        asked.append(tuple(columns))
        return pd.DataFrame({column: [np.nan] + [7.0] * (len(chunk) - 1) for column in columns})

    found = fill_gaps(sites, empty, evaluate, chunk_size=50)

    assert set(found) == {"A"} and set(found["A"]) == {"x"}
    assert found["A"]["x"].value == 7.0
    assert 0 < found["A"]["x"].node.distance_km < CELL_KM
    assert asked == [("x",)]


def test_apply_borrowed_marks_neighbour_and_leaves_the_inputs_alone():
    index = pd.Index(["A", "B"], name="site_name")
    values = pd.DataFrame({"x": [np.nan, 2.0]}, index=index)
    status = pd.DataFrame({"x": [NO_DATA, OK]}, index=index)
    borrowed = {"A": {"x": Borrowed(5.0, _node(1.5, 3, 4))}}
    new_values, new_status = apply_borrowed(values, status, borrowed)

    assert new_values.loc["A", "x"] == 5.0 and new_status.loc["A", "x"] == NEIGHBOUR
    assert new_status.loc["B", "x"] == OK
    assert np.isnan(values.loc["A", "x"]) and status.loc["A", "x"] == NO_DATA


def test_borrowed_table_says_how_far_each_value_came():
    node = Node(distance_km=1.25, row=3, col=4, lat=10.5, lon=20.5)
    table = borrowed_table({"A": {"x": Borrowed(5.0, node)}})

    assert list(table.columns) == BORROWED_COLUMNS
    assert table.iloc[0].tolist() == ["A", "x", 1.25, 10.5, 20.5]
    assert list(borrowed_table({}).columns) == BORROWED_COLUMNS
