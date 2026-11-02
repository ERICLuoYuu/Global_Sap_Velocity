import numpy as np
import pandas as pd
import pytest

from src.serving.build_inputs import attach_statics, latitude_rows, within_bbox


def _cells():
    return pd.DataFrame({"row": [0, 1], "col": [2, 3], "pft": ["ENF", "DBF"]})


def test_statics_join_on_grid_row_and_col():
    statics = pd.DataFrame({"row": [1, 0], "col": [3, 2], "elevation": [100.0, 200.0]})
    joined = attach_statics(_cells(), statics)
    assert joined["elevation"].tolist() == [200.0, 100.0]
    assert joined["pft"].tolist() == ["ENF", "DBF"]
    partial = attach_statics(_cells(), statics.iloc[:1])
    assert np.isnan(partial.loc[0, "elevation"]) and partial.loc[1, "elevation"] == 100.0


def test_statics_tables_that_cannot_join_are_refused():
    with pytest.raises(ValueError):
        attach_statics(_cells(), pd.DataFrame({"row": [0], "col": [2], "sap_velocity": [1.0]}))
    with pytest.raises(KeyError):
        attach_statics(_cells(), pd.DataFrame({"row": [0], "elevation": [1.0]}))
    with pytest.raises(pd.errors.MergeError):
        attach_statics(_cells(), pd.DataFrame({"row": [0, 0], "col": [2, 2], "elevation": [1.0, 2.0]}))
    with pytest.raises(ValueError):
        attach_statics(_cells().assign(elevation=5.0), pd.DataFrame({"row": [0], "col": [2], "elevation": [1.0]}))


def test_latitude_rows_are_the_distinct_rows_in_order():
    rows, latitudes = latitude_rows(pd.DataFrame({"row": [2, 0, 2], "latitude": [49.8, 50.0, 49.8]}))
    np.testing.assert_array_equal(rows, [0, 2])
    np.testing.assert_array_equal(latitudes, [50.0, 49.8])
    with pytest.raises(ValueError):
        latitude_rows(pd.DataFrame({"row": [2, 2], "latitude": [49.8, 49.7]}))


def test_bbox_keeps_cells_inside_in_order_and_refuses_empty_or_inverted_boxes():
    cells = pd.DataFrame({"row": [0, 1, 2], "latitude": [50.0, 45.0, 10.0], "longitude": [8.0, 9.0, 8.5]})
    inside = within_bbox(cells, (5.0, 40.0, 10.0, 55.0))
    assert inside["row"].tolist() == [0, 1]
    with pytest.raises(ValueError):
        within_bbox(cells, (5.0, 60.0, 10.0, 70.0))
    with pytest.raises(ValueError):
        within_bbox(cells, (10.0, 40.0, 5.0, 55.0))
