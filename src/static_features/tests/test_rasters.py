from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from src.static_features.rasters import (
    PixelHit,
    centred_span,
    pixel_index,
    pixel_with_reach,
    require_global_axes,
    tiff_reader,
)
from src.static_features.results import NEIGHBOUR, NO_DATA, OK, ExtractionFailure


def _grid(points, shape=(7, 7)):
    array = np.full(shape, -1.0)
    for (row, col), value in points.items():
        array[row, col] = value
    return array


def _hit(array, row, col, reach, **kwargs):
    def read_block(row_start, row_stop, col_start, col_stop):
        return array[row_start:row_stop, col_start:col_stop]

    return pixel_with_reach(read_block, array.shape, row, col, reach, lambda block: block != -1, **kwargs)


def _assert_no_data(hit):
    assert hit.status == NO_DATA and np.isnan(hit.value) and hit.row is None and hit.col is None


def test_valid_own_pixel_is_ok_even_with_valid_neighbours():
    assert _hit(_grid({(3, 3): 5.0, (3, 4): 9.0, (2, 2): 7.0}), 3, 3, 3) == PixelHit(5.0, OK, 3, 3)


def test_nearest_beats_the_corner():
    # The CHN_YUN_YUN defect: scanning from the corner took (-1, -1) while land sat at (0, -1).
    assert _hit(_grid({(2, 2): 7.0, (3, 2): 8.0}), 3, 3, 3) == PixelHit(8.0, NEIGHBOUR, 3, 2)


def test_tie_keeps_first_in_row_then_col_order():
    assert _hit(_grid({(2, 3): 1.0, (4, 3): 2.0}), 3, 3, 1) == PixelHit(1.0, NEIGHBOUR, 2, 3)
    assert _hit(_grid({(3, 4): 1.0, (3, 2): 2.0}), 3, 3, 1) == PixelHit(2.0, NEIGHBOUR, 3, 2)


def test_reach_limits_the_search():
    array = _grid({(3, 5): 1.0})
    _assert_no_data(_hit(array, 3, 3, 1))
    assert _hit(array, 3, 3, 2) == PixelHit(1.0, NEIGHBOUR, 3, 5)


def test_reach_is_a_square_so_the_corner_counts():
    # extract_raster_value searches a square window, and unchanged products must reproduce it.
    assert _hit(_grid({(2, 2): 7.0}), 3, 3, 1) == PixelHit(7.0, NEIGHBOUR, 2, 2)


def test_wraps_the_dateline():
    west_edge = _grid({(3, 6): 4.0})
    assert _hit(west_edge, 3, 0, 1) == PixelHit(4.0, NEIGHBOUR, 3, 6)
    _assert_no_data(_hit(west_edge, 3, 0, 1, wrap_cols=False))

    east_edge = _grid({(3, 0): 4.0})
    assert _hit(east_edge, 3, 6, 1) == PixelHit(4.0, NEIGHBOUR, 3, 0)
    _assert_no_data(_hit(east_edge, 3, 6, 1, wrap_cols=False))


def test_wrapped_distance_is_measured_across_the_seam():
    # From col 6, col 0 is one pixel away and col 4 two.
    assert _hit(_grid({(3, 0): 1.0, (3, 4): 2.0}), 3, 6, 2) == PixelHit(1.0, NEIGHBOUR, 3, 0)


def test_clips_at_the_poles():
    assert _hit(_grid({(1, 3): 2.0}), 0, 3, 1) == PixelHit(2.0, NEIGHBOUR, 1, 3)
    assert _hit(_grid({(5, 3): 2.0}), 6, 3, 1) == PixelHit(2.0, NEIGHBOUR, 5, 3)


def test_rows_do_not_wrap():
    _assert_no_data(_hit(_grid({(6, 3): 2.0}), 0, 3, 1))


def test_own_pixel_outside_the_grid_is_refused():
    with pytest.raises(ValueError):
        _hit(_grid({}), 7, 3, 1)
    with pytest.raises(ValueError):
        _hit(_grid({}), 3, -1, 1)


def test_value_is_a_plain_float():
    assert type(_hit(_grid({(3, 3): 5.0}), 3, 3, 1).value) is float


def test_centred_span_hand_computed():
    assert centred_span(0.3, 1.0, 0.0, 0.25, 8) == (1, 4)  # centres .375 .625 .875
    assert centred_span(0.3, 1.0, 1.0, -0.25, 8) == (0, 3)  # descending axis, same centres
    assert centred_span(0.125, 0.375, 0.0, 0.25, 8) == (0, 1)  # low edge in, high edge out
    start, stop = centred_span(0.13, 0.14, 0.0, 0.25, 8)  # no centre inside
    assert start == stop


def test_centred_span_clips_to_the_array():
    assert centred_span(-1.0, 0.3, 0.0, 0.25, 8) == (0, 1)
    assert centred_span(1.8, 9.0, 0.0, 0.25, 8) == (7, 8)
    start, stop = centred_span(5.0, 6.0, 0.0, 0.25, 8)
    assert start == stop


def test_pixel_centred_exactly_on_a_cell_edge_belongs_to_the_cell_above_it():
    # GAMI: 1125 pixels per degree, so every fourth cell edge passes through a pixel centre.
    # Pixel 37 is centred on 90 - 4/120, the low edge of cell 3 and the high edge of cell 4.
    assert centred_span(90 - 4 / 120, 90 - 3 / 120, 90.0, -1 / 1125, 202500) == (28, 38)
    assert centred_span(90 - 5 / 120, 90 - 4 / 120, 90.0, -1 / 1125, 202500) == (38, 47)


@pytest.mark.parametrize("origin, step", [(90.0, -1 / 1125), (-180.0, 1 / 1125), (90.0, -1 / 1008)])
def test_neighbouring_cells_share_no_pixel_and_lose_none(origin, step):
    # Whatever rounding does to the edges, each pixel belongs to exactly one cell.
    size = 202500
    sign = 1 if step > 0 else -1
    spans = []
    for cell in range(4000):
        first = origin + sign * cell / 120
        second = origin + sign * (cell + 1) / 120
        spans.append(centred_span(min(first, second), max(first, second), origin, step, size))
    previous_stop = spans[0][1]
    for start, stop in spans[1:]:
        assert start == previous_stop
        previous_stop = stop


def test_tiff_reader_reads_the_requested_window_of_the_requested_band(tmp_path):
    path = tmp_path / "two_bands.tif"
    first = np.arange(16, dtype="float32").reshape(4, 4)
    profile = {
        "driver": "GTiff", "height": 4, "width": 4, "count": 2, "dtype": "float32",
        "crs": "EPSG:4326", "transform": from_origin(0, 4, 1, 1),
    }  # fmt: skip
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(first, 1)
        dst.write(first + 100, 2)

    with rasterio.open(path) as src:
        np.testing.assert_array_equal(tiff_reader(src, 1)(1, 3, 2, 4), [[6, 7], [10, 11]])
        np.testing.assert_array_equal(tiff_reader(src, 2)(0, 1, 0, 2), [[100, 101]])


def test_pixel_index_hand_computed():
    assert pixel_index(51.5, 90.0, -0.05, 3600) == 770
    assert pixel_index(10.25, -180.0, 0.05, 7200) == 3805
    assert pixel_index(-90.0, 90.0, -0.05, 3600) == 3599  # the far edge clamps into the last pixel
    assert pixel_index(180.0, -180.0, 0.05, 7200) == 7199


def test_pixel_index_refuses_a_coordinate_outside_the_grid():
    with pytest.raises(ValueError):
        pixel_index(91.0, 90.0, -0.05, 3600)
    with pytest.raises(ValueError):
        pixel_index(float("nan"), 90.0, -0.05, 3600)


def test_require_global_axes_accepts_pixel_centres():
    latitude = 90.0 - (np.arange(18) + 0.5) * 10.0
    longitude = -180.0 + (np.arange(36) + 0.5) * 10.0
    require_global_axes("p", latitude, longitude)


def test_require_global_axes_refuses_south_first():
    latitude = -90.0 + (np.arange(18) + 0.5) * 10.0
    longitude = -180.0 + (np.arange(36) + 0.5) * 10.0
    with pytest.raises(ExtractionFailure, match="p: latitude"):
        require_global_axes("p", latitude, longitude)


def test_require_global_axes_refuses_a_half_pixel_shift():
    latitude = 90.0 - (np.arange(18) + 0.5) * 10.0
    longitude = -180.0 + np.arange(36) * 10.0
    with pytest.raises(ExtractionFailure, match="p: longitude"):
        require_global_axes("p", latitude, longitude)
