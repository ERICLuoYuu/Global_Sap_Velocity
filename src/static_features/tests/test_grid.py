from __future__ import annotations

import pytest
from src.static_features.grid import cell_bounds, cell_of


def test_cell_of_hand_computed():
    assert cell_of(90.0, -180.0) == (0, 0)
    assert cell_of(51.5, 10.25) == (4620, 22830)
    assert cell_of(-90.0, 180.0) == (21599, 43199)  # the far edges clamp into the last cell


def test_cell_of_real_site_hand_computed():
    # GBR_DEV: (90 - 56.03333) * 120 = 4076.0004, (-3.716667 + 180) * 120 = 21153.99996
    assert cell_of(56.03333, -3.716667) == (4076, 21153)


def test_cell_bounds_inverts_cell_of():
    assert cell_bounds(4620, 22830) == pytest.approx((10.25, 51.5 - 1 / 120, 10.25 + 1 / 120, 51.5))


def test_a_coordinate_lies_inside_the_bounds_of_its_cell():
    west, south, east, north = cell_bounds(*cell_of(56.03333, -3.716667))
    assert west <= -3.716667 < east
    assert south < 56.03333 <= north


@pytest.mark.parametrize(
    "lat, lon",
    [(91.0, 0.0), (-90.5, 0.0), (0.0, 181.0), (0.0, -180.5), (float("nan"), 0.0), (0.0, float("nan"))],
)
def test_cell_of_refuses_outside_earth_and_nan(lat, lon):
    with pytest.raises(ValueError):
        cell_of(lat, lon)


@pytest.mark.parametrize("row, col", [(-1, 0), (21600, 0), (0, -1), (0, 43200)])
def test_cell_bounds_refuses_outside_the_grid(row, col):
    with pytest.raises(ValueError):
        cell_bounds(row, col)
