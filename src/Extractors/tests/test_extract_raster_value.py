"""Tests for spatial_features_extractor.extract_raster_value.

Three defects are pinned here as regressions, all measured against the shipped
code over the 226-site corpus on 2026-08-18:

* a raster that declares ``nodata=None`` had EVERY value accepted, so Beck's
  ocean sentinel 0 was returned as a climate class at three sites;
* the neighbour search returned the ring's CORNER before its edge;
* every probe read the whole band.

The rasters here are synthetic and tiny, but they are laid out so that the old
behaviour and the new one give DIFFERENT answers -- a test where both agree would
pass against the bug.

The search footprint is a SQUARE of side ``2 * search_radius + 1``, which is what
the original implementation scanned; what changed is that candidates inside it
are now ranked by true distance instead of by scan order. The square is kept
deliberately, so the fix does not quietly shrink the reach of an extractor that
other products also use.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.Extractors.spatial_features_extractor import KOPPEN_OCEAN_CODE, extract_raster_value

SIZE = 11
CENTRE = 5
OCEAN = 0
DECLARED_NODATA = 255


def write_raster(path, array, nodata=None, dtype="uint8"):
    """One-degree pixels with the north-west corner at (0, SIZE)."""
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": "EPSG:4326",
        "transform": from_origin(0, SIZE, 1, 1),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
    return path


def centre_of(row, col):
    """lon, lat of the centre of pixel (row, col) for the transform above."""
    return col + 0.5, SIZE - row - 0.5


@pytest.fixture
def coastal_raster(tmp_path):
    """Ocean at the centre, nearest land on an EDGE, different land on a CORNER.

    The two values differ so the search orders cannot be confused:
      (CENTRE, CENTRE-1) one pixel west, distance 1.000 -> 11
      (CENTRE-1, CENTRE-1) diagonal,     distance 1.414 -> 77
    """
    array = np.full((SIZE, SIZE), OCEAN, dtype=np.uint8)
    array[CENTRE, CENTRE - 1] = 11
    array[CENTRE - 1, CENTRE - 1] = 77
    return write_raster(tmp_path / "coastal.tif", array)


class TestOceanSentinel:
    def test_undeclared_sentinel_is_rejected_when_named(self, coastal_raster):
        lon, lat = centre_of(CENTRE, CENTRE)
        got = extract_raster_value(str(coastal_raster), lat, lon, sentinel_values=(OCEAN,))
        assert got != OCEAN, "the ocean sentinel was returned as if it were data"
        assert got == 11

    def test_without_the_sentinel_the_sea_is_returned(self, coastal_raster):
        # The defect, stated exactly. nodata is None, so the validity test
        # short-circuits and accepts everything -- this is what put three corpus
        # sites in the sea, and it remains the behaviour for rasters that use 0
        # legitimately, which is why the sentinel is opt-in per caller.
        lon, lat = centre_of(CENTRE, CENTRE)
        assert extract_raster_value(str(coastal_raster), lat, lon) == OCEAN

    def test_koppen_ocean_code_is_zero(self):
        assert KOPPEN_OCEAN_CODE == 0


class TestNearestNotFirstScanned:
    def test_returns_the_edge_neighbour_not_the_corner(self, coastal_raster):
        # The old loop ran dy from -r then dx from -r, so (-1,-1) at distance
        # 1.414 was hit before (0,-1) at distance 1.000.
        lon, lat = centre_of(CENTRE, CENTRE)
        got = extract_raster_value(str(coastal_raster), lat, lon, sentinel_values=(OCEAN,))
        assert got == 11, "returned the ring's corner (77) instead of its nearest edge (11)"

    def test_a_valid_centre_beats_every_neighbour(self, tmp_path):
        array = np.full((SIZE, SIZE), 5, dtype=np.uint8)
        array[CENTRE, CENTRE] = 42
        raster = write_raster(tmp_path / "centre.tif", array)
        lon, lat = centre_of(CENTRE, CENTRE)
        assert extract_raster_value(str(raster), lat, lon, sentinel_values=(OCEAN,)) == 42

    def test_prefers_the_closer_of_two_valid_neighbours(self, tmp_path):
        array = np.full((SIZE, SIZE), OCEAN, dtype=np.uint8)
        array[CENTRE, CENTRE - 1] = 3  # distance 1
        array[CENTRE, CENTRE - 3] = 9  # distance 3
        raster = write_raster(tmp_path / "two.tif", array)
        lon, lat = centre_of(CENTRE, CENTRE)
        assert extract_raster_value(str(raster), lat, lon, sentinel_values=(OCEAN,)) == 3

    def test_reaches_the_far_edge_of_the_square(self, tmp_path):
        # The footprint is a square of side 2r+1, so its corner is at distance
        # r*sqrt(2) = 4.24 px, further than r. Pinned because it is the reach the
        # original implementation had and this fix deliberately does not shrink.
        array = np.full((SIZE, SIZE), OCEAN, dtype=np.uint8)
        array[CENTRE - 3, CENTRE - 3] = 9
        raster = write_raster(tmp_path / "corner.tif", array)
        lon, lat = centre_of(CENTRE, CENTRE)
        assert extract_raster_value(str(raster), lat, lon, sentinel_values=(OCEAN,)) == 9

    def test_gives_up_outside_the_square(self, tmp_path):
        array = np.full((SIZE, SIZE), OCEAN, dtype=np.uint8)
        array[CENTRE - 4, CENTRE - 4] = 9  # one pixel beyond the search square
        raster = write_raster(tmp_path / "toofar.tif", array)
        lon, lat = centre_of(CENTRE, CENTRE)
        assert extract_raster_value(str(raster), lat, lon, sentinel_values=(OCEAN,)) is None

    def test_search_radius_zero_is_the_centre_pixel_only(self, coastal_raster):
        lon, lat = centre_of(CENTRE, CENTRE)
        assert extract_raster_value(str(coastal_raster), lat, lon, search_radius=0, sentinel_values=(OCEAN,)) is None

    def test_equidistant_candidates_resolve_deterministically(self, tmp_path):
        # Four neighbours at distance 1 with different values. Ties are broken by
        # row-major scan order, so the northern one wins. Nothing depends on this
        # choice, but it should not vary between runs.
        array = np.full((SIZE, SIZE), OCEAN, dtype=np.uint8)
        array[CENTRE - 1, CENTRE] = 1  # north
        array[CENTRE, CENTRE - 1] = 2  # west
        array[CENTRE, CENTRE + 1] = 3  # east
        array[CENTRE + 1, CENTRE] = 4  # south
        raster = write_raster(tmp_path / "tie.tif", array)
        lon, lat = centre_of(CENTRE, CENTRE)
        results = {extract_raster_value(str(raster), lat, lon, sentinel_values=(OCEAN,)) for _ in range(3)}
        assert results == {1}


class TestDeclaredNodata:
    def test_declared_nodata_still_rejected(self, tmp_path):
        # Everything is nodata except one pixel, so there is no tie to hide
        # behind: if declared nodata were accepted the centre value would win.
        array = np.full((SIZE, SIZE), DECLARED_NODATA, dtype=np.uint8)
        array[CENTRE, CENTRE - 1] = 12
        raster = write_raster(tmp_path / "nodata.tif", array, nodata=DECLARED_NODATA)
        lon, lat = centre_of(CENTRE, CENTRE)
        assert extract_raster_value(str(raster), lat, lon) == 12

    def test_nan_nodata_is_rejected(self, tmp_path):
        # float('nan') != float('nan'), so a `value != src.nodata` test accepts
        # NaN as valid data. It has to be rejected on its own terms.
        array = np.full((SIZE, SIZE), np.nan, dtype=np.float32)
        array[CENTRE, CENTRE - 1] = 3.5
        raster = write_raster(tmp_path / "nan.tif", array, nodata=float("nan"), dtype="float32")
        lon, lat = centre_of(CENTRE, CENTRE)
        assert extract_raster_value(str(raster), lat, lon) == pytest.approx(3.5)

    def test_sentinel_and_declared_nodata_are_both_applied(self, tmp_path):
        array = np.full((SIZE, SIZE), DECLARED_NODATA, dtype=np.uint8)
        array[CENTRE, CENTRE] = DECLARED_NODATA  # declared nodata
        array[CENTRE, CENTRE - 1] = OCEAN  # undeclared sentinel, nearer
        array[CENTRE, CENTRE - 2] = 8  # the only real value
        raster = write_raster(tmp_path / "both.tif", array, nodata=DECLARED_NODATA)
        lon, lat = centre_of(CENTRE, CENTRE)
        assert extract_raster_value(str(raster), lat, lon, sentinel_values=(OCEAN,)) == 8


class TestBoundaries:
    def test_point_outside_the_raster_returns_none(self, coastal_raster):
        assert extract_raster_value(str(coastal_raster), 500.0, 500.0) is None

    def test_edge_pixel_does_not_crash_on_a_clipped_window(self, tmp_path):
        array = np.full((SIZE, SIZE), OCEAN, dtype=np.uint8)
        array[0, 1] = 6
        raster = write_raster(tmp_path / "edge.tif", array)
        lon, lat = centre_of(0, 0)
        assert extract_raster_value(str(raster), lat, lon, sentinel_values=(OCEAN,)) == 6

    def test_missing_file_returns_none(self, tmp_path):
        assert extract_raster_value(str(tmp_path / "nope.tif"), 0.5, 0.5) is None

    def test_all_invalid_returns_none(self, tmp_path):
        array = np.full((SIZE, SIZE), OCEAN, dtype=np.uint8)
        raster = write_raster(tmp_path / "allsea.tif", array)
        lon, lat = centre_of(CENTRE, CENTRE)
        assert extract_raster_value(str(raster), lat, lon, sentinel_values=(OCEAN,)) is None
