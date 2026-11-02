"""Tests for src/Extractors/terrain_aspect.py.

The two defects this module exists to remove are both invisible in summary
statistics, so they are pinned here as explicit regressions:

* averaging aspect DEGREES turns a north-facing site into a south-facing one;
* a flat pixel answers ``aspect = 0``, which reads as due north.

Pure functions only -- nothing here talks to Earth Engine.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from src.Extractors.terrain_aspect import (
    DEM_NOMINAL_SCALE_M,
    FLAT_SLOPE_DEG,
    SECTORS,
    compass_sector,
    standardize_aspect,
)

SLOPED = 12.0  # any slope comfortably above FLAT_SLOPE_DEG


class TestCompassSector:
    def test_north_spans_the_seam(self):
        assert compass_sector(355) == "N"
        assert compass_sector(0) == "N"
        assert compass_sector(5) == "N"
        assert compass_sector(359.99) == "N"

    def test_bin_edges(self):
        # N is 337.5..22.5; the edge belongs to the sector it opens.
        assert compass_sector(22.49) == "N"
        assert compass_sector(22.5) == "NE"
        assert compass_sector(337.49) == "NW"
        assert compass_sector(337.5) == "N"

    def test_cardinals(self):
        assert compass_sector(90) == "E"
        assert compass_sector(180) == "S"
        assert compass_sector(270) == "W"

    def test_wraps_and_negatives(self):
        assert compass_sector(360) == "N"
        assert compass_sector(720) == "N"
        assert compass_sector(-90) == "W"

    def test_absent_readings(self):
        assert compass_sector(None) is None
        assert compass_sector(float("nan")) is None
        assert compass_sector(np.nan) is None

    def test_every_sector_is_reachable(self):
        # 106 of 226 sites landing in one sector was the signature of the bug;
        # a correct binning must be able to produce all eight.
        produced = {compass_sector(deg) for deg in range(0, 360, 5)}
        assert produced == set(SECTORS)


class TestLinearMeanRegression:
    """The defect being fixed, stated as an executable fact."""

    def test_linear_mean_of_a_north_window_points_south_east(self):
        window = [350.0, 355.0, 0.0, 5.0, 10.0]
        # Every pixel in this window faces north...
        assert {compass_sector(deg) for deg in window} == {"N"}
        # ...yet the arithmetic mean the old extractor computed does not.
        linear_mean = sum(window) / len(window)
        assert linear_mean == pytest.approx(144.0)
        assert compass_sector(linear_mean) == "SE"

    def test_a_south_facing_window_is_unharmed(self):
        # The seam is the whole problem: away from it the linear mean is fine,
        # which is why the corpus histogram piled up in the south rather than
        # looking obviously broken.
        window = [170.0, 175.0, 180.0, 185.0, 190.0]
        assert compass_sector(sum(window) / len(window)) == "S"


class TestFlatTerrainGuard:
    def test_flat_ground_is_the_zero_vector_not_north(self):
        # Earth Engine answers aspect=0 where the gradient vanishes. Treating
        # that as a bearing invents a north-facing site.
        result = standardize_aspect(0.0, 0.0)
        assert result["aspect_sin"] == 0.0
        assert result["aspect_cos"] == 0.0
        assert result["aspect_standardized"] is None
        assert result["aspect_category"] is None

    def test_usa_wvf_regression(self):
        # USA_WVF: centre pixel dead flat, but the 250 m buffered slope is 11.5
        # degrees. A guard fed the buffered slope passes it through and records
        # aspect 0 as "N". The guard must see the centre pixel's own slope.
        centre_slope, buffered_slope, aspect = 0.0, 11.533, 0.0
        assert standardize_aspect(aspect, centre_slope)["aspect_category"] is None
        assert standardize_aspect(aspect, buffered_slope)["aspect_category"] == "N"

    def test_threshold_is_exclusive_below(self):
        just_flat = standardize_aspect(90.0, FLAT_SLOPE_DEG - 1e-9)
        just_sloped = standardize_aspect(90.0, FLAT_SLOPE_DEG + 1e-9)
        assert just_flat["aspect_category"] is None
        assert just_sloped["aspect_category"] == "E"

    def test_negative_slope_is_judged_on_magnitude(self):
        assert standardize_aspect(90.0, -0.05)["aspect_category"] is None
        assert standardize_aspect(90.0, -5.0)["aspect_category"] == "E"

    def test_missing_slope_is_flat(self):
        for absent in (None, float("nan")):
            result = standardize_aspect(90.0, absent)
            assert result["aspect_sin"] == 0.0
            assert result["aspect_cos"] == 0.0


class TestMissingAspect:
    def test_absent_reading_is_null_not_zero(self):
        # No slope means "no direction exists" -> zero vector.
        # No reading means "we do not know" -> null. These are different claims
        # and the encoding keeps them apart.
        result = standardize_aspect(None, SLOPED)
        assert result["aspect_sin"] is None
        assert result["aspect_cos"] is None
        assert result["aspect_standardized"] is None

    def test_nan_reading_is_null(self):
        assert standardize_aspect(float("nan"), SLOPED)["aspect_sin"] is None


class TestUnitVector:
    def test_cardinal_directions(self):
        north = standardize_aspect(0.0, SLOPED)
        assert north["aspect_sin"] == pytest.approx(0.0, abs=1e-12)
        assert north["aspect_cos"] == pytest.approx(1.0)

        east = standardize_aspect(90.0, SLOPED)
        assert east["aspect_sin"] == pytest.approx(1.0)
        assert east["aspect_cos"] == pytest.approx(0.0, abs=1e-12)

        south = standardize_aspect(180.0, SLOPED)
        assert south["aspect_cos"] == pytest.approx(-1.0)

    def test_always_a_unit_vector_when_a_direction_exists(self):
        for deg in range(0, 360, 7):
            result = standardize_aspect(float(deg), SLOPED)
            magnitude = math.hypot(result["aspect_sin"], result["aspect_cos"])
            assert magnitude == pytest.approx(1.0)

    def test_360_and_0_are_the_same_place(self):
        at_zero = standardize_aspect(0.0, SLOPED)
        at_360 = standardize_aspect(360.0, SLOPED)
        assert at_360["aspect_standardized"] == at_zero["aspect_standardized"] == 0.0
        assert at_360["aspect_category"] == at_zero["aspect_category"] == "N"
        assert at_360["aspect_sin"] == pytest.approx(at_zero["aspect_sin"], abs=1e-12)
        assert at_360["aspect_cos"] == pytest.approx(at_zero["aspect_cos"])

    def test_standardized_is_always_in_range(self):
        for deg in (-450.0, -1.0, 0.0, 359.9, 360.0, 725.0):
            result = standardize_aspect(deg, SLOPED)
            assert 0.0 <= result["aspect_standardized"] < 360.0


class TestSamplingConstant:
    def test_dem_scale_is_the_measured_native_scale_not_30(self):
        # ASTER GDEM is 1 arc-second. 30 m is FINER than native, and asking Earth
        # Engine for a finer grid makes it resample the elevation surface and
        # differentiate the resample: over the corpus that moved aspect at 110 of
        # 226 sites and slope at 113.
        assert DEM_NOMINAL_SCALE_M > 30.0
        assert pytest.approx(30.922081, abs=1e-6) == DEM_NOMINAL_SCALE_M
