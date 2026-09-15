"""Edge cases of the servable contract: the grid's boundaries, and the day that broke the rebuild.

The unit suite proves the ordinary path. These are the inputs where a normalisation usually goes
wrong: the antimeridian, the poles, a cube that does not carry every layer, and a day whose soil
moisture never moves -- the case the retired ``rebuild_daily`` could not express at all, because it
recovered the hour count as ``raw_sum / raw_mean`` and that day's mean is zero.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_contract.merge_contract import (
    FeatureContractError,
    cell_rows,
    preflight_cells,
    resolve_site_cells,
    swc_climatology_moments,
    write_sidecar,
)
from src.feature_contract.swvl_climatology import SwvlClimatology

SWC1 = "volumetric_soil_water_layer_1"
LAYERS = (1, 2, 3, 4)


def _global_clim(sigma=0.05, mean=0.30, layers=LAYERS, dead=()):
    """A coarse whole-Earth cube: lat 89.5..-89.5, lon -179.5..179.5, so the lon axis wraps.

    ``dead`` names (i, j) cells whose sigma sits at the invalid floor, to force a fallback.
    """
    lats = np.arange(89.5, -90.0, -1.0)
    lons = np.arange(-179.5, 180.0, 1.0)
    grid = np.full((len(lats), len(lons)), float(sigma))
    for i, j in dead:
        grid[i, j] = 0.0
    return SwvlClimatology(
        lats=lats,
        lons=lons,
        sigma={layer: grid.copy() for layer in layers},
        mean={layer: np.full((len(lats), len(lons)), float(mean)) for layer in layers},
        path="/fake/global.nc",
        sha256="0" * 64,
        attrs={},
    )


class TestTheGridsBoundaries:
    def test_a_site_just_east_of_the_antimeridian_is_placed(self):
        cells = resolve_site_cells(_global_clim(), "PACIFIC", 0.0, 179.9, (1,))
        assert cells[1].cell_lon == pytest.approx(179.5)

    def test_a_site_just_west_of_it_is_placed_too(self):
        cells = resolve_site_cells(_global_clim(), "PACIFIC", 0.0, -179.9, (1,))
        assert cells[1].cell_lon == pytest.approx(-179.5)

    def test_the_search_window_crosses_the_antimeridian_rather_than_stopping_at_it(self):
        """The nearest valid cell to 179.9E may lie at 179.5W. A window that clipped at the edge of
        the array would either fail or reach for a cell thousands of km away."""
        clim = _global_clim(dead=[(89, 359)])  # kill the cell at (0.5N, 179.5E)
        cells = resolve_site_cells(clim, "PACIFIC", 0.4, 179.9, (1,))
        assert cells[1].fallback is True
        assert cells[1].distance_km < 200.0

    def test_a_polar_site_does_not_reach_across_the_pole(self):
        """Clamping at row 0 must not wrap to the far side of the planet."""
        cells = resolve_site_cells(_global_clim(), "ARCTIC", 89.9, 0.0, (1,))
        assert cells[1].cell_lat == pytest.approx(89.5)
        assert cells[1].distance_km < 100.0

    def test_two_sites_in_one_cell_get_the_same_moments(self):
        clim = _global_clim()
        a = swc_climatology_moments(resolve_site_cells(clim, "A", 0.4, 0.4, (1,)))
        b = swc_climatology_moments(resolve_site_cells(clim, "B", 0.6, 0.6, (1,)))
        assert a == b


class TestACubeThatDoesNotCarryEveryLayer:
    def test_a_missing_layer_names_the_site_and_the_layer(self):
        """KeyError from lookup_cell is a LookupError, so it must arrive as a contract failure --
        not as a bare KeyError the merge loop would swallow as an ordinary messy site."""
        shallow = _global_clim(layers=(1,))
        with pytest.raises(FeatureContractError, match="DEEP_SITE"):
            resolve_site_cells(shallow, "DEEP_SITE", 0.0, 0.0, LAYERS)

    def test_it_says_which_layer_is_missing(self):
        shallow = _global_clim(layers=(1,))
        with pytest.raises(FeatureContractError, match="layer 2"):
            resolve_site_cells(shallow, "DEEP_SITE", 0.0, 0.0, LAYERS)


class TestTheDayTheRebuildCouldNotExpress:
    """raw_mean == 0 makes n = raw_sum / raw_mean undefined; normalising hourly never needs n."""

    @staticmethod
    def _daily_from_hourly(values, mu, sigma):
        """What the merge does: normalise the HOURS, then aggregate."""
        bare = np.asarray(values) / sigma
        zscore = (np.asarray(values) - mu) / sigma
        return {
            "raw_sum": float(np.sum(values)),
            "raw_mean": float(np.mean(values)),
            "bare_sum": float(np.sum(bare)),
            "zscore_sum": float(np.sum(zscore)),
        }

    def test_a_day_of_zeroes_still_has_a_defined_zscore_sum(self):
        mu, sigma, n = 0.30, 0.05, 24
        out = self._daily_from_hourly([0.0] * n, mu, sigma)
        assert out["raw_mean"] == 0.0
        # The rebuild's route: n = raw_sum / raw_mean = 0 / 0. The warning is the finding.
        with np.errstate(invalid="ignore"):
            assert np.isnan(np.float64(out["raw_sum"]) / np.float64(out["raw_mean"]))
        # The merge's route needs no n at all.
        assert out["zscore_sum"] == pytest.approx((0.0 - n * mu) / sigma)
        assert out["zscore_sum"] == pytest.approx(-144.0)

    def test_the_identity_holds_for_an_ordinary_day_too(self):
        mu, sigma = 0.30, 0.05
        values = [0.25 + 0.001 * k for k in range(24)]
        out = self._daily_from_hourly(values, mu, sigma)
        assert out["zscore_sum"] == pytest.approx((out["raw_sum"] - 24 * mu) / sigma)

    def test_and_for_a_short_day(self):
        mu, sigma = 0.30, 0.05
        values = [0.31] * 12
        out = self._daily_from_hourly(values, mu, sigma)
        assert out["zscore_sum"] == pytest.approx((out["raw_sum"] - 12 * mu) / sigma)


class TestPreflightAtTheEdges:
    def test_no_site_placed_means_the_first_site_raises_not_two_hundred_silent_drops(self):
        """If the filename convention ever drifts, _preflight_sites returns nothing. The merge then
        raises on its first site rather than dropping every one of them into the blanket except."""
        assert preflight_cells(_global_clim(), []) == {}

    def test_preflight_keeps_every_site_it_placed(self):
        cells = preflight_cells(_global_clim(), [("A", 0.0, 0.0), ("B", 45.0, 90.0)], (1,))
        assert sorted(cells) == ["A", "B"]

    def test_a_site_with_no_cells_records_no_rows(self):
        assert cell_rows("A", 0.0, 0.0, {}) == []

    def test_and_a_sidecar_of_no_rows_is_refused(self, tmp_path):
        with pytest.raises(FeatureContractError):
            write_sidecar(tmp_path, {"name": "servable"}, cell_rows("A", 0.0, 0.0, {}))


class TestCoordinatesThatAreNotNumbers:
    @pytest.mark.parametrize("lat,lon", [("fifty", 8.0), (None, 8.0), (50.9, "east")])
    def test_they_arrive_as_a_contract_failure_naming_the_site(self, lat, lon):
        with pytest.raises(FeatureContractError, match="ODD_SITE"):
            resolve_site_cells(_global_clim(), "ODD_SITE", lat, lon, (1,))
