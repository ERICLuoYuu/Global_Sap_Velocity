"""What the servable contract refuses, and what it says when it does.

Every refusal here ends a run rather than a site. The merge's per-site loop catches Exception and
continues, so a contract failure that arrived as an ordinary exception would shrink the corpus by one
site and print a single line; each of these must therefore be a FeatureContractError, and each must
name the thing the operator has to go and fix.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.feature_contract.definitions import (
    ERA5_HOURLY_MEAN_PRECIP_COL,
    SERVABLE_CONTRACT,
    validate_contract,
)
from src.feature_contract.merge_contract import (
    FeatureContractError,
    apply_servable_precip,
    contract_block,
    load_cube,
    rename_gauge_columns,
    require_era5_precip,
    resolve_site_cells,
    write_sidecar,
)
from src.feature_contract.swvl_climatology import SwvlClimatology

LATS = np.array([51.0, 50.9, 50.8])
LONS = np.array([7.9, 8.0, 8.1])


def _clim(sigma=0.05, mean=0.30):
    return SwvlClimatology(
        lats=LATS,
        lons=LONS,
        sigma={1: np.full((3, 3), float(sigma))},
        mean={1: np.full((3, 3), float(mean))},
        path="/fake/cube.nc",
        sha256="0" * 64,
        attrs={},
    )


def _daily(n=2, era5=1e-4):
    return pd.DataFrame(
        {
            "TIMESTAMP": pd.date_range("2015-06-01", periods=n, freq="D", tz="UTC"),
            ERA5_HOURLY_MEAN_PRECIP_COL: era5,
            "precip": 0.125,
            "precip_min": 0.0,
            "precip_max": 1.0,
            "precip_sum": 3.0,
            "precip_coverage": 1.0,
        }
    )


class TestACubeItCannotUse:
    def test_a_directory_is_not_a_cube(self, tmp_path):
        with pytest.raises(FeatureContractError, match="not a file"):
            load_cube(tmp_path)

    def test_a_cube_without_the_sigma_variable_is_refused(self, tmp_path):
        path = tmp_path / "no_sigma.nc"
        xr.Dataset(
            {"swvl_mean": (("layer", "lat", "lon"), np.full((1, 3, 3), 0.3, dtype=np.float32))},
            coords={"layer": np.array([1], dtype=np.int8), "lat": LATS, "lon": LONS},
        ).to_netcdf(path)
        with pytest.raises(FeatureContractError, match="no_sigma"):
            load_cube(path)

    def test_a_cube_on_the_zero_to_three_sixty_grid_is_refused(self, tmp_path):
        """ERA5 ships both conventions. Normalising against the wrong one places every site in the
        wrong hemisphere without failing, so the axis is checked rather than trusted."""
        path = tmp_path / "wrong_axis.nc"
        xr.Dataset(
            {
                "swvl_sigma": (("layer", "lat", "lon"), np.full((1, 3, 3), 0.05, dtype=np.float32)),
                "swvl_mean": (("layer", "lat", "lon"), np.full((1, 3, 3), 0.3, dtype=np.float32)),
            },
            coords={"layer": np.array([1], dtype=np.int8), "lat": LATS, "lon": np.array([0.0, 180.0, 359.0])},
        ).to_netcdf(path)
        with pytest.raises(FeatureContractError, match="wrong_axis"):
            load_cube(path)


class TestASiteItCannotPlace:
    def test_a_dead_centre_cell_with_no_room_to_search_is_refused(self):
        dead = _clim(sigma=0.0)
        with pytest.raises(FeatureContractError, match="SITE_A"):
            resolve_site_cells(dead, "SITE_A", 50.9, 8.0, (1,), max_radius_cells=0)

    def test_the_message_carries_the_coordinate_that_failed(self):
        with pytest.raises(FeatureContractError, match="70"):
            resolve_site_cells(_clim(), "SITE_B", -70.0, 170.0, (1,))


class TestAnExtractItCannotUse:
    def test_an_empty_frame_is_refused(self):
        with pytest.raises(FeatureContractError, match=ERA5_HOURLY_MEAN_PRECIP_COL):
            require_era5_precip(pd.DataFrame(), "/data/extract.csv")

    def test_a_frame_of_other_variables_is_refused(self):
        frame = pd.DataFrame({"site_name": ["A"], "temperature_2m": [280.0], "u_component_of_wind_10m": [1.0]})
        with pytest.raises(FeatureContractError, match="extract.csv"):
            require_era5_precip(frame, "/data/extract.csv")


class TestPrecipOnAwkwardFrames:
    def test_a_frame_with_no_gauge_precip_at_all_still_gets_the_servable_column(self):
        """A site whose env record has no rain gauge is normal; ERA5 supplies precip regardless."""
        frame = _daily().drop(columns=["precip", "precip_min", "precip_max", "precip_sum", "precip_coverage"])
        out = apply_servable_precip(frame)
        np.testing.assert_allclose(out["precip"].to_numpy(), 2.4)

    def test_applying_it_twice_changes_nothing(self):
        """It always recomputes from the ERA5 column, so a second pass cannot compound the first --
        unlike a transform that reads its own output."""
        once = apply_servable_precip(_daily())
        twice = apply_servable_precip(once)
        np.testing.assert_allclose(twice["precip"].to_numpy(), once["precip"].to_numpy())
        assert list(twice.columns) == list(once.columns)

    def test_a_frame_with_no_rows_keeps_the_column_and_stays_empty(self):
        out = apply_servable_precip(_daily().iloc[:0])
        assert "precip" in out.columns
        assert len(out) == 0

    def test_the_gauge_rename_only_touches_what_is_there(self):
        gauge = pd.DataFrame({"precip": [1.0], "precip_sum": [2.0]})
        out = rename_gauge_columns(gauge, "precip")
        assert list(out.columns) == ["precip_gauge", "precip_gauge_sum"]


class TestABadgeItRefusesToWrite:
    def test_a_row_missing_a_field_is_refused(self, tmp_path):
        row = {"site": "A", "site_lat": 1.0, "site_lon": 2.0, "layer": 1}
        with pytest.raises(FeatureContractError, match="cells record"):
            write_sidecar(tmp_path, {"name": SERVABLE_CONTRACT}, [row])

    def test_a_row_with_an_extra_field_is_refused_too(self, tmp_path):
        cells = resolve_site_cells(_clim(), "A", 50.9, 8.0, (1,))
        from src.feature_contract.merge_contract import cell_rows

        rows = cell_rows("A", 50.9, 8.0, cells)
        rows[0]["constant_raw"] = True
        with pytest.raises(FeatureContractError):
            write_sidecar(tmp_path, {"name": SERVABLE_CONTRACT}, rows)

    def test_neither_file_survives_a_refusal(self, tmp_path):
        from src.feature_contract.merge_contract import CELLS_FILE, SIDECAR_FILE

        with pytest.raises(FeatureContractError):
            write_sidecar(tmp_path, {"name": SERVABLE_CONTRACT}, [])
        assert not (tmp_path / SIDECAR_FILE).exists()
        assert not (tmp_path / CELLS_FILE).exists()


class TestAContractNameItDoesNotKnow:
    @pytest.mark.parametrize("name", ["servible", "SERVABLE", "", "gridded", None])
    def test_an_unknown_contract_is_refused(self, name):
        with pytest.raises(ValueError, match="unknown feature contract"):
            validate_contract(name)

    def test_the_two_it_knows_pass_through(self):
        assert validate_contract("site") == "site"
        assert validate_contract(SERVABLE_CONTRACT) == SERVABLE_CONTRACT


class TestTheBlockWithoutAGitSha:
    def test_a_missing_sha_is_recorded_as_null_rather_than_invented(self):
        from src.feature_contract.merge_contract import cell_rows

        rows = cell_rows("A", 50.9, 8.0, resolve_site_cells(_clim(), "A", 50.9, 8.0, (1,)))
        assert contract_block(_clim(), (1,), 3, rows, None)["git_sha"] is None
