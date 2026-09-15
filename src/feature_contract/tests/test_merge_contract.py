"""The servable feature contract as the merge applies it.

``src.feature_contract.merge_contract`` holds every decision the Stage-4 merge needs to emit a
servable tree in one pass: which climatology cell a site normalises against, how the gauge columns
step aside, and what the tree records about itself. The merge itself only calls these.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.feature_contract.definitions import (
    ERA5_HOURLY_MEAN_PRECIP_COL,
    SERVABLE_CONTRACT,
    SERVABLE_RAIN_EVENT_MM_PER_DAY,
)
from src.feature_contract.merge_contract import (
    CELL_FIELDS,
    CELLS_FILE,
    DEFAULT_SWVL_CUBE,
    SIDECAR_FILE,
    FeatureContractError,
    apply_servable_precip,
    cell_rows,
    contract_block,
    load_cube,
    preflight_cells,
    rename_gauge_columns,
    require_era5_precip,
    resolve_site_cells,
    swc_climatology_moments,
    write_sidecar,
)
from src.feature_contract.swvl_climatology import MIN_VALID_SIGMA, SwvlClimatology

SWC1 = "volumetric_soil_water_layer_1"
LAYERS = (1, 2, 3, 4)


def _clim(sigma=0.05, mean=0.30):
    """A 3x3 climatology centred on (50.9, 8.0), every cell identical."""
    lats = np.array([51.0, 50.9, 50.8])
    lons = np.array([7.9, 8.0, 8.1])
    grid = np.full((3, 3), float(sigma))
    means = np.full((3, 3), float(mean))
    return SwvlClimatology(
        lats=lats,
        lons=lons,
        sigma={layer: grid.copy() for layer in LAYERS},
        mean={layer: means.copy() for layer in LAYERS},
        path="/fake/swvl_cube.nc",
        sha256="0" * 64,
        attrs={"source": "unit test"},
    )


def _daily(n=3, era5_mean_m_per_hour=1e-4):
    """A merge-shaped daily frame: the ERA5 column, plus the gauge statistics the merge starts with."""
    return pd.DataFrame(
        {
            "TIMESTAMP": pd.date_range("2015-06-01", periods=n, freq="D", tz="UTC"),
            ERA5_HOURLY_MEAN_PRECIP_COL: era5_mean_m_per_hour,
            "total_precipitation_hourly_sum": era5_mean_m_per_hour * 24,
            "precip": 0.125,
            "precip_min": 0.0,
            "precip_max": 1.0,
            "precip_sum": 3.0,
            "precip_coverage": 1.0,
        }
    )


class TestCubeLoading:
    def test_a_missing_cube_names_the_path(self, tmp_path):
        missing = tmp_path / "not_here.nc"
        with pytest.raises(FeatureContractError, match=str(missing.name)):
            load_cube(missing)

    def test_an_unreadable_cube_names_the_path(self, tmp_path):
        broken = tmp_path / "broken.nc"
        broken.write_text("this is not netcdf")
        with pytest.raises(FeatureContractError, match=str(broken.name)):
            load_cube(broken)

    def test_the_default_cube_is_the_v2_climatology(self):
        assert str(DEFAULT_SWVL_CUBE).endswith("swvl_sigma_climatology_v2_1995_2019.nc")


class TestCellResolution:
    def test_one_cell_per_layer(self):
        cells = resolve_site_cells(_clim(), "SITE_A", 50.9, 8.0, LAYERS)
        assert sorted(cells) == list(LAYERS)
        assert cells[1].sigma == pytest.approx(0.05)
        assert cells[1].mean == pytest.approx(0.30)
        assert cells[1].fallback is False

    def test_no_valid_cell_names_the_site_and_the_layer(self):
        dead = _clim(sigma=0.0)
        with pytest.raises(FeatureContractError, match="SITE_A"):
            resolve_site_cells(dead, "SITE_A", 50.9, 8.0, LAYERS)

    def test_a_site_off_the_grid_names_the_site(self):
        with pytest.raises(FeatureContractError, match="SITE_B"):
            resolve_site_cells(_clim(), "SITE_B", -70.0, 170.0, LAYERS)

    def test_a_site_without_a_coordinate_names_the_site(self):
        with pytest.raises(FeatureContractError, match="SITE_C"):
            resolve_site_cells(_clim(), "SITE_C", float("nan"), 8.0, LAYERS)

    def test_preflight_resolves_every_site(self):
        cells = preflight_cells(_clim(), [("A", 50.9, 8.0), ("B", 51.0, 7.9)], LAYERS)
        assert sorted(cells) == ["A", "B"]
        assert sorted(cells["A"]) == list(LAYERS)

    def test_preflight_raises_on_the_first_site_it_cannot_place(self):
        with pytest.raises(FeatureContractError, match="B"):
            preflight_cells(_clim(), [("A", 50.9, 8.0), ("B", -70.0, 170.0)], LAYERS)

    def test_sigma_exactly_at_the_floor_is_valid(self):
        cells = resolve_site_cells(_clim(sigma=MIN_VALID_SIGMA), "SITE_A", 50.9, 8.0, (1,))
        assert cells[1].sigma == pytest.approx(MIN_VALID_SIGMA)


class TestMomentsHandover:
    def test_the_moments_are_the_mapping_add_swc_representations_takes(self):
        moments = swc_climatology_moments(resolve_site_cells(_clim(), "A", 50.9, 8.0, LAYERS))
        assert set(moments) == {f"volumetric_soil_water_layer_{layer}" for layer in LAYERS}
        assert moments[SWC1] == (pytest.approx(0.30), pytest.approx(0.05))

    def test_the_order_is_mean_then_sigma(self):
        """add_swc_representations unpacks (mean, std); swapping them would silently invert the z-score."""
        clim = _clim(sigma=0.05, mean=0.30)
        mean, sigma = swc_climatology_moments(resolve_site_cells(clim, "A", 50.9, 8.0, (1,)))[SWC1]
        assert mean > sigma


class TestCellsRecord:
    def test_the_fields_are_the_ones_rebuild_daily_writes(self):
        rows = cell_rows("SITE_A", 50.9, 8.0, resolve_site_cells(_clim(), "SITE_A", 50.9, 8.0, LAYERS))
        assert len(rows) == len(LAYERS)
        assert list(rows[0]) == list(CELL_FIELDS)
        assert rows[0]["site"] == "SITE_A"
        assert rows[0]["site_lat"] == pytest.approx(50.9)

    def test_one_row_per_layer_in_layer_order(self):
        rows = cell_rows("SITE_A", 50.9, 8.0, resolve_site_cells(_clim(), "SITE_A", 50.9, 8.0, LAYERS))
        assert [row["layer"] for row in rows] == list(LAYERS)


class TestGaugeColumnsStepAside:
    def test_every_gauge_statistic_takes_the_gauge_name(self):
        gauge = pd.DataFrame(
            {name: [1.0] for name in ("precip", "precip_min", "precip_max", "precip_sum", "precip_coverage")}
        )
        out = rename_gauge_columns(gauge, "precip")
        assert list(out.columns) == [
            "precip_gauge",
            "precip_gauge_min",
            "precip_gauge_max",
            "precip_gauge_sum",
            "precip_gauge_coverage",
        ]

    def test_the_input_is_not_modified(self):
        gauge = pd.DataFrame({"precip": [1.0]})
        rename_gauge_columns(gauge, "precip")
        assert list(gauge.columns) == ["precip"]


class TestServablePrecip:
    def test_precip_is_the_era5_daily_total_in_mm(self):
        out = apply_servable_precip(_daily())
        np.testing.assert_allclose(out["precip"].to_numpy(), 1e-4 * 24 * 1000)

    def test_precip_agrees_with_the_era5_daily_sum(self):
        out = apply_servable_precip(_daily())
        np.testing.assert_allclose(out["precip"].to_numpy(), out["total_precipitation_hourly_sum"].to_numpy() * 1000)

    def test_a_tiny_negative_reanalysis_value_becomes_zero(self):
        out = apply_servable_precip(_daily(era5_mean_m_per_hour=-1e-12))
        assert (out["precip"] == 0.0).all()

    def test_a_day_without_era5_hours_stays_missing(self):
        frame = _daily()
        frame.loc[1, ERA5_HOURLY_MEAN_PRECIP_COL] = np.nan
        out = apply_servable_precip(frame)
        assert np.isnan(out["precip"].iloc[1])

    def test_the_stale_gauge_statistics_are_dropped(self):
        out = apply_servable_precip(_daily())
        assert not {"precip_min", "precip_max", "precip_sum"} & set(out.columns)

    def test_precip_keeps_its_column_position(self):
        frame = _daily()
        before = [c for c in frame.columns if c not in ("precip_min", "precip_max", "precip_sum")]
        assert list(apply_servable_precip(frame).columns) == before

    def test_without_the_era5_column_it_refuses_rather_than_keeping_the_gauge(self):
        frame = _daily().drop(columns=[ERA5_HOURLY_MEAN_PRECIP_COL])
        with pytest.raises(FeatureContractError, match=ERA5_HOURLY_MEAN_PRECIP_COL):
            apply_servable_precip(frame)

    def test_the_input_is_not_modified(self):
        frame = _daily()
        apply_servable_precip(frame)
        assert frame["precip"].iloc[0] == pytest.approx(0.125)
        assert "precip_sum" in frame.columns


class TestExtractPreflight:
    """The ERA5 column is a property of the extract, so it is settled once, before any site."""

    def test_an_extract_that_carries_the_column_passes(self):
        require_era5_precip(pd.DataFrame({"site_name": ["A"], ERA5_HOURLY_MEAN_PRECIP_COL: [1e-4]}), "/x.csv")

    def test_an_extract_without_it_names_the_column_and_the_file(self):
        frame = pd.DataFrame({"site_name": ["A"], "temperature_2m": [280.0]})
        with pytest.raises(FeatureContractError, match=ERA5_HOURLY_MEAN_PRECIP_COL):
            require_era5_precip(frame, "/data/era5_extracted_data.csv")
        with pytest.raises(FeatureContractError, match="era5_extracted_data"):
            require_era5_precip(frame, "/data/era5_extracted_data.csv")


class TestContractBlock:
    @staticmethod
    def _rows(fallback=0):
        rows = cell_rows("A", 50.9, 8.0, resolve_site_cells(_clim(), "A", 50.9, 8.0, LAYERS))
        for k in range(fallback):
            rows[k]["fallback"] = True
        return rows

    def test_it_names_the_servable_contract_and_the_cube(self):
        block = contract_block(_clim(), LAYERS, 3, self._rows(), "abc1234")
        assert block["name"] == SERVABLE_CONTRACT
        assert block["swvl"]["cube"] == "/fake/swvl_cube.nc"
        assert block["swvl"]["cube_sha256"] == "0" * 64
        assert block["git_sha"] == "abc1234"

    def test_it_counts_the_fallback_cells(self):
        assert contract_block(_clim(), LAYERS, 3, self._rows(fallback=2), None)["swvl"]["n_fallback_cells"] == 2

    def test_it_counts_the_sites_the_badge_covers(self):
        """A layer is one row, so counting rows would report four sites for one."""
        rows = self._rows() + cell_rows("B", 51.0, 7.9, resolve_site_cells(_clim(), "B", 51.0, 7.9, LAYERS))
        assert contract_block(_clim(), LAYERS, 3, rows, None)["swvl"]["n_sites"] == 2

    def test_the_precip_memory_block_describes_mm_per_day(self):
        memory = contract_block(_clim(), LAYERS, 3, self._rows(), None)["precip_memory"]
        assert memory["source_column"] == ERA5_HOURLY_MEAN_PRECIP_COL
        assert memory["rain_event_mm_per_day"] == SERVABLE_RAIN_EVENT_MM_PER_DAY
        assert "mm" in memory["units"]

    def test_it_agrees_with_rebuild_daily_on_every_key_a_gate_reads(self):
        """The two builders describe different constructions; the gates must accept either."""
        from pathlib import Path

        from src.feature_contract.rebuild_daily import _contract_block

        rows = self._rows(fallback=1)
        mine = contract_block(_clim(), LAYERS, 3, rows, "abc1234")
        theirs = _contract_block(_clim(), Path("/src/tree"), LAYERS, 3, rows, "abc1234")
        assert mine["name"] == theirs["name"]
        assert mine["swvl"]["definition"] == theirs["swvl"]["definition"]
        assert mine["swvl"]["n_fallback_cells"] == theirs["swvl"]["n_fallback_cells"]
        assert mine["swvl"]["min_valid_sigma"] == theirs["swvl"]["min_valid_sigma"]
        assert mine["swvl"]["clip"] == theirs["swvl"]["clip"]
        assert mine["precip_memory"]["source_column"] == theirs["precip_memory"]["source_column"]
        assert mine["precip_memory"]["rain_event_mm_per_day"] == theirs["precip_memory"]["rain_event_mm_per_day"]


class TestSidecar:
    def test_it_writes_the_block_and_the_cells_beside_the_daily_directory(self, tmp_path):
        rows = cell_rows("A", 50.9, 8.0, resolve_site_cells(_clim(), "A", 50.9, 8.0, LAYERS))
        block = contract_block(_clim(), LAYERS, 3, rows, None)
        write_sidecar(tmp_path, block, rows)

        with open(tmp_path / SIDECAR_FILE) as fh:
            assert json.load(fh)["name"] == SERVABLE_CONTRACT
        cells = pd.read_csv(tmp_path / CELLS_FILE)
        assert list(cells.columns) == list(CELL_FIELDS)
        assert len(cells) == len(LAYERS)

    def test_a_tree_that_retained_no_site_is_not_labelled_servable(self, tmp_path):
        """The badge is what every downstream gate reads; it must not outlive the data it describes."""
        with pytest.raises(FeatureContractError, match="servable"):
            write_sidecar(tmp_path, contract_block(_clim(), LAYERS, 3, [], None), [])
        assert not (tmp_path / SIDECAR_FILE).exists()
        assert not (tmp_path / CELLS_FILE).exists()

    def test_a_row_that_does_not_match_the_fields_is_refused(self, tmp_path):
        with pytest.raises(FeatureContractError):
            write_sidecar(tmp_path, {"name": SERVABLE_CONTRACT}, [{"site": "A"}])
