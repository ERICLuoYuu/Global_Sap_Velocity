"""Tests for running the comparison against a servable (serving-store) prediction.

Covers the four silent-failure traps found in the 2026-09-16 review:
  1. validity thresholds calibrated in days but applied to months
  2. reindex_like tolerance is zero on the time axis, so a period mismatch is all-NaN
  3. the preprocessed cache is keyed on filename only, so a stale run is reused
  4. a product that fails preprocessing disappears and the run still exits 0
plus the loader for the serving NetCDF cube itself.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from src.transpiration_comparison.preprocessing.alignment import (
    align_to_reference,
    assert_common_time_axis,
)
from src.transpiration_comparison.products.serving_sap_velocity import (
    ServingSapVelocityProduct,
)
from src.transpiration_comparison.provenance import (
    RunProvenance,
    assert_provenance_matches,
    write_provenance,
)

from src.transpiration_comparison.config import (
    Paths,
    SpatialDomain,
    TimePeriod,
    min_valid_timesteps,
)
from src.transpiration_comparison.preprocessing.normalize import zscore_normalize

# --------------------------------------------------------------------------
# 1. validity thresholds follow the frame, not a hard-coded day count
# --------------------------------------------------------------------------


class TestMinValidTimesteps:
    def test_thirty_six_months_reproduces_the_published_threshold(self):
        # the 2016-2018 run used 30 of 36; keep that exact mask
        assert min_valid_timesteps(36) == 30  # nosec B101

    def test_twelve_months_keeps_a_usable_threshold(self):
        assert min_valid_timesteps(12) == 10  # nosec B101

    def test_never_demands_more_steps_than_the_frame_has(self):
        assert min_valid_timesteps(4) == 4  # nosec B101


class TestZscoreDerivesMinValid:
    def test_a_twelve_step_frame_is_not_wholly_masked(self):
        """The trap: MIN_VALID_TIMESTEPS=30 masked every pixel of a one-year run."""
        time = pd.date_range("2015-01-31", periods=12, freq="ME")
        lat = np.arange(45.0, 47.0, 0.1)
        lon = np.arange(5.0, 7.0, 0.1)
        rng = np.random.default_rng(0)
        data = rng.normal(size=(12, len(lat), len(lon))).astype(np.float32)
        ds = xr.Dataset(
            {"sap_velocity": (["time", "lat", "lon"], data)},
            coords={"time": time, "lat": lat, "lon": lon},
        )

        out = zscore_normalize(ds, "sap_velocity")

        assert int(out["sap_velocity_zscore"].notnull().sum()) == data.size  # nosec B101

    def test_an_explicit_min_valid_still_wins(self):
        time = pd.date_range("2015-01-31", periods=12, freq="ME")
        ds = xr.Dataset(
            {"v": (["time", "lat", "lon"], np.ones((12, 2, 2), dtype=np.float32))},
            coords={"time": time, "lat": [1.0, 2.0], "lon": [1.0, 2.0]},
        )
        out = zscore_normalize(ds, "v", min_valid=30)
        assert int(out["v_zscore"].notnull().sum()) == 0  # nosec B101


# --------------------------------------------------------------------------
# 2. a period mismatch must raise, not silently produce NaN
# --------------------------------------------------------------------------


def _frame(start: str, periods: int, lat, lon, name="transpiration") -> xr.Dataset:
    time = pd.date_range(start, periods=periods, freq="ME")
    data = np.ones((periods, len(lat), len(lon)), dtype=np.float32)
    return xr.Dataset(
        {name: (["time", "lat", "lon"], data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )


class TestCommonTimeAxis:
    def test_a_different_period_raises_naming_the_product(self):
        lat = np.arange(45.0, 46.0, 0.1)
        lon = np.arange(5.0, 6.0, 0.1)
        datasets = {
            "sap_velocity": _frame("2015-01-31", 12, lat, lon, "sap_velocity"),
            "gleam": _frame("2016-01-31", 36, lat, lon),
        }
        with pytest.raises(ValueError, match="gleam"):
            assert_common_time_axis(datasets)

    def test_an_offset_stamp_raises(self):
        """tolerance=0.06 on a datetime axis is Timedelta(0): an hour of drift is all-NaN."""
        lat = np.arange(45.0, 46.0, 0.1)
        lon = np.arange(5.0, 6.0, 0.1)
        a = _frame("2016-01-31", 36, lat, lon, "sap_velocity")
        b = _frame("2016-01-31", 36, lat, lon)
        b = b.assign_coords(time=b.time + pd.Timedelta(hours=1))
        with pytest.raises(ValueError, match="gleam"):
            assert_common_time_axis({"sap_velocity": a, "gleam": b})

    def test_identical_axes_pass(self):
        lat = np.arange(45.0, 46.0, 0.1)
        lon = np.arange(5.0, 6.0, 0.1)
        a = _frame("2016-01-31", 36, lat, lon, "sap_velocity")
        b = _frame("2016-01-31", 36, lat, lon)
        assert_common_time_axis({"sap_velocity": a, "gleam": b})


class TestAlignToReference:
    def test_a_half_cell_offset_grid_is_not_dropped(self):
        ref_lat, ref_lon = np.arange(45.0, 46.0, 0.1), np.arange(5.0, 6.0, 0.1)
        off_lat, off_lon = ref_lat + 0.05, ref_lon + 0.05
        datasets = {
            "sap_velocity": _frame("2016-01-31", 36, ref_lat, ref_lon, "sap_velocity"),
            "gleam": _frame("2016-01-31", 36, off_lat, off_lon),
        }

        out = align_to_reference(datasets, "sap_velocity", method="nearest")

        assert np.allclose(out["gleam"].lat.values, ref_lat)  # nosec B101
        assert int(out["gleam"]["transpiration"].notnull().sum()) > 0  # nosec B101

    def test_alignment_leaves_the_time_axis_untouched(self):
        ref_lat, ref_lon = np.arange(45.0, 46.0, 0.1), np.arange(5.0, 6.0, 0.1)
        datasets = {
            "sap_velocity": _frame("2016-01-31", 36, ref_lat, ref_lon, "sap_velocity"),
            "gleam": _frame("2016-01-31", 36, ref_lat + 0.05, ref_lon + 0.05),
        }
        out = align_to_reference(datasets, "sap_velocity", method="nearest")
        assert out["gleam"].sizes["time"] == 36  # nosec B101

    def test_linear_centres_the_value_between_both_neighbours(self):
        """A half-cell offset averages the two straddling cells instead of picking one."""
        ref_lat, ref_lon = np.array([45.0, 45.1]), np.array([5.0, 5.1])
        src_lat = np.array([44.95, 45.05, 45.15])
        time = pd.date_range("2016-01-31", periods=2, freq="ME")
        column = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        values = np.array([column] * 2, dtype=np.float32)
        src = xr.Dataset(
            {"transpiration": (["time", "lat", "lon"], values)},
            coords={"time": time, "lat": src_lat, "lon": ref_lon},
        )
        ref = _frame("2016-01-31", 2, ref_lat, ref_lon, "sap_velocity")

        out = align_to_reference({"sap_velocity": ref, "gleam": src}, "sap_velocity", method="linear")

        got = out["gleam"]["transpiration"].isel(time=0, lon=0).values
        assert got[0] == pytest.approx(5.0)  # nosec B101
        assert got[1] == pytest.approx(15.0)  # nosec B101

    def test_a_dimension_already_on_the_reference_grid_is_not_interpolated(self):
        """ERA5-Land already sits on the prediction grid.

        Interpolating a matching axis is not free: scipy evaluates the
        bracketing interval even at a node, so one NaN cell spreads into its
        neighbour and every coastline loses a row for nothing.
        """
        ref_lat, ref_lon = np.array([45.0, 45.1, 45.2]), np.array([5.0, 5.1])
        time = pd.date_range("2016-01-31", periods=2, freq="ME")
        values = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
        values[:, 1, :] = np.nan  # a hole linear interpolation would spread
        src = xr.Dataset(
            {"transpiration": (["time", "lat", "lon"], values.copy())},
            coords={"time": time, "lat": ref_lat, "lon": ref_lon},
        )
        ref = _frame("2016-01-31", 2, ref_lat, ref_lon, "sap_velocity")

        out = align_to_reference({"sap_velocity": ref, "gleam": src}, "sap_velocity", method="linear")

        np.testing.assert_array_equal(out["gleam"]["transpiration"].values, values)


# --------------------------------------------------------------------------
# 3. the preprocessed cache must name the run that produced it
# --------------------------------------------------------------------------


def _prov(run_id="run_a", start="2016-01-01", end="2018-12-31"):
    return RunProvenance(
        sap_source="serving",
        run_id=run_id,
        period_start=start,
        period_end=end,
        prediction_files=("sap_velocity_run_a_2016.nc",),
    )


class TestProvenance:
    def test_a_cache_from_another_run_is_refused(self, tmp_path):
        write_provenance(tmp_path, _prov(run_id="old_model"))
        with pytest.raises(ValueError, match="old_model"):
            assert_provenance_matches(tmp_path, _prov(run_id="servable_smzp"))

    def test_a_cache_from_another_period_is_refused(self, tmp_path):
        write_provenance(tmp_path, _prov(start="2016-01-01", end="2018-12-31"))
        with pytest.raises(ValueError, match="2015"):
            assert_provenance_matches(tmp_path, _prov(start="2015-01-01", end="2015-12-31"))

    def test_the_same_run_is_accepted(self, tmp_path):
        write_provenance(tmp_path, _prov())
        assert_provenance_matches(tmp_path, _prov())

    def test_an_empty_directory_is_accepted(self, tmp_path):
        assert_provenance_matches(tmp_path, _prov())

    def test_the_sidecar_is_readable_json(self, tmp_path):
        path = write_provenance(tmp_path, _prov())
        payload = json.loads(path.read_text())
        assert payload["run_id"] == "run_a"  # nosec B101
        assert payload["sap_source"] == "serving"  # nosec B101


# --------------------------------------------------------------------------
# 4. the serving NetCDF loader
# --------------------------------------------------------------------------


@pytest.fixture
def serving_store(tmp_path):
    """A miniature serving store: two years, descending lat, masks, cells.parquet."""
    run_id = "smzp"
    pred_dir = tmp_path / "predictions" / run_id
    pred_dir.mkdir(parents=True)

    lat_desc = np.array([46.0, 45.9, 45.8, 45.7])  # descending, as the store writes it
    lon = np.array([5.0, 5.1, 5.2])

    for year in (2016, 2017):
        time = pd.date_range(f"{year}-01-01", periods=5, freq="D")
        data = np.full((len(time), len(lat_desc), len(lon)), float(year), dtype=np.float32)
        served = np.ones((len(lat_desc), len(lon)), dtype=np.uint8)
        unbuilt = np.zeros((len(lat_desc), len(lon)), dtype=np.uint8)
        unbuilt[0, 0] = 1  # one cell the trainer could not build
        served[0, 0] = 0
        data[:, 0, 0] = np.nan
        xr.Dataset(
            {
                "sap_velocity": (["time", "lat", "lon"], data),
                "served_cell": (["lat", "lon"], served),
                "unbuilt_cell": (["lat", "lon"], unbuilt),
            },
            coords={"time": time, "lat": lat_desc, "lon": lon},
            attrs={"run_id": run_id, "feature_contract": "servable"},
        ).to_netcdf(pred_dir / f"sap_velocity_{run_id}_{year}.nc")

    lats, lons = np.meshgrid(lat_desc, lon, indexing="ij")
    pd.DataFrame(
        {
            "latitude": lats.ravel(),
            "longitude": lons.ravel(),
            "pft": ["ENF", "EBF", "DBF"] * 4,
        }
    ).to_parquet(tmp_path / "cells.parquet", index=False)
    return tmp_path, run_id


def _product(store, run_id, start="2016-01-01", end="2017-12-31"):
    return ServingSapVelocityProduct(
        store=store,
        run_id=run_id,
        paths=Paths(base=store),
        period=TimePeriod(start=start, end=end),
        domain=SpatialDomain(),
    )


class TestServingLoader:
    def test_a_missing_year_raises_naming_it(self, serving_store):
        store, run_id = serving_store
        product = _product(store, run_id, start="2016-01-01", end="2018-12-31")
        with pytest.raises(FileNotFoundError, match="2018"):
            product.download()

    def test_load_returns_ascending_latitude(self, serving_store):
        store, run_id = serving_store
        ds = _product(store, run_id).load()
        assert ds.lat.values[0] < ds.lat.values[-1]  # nosec B101

    def test_load_concatenates_every_year_in_the_period(self, serving_store):
        store, run_id = serving_store
        ds = _product(store, run_id).load()
        years = set(pd.to_datetime(ds.time.values).year)
        assert years == {2016, 2017}  # nosec B101

    def test_load_honours_a_narrower_period(self, serving_store):
        store, run_id = serving_store
        ds = _product(store, run_id, start="2016-01-01", end="2016-12-31").load()
        assert set(pd.to_datetime(ds.time.values).year) == {2016}  # nosec B101

    def test_the_mask_variables_do_not_reach_the_comparison(self, serving_store):
        """zonal_mean_profile takes raw_vars[0]; a mask left in place would be plotted."""
        store, run_id = serving_store
        ds = _product(store, run_id).load()
        assert list(ds.data_vars) == ["sap_velocity"]  # nosec B101

    def test_unserved_cells_are_nan(self, serving_store):
        store, run_id = serving_store
        ds = _product(store, run_id).load()
        corner = ds["sap_velocity"].sel(lat=46.0, lon=5.0)
        assert bool(corner.isnull().all())  # nosec B101

    def test_pft_map_comes_from_cells_parquet(self, serving_store):
        store, run_id = serving_store
        pft = _product(store, run_id).load_pft_map()
        assert pft.dims == ("lat", "lon")  # nosec B101
        assert str(pft.sel(lat=45.7, lon=5.0).values) == "ENF"  # nosec B101

    def test_pft_map_latitude_matches_the_cube(self, serving_store):
        store, run_id = serving_store
        product = _product(store, run_id)
        assert np.allclose(product.load_pft_map().lat.values, product.load().lat.values)  # nosec B101


# --------------------------------------------------------------------------
# 5. a product that fails must fail the run
# --------------------------------------------------------------------------


class TestFailurePropagation:
    def test_a_failing_product_aborts_the_run(self, tmp_path):
        from src.transpiration_comparison.pipeline import TranspirationComparisonPipeline

        paths = Paths(base=tmp_path)
        paths.ensure_dirs()
        pipeline = TranspirationComparisonPipeline(paths=paths, period=TimePeriod(), domain=SpatialDomain())
        # gleam has no files anywhere under tmp_path, so its loader raises
        with pytest.raises(RuntimeError, match="gleam"):
            pipeline.preprocess_products(["gleam"])

    def test_an_allowed_product_may_be_missing(self, tmp_path):
        from src.transpiration_comparison.pipeline import TranspirationComparisonPipeline

        paths = Paths(base=tmp_path)
        paths.ensure_dirs()
        pipeline = TranspirationComparisonPipeline(paths=paths, period=TimePeriod(), domain=SpatialDomain())
        pipeline.preprocess_products(["gleam"], allow_missing=["gleam"])
