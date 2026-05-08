"""Tests for the Fan et al. 2017 rooting-depth extractor.

Unit tests monkeypatch the dataset opener with in-memory synthetic
xr.Datasets, so no network access is required for the default test run.
A single integration test hits the live USC Santiago THREDDS server and
is gated behind the ``FAN2017_LIVE_TESTS`` environment variable.

Reference
---------
Fan Y, Miguez-Macho G, Jobbágy EG, Jackson RB, Otero-Casal C (2017).
Hydrologic regulation of plant rooting depth. PNAS 114(40):10572-10577.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.Extractors.extract_root_depth_fan2017 import (
    FEATURE_COLUMNS,
    THREDDS_DODS_BASE,
    VARIABLE_SCHEMA,
    RootDepthResult,
    _nearest_valid_pixel_indices,
    extract_root_depth_for_sites,
    fetch_all_variables_at_point,
    opendap_url,
    pick_continent,
)

# ============================================================================
# Synthetic dataset helpers
# ============================================================================


def _synthetic_continent_datasets(
    mask_holes: tuple[tuple[int, int], ...] = (),
    lat_bounds: tuple[float, float] = (45.0, 48.0),
    lon_bounds: tuple[float, float] = (10.0, 13.0),
    n_lat: int = 30,
    n_lon: int = 30,
) -> dict[str, xr.Dataset]:
    """Build an in-memory 3-file synthetic dataset for one 'continent'.

    All six Fan 2017 variables are populated with deterministic functions of
    the grid indices so tests can assert exact lookups.

    Matches the REAL Fan 2017 file sign conventions:

    * ``ETDEPTH`` and ``INFDEPTH`` are stored as **negative** z-values (depth
      below surface). The extractor applies ``sign=-1`` on read to flip them
      to positive magnitudes in the output CSV — so a synthetic raw of
      ``-2.65 m`` here becomes ``+2.65 m`` after ``fetch_all_variables_at_point``.
    * ``DD``, ``FRDD``, ``FLUSHRATE``, ``RTIME`` are stored positively.
    """
    lat = np.linspace(lat_bounds[0], lat_bounds[1], n_lat, dtype=np.float32)
    lon = np.linspace(lon_bounds[0], lon_bounds[1], n_lon, dtype=np.float32)

    mask = np.ones((n_lat, n_lon), dtype=np.int8)
    for i, j in mask_holes:
        mask[i, j] = 0

    time = np.array([0], dtype=np.int8)
    coords = {
        "time": ("time", time),
        "lat": ("lat", lat, {"units": "degrees_N"}),
        "lon": ("lon", lon, {"units": "degrees_E"}),
    }

    i_idx, j_idx = np.meshgrid(np.arange(n_lat), np.arange(n_lon), indexing="ij")
    # NEGATIVE z-values below surface, matching the real Fan 2017 file.
    etdepth = (-(1.0 + i_idx * 0.1 + j_idx * 0.01)).astype(np.float32)
    infdepth = (-(2.0 + i_idx * 0.1)).astype(np.float32)
    # Positive flux / frequency / time — stored as-is in the real file.
    dd = (0.5 + j_idx * 0.01).astype(np.float32)
    frdd = (0.1 + (i_idx + j_idx) * 0.001).astype(np.float32)
    flushrate = (1.5 + i_idx * 0.05).astype(np.float32)
    rtime = (1e6 + i_idx * 1000).astype(np.float32)

    ds_etdepth = xr.Dataset(
        data_vars={
            "ETDEPTH": (("time", "lat", "lon"), etdepth[None, :, :], {"units": "m"}),
            "mask": (("lat", "lon"), mask, {"units": "1=domain"}),
        },
        coords=coords,
    )
    ds_inf = xr.Dataset(
        data_vars={
            "INFDEPTH": (("time", "lat", "lon"), infdepth[None, :, :], {"units": "m"}),
            "DD": (("time", "lat", "lon"), dd[None, :, :], {"units": "mm/day"}),
            "FRDD": (("time", "lat", "lon"), frdd[None, :, :], {"units": "fraction"}),
            "mask": (("lat", "lon"), mask, {"units": "1=domain"}),
        },
        coords=coords,
    )
    ds_flush = xr.Dataset(
        data_vars={
            "FLUSHRATE": (("time", "lat", "lon"), flushrate[None, :, :], {"units": "mm/day"}),
            "RTIME": (("time", "lat", "lon"), rtime[None, :, :], {"units": "s"}),
            "mask": (("lat", "lon"), mask, {"units": "1=domain"}),
        },
        coords=coords,
    )
    return {
        "ETDEPTH": ds_etdepth,
        "INFDEPTH_DD_FRDD": ds_inf,
        "FLUSHRATE_RTIME": ds_flush,
    }


@pytest.fixture
def fake_opener():
    """Callable with the same signature as ``open_continent_datasets`` that
    serves synthetic in-memory datasets instead of hitting the network."""

    def _opener(continent):
        return _synthetic_continent_datasets()

    return _opener


@pytest.fixture
def fake_opener_with_center_hole():
    """Variant with a masked hole at grid index (15, 15).

    With lat_bounds=(45, 48), lon_bounds=(10, 13), and n_lat=n_lon=30, the
    pixel centre at (15, 15) lands near (46.55, 11.55), so a query at
    (46.6, 11.6) will snap to the masked hole and trigger fallback.
    """

    def _opener(continent):
        return _synthetic_continent_datasets(mask_holes=((15, 15),))

    return _opener


# ============================================================================
# Pure-function tests
# ============================================================================


@pytest.mark.unit
class TestPickContinent:
    @pytest.mark.parametrize(
        "lat,lon,expected",
        [
            (47.67, 11.45, "EURASIA"),  # DE-Hai (Hainich)
            (39.08, -96.56, "NAMERICA"),  # Konza prairie
            (-3.15, -60.00, "SAMERICA"),  # near Manaus
            (-1.07, 35.00, "AFRICA"),  # Mau Forest, Kenya
            (-35.66, 148.15, "AUSTRALIA"),  # Tumbarumba
        ],
    )
    def test_interior_sites_resolve(self, lat, lon, expected):
        assert pick_continent(lat, lon) == expected

    def test_mid_pacific_is_oob(self):
        assert pick_continent(0.0, -170.0) is None

    def test_first_match_wins_on_overlap(self):
        # (10, 10) falls in both EURASIA bbox (lat>=0, lon>=-14) and
        # AFRICA bbox (lat<40, lon<55). EURASIA comes first.
        assert pick_continent(10.0, 10.0) == "EURASIA"


@pytest.mark.unit
class TestOpendapUrl:
    @pytest.mark.parametrize(
        "continent,file_kind,tail",
        [
            ("EURASIA", "ETDEPTH", "EURASIA_ETDEPTH.nc"),
            ("NAMERICA", "INFDEPTH_DD_FRDD", "NAMERICA_INFDEPTH_DD_FRDD.nc"),
            ("AFRICA", "FLUSHRATE_RTIME", "AFRICA_FLUSHRATE_RTIME.nc"),
        ],
    )
    def test_url_shape(self, continent, file_kind, tail):
        assert opendap_url(continent, file_kind) == f"{THREDDS_DODS_BASE}/{tail}"


@pytest.mark.unit
class TestVariableSchema:
    def test_feature_columns_are_unique(self):
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))

    def test_all_six_source_variables_registered(self):
        expected = {"ETDEPTH", "INFDEPTH", "DD", "FRDD", "FLUSHRATE", "RTIME"}
        assert {spec.source_var for spec in VARIABLE_SCHEMA} == expected

    def test_root_depth_is_first_feature_column(self):
        assert FEATURE_COLUMNS[0] == "root_depth"

    def test_schema_covers_three_file_kinds(self):
        kinds = {spec.file_kind for spec in VARIABLE_SCHEMA}
        assert kinds == {"ETDEPTH", "INFDEPTH_DD_FRDD", "FLUSHRATE_RTIME"}


# ============================================================================
# Nearest-valid-pixel search
# ============================================================================


@pytest.mark.unit
class TestNearestValidPixel:
    def _grid(self):
        lat_arr = np.linspace(0.0, 9.0, 10, dtype=np.float32)
        lon_arr = np.linspace(0.0, 9.0, 10, dtype=np.float32)
        return lat_arr, lon_arr

    def test_unmasked_nearest_returns_radius_zero(self):
        mask = np.ones((10, 10), dtype=np.int8)
        lat_arr, lon_arr = self._grid()
        li, lj, valid, r = _nearest_valid_pixel_indices(mask, lat_arr, lon_arr, 5.0, 5.0, max_radius_px=3)
        assert (li, lj) == (5, 5)
        assert valid is True
        assert r == 0

    def test_single_masked_hole_falls_back_one_pixel(self):
        mask = np.ones((10, 10), dtype=np.int8)
        mask[5, 5] = 0
        lat_arr, lon_arr = self._grid()
        li, lj, valid, r = _nearest_valid_pixel_indices(mask, lat_arr, lon_arr, 5.0, 5.0, max_radius_px=3)
        assert valid is False
        assert r == 1
        assert mask[li, lj] == 1

    def test_all_masked_within_radius_returns_minus_one(self):
        mask = np.zeros((10, 10), dtype=np.int8)
        lat_arr, lon_arr = self._grid()
        _, _, valid, r = _nearest_valid_pixel_indices(mask, lat_arr, lon_arr, 5.0, 5.0, max_radius_px=3)
        assert valid is False
        assert r == -1

    def test_edge_of_grid_does_not_index_error(self):
        mask = np.ones((10, 10), dtype=np.int8)
        lat_arr, lon_arr = self._grid()
        li, lj, valid, r = _nearest_valid_pixel_indices(mask, lat_arr, lon_arr, 0.0, 0.0, max_radius_px=3)
        assert (li, lj) == (0, 0)
        assert valid is True


# ============================================================================
# Single-site fetch (all 6 variables in one shot)
# ============================================================================


@pytest.mark.unit
class TestFetchAllVariablesAtPoint:
    def test_returns_all_six_feature_columns(self, fake_opener):
        result = fetch_all_variables_at_point(
            lat=46.6,
            lon=11.6,
            site_name="synthetic",
            dataset_cache={},
            opener=fake_opener,
        )
        assert isinstance(result, RootDepthResult)
        assert set(result.values.keys()) == set(FEATURE_COLUMNS)
        for value in result.values.values():
            assert value is not None

    def test_mask_valid_when_nearest_pixel_unmasked(self, fake_opener):
        result = fetch_all_variables_at_point(
            lat=46.6,
            lon=11.6,
            site_name="good",
            dataset_cache={},
            opener=fake_opener,
        )
        assert result.mask_valid is True
        assert result.fallback_radius_px == 0
        assert result.continent == "EURASIA"

    def test_fallback_when_nearest_pixel_masked(self, fake_opener_with_center_hole):
        result = fetch_all_variables_at_point(
            lat=46.6,
            lon=11.6,
            site_name="masked",
            dataset_cache={},
            opener=fake_opener_with_center_hole,
        )
        assert result.mask_valid is False
        assert result.fallback_radius_px >= 1
        for value in result.values.values():
            assert value is not None

    def test_dataset_cache_prevents_reopen(self, fake_opener):
        open_calls: list[str] = []

        def counting_opener(continent):
            open_calls.append(continent)
            return _synthetic_continent_datasets()

        cache: dict = {}
        for _ in range(5):
            fetch_all_variables_at_point(
                lat=46.6,
                lon=11.6,
                site_name="repeat",
                dataset_cache=cache,
                opener=counting_opener,
            )
        assert open_calls == ["EURASIA"]

    def test_oob_site_returns_all_none_values(self, fake_opener):
        result = fetch_all_variables_at_point(
            lat=0.0,
            lon=-170.0,
            site_name="ghost",
            dataset_cache={},
            opener=fake_opener,
        )
        assert result.continent is None
        assert result.fallback_radius_px == -1
        assert all(v is None for v in result.values.values())

    def test_root_depth_matches_synthetic_encoding(self, fake_opener):
        # Synthetic ETDEPTH = -(1 + i*0.1 + j*0.01) — negative, matching the
        # real Fan 2017 file convention. On a 30x30 grid on (45, 48) x (10, 13),
        # the query (46.6, 11.6) snaps to index (15, 15), so raw = -2.65.
        # The extractor applies sign=-1, returning +2.65 in the output.
        result = fetch_all_variables_at_point(
            lat=46.6,
            lon=11.6,
            site_name="encoded",
            dataset_cache={},
            opener=fake_opener,
        )
        assert result.values["root_depth"] == pytest.approx(2.65, abs=0.15)

    def test_sign_convention_flips_depth_variables(self, fake_opener):
        """Lock in the sign flip: ETDEPTH/INFDEPTH get sign=-1, others not.

        Source-negative depths must become positive in the output, while
        flux / frequency / time variables must pass through unchanged.
        Without this test, a future refactor could accidentally apply sign
        to all variables or to none.
        """
        result = fetch_all_variables_at_point(
            lat=46.6,
            lon=11.6,
            site_name="sign_check",
            dataset_cache={},
            opener=fake_opener,
        )
        # Depth variables: synthetic raw is negative, module negates → positive
        assert result.values["root_depth"] > 0
        assert result.values["infiltration_depth"] > 0
        # Non-depth variables: synthetic raw is positive, module passes through
        assert result.values["deep_drainage"] > 0
        assert result.values["drainage_frequency"] > 0
        assert result.values["regolith_flush_rate"] > 0
        assert result.values["gw_residence_time"] > 0


# ============================================================================
# Batch extraction
# ============================================================================


def _write_three_site_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "site_name": ["site_a", "site_b", "site_c"],
            "lat": [46.6, 47.0, 45.8],
            "lon": [11.6, 12.0, 10.5],
        }
    ).to_csv(path, index=False)


@pytest.mark.unit
class TestBatchExtract:
    def test_fresh_run_produces_all_rows_and_columns(self, tmp_path, fake_opener):
        input_csv = tmp_path / "site_info.csv"
        output_csv = tmp_path / "root_depth.csv"
        _write_three_site_csv(input_csv)

        out_df = extract_root_depth_for_sites(
            input_csv,
            output_csv,
            flush_every=2,
            opener=fake_opener,
        )
        assert len(out_df) == 3
        for fn in FEATURE_COLUMNS:
            assert fn in out_df.columns
        for col in (
            "root_depth_mask_valid",
            "root_depth_fallback_px",
            "root_depth_continent",
            "root_depth_lat_pixel",
            "root_depth_lon_pixel",
        ):
            assert col in out_df.columns

    def test_resume_preserves_existing_values(self, tmp_path, fake_opener):
        input_csv = tmp_path / "site_info.csv"
        output_csv = tmp_path / "root_depth.csv"
        _write_three_site_csv(input_csv)

        pre_existing = pd.DataFrame(
            {
                "site_name": ["site_a"],
                "root_depth": [1.23],
                "infiltration_depth": [4.56],
                "deep_drainage": [0.78],
                "drainage_frequency": [0.12],
                "regolith_flush_rate": [1.9],
                "gw_residence_time": [5.0e6],
                "root_depth_mask_valid": [True],
                "root_depth_fallback_px": [0],
                "root_depth_continent": ["EURASIA"],
                "root_depth_lat_pixel": [46.6],
                "root_depth_lon_pixel": [11.6],
            }
        )
        pre_existing.to_csv(output_csv, index=False)

        out_df = extract_root_depth_for_sites(
            input_csv,
            output_csv,
            resume=True,
            flush_every=10,
            opener=fake_opener,
        )
        assert len(out_df) == 3
        row_a = out_df[out_df["site_name"] == "site_a"].iloc[0]
        assert row_a["root_depth"] == pytest.approx(1.23)

    def test_missing_required_column_raises(self, tmp_path, fake_opener):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"lat": [0.0], "lon": [0.0]}).to_csv(bad_csv, index=False)

        with pytest.raises(ValueError, match="missing columns"):
            extract_root_depth_for_sites(
                bad_csv,
                tmp_path / "out.csv",
                opener=fake_opener,
            )

    def test_duplicate_site_name_processed_once(self, tmp_path, fake_opener):
        """A duplicate row in input_csv must NOT produce two output rows.

        Regression guard for the done-set update: without `done.add()` inside
        the loop, the second occurrence would get fetched and appended again.
        """
        input_csv = tmp_path / "site_info.csv"
        output_csv = tmp_path / "root_depth.csv"
        pd.DataFrame(
            {
                "site_name": ["site_a", "site_a", "site_b"],
                "lat": [46.6, 46.6, 47.0],
                "lon": [11.6, 11.6, 12.0],
            }
        ).to_csv(input_csv, index=False)

        out_df = extract_root_depth_for_sites(
            input_csv,
            output_csv,
            flush_every=5,
            opener=fake_opener,
        )
        assert len(out_df) == 2
        assert sorted(out_df["site_name"].tolist()) == ["site_a", "site_b"]

    def test_all_sites_already_done_returns_without_error(self, tmp_path, fake_opener):
        """Resume path where every input site is already in the output CSV.

        Smoke test for the early-exit path; no network calls should be
        attempted because the opener is counting calls and we assert 0.
        """
        input_csv = tmp_path / "site_info.csv"
        output_csv = tmp_path / "root_depth.csv"
        _write_three_site_csv(input_csv)

        pre_existing = pd.DataFrame(
            {
                "site_name": ["site_a", "site_b", "site_c"],
                "root_depth": [1.0, 2.0, 3.0],
                "infiltration_depth": [4.0, 5.0, 6.0],
                "deep_drainage": [0.1, 0.2, 0.3],
                "drainage_frequency": [0.01, 0.02, 0.03],
                "regolith_flush_rate": [1.0, 2.0, 3.0],
                "gw_residence_time": [1.0e6, 2.0e6, 3.0e6],
                "root_depth_mask_valid": [True, True, True],
                "root_depth_fallback_px": [0, 0, 0],
                "root_depth_continent": ["EURASIA"] * 3,
                "root_depth_lat_pixel": [46.6, 47.0, 45.8],
                "root_depth_lon_pixel": [11.6, 12.0, 10.5],
            }
        )
        pre_existing.to_csv(output_csv, index=False)

        call_counter: list[str] = []

        def counting_opener(continent):
            call_counter.append(continent)
            return _synthetic_continent_datasets()

        out_df = extract_root_depth_for_sites(
            input_csv,
            output_csv,
            resume=True,
            opener=counting_opener,
        )
        assert len(out_df) == 3
        assert call_counter == []


@pytest.mark.unit
class TestFullyMaskedDataset:
    """Regression guard: a dataset with no valid pixels within max_radius_px
    must yield all-None feature values AND NaN pixel coordinates."""

    def test_fully_masked_returns_nan_pixel_and_none_values(self):
        def opener_all_masked(_continent):
            return _synthetic_continent_datasets(
                mask_holes=tuple((i, j) for i in range(30) for j in range(30)),
            )

        result = fetch_all_variables_at_point(
            lat=46.6,
            lon=11.6,
            site_name="fully_masked",
            dataset_cache={},
            max_radius_px=3,
            opener=opener_all_masked,
        )
        assert result.fallback_radius_px == -1
        assert result.mask_valid is False
        # All feature values must be None
        for value in result.values.values():
            assert value is None
        # Pixel coordinates must be NaN, not the masked nearest pixel
        assert np.isnan(result.lat_pixel)
        assert np.isnan(result.lon_pixel)


# ============================================================================
# CF decoding round-trip — guards against xarray version drift
# ============================================================================


@pytest.mark.unit
class TestCfDecodingRoundtrip:
    def test_int16_packed_value_decodes_to_meters(self, tmp_path):
        # Test-specific scale/offset — the real file's constants
        # (scale=0.01526, offset=-499.99) would require a packed value of
        # ~32929 which OVERFLOWS signed Int16 (max 32767). The real file
        # relies on Int16 values staying within a narrow range near the top
        # of the signed max, storing depths as negative z; for the roundtrip
        # test we just need to exercise xarray's CF decode path with values
        # that pack cleanly. Test constants chosen so 2.5m → packed 25.
        nc_path = tmp_path / "synth_etdepth.nc"
        scale = 0.1
        offset = 0.0
        known_depth_m = 2.5
        packed = int(round((known_depth_m - offset) / scale))
        assert -32768 <= packed <= 32767, f"Test design error: packed={packed} overflows Int16"

        raw = np.full((1, 4, 4), packed, dtype=np.int16)
        mask = np.ones((4, 4), dtype=np.int8)

        ds = xr.Dataset(
            data_vars={
                "ETDEPTH": (
                    ("time", "lat", "lon"),
                    raw,
                    {"units": "m", "scale_factor": scale, "add_offset": offset},
                ),
                "mask": (("lat", "lon"), mask),
            },
            coords={
                "time": ("time", np.array([0], dtype=np.int8)),
                "lat": ("lat", np.linspace(40.0, 50.0, 4, dtype=np.float32)),
                "lon": ("lon", np.linspace(10.0, 20.0, 4, dtype=np.float32)),
            },
        )
        ds.to_netcdf(nc_path)
        ds.close()

        reopened = xr.open_dataset(nc_path, engine="netcdf4", decode_cf=True)
        try:
            decoded = float(reopened["ETDEPTH"].isel(time=0, lat=0, lon=0).values)
        finally:
            reopened.close()

        assert decoded == pytest.approx(known_depth_m, abs=1e-3)


# ============================================================================
# Integration test — live network, gated by FAN2017_LIVE_TESTS=1
# ============================================================================


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("FAN2017_LIVE_TESTS"),
    reason="Live network test; set FAN2017_LIVE_TESTS=1 to enable.",
)
def test_de_hai_live_opendap_fetch():
    """Smoke test against the live USC Santiago THREDDS server.

    DE-Hai (Hainich, Germany; 51.08 N, 10.45 E) is a real SAPFLUXNET site.
    Its biologically plausible effective root depth for a temperate beech
    forest sits comfortably between 0.3 m and 20 m.
    """
    result = fetch_all_variables_at_point(
        lat=51.08,
        lon=10.45,
        site_name="DE-Hai",
        dataset_cache={},
    )
    assert result.continent == "EURASIA"
    assert result.mask_valid is True
    assert result.fallback_radius_px == 0
    assert result.values["root_depth"] is not None
    assert 0.3 <= result.values["root_depth"] <= 20.0
    assert result.values["deep_drainage"] is not None
    assert result.values["gw_residence_time"] is not None
