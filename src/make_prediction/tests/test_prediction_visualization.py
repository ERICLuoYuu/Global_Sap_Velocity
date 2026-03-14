"""Unit tests for prediction_visualization_hpc module."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
import rasterio

from rasterio.crs import CRS

from src.make_prediction.prediction_visualization_hpc import (
    _sanitize_filename,
    _write_geotiff,
    build_parser,
    compute_composites,
    discover_timestamps,
    get_valid_block_size,
    rasterize_all_timestamps,
    rasterize_timestamp,
    read_prediction_file,
    run_batch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path: Path) -> str:
    """Create a small sample prediction CSV for testing.

    Returns
    -------
    str
        Path to the CSV file.
    """
    np.random.seed(42)
    n_pixels = 50
    timestamps = ["2015-07-01 12:00:00+00:00", "2015-07-01 13:00:00+00:00"]
    rows: list = []
    for ts in timestamps:
        for _ in range(n_pixels):
            lat = round(np.random.uniform(-60, 78), 1)
            lon = round(np.random.uniform(-180, 180), 1)
            rows.append({
                "timestamp.1": ts,
                "latitude": lat,
                "longitude": lon,
                "sw_in": np.random.uniform(0, 400),
                "sap_velocity_cnn_lstm": np.random.uniform(0, 5),
            })
    df = pd.DataFrame(rows)
    csv_path = str(tmp_path / "test_predictions.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_geotiffs(tmp_path: Path) -> List[str]:
    """Create small sample GeoTIFFs for composite testing.

    Returns
    -------
    list of str
        Paths to the GeoTIFF files.
    """
    tif_paths: list = []
    np.random.seed(99)
    h, w = 10, 20
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": rasterio.float32,
        "crs": "+proj=longlat +datum=WGS84 +no_defs",
        "transform": rasterio.transform.from_origin(-180, 78, 0.1, 0.1),
        "nodata": np.nan,
    }
    for i in range(3):
        data = np.random.uniform(0, 5, (h, w)).astype(np.float32)
        # Add some NaN pixels
        data[0, 0] = np.nan
        path = str(tmp_path / f"ts_{i}.tif")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
        tif_paths.append(path)
    return tif_paths


# ---------------------------------------------------------------------------
# Tests: get_valid_block_size
# ---------------------------------------------------------------------------

class TestGetValidBlockSize:
    """Tests for GeoTIFF block size calculation."""

    def test_large_dimension_returns_preferred(self) -> None:
        """Dimensions >= preferred should return preferred block size."""
        assert get_valid_block_size(1000) == 256

    def test_small_dimension_rounds_down(self) -> None:
        """Dimensions < preferred should round down to nearest multiple."""
        assert get_valid_block_size(100) == 96  # 100 // 16 * 16
        assert get_valid_block_size(48) == 48

    def test_very_small_dimension_returns_minimum(self) -> None:
        """Very small dimensions should return the minimum multiple."""
        assert get_valid_block_size(10) == 16

    def test_zero_dimension_returns_minimum(self) -> None:
        """Zero dimension should return the block_multiple."""
        assert get_valid_block_size(0) == 16

    def test_negative_dimension_returns_minimum(self) -> None:
        """Negative dimension should return block_multiple."""
        assert get_valid_block_size(-5) == 16

    def test_custom_preferred_and_multiple(self) -> None:
        """Custom preferred_block_size and block_multiple work correctly."""
        # 500 < 512 so returns (500 // 32) * 32 = 480
        assert get_valid_block_size(500, preferred_block_size=512, block_multiple=32) == 480
        assert get_valid_block_size(600, preferred_block_size=512, block_multiple=32) == 512
        assert get_valid_block_size(200, preferred_block_size=512, block_multiple=32) == 192


# ---------------------------------------------------------------------------
# Tests: discover_timestamps
# ---------------------------------------------------------------------------

class TestDiscoverTimestamps:
    """Tests for timestamp discovery from CSV files."""

    def test_finds_all_timestamps(self, sample_csv: str) -> None:
        """Should discover all unique timestamps."""
        ts_list = discover_timestamps(sample_csv)
        assert len(ts_list) == 2
        assert "2015-07-01 12:00:00+00:00" in ts_list
        assert "2015-07-01 13:00:00+00:00" in ts_list

    def test_returns_sorted(self, sample_csv: str) -> None:
        """Timestamps should be sorted."""
        ts_list = discover_timestamps(sample_csv)
        assert ts_list == sorted(ts_list)

    def test_empty_csv_returns_empty(self, tmp_path: Path) -> None:
        """Empty CSV should return empty list."""
        csv_path = str(tmp_path / "empty.csv")
        pd.DataFrame({"timestamp.1": []}).to_csv(csv_path, index=False)
        ts_list = discover_timestamps(csv_path)
        assert ts_list == []


# ---------------------------------------------------------------------------
# Tests: _write_geotiff
# ---------------------------------------------------------------------------

class TestWriteGeotiff:
    """Tests for the core GeoTIFF writing function."""

    def test_creates_valid_geotiff(self, tmp_path: Path) -> None:
        """Should create a readable GeoTIFF with correct dimensions."""
        df = pd.DataFrame({
            "latitude": [0.0, 0.1, 0.2],
            "longitude": [10.0, 10.1, 10.2],
            "value": [1.0, 2.0, 3.0],
        })
        tif_path = str(tmp_path / "test.tif")
        _write_geotiff(df, tif_path, "value", resolution=0.1)

        assert os.path.exists(tif_path)
        with rasterio.open(tif_path) as src:
            assert src.count == 1
            assert src.crs is not None
            data = src.read(1)
            assert not np.all(np.isnan(data))

    def test_handles_nan_values(self, tmp_path: Path) -> None:
        """NaN values should be preserved as nodata."""
        df = pd.DataFrame({
            "latitude": [0.0, 0.1],
            "longitude": [10.0, 10.1],
            "value": [1.0, np.nan],
        })
        tif_path = str(tmp_path / "nan_test.tif")
        _write_geotiff(df, tif_path, "value", resolution=0.1)

        with rasterio.open(tif_path) as src:
            data = src.read(1)
            valid_count = np.sum(~np.isnan(data))
            assert valid_count == 1

    def test_custom_extent(self, tmp_path: Path) -> None:
        """Should respect custom lat/lon range."""
        df = pd.DataFrame({
            "latitude": [10.0, 10.5],
            "longitude": [20.0, 20.5],
            "value": [1.0, 2.0],
        })
        tif_path = str(tmp_path / "extent_test.tif")
        _write_geotiff(
            df, tif_path, "value", resolution=0.1,
            lat_range=(9.0, 11.0), lon_range=(19.0, 21.0),
        )

        with rasterio.open(tif_path) as src:
            assert src.bounds.left == pytest.approx(19.0)
            assert src.bounds.top == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# Tests: rasterize_timestamp
# ---------------------------------------------------------------------------

class TestRasterizeTimestamp:
    """Tests for per-timestamp rasterization."""

    def test_produces_geotiff(self, sample_csv: str, tmp_path: Path) -> None:
        """Should produce a valid GeoTIFF for a known timestamp."""
        tif_path = str(tmp_path / "ts_output.tif")
        result = rasterize_timestamp(
            csv_path=sample_csv,
            timestamp_value="2015-07-01 12:00:00+00:00",
            output_tif=tif_path,
            sw_in_threshold=0,  # Disable to ensure data passes
            resolution=1.0,  # Coarse for speed
        )
        assert result is not None
        assert os.path.exists(result)

    def test_missing_timestamp_returns_none(
        self, sample_csv: str, tmp_path: Path,
    ) -> None:
        """Non-existent timestamp should return None."""
        tif_path = str(tmp_path / "missing_ts.tif")
        result = rasterize_timestamp(
            csv_path=sample_csv,
            timestamp_value="2099-01-01 00:00:00+00:00",
            output_tif=tif_path,
            sw_in_threshold=0,
            resolution=1.0,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Tests: compute_composites
# ---------------------------------------------------------------------------

class TestComputeComposites:
    """Tests for composite map generation."""

    def test_produces_all_stats(
        self, sample_geotiffs: List[str], tmp_path: Path,
    ) -> None:
        """Should produce one GeoTIFF per requested stat."""
        out_dir = str(tmp_path / "composites")
        results = compute_composites(
            tif_paths=sample_geotiffs,
            output_dir=out_dir,
            stats=("mean", "median", "max", "count"),
            render_png=False,
        )
        assert len(results) == 4
        for stat, path in results.items():
            assert os.path.exists(path)
            with rasterio.open(path) as src:
                data = src.read(1)
                assert data.shape == (10, 20)

    def test_count_composite_values(
        self, sample_geotiffs: List[str], tmp_path: Path,
    ) -> None:
        """Count composite should have correct values."""
        out_dir = str(tmp_path / "composites_count")
        results = compute_composites(
            tif_paths=sample_geotiffs,
            output_dir=out_dir,
            stats=("count",),
            render_png=False,
        )
        with rasterio.open(results["count"]) as src:
            data = src.read(1)
            # Pixel (0, 0) has NaN in all 3 files → count = 0
            assert data[0, 0] == pytest.approx(0.0)
            # Most pixels should have count == 3
            assert np.max(data) == pytest.approx(3.0)

    def test_empty_input_returns_empty(self, tmp_path: Path) -> None:
        """Empty input list should return empty dict."""
        out_dir = str(tmp_path / "empty_composites")
        results = compute_composites(
            tif_paths=[],
            output_dir=out_dir,
            render_png=False,
        )
        assert results == {}


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLI:
    """Tests for argparse CLI configuration."""

    def test_required_args(self) -> None:
        """Should require --input-dir and --run-id."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self) -> None:
        """Should set sensible defaults."""
        parser = build_parser()
        args = parser.parse_args([
            "--input-dir", "/tmp/input",
            "--run-id", "test_run",
        ])
        assert args.value_column == "sap_velocity_cnn_lstm"
        assert args.sw_threshold == 15.0
        assert args.resolution == 0.1
        assert args.no_png is False
        assert args.dask_blocksize == "256MB"

    def test_custom_args(self) -> None:
        """Should accept custom arguments."""
        parser = build_parser()
        args = parser.parse_args([
            "--input-dir", "/data/pred",
            "--run-id", "custom_run",
            "--value-column", "sap_velocity_ensemble",
            "--sw-threshold", "0",
            "--resolution", "0.05",
            "--no-png",
            "--stats", "mean", "max",
            "--lat-range", "-60", "78",
        ])
        assert args.value_column == "sap_velocity_ensemble"
        assert args.sw_threshold == 0.0
        assert args.resolution == 0.05
        assert args.no_png is True
        assert args.stats == ["mean", "max"]
        assert args.lat_range == [-60.0, 78.0]


# ---------------------------------------------------------------------------
# Tests: _sanitize_filename
# ---------------------------------------------------------------------------

class TestSanitizeFilename:
    """Tests for timestamp-to-filename sanitization."""

    def test_basic_timestamp(self) -> None:
        """ISO timestamp becomes filesystem-safe."""
        assert _sanitize_filename("2015-07-01 12:00:00") == "2015-07-01_12-00-00"

    def test_timezone_stripped(self) -> None:
        """Timezone offset after '+' is stripped."""
        assert _sanitize_filename("2015-07-01 12:00:00+00:00") == "2015-07-01_12-00-00"

    def test_special_chars_removed(self) -> None:
        """Characters outside the whitelist are removed."""
        result = _sanitize_filename("../../evil/path")
        # Dots and hyphens allowed; slashes stripped
        assert "/" not in result
        assert "\\" not in result

    def test_empty_returns_unknown(self) -> None:
        """Empty string returns 'unknown'."""
        assert _sanitize_filename("") == "unknown"


# ---------------------------------------------------------------------------
# Tests: rasterize_all_timestamps (single-pass)
# ---------------------------------------------------------------------------

class TestRasterizeAllTimestamps:
    """Tests for single-pass CSV rasterization."""

    def test_produces_tifs_for_all_timestamps(
        self, sample_csv: str, tmp_path: Path,
    ) -> None:
        """Should produce one GeoTIFF per unique timestamp."""
        out_dir = str(tmp_path / "single_pass")
        results = rasterize_all_timestamps(
            csv_path=sample_csv,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=1.0,
        )
        assert len(results) == 2
        for ts_str, tif_path in results:
            assert os.path.exists(tif_path)

    def test_resume_skips_existing(
        self, sample_csv: str, tmp_path: Path,
    ) -> None:
        """Should skip timestamps whose TIFs already exist."""
        out_dir = str(tmp_path / "resume")
        # Run once
        results1 = rasterize_all_timestamps(
            csv_path=sample_csv,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=1.0,
        )
        # Run again — should skip (no errors, same results)
        results2 = rasterize_all_timestamps(
            csv_path=sample_csv,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=1.0,
        )
        assert len(results2) == len(results1)


# ---------------------------------------------------------------------------
# Tests: run_batch path validation
# ---------------------------------------------------------------------------

class TestRunBatchPathValidation:
    """Tests for path traversal protection in run_batch."""

    def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        """run_id with '..' should be rejected."""
        with pytest.raises(ValueError, match="escapes base"):
            run_batch(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                run_id="../../etc",
            )

    def test_valid_run_id_accepted(
        self, sample_csv: str, tmp_path: Path,
    ) -> None:
        """Normal run_id should not raise."""
        # Create a CSV in the input dir
        import shutil
        in_dir = str(tmp_path / "in")
        os.makedirs(in_dir, exist_ok=True)
        shutil.copy(sample_csv, os.path.join(in_dir, "test_predictions.csv"))

        # Should not raise (will raise FileNotFoundError from glob if
        # pattern doesn't match, but NOT ValueError)
        try:
            run_batch(
                input_dir=in_dir,
                output_dir=str(tmp_path / "out"),
                run_id="valid_run_2025",
                file_glob="*predictions*.csv",
                sw_in_threshold=0,
                resolution=1.0,
                stats=(),
                render_png=False,
            )
        except FileNotFoundError:
            pytest.fail("run_batch raised FileNotFoundError unexpectedly")




# ---------------------------------------------------------------------------
# Fixtures for WS4 expanded tests
# ---------------------------------------------------------------------------

@pytest.fixture
def known_pixel_csv(tmp_path: Path) -> str:
    """Create CSV with known (lat, lon, value) triples for pixel placement.

    Coordinates are grid-aligned at 0.1 degree resolution.

    Returns
    -------
    str
        Path to the CSV file.
    """
    rows = [
        {"timestamp.1": "2015-07-01", "latitude": 0.05, "longitude": 0.05,
         "sap_velocity_xgb": 1.5, "sw_in": 100.0},
        {"timestamp.1": "2015-07-01", "latitude": 10.05, "longitude": 20.05,
         "sap_velocity_xgb": 3.0, "sw_in": 200.0},
        {"timestamp.1": "2015-07-01", "latitude": -30.05, "longitude": -90.05,
         "sap_velocity_xgb": 0.5, "sw_in": 50.0},
    ]
    df = pd.DataFrame(rows)
    csv_path = str(tmp_path / "known_pixels.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_parquet(tmp_path: Path, sample_csv: str) -> str:
    """Create a Parquet version of the sample CSV.

    Returns
    -------
    str
        Path to the Parquet file.
    """
    df = pd.read_csv(sample_csv)
    parquet_path = str(tmp_path / "test_predictions.parquet")
    df.to_parquet(parquet_path, engine="pyarrow", compression="gzip")
    return parquet_path


@pytest.fixture
def hourly_csv(tmp_path: Path) -> str:
    """Create CSV with hourly data (24 hours) for a single day.

    Returns
    -------
    str
        Path to the CSV file.
    """
    rows = []
    np.random.seed(55)
    lats = [0.05, 10.05]
    lons = [0.05, 20.05]
    for hour in range(24):
        for lat, lon in zip(lats, lons):
            rows.append({
                "timestamp.1": f"2015-07-01 {hour:02d}:00:00+00:00",
                "latitude": lat,
                "longitude": lon,
                "sw_in": 300.0 if 6 <= hour <= 18 else 0.0,
                "sap_velocity_xgb": np.random.uniform(0, 5),
            })
    df = pd.DataFrame(rows)
    csv_path = str(tmp_path / "hourly_predictions.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def multi_ts_csv(tmp_path: Path) -> str:
    """Create CSV with 3 distinct timestamps and non-overlapping values.

    Each timestamp has unique value ranges to verify no cross-contamination.

    Returns
    -------
    str
        Path to the CSV file.
    """
    rows = []
    np.random.seed(77)
    timestamps = ["2015-07-01", "2015-07-02", "2015-07-03"]
    value_ranges = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    for ts, (vmin, vmax) in zip(timestamps, value_ranges):
        for _ in range(30):
            rows.append({
                "timestamp.1": ts,
                "latitude": round(np.random.uniform(-10, 10), 1),
                "longitude": round(np.random.uniform(-10, 10), 1),
                "sw_in": 200.0,
                "sap_velocity_xgb": np.random.uniform(vmin, vmax),
            })
    df = pd.DataFrame(rows)
    csv_path = str(tmp_path / "multi_ts_predictions.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# WS4: Spatial Correctness Tests
# ---------------------------------------------------------------------------

class TestSpatialCorrectness:
    """Tests for spatial correctness of rasterized GeoTIFFs."""

    def test_pixel_coordinate_placement(
        self, known_pixel_csv: str, tmp_path: Path,
    ) -> None:
        """Known (lat, lon, value) triples appear at correct pixel positions."""
        out_dir = str(tmp_path / "pixel_test")
        results = rasterize_all_timestamps(
            csv_path=known_pixel_csv,
            output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp.1",
            sw_in_threshold=0,
            resolution=0.1,
        )
        assert len(results) == 1
        _, tif_path = results[0]

        with rasterio.open(tif_path) as src:
            data = src.read(1)
            transform = src.transform
            # Check pixel for (lat=0.05, lon=0.05) -> value ~1.5
            col, row = ~transform * (0.05, 0.05)
            row, col = int(row), int(col)
            assert data[row, col] == pytest.approx(1.5, abs=0.1)

    def test_coordinate_to_pixel_roundtrip(
        self, known_pixel_csv: str, tmp_path: Path,
    ) -> None:
        """Convert (lat,lon)->pixel->back; error < resolution/2."""
        out_dir = str(tmp_path / "roundtrip_test")
        results = rasterize_all_timestamps(
            csv_path=known_pixel_csv,
            output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp.1",
            sw_in_threshold=0,
            resolution=0.1,
        )
        _, tif_path = results[0]

        with rasterio.open(tif_path) as src:
            transform = src.transform
            test_points = [(0.05, 0.05), (10.05, 20.05), (-30.05, -90.05)]
            for lat, lon in test_points:
                col, row = ~transform * (lon, lat)
                lon_back, lat_back = transform * (int(col) + 0.5, int(row) + 0.5)
                assert abs(lat_back - lat) <= 0.051, (
                    f"Lat roundtrip error for ({lat}, {lon})"
                )
                assert abs(lon_back - lon) <= 0.051, (
                    f"Lon roundtrip error for ({lat}, {lon})"
                )

    def test_nodata_for_missing_coordinates(
        self, known_pixel_csv: str, tmp_path: Path,
    ) -> None:
        """Pixels without predictions must be NaN (nodata)."""
        out_dir = str(tmp_path / "nodata_test")
        results = rasterize_all_timestamps(
            csv_path=known_pixel_csv,
            output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp.1",
            sw_in_threshold=0,
            resolution=0.1,
        )
        _, tif_path = results[0]

        with rasterio.open(tif_path) as src:
            data = src.read(1)
            # Most of the grid should be NaN (only 3 pixels have data)
            nan_fraction = np.sum(np.isnan(data)) / data.size
            assert nan_fraction > 0.99, (
                f"Expected >99% NaN but got {nan_fraction:.1%}"
            )

    def test_grid_dimensions_match_resolution(self, tmp_path: Path) -> None:
        """Default extent [-60,78]x[-180,180] at 0.1 deg -> 1380x3600."""
        df = pd.DataFrame({
            "latitude": [0.0],
            "longitude": [0.0],
            "value": [1.0],
        })
        tif_path = str(tmp_path / "dims_test.tif")
        _write_geotiff(
            df, tif_path, "value", resolution=0.1,
            lat_range=(-60.0, 78.0), lon_range=(-180.0, 180.0),
        )
        with rasterio.open(tif_path) as src:
            # Grid includes upper bound pixel, so (78-(-60))/0.1 + 1 = 1381
            assert src.height == 1381, f"Expected 1381 rows, got {src.height}"
            assert src.width == 3601, f"Expected 3601 cols, got {src.width}"

    def test_no_duplicate_pixels(self, tmp_path: Path) -> None:
        """Duplicate (lat, lon) rows should be averaged, not overwritten."""
        rows = [
            {"timestamp.1": "2015-07-01", "latitude": 0.05, "longitude": 0.05,
             "sap_velocity_xgb": 2.0, "sw_in": 100.0},
            {"timestamp.1": "2015-07-01", "latitude": 0.05, "longitude": 0.05,
             "sap_velocity_xgb": 4.0, "sw_in": 100.0},
        ]
        df = pd.DataFrame(rows)
        csv_path = str(tmp_path / "dup_pixels.csv")
        df.to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "dup_test")
        results = rasterize_all_timestamps(
            csv_path=csv_path,
            output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp.1",
            sw_in_threshold=0,
            resolution=0.1,
        )
        _, tif_path = results[0]

        with rasterio.open(tif_path) as src:
            data = src.read(1)
            transform = src.transform
            col, row = ~transform * (0.05, 0.05)
            row, col = int(row), int(col)
            # Mean of 2.0 and 4.0 = 3.0
            assert data[row, col] == pytest.approx(3.0, abs=0.01), (
                f"Expected mean 3.0 but got {data[row, col]}"
            )


# ---------------------------------------------------------------------------
# WS4: Temporal Grouping Tests
# ---------------------------------------------------------------------------

class TestTemporalGrouping:
    """Tests for temporal grouping and per-timestamp map generation."""

    def test_single_timestamp_produces_one_tif(
        self, tmp_path: Path,
    ) -> None:
        """Input with one unique timestamp -> exactly one .tif file."""
        df = pd.DataFrame({
            "timestamp.1": ["2015-07-01"] * 5,
            "latitude": [0.0, 1.0, 2.0, 3.0, 4.0],
            "longitude": [0.0, 1.0, 2.0, 3.0, 4.0],
            "sap_velocity_xgb": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        csv_path = str(tmp_path / "single_ts.csv")
        df.to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "single_ts_out")
        results = rasterize_all_timestamps(
            csv_path=csv_path,
            output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp.1",
            sw_in_threshold=0,
            resolution=1.0,
        )
        assert len(results) == 1

    def test_multiple_timestamps_produce_multiple_tifs(
        self, multi_ts_csv: str, tmp_path: Path,
    ) -> None:
        """Input with N unique timestamps -> N .tif files."""
        out_dir = str(tmp_path / "multi_ts_out")
        results = rasterize_all_timestamps(
            csv_path=multi_ts_csv,
            output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp.1",
            sw_in_threshold=0,
            resolution=1.0,
        )
        assert len(results) == 3
        # Filenames should contain timestamp strings
        for ts_str, tif_path in results:
            assert ts_str in ["2015-07-01", "2015-07-02", "2015-07-03"]
            assert os.path.exists(tif_path)

    def test_no_cross_contamination_between_timestamps(
        self, multi_ts_csv: str, tmp_path: Path,
    ) -> None:
        """Values from timestamp A must not appear in GeoTIFF for B."""
        out_dir = str(tmp_path / "contam_test")
        results = rasterize_all_timestamps(
            csv_path=multi_ts_csv,
            output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp.1",
            sw_in_threshold=0,
            resolution=1.0,
        )
        # Value ranges: ts1=[1,2], ts2=[3,4], ts3=[5,6]
        for ts_str, tif_path in results:
            with rasterio.open(tif_path) as src:
                data = src.read(1)
                valid = data[~np.isnan(data)]
                if ts_str == "2015-07-01":
                    assert np.all(valid >= 0.9) and np.all(valid <= 2.1), (
                        f"Timestamp 1 has out-of-range values: "
                        f"[{valid.min():.2f}, {valid.max():.2f}]"
                    )
                elif ts_str == "2015-07-02":
                    assert np.all(valid >= 2.9) and np.all(valid <= 4.1), (
                        f"Timestamp 2 has out-of-range values: "
                        f"[{valid.min():.2f}, {valid.max():.2f}]"
                    )
                elif ts_str == "2015-07-03":
                    assert np.all(valid >= 4.9) and np.all(valid <= 6.1), (
                        f"Timestamp 3 has out-of-range values: "
                        f"[{valid.min():.2f}, {valid.max():.2f}]"
                    )

    def test_hourly_to_daily_aggregation(
        self, hourly_csv: str, tmp_path: Path,
    ) -> None:
        """Hourly input aggregated to daily: one map per day with mean."""
        out_dir = str(tmp_path / "hourly_agg")
        results = rasterize_all_timestamps(
            csv_path=hourly_csv,
            output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp.1",
            sw_in_threshold=0,
            resolution=0.1,
            hourly_mode=True,
            hourly_agg="mean",
        )
        # Should aggregate to 1 day
        assert len(results) == 1
        ts_str, tif_path = results[0]
        assert "2015-07-01" in ts_str

        # Verify value is reasonable (mean of random uniform [0,5])
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            valid = data[~np.isnan(data)]
            assert len(valid) > 0
            assert np.all(valid >= 0) and np.all(valid <= 5)


# ---------------------------------------------------------------------------
# WS4: Composite Correctness Tests
# ---------------------------------------------------------------------------

class TestCompositeCorrectness:
    """Tests for composite map correctness."""

    def test_composite_mean_equals_manual(
        self, sample_geotiffs: List[str], tmp_path: Path,
    ) -> None:
        """Composite mean matches numpy nanmean across input TIFFs."""
        # Read all input data
        arrays = []
        for tif_path in sample_geotiffs:
            with rasterio.open(tif_path) as src:
                arrays.append(src.read(1))
        expected_mean = np.nanmean(np.stack(arrays, axis=0), axis=0)

        out_dir = str(tmp_path / "mean_check")
        results = compute_composites(
            tif_paths=sample_geotiffs,
            output_dir=out_dir,
            stats=("mean",),
            render_png=False,
        )
        with rasterio.open(results["mean"]) as src:
            actual_mean = src.read(1)

        # Compare only non-NaN pixels
        mask = ~np.isnan(expected_mean) & ~np.isnan(actual_mean)
        np.testing.assert_allclose(
            actual_mean[mask], expected_mean[mask], atol=1e-5,
        )

    def test_composite_count_reflects_valid_days(
        self, tmp_path: Path,
    ) -> None:
        """Pixel present in 2/3 timestamps -> count = 2."""
        h, w = 5, 5
        profile = {
            "driver": "GTiff", "height": h, "width": w, "count": 1,
            "dtype": rasterio.float32,
            "crs": "+proj=longlat +datum=WGS84",
            "transform": rasterio.transform.from_origin(-180, 78, 0.1, 0.1),
            "nodata": np.nan,
        }
        tifs = []
        for i in range(3):
            data = np.ones((h, w), dtype=np.float32) * (i + 1)
            if i == 2:
                data[2, 3] = np.nan  # Missing in 3rd file
            path = str(tmp_path / f"count_{i}.tif")
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data, 1)
            tifs.append(path)

        out_dir = str(tmp_path / "count_check")
        results = compute_composites(
            tif_paths=tifs,
            output_dir=out_dir,
            stats=("count",),
            render_png=False,
        )
        with rasterio.open(results["count"]) as src:
            count_data = src.read(1)
        # Pixel (2,3) only in 2 files
        assert count_data[2, 3] == pytest.approx(2.0)
        # Other pixels in all 3
        assert count_data[0, 1] == pytest.approx(3.0)

    def test_composite_std_with_single_value(
        self, tmp_path: Path,
    ) -> None:
        """Pixel with 1 observation -> std = 0 (ddof=0)."""
        h, w = 3, 3
        profile = {
            "driver": "GTiff", "height": h, "width": w, "count": 1,
            "dtype": rasterio.float32,
            "crs": "+proj=longlat +datum=WGS84",
            "transform": rasterio.transform.from_origin(-180, 78, 0.1, 0.1),
            "nodata": np.nan,
        }
        # Only one file has data at (1,1); others are NaN there
        tifs = []
        for i in range(3):
            data = np.full((h, w), np.nan, dtype=np.float32)
            if i == 0:
                data[1, 1] = 5.0  # Only observation
            else:
                data[0, 0] = float(i)  # Data elsewhere
            path = str(tmp_path / f"std_{i}.tif")
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data, 1)
            tifs.append(path)

        out_dir = str(tmp_path / "std_check")
        results = compute_composites(
            tif_paths=tifs,
            output_dir=out_dir,
            stats=("std",),
            render_png=False,
        )
        with rasterio.open(results["std"]) as src:
            std_data = src.read(1)
        # Single observation -> std = 0
        assert std_data[1, 1] == pytest.approx(0.0, abs=1e-5)

    def test_composite_ignores_nodata(
        self, tmp_path: Path,
    ) -> None:
        """NoData pixels should not affect aggregation of valid pixels."""
        h, w = 3, 3
        profile = {
            "driver": "GTiff", "height": h, "width": w, "count": 1,
            "dtype": rasterio.float32,
            "crs": "+proj=longlat +datum=WGS84",
            "transform": rasterio.transform.from_origin(-180, 78, 0.1, 0.1),
            "nodata": np.nan,
        }
        values_at_00 = [2.0, 4.0, np.nan]  # 2 valid, 1 NaN
        tifs = []
        for i, v in enumerate(values_at_00):
            data = np.full((h, w), np.nan, dtype=np.float32)
            data[0, 0] = v
            path = str(tmp_path / f"nodata_{i}.tif")
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data, 1)
            tifs.append(path)

        out_dir = str(tmp_path / "nodata_check")
        results = compute_composites(
            tif_paths=tifs,
            output_dir=out_dir,
            stats=("mean",),
            render_png=False,
        )
        with rasterio.open(results["mean"]) as src:
            mean_data = src.read(1)
        # nanmean([2.0, 4.0]) = 3.0
        assert mean_data[0, 0] == pytest.approx(3.0, abs=1e-5)


# ---------------------------------------------------------------------------
# WS4: Format and I/O Tests
# ---------------------------------------------------------------------------

class TestFormatAndIO:
    """Tests for format handling and GeoTIFF metadata."""

    def test_parquet_input_produces_same_output(
        self, sample_csv: str, sample_parquet: str, tmp_path: Path,
    ) -> None:
        """Same data as CSV and Parquet -> identical GeoTIFF output."""
        csv_dir = str(tmp_path / "csv_out")
        parquet_dir = str(tmp_path / "parquet_out")

        csv_results = rasterize_all_timestamps(
            csv_path=sample_csv,
            output_dir=csv_dir,
            sw_in_threshold=0,
            resolution=1.0,
        )
        parquet_results = rasterize_all_timestamps(
            csv_path=sample_parquet,
            output_dir=parquet_dir,
            sw_in_threshold=0,
            resolution=1.0,
        )

        assert len(csv_results) == len(parquet_results)

        for (_, csv_tif), (_, pq_tif) in zip(csv_results, parquet_results):
            with rasterio.open(csv_tif) as src_csv:
                data_csv = src_csv.read(1)
            with rasterio.open(pq_tif) as src_pq:
                data_pq = src_pq.read(1)

            assert data_csv.shape == data_pq.shape
            # Compare non-NaN values
            mask = ~np.isnan(data_csv) & ~np.isnan(data_pq)
            np.testing.assert_allclose(
                data_csv[mask], data_pq[mask], atol=1e-5,
            )

    def test_geotiff_has_correct_crs(
        self, known_pixel_csv: str, tmp_path: Path,
    ) -> None:
        """Output GeoTIFF CRS must be EPSG:4326."""
        out_dir = str(tmp_path / "crs_test")
        results = rasterize_all_timestamps(
            csv_path=known_pixel_csv,
            output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp.1",
            sw_in_threshold=0,
            resolution=0.1,
        )
        _, tif_path = results[0]
        with rasterio.open(tif_path) as src:
            assert src.crs == CRS.from_epsg(4326), (
                f"Expected EPSG:4326 but got {src.crs}"
            )

    def test_geotiff_nodata_value(
        self, known_pixel_csv: str, tmp_path: Path,
    ) -> None:
        """GeoTIFF nodata sentinel must be set (NaN for float32)."""
        out_dir = str(tmp_path / "nodata_meta_test")
        results = rasterize_all_timestamps(
            csv_path=known_pixel_csv,
            output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp.1",
            sw_in_threshold=0,
            resolution=0.1,
        )
        _, tif_path = results[0]
        with rasterio.open(tif_path) as src:
            # nodata should be NaN (as np.nan) or a numeric sentinel
            nodata = src.nodata
            assert nodata is not None, "GeoTIFF nodata not set"
            # For float32, NaN is the standard nodata sentinel
            assert np.isnan(nodata) or nodata == -9999, (
                f"Unexpected nodata value: {nodata}"
            )
