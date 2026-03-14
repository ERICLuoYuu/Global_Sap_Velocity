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

from src.make_prediction.prediction_visualization_hpc import (
    _sanitize_filename,
    _write_geotiff,
    build_parser,
    compute_composites,
    discover_timestamps,
    get_valid_block_size,
    rasterize_all_timestamps,
    rasterize_timestamp,
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
                csv_glob="*predictions*.csv",
                sw_in_threshold=0,
                resolution=1.0,
                stats=(),
                render_png=False,
            )
        except FileNotFoundError:
            pytest.fail("run_batch raised FileNotFoundError unexpectedly")

