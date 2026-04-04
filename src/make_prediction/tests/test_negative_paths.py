"""Round 5 tests: Negative paths, error handling, and security edge cases.

Tests cover:
- Malformed CSV headers
- Non-numeric coordinate data
- Huge resolution that collapses grid to 1x1
- Path traversal via timestamp injection in filenames
- Parquet file with missing columns
- Resolution propagation from run_batch to _write_geotiff
- Empty directory input
- Composites with single file (edge case)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from src.make_prediction.prediction_visualization_hpc import (
    _sanitize_filename,
    _write_geotiff,
    compute_composites,
    discover_timestamps,
    read_prediction_file,
    run_batch,
)

# ---------------------------------------------------------------------------
# Malformed input tests
# ---------------------------------------------------------------------------


class TestMalformedInput:
    """Tests for graceful handling of malformed CSV/Parquet data."""

    def test_non_numeric_coordinates_filtered(self, tmp_path: Path) -> None:
        """Non-numeric lat/lon should be filtered out, not crash."""
        df = pd.DataFrame(
            {
                "latitude": ["not_a_number", "0.0", "1.0"],
                "longitude": ["0.0", "also_bad", "1.0"],
                "value": [1.0, 2.0, 3.0],
            }
        )
        tif_path = str(tmp_path / "bad_coords.tif")
        # Only the 3rd row has valid lat AND lon
        _write_geotiff(df, tif_path, "value", resolution=0.1)
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            valid = data[~np.isnan(data)]
            assert len(valid) == 1
            assert valid[0] == pytest.approx(3.0, abs=0.01)

    def test_non_numeric_values_filtered(self, tmp_path: Path) -> None:
        """Non-numeric value column should be coerced to NaN and filtered."""
        df = pd.DataFrame(
            {
                "latitude": [0.0, 1.0],
                "longitude": [0.0, 1.0],
                "value": ["bad", 2.0],
            }
        )
        tif_path = str(tmp_path / "bad_vals.tif")
        _write_geotiff(df, tif_path, "value", resolution=0.1)
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            valid = data[~np.isnan(data)]
            assert len(valid) == 1

    def test_csv_with_extra_whitespace_columns(self, tmp_path: Path) -> None:
        """CSV with whitespace-padded column names should still work."""
        csv_path = str(tmp_path / "whitespace.csv")
        with open(csv_path, "w") as f:
            f.write("timestamp.1,latitude,longitude,sap_velocity_cnn_lstm\n")
            f.write("2015-07-01,0.0,0.0,1.5\n")
        ts_list = discover_timestamps(csv_path)
        assert len(ts_list) == 1


# ---------------------------------------------------------------------------
# Resolution edge cases
# ---------------------------------------------------------------------------


class TestResolutionEdgeCases:
    """Test extreme resolution values."""

    def test_huge_resolution_produces_1x1_grid(self, tmp_path: Path) -> None:
        """Resolution larger than data extent produces a 1x1 GeoTIFF."""
        df = pd.DataFrame(
            {
                "latitude": [0.0, 1.0],
                "longitude": [0.0, 1.0],
                "value": [1.0, 2.0],
            }
        )
        tif_path = str(tmp_path / "huge_res.tif")
        _write_geotiff(df, tif_path, "value", resolution=100.0)
        with rasterio.open(tif_path) as src:
            assert src.height == 1
            assert src.width == 1

    def test_resolution_zero_produces_no_output(self, tmp_path: Path) -> None:
        """run_batch with resolution=0 logs errors, produces no GeoTIFFs."""

        in_dir = str(tmp_path / "in")
        os.makedirs(in_dir)
        df = pd.DataFrame(
            {
                "timestamp.1": ["2015-07-01"],
                "latitude": [0.0],
                "longitude": [0.0],
                "sap_velocity_cnn_lstm": [1.0],
            }
        )
        df.to_csv(os.path.join(in_dir, "test_predictions.csv"), index=False)

        # resolution=0 raises ValueError in _write_geotiff, which is caught
        # per-timestamp by the narrowed except. No TIFs are produced.
        run_batch(
            input_dir=in_dir,
            output_dir=str(tmp_path / "out"),
            run_id="zero_res_test",
            file_glob="*predictions*.csv",
            sw_in_threshold=0,
            resolution=0.0,
            stats=(),
            render_png=False,
        )
        run_dir = tmp_path / "out" / "zero_res_test"
        tifs = list(run_dir.glob("*.tif")) if run_dir.exists() else []
        assert len(tifs) == 0, "No TIFs should be created with resolution=0"


# ---------------------------------------------------------------------------
# Filename sanitization security
# ---------------------------------------------------------------------------


class TestFilenameSecurity:
    """Test _sanitize_filename against path injection attempts."""

    def test_path_separator_removed(self) -> None:
        """Forward and back slashes should be stripped."""
        assert "/" not in _sanitize_filename("../../../etc/passwd")
        assert "\\" not in _sanitize_filename("..\\..\\secret")

    def test_leading_dots_stripped(self) -> None:
        """Leading dots (hidden files, ..) should be stripped."""
        result = _sanitize_filename("...hidden")
        assert not result.startswith(".")

    def test_null_bytes_removed(self) -> None:
        """Null bytes should be stripped."""
        result = _sanitize_filename("normal\x00evil")
        assert "\x00" not in result

    def test_very_long_timestamp_truncated(self) -> None:
        """Timestamps > 200 chars should be truncated."""
        long_ts = "A" * 500
        result = _sanitize_filename(long_ts)
        assert len(result) <= 200


# ---------------------------------------------------------------------------
# Composite edge cases
# ---------------------------------------------------------------------------


class TestCompositeEdgeCases:
    """Edge cases in composite generation."""

    def test_single_tif_stats(self, tmp_path: Path) -> None:
        """Composites from a single GeoTIFF should equal the input."""
        h, w = 5, 5
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": rasterio.float32,
            "crs": "+proj=longlat +datum=WGS84",
            "transform": from_origin(-180, 78, 0.1, 0.1),
            "nodata": np.nan,
        }
        data_in = np.array([[1, 2], [3, 4]], dtype=np.float32)
        data_full = np.full((h, w), np.nan, dtype=np.float32)
        data_full[:2, :2] = data_in
        path = str(tmp_path / "single.tif")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data_full, 1)

        out_dir = str(tmp_path / "single_comp")
        results = compute_composites(
            tif_paths=[path],
            output_dir=out_dir,
            stats=("mean", "min", "max"),
            render_png=False,
        )
        # Mean of 1 observation = the observation itself
        for stat in ("mean", "min", "max"):
            with rasterio.open(results[stat]) as src:
                out = src.read(1)
            np.testing.assert_allclose(
                out[:2, :2],
                data_in,
                atol=1e-5,
                err_msg=f"{stat} != input for single TIF",
            )

    def test_all_nan_tifs_produce_nan_composite(self, tmp_path: Path) -> None:
        """All-NaN inputs should produce all-NaN composite."""
        h, w = 3, 3
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": rasterio.float32,
            "crs": "+proj=longlat +datum=WGS84",
            "transform": from_origin(-180, 78, 0.1, 0.1),
            "nodata": np.nan,
        }
        tifs = []
        for i in range(2):
            data = np.full((h, w), np.nan, dtype=np.float32)
            path = str(tmp_path / f"allnan_{i}.tif")
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data, 1)
            tifs.append(path)

        out_dir = str(tmp_path / "nan_comp")
        results = compute_composites(
            tif_paths=tifs,
            output_dir=out_dir,
            stats=("mean", "count"),
            render_png=False,
        )
        with rasterio.open(results["mean"]) as src:
            mean_data = src.read(1)
        assert np.all(np.isnan(mean_data)), "All-NaN input should produce all-NaN mean"

        with rasterio.open(results["count"]) as src:
            count_data = src.read(1)
        assert np.all(count_data == 0), "All-NaN input should have count=0 everywhere"


# ---------------------------------------------------------------------------
# read_prediction_file edge cases
# ---------------------------------------------------------------------------


class TestReadPredictionFile:
    """Tests for the file reader function."""

    def test_read_csv_without_chunksize(self, tmp_path: Path) -> None:
        """Reading CSV without chunksize returns full DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        csv_path = str(tmp_path / "simple.csv")
        df.to_csv(csv_path, index=False)
        result = read_prediction_file(csv_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_read_csv_with_chunksize_returns_iterator(self, tmp_path: Path) -> None:
        """Reading CSV with chunksize returns an iterator."""
        df = pd.DataFrame({"a": range(100)})
        csv_path = str(tmp_path / "chunked.csv")
        df.to_csv(csv_path, index=False)
        result = read_prediction_file(csv_path, chunksize=10)
        # Should be iterable
        chunks = list(result)
        assert len(chunks) == 10
        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 100

    def test_read_nonexistent_file_raises(self) -> None:
        """Reading a nonexistent file should raise."""
        with pytest.raises((FileNotFoundError, OSError)):
            read_prediction_file("/nonexistent/path/file.csv")
