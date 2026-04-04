"""Round 1 tests for review fixes (C1, H5, H6, H7, M1, M4, M5).

These tests verify the specific fixes applied in the review loop:
- C1+H3: Chunk-level aggregation uses intended hourly_agg method
- H5: file_glob path traversal guard
- H6: Resolution lower bound + max raster dimension
- H7: Parquet-aware header read in rasterize_timestamp
- M1: vmin == vmax guard in plot_map
- M4: lat_range/lon_range Earth bounds validation
- M5: Leading hyphens stripped in _sanitize_filename
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from src.make_prediction.prediction_visualization_hpc import (
    _sanitize_filename,
    _write_geotiff,
    plot_map,
    rasterize_all_timestamps,
    rasterize_timestamp,
    run_batch,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hourly_known_csv(tmp_path: Path) -> str:
    """CSV with hourly data where max != mean, to verify C1 fix.

    Pixel (0.05, 0.05) has values [1, 2, 3, 10] across 4 hours.
    mean=4.0, max=10, min=1, sum=16
    """
    rows = []
    values = [1.0, 2.0, 3.0, 10.0]
    for hour, val in enumerate(values):
        rows.append(
            {
                "timestamp.1": f"2015-07-01 {hour + 8:02d}:00:00+00:00",
                "latitude": 0.05,
                "longitude": 0.05,
                "sap_velocity_cnn_lstm": val,
            }
        )
    df = pd.DataFrame(rows)
    csv_path = str(tmp_path / "hourly_known.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def uniform_geotiff(tmp_path: Path) -> str:
    """GeoTIFF where all valid pixels have the same value (for M1 test)."""
    h, w = 10, 20
    data = np.full((h, w), 3.14, dtype=np.float32)
    data[0, 0] = np.nan
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
    tif_path = str(tmp_path / "uniform.tif")
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(data, 1)
    return tif_path


# ---------------------------------------------------------------------------
# C1+H3: Chunk-level aggregation uses correct hourly_agg
# ---------------------------------------------------------------------------


class TestC1ChunkAggregation:
    """Verify that chunk-level aggregation respects hourly_agg parameter."""

    def test_hourly_max_returns_true_max(
        self,
        hourly_known_csv: str,
        tmp_path: Path,
    ) -> None:
        """hourly_agg='max' must return the true maximum, not max of means."""
        out_dir = str(tmp_path / "c1_max")
        results = rasterize_all_timestamps(
            csv_path=hourly_known_csv,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=0.1,
            hourly_mode=True,
            hourly_agg="max",
        )
        assert len(results) == 1
        _, tif_path = results[0]
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            valid = data[~np.isnan(data)]
            assert valid[0] == pytest.approx(10.0, abs=0.01), (
                f"Expected max 10.0 but got {valid[0]} — C1 bug still present"
            )

    def test_hourly_min_returns_true_min(
        self,
        hourly_known_csv: str,
        tmp_path: Path,
    ) -> None:
        """hourly_agg='min' must return the true minimum."""
        out_dir = str(tmp_path / "c1_min")
        results = rasterize_all_timestamps(
            csv_path=hourly_known_csv,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=0.1,
            hourly_mode=True,
            hourly_agg="min",
        )
        assert len(results) == 1
        _, tif_path = results[0]
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            valid = data[~np.isnan(data)]
            assert valid[0] == pytest.approx(1.0, abs=0.01), (
                f"Expected min 1.0 but got {valid[0]} — C1 bug still present"
            )

    def test_hourly_mean_still_correct(
        self,
        hourly_known_csv: str,
        tmp_path: Path,
    ) -> None:
        """hourly_agg='mean' (the default) should still produce correct mean."""
        out_dir = str(tmp_path / "c1_mean")
        results = rasterize_all_timestamps(
            csv_path=hourly_known_csv,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=0.1,
            hourly_mode=True,
            hourly_agg="mean",
        )
        assert len(results) == 1
        _, tif_path = results[0]
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            valid = data[~np.isnan(data)]
            expected_mean = (1.0 + 2.0 + 3.0 + 10.0) / 4
            assert valid[0] == pytest.approx(expected_mean, abs=0.01), (
                f"Expected mean {expected_mean} but got {valid[0]}"
            )

    def test_hourly_sum_returns_true_sum(
        self,
        hourly_known_csv: str,
        tmp_path: Path,
    ) -> None:
        """hourly_agg='sum' must return the true sum across hours."""
        out_dir = str(tmp_path / "c1_sum")
        results = rasterize_all_timestamps(
            csv_path=hourly_known_csv,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=0.1,
            hourly_mode=True,
            hourly_agg="sum",
        )
        assert len(results) == 1
        _, tif_path = results[0]
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            valid = data[~np.isnan(data)]
            assert valid[0] == pytest.approx(16.0, abs=0.01), (
                f"Expected sum 16.0 but got {valid[0]} — H3 bug still present"
            )


# ---------------------------------------------------------------------------
# H5: file_glob path traversal guard
# ---------------------------------------------------------------------------


class TestH5FileGlobTraversal:
    """Verify that file_glob with unsafe path components is rejected."""

    def test_dotdot_in_glob_rejected(self, tmp_path: Path) -> None:
        """file_glob containing '..' should raise ValueError."""
        with pytest.raises(ValueError, match="unsafe path"):
            run_batch(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                run_id="test",
                file_glob="../../*.csv",
            )

    def test_absolute_glob_rejected(self, tmp_path: Path) -> None:
        """file_glob starting with '/' should raise ValueError."""
        with pytest.raises(ValueError, match="unsafe path"):
            run_batch(
                input_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                run_id="test",
                file_glob="/etc/passwd",
            )

    def test_normal_glob_accepted(self, tmp_path: Path) -> None:
        """Normal glob patterns should still work."""
        in_dir = str(tmp_path / "in")
        os.makedirs(in_dir, exist_ok=True)
        df = pd.DataFrame(
            {
                "timestamp.1": ["2015-07-01"],
                "latitude": [0.0],
                "longitude": [0.0],
                "sap_velocity_cnn_lstm": [1.0],
            }
        )
        df.to_csv(os.path.join(in_dir, "test_predictions.csv"), index=False)

        # Should not raise ValueError for path traversal
        try:
            run_batch(
                input_dir=in_dir,
                output_dir=str(tmp_path / "out"),
                run_id="ok",
                file_glob="*predictions*.csv",
                sw_in_threshold=0,
                resolution=1.0,
                stats=(),
                render_png=False,
            )
        except ValueError as exc:
            if "unsafe" in str(exc):
                pytest.fail(f"Normal glob rejected: {exc}")


# ---------------------------------------------------------------------------
# H6: Resolution lower bound + max raster dimension
# ---------------------------------------------------------------------------


class TestH6ResolutionBounds:
    """Verify resolution and raster dimension guards."""

    def test_tiny_resolution_rejected(self, tmp_path: Path) -> None:
        """Resolution below 0.001 should raise ValueError."""
        df = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "value": [1.0],
            }
        )
        tif_path = str(tmp_path / "tiny_res.tif")
        with pytest.raises(ValueError, match="below minimum"):
            _write_geotiff(df, tif_path, "value", resolution=0.0001)

    def test_zero_resolution_rejected(self, tmp_path: Path) -> None:
        """Resolution of 0 should raise ValueError."""
        df = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "value": [1.0],
            }
        )
        tif_path = str(tmp_path / "zero_res.tif")
        with pytest.raises(ValueError, match="positive"):
            _write_geotiff(df, tif_path, "value", resolution=0.0)

    def test_negative_resolution_rejected(self, tmp_path: Path) -> None:
        """Negative resolution should raise ValueError."""
        df = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "value": [1.0],
            }
        )
        tif_path = str(tmp_path / "neg_res.tif")
        with pytest.raises(ValueError, match="positive"):
            _write_geotiff(df, tif_path, "value", resolution=-0.1)

    def test_large_raster_dimension_rejected(self, tmp_path: Path) -> None:
        """Huge lat/lon range with fine resolution should hit dimension guard."""
        df = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "value": [1.0],
            }
        )
        tif_path = str(tmp_path / "huge_dim.tif")
        with pytest.raises(ValueError, match="exceed limit"):
            _write_geotiff(
                df,
                tif_path,
                "value",
                resolution=0.001,
                lat_range=(-90.0, 90.0),
                lon_range=(-180.0, 180.0),
            )

    def test_valid_resolution_passes(self, tmp_path: Path) -> None:
        """Valid resolution=0.1 with normal range should succeed."""
        df = pd.DataFrame(
            {
                "latitude": [0.0, 1.0],
                "longitude": [0.0, 1.0],
                "value": [1.0, 2.0],
            }
        )
        tif_path = str(tmp_path / "valid_res.tif")
        _write_geotiff(df, tif_path, "value", resolution=0.1)
        assert os.path.exists(tif_path)


# ---------------------------------------------------------------------------
# H7: Parquet-aware header read in rasterize_timestamp
# ---------------------------------------------------------------------------


class TestH7ParquetHeaderRead:
    """Verify rasterize_timestamp handles Parquet files correctly."""

    def test_parquet_input_does_not_crash(self, tmp_path: Path) -> None:
        """rasterize_timestamp with .parquet input should not crash on header read."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "timestamp.1": ["2015-07-01 12:00:00+00:00"] * 10,
                "latitude": np.linspace(-10, 10, 10),
                "longitude": np.linspace(-10, 10, 10),
                "sap_velocity_cnn_lstm": np.random.uniform(0, 5, 10),
                "sw_in": [200.0] * 10,
            }
        )
        pq_path = str(tmp_path / "test.parquet")
        df.to_parquet(pq_path, engine="pyarrow")

        tif_path = str(tmp_path / "h7_output.tif")
        result = rasterize_timestamp(
            csv_path=pq_path,
            timestamp_value="2015-07-01 12:00:00+00:00",
            output_tif=tif_path,
            sw_in_threshold=0,
            resolution=1.0,
        )
        # Should either succeed or return None, but NOT crash
        if result is not None:
            assert os.path.exists(result)


# ---------------------------------------------------------------------------
# M1: vmin == vmax guard in plot_map
# ---------------------------------------------------------------------------


class TestM1VminVmax:
    """Verify plot_map handles uniform-value GeoTIFFs without crashing."""

    def test_uniform_tiff_does_not_crash(
        self,
        uniform_geotiff: str,
        tmp_path: Path,
    ) -> None:
        """A GeoTIFF with all-same values should produce a PNG (not crash)."""
        png_path = str(tmp_path / "uniform.png")
        result = plot_map(
            tif_path=uniform_geotiff,
            output_png=png_path,
            dpi=72,
        )
        # Should produce a PNG (not None from exception)
        assert result is not None
        assert os.path.exists(png_path)


# ---------------------------------------------------------------------------
# M4: lat_range / lon_range Earth bounds validation
# ---------------------------------------------------------------------------


class TestM4EarthBounds:
    """Verify lat_range/lon_range are validated against Earth bounds."""

    def test_lat_range_outside_earth_rejected(self, tmp_path: Path) -> None:
        """lat_range exceeding [-90, 90] should raise ValueError."""
        df = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "value": [1.0],
            }
        )
        tif_path = str(tmp_path / "bad_lat.tif")
        with pytest.raises(ValueError, match="Earth bounds"):
            _write_geotiff(
                df,
                tif_path,
                "value",
                resolution=0.1,
                lat_range=(-100.0, 100.0),
            )

    def test_lon_range_outside_earth_rejected(self, tmp_path: Path) -> None:
        """lon_range exceeding [-180, 180] should raise ValueError."""
        df = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "value": [1.0],
            }
        )
        tif_path = str(tmp_path / "bad_lon.tif")
        with pytest.raises(ValueError, match="Earth bounds"):
            _write_geotiff(
                df,
                tif_path,
                "value",
                resolution=0.1,
                lon_range=(-200.0, 200.0),
            )

    def test_valid_global_range_accepted(self, tmp_path: Path) -> None:
        """Standard global range [-60, 78] x [-180, 180] should work."""
        df = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "value": [1.0],
            }
        )
        tif_path = str(tmp_path / "valid_range.tif")
        _write_geotiff(
            df,
            tif_path,
            "value",
            resolution=1.0,
            lat_range=(-60.0, 78.0),
            lon_range=(-180.0, 180.0),
        )
        assert os.path.exists(tif_path)


# ---------------------------------------------------------------------------
# M5: Leading hyphens stripped in _sanitize_filename
# ---------------------------------------------------------------------------


class TestM5SanitizeHyphens:
    """Verify leading hyphens are stripped from sanitized filenames."""

    def test_leading_hyphens_stripped(self) -> None:
        """Filenames must not start with hyphens (shell confusion)."""
        result = _sanitize_filename("--dangerous-flag")
        assert not result.startswith("-"), f"Leading hyphen not stripped: {result!r}"

    def test_leading_dots_still_stripped(self) -> None:
        """Leading dots should still be stripped (hidden files)."""
        result = _sanitize_filename("...hidden")
        assert not result.startswith("."), f"Leading dot not stripped: {result!r}"

    def test_normal_timestamp_unchanged(self) -> None:
        """Normal timestamp should produce expected output."""
        result = _sanitize_filename("2015-07-01 12:00:00+00:00")
        assert result == "2015-07-01_12-00-00"
