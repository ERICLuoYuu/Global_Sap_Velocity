"""Round 2 tests: Negative paths & error handling for review fixes.

Targets:
- Corrupt TIF detection during resume
- Dask parquet path via rasterize_timestamp
- _rasterize_timestamp_pandas column resolution
- Inverted lat/lon ranges
- CLI argument edge cases
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
    build_parser,
    rasterize_all_timestamps,
    rasterize_timestamp,
)

# ---------------------------------------------------------------------------
# Corrupt TIF resume detection
# ---------------------------------------------------------------------------


class TestCorruptTifResume:
    """Verify that corrupt TIFs are detected and regenerated during resume."""

    def test_corrupt_tif_is_regenerated(self, tmp_path: Path) -> None:
        """A truncated TIF should be detected and overwritten on second run."""
        # Create a valid CSV
        df = pd.DataFrame(
            {
                "timestamp.1": ["2015-07-01"] * 5,
                "latitude": [0.0, 1.0, 2.0, 3.0, 4.0],
                "longitude": [0.0, 1.0, 2.0, 3.0, 4.0],
                "sap_velocity_cnn_lstm": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        csv_path = str(tmp_path / "predictions.csv")
        df.to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "tifs")

        # Run once to get valid TIFs
        results1 = rasterize_all_timestamps(
            csv_path=csv_path,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=1.0,
        )
        assert len(results1) == 1
        _, tif_path = results1[0]

        # Read original data
        with rasterio.open(tif_path) as src:
            original_data = src.read(1).copy()

        # Corrupt the TIF by truncating it
        with open(tif_path, "wb") as f:
            f.write(b"NOT A VALID TIFF FILE")

        # Re-run — should detect corruption and regenerate
        results2 = rasterize_all_timestamps(
            csv_path=csv_path,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=1.0,
        )
        assert len(results2) == 1
        _, tif_path2 = results2[0]

        # Verify regenerated TIF matches original
        with rasterio.open(tif_path2) as src:
            regenerated_data = src.read(1)
        np.testing.assert_array_equal(
            regenerated_data,
            original_data,
            err_msg="Regenerated TIF does not match original",
        )

    def test_valid_tif_is_skipped(self, tmp_path: Path) -> None:
        """A valid existing TIF should be skipped (not re-created)."""
        df = pd.DataFrame(
            {
                "timestamp.1": ["2015-07-01"] * 3,
                "latitude": [0.0, 1.0, 2.0],
                "longitude": [0.0, 1.0, 2.0],
                "sap_velocity_cnn_lstm": [1.0, 2.0, 3.0],
            }
        )
        csv_path = str(tmp_path / "predictions.csv")
        df.to_csv(csv_path, index=False)
        out_dir = str(tmp_path / "tifs")

        results1 = rasterize_all_timestamps(
            csv_path=csv_path,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=1.0,
        )
        _, tif_path = results1[0]
        mtime_before = os.path.getmtime(tif_path)

        # Second run — should skip
        results2 = rasterize_all_timestamps(
            csv_path=csv_path,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=1.0,
        )
        mtime_after = os.path.getmtime(tif_path)
        assert mtime_before == mtime_after, "Valid TIF was modified on resume"


# ---------------------------------------------------------------------------
# Dask parquet path (rasterize_timestamp)
# ---------------------------------------------------------------------------


class TestDaskParquetPath:
    """Verify the Dask parquet branch in rasterize_timestamp."""

    def test_parquet_produces_same_as_csv(self, tmp_path: Path) -> None:
        """CSV and parquet input should produce identical GeoTIFFs."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "timestamp.1": ["2015-07-01 12:00:00"] * 20,
                "latitude": np.round(np.random.uniform(-10, 10, 20), 1),
                "longitude": np.round(np.random.uniform(-10, 10, 20), 1),
                "sap_velocity_cnn_lstm": np.random.uniform(0, 5, 20),
                "sw_in": [200.0] * 20,
            }
        )
        csv_path = str(tmp_path / "test.csv")
        pq_path = str(tmp_path / "test.parquet")
        df.to_csv(csv_path, index=False)
        df.to_parquet(pq_path, engine="pyarrow")

        csv_tif = str(tmp_path / "csv_out.tif")
        pq_tif = str(tmp_path / "pq_out.tif")

        rasterize_timestamp(
            csv_path=csv_path,
            timestamp_value="2015-07-01 12:00:00",
            output_tif=csv_tif,
            sw_in_threshold=0,
            resolution=1.0,
        )
        rasterize_timestamp(
            csv_path=pq_path,
            timestamp_value="2015-07-01 12:00:00",
            output_tif=pq_tif,
            sw_in_threshold=0,
            resolution=1.0,
        )

        with rasterio.open(csv_tif) as src_csv:
            data_csv = src_csv.read(1)
        with rasterio.open(pq_tif) as src_pq:
            data_pq = src_pq.read(1)

        assert data_csv.shape == data_pq.shape
        mask = ~np.isnan(data_csv) & ~np.isnan(data_pq)
        np.testing.assert_allclose(
            data_csv[mask],
            data_pq[mask],
            atol=1e-5,
            err_msg="CSV and Parquet produce different GeoTIFFs",
        )

    def test_parquet_missing_timestamp_returns_none(self, tmp_path: Path) -> None:
        """Parquet input with non-existent timestamp returns None."""
        df = pd.DataFrame(
            {
                "timestamp.1": ["2015-07-01"] * 5,
                "latitude": [0.0, 1.0, 2.0, 3.0, 4.0],
                "longitude": [0.0, 1.0, 2.0, 3.0, 4.0],
                "sap_velocity_cnn_lstm": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        pq_path = str(tmp_path / "test.parquet")
        df.to_parquet(pq_path, engine="pyarrow")

        result = rasterize_timestamp(
            csv_path=pq_path,
            timestamp_value="2099-01-01",
            output_tif=str(tmp_path / "missing.tif"),
            sw_in_threshold=0,
            resolution=1.0,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Inverted / invalid lat_range and lon_range
# ---------------------------------------------------------------------------


class TestInvertedRanges:
    """Verify inverted ranges are rejected cleanly."""

    def test_inverted_lat_range_rejected(self, tmp_path: Path) -> None:
        """lat_range where min > max should raise ValueError."""
        df = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "value": [1.0],
            }
        )
        tif_path = str(tmp_path / "inv_lat.tif")
        with pytest.raises(ValueError, match="inverted"):
            _write_geotiff(
                df,
                tif_path,
                "value",
                resolution=0.1,
                lat_range=(10.0, -10.0),
            )

    def test_inverted_lon_range_rejected(self, tmp_path: Path) -> None:
        """lon_range where min > max should raise ValueError."""
        df = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "value": [1.0],
            }
        )
        tif_path = str(tmp_path / "inv_lon.tif")
        with pytest.raises(ValueError, match="inverted"):
            _write_geotiff(
                df,
                tif_path,
                "value",
                resolution=0.1,
                lon_range=(10.0, -10.0),
            )

    def test_equal_lat_range_rejected(self, tmp_path: Path) -> None:
        """lat_range where min == max (zero width) should raise."""
        df = pd.DataFrame(
            {
                "latitude": [5.0],
                "longitude": [5.0],
                "value": [1.0],
            }
        )
        tif_path = str(tmp_path / "zero_lat.tif")
        with pytest.raises(ValueError, match="inverted|zero-width"):
            _write_geotiff(
                df,
                tif_path,
                "value",
                resolution=0.1,
                lat_range=(5.0, 5.0),
            )


# ---------------------------------------------------------------------------
# CLI edge cases
# ---------------------------------------------------------------------------


class TestCLIEdgeCases:
    """CLI argument parsing edge cases."""

    def test_time_scale_hourly_accepted(self) -> None:
        """--time-scale hourly should be accepted."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--input-dir",
                "/tmp/in",
                "--run-id",
                "test",
                "--time-scale",
                "hourly",
            ]
        )
        assert args.time_scale == "hourly"

    def test_hourly_agg_choices(self) -> None:
        """All hourly_agg choices should be accepted."""
        parser = build_parser()
        for method in ["mean", "median", "max", "min", "sum"]:
            args = parser.parse_args(
                [
                    "--input-dir",
                    "/tmp/in",
                    "--run-id",
                    "test",
                    "--hourly-agg",
                    method,
                ]
            )
            assert args.hourly_agg == method

    def test_invalid_stat_rejected(self) -> None:
        """Invalid stat name should be rejected by argparse."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--input-dir",
                    "/tmp/in",
                    "--run-id",
                    "test",
                    "--stats",
                    "variance",
                ]
            )

    def test_value_columns_multiple(self) -> None:
        """--value-columns should accept multiple model columns."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--input-dir",
                "/tmp/in",
                "--run-id",
                "test",
                "--value-columns",
                "sap_velocity_xgb",
                "sap_velocity_rf",
            ]
        )
        assert args.value_columns == ["sap_velocity_xgb", "sap_velocity_rf"]


# ---------------------------------------------------------------------------
# _sanitize_filename edge cases
# ---------------------------------------------------------------------------


class TestSanitizeFilenameEdgeCases:
    """Edge cases for filename sanitization."""

    def test_very_long_input_truncated(self) -> None:
        """Input longer than 200 chars should be truncated."""
        long_ts = "2015-07-01_" * 50  # 550 chars
        result = _sanitize_filename(long_ts)
        assert len(result) <= 200

    def test_all_special_chars_returns_unknown(self) -> None:
        """Input with only special chars should return 'unknown'."""
        result = _sanitize_filename("@#$%^&*()")
        assert result == "unknown"

    def test_dots_only_returns_unknown(self) -> None:
        """Input of only dots should return 'unknown' (after lstrip)."""
        result = _sanitize_filename("...")
        assert result == "unknown"

    def test_hyphens_only_returns_unknown(self) -> None:
        """Input of only hyphens should return 'unknown' (after lstrip)."""
        result = _sanitize_filename("---")
        assert result == "unknown"
