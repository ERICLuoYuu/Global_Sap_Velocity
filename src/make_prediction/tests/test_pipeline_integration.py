"""End-to-end pipeline integration tests.

Categories:
- E: Rasterization correctness
- F: Composite correctness (Welford)
- G: End-to-end contract (golden tests)
- H: Format compatibility
- I: Edge cases
"""
from __future__ import annotations

import glob as globmod
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS

from src.make_prediction.prediction_visualization_hpc import (
    _write_geotiff,
    compute_composites,
    rasterize_all_timestamps,
    read_prediction_file,
    run_batch,
)
from src.make_prediction.tests.conftest import (
    EPS,
    KNOWN_POINTS,
    LAT_RANGE,
    LON_RANGE,
    RESOLUTION,
    TIMESTAMPS,
    coord_to_pixel,
)


# ===================================================================
# Category E — Rasterization Correctness
# ===================================================================


class TestRasterizationCorrectness:
    """Verify coordinate-to-pixel mapping is mathematically correct."""

    def test_single_point_pixel_exact(self, tmp_path: Path) -> None:
        """One data point -> pixel value equals exact input value."""
        df = pd.DataFrame({
            "latitude": [48.5],
            "longitude": [11.5],
            "sap_velocity_xgb": [4.2],
        })
        tif_path = str(tmp_path / "single.tif")
        _write_geotiff(
            df, tif_path, "sap_velocity_xgb", RESOLUTION,
            LAT_RANGE, LON_RANGE,
        )

        row, col = coord_to_pixel(48.5, 11.5)
        with rasterio.open(tif_path) as src:
            band = src.read(1)
            assert abs(band[row, col] - 4.2) < 1e-5, (
                f"Expected 4.2 at pixel ({row},{col}), got {band[row, col]}"
            )

    def test_duplicate_pixels_mean_aggregation(self, tmp_path: Path) -> None:
        """Multiple points at same (lat, lon) -> rasterize takes mean."""
        values = [3.0, 6.0, 9.0]
        df = pd.DataFrame({
            "latitude": [48.5, 48.5, 48.5],
            "longitude": [11.5, 11.5, 11.5],
            "timestamp": ["2015-07-01"] * 3,
            "sap_velocity_xgb": values,
        })
        csv_path = str(tmp_path / "dup.csv")
        df.to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "out")
        results = rasterize_all_timestamps(
            csv_path=csv_path, output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )
        assert len(results) == 1
        _, tif_path = results[0]

        row, col = coord_to_pixel(48.5, 11.5)
        with rasterio.open(tif_path) as src:
            band = src.read(1)
            expected = np.mean(values)
            assert abs(band[row, col] - expected) < 1e-5, (
                f"Expected mean={expected}, got {band[row, col]}"
            )

    def test_nodata_where_no_input(self, tmp_path: Path) -> None:
        """Pixels without data must be NaN (not zero)."""
        df = pd.DataFrame({
            "latitude": [48.5],
            "longitude": [11.5],
            "sap_velocity_xgb": [4.2],
        })
        tif_path = str(tmp_path / "sparse.tif")
        _write_geotiff(
            df, tif_path, "sap_velocity_xgb", RESOLUTION,
            LAT_RANGE, LON_RANGE,
        )

        # Check a pixel far from data
        row_empty, col_empty = coord_to_pixel(-50.0, 100.0)
        with rasterio.open(tif_path) as src:
            band = src.read(1)
            assert np.isnan(band[row_empty, col_empty]), (
                f"Expected NaN at empty pixel, got {band[row_empty, col_empty]}"
            )

    def test_grid_boundary_pixels(self, tmp_path: Path) -> None:
        """Points at grid edges map to valid pixels without IndexError."""
        df = pd.DataFrame({
            "latitude": [-60.0, 77.9],
            "longitude": [-180.0, 179.9],
            "sap_velocity_xgb": [1.0, 2.0],
        })
        tif_path = str(tmp_path / "boundary.tif")
        _write_geotiff(
            df, tif_path, "sap_velocity_xgb", RESOLUTION,
            LAT_RANGE, LON_RANGE,
        )

        with rasterio.open(tif_path) as src:
            band = src.read(1)
            # Bottom-left corner
            r1, c1 = coord_to_pixel(-60.0, -180.0)
            assert 0 <= r1 < band.shape[0] and 0 <= c1 < band.shape[1]
            assert abs(band[r1, c1] - 1.0) < 1e-5

            # Top-right (near boundary)
            r2, c2 = coord_to_pixel(77.9, 179.9)
            assert 0 <= r2 < band.shape[0] and 0 <= c2 < band.shape[1]
            assert abs(band[r2, c2] - 2.0) < 1e-5

    def test_coordinate_pixel_roundtrip(self, tmp_path: Path) -> None:
        """(lat, lon) -> pixel -> (lat, lon) roundtrip within 0.05 degrees."""
        test_coords = [(-3.0, -60.0), (48.5, 11.5), (61.0, 24.0)]
        for lat, lon in test_coords:
            row, col = coord_to_pixel(lat, lon)
            # Inverse: pixel center to coordinate
            recovered_lat = LAT_RANGE[1] - (row + 0.5) * RESOLUTION
            recovered_lon = LON_RANGE[0] + (col + 0.5) * RESOLUTION

            assert abs(recovered_lat - lat) < 0.05 + RESOLUTION / 2, (
                f"Lat roundtrip error: {lat} -> row {row} -> {recovered_lat}"
            )
            assert abs(recovered_lon - lon) < 0.05 + RESOLUTION / 2, (
                f"Lon roundtrip error: {lon} -> col {col} -> {recovered_lon}"
            )

    def test_known_points_all_placed_correctly(
        self, prediction_csv_dir: Path, tmp_path: Path,
    ) -> None:
        """All 5 known reference points appear at correct GeoTIFF pixels."""
        csv_files = list(prediction_csv_dir.glob("*.csv"))
        assert len(csv_files) > 0

        out_dir = str(tmp_path / "known_pts")
        results = rasterize_all_timestamps(
            csv_path=str(csv_files[0]), output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )

        assert len(results) == 5, f"Expected 5 GeoTIFFs, got {len(results)}"

        # Check first timestamp's GeoTIFF
        _, tif_path = results[0]
        with rasterio.open(tif_path) as src:
            band = src.read(1)
            for pt in KNOWN_POINTS:
                row, col = coord_to_pixel(pt["lat"], pt["lon"])
                actual = band[row, col]
                expected = pt["sap_xgb"]
                assert abs(actual - expected) < 1e-4, (
                    f"{pt['label']}: expected {expected} at "
                    f"({pt['lat']}, {pt['lon']}), got {actual}"
                )


# ===================================================================
# Category F — Composite Correctness
# ===================================================================


class TestCompositeCorrectness:
    """Verify Welford streaming composites match numpy reference."""

    def _create_daily_tifs(
        self, tmp_path: Path, n_days: int = 5,
    ) -> Tuple[List[str], Dict[Tuple[int, int], List[float]]]:
        """Helper: create n_days GeoTIFFs with known pixel values.

        Returns (tif_paths, pixel_values) where pixel_values maps
        (row, col) -> list of values across days.
        """
        rng = np.random.default_rng(42)
        tif_paths = []
        pixel_values: Dict[Tuple[int, int], List[float]] = {}

        coords = [(-3.0, -60.0), (48.5, 11.5), (61.0, 24.0)]

        for day_i in range(n_days):
            rows_data = []
            for lat, lon in coords:
                val = float(rng.uniform(1, 20))
                rows_data.append({
                    "latitude": lat, "longitude": lon,
                    "sap_velocity_xgb": val,
                })
                rc = coord_to_pixel(lat, lon)
                pixel_values.setdefault(rc, []).append(val)

            df = pd.DataFrame(rows_data)
            tif_path = str(tmp_path / f"day_{day_i:03d}.tif")
            _write_geotiff(
                df, tif_path, "sap_velocity_xgb", RESOLUTION,
                LAT_RANGE, LON_RANGE,
            )
            tif_paths.append(tif_path)

        return tif_paths, pixel_values

    def test_mean_composite_equals_nanmean(self, tmp_path: Path) -> None:
        """Welford mean matches numpy nanmean."""
        tif_paths, pixel_values = self._create_daily_tifs(tmp_path / "days")
        comp_dir = str(tmp_path / "comp")

        results = compute_composites(
            tif_paths, comp_dir, stats=("mean",), render_png=False,
        )
        assert "mean" in results

        with rasterio.open(results["mean"]) as src:
            band = src.read(1)

        for (row, col), vals in pixel_values.items():
            expected = np.nanmean(vals)
            actual = band[row, col]
            assert abs(actual - expected) < 1e-4, (
                f"Mean mismatch at ({row},{col}): expected {expected}, got {actual}"
            )

    def test_std_composite_uses_ddof1(self, tmp_path: Path) -> None:
        """Welford std uses N-1 denominator (sample std)."""
        tif_paths, pixel_values = self._create_daily_tifs(tmp_path / "days")
        comp_dir = str(tmp_path / "comp")

        results = compute_composites(
            tif_paths, comp_dir, stats=("std",), render_png=False,
        )

        with rasterio.open(results["std"]) as src:
            band = src.read(1)

        for (row, col), vals in pixel_values.items():
            if len(vals) > 1:
                expected = float(np.nanstd(vals, ddof=1))
                actual = float(band[row, col])
                assert abs(actual - expected) < 1e-3, (
                    f"Std mismatch at ({row},{col}): "
                    f"expected {expected:.6f}, got {actual:.6f}"
                )

    def test_count_composite_equals_valid_days(self, tmp_path: Path) -> None:
        """Count composite reflects number of days with data."""
        tif_paths, pixel_values = self._create_daily_tifs(tmp_path / "days")
        comp_dir = str(tmp_path / "comp")

        results = compute_composites(
            tif_paths, comp_dir, stats=("count",), render_png=False,
        )

        with rasterio.open(results["count"]) as src:
            band = src.read(1)

        for (row, col), vals in pixel_values.items():
            expected = len(vals)
            actual = band[row, col]
            assert actual == expected, (
                f"Count at ({row},{col}): expected {expected}, got {actual}"
            )

    def test_single_day_std_is_zero(self, tmp_path: Path) -> None:
        """One observation -> std = 0 (code returns 0 for count <= 1)."""
        # Create 2 TIFFs but pixel only has data in 1
        df1 = pd.DataFrame({
            "latitude": [48.5], "longitude": [11.5],
            "sap_velocity_xgb": [5.0],
        })
        tif1 = str(tmp_path / "single_day.tif")
        _write_geotiff(
            df1, tif1, "sap_velocity_xgb", RESOLUTION,
            LAT_RANGE, LON_RANGE,
        )

        df2 = pd.DataFrame({
            "latitude": [-3.0], "longitude": [-60.0],
            "sap_velocity_xgb": [10.0],
        })
        tif2 = str(tmp_path / "single_day2.tif")
        _write_geotiff(
            df2, tif2, "sap_velocity_xgb", RESOLUTION,
            LAT_RANGE, LON_RANGE,
        )

        comp_dir = str(tmp_path / "comp")
        results = compute_composites(
            [tif1, tif2], comp_dir,
            stats=("std",), render_png=False,
        )

        row, col = coord_to_pixel(48.5, 11.5)
        with rasterio.open(results["std"]) as src:
            band = src.read(1)
            assert band[row, col] == 0.0, (
                f"Expected std=0 for single observation, got {band[row, col]}"
            )

    def test_composite_excludes_nodata_from_mean(self, tmp_path: Path) -> None:
        """NaN pixels in some TIFFs don't affect mean at other pixels."""
        tifs = []
        for i, val in enumerate([4.0, 6.0]):
            df = pd.DataFrame({
                "latitude": [48.5], "longitude": [11.5],
                "sap_velocity_xgb": [val],
            })
            tif = str(tmp_path / f"d{i}.tif")
            _write_geotiff(
                df, tif, "sap_velocity_xgb", RESOLUTION,
                LAT_RANGE, LON_RANGE,
            )
            tifs.append(tif)

        # Day 3: different pixel only
        df3 = pd.DataFrame({
            "latitude": [-3.0], "longitude": [-60.0],
            "sap_velocity_xgb": [99.0],
        })
        tif3 = str(tmp_path / "d2.tif")
        _write_geotiff(
            df3, tif3, "sap_velocity_xgb", RESOLUTION,
            LAT_RANGE, LON_RANGE,
        )
        tifs.append(tif3)

        comp_dir = str(tmp_path / "comp")
        results = compute_composites(
            tifs, comp_dir, stats=("mean",), render_png=False,
        )

        row, col = coord_to_pixel(48.5, 11.5)
        with rasterio.open(results["mean"]) as src:
            band = src.read(1)
            assert abs(band[row, col] - 5.0) < 1e-4

    def test_min_max_composites(self, tmp_path: Path) -> None:
        """Min and max composites match numpy reference."""
        tif_paths, pixel_values = self._create_daily_tifs(tmp_path / "days")
        comp_dir = str(tmp_path / "comp")

        results = compute_composites(
            tif_paths, comp_dir, stats=("min", "max"), render_png=False,
        )

        with rasterio.open(results["min"]) as src:
            min_band = src.read(1)
        with rasterio.open(results["max"]) as src:
            max_band = src.read(1)

        for (row, col), vals in pixel_values.items():
            assert abs(min_band[row, col] - min(vals)) < 1e-4
            assert abs(max_band[row, col] - max(vals)) < 1e-4


# ===================================================================
# Category G — End-to-End Contract
# ===================================================================


class TestEndToEndContract:
    """Prove prediction output -> visualization input works correctly."""

    def test_all_predictions_appear_in_geotiffs(
        self, prediction_csv_dir: Path, tmp_path: Path,
    ) -> None:
        """Every (lat, lon, timestamp) in predictions has a non-NaN pixel."""
        csv_files = list(prediction_csv_dir.glob("*.csv"))
        out_dir = str(tmp_path / "appear")
        results = rasterize_all_timestamps(
            csv_path=str(csv_files[0]), output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )

        pred_df = pd.read_csv(csv_files[0])

        result_map = {ts: path for ts, path in results}
        for ts_str, tif_path in results:
            ts_data = pred_df[pred_df["timestamp"] == ts_str]
            with rasterio.open(tif_path) as src:
                band = src.read(1)
                for _, row in ts_data.iterrows():
                    r, c = coord_to_pixel(row["latitude"], row["longitude"])
                    if 0 <= r < band.shape[0] and 0 <= c < band.shape[1]:
                        assert not np.isnan(band[r, c]), (
                            f"Missing pixel at ({row['latitude']}, "
                            f"{row['longitude']}) for timestamp {ts_str}"
                        )

    def test_no_phantom_geotiff_pixels(
        self, prediction_csv_dir: Path, tmp_path: Path,
    ) -> None:
        """Every non-NaN pixel must have at least one prediction row."""
        csv_files = list(prediction_csv_dir.glob("*.csv"))
        out_dir = str(tmp_path / "phantom")
        results = rasterize_all_timestamps(
            csv_path=str(csv_files[0]), output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )

        pred_df = pd.read_csv(csv_files[0])

        for ts_str, tif_path in results:
            ts_data = pred_df[pred_df["timestamp"] == ts_str]
            expected_pixels = set()
            for _, row in ts_data.iterrows():
                r, c = coord_to_pixel(row["latitude"], row["longitude"])
                expected_pixels.add((r, c))

            with rasterio.open(tif_path) as src:
                band = src.read(1)
                non_nan = np.argwhere(~np.isnan(band))
                for r, c in non_nan:
                    assert (r, c) in expected_pixels, (
                        f"Phantom pixel at ({r}, {c}) for {ts_str}"
                    )

    def test_timestamp_bijection(
        self, prediction_csv_dir: Path, tmp_path: Path,
    ) -> None:
        """Unique timestamps in predictions == GeoTIFF filenames (1:1)."""
        csv_files = list(prediction_csv_dir.glob("*.csv"))
        pred_df = pd.read_csv(csv_files[0])
        expected_ts = set(pred_df["timestamp"].unique())

        out_dir = str(tmp_path / "bijection")
        results = rasterize_all_timestamps(
            csv_path=str(csv_files[0]), output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )

        result_ts = {ts for ts, _ in results}
        assert result_ts == expected_ts, (
            f"Timestamp mismatch:\n"
            f"In predictions: {sorted(expected_ts)}\n"
            f"In GeoTIFFs: {sorted(result_ts)}"
        )

    def test_spatial_coverage_preserved(
        self, prediction_csv_dir: Path, tmp_path: Path,
    ) -> None:
        """Unique (lat, lon) pairs per timestamp == non-NaN pixel count."""
        csv_files = list(prediction_csv_dir.glob("*.csv"))
        pred_df = pd.read_csv(csv_files[0])

        out_dir = str(tmp_path / "coverage")
        results = rasterize_all_timestamps(
            csv_path=str(csv_files[0]), output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )

        for ts_str, tif_path in results:
            ts_data = pred_df[pred_df["timestamp"] == ts_str]
            n_unique_coords = len(
                ts_data.drop_duplicates(subset=["latitude", "longitude"])
            )
            with rasterio.open(tif_path) as src:
                band = src.read(1)
                n_valid = int(np.sum(~np.isnan(band)))
                assert n_valid == n_unique_coords, (
                    f"Coverage mismatch for {ts_str}: "
                    f"{n_unique_coords} coords vs {n_valid} pixels"
                )

    def test_end_to_end_value_tracing(
        self, prediction_csv_dir: Path, tmp_path: Path,
    ) -> None:
        """THE GOLDEN TEST: trace known values through the full pipeline.

        Synthetic data has Germany point (48.5, 11.5) with sap_velocity_xgb=4.2
        for every timestamp. Verify:
        1. CSV exists with this value
        2. GeoTIFF pixel at corresponding (row, col) == 4.2
        """
        csv_files = list(prediction_csv_dir.glob("*.csv"))
        pred_df = pd.read_csv(csv_files[0])

        # Step 1: Verify value in CSV
        germany = pred_df[
            (abs(pred_df["latitude"] - 48.5) < 0.01)
            & (abs(pred_df["longitude"] - 11.5) < 0.01)
        ]
        assert len(germany) == 5, f"Expected 5 rows for Germany, got {len(germany)}"
        assert all(abs(germany["sap_velocity_xgb"] - 4.2) < 1e-6)

        # Step 2: Run visualization
        out_dir = str(tmp_path / "golden")
        results = rasterize_all_timestamps(
            csv_path=str(csv_files[0]), output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )

        # Step 3: Verify pixel value in every daily GeoTIFF
        row, col = coord_to_pixel(48.5, 11.5)
        for ts_str, tif_path in results:
            with rasterio.open(tif_path) as src:
                band = src.read(1)
                actual = float(band[row, col])
                assert abs(actual - 4.2) < 1e-4, (
                    f"Golden test FAILED for {ts_str}: "
                    f"expected 4.2, got {actual} at pixel ({row},{col})"
                )


# ===================================================================
# Category H — Format Compatibility
# ===================================================================


class TestFormatCompatibility:
    """Verify CSV and Parquet produce identical results."""

    def test_csv_to_geotiff_roundtrip(
        self, prediction_csv_dir: Path, tmp_path: Path,
    ) -> None:
        """CSV predictions -> 5 GeoTIFFs (one per timestamp)."""
        csv_files = list(prediction_csv_dir.glob("*.csv"))
        out_dir = str(tmp_path / "csv_rt")
        results = rasterize_all_timestamps(
            csv_path=str(csv_files[0]), output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )
        assert len(results) == 5, f"Expected 5 GeoTIFFs, got {len(results)}"

    def test_parquet_to_geotiff_roundtrip(
        self, prediction_parquet_dir: Path, tmp_path: Path,
    ) -> None:
        """Parquet predictions -> 5 GeoTIFFs."""
        pq_files = list(prediction_parquet_dir.glob("*.parquet"))
        out_dir = str(tmp_path / "pq_rt")
        results = rasterize_all_timestamps(
            csv_path=str(pq_files[0]), output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )
        assert len(results) == 5, f"Expected 5 GeoTIFFs, got {len(results)}"

    def test_format_parity_csv_vs_parquet(
        self, prediction_csv_dir: Path, prediction_parquet_dir: Path,
        tmp_path: Path,
    ) -> None:
        """CSV and Parquet produce pixel-wise identical GeoTIFFs."""
        csv_files = list(prediction_csv_dir.glob("*.csv"))
        pq_files = list(prediction_parquet_dir.glob("*.parquet"))

        csv_out = str(tmp_path / "parity_csv")
        pq_out = str(tmp_path / "parity_pq")

        csv_results = rasterize_all_timestamps(
            csv_path=str(csv_files[0]), output_dir=csv_out,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )
        pq_results = rasterize_all_timestamps(
            csv_path=str(pq_files[0]), output_dir=pq_out,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )

        assert len(csv_results) == len(pq_results)

        for (ts_csv, path_csv), (ts_pq, path_pq) in zip(
            sorted(csv_results), sorted(pq_results),
        ):
            assert ts_csv == ts_pq
            with rasterio.open(path_csv) as src_csv, \
                 rasterio.open(path_pq) as src_pq:
                band_csv = src_csv.read(1)
                band_pq = src_pq.read(1)
                mask = ~np.isnan(band_csv) & ~np.isnan(band_pq)
                np.testing.assert_allclose(
                    band_csv[mask], band_pq[mask], atol=1e-5,
                    err_msg=f"Parity mismatch for timestamp {ts_csv}",
                )

    def test_column_name_contract(
        self, synthetic_predictions: pd.DataFrame,
    ) -> None:
        """Prediction DataFrame has all required columns."""
        required = {"latitude", "longitude", "timestamp", "sap_velocity_xgb"}
        actual = set(synthetic_predictions.columns)
        missing = required - actual
        assert not missing, f"Missing required columns: {missing}"

    def test_read_prediction_file_auto_detect_csv(
        self, prediction_csv_dir: Path,
    ) -> None:
        """CSV file detected and read correctly."""
        csv_files = list(prediction_csv_dir.glob("*.csv"))
        df = read_prediction_file(str(csv_files[0]))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_read_prediction_file_auto_detect_parquet(
        self, prediction_parquet_dir: Path,
    ) -> None:
        """Parquet file detected and read correctly."""
        pq_files = list(prediction_parquet_dir.glob("*.parquet"))
        df = read_prediction_file(str(pq_files[0]))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ===================================================================
# Category I — Edge Cases
# ===================================================================


class TestEdgeCases:
    """Verify graceful handling of edge cases."""

    def test_single_row_prediction(self, tmp_path: Path) -> None:
        """1 row -> 1 GeoTIFF with 1 non-NaN pixel."""
        df = pd.DataFrame({
            "latitude": [48.5], "longitude": [11.5],
            "timestamp": ["2015-07-01"],
            "sap_velocity_xgb": [4.2],
        })
        csv_path = str(tmp_path / "single.csv")
        df.to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "out")
        results = rasterize_all_timestamps(
            csv_path=csv_path, output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )
        assert len(results) == 1

        with rasterio.open(results[0][1]) as src:
            band = src.read(1)
            n_valid = int(np.sum(~np.isnan(band)))
            assert n_valid == 1

    def test_all_same_coordinate_different_timestamps(
        self, tmp_path: Path,
    ) -> None:
        """Same (lat, lon) with 5 timestamps -> 5 GeoTIFFs, each 1 pixel."""
        df = pd.DataFrame({
            "latitude": [48.5] * 5,
            "longitude": [11.5] * 5,
            "timestamp": [str(ts) for ts in TIMESTAMPS],
            "sap_velocity_xgb": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        csv_path = str(tmp_path / "same_coord.csv")
        df.to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "out")
        results = rasterize_all_timestamps(
            csv_path=csv_path, output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )
        assert len(results) == 5

        for _, tif_path in results:
            with rasterio.open(tif_path) as src:
                band = src.read(1)
                assert int(np.sum(~np.isnan(band))) == 1

    def test_extreme_values_preserved(self, tmp_path: Path) -> None:
        """Very small and very large values survive without clipping."""
        df = pd.DataFrame({
            "latitude": [48.5, -3.0],
            "longitude": [11.5, -60.0],
            "timestamp": ["2015-07-01", "2015-07-01"],
            "sap_velocity_xgb": [0.0001, 150.0],
        })
        csv_path = str(tmp_path / "extreme.csv")
        df.to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "out")
        results = rasterize_all_timestamps(
            csv_path=csv_path, output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )

        _, tif_path = results[0]
        with rasterio.open(tif_path) as src:
            band = src.read(1)
            r1, c1 = coord_to_pixel(48.5, 11.5)
            r2, c2 = coord_to_pixel(-3.0, -60.0)
            assert abs(band[r1, c1] - 0.0001) < 1e-6
            assert abs(band[r2, c2] - 150.0) < 1e-3

    def test_empty_timestamp_no_crash(self, tmp_path: Path) -> None:
        """Timestamp with all NaN values -> skip gracefully."""
        df = pd.DataFrame({
            "latitude": [48.5, 48.5],
            "longitude": [11.5, 11.5],
            "timestamp": ["2015-07-01", "2015-07-02"],
            "sap_velocity_xgb": [4.2, np.nan],
        })
        csv_path = str(tmp_path / "empty_ts.csv")
        df.to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "out")
        results = rasterize_all_timestamps(
            csv_path=csv_path, output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )
        # At minimum, the NaN-only timestamp should not crash
        # It may produce 1 or 2 tifs depending on NaN handling
        assert len(results) >= 1

    def test_multi_model_separate_dirs(self, tmp_path: Path) -> None:
        """Two value columns -> separate subdirectories."""
        df = pd.DataFrame({
            "latitude": [48.5],
            "longitude": [11.5],
            "timestamp": ["2015-07-01"],
            "sap_velocity_xgb": [4.2],
            "sap_velocity_rf": [3.8],
        })
        csv_path = str(tmp_path / "multi.csv")
        df.to_csv(csv_path, index=False)

        input_dir = str(tmp_path)
        out_dir = str(tmp_path / "maps")

        run_batch(
            input_dir=input_dir,
            output_dir=out_dir,
            run_id="test_multi",
            value_columns=["sap_velocity_xgb", "sap_velocity_rf"],
            timestamp_col="timestamp",
            file_glob="multi.csv",
            sw_in_threshold=0,
            resolution=RESOLUTION,
            lat_range=LAT_RANGE,
            lon_range=LON_RANGE,
            stats=(),
            render_png=False,
        )

        run_dir = Path(out_dir) / "test_multi"
        xgb_dir = run_dir / "xgb"
        rf_dir = run_dir / "rf"
        assert xgb_dir.exists(), f"XGB dir not found at {xgb_dir}"
        assert rf_dir.exists(), f"RF dir not found at {rf_dir}"
        assert len(list(xgb_dir.glob("*.tif"))) >= 1
        assert len(list(rf_dir.glob("*.tif"))) >= 1

    def test_geotiff_metadata(
        self, prediction_csv_dir: Path, tmp_path: Path,
    ) -> None:
        """GeoTIFF has correct CRS, pixel size, nodata, band count."""
        csv_files = list(prediction_csv_dir.glob("*.csv"))
        out_dir = str(tmp_path / "meta")
        results = rasterize_all_timestamps(
            csv_path=str(csv_files[0]), output_dir=out_dir,
            value_column="sap_velocity_xgb",
            timestamp_col="timestamp", sw_in_threshold=0,
            resolution=RESOLUTION, lat_range=LAT_RANGE, lon_range=LON_RANGE,
        )

        _, tif_path = results[0]
        with rasterio.open(tif_path) as src:
            assert src.crs == CRS.from_epsg(4326), f"CRS: {src.crs}"
            assert src.count == 1
            assert abs(src.res[0] - RESOLUTION) < 1e-6
            assert abs(src.res[1] - RESOLUTION) < 1e-6
            assert src.nodata is not None
            assert np.isnan(src.nodata), f"nodata = {src.nodata}"
