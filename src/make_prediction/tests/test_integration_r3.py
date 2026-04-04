"""Round 6 tests: Integration, regression for review fixes, cross-module tests.

Tests cover:
- End-to-end pipeline with all fixes active
- Composite min/max correctness (regression for M1 CRS fix)
- Multi-file batch processing
- CLI arg parsing round-trip with new options
- Welford streaming vs numpy reference for all stats
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
    build_parser,
    compute_composites,
    main,
    run_batch,
)

# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Full pipeline integration with all review fixes active."""

    def test_pipeline_with_explicit_ranges(self, tmp_path: Path) -> None:
        """End-to-end: explicit lat/lon range -> GeoTIFFs -> composites."""
        np.random.seed(42)
        rows = []
        for ts in ["2015-07-01", "2015-07-02"]:
            for _ in range(50):
                rows.append(
                    {
                        "timestamp.1": ts,
                        "latitude": round(np.random.uniform(-10, 10), 1),
                        "longitude": round(np.random.uniform(-10, 10), 1),
                        "sw_in": np.random.uniform(20, 400),
                        "sap_velocity_cnn_lstm": np.random.uniform(0.5, 4.0),
                    }
                )
        df = pd.DataFrame(rows)
        in_dir = str(tmp_path / "in")
        os.makedirs(in_dir)
        df.to_csv(os.path.join(in_dir, "predictions.csv"), index=False)

        run_batch(
            input_dir=in_dir,
            output_dir=str(tmp_path / "out"),
            run_id="e2e_test",
            file_glob="*predictions*.csv",
            sw_in_threshold=15.0,
            resolution=0.5,
            lat_range=(-10.0, 10.0),
            lon_range=(-10.0, 10.0),
            stats=("mean", "count"),
            render_png=False,
        )

        run_dir = tmp_path / "out" / "e2e_test"
        assert run_dir.exists()
        tifs = list(run_dir.glob("sap_velocity_*.tif"))
        assert len(tifs) == 2, f"Expected 2 timestamp TIFs, got {len(tifs)}"

        comp_dir = run_dir / "composites"
        assert comp_dir.exists()
        comp_tifs = list(comp_dir.glob("*.tif"))
        assert len(comp_tifs) >= 2

        # Validate composite mean is within data range
        with rasterio.open(str(comp_dir / "sap_velocity_composite_mean.tif")) as src:
            data = src.read(1)
            valid = data[~np.isnan(data)]
            assert len(valid) > 0
            assert np.all(valid >= 0), "Negative values in mean composite"
            assert np.all(valid <= 5), "Values exceed data range in composite"

    def test_multi_file_batch(self, tmp_path: Path) -> None:
        """Multiple CSV files in input dir should all be processed."""
        np.random.seed(99)
        in_dir = str(tmp_path / "multi_in")
        os.makedirs(in_dir)

        for day in range(1, 4):
            rows = []
            for _ in range(20):
                rows.append(
                    {
                        "timestamp.1": f"2015-07-{day:02d}",
                        "latitude": round(np.random.uniform(0, 5), 1),
                        "longitude": round(np.random.uniform(0, 5), 1),
                        "sap_velocity_cnn_lstm": np.random.uniform(1, 3),
                    }
                )
            df = pd.DataFrame(rows)
            df.to_csv(
                os.path.join(in_dir, f"predictions_day{day}.csv"),
                index=False,
            )

        run_batch(
            input_dir=in_dir,
            output_dir=str(tmp_path / "out"),
            run_id="multi_file",
            file_glob="predictions_*.csv",
            sw_in_threshold=0,
            resolution=1.0,
            stats=("mean",),
            render_png=False,
        )

        run_dir = tmp_path / "out" / "multi_file"
        tifs = list(run_dir.glob("sap_velocity_*.tif"))
        # 3 files × 1 timestamp each = 3 TIFs
        assert len(tifs) == 3, f"Expected 3 TIFs from 3 files, got {len(tifs)}"


# ---------------------------------------------------------------------------
# Welford streaming accuracy
# ---------------------------------------------------------------------------


class TestWelfordAccuracy:
    """Verify Welford streaming stats match numpy reference."""

    @pytest.fixture
    def reference_tifs(self, tmp_path: Path) -> list[str]:
        """Create 10 TIFs with known random data for stat comparison."""
        np.random.seed(77)
        h, w = 8, 10
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
        for i in range(10):
            data = np.random.uniform(0, 10, (h, w)).astype(np.float32)
            # Add some NaN
            data[0, 0] = np.nan
            if i % 3 == 0:
                data[2, 3] = np.nan
            path = str(tmp_path / f"welford_{i}.tif")
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data, 1)
            tifs.append(path)
        return tifs

    def test_mean_matches_numpy(
        self,
        reference_tifs: list[str],
        tmp_path: Path,
    ) -> None:
        """Welford mean should match np.nanmean within float32 tolerance."""
        # Compute reference
        arrays = []
        for path in reference_tifs:
            with rasterio.open(path) as src:
                arrays.append(src.read(1).astype(np.float64))
        ref_mean = np.nanmean(np.stack(arrays), axis=0).astype(np.float32)

        out_dir = str(tmp_path / "welford_out")
        results = compute_composites(
            tif_paths=reference_tifs,
            output_dir=out_dir,
            stats=("mean",),
            render_png=False,
        )
        with rasterio.open(results["mean"]) as src:
            actual = src.read(1)

        mask = ~np.isnan(ref_mean) & ~np.isnan(actual)
        np.testing.assert_allclose(
            actual[mask],
            ref_mean[mask],
            atol=1e-4,
            err_msg="Welford mean diverges from numpy reference",
        )

    def test_std_matches_numpy(
        self,
        reference_tifs: list[str],
        tmp_path: Path,
    ) -> None:
        """Welford std should match np.nanstd(ddof=1) within tolerance."""
        arrays = []
        for path in reference_tifs:
            with rasterio.open(path) as src:
                arrays.append(src.read(1).astype(np.float64))
        stack = np.stack(arrays)
        ref_std = np.nanstd(stack, axis=0, ddof=1).astype(np.float32)

        out_dir = str(tmp_path / "std_out")
        results = compute_composites(
            tif_paths=reference_tifs,
            output_dir=out_dir,
            stats=("std",),
            render_png=False,
        )
        with rasterio.open(results["std"]) as src:
            actual = src.read(1)

        mask = ~np.isnan(ref_std) & ~np.isnan(actual)
        np.testing.assert_allclose(
            actual[mask],
            ref_std[mask],
            atol=1e-3,
            err_msg="Welford std diverges from numpy reference",
        )

    def test_min_max_match_numpy(
        self,
        reference_tifs: list[str],
        tmp_path: Path,
    ) -> None:
        """Streaming min/max should exactly match numpy reference."""
        arrays = []
        for path in reference_tifs:
            with rasterio.open(path) as src:
                arrays.append(src.read(1).astype(np.float64))
        stack = np.stack(arrays)
        ref_min = np.nanmin(stack, axis=0).astype(np.float32)
        ref_max = np.nanmax(stack, axis=0).astype(np.float32)

        out_dir = str(tmp_path / "minmax_out")
        results = compute_composites(
            tif_paths=reference_tifs,
            output_dir=out_dir,
            stats=("min", "max"),
            render_png=False,
        )
        with rasterio.open(results["min"]) as src:
            actual_min = src.read(1)
        with rasterio.open(results["max"]) as src:
            actual_max = src.read(1)

        mask = ~np.isnan(ref_min) & ~np.isnan(actual_min)
        np.testing.assert_allclose(
            actual_min[mask],
            ref_min[mask],
            atol=1e-5,
            err_msg="Streaming min diverges from numpy",
        )
        np.testing.assert_allclose(
            actual_max[mask],
            ref_max[mask],
            atol=1e-5,
            err_msg="Streaming max diverges from numpy",
        )


# ---------------------------------------------------------------------------
# CLI round-trip
# ---------------------------------------------------------------------------


class TestCLIRoundTrip:
    """Test CLI parsing with new options."""

    def test_hourly_options_parsed(self) -> None:
        """--time-scale hourly + --hourly-agg median should parse correctly."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--input-dir",
                "/tmp/in",
                "--run-id",
                "hourly_test",
                "--time-scale",
                "hourly",
                "--hourly-agg",
                "median",
                "--hourly-maps",
            ]
        )
        assert args.time_scale == "hourly"
        assert args.hourly_agg == "median"
        assert args.hourly_maps is True

    def test_multi_value_columns_parsed(self) -> None:
        """--value-columns should accept multiple values."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--input-dir",
                "/tmp/in",
                "--run-id",
                "multi",
                "--value-columns",
                "sap_velocity_xgb",
                "sap_velocity_rf",
            ]
        )
        assert args.value_columns == ["sap_velocity_xgb", "sap_velocity_rf"]

    def test_main_with_argv(self, tmp_path: Path) -> None:
        """main() should accept argv list and run successfully."""
        np.random.seed(42)
        in_dir = str(tmp_path / "cli_in")
        os.makedirs(in_dir)
        df = pd.DataFrame(
            {
                "timestamp.1": ["2015-07-01"] * 10,
                "latitude": np.linspace(0, 5, 10),
                "longitude": np.linspace(0, 5, 10),
                "sap_velocity_cnn_lstm": np.random.uniform(0, 5, 10),
            }
        )
        df.to_csv(os.path.join(in_dir, "test_predictions.csv"), index=False)

        main(
            argv=[
                "--input-dir",
                in_dir,
                "--output-dir",
                str(tmp_path / "cli_out"),
                "--run-id",
                "cli_test",
                "--sw-threshold",
                "0",
                "--resolution",
                "1.0",
                "--stats",
                "mean",
                "--no-png",
            ]
        )

        run_dir = tmp_path / "cli_out" / "cli_test"
        assert run_dir.exists()
        tifs = list(run_dir.glob("sap_velocity_*.tif"))
        assert len(tifs) >= 1
