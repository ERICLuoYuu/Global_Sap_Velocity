"""Round 4 tests: Review fix verification + edge cases for new guards.

Tests cover:
- C1: resolution <= 0 raises ValueError
- C2: inverted lat/lon ranges raise ValueError
- H1: corrupt TIF is regenerated on resume
- H2: median memory guard fallback
- H3: variance division no RuntimeWarning
- M1: CRS mismatch skips file in composites
- M2: narrowed except only catches expected errors
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from src.make_prediction.prediction_visualization_hpc import (
    _write_geotiff,
    compute_composites,
    rasterize_all_timestamps,
)

# ---------------------------------------------------------------------------
# C1: Resolution validation
# ---------------------------------------------------------------------------


class TestResolutionValidation:
    """C1 fix: _write_geotiff must reject resolution <= 0."""

    def test_zero_resolution_raises(self, tmp_path: Path) -> None:
        """resolution=0 should raise ValueError."""
        df = pd.DataFrame(
            {
                "latitude": [0.0, 1.0],
                "longitude": [0.0, 1.0],
                "value": [1.0, 2.0],
            }
        )
        tif_path = str(tmp_path / "zero_res.tif")
        with pytest.raises(ValueError, match="resolution must be positive"):
            _write_geotiff(df, tif_path, "value", resolution=0.0)

    def test_negative_resolution_raises(self, tmp_path: Path) -> None:
        """resolution=-0.1 should raise ValueError."""
        df = pd.DataFrame(
            {
                "latitude": [0.0, 1.0],
                "longitude": [0.0, 1.0],
                "value": [1.0, 2.0],
            }
        )
        tif_path = str(tmp_path / "neg_res.tif")
        with pytest.raises(ValueError, match="resolution must be positive"):
            _write_geotiff(df, tif_path, "value", resolution=-0.1)

    def test_very_small_positive_resolution_ok(self, tmp_path: Path) -> None:
        """Very small positive resolution should not raise."""
        df = pd.DataFrame(
            {
                "latitude": [0.0, 0.001],
                "longitude": [0.0, 0.001],
                "value": [1.0, 2.0],
            }
        )
        tif_path = str(tmp_path / "tiny_res.tif")
        _write_geotiff(df, tif_path, "value", resolution=0.001)
        assert os.path.exists(tif_path)


# ---------------------------------------------------------------------------
# C2: Range validation
# ---------------------------------------------------------------------------


class TestRangeValidation:
    """C2 fix: inverted explicit ranges must raise ValueError."""

    def test_inverted_lat_range_raises(self, tmp_path: Path) -> None:
        """lat_range=(78, -60) should raise ValueError."""
        df = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "value": [1.0],
            }
        )
        tif_path = str(tmp_path / "inv_lat.tif")
        with pytest.raises(ValueError, match="lat_range inverted"):
            _write_geotiff(
                df,
                tif_path,
                "value",
                resolution=0.1,
                lat_range=(78.0, -60.0),
            )

    def test_inverted_lon_range_raises(self, tmp_path: Path) -> None:
        """lon_range=(180, -180) should raise ValueError."""
        df = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "value": [1.0],
            }
        )
        tif_path = str(tmp_path / "inv_lon.tif")
        with pytest.raises(ValueError, match="lon_range inverted"):
            _write_geotiff(
                df,
                tif_path,
                "value",
                resolution=0.1,
                lat_range=(-60.0, 78.0),
                lon_range=(180.0, -180.0),
            )

    def test_equal_lat_range_raises(self, tmp_path: Path) -> None:
        """lat_range=(10.0, 10.0) should raise ValueError (zero-width)."""
        df = pd.DataFrame(
            {
                "latitude": [10.0],
                "longitude": [0.0],
                "value": [1.0],
            }
        )
        tif_path = str(tmp_path / "eq_lat.tif")
        with pytest.raises(ValueError, match="lat_range inverted or zero-width"):
            _write_geotiff(
                df,
                tif_path,
                "value",
                resolution=0.1,
                lat_range=(10.0, 10.0),
            )

    def test_data_derived_single_point_ok(self, tmp_path: Path) -> None:
        """Single point without explicit range should still produce 1-pixel TIF."""
        df = pd.DataFrame(
            {
                "latitude": [45.0],
                "longitude": [10.0],
                "value": [2.5],
            }
        )
        tif_path = str(tmp_path / "single_ok.tif")
        # No lat_range/lon_range -> data-derived, should NOT raise
        _write_geotiff(df, tif_path, "value", resolution=0.1)
        assert os.path.exists(tif_path)


# ---------------------------------------------------------------------------
# H1: Corrupt TIF resume
# ---------------------------------------------------------------------------


class TestCorruptTIFResume:
    """H1 fix: corrupted TIF should be regenerated, not skipped."""

    def test_corrupt_tif_regenerated(self, tmp_path: Path) -> None:
        """A truncated TIF should be detected and regenerated."""
        # Create valid CSV
        df = pd.DataFrame(
            {
                "timestamp.1": ["2015-07-01"] * 5,
                "latitude": [0.0, 1.0, 2.0, 3.0, 4.0],
                "longitude": [0.0, 1.0, 2.0, 3.0, 4.0],
                "sap_velocity_cnn_lstm": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "corrupt_test")
        os.makedirs(out_dir, exist_ok=True)

        # Create a corrupt TIF (just random bytes)
        corrupt_name = "sap_velocity_2015-07-01.tif"
        corrupt_path = os.path.join(out_dir, corrupt_name)
        with open(corrupt_path, "wb") as f:
            f.write(b"THIS IS NOT A VALID TIFF FILE\x00\x00")

        # Run rasterization - should regenerate the corrupt file
        results = rasterize_all_timestamps(
            csv_path=csv_path,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=1.0,
        )

        # Should have produced a valid result
        assert len(results) == 1
        _, tif_path = results[0]
        # The regenerated file should be readable
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            assert not np.all(np.isnan(data))

    def test_valid_tif_not_regenerated(self, tmp_path: Path) -> None:
        """A valid TIF should be skipped (not re-created)."""
        df = pd.DataFrame(
            {
                "timestamp.1": ["2015-07-01"] * 3,
                "latitude": [0.0, 1.0, 2.0],
                "longitude": [0.0, 1.0, 2.0],
                "sap_velocity_cnn_lstm": [1.0, 2.0, 3.0],
            }
        )
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)
        out_dir = str(tmp_path / "valid_test")

        # First run
        results1 = rasterize_all_timestamps(
            csv_path=csv_path,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=1.0,
        )
        assert len(results1) == 1
        _, tif_path = results1[0]
        mtime1 = os.path.getmtime(tif_path)

        # Second run — should skip
        results2 = rasterize_all_timestamps(
            csv_path=csv_path,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=1.0,
        )
        assert len(results2) == 1
        mtime2 = os.path.getmtime(results2[0][1])
        # File should NOT have been regenerated
        assert mtime1 == mtime2, "Valid TIF was regenerated unnecessarily"


# ---------------------------------------------------------------------------
# H2: Median memory guard
# ---------------------------------------------------------------------------


class TestMedianMemoryGuard:
    """H2 fix: median composite should fallback when stack too large."""

    def test_small_stack_uses_real_median(
        self,
        tmp_path: Path,
    ) -> None:
        """Small stack (well under 16GB) should compute real median."""
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
        # Values: 1, 2, 3 -> median = 2
        for i in range(3):
            data = np.full((h, w), float(i + 1), dtype=np.float32)
            path = str(tmp_path / f"med_{i}.tif")
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data, 1)
            tifs.append(path)

        out_dir = str(tmp_path / "med_out")
        results = compute_composites(
            tif_paths=tifs,
            output_dir=out_dir,
            stats=("median",),
            render_png=False,
        )
        with rasterio.open(results["median"]) as src:
            data = src.read(1)
        # Real median of [1, 2, 3] = 2
        assert data[1, 1] == pytest.approx(2.0, abs=0.01)


# ---------------------------------------------------------------------------
# H3: RuntimeWarning suppressed
# ---------------------------------------------------------------------------


class TestVarianceWarning:
    """H3 fix: variance division should not emit RuntimeWarning."""

    def test_no_runtime_warning_in_composites(
        self,
        tmp_path: Path,
    ) -> None:
        """compute_composites with mixed data should not warn about division."""
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
        tifs = []
        for i in range(3):
            data = np.full((h, w), np.nan, dtype=np.float32)
            data[0, 0] = float(i + 1)  # Only one pixel has data in all 3
            if i == 0:
                data[2, 2] = 10.0  # Only 1 observation here
            path = str(tmp_path / f"warn_{i}.tif")
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data, 1)
            tifs.append(path)

        out_dir = str(tmp_path / "warn_out")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_composites(
                tif_paths=tifs,
                output_dir=out_dir,
                stats=("std",),
                render_png=False,
            )
        runtime_warns = [
            w for w in caught if issubclass(w.category, RuntimeWarning) and "divide" in str(w.message).lower()
        ]
        assert len(runtime_warns) == 0, f"Got {len(runtime_warns)} RuntimeWarning(s) about division"


# ---------------------------------------------------------------------------
# M1: CRS mismatch
# ---------------------------------------------------------------------------


class TestCRSMismatch:
    """M1 fix: composites should skip files with mismatched CRS."""

    def test_mismatched_crs_skipped(self, tmp_path: Path) -> None:
        """GeoTIFF with different CRS should be excluded from composite."""
        h, w = 5, 5
        transform = from_origin(-180, 78, 0.1, 0.1)
        profile_4326 = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": rasterio.float32,
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": np.nan,
        }
        profile_3857 = profile_4326.copy()
        profile_3857["crs"] = "EPSG:3857"

        # Two files with EPSG:4326, one with EPSG:3857
        tifs = []
        for i in range(2):
            data = np.full((h, w), 2.0, dtype=np.float32)
            path = str(tmp_path / f"crs_ok_{i}.tif")
            with rasterio.open(path, "w", **profile_4326) as dst:
                dst.write(data, 1)
            tifs.append(path)

        # Mismatched CRS file
        data_bad = np.full((h, w), 100.0, dtype=np.float32)
        bad_path = str(tmp_path / "crs_bad.tif")
        with rasterio.open(bad_path, "w", **profile_3857) as dst:
            dst.write(data_bad, 1)
        tifs.append(bad_path)

        out_dir = str(tmp_path / "crs_out")
        results = compute_composites(
            tif_paths=tifs,
            output_dir=out_dir,
            stats=("mean",),
            render_png=False,
        )
        with rasterio.open(results["mean"]) as src:
            data = src.read(1)
        # Mean should be 2.0 (bad file excluded), not (2+2+100)/3 = 34.67
        valid = data[~np.isnan(data)]
        assert np.all(valid < 5.0), f"CRS-mismatched file was included: mean={valid.mean():.1f}"


# ---------------------------------------------------------------------------
# M2: Narrowed except
# ---------------------------------------------------------------------------


class TestNarrowedExcept:
    """M2 fix: rasterize loop should only catch expected exceptions."""

    def test_rasterize_with_valid_data_succeeds(
        self,
        tmp_path: Path,
    ) -> None:
        """Normal rasterization should still work after narrowing except."""
        df = pd.DataFrame(
            {
                "timestamp.1": ["2015-07-01"] * 10,
                "latitude": np.linspace(0, 5, 10),
                "longitude": np.linspace(0, 5, 10),
                "sap_velocity_cnn_lstm": np.random.uniform(0, 5, 10),
            }
        )
        csv_path = str(tmp_path / "narrow_test.csv")
        df.to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "narrow_out")
        results = rasterize_all_timestamps(
            csv_path=csv_path,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=1.0,
        )
        assert len(results) == 1
