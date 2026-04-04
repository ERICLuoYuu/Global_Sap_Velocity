"""Round 3 tests: Integration & regression for all review fixes.

End-to-end tests that exercise multiple fixes simultaneously,
plus regression tests ensuring fixes didn't break existing behavior.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from src.make_prediction.prediction_visualization_hpc import (
    _write_geotiff,
    compute_composites,
    rasterize_all_timestamps,
    run_batch,
)

# ---------------------------------------------------------------------------
# End-to-end: full pipeline with hourly data + composites
# ---------------------------------------------------------------------------


class TestEndToEndHourlyPipeline:
    """Full pipeline: hourly CSV -> daily GeoTIFFs -> composites."""

    def test_hourly_run_batch_mean(self, tmp_path: Path) -> None:
        """run_batch with hourly data, daily aggregation, then composites."""
        np.random.seed(42)
        rows = []
        for day in range(1, 4):
            for hour in range(0, 24, 6):
                for _ in range(10):
                    rows.append(
                        {
                            "timestamp.1": f"2015-07-{day:02d} {hour:02d}:00:00",
                            "latitude": round(np.random.uniform(-5, 5), 1),
                            "longitude": round(np.random.uniform(-5, 5), 1),
                            "sw_in": 200.0,
                            "sap_velocity_cnn_lstm": np.random.uniform(0, 5),
                        }
                    )
        df = pd.DataFrame(rows)
        in_dir = str(tmp_path / "in")
        os.makedirs(in_dir)
        df.to_csv(os.path.join(in_dir, "hourly_predictions.csv"), index=False)

        run_batch(
            input_dir=in_dir,
            output_dir=str(tmp_path / "out"),
            run_id="hourly_test",
            file_glob="*predictions*.csv",
            time_scale="hourly",
            hourly_agg="mean",
            sw_in_threshold=0,
            resolution=1.0,
            stats=("mean", "count"),
            render_png=False,
        )

        run_dir = tmp_path / "out" / "hourly_test"
        assert run_dir.exists()
        tifs = list(run_dir.glob("sap_velocity_*.tif"))
        # 3 days of hourly data -> 3 daily TIFs
        assert len(tifs) == 3, f"Expected 3 daily TIFs but got {len(tifs)}"

        comp_dir = run_dir / "composites"
        assert comp_dir.exists()
        assert (comp_dir / "sap_velocity_composite_mean.tif").exists()
        assert (comp_dir / "sap_velocity_composite_count.tif").exists()


# ---------------------------------------------------------------------------
# End-to-end: parquet through rasterize_all_timestamps + composites
# ---------------------------------------------------------------------------


class TestEndToEndParquet:
    """Full pipeline with parquet input format."""

    def test_parquet_through_rasterize_all(self, tmp_path: Path) -> None:
        """Parquet input through rasterize_all_timestamps -> composites."""
        np.random.seed(99)
        rows = []
        for ts in ["2015-07-01", "2015-07-02"]:
            for _ in range(20):
                rows.append(
                    {
                        "timestamp.1": ts,
                        "latitude": round(np.random.uniform(-5, 5), 1),
                        "longitude": round(np.random.uniform(-5, 5), 1),
                        "sap_velocity_cnn_lstm": np.random.uniform(0, 5),
                    }
                )
        df = pd.DataFrame(rows)
        pq_path = str(tmp_path / "predictions.parquet")
        df.to_parquet(pq_path, engine="pyarrow")

        out_dir = str(tmp_path / "pq_out")
        results = rasterize_all_timestamps(
            csv_path=pq_path,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=1.0,
        )
        assert len(results) == 2

        tif_paths = [p for _, p in results]
        comp_dir = str(tmp_path / "pq_composites")
        comp_results = compute_composites(
            tif_paths=tif_paths,
            output_dir=comp_dir,
            stats=("mean", "max"),
            render_png=False,
        )
        assert len(comp_results) == 2
        for stat, path in comp_results.items():
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Regression: C1 fix didn't break daily (non-hourly) mode
# ---------------------------------------------------------------------------


class TestC1RegressionDailyMode:
    """Ensure C1 fix (chunk agg) didn't break daily (non-hourly) operation."""

    def test_daily_mode_unaffected(self, tmp_path: Path) -> None:
        """Daily data without hourly_mode should use mean agg as before."""
        np.random.seed(77)
        rows = []
        for ts in ["2015-07-01", "2015-07-02"]:
            for _ in range(30):
                rows.append(
                    {
                        "timestamp.1": ts,
                        "latitude": round(np.random.uniform(-10, 10), 1),
                        "longitude": round(np.random.uniform(-10, 10), 1),
                        "sap_velocity_cnn_lstm": np.random.uniform(1, 4),
                    }
                )
        df = pd.DataFrame(rows)
        csv_path = str(tmp_path / "daily.csv")
        df.to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "daily_out")
        results = rasterize_all_timestamps(
            csv_path=csv_path,
            output_dir=out_dir,
            sw_in_threshold=0,
            resolution=1.0,
            # Default: hourly_mode=False, hourly_agg="mean"
        )
        assert len(results) == 2
        for _, tif_path in results:
            with rasterio.open(tif_path) as src:
                data = src.read(1)
                valid = data[~np.isnan(data)]
                assert len(valid) > 0
                assert np.all(valid >= 0.5) and np.all(valid <= 4.5)


# ---------------------------------------------------------------------------
# Regression: H6 didn't break normal resolution values
# ---------------------------------------------------------------------------


class TestH6RegressionNormalResolution:
    """Ensure H6 bounds check doesn't reject valid scientific resolutions."""

    def test_001_degree_resolution(self, tmp_path: Path) -> None:
        """0.01 degree (~1km) resolution should be accepted."""
        df = pd.DataFrame(
            {
                "latitude": [0.0, 0.5],
                "longitude": [0.0, 0.5],
                "value": [1.0, 2.0],
            }
        )
        tif_path = str(tmp_path / "fine.tif")
        _write_geotiff(df, tif_path, "value", resolution=0.01)
        assert os.path.exists(tif_path)

    def test_005_degree_resolution(self, tmp_path: Path) -> None:
        """0.05 degree (~5km) resolution should be accepted."""
        df = pd.DataFrame(
            {
                "latitude": [0.0, 1.0],
                "longitude": [0.0, 1.0],
                "value": [1.0, 2.0],
            }
        )
        tif_path = str(tmp_path / "medium.tif")
        _write_geotiff(df, tif_path, "value", resolution=0.05)
        assert os.path.exists(tif_path)


# ---------------------------------------------------------------------------
# Regression: Multi-model pipeline still works after all fixes
# ---------------------------------------------------------------------------


class TestMultiModelRegression:
    """Ensure multi-model pipeline works with all review fixes applied."""

    def test_two_models_with_composites(self, tmp_path: Path) -> None:
        """Two value columns -> separate subdirs with composites."""
        np.random.seed(123)
        rows = []
        for ts in ["2015-07-01", "2015-07-02"]:
            for _ in range(20):
                rows.append(
                    {
                        "timestamp.1": ts,
                        "latitude": round(np.random.uniform(-5, 5), 1),
                        "longitude": round(np.random.uniform(-5, 5), 1),
                        "sw_in": 200.0,
                        "sap_velocity_xgb": np.random.uniform(0, 5),
                        "sap_velocity_rf": np.random.uniform(0, 5),
                    }
                )
        df = pd.DataFrame(rows)
        in_dir = str(tmp_path / "in")
        os.makedirs(in_dir)
        df.to_csv(os.path.join(in_dir, "multi_predictions.csv"), index=False)

        run_batch(
            input_dir=in_dir,
            output_dir=str(tmp_path / "out"),
            run_id="multi",
            value_columns=["sap_velocity_xgb", "sap_velocity_rf"],
            file_glob="*predictions*.csv",
            sw_in_threshold=0,
            resolution=1.0,
            stats=("mean",),
            render_png=False,
        )

        run_dir = tmp_path / "out" / "multi"
        assert (run_dir / "xgb").exists()
        assert (run_dir / "rf").exists()
        assert len(list((run_dir / "xgb").glob("sap_velocity_*.tif"))) == 2
        assert len(list((run_dir / "rf").glob("sap_velocity_*.tif"))) == 2
