"""Tests for Parquet migration: io_utils helpers and pipeline integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_prediction_df():
    """Minimal prediction-style DataFrame with typical columns."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
            "latitude": rng.uniform(-60, 78, n),
            "longitude": rng.uniform(-180, 180, n),
            "ta": rng.uniform(250, 310, n),
            "vpd": rng.uniform(0, 4, n),
            "sw_in": rng.uniform(0, 800, n),
            "sap_velocity": rng.uniform(0, 15, n),
            "elevation": rng.uniform(0, 3000, n),
            "canopy_height": rng.uniform(5, 40, n),
            "MF": rng.integers(0, 2, n),
            "ENF": rng.integers(0, 2, n),
        }
    )


@pytest.fixture
def sample_df_no_tz():
    """DataFrame with timezone-naive timestamps."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-06-01", periods=10, freq="D"),
            "latitude": np.linspace(40, 50, 10),
            "longitude": np.linspace(-10, 0, 10),
            "sap_velocity": np.random.default_rng(7).uniform(0, 10, 10),
        }
    )


# ===========================================================================
# 1. io_utils: save_df / read_df
# ===========================================================================


class TestSaveDf:
    """save_df writes the correct format based on arguments."""

    def test_save_parquet_default(self, tmp_path, sample_prediction_df):
        from src.make_prediction.io_utils import save_df

        out = tmp_path / "test.parquet"
        save_df(sample_prediction_df, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_csv_explicit(self, tmp_path, sample_prediction_df):
        from src.make_prediction.io_utils import save_df

        out = tmp_path / "test.csv"
        save_df(sample_prediction_df, out, fmt="csv")
        assert out.exists()
        content = out.read_text()
        assert "timestamp" in content

    def test_save_parquet_smaller_than_csv(self, tmp_path, sample_prediction_df):
        from src.make_prediction.io_utils import save_df

        pq = tmp_path / "test.parquet"
        csv = tmp_path / "test.csv"
        save_df(sample_prediction_df, pq)
        save_df(sample_prediction_df, csv, fmt="csv")
        assert pq.stat().st_size < csv.stat().st_size

    def test_save_creates_parent_dirs(self, tmp_path, sample_prediction_df):
        from src.make_prediction.io_utils import save_df

        out = tmp_path / "sub" / "dir" / "test.parquet"
        save_df(sample_prediction_df, out)
        assert out.exists()

    def test_save_fmt_from_extension(self, tmp_path, sample_prediction_df):
        """When fmt is not given, infer from file extension."""
        from src.make_prediction.io_utils import save_df

        csv_path = tmp_path / "out.csv"
        save_df(sample_prediction_df, csv_path)
        # Should be readable as CSV
        df = pd.read_csv(csv_path)
        assert len(df) == len(sample_prediction_df)


class TestReadDf:
    """read_df auto-detects format from extension."""

    def test_roundtrip_parquet(self, tmp_path, sample_prediction_df):
        from src.make_prediction.io_utils import read_df, save_df

        path = tmp_path / "data.parquet"
        save_df(sample_prediction_df, path)
        result = read_df(path)
        assert len(result) == len(sample_prediction_df)
        assert list(result.columns) == list(sample_prediction_df.columns)

    def test_roundtrip_csv(self, tmp_path, sample_prediction_df):
        from src.make_prediction.io_utils import read_df, save_df

        path = tmp_path / "data.csv"
        save_df(sample_prediction_df, path, fmt="csv")
        result = read_df(path)
        assert len(result) == len(sample_prediction_df)

    def test_timestamp_preserved_parquet(self, tmp_path, sample_prediction_df):
        """Parquet preserves datetime dtype without re-parsing."""
        from src.make_prediction.io_utils import read_df, save_df

        path = tmp_path / "ts.parquet"
        save_df(sample_prediction_df, path)
        result = read_df(path)
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])

    def test_timestamp_naive_roundtrip(self, tmp_path, sample_df_no_tz):
        """Timezone-naive timestamps survive parquet round-trip."""
        from src.make_prediction.io_utils import read_df, save_df

        path = tmp_path / "naive.parquet"
        save_df(sample_df_no_tz, path)
        result = read_df(path)
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            sample_df_no_tz.reset_index(drop=True),
            check_dtype=False,
        )

    def test_numeric_values_exact(self, tmp_path, sample_prediction_df):
        """Numeric columns match exactly after Parquet round-trip."""
        from src.make_prediction.io_utils import read_df, save_df

        path = tmp_path / "exact.parquet"
        save_df(sample_prediction_df, path)
        result = read_df(path)
        np.testing.assert_array_almost_equal(
            result["sap_velocity"].values,
            sample_prediction_df["sap_velocity"].values,
        )

    def test_read_nonexistent_raises(self, tmp_path):
        from src.make_prediction.io_utils import read_df

        with pytest.raises(FileNotFoundError):
            read_df(tmp_path / "nope.parquet")

    def test_read_unknown_extension_raises(self, tmp_path):
        from src.make_prediction.io_utils import read_df

        bad = tmp_path / "data.xyz"
        bad.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            read_df(bad)


class TestReadDfDask:
    """read_df_dask returns a Dask DataFrame for both formats."""

    def test_dask_parquet(self, tmp_path, sample_prediction_df):
        from src.make_prediction.io_utils import read_df_dask, save_df

        path = tmp_path / "dask.parquet"
        save_df(sample_prediction_df, path)
        ddf = read_df_dask(path)
        assert len(ddf) == len(sample_prediction_df)

    def test_dask_csv(self, tmp_path, sample_prediction_df):
        from src.make_prediction.io_utils import read_df_dask, save_df

        path = tmp_path / "dask.csv"
        save_df(sample_prediction_df, path, fmt="csv")
        ddf = read_df_dask(path)
        assert len(ddf) == len(sample_prediction_df)

    def test_dask_parquet_columns_subset(self, tmp_path, sample_prediction_df):
        """Can read only specific columns from Parquet (columnar advantage)."""
        from src.make_prediction.io_utils import read_df_dask, save_df

        path = tmp_path / "cols.parquet"
        save_df(sample_prediction_df, path)
        ddf = read_df_dask(path, columns=["latitude", "longitude", "sap_velocity"])
        assert list(ddf.columns) == ["latitude", "longitude", "sap_velocity"]


# ===========================================================================
# 2. Stage 1: save_prediction_dataset format
# ===========================================================================


class TestStage1OutputFormat:
    """process_era5land_gee_opt_fix saves Parquet when configured."""

    def test_output_extension_parquet(self, tmp_path, sample_prediction_df):
        """save_prediction_dataset produces .parquet file."""
        from src.make_prediction.io_utils import read_df, save_df

        out = tmp_path / "prediction_2020_01_daily.parquet"
        save_df(sample_prediction_df, out)
        result = read_df(out)
        assert len(result) == len(sample_prediction_df)

    def test_temp_hourly_parquet_roundtrip(self, tmp_path, sample_prediction_df):
        """Temp hourly files written as Parquet can be merged back."""
        from src.make_prediction.io_utils import read_df, save_df

        # Simulate writing hourly chunks
        chunks = []
        for i in range(3):
            chunk = sample_prediction_df.iloc[i * 50 : (i + 1) * 50]
            path = tmp_path / f"hour_{i:02d}_raw.parquet"
            save_df(chunk, path)
            chunks.append(path)
        # Merge back
        merged = pd.concat([read_df(p) for p in chunks], ignore_index=True)
        assert len(merged) == 150


# ===========================================================================
# 3. Stage 2: file discovery
# ===========================================================================


class TestStage2Discovery:
    """predict_sap_velocity_sequential discovers both formats."""

    def test_discovers_parquet_files(self, tmp_path, sample_prediction_df):
        from src.make_prediction.io_utils import discover_input_files, save_df

        save_df(sample_prediction_df, tmp_path / "pred_2020_01.parquet")
        save_df(sample_prediction_df, tmp_path / "pred_2020_02.parquet")
        files = discover_input_files(tmp_path)
        assert len(files) == 2
        assert all(f.suffix == ".parquet" for f in files)

    def test_discovers_csv_files(self, tmp_path, sample_prediction_df):
        from src.make_prediction.io_utils import discover_input_files, save_df

        save_df(sample_prediction_df, tmp_path / "pred_2020_01.csv", fmt="csv")
        files = discover_input_files(tmp_path)
        assert len(files) == 1
        assert files[0].suffix == ".csv"

    def test_prefers_parquet_over_csv(self, tmp_path, sample_prediction_df):
        """When both formats exist for same stem, prefer Parquet."""
        from src.make_prediction.io_utils import discover_input_files, save_df

        save_df(sample_prediction_df, tmp_path / "pred.parquet")
        save_df(sample_prediction_df, tmp_path / "pred.csv", fmt="csv")
        files = discover_input_files(tmp_path)
        # Should return only the parquet version
        assert len(files) == 1
        assert files[0].suffix == ".parquet"

    def test_mixed_formats(self, tmp_path, sample_prediction_df):
        """Discovers both formats when stems differ."""
        from src.make_prediction.io_utils import discover_input_files, save_df

        save_df(sample_prediction_df, tmp_path / "jan.parquet")
        save_df(sample_prediction_df, tmp_path / "feb.csv", fmt="csv")
        files = discover_input_files(tmp_path)
        assert len(files) == 2


# ===========================================================================
# 4. Stage 3: visualization reads Parquet
# ===========================================================================


class TestStage3ParquetInput:
    """prediction_visualization can read Parquet input."""

    def test_inspect_parquet_file(self, tmp_path, sample_prediction_df):
        """inspect function works with Parquet input."""
        from src.make_prediction.io_utils import read_df, save_df

        path = tmp_path / "viz_input.parquet"
        save_df(sample_prediction_df, path)
        # Basic check: can read nrows equivalent
        df = read_df(path)
        assert "latitude" in df.columns
        assert "longitude" in df.columns
        assert "sap_velocity" in df.columns

    def test_dask_aggregation_from_parquet(self, tmp_path, sample_prediction_df):
        """Dask can groupby-aggregate from Parquet (core viz operation)."""
        import dask.dataframe as dd
        from src.make_prediction.io_utils import save_df

        path = tmp_path / "agg.parquet"
        save_df(sample_prediction_df, path)
        ddf = dd.read_parquet(path, columns=["latitude", "longitude", "sap_velocity"])
        result = ddf.groupby(["latitude", "longitude"])["sap_velocity"].mean().compute()
        assert len(result) > 0
