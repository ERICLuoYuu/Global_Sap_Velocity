"""Unit tests for ``src.Extractors.compute_swvl_sigma``.

Hermetic — no parquet files, no network. Synthetic frames cover:
  * Welford accumulator vs. ``np.std(ddof=1)`` on a known sample.
  * Layer auto-detection from a parquet schema.
  * Per-cell σ aggregation across multiple frames.
  * ``min_count`` gate masks under-sampled cells.
  * NetCDF round-trip preserves attrs and layer count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xarray as xr

from src.Extractors import compute_swvl_sigma as mod

pytestmark = pytest.mark.unit


def _make_frame(coords, values_by_layer):
    """Build a DataFrame with lat, lon, and one column per layer in values_by_layer."""
    lats, lons = zip(*coords)  # noqa: B905 — Python 3.9 compat (no strict=)
    df = pd.DataFrame({"latitude": lats, "longitude": lons})
    for layer, values in values_by_layer.items():
        df[mod.LAYER_PATTERN.format(layer=layer)] = values
    return df


# -- Welford accumulator -----------------------------------------------------


def test_welford_matches_numpy_std_on_known_sample():
    rng = np.random.default_rng(seed=42)
    values = rng.normal(loc=0.25, scale=0.05, size=500).astype(np.float64)

    cell = mod.WelfordCell()
    cell.update_block(values)

    assert cell.n == 500
    assert cell.mean == pytest.approx(values.mean(), rel=1e-12, abs=1e-12)
    # Welford gives sample variance (ddof=1).
    assert cell.variance() == pytest.approx(values.var(ddof=1), rel=1e-10)
    assert cell.sigma() == pytest.approx(values.std(ddof=1), rel=1e-10)


def test_welford_returns_nan_for_lt_two_samples():
    cell = mod.WelfordCell()
    assert np.isnan(cell.variance())
    cell.update_block(np.array([0.5]))
    assert np.isnan(cell.variance())


def test_welford_block_update_equivalent_to_one_at_a_time():
    rng = np.random.default_rng(seed=7)
    a = rng.normal(size=200)
    b = rng.normal(size=300)

    block = mod.WelfordCell()
    block.update_block(a)
    block.update_block(b)

    one_by_one = mod.WelfordCell()
    for x in np.concatenate([a, b]):
        one_by_one.update_block(np.array([x]))

    assert block.n == one_by_one.n
    assert block.mean == pytest.approx(one_by_one.mean, rel=1e-12)
    assert block.variance() == pytest.approx(one_by_one.variance(), rel=1e-10)


# -- Per-cell aggregation ----------------------------------------------------


def test_accumulator_aggregates_same_cell_across_frames():
    acc = mod.SigmaAccumulator(layers=(1,))
    coords = [(10.1, 20.2)] * 10
    frame_a = _make_frame(coords, {1: np.linspace(0.10, 0.20, 10)})
    frame_b = _make_frame(coords, {1: np.linspace(0.30, 0.40, 10)})

    acc.ingest_frame(frame_a)
    acc.ingest_frame(frame_b)

    cell = acc.cells[1][(10.1, 20.2)]
    assert cell.n == 20
    expected = np.concatenate([np.linspace(0.10, 0.20, 10), np.linspace(0.30, 0.40, 10)])
    assert cell.mean == pytest.approx(expected.mean(), rel=1e-10)
    assert cell.sigma() == pytest.approx(expected.std(ddof=1), rel=1e-10)


def test_accumulator_rounds_coords_to_one_decimal():
    """Float drift across years must not split one cell into two."""
    acc = mod.SigmaAccumulator(layers=(1,))
    coords = [(10.1 + 1e-7, 20.2 - 1e-7), (10.1 - 1e-8, 20.2 + 1e-8)]
    frame = _make_frame(coords, {1: [0.15, 0.25]})
    acc.ingest_frame(frame)

    # Both jittered points round to the same key → exactly one cell.
    assert len(acc.cells[1]) == 1


def test_accumulator_ignores_nonfinite_values():
    acc = mod.SigmaAccumulator(layers=(1,))
    coords = [(0.0, 0.0)] * 4
    frame = _make_frame(coords, {1: [0.1, np.nan, 0.3, np.inf]})
    acc.ingest_frame(frame)

    cell = acc.cells[1][(0.0, 0.0)]
    assert cell.n == 2  # only 0.1 and 0.3 are finite
    assert cell.mean == pytest.approx(0.2)


# -- Materialization + min_count gate ----------------------------------------


def test_materialize_layer_applies_min_count_gate():
    rng = np.random.default_rng(seed=1)
    acc = mod.SigmaAccumulator(layers=(1,))

    # Cell A: 100 obs (passes gate=60). Cell B: 30 obs (fails).
    coords_a = [(0.0, 0.0)] * 100
    coords_b = [(0.1, 0.1)] * 30
    acc.ingest_frame(_make_frame(coords_a, {1: rng.normal(0.25, 0.05, 100)}))
    acc.ingest_frame(_make_frame(coords_b, {1: rng.normal(0.25, 0.05, 30)}))

    lats = np.array([0.0, 0.1], dtype=np.float32)
    lons = np.array([0.0, 0.1], dtype=np.float32)
    sigma, mean, n = mod.materialize_layer(
        acc.cells[1],
        lats,
        lons,
        min_count=60,
        eps_sigma=1e-10,
    )

    # Cell A passes; cell B is masked.
    assert np.isfinite(sigma[0, 0])
    assert np.isnan(sigma[1, 1])
    assert n[0, 0] == 100
    assert n[1, 1] == 30


def test_materialize_layer_clips_sigma_at_eps():
    """Constant-valued cells should produce eps_sigma, not 0 or NaN."""
    acc = mod.SigmaAccumulator(layers=(1,))
    coords = [(5.0, 5.0)] * 200
    acc.ingest_frame(_make_frame(coords, {1: np.full(200, 0.30)}))

    lats = np.array([5.0], dtype=np.float32)
    lons = np.array([5.0], dtype=np.float32)
    sigma, _, _ = mod.materialize_layer(
        acc.cells[1],
        lats,
        lons,
        min_count=60,
        eps_sigma=1e-10,
    )
    # Sample variance of a constant series is 0; clipped up to eps.
    assert sigma[0, 0] == pytest.approx(1e-10, abs=1e-15)


# -- Layer auto-detection ----------------------------------------------------


def test_detect_layers_from_parquet_schema(tmp_path: Path):
    df = _make_frame(
        [(0.0, 0.0), (0.1, 0.1)],
        {1: [0.2, 0.3], 3: [0.25, 0.35]},
    )
    pq.write_table(pa.Table.from_pandas(df), tmp_path / "shard.parquet")
    layers = mod.detect_layers(tmp_path / "shard.parquet")
    assert layers == (1, 3)


def test_detect_layers_raises_when_no_swvl_columns(tmp_path: Path):
    df = pd.DataFrame({"latitude": [0.0], "longitude": [0.0], "ta": [15.0]})
    pq.write_table(pa.Table.from_pandas(df), tmp_path / "shard.parquet")
    with pytest.raises(RuntimeError, match="No volumetric_soil_water_layer"):
        mod.detect_layers(tmp_path / "shard.parquet")


# -- CLI absolute-glob handling (Python 3.9 regression) ----------------------


def test_main_resolves_absolute_glob_pattern(tmp_path: Path):
    """Python 3.9's Path.glob rejects absolute patterns; main() must use
    the stdlib `glob` module for absolute patterns. This test runs the full
    main() against a tiny absolute-path glob and asserts a NetCDF lands."""
    # Build two synthetic parquet shards with shared cells so σ is finite.
    rng = np.random.default_rng(seed=11)
    for fname in ["a.parquet", "b.parquet"]:
        df = _make_frame(
            [(0.0, 0.0)] * 50 + [(0.1, 0.1)] * 50,
            {1: rng.normal(0.25, 0.05, 100)},
        )
        pq.write_table(pa.Table.from_pandas(df), tmp_path / fname)

    out_path = tmp_path / "out" / "sigma.nc"
    abs_glob = str(tmp_path / "*.parquet")
    assert Path(abs_glob).is_absolute()  # confirms we're testing the right code path

    rc = mod.main(
        [
            "--input-glob",
            abs_glob,
            "--output",
            str(out_path),
            "--min-count",
            "60",
        ]
    )
    assert rc == 0, "main() should exit 0 on success"
    assert out_path.exists(), "expected NetCDF output file at the absolute path"

    ds = xr.open_dataset(out_path)
    try:
        assert "swvl1_sigma" in ds.data_vars
    finally:
        ds.close()


# -- NetCDF round-trip -------------------------------------------------------


def test_write_netcdf_round_trip_preserves_contract(tmp_path: Path):
    rng = np.random.default_rng(seed=99)
    lats = np.array([10.0, 10.1], dtype=np.float32)
    lons = np.array([20.0, 20.1], dtype=np.float32)

    sigma1 = rng.uniform(0.01, 0.05, size=(2, 2)).astype(np.float32)
    mean1 = rng.uniform(0.20, 0.30, size=(2, 2)).astype(np.float32)
    n1 = np.full((2, 2), 100, dtype=np.int32)

    out_path = tmp_path / "sigma.nc"
    mod.write_netcdf(
        out_path,
        lats,
        lons,
        arrays_by_layer={1: (sigma1, mean1, n1)},
        source="test fixture",
        min_count=60,
        eps_sigma=1e-10,
        version="v1",
    )

    ds = xr.open_dataset(out_path)
    try:
        assert "swvl1_sigma" in ds.data_vars
        assert "swvl1_mean" in ds.data_vars
        assert "n_swvl1" in ds.data_vars
        assert ds.attrs["source"] == "test fixture"
        assert ds.attrs["min_count"] == 60
        assert ds.attrs["version"] == "v1"
        np.testing.assert_array_equal(ds["lat"].values, lats)
        np.testing.assert_array_equal(ds["lon"].values, lons)
        np.testing.assert_array_almost_equal(ds["swvl1_sigma"].values, sigma1)
    finally:
        ds.close()


def test_write_netcdf_handles_multiple_layers(tmp_path: Path):
    lats = np.array([0.0], dtype=np.float32)
    lons = np.array([0.0], dtype=np.float32)
    arrays = {
        layer: (
            np.full((1, 1), 0.05 * layer, dtype=np.float32),
            np.full((1, 1), 0.25, dtype=np.float32),
            np.full((1, 1), 100, dtype=np.int32),
        )
        for layer in (1, 2, 3, 4)
    }
    out_path = tmp_path / "sigma.nc"
    mod.write_netcdf(
        out_path,
        lats,
        lons,
        arrays_by_layer=arrays,
        source="test",
        min_count=60,
        eps_sigma=1e-10,
        version="v1",
    )
    ds = xr.open_dataset(out_path)
    try:
        for layer in (1, 2, 3, 4):
            assert f"swvl{layer}_sigma" in ds.data_vars
            assert f"swvl{layer}_mean" in ds.data_vars
            assert f"n_swvl{layer}" in ds.data_vars
    finally:
        ds.close()
