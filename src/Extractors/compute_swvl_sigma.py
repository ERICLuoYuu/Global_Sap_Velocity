"""Compute per-cell σ-map for ERA5-Land soil-moisture layers.

Streams stage2 prediction parquets through a Welford accumulator and writes
``swvl_sigma_v1.nc`` per the Section 5 contract of
``.claude/plan/rooted-ingestion-redesign.md``.

Design notes
------------
* **Auto-detect layers.** Scans the first parquet's schema for any column named
  ``volumetric_soil_water_layer_{1..4}`` and processes only those. Phase 0a sees
  swvl1 only (existing cache); after Phase 1 the same script processes all 4.
* **Welford streaming.** Maintains ``(n, mean, M2)`` per (lat, lon, layer) tuple.
  Numerically stable for σ ≪ |μ| (which is the regime for swvl in arid cells).
* **Per-cell key.** Coords are rounded to 1 decimal place before keying — parquets
  carry 0.1°-aligned coords but float drift across years can introduce ±1e-7
  jitter; rounding once keeps dictionary keys stable.
* **Cell gate.** ``min_count`` (default 60) drops cells with too-few obs to
  estimate σ reliably. ``eps_sigma`` (default 1e-10) clips σ from below so
  downstream division never divides by zero.

CLI
---
python -m src.Extractors.compute_swvl_sigma \\
    --input-glob "/scratch/tmp/yluo2/gsv/outputs/data_for_prediction/stage2_no_precip/*.parquet" \\
    --output /scratch/tmp/yluo2/gsv/outputs/derived/swvl_sigma_v1.nc \\
    --min-count 60 \\
    --version v1
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

LOGGER = logging.getLogger("compute_swvl_sigma")

LAYER_PATTERN = "volumetric_soil_water_layer_{layer}"
LAT_COL = "latitude"
LON_COL = "longitude"
COORD_DECIMALS = 1
DEFAULT_MIN_COUNT = 60
DEFAULT_EPS_SIGMA = 1e-10


@dataclass
class WelfordCell:
    """Streaming variance accumulator for one (lat, lon, layer) cell.

    Updates ``(n, mean, M2)`` with each observation. ``variance = M2 / (n - 1)``
    once finalized. See Knuth TAOCP vol. 2 § 4.2.2 / Welford 1962.
    """

    n: int = 0
    mean: float = 0.0
    M2: float = 0.0  # noqa: N815 — match Welford notation

    def update_block(self, values: np.ndarray) -> None:
        """Update from a block of finite-valued observations."""
        for x in values:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def variance(self) -> float:
        if self.n < 2:
            return float("nan")
        return self.M2 / (self.n - 1)

    def sigma(self) -> float:
        v = self.variance()
        return float("nan") if not np.isfinite(v) or v < 0 else float(np.sqrt(v))


@dataclass
class SigmaAccumulator:
    """Per-layer dict of WelfordCell, keyed by (lat_rounded, lon_rounded)."""

    layers: tuple[int, ...]
    cells: dict[int, dict[tuple[float, float], WelfordCell]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for layer in self.layers:
            self.cells.setdefault(layer, {})

    def ingest_frame(self, df: pd.DataFrame) -> None:
        """Update accumulators from one parquet frame.

        Frame must carry ``latitude``, ``longitude`` and at least one of the
        target layer columns. Rows with non-finite layer values are skipped per
        layer; rows with non-finite coords are dropped wholesale.
        """
        coord_mask = np.isfinite(df[LAT_COL].to_numpy()) & np.isfinite(df[LON_COL].to_numpy())
        if not coord_mask.all():
            df = df.loc[coord_mask].reset_index(drop=True)

        lat_keys = np.round(df[LAT_COL].to_numpy(dtype=np.float64), COORD_DECIMALS)
        lon_keys = np.round(df[LON_COL].to_numpy(dtype=np.float64), COORD_DECIMALS)

        for layer in self.layers:
            col = LAYER_PATTERN.format(layer=layer)
            if col not in df.columns:
                continue
            values = df[col].to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            if not finite.any():
                continue

            sub_lat = lat_keys[finite]
            sub_lon = lon_keys[finite]
            sub_val = values[finite]

            store = self.cells[layer]
            grouper = pd.DataFrame({"lat": sub_lat, "lon": sub_lon, "v": sub_val})
            for (lat_k, lon_k), block in grouper.groupby(["lat", "lon"], sort=False):
                cell = store.get((lat_k, lon_k))
                if cell is None:
                    cell = WelfordCell()
                    store[(lat_k, lon_k)] = cell
                cell.update_block(block["v"].to_numpy(dtype=np.float64))


def detect_layers(parquet_path: Path) -> tuple[int, ...]:
    """Inspect the first parquet's schema and return the swvl layers present."""
    schema = pq.read_schema(parquet_path)
    available = set(schema.names)
    present = tuple(layer for layer in (1, 2, 3, 4) if LAYER_PATTERN.format(layer=layer) in available)
    if not present:
        raise RuntimeError(
            f"No volumetric_soil_water_layer_{{1..4}} columns in {parquet_path}; is this the wrong cache?"
        )
    return present


def iter_parquet_frames(paths: Sequence[Path], columns: Sequence[str]) -> Iterable[pd.DataFrame]:
    """Yield parquet frames one file at a time with only the columns we need."""
    for p in paths:
        try:
            yield pd.read_parquet(p, columns=list(columns))
        except Exception as exc:  # noqa: BLE001 — parquet failures vary; log and continue
            LOGGER.warning("Skipping %s: %s", p, exc)


def build_grid(cells_by_layer: dict[int, dict[tuple[float, float], WelfordCell]]) -> tuple[np.ndarray, np.ndarray]:
    """Construct regular 0.1° lat/lon axes spanning all observed cells."""
    all_lat = set()
    all_lon = set()
    for store in cells_by_layer.values():
        for lat_k, lon_k in store:
            all_lat.add(lat_k)
            all_lon.add(lon_k)
    if not all_lat or not all_lon:
        raise RuntimeError("No cells accumulated — check input parquets and layer detection.")

    lat_min, lat_max = min(all_lat), max(all_lat)
    lon_min, lon_max = min(all_lon), max(all_lon)
    n_lat = int(round((lat_max - lat_min) / 0.1)) + 1
    n_lon = int(round((lon_max - lon_min) / 0.1)) + 1
    lats = np.round(np.linspace(lat_min, lat_max, n_lat), COORD_DECIMALS).astype(np.float32)
    lons = np.round(np.linspace(lon_min, lon_max, n_lon), COORD_DECIMALS).astype(np.float32)
    return lats, lons


def materialize_layer(
    store: dict[tuple[float, float], WelfordCell],
    lats: np.ndarray,
    lons: np.ndarray,
    min_count: int,
    eps_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project the cell dict onto the (lat, lon) grid → sigma, mean, n arrays."""
    sigma = np.full((lats.size, lons.size), np.nan, dtype=np.float32)
    mean = np.full((lats.size, lons.size), np.nan, dtype=np.float32)
    n = np.zeros((lats.size, lons.size), dtype=np.int32)

    lat_index = {round(float(v), COORD_DECIMALS): i for i, v in enumerate(lats)}
    lon_index = {round(float(v), COORD_DECIMALS): i for i, v in enumerate(lons)}

    for (lat_k, lon_k), cell in store.items():
        i = lat_index.get(lat_k)
        j = lon_index.get(lon_k)
        if i is None or j is None:
            continue
        n[i, j] = cell.n
        if cell.n >= min_count:
            mean[i, j] = cell.mean
            s = cell.sigma()
            if np.isfinite(s):
                sigma[i, j] = max(s, eps_sigma)
    return sigma, mean, n


def write_netcdf(
    out_path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    arrays_by_layer: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    source: str,
    min_count: int,
    eps_sigma: float,
    version: str,
) -> None:
    """Write the σ-map to NetCDF per Section 5 contract."""
    data_vars: dict[str, xr.DataArray] = {}
    for layer, (sigma, mean, n) in arrays_by_layer.items():
        data_vars[f"swvl{layer}_sigma"] = xr.DataArray(sigma, dims=("lat", "lon"), attrs={"units": "m3 m-3"})
        data_vars[f"swvl{layer}_mean"] = xr.DataArray(mean, dims=("lat", "lon"), attrs={"units": "m3 m-3"})
        data_vars[f"n_swvl{layer}"] = xr.DataArray(n, dims=("lat", "lon"), attrs={"long_name": "valid daily obs"})

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "lat": xr.DataArray(lats, dims="lat", attrs={"units": "degrees_north"}),
            "lon": xr.DataArray(lons, dims="lon", attrs={"units": "degrees_east"}),
        },
        attrs={
            "source": source,
            "method": "streaming Welford accumulator",
            "min_count": int(min_count),
            "eps_sigma": float(eps_sigma),
            "version": version,
        },
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in list(data_vars) + ["lat", "lon"]}
    ds.to_netcdf(out_path, encoding=encoding, engine="netcdf4")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-glob", required=True, help="Glob for source parquets")
    parser.add_argument("--output", required=True, type=Path, help="NetCDF output path")
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    parser.add_argument("--eps-sigma", type=float, default=DEFAULT_EPS_SIGMA)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--source-label", default="ERA5-Land 2016-2018 daily, raw GEE pull pass")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N parquets (0 = all)")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Path.glob() in Python 3.9 rejects absolute patterns ("Non-relative
    # patterns are unsupported"), so split absolute globs into a parent dir
    # and a relative pattern. Use the system glob module for safety.
    import glob as _glob

    if "*" in args.input_glob or "?" in args.input_glob or "[" in args.input_glob:
        paths = sorted(Path(p) for p in _glob.glob(args.input_glob))
    else:
        paths = [Path(args.input_glob)]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        LOGGER.error("No parquets matched %s", args.input_glob)
        return 2
    LOGGER.info("Matched %d parquets", len(paths))

    layers = detect_layers(paths[0])
    LOGGER.info("Detected swvl layers: %s", layers)

    columns = [LAT_COL, LON_COL] + [LAYER_PATTERN.format(layer=layer) for layer in layers]
    accumulator = SigmaAccumulator(layers=layers)

    for i, frame in enumerate(iter_parquet_frames(paths, columns), 1):
        accumulator.ingest_frame(frame)
        if i % 25 == 0 or i == len(paths):
            sample_layer = layers[0]
            LOGGER.info(
                "Processed %d/%d parquets (cells in layer %d: %d)",
                i,
                len(paths),
                sample_layer,
                len(accumulator.cells[sample_layer]),
            )

    lats, lons = build_grid(accumulator.cells)
    LOGGER.info("Grid: %d lat × %d lon", lats.size, lons.size)

    arrays_by_layer: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for layer in layers:
        sigma, mean, n = materialize_layer(
            accumulator.cells[layer],
            lats,
            lons,
            min_count=args.min_count,
            eps_sigma=args.eps_sigma,
        )
        valid = int(np.isfinite(sigma).sum())
        LOGGER.info("Layer %d: %d/%d cells passed min_count=%d gate", layer, valid, sigma.size, args.min_count)
        arrays_by_layer[layer] = (sigma, mean, n)

    write_netcdf(
        args.output,
        lats,
        lons,
        arrays_by_layer,
        source=args.source_label,
        min_count=args.min_count,
        eps_sigma=args.eps_sigma,
        version=args.version,
    )
    LOGGER.info("Wrote %s", args.output)
    LOGGER.info("SHA-256: %s", sha256_of(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
