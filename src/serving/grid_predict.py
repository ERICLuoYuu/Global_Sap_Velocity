"""Stage C: sap velocity for every served cell and day of a year.

    python -m src.serving.grid_predict plan     --store S --run_id R [--models_root models] [--model_type xgb]
    python -m src.serving.grid_predict run      --store S --run_id R --year Y --block K --n_blocks N [--processes P]
    python -m src.serving.grid_predict assemble --store S --run_id R --year Y --n_blocks N

``plan`` works out from the model alone what a run reads: the base columns its features depend on and the
days of history a cell must be engineered over before its first predicted day (the servability report's
``spin_up_days`` plus :data:`SPIN_UP_MARGIN_DAYS`). It refuses a model with a never-servable feature, a
static layer the store's cells lack, or a feature that comes out differently from a frame holding only the
columns it claims to need.

``run`` engineers each cell of a block over [1 January - history, 31 December] with the trainer's own
functions, keeps the year's rows, predicts and writes the block. History comes from the previous year's
store, so 1 January is predicted exactly as a mid-year day is; ``--cold_start`` instead starts on
1 January and blanks the days that lack history. A day of ERA5-Land that Stage A did not mark refuses the run.
A cell whose inputs cannot build a model feature (no soil within 1 km of it, say) is left unpredicted and
recorded in its block, as the trainer skips a site that lacks a feature. The run first repeats plan's check
that a served frame builds every feature, so such a gap is down to the cell's inputs, not to the code. A cell
carrying a forest type the model has no class for instead refuses the run: unlike a type the raster left
blank, it would one-hot to all zeros and serve as quietly as a real class.

``assemble`` writes a year's blocks into one NetCDF on the served grid, with the cells left unpredicted so marked.

``run --aoa_reference REF`` also stores the dissimilarity index (Meyer & Pebesma 2021) of every row it predicts,
computed on the same model matrix, so DI and prediction exist for exactly the same cell-days. The reference must
name the model's features in the model's order. ``assemble`` then writes ``DI`` and an ``aoa`` mask beside
``sap_velocity``, and refuses blocks that mix runs with and without DI, or runs against different references.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.feature_contract.definitions import SERVABLE_CONTRACT, SWC_LAYERS, swc_anomaly_col, swc_level_col
from src.hyperparameter_optimization.shap_constants import PFT_COLUMNS
from src.serving.acquire_era5 import STORE_COLUMNS
from src.serving.base_columns import (
    CELL_COLUMNS,
    DAY_COLUMN,
    LAI_COLUMNS,
    PFT_COLUMN,
    SENSOR_SUBSTITUTED,
    SOLAR_COLUMNS,
    STATIC_COLUMNS,
    TIME_COLUMN,
)
from src.serving.store import Store, load_grid, read_year_column, write_json, year_days

logger = logging.getLogger("grid_predict")

SPIN_UP_MARGIN_DAYS = 7
CELL_CHUNK = 128
UNITS = "cm3 cm-2 h-1"
NETCDF_CHUNK = (460, 600)
AOA_FILL = 255
ANOMALY_TO_LEVEL = {swc_anomaly_col(layer): swc_level_col(layer) for layer in SWC_LAYERS}
ALWAYS_PRESENT = (TIME_COLUMN, DAY_COLUMN, *CELL_COLUMNS)

_STATE = {}


def series_columns(required: Sequence[str]) -> tuple:
    """The stored day-by-day series a frame needs for ``required`` (an anomaly is built from its level)."""
    names = set()
    for name in required:
        if name in ANOMALY_TO_LEVEL:
            names.add(ANOMALY_TO_LEVEL[name])
        elif name in STORE_COLUMNS or name in SOLAR_COLUMNS or name in LAI_COLUMNS:
            names.add(name)
    return tuple(sorted(names))


def check_required_columns_suffice(feature_names: Sequence[str], required: Sequence[str]) -> None:
    """Refuse unless every feature is built, and identical, from a frame holding only what a served frame will hold.

    ``run_block`` repeats it: once it holds, a feature a served cell does not get is down to that cell's inputs.
    """
    from src.serving.dependencies import synthetic_cell  # noqa: PLC0415
    from src.serving.engineer import engineer_cell  # noqa: PLC0415

    kept = set(required) | set(series_columns(required)) | set(ALWAYS_PRESENT)
    full = synthetic_cell()
    whole = engineer_cell(full)
    reduced = engineer_cell(full[[name for name in full.columns if name in kept]])
    unbuilt = [name for name in feature_names if name not in PFT_COLUMNS and name not in whole.columns]
    if unbuilt:
        raise ValueError(f"feature(s) {unbuilt} are not built even from a cell holding every base column")
    differing = []
    for name in feature_names:
        if name in PFT_COLUMNS:
            continue
        if name not in reduced.columns or not np.array_equal(
            whole[name].to_numpy(dtype=float), reduced[name].to_numpy(dtype=float), equal_nan=True
        ):
            differing.append(name)
    if differing:
        raise ValueError(f"feature(s) {differing} change when the frame holds only their reported base columns")


def check_forest_types_are_model_classes(cells: pd.DataFrame, feature_names: Sequence[str]) -> None:
    """Refuse a cells table carrying a forest type the model has no class for.

    A cell the raster left blank is left unpredicted, as a cell with no soil is. A name outside
    :data:`PFT_COLUMNS` is a defect in the table: it one-hots to all zeros, which serves as quietly as a real
    class, so every block of the year refuses, not only the one holding the cell.
    """
    if not any(name in PFT_COLUMNS for name in feature_names):
        return
    unknown = sorted({str(value) for value in cells[PFT_COLUMN].dropna().unique() if str(value) not in PFT_COLUMNS})
    if unknown:
        raise ValueError(f"the cells carry forest type(s) {unknown}; the model's classes are {sorted(PFT_COLUMNS)}")


def plan_run(store: Store, run_id: str, models_root, model_type: str = "xgb") -> dict:
    import pyarrow.parquet as pq  # noqa: PLC0415

    from src.serving.dependencies import required_base_columns, servability_report, spin_up_days  # noqa: PLC0415
    from src.serving.model_bundle import load_bundle  # noqa: PLC0415

    bundle = load_bundle(models_root, run_id, model_type)
    names = list(bundle.feature_names)
    report = servability_report(names)
    spin_up = spin_up_days(report)
    required = required_base_columns(report)
    cell_columns = set(pq.read_schema(store.cells).names)
    absent = [name for name in required if name in STATIC_COLUMNS and name not in cell_columns]
    if absent:
        raise ValueError(f"{run_id} reads static layer(s) {absent}, which {store.cells} does not hold")
    check_required_columns_suffice(names, required)
    series = series_columns(required)
    plan = {
        "run_id": run_id,
        "model_type": model_type,
        "models_root": str(models_root),
        "feature_contract": SERVABLE_CONTRACT,
        "feature_names": names,
        "required_base_columns": list(required),
        "series_columns": list(series),
        "static_columns": [name for name in required if name in STATIC_COLUMNS],
        "spin_up_days": spin_up,
        "history_days": spin_up + SPIN_UP_MARGIN_DAYS,
        "event_dependent": report.loc[report["event_dependent"], "feature"].tolist(),
        "sensor_substituted_inputs": sorted(set(required) & SENSOR_SUBSTITUTED),
    }
    path = Path(store.plan(run_id))
    write_json(path, plan)
    report.to_csv(path.with_name("servability.csv"), index=False)
    return plan


def span_days(year: int, history_days: int) -> pd.DatetimeIndex:
    first = pd.Timestamp(f"{year}-01-01", tz="UTC") - pd.Timedelta(days=int(history_days))
    return pd.date_range(first, pd.Timestamp(f"{year}-12-31", tz="UTC"), freq="D")


def missing_era5_days(store: Store, days: pd.DatetimeIndex) -> list:
    return [day for day in days if not Path(store.era5_done(day.year, day)).exists()]


def _year_block(store: Store, cells: pd.DataFrame, year: int, column: str, start: int, stop: int, offsets):
    n_days, n_cells = len(year_days(year)), len(cells)
    if column in STORE_COLUMNS:
        return np.asarray(read_year_column(store.era5(year, column), n_days, n_cells)[offsets, start:stop], dtype=float)
    if column in LAI_COLUMNS:
        return np.asarray(read_year_column(store.lai(year), n_days, n_cells)[offsets, start:stop], dtype=float)
    if column in SOLAR_COLUMNS:
        return _solar_block(store, cells, year, column, start, stop)[offsets]
    raise KeyError(f"{column!r} is not a stored series")


def _solar_block(store: Store, cells: pd.DataFrame, year: int, column: str, start: int, stop: int) -> np.ndarray:
    from src.serving.solar_grid import cell_solar  # noqa: PLC0415

    key = ("solar", year, start, stop)
    if key not in _STATE:
        path = Path(store.solar(year))
        if not path.exists():
            raise FileNotFoundError(f"{path} has not been built")
        block = cells.iloc[start:stop]
        with np.load(path) as solar:
            arrays = {name: solar[name] for name in solar.files}
        index = np.searchsorted(arrays["rows"], block["row"].to_numpy())
        if np.any(index >= len(arrays["rows"])) or np.any(arrays["rows"][np.minimum(index, len(arrays["rows"]) - 1)] != block["row"].to_numpy()):
            raise ValueError(f"{path} lacks a grid row of this block")
        ext_rad, day_length = cell_solar(arrays, index, block["longitude"].to_numpy())
        _STATE[key] = {"ext_rad": ext_rad, "day_length": day_length}
    return _STATE[key][column]


def load_series(store: Store, cells: pd.DataFrame, start: int, stop: int, days: pd.DatetimeIndex, columns) -> dict:
    """Base column -> float64 (days, cells start:stop), stitched across calendar years."""
    out = {column: np.empty((len(days), stop - start)) for column in columns}
    for year in np.unique(days.year):
        chosen = np.asarray(days.year == year)
        offsets = np.asarray((days[chosen] - pd.Timestamp(f"{year}-01-01", tz="UTC")).days)
        for column in columns:
            out[column][chosen] = _year_block(store, cells, int(year), column, start, stop, offsets)
    return out


def predict_cells(
    days, series: dict, records: Sequence, feature_names, bundle, n_predicted: int, di_reference=None
) -> tuple:
    """float32 (n_predicted, cells): each cell engineered over ``days``, its last ``n_predicted`` rows predicted.

    Also the cells whose inputs cannot build a feature, left unpredicted: (position in ``records``, features absent);
    and, given a :class:`src.serving.aoa_di.DIReference`, the dissimilarity index of the predicted rows (else None).
    """
    from src.serving.engineer import AbsentFeatures, engineer_cell, model_matrix  # noqa: PLC0415
    from src.serving.frames import cell_frame  # noqa: PLC0415

    predicted = np.full((n_predicted, len(records)), np.nan, dtype=np.float32)
    di = None if di_reference is None else np.full_like(predicted, np.nan)
    built, matrices, unbuilt = [], [], []
    for j, record in enumerate(records):
        frame = cell_frame(days, {name: values[:, j] for name, values in series.items()}, record)
        if not frame[TIME_COLUMN].is_monotonic_increasing:
            raise ValueError("solar timestamps are not in day order")
        engineered = engineer_cell(frame)
        if len(engineered) != len(days):
            raise ValueError(f"feature engineering returned {len(engineered)} rows for {len(days)} days")
        cell_id = f"row{record['row']}_col{record['col']}"
        try:
            matrix = model_matrix(engineered, feature_names, cell_id)
        except AbsentFeatures as absent:
            unbuilt.append((j, absent.missing))
            continue
        built.append(j)
        matrices.append(matrix.iloc[len(days) - n_predicted :])
    if matrices:
        matrix = pd.concat(matrices, ignore_index=True)
        values = np.asarray(bundle.predict(matrix), dtype=np.float32)
        predicted[:, built] = values.reshape(len(built), n_predicted).T
        if di is not None:
            from src.serving.aoa_di import matrix_di  # noqa: PLC0415

            index = matrix_di(di_reference, matrix.to_numpy(dtype=np.float32))
            di[:, built] = index.reshape(len(built), n_predicted).T
    return predicted, unbuilt, di


def _init_worker(models_root, run_id, model_type, aoa_reference=None) -> None:
    from src.serving.model_bundle import load_bundle  # noqa: PLC0415

    _STATE["bundle"] = load_bundle(models_root, run_id, model_type)
    _STATE["di_reference"] = None
    if aoa_reference is not None:
        from src.serving.aoa_di import load_di_reference  # noqa: PLC0415

        _STATE["di_reference"] = load_di_reference(aoa_reference, _STATE["bundle"].feature_names)


def _predict_chunk(bounds) -> tuple:
    low, high = bounds
    series = {name: values[:, low:high] for name, values in _STATE["series"].items()}
    values, unbuilt, di = predict_cells(
        _STATE["days"],
        series,
        _STATE["records"][low:high],
        _STATE["feature_names"],
        _STATE["bundle"],
        _STATE["n_predicted"],
        _STATE.get("di_reference"),
    )
    return low, values, [(low + position, missing) for position, missing in unbuilt], di


def block_bounds(n_cells: int, block: int, n_blocks: int) -> tuple:
    if not 0 <= block < n_blocks:
        raise ValueError(f"block {block} outside 0..{n_blocks - 1}")
    edges = np.linspace(0, n_cells, n_blocks + 1).round().astype(int)
    return int(edges[block]), int(edges[block + 1])


def run_block(
    store: Store,
    run_id: str,
    year: int,
    block: int,
    n_blocks: int,
    processes: int = 1,
    cold_start: bool = False,
    aoa_reference=None,
) -> Path:
    plan = json.loads(Path(store.plan(run_id)).read_text())
    check_required_columns_suffice(plan["feature_names"], plan["required_base_columns"])
    di_reference = None
    if aoa_reference is not None:
        from src.serving.aoa_di import load_di_reference  # noqa: PLC0415

        aoa_reference = str(aoa_reference)
        di_reference = load_di_reference(aoa_reference, plan["feature_names"])
    cells = pd.read_parquet(store.cells)
    check_forest_types_are_model_classes(cells, plan["feature_names"])
    start, stop = block_bounds(len(cells), block, n_blocks)
    if stop <= start:
        raise ValueError(f"block {block} of {n_blocks} holds no cell")
    history = 0 if cold_start else int(plan["history_days"])
    days = span_days(year, history)
    columns = plan["series_columns"]
    if any(column in STORE_COLUMNS for column in columns):
        missing = missing_era5_days(store, days)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} day(s) of ERA5-Land not acquired, first {missing[0].date()}, last {missing[-1].date()}"
            )
    began = time.time()
    _STATE.clear()
    _STATE.update(
        series=load_series(store, cells, start, stop, days, columns),
        records=cells.iloc[start:stop].to_dict("records"),
        days=days,
        feature_names=plan["feature_names"],
        n_predicted=len(year_days(year)),
    )
    n_cells = stop - start
    logger.info("block %d/%d: cells %d..%d, %d days loaded in %.0f s", block, n_blocks, start, stop, len(days), time.time() - began)
    predictions = np.full((_STATE["n_predicted"], n_cells), np.nan, dtype=np.float32)
    di_all = None if di_reference is None else np.full_like(predictions, np.nan)
    unbuilt = []
    chunks = [(low, min(low + CELL_CHUNK, n_cells)) for low in range(0, n_cells, CELL_CHUNK)]
    bundle_args = (plan["models_root"], run_id, plan["model_type"], aoa_reference)
    done = 0

    def collect(low, values, chunk_unbuilt, di):
        predictions[:, low : low + values.shape[1]] = values
        if di_all is not None:
            di_all[:, low : low + values.shape[1]] = di
        unbuilt.extend(chunk_unbuilt)

    if processes > 1:
        with Pool(processes, initializer=_init_worker, initargs=bundle_args) as pool:
            for result in pool.imap_unordered(_predict_chunk, chunks):
                collect(*result)
                done += result[1].shape[1]
                logger.info("%d/%d cells (%.0f s)", done, n_cells, time.time() - began)
    else:
        _init_worker(*bundle_args)
        for bounds in chunks:
            collect(*_predict_chunk(bounds))
    if cold_start:
        predictions[: int(plan["history_days"])] = np.nan
        if di_all is not None:
            di_all[: int(plan["history_days"])] = np.nan
    unbuilt.sort()
    if unbuilt:
        by_features = Counter(",".join(missing) for _, missing in unbuilt)
        logger.info("%d cell(s) left unpredicted, their inputs cannot build: %s", len(unbuilt), dict(by_features))
    path = Path(store.block(run_id, year, block))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    arrays = {}
    if di_reference is not None:
        arrays = {
            "di": di_all,
            "aoa_reference_path": np.array(di_reference.path),
            "aoa_reference_sha256": np.array(di_reference.sha256),
            "aoa_threshold": np.float64(di_reference.threshold),
            "aoa_d_bar": np.float64(di_reference.d_bar),
        }
        finite = np.isfinite(di_all)
        if finite.any():
            logger.info("DI: %.1f%% of predicted cell-days inside the AOA", 100.0 * np.mean(di_all[finite] <= di_reference.threshold))
    with open(temporary, "wb") as handle:
        np.savez(
            handle,
            cells=np.arange(start, stop),
            year=year,
            predictions=predictions,
            cold_start=cold_start,
            unbuilt_cells=np.array([start + position for position, _ in unbuilt], dtype=np.int64),
            unbuilt_features=np.array([",".join(missing) for _, missing in unbuilt], dtype=str),
            **arrays,
        )
    os.replace(temporary, path)
    logger.info("wrote %s: %.1f%% of cell-days predicted", path, 100.0 * np.isfinite(predictions).mean())
    return path


def assemble(store: Store, run_id: str, year: int, n_blocks: int) -> Path:
    import netCDF4  # noqa: PLC0415

    plan = json.loads(Path(store.plan(run_id)).read_text())
    grid = load_grid(store)
    cells = pd.read_parquet(store.cells, columns=["row", "col"])
    days = year_days(year)
    predictions = np.full((len(days), len(cells)), np.nan, dtype=np.float32)
    covered = np.zeros(len(cells), dtype=bool)
    unbuilt = np.zeros(len(cells), dtype=bool)
    cold_start = False
    di, carries_di, references = None, set(), {}
    for block in range(n_blocks):
        with np.load(store.block(run_id, year, block)) as data:
            index, values = data["cells"], data["predictions"]
            if int(data["year"]) != year or values.shape != (len(days), len(index)):
                raise ValueError(f"block {block} does not hold {year} for its cells")
            if covered[index].any():
                raise ValueError(f"block {block} overlaps an earlier block")
            covered[index] = True
            predictions[:, index] = values
            unbuilt[data["unbuilt_cells"]] = True
            cold_start = cold_start or bool(data["cold_start"])
            carries_di.add("di" in data.files)
            if "di" in data.files:
                if di is None:
                    di = np.full_like(predictions, np.nan)
                di[:, index] = data["di"]
                references[str(data["aoa_reference_sha256"])] = (
                    str(data["aoa_reference_path"]),
                    float(data["aoa_threshold"]),
                    float(data["aoa_d_bar"]),
                )
    if len(carries_di) > 1:
        raise ValueError("AOA: some blocks carry DI and some do not; rerun them with the same --aoa_reference")
    if len(references) > 1:
        raise ValueError(f"blocks were run against {len(references)} different AOA references")
    if not covered.all():
        raise ValueError(f"{int((~covered).sum())} cell(s) are in no block; is --n_blocks the one the run used?")
    logger.info("%d served cell(s) left unpredicted: their inputs cannot build every feature", int(unbuilt.sum()))

    rows, cols = cells["row"].to_numpy(), cells["col"].to_numpy()
    path = Path(store.assembled(run_id, year))
    temporary = path.with_name(path.stem + ".tmp.nc")
    with netCDF4.Dataset(temporary, "w", format="NETCDF4") as nc:
        nc.createDimension("time", len(days))
        nc.createDimension("lat", grid.n_rows)
        nc.createDimension("lon", grid.n_cols)
        time_var = nc.createVariable("time", "i4", ("time",))
        time_var.units = f"days since {year}-01-01 00:00:00"
        time_var.calendar = "standard"
        time_var[:] = np.arange(len(days))
        for name, size, values, units in (
            ("lat", "lat", grid.latitudes(), "degrees_north"),
            ("lon", "lon", grid.longitudes(), "degrees_east"),
        ):
            variable = nc.createVariable(name, "f8", (size,))
            variable.units = units
            variable[:] = values
        chunks = (1, min(NETCDF_CHUNK[0], grid.n_rows), min(NETCDF_CHUNK[1], grid.n_cols))
        sap = nc.createVariable(
            "sap_velocity", "f4", ("time", "lat", "lon"), zlib=True, complevel=4, chunksizes=chunks, fill_value=np.float32(np.nan)
        )
        sap.units = UNITS
        sap.long_name = "daily mean sap velocity"
        for t in range(len(days)):
            plane = np.full((grid.n_rows, grid.n_cols), np.nan, dtype=np.float32)
            plane[rows, cols] = predictions[t]
            sap[t] = plane
        served = nc.createVariable("served_cell", "u1", ("lat", "lon"), zlib=True)
        mask = np.zeros((grid.n_rows, grid.n_cols), dtype=np.uint8)
        mask[rows, cols] = 1
        served[:] = mask
        marked = nc.createVariable("unbuilt_cell", "u1", ("lat", "lon"), zlib=True)
        marked.long_name = "served cell whose inputs cannot build every model feature, left unpredicted"
        unbuilt_mask = np.zeros((grid.n_rows, grid.n_cols), dtype=np.uint8)
        unbuilt_mask[rows[unbuilt], cols[unbuilt]] = 1
        marked[:] = unbuilt_mask
        if di is not None:
            sha256, (reference_path, threshold, d_bar) = next(iter(references.items()))
            di_var = nc.createVariable(
                "DI", "f4", ("time", "lat", "lon"), zlib=True, complevel=4, chunksizes=chunks, fill_value=np.float32(np.nan)
            )
            di_var.long_name = "dissimilarity index to the training data (Meyer & Pebesma 2021)"
            di_var.threshold = threshold
            aoa_var = nc.createVariable(
                "aoa", "u1", ("time", "lat", "lon"), zlib=True, complevel=4, chunksizes=chunks, fill_value=np.uint8(AOA_FILL)
            )
            aoa_var.long_name = "1 inside the area of applicability (DI <= threshold), 0 outside"
            for t in range(len(days)):
                plane = np.full((grid.n_rows, grid.n_cols), np.nan, dtype=np.float32)
                plane[rows, cols] = di[t]
                di_var[t] = plane
                inside = np.full((grid.n_rows, grid.n_cols), AOA_FILL, dtype=np.uint8)
                finite = np.isfinite(plane)
                inside[finite] = (plane[finite] <= threshold).astype(np.uint8)
                aoa_var[t] = inside
            nc.aoa_reference = reference_path
            nc.aoa_reference_sha256 = sha256
            nc.aoa_threshold = threshold
            nc.aoa_d_bar = d_bar
        nc.run_id = run_id
        nc.feature_contract = plan["feature_contract"]
        nc.n_features = len(plan["feature_names"])
        nc.history_days = int(plan["history_days"])
        nc.cold_start = int(cold_start)
        nc.n_unbuilt_cells = int(unbuilt.sum())
        nc.sensor_substituted_inputs = ", ".join(plan["sensor_substituted_inputs"]) or "none"
        nc.source = "src.serving.grid_predict"
    os.replace(temporary, path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "assemble"):
        command = commands.add_parser(name)
        command.add_argument("--store", required=True)
        command.add_argument("--run_id", required=True)
        if name == "plan":
            command.add_argument("--models_root", default="models")
            command.add_argument("--model_type", default="xgb")
        else:
            command.add_argument("--year", type=int, required=True)
            command.add_argument("--n_blocks", type=int, required=True)
        if name == "run":
            command.add_argument("--block", type=int, required=True)
            command.add_argument("--processes", type=int, default=1)
            command.add_argument("--cold_start", action="store_true")
            command.add_argument("--aoa_reference", default=None, help="FINAL_aoa_reference_<run>.npz: store DI too")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    store = Store(Path(args.store))
    if args.command == "plan":
        plan = plan_run(store, args.run_id, args.models_root, args.model_type)
        print(json.dumps({key: value for key, value in plan.items() if key != "feature_names"}, indent=2))
    elif args.command == "run":
        run_block(
            store, args.run_id, args.year, args.block, args.n_blocks, args.processes, args.cold_start, args.aoa_reference
        )
    else:
        logger.info("wrote %s", assemble(store, args.run_id, args.year, args.n_blocks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
