import json
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.feature_contract.definitions import SWC_LAYERS, swc_level_col
from src.hyperparameter_optimization.target_transformer import TargetTransformer
from src.serving import grid_predict
from src.serving.acquire_era5 import STORE_COLUMNS
from src.serving.engineer import engineer_cell, model_matrix
from src.serving.frames import cell_frame
from src.serving.grid import Grid
from src.serving.model_bundle import bundle_paths, load_bundle
from src.serving.store import Store, save_grid, write_json, year_days

RUN = "grid_unit_run"
YEAR = 2015
PFT = ("DBF", "ENF")
FEATURES = ["DBF", "ENF", "sw_in", "ta", "ta_roll7d_mean", "volumetric_soil_water_layer_1_zscore", "vpd"]
READ = ("sw_in", "ta", "vpd", swc_level_col(1))
GRID = Grid(west=5.95, north=50.05, step=0.1, n_rows=3, n_cols=4)
ROWS, COLS = [0, 1, 2], [1, 3, 0]
SOIL_RUN = "grid_soil_run"
SOIL_FEATURES = [*FEATURES, "soil_water_deficit"]
SOIL_READ = (*READ, swc_level_col(2))
PFT_RUN = "grid_pft_run"


def _write_bundle(models_root, features, run):
    numeric = [name for name in features if name not in PFT]
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, len(features))).astype(np.float32)
    scaler = StandardScaler().fit(pd.DataFrame(x[:, [features.index(n) for n in numeric]], columns=numeric))
    transformer = TargetTransformer(method="log1p")
    model = LinearRegression().fit(x, transformer.fit_transform(np.abs(rng.normal(size=200)) * 5))
    config = {
        "feature_names": features,
        "feature_contract": "servable",
        "time_scale": "daily",
        "data_info": {"IS_WINDOWING": False},
        "preprocessing": {"target_transform": "log1p"},
    }
    paths = bundle_paths(models_root, run)
    paths.config.parent.mkdir(parents=True, exist_ok=True)
    paths.config.write_text(json.dumps(config))
    joblib.dump(model, paths.model)
    joblib.dump(scaler, paths.scaler)
    joblib.dump(transformer, paths.transformer)


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    root = tmp_path_factory.mktemp("serving")
    store = Store(root / "store")
    Path(store.root).mkdir(parents=True)
    save_grid(store, GRID)
    cells = pd.DataFrame(
        {
            "row": ROWS,
            "col": COLS,
            "latitude": GRID.latitudes()[ROWS],
            "longitude": GRID.longitudes()[COLS],
            "igbp": [1, 4, 1],
            "pft": ["ENF", "DBF", "ENF"],
        }
    )
    for layer in SWC_LAYERS:
        cells[f"swvl_mean_{layer}"] = 0.3
        cells[f"swvl_sigma_{layer}"] = 0.06
        cells[f"swvl_fallback_{layer}"] = False
    cells.loc[2, "swvl_sigma_1"] = np.nan
    cells.to_parquet(store.cells, index=False)

    rng = np.random.default_rng(11)
    for year in (YEAR - 1, YEAR):
        days = year_days(year)
        season = np.sin(2 * np.pi * days.dayofyear.to_numpy()[:, None] / 365.25)
        shape = (len(days), len(cells))
        values = {column: rng.uniform(0.1, 1.0, size=shape) for column in STORE_COLUMNS}
        values["ta"] = 10 + 10 * season + rng.normal(0, 2, shape)
        values["vpd"] = 0.8 + 0.5 * season + rng.uniform(0, 0.3, shape)
        values["sw_in"] = 180 + 80 * season + rng.uniform(0, 40, shape)
        values[swc_level_col(1)] = 0.3 + 0.05 * season + rng.normal(0, 0.01, shape)
        for column, array in values.items():
            path = Path(store.era5(year, column))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, array.astype(np.float32))
        for day in days:
            write_json(store.era5_done(year, day), {"day": str(day.date())})

    models_root = root / "models"
    _write_bundle(models_root, FEATURES, RUN)
    plan = grid_predict.plan_run(store, RUN, models_root)
    return store, models_root, pd.read_parquet(store.cells), plan


@pytest.fixture(scope="module")
def served_soil(served, tmp_path_factory):
    """``served``'s store with soil hydraulics at cells 0 and 2 but not at cell 1, and a model reading soil_water_deficit.

    Cell 0 lacks a soil-moisture climatology instead: its values are NaN, yet every feature is built.
    """
    root = tmp_path_factory.mktemp("serving_soil")
    store = Store(root / "store")
    shutil.copytree(served[0].root, store.root)
    cells = pd.read_parquet(store.cells)
    cells["soil_theta_wp"] = [0.12, np.nan, 0.12]
    cells["soil_theta_fc"] = [0.31, np.nan, 0.31]
    cells["swvl_sigma_1"] = [np.nan, 0.06, 0.06]
    cells.to_parquet(store.cells, index=False)
    models_root = root / "models"
    _write_bundle(models_root, SOIL_FEATURES, SOIL_RUN)
    plan = grid_predict.plan_run(store, SOIL_RUN, models_root)
    return store, models_root, cells, plan


@pytest.fixture
def served_pft(served, tmp_path_factory):
    """A copy of ``served``'s store whose forest type at one cell is whatever the test sets, the run re-planned."""
    root = tmp_path_factory.mktemp("serving_pft")
    store = Store(root / "store")
    shutil.copytree(served[0].root, store.root)
    models_root = root / "models"
    _write_bundle(models_root, FEATURES, PFT_RUN)

    def set_forest_type(cell, value):
        cells = pd.read_parquet(store.cells)
        cells["pft"] = cells["pft"].astype(object)
        cells.loc[cell, "pft"] = value
        cells.to_parquet(store.cells, index=False)
        grid_predict.plan_run(store, PFT_RUN, models_root)
        return store, models_root, cells

    return set_forest_type


def _reference(store, models_root, cells, cell, features=FEATURES, run=RUN, columns=READ):
    days = pd.date_range(f"{YEAR - 1}-01-01", f"{YEAR}-12-31", freq="D", tz="UTC")
    series = {
        column: np.concatenate([np.load(store.era5(year, column))[:, cell] for year in (YEAR - 1, YEAR)]).astype(float)
        for column in columns
    }
    matrix = model_matrix(engineer_cell(cell_frame(days, series, cells.iloc[cell].to_dict())), features, "reference")
    return load_bundle(models_root, run).predict(matrix)[-len(year_days(YEAR)) :]


def test_plan_names_the_series_and_history_the_model_needs(served):
    store, _, _, plan = served
    assert plan["spin_up_days"] >= 6
    assert plan["history_days"] == plan["spin_up_days"] + grid_predict.SPIN_UP_MARGIN_DAYS
    assert set(READ) <= set(plan["series_columns"])
    assert "volumetric_soil_water_layer_1_zscore" in plan["required_base_columns"]
    assert plan["static_columns"] == []
    assert Path(store.plan(RUN)).exists() and Path(store.plan(RUN)).with_name("servability.csv").exists()


def test_blocks_reproduce_one_continuous_record_on_the_grid(served):
    import netCDF4

    store, models_root, cells, _ = served
    for block in (0, 1):
        grid_predict.run_block(store, RUN, YEAR, block, 2)
    path = grid_predict.assemble(store, RUN, YEAR, 2)
    with netCDF4.Dataset(path) as nc:
        nc.set_auto_mask(False)
        sap = nc["sap_velocity"][:]
        served_cell = nc["served_cell"][:]
        np.testing.assert_allclose(nc["lat"][:], GRID.latitudes())
    assert sap.shape == (365, GRID.n_rows, GRID.n_cols)
    for cell in (0, 1):
        got = sap[:, ROWS[cell], COLS[cell]]
        assert np.isfinite(got).mean() > 0.95
        np.testing.assert_allclose(got, _reference(store, models_root, cells, cell), rtol=1e-5)
    assert np.isnan(sap[:, ROWS[2], COLS[2]]).all()
    assert served_cell.sum() == 3
    outside = np.ones((GRID.n_rows, GRID.n_cols), dtype=bool)
    outside[ROWS, COLS] = False
    assert np.isnan(sap[:, outside]).all()


def test_a_worker_pool_matches_one_process(served, monkeypatch):
    store, _, _, _ = served
    monkeypatch.setattr(grid_predict, "CELL_CHUNK", 1)
    with np.load(grid_predict.run_block(store, RUN, YEAR, 0, 1, processes=1)) as data:
        single = data["predictions"].copy()
    with np.load(grid_predict.run_block(store, RUN, YEAR, 0, 1, processes=2)) as data:
        pooled = data["predictions"].copy()
    np.testing.assert_array_equal(single, pooled)


def test_a_cold_start_blanks_only_the_days_without_history(served):
    store, models_root, cells, plan = served
    history = plan["history_days"]
    with np.load(grid_predict.run_block(store, RUN, YEAR, 0, 1, cold_start=True)) as data:
        predictions = data["predictions"].copy()
    assert np.isnan(predictions[:history]).all()
    np.testing.assert_allclose(predictions[history:, 0], _reference(store, models_root, cells, 0)[history:], rtol=1e-5)


def test_an_unacquired_history_day_refuses_the_run(served):
    store = served[0]
    marker = Path(store.era5_done(YEAR - 1, pd.Timestamp(f"{YEAR - 1}-12-30", tz="UTC")))
    content = marker.read_text()
    os.remove(marker)
    try:
        with pytest.raises(FileNotFoundError, match="not acquired"):
            grid_predict.run_block(store, RUN, YEAR, 0, 1)
    finally:
        marker.write_text(content)


def test_plan_refuses_a_static_layer_the_cells_lack(served):
    store, models_root, _, _ = served
    _write_bundle(models_root, [*FEATURES, "elevation"], "static_run")
    with pytest.raises(ValueError, match="static layer"):
        grid_predict.plan_run(store, "static_run", models_root)


def test_assemble_refuses_missing_and_overlapping_blocks(served):
    store = served[0]
    grid_predict.run_block(store, RUN, YEAR, 0, 1)
    for leftover in Path(store.block(RUN, YEAR, 0)).parent.glob("block_001.npz"):
        leftover.unlink()
    with pytest.raises(FileNotFoundError):
        grid_predict.assemble(store, RUN, YEAR, 2)
    grid_predict.run_block(store, RUN, YEAR, 1, 2)
    with pytest.raises(ValueError, match="overlaps"):
        grid_predict.assemble(store, RUN, YEAR, 2)


def test_block_bounds_tile_the_cells():
    bounds = [grid_predict.block_bounds(10, block, 3) for block in range(3)]
    assert bounds[0][0] == 0 and bounds[-1][1] == 10
    assert all(bounds[k][1] == bounds[k + 1][0] for k in range(2))
    with pytest.raises(ValueError):
        grid_predict.block_bounds(10, 3, 3)


# One chunk holding all three cells, one cell per chunk through a pool, one cell per block: the unpredicted
# cell 1 is then between predicted columns, at a chunk offset, and at a block offset.
@pytest.mark.parametrize("n_blocks, chunk, processes", [(1, 128, 1), (1, 1, 2), (3, 1, 1)])
def test_a_cell_whose_inputs_cannot_build_a_feature_is_left_unpredicted_and_recorded(
    served_soil, monkeypatch, n_blocks, chunk, processes
):
    store, models_root, cells, _ = served_soil
    monkeypatch.setattr(grid_predict, "CELL_CHUNK", chunk)
    predictions = np.full((len(year_days(YEAR)), len(cells)), np.nan, dtype=np.float32)
    unbuilt_cells, unbuilt_features = [], []
    for block in range(n_blocks):
        with np.load(grid_predict.run_block(store, SOIL_RUN, YEAR, block, n_blocks, processes=processes)) as data:
            predictions[:, data["cells"]] = data["predictions"]
            unbuilt_cells += data["unbuilt_cells"].tolist()
            unbuilt_features += data["unbuilt_features"].tolist()
    # Cell 1 has no soil, so no soil_water_deficit: no prediction, and its block records which cell and why.
    assert unbuilt_cells == [1]
    assert unbuilt_features == ["soil_water_deficit"]
    assert np.isnan(predictions[:, 1]).all()
    # Cell 0 has no climatology: NaN values, not an absent feature, so nothing is recorded.
    assert np.isnan(predictions[:, 0]).all()
    np.testing.assert_allclose(
        predictions[:, 2], _reference(store, models_root, cells, 2, SOIL_FEATURES, SOIL_RUN, SOIL_READ), rtol=1e-5
    )


def test_assemble_marks_the_cells_left_unpredicted(served_soil):
    import netCDF4

    store = served_soil[0]
    for block in (0, 1):
        grid_predict.run_block(store, SOIL_RUN, YEAR, block, 2)
    with netCDF4.Dataset(grid_predict.assemble(store, SOIL_RUN, YEAR, 2)) as nc:
        nc.set_auto_mask(False)
        unbuilt = nc["unbuilt_cell"][:]
    expected = np.zeros((GRID.n_rows, GRID.n_cols), dtype=np.uint8)
    expected[ROWS[1], COLS[1]] = 1
    np.testing.assert_array_equal(unbuilt, expected)


def test_a_run_refuses_when_even_complete_inputs_cannot_build_a_feature(served_soil, monkeypatch):
    from src.serving import engineer

    store = served_soil[0]
    build = engineer.apply_all_feature_engineering

    def build_without_deficit(frame, *args, **kwargs):
        frame, names = build(frame, *args, **kwargs)
        return frame.drop(columns="soil_water_deficit", errors="ignore"), names

    # Were every cell merely left unpredicted, a feature-engineering change would pass for missing soil everywhere.
    monkeypatch.setattr(engineer, "apply_all_feature_engineering", build_without_deficit)
    with pytest.raises(ValueError, match="soil_water_deficit"):
        grid_predict.run_block(store, SOIL_RUN, YEAR, 0, 2)


def test_a_cell_with_no_forest_type_is_left_unpredicted_and_recorded(served_pft):
    store, models_root, cells = served_pft(1, None)
    with np.load(grid_predict.run_block(store, PFT_RUN, YEAR, 0, 1)) as data:
        predictions, recorded = data["predictions"].copy(), data["unbuilt_cells"].tolist()
        missing = data["unbuilt_features"].tolist()
    # No one-hot can be built for it, so it cannot supply the model's PFT features: skipped, as a soil-less cell is.
    assert recorded == [1]
    assert missing == ["DBF,ENF"]
    assert np.isnan(predictions[:, 1]).all()
    np.testing.assert_allclose(predictions[:, 0], _reference(store, models_root, cells, 0, run=PFT_RUN), rtol=1e-5)


def test_a_run_refuses_a_forest_type_that_is_not_one_of_the_model_classes(served_pft):
    store, _, _ = served_pft(1, "Mixed Forest")
    # An unrecognised name one-hots to all zeros, which would serve as silently as a real class. Block 0 of 3
    # holds cell 0 alone and would otherwise finish: the table, not the block, is what the run refuses.
    with pytest.raises(ValueError, match="Mixed Forest"):
        grid_predict.run_block(store, PFT_RUN, YEAR, 0, 3)
# --- Area of applicability: DI beside the predictions -------------------------------------------------------------


@pytest.fixture(scope="module")
def di_reference(served, tmp_path_factory):
    """An AOA reference built from cells 0 and 1's served matrices of the previous January-March, one fold per cell.

    Winter only, so the served year's summer lies outside it and the mask holds both values.
    """
    from src.aoa.prepare import build_aoa_reference

    store, _, cells, _ = served
    days = year_days(YEAR - 1)
    rows, folds = [], []
    for cell in (0, 1):
        series = {column: np.load(store.era5(YEAR - 1, column))[:, cell].astype(float) for column in READ}
        matrix = model_matrix(engineer_cell(cell_frame(days, series, cells.iloc[cell].to_dict())), FEATURES, "ref")
        winter = matrix.iloc[:90]
        finite = winter[np.isfinite(winter.to_numpy()).all(axis=1)].to_numpy(dtype=float)
        rows.append(finite)
        folds.append(np.full(len(finite), cell))
    root = tmp_path_factory.mktemp("aoa_reference")
    weights = np.linspace(0.2, 1.0, len(FEATURES))
    return build_aoa_reference(np.vstack(rows), np.concatenate(folds), weights, FEATURES, root, "grid_unit")


def _expected_di(store, cells, cell, reference_path):
    from src.serving.aoa_di import load_di_reference, matrix_di

    days = pd.date_range(f"{YEAR - 1}-01-01", f"{YEAR}-12-31", freq="D", tz="UTC")
    series = {
        column: np.concatenate([np.load(store.era5(year, column))[:, cell] for year in (YEAR - 1, YEAR)]).astype(float)
        for column in READ
    }
    matrix = model_matrix(engineer_cell(cell_frame(days, series, cells.iloc[cell].to_dict())), FEATURES, "expected")
    return matrix_di(load_di_reference(reference_path, FEATURES), matrix.to_numpy(dtype=np.float32))[
        -len(year_days(YEAR)) :
    ]


@pytest.mark.parametrize("n_blocks, chunk, processes", [(1, 128, 1), (2, 1, 2)])
def test_blocks_carry_the_di_of_the_rows_they_predicted(served, di_reference, monkeypatch, n_blocks, chunk, processes):
    store, _, cells, _ = served
    monkeypatch.setattr(grid_predict, "CELL_CHUNK", chunk)
    predictions = np.full((len(year_days(YEAR)), len(cells)), np.nan, dtype=np.float32)
    di = np.full_like(predictions, np.nan)
    for block in range(n_blocks):
        path = grid_predict.run_block(
            store, RUN, YEAR, block, n_blocks, processes=processes, aoa_reference=di_reference
        )
        with np.load(path) as data:
            predictions[:, data["cells"]] = data["predictions"]
            di[:, data["cells"]] = data["di"]
            assert str(data["aoa_reference_sha256"]) and float(data["aoa_threshold"]) > 0
    np.testing.assert_array_equal(np.isfinite(di), np.isfinite(predictions))
    for cell in (0, 1):
        np.testing.assert_allclose(di[:, cell], _expected_di(store, cells, cell, di_reference), rtol=1e-5)
    assert np.isnan(di[:, 2]).all()


def test_a_block_without_a_reference_carries_no_di(served):
    store = served[0]
    with np.load(grid_predict.run_block(store, RUN, YEAR, 0, 1)) as data:
        assert "di" not in data.files


def test_a_cold_start_blanks_the_di_with_the_predictions(served, di_reference):
    store, _, _, plan = served
    with np.load(grid_predict.run_block(store, RUN, YEAR, 0, 1, cold_start=True, aoa_reference=di_reference)) as data:
        di, predictions = data["di"].copy(), data["predictions"].copy()
    assert np.isnan(di[: plan["history_days"]]).all()
    np.testing.assert_array_equal(np.isfinite(di), np.isfinite(predictions))


def test_a_reference_for_other_features_refuses_the_run_before_any_work(served, di_reference, tmp_path, monkeypatch):
    from src.aoa.prepare import build_aoa_reference

    store = served[0]
    X = np.random.default_rng(1).normal(size=(30, len(FEATURES)))
    other = build_aoa_reference(X, np.tile([0, 1, 2], 10), np.ones(len(FEATURES)), FEATURES[::-1], tmp_path, "other")

    def no_series(*args, **kwargs):
        raise AssertionError("series were loaded before the reference was checked")

    monkeypatch.setattr(grid_predict, "load_series", no_series)
    with pytest.raises(ValueError, match="feature"):
        grid_predict.run_block(store, RUN, YEAR, 0, 1, aoa_reference=other)


def test_assemble_writes_di_and_the_aoa_mask(served, di_reference):
    import netCDF4

    from src.aoa.prepare import load_aoa_reference

    store = served[0]
    for block in (0, 1):
        grid_predict.run_block(store, RUN, YEAR, block, 2, aoa_reference=di_reference)
    threshold = load_aoa_reference(di_reference)["threshold"]
    with netCDF4.Dataset(grid_predict.assemble(store, RUN, YEAR, 2)) as nc:
        nc.set_auto_mask(False)
        sap, di, aoa = nc["sap_velocity"][:], nc["DI"][:], nc["aoa"][:]
        assert nc["DI"].threshold == pytest.approx(threshold)
        assert nc.aoa_reference_sha256 and nc.aoa_threshold == pytest.approx(threshold)
    finite = np.isfinite(di)
    np.testing.assert_array_equal(finite, np.isfinite(sap))
    np.testing.assert_array_equal(aoa[finite], (di[finite] <= threshold).astype(np.uint8))
    assert (aoa[~finite] == grid_predict.AOA_FILL).all()
    assert {0, 1} <= set(np.unique(aoa[finite]).tolist())


def test_assemble_refuses_blocks_with_and_without_di(served, di_reference):
    store = served[0]
    grid_predict.run_block(store, RUN, YEAR, 0, 2, aoa_reference=di_reference)
    grid_predict.run_block(store, RUN, YEAR, 1, 2)
    with pytest.raises(ValueError, match="AOA"):
        grid_predict.assemble(store, RUN, YEAR, 2)


def test_assemble_refuses_blocks_from_different_references(served, di_reference, tmp_path):
    from src.aoa.prepare import build_aoa_reference

    store = served[0]
    X = np.random.default_rng(2).normal(size=(30, len(FEATURES)))
    other = build_aoa_reference(X, np.tile([0, 1, 2], 10), np.ones(len(FEATURES)), FEATURES, tmp_path, "other")
    grid_predict.run_block(store, RUN, YEAR, 0, 2, aoa_reference=di_reference)
    grid_predict.run_block(store, RUN, YEAR, 1, 2, aoa_reference=other)
    with pytest.raises(ValueError, match="reference"):
        grid_predict.assemble(store, RUN, YEAR, 2)


def test_the_cli_passes_the_reference(served, di_reference):
    store = served[0]
    argv = ["run", "--store", str(store.root), "--run_id", RUN, "--year", str(YEAR), "--block", "0", "--n_blocks", "1"]
    assert grid_predict.main([*argv, "--aoa_reference", str(di_reference)]) == 0
    with np.load(store.block(RUN, YEAR, 0)) as data:
        assert "di" in data.files
