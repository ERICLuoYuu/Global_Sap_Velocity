import pickle

import numpy as np
import pandas as pd
import pytest

from src.feature_contract.definitions import SERVABLE_CONTRACT
from src.hyperparameter_optimization.feature_engineering import apply_all_feature_engineering
from src.hyperparameter_optimization.training_utils import add_time_features
from src.serving.dependencies import synthetic_cell
from src.serving.engineer import AbsentFeatures, engineer_cell, model_matrix


@pytest.fixture(scope="module")
def cell():
    return synthetic_cell(n_days=90)


def test_datetime_index_gives_what_the_trainer_gets_from_csv_strings(cell):
    served = engineer_cell(cell)
    # The trainer reads solar_TIMESTAMP as text (read_csv parses only TIMESTAMP) and indexes on the strings.
    as_read = cell.assign(solar_TIMESTAMP=cell["solar_TIMESTAMP"].astype(str))
    trainer = as_read.set_index("solar_TIMESTAMP")
    trainer.sort_index(inplace=True)
    trainer = add_time_features(trainer, datetime_column=None)
    trainer, _ = apply_all_feature_engineering(trainer, "daily", contract=SERVABLE_CONTRACT)
    numeric = [c for c in trainer.columns if pd.api.types.is_numeric_dtype(trainer[c])]
    assert set(numeric) <= set(served.columns)
    np.testing.assert_array_equal(served[numeric].to_numpy(dtype=float), trainer[numeric].to_numpy(dtype=float))


def test_rows_are_engineered_in_time_order_whatever_the_input_order(cell):
    shuffled = cell.sample(frac=1.0, random_state=3)
    pd.testing.assert_frame_equal(engineer_cell(shuffled), engineer_cell(cell))


def test_required_columns_are_enforced(cell):
    with pytest.raises(KeyError):
        engineer_cell(cell.drop(columns="latitude"))


def test_model_matrix_order_dtype_onehot_and_missing_history(cell):
    engineered = engineer_cell(cell)
    names = ["ta", "ENF", "vpd_lag1d", "DBF"]
    before = list(engineered.columns)
    matrix = model_matrix(engineered, names, "cell")
    assert list(matrix.columns) == names
    assert set(matrix.dtypes) == {np.dtype("float32")}
    assert (matrix["ENF"] == 1.0).all() and (matrix["DBF"] == 0.0).all()
    assert np.isnan(matrix["vpd_lag1d"].iloc[0]) and matrix["vpd_lag1d"].iloc[1:].notna().all()
    assert list(engineered.columns) == before


def test_model_matrix_names_every_feature_it_cannot_supply(cell):
    with pytest.raises(AbsentFeatures, match="soil_moisture_rel") as raised:
        model_matrix(engineer_cell(cell), ["ta", "soil_moisture_rel", "not_a_feature"], "cell")
    # Still a KeyError to callers that catch one, and the names survive a trip through a worker pool.
    assert isinstance(raised.value, KeyError)
    assert raised.value.missing == ("soil_moisture_rel", "not_a_feature")
    assert pickle.loads(pickle.dumps(raised.value)).missing == ("soil_moisture_rel", "not_a_feature")


def test_model_matrix_names_the_pft_features_a_cell_with_no_forest_type_cannot_supply(cell):
    # A cell the PFT raster left blank cannot be one-hot encoded, so it cannot supply those features either.
    with pytest.raises(AbsentFeatures, match="ENF") as raised:
        model_matrix(engineer_cell(cell.assign(pft=np.nan)), ["ta", "ENF", "DBF"], "cell")
    assert raised.value.missing == ("ENF", "DBF")


def test_infinity_reads_as_missing(cell):
    engineered = engineer_cell(cell)
    engineered.iloc[5, engineered.columns.get_loc("ta")] = np.inf
    assert np.isnan(model_matrix(engineered, ["ta"], "cell")["ta"].iloc[5])
