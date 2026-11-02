import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.hyperparameter_optimization.target_transformer import TargetTransformer
from src.serving.model_bundle import bundle_paths, load_bundle

RUN = "unit_run"
NAMES = ["DBF", "ENF", "sw_in", "ta", "vpd"]
NUMERIC = ["sw_in", "ta", "vpd"]


def _frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, len(NAMES))), columns=NAMES).astype(np.float32)
    frame["DBF"] = 0.0
    frame["ENF"] = 1.0
    return frame


def _write_run(root, config_overrides=None, scaler_columns=NUMERIC, transformer_method="log1p"):
    frame = _frame()
    y = np.abs(np.random.default_rng(1).normal(size=len(frame))) * 5
    scaler = StandardScaler().fit(pd.DataFrame(frame[scaler_columns].to_numpy(), columns=scaler_columns))
    values = frame.to_numpy(dtype=np.float32)
    index = [NAMES.index(c) for c in NUMERIC]
    values[:, index] = StandardScaler().fit_transform(values[:, index])
    transformer = TargetTransformer(method=transformer_method)
    model = LinearRegression().fit(values, transformer.fit_transform(y))
    config = {
        "feature_names": NAMES,
        "feature_contract": "servable",
        "time_scale": "daily",
        "data_info": {"IS_WINDOWING": False},
        "preprocessing": {"target_transform": "log1p"},
    }
    config.update(config_overrides or {})
    paths = bundle_paths(root, RUN)
    paths.config.parent.mkdir(parents=True, exist_ok=True)
    paths.config.write_text(json.dumps(config))
    joblib.dump(model, paths.model)
    joblib.dump(scaler, paths.scaler)
    joblib.dump(transformer, paths.transformer)
    return model, scaler, transformer


def test_prediction_scales_numeric_columns_only_and_inverts_the_target(tmp_path):
    model, scaler, transformer = _write_run(tmp_path)
    bundle = load_bundle(tmp_path, RUN)
    frame = _frame(n=10, seed=5)
    expected = frame.to_numpy(dtype=np.float32, copy=True)
    index = [NAMES.index(c) for c in NUMERIC]
    expected[:, index] = scaler.transform(pd.DataFrame(expected[:, index], columns=NUMERIC))
    np.testing.assert_allclose(bundle.predict(frame), np.expm1(model.predict(expected)))


def test_rows_with_a_missing_feature_are_not_predicted(tmp_path):
    _write_run(tmp_path)
    bundle = load_bundle(tmp_path, RUN)
    frame = _frame(n=6, seed=2)
    whole = bundle.predict(frame)
    frame.iloc[2, frame.columns.get_loc("vpd")] = np.nan
    got = bundle.predict(frame)
    assert np.isnan(got[2])
    np.testing.assert_allclose(np.delete(got, 2), np.delete(whole, 2))


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"feature_contract": "site"}, ValueError),
        ({"data_info": {"IS_WINDOWING": True}}, NotImplementedError),
        ({"time_scale": "hourly"}, ValueError),
        ({"feature_names": NAMES[:-1] + ["volumetric_soil_water_layer_1"]}, ValueError),
    ],
    ids=["site-contract", "windowed", "hourly", "retired-swc-name"],
)
def test_unservable_configs_are_refused(tmp_path, overrides, error):
    _write_run(tmp_path, config_overrides=overrides)
    with pytest.raises(error):
        load_bundle(tmp_path, RUN)


def test_scaler_fitted_in_another_column_order_is_refused(tmp_path):
    _write_run(tmp_path, scaler_columns=list(reversed(NUMERIC)))
    with pytest.raises(ValueError, match="scaler"):
        load_bundle(tmp_path, RUN)


def test_transformer_that_is_not_the_configs_is_refused(tmp_path):
    _write_run(tmp_path, transformer_method="sqrt")
    with pytest.raises(ValueError, match="transformer"):
        load_bundle(tmp_path, RUN)


def test_matrix_in_another_column_order_is_refused(tmp_path):
    _write_run(tmp_path)
    bundle = load_bundle(tmp_path, RUN)
    with pytest.raises(ValueError, match="order"):
        bundle.predict(_frame()[list(reversed(NAMES))])


def test_missing_model_file_is_reported(tmp_path):
    _write_run(tmp_path)
    bundle_paths(tmp_path, RUN).model.unlink()
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path, RUN)
