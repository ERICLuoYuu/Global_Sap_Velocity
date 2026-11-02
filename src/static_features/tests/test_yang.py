from __future__ import annotations

import zipfile

import h5py
import numpy as np
import pandas as pd
import pytest
from src.static_features.fetch import Download
from src.static_features.products import yang
from src.static_features.products.yang import MODELS, extract, fetch_inputs, layers_at, load_ensemble
from src.static_features.results import NEIGHBOUR, NO_DATA, OK, ExtractionFailure, RunContext, validate

N_COLS, N_ROWS = 36, 18  # 10 degree pixels, so the site at (45, 5) is row 4, col 18


def _mat_directory(tmp_path):
    return tmp_path / "yang_2024" / "global-wood-density-main" / "Global_wood_density_predictions"


def _write(tmp_path, means=None, layers=None):
    """means: model -> {(row, col): value}, 0 elsewhere. layers: model -> {(row, col): 8 values}."""
    directory = _mat_directory(tmp_path)
    directory.mkdir(parents=True)
    for model in MODELS:
        wd_mean = np.zeros((N_COLS, N_ROWS))  # as h5py sees a MATLAB array: (lon, lat)
        for (row, col), value in (means or {}).get(model, {}).items():
            wd_mean[col, row] = value
        wd = np.zeros((8, N_COLS, N_ROWS))
        for (row, col), values in (layers or {}).get(model, {}).items():
            wd[:, col, row] = values
        with h5py.File(directory / f"{model}_all.mat", "w") as handle:
            handle["wd_mean"] = wd_mean
            handle["wd"] = wd
    return directory


def _all_models(points):
    return {model: points for model in MODELS}


def _context(tmp_path, lat=45.0, lon=5.0):
    sites = pd.DataFrame({"site_name": ["A"], "lat": [lat], "lon": [lon]})
    return RunContext(sites=sites, raw_root=tmp_path, output_root=tmp_path / "out", project="p")


def test_yang_axes_are_lon_then_lat(tmp_path):
    # Written at [col, row] as MATLAB's column-major file holds it; only the right pixel holds a value.
    _write(tmp_path, _all_models({(4, 18): 0.5}))
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "wood_density"] == 0.5
    assert result.status.loc["A", "wood_density"] == OK
    validate(result, ["A"])


def test_load_ensemble_is_row_first(tmp_path):
    directory = _write(tmp_path, _all_models({(4, 18): 0.5}))
    ensemble = load_ensemble(directory)

    assert ensemble.shape == (N_ROWS, N_COLS)
    assert ensemble[4, 18] == 0.5


def test_yang_zero_is_no_data_not_a_density(tmp_path):
    _write(tmp_path, _all_models({}))
    result = extract(_context(tmp_path))

    assert np.isnan(result.values.loc["A", "wood_density"])
    assert result.status.loc["A", "wood_density"] == NO_DATA
    assert result.status.loc["A", "wood_density_sd"] == NO_DATA
    validate(result, ["A"])


def test_yang_ensemble_is_the_mean_of_four_models(tmp_path):
    means = {
        model: {(4, 18): value}
        for model, value in (("rf", 0.4), ("lightgbm", 0.5), ("xgboost", 0.6), ("lightgbm_rf", 0.7))
    }
    _write(tmp_path, means)
    assert extract(_context(tmp_path)).values.loc["A", "wood_density"] == pytest.approx(0.55)


def test_yang_pixel_counts_only_with_all_four_models(tmp_path):
    means = _all_models({(4, 18): 0.5})
    means["xgboost"] = {}  # 0 in one model masks the pixel
    _write(tmp_path, means)
    assert extract(_context(tmp_path)).status.loc["A", "wood_density"] == NO_DATA


def test_yang_nan_in_one_model_masks_the_pixel(tmp_path):
    means = _all_models({(4, 18): 0.5})
    means["rf"] = {(4, 18): float("nan")}
    _write(tmp_path, means)
    assert extract(_context(tmp_path)).status.loc["A", "wood_density"] == NO_DATA


def test_yang_sd_spans_all_32_layers(tmp_path):
    # 16 layers at 0.4 and 16 at 0.6: mean 0.5, sample variance 32/31 x 0.01, sd 0.10160.
    layers = {
        "rf": {(4, 18): [0.4] * 8},
        "lightgbm": {(4, 18): [0.4] * 8},
        "xgboost": {(4, 18): [0.6] * 8},
        "lightgbm_rf": {(4, 18): [0.6] * 8},
    }
    directory = _write(tmp_path, _all_models({(4, 18): 0.5}), layers)

    assert len(layers_at(directory, 4, 18)) == 32
    assert extract(_context(tmp_path)).values.loc["A", "wood_density_sd"] == pytest.approx(0.10160, abs=1e-4)


def test_yang_neighbour_takes_value_and_sd_from_the_chosen_pixel(tmp_path):
    layers = _all_models({(4, 19): [0.5, 0.7] * 4})
    _write(tmp_path, _all_models({(4, 19): 0.6}), layers)
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "wood_density"] == 0.6
    assert result.values.loc["A", "wood_density_sd"] > 0
    assert result.status.loc["A", "wood_density"] == NEIGHBOUR
    assert result.status.loc["A", "wood_density_sd"] == NEIGHBOUR


def test_yang_beyond_one_pixel_is_no_data(tmp_path):
    _write(tmp_path, _all_models({(4, 20): 0.6}))
    assert extract(_context(tmp_path)).status.loc["A", "wood_density"] == NO_DATA


def test_yang_wraps_the_dateline(tmp_path):
    _write(tmp_path, _all_models({(4, 35): 0.6}))
    result = extract(_context(tmp_path, lat=45.0, lon=-175.0))
    assert result.values.loc["A", "wood_density"] == 0.6
    assert result.status.loc["A", "wood_density"] == NEIGHBOUR


def _fake_zip(tmp_path):
    path = tmp_path / "downloaded.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for model in MODELS:
            archive.writestr(f"global-wood-density-main/Global_wood_density_predictions/{model}_all.mat", b"mat")
        archive.writestr("global-wood-density-main/Global_wood_density_predictions/other.mat", b"other")
        archive.writestr("global-wood-density-main/README.md", b"readme")
    return path


def _fetch_context(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    return RunContext(sites=pd.DataFrame(), raw_root=raw, output_root=tmp_path / "out", project="p")


def test_fetch_inputs_unpacks_only_the_four_prediction_files(tmp_path, monkeypatch):
    archive = _fake_zip(tmp_path)
    item = Download("https://example.org", "global-wood-density-main.zip", md5=yang.ZIP_MD5, size=1)
    monkeypatch.setattr(yang, "zenodo_downloads", lambda record_id, directory: [item])
    monkeypatch.setattr(yang, "fetch", lambda download, directory: archive)
    context = _fetch_context(tmp_path)

    fetch_inputs(context)

    unpacked = context.raw_root / "yang_2024" / "global-wood-density-main" / "Global_wood_density_predictions"
    assert sorted(path.name for path in unpacked.iterdir()) == sorted(f"{model}_all.mat" for model in MODELS)


def test_fetch_inputs_refuses_a_record_whose_archive_changed(tmp_path, monkeypatch):
    item = Download("https://example.org", "global-wood-density-main.zip", md5="0" * 32, size=1)
    monkeypatch.setattr(yang, "zenodo_downloads", lambda record_id, directory: [item])
    with pytest.raises(ExtractionFailure, match="global-wood-density-main.zip"):
        fetch_inputs(_fetch_context(tmp_path))
