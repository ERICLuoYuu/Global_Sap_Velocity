"""Yang 2024 wood density: the native 0.04 degree pixel, mean of four models, with a 1-pixel neighbour rule.

The deposited maps are 9000 x 4500 although the record says 0.01 degrees. The files carry no coordinates and 0 is their
mask, so orientation and validity rest on what opening them showed on 2026-11-02.
"""

from __future__ import annotations

import math
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from src.static_features.fetch import fetch, sha256_of, zenodo_downloads
from src.static_features.rasters import pixel_index, pixel_with_reach
from src.static_features.results import NO_DATA, ExtractionFailure, ProductResult, RunContext

RECORD_ID = 10692059
ZIP_NAME = "global-wood-density-main.zip"
ZIP_MD5 = "79168d8122ecfd684b51d24a962470ce"
MAT_SUBDIRECTORY = "global-wood-density-main/Global_wood_density_predictions"
MODELS = ("rf", "lightgbm", "xgboost", "lightgbm_rf")
REACH = 1


def _model_path(directory: Path, model: str) -> Path:
    return Path(directory) / f"{model}_all.mat"


def load_ensemble(directory: Path) -> np.ndarray:
    """Mean of the four wd_mean maps, row-first, NaN where any model is not finite and above 0."""
    total = valid = None
    for model in MODELS:
        with h5py.File(_model_path(directory, model), "r") as handle:
            grid = handle["wd_mean"][()].T  # MATLAB stores column-major, so h5py's axis order is (lon, lat)
        model_valid = np.isfinite(grid) & (grid > 0)
        if total is None:
            total, valid = np.zeros(grid.shape), model_valid
        else:
            valid &= model_valid
        total += np.where(model_valid, grid, 0.0)
    ensemble = total / len(MODELS)
    ensemble[~valid] = np.nan
    return ensemble


def layers_at(directory: Path, row: int, col: int) -> np.ndarray:
    """The 32 cross-validation layers at one pixel: 8 from each of the four models."""
    layers = []
    for model in MODELS:
        with h5py.File(_model_path(directory, model), "r") as handle:
            layers.append(handle["wd"][:, col, row])
    return np.concatenate(layers)


def _is_present(block: np.ndarray) -> np.ndarray:
    return ~np.isnan(block)


def extract(context: RunContext) -> ProductResult:
    directory = context.raw_root / "yang_2024" / MAT_SUBDIRECTORY
    ensemble = load_ensemble(directory)
    n_rows, n_cols = ensemble.shape
    if n_cols != 2 * n_rows:
        raise ExtractionFailure(f"yang: maps of shape {ensemble.shape} are not a global grid of square pixels")
    step = 360.0 / n_cols

    def read_block(row_start: int, row_stop: int, col_start: int, col_stop: int) -> np.ndarray:
        return ensemble[row_start:row_stop, col_start:col_stop]

    values, status = [], []
    for site in context.sites.itertuples(index=False):
        row = pixel_index(site.lat, 90.0, -step, n_rows)
        col = pixel_index(site.lon, -180.0, step, n_cols)
        hit = pixel_with_reach(read_block, ensemble.shape, row, col, REACH, _is_present)
        if hit.row is None:
            values.append((math.nan, math.nan))
            status.append((NO_DATA, NO_DATA))
            continue
        spread = float(np.std(layers_at(directory, hit.row, hit.col), ddof=1))
        values.append((hit.value, spread))
        status.append((hit.status, NO_DATA if math.isnan(spread) else hit.status))

    columns = ["wood_density", "wood_density_sd"]
    index = pd.Index(context.sites["site_name"], name="site_name")
    return ProductResult(
        product="yang",
        values=pd.DataFrame(values, index=index, columns=columns),
        status=pd.DataFrame(status, index=index, columns=columns),
        coverage=pd.DataFrame(index=index),
        by_year=None,
        extras={},
        provenance={
            "source": f"https://zenodo.org/records/{RECORD_ID}, {ZIP_NAME}",
            "version": "Yang et al. 2024, four-model ensemble, deposited at 0.04 degrees",
            "checksum": {
                ZIP_NAME: f"md5:{ZIP_MD5}",
                **{f"{model}_all.mat": sha256_of(_model_path(directory, model)) for model in MODELS},
            },
            "time_window": None,
            "method": (
                "wood_density is the mean of the four models' wd_mean, valid where all four are finite and above 0. "
                "wood_density_sd is the sample standard deviation over all 32 cross-validation layers at the chosen pixel. "
                "Native pixel, else the nearest valid pixel within 1, longitude wrapping. Forest only."
            ),
        },
    )


def fetch_inputs(context: RunContext) -> None:
    directory = context.raw_root / "yang_2024"
    items = [item for item in zenodo_downloads(RECORD_ID, directory) if item.name == ZIP_NAME]
    if not items or items[0].md5 != ZIP_MD5:
        raise ExtractionFailure(f"yang: record {RECORD_ID} no longer holds {ZIP_NAME} with md5 {ZIP_MD5}")
    archive = fetch(items[0], directory)
    with zipfile.ZipFile(archive) as opened:
        for model in MODELS:
            member = f"{MAT_SUBDIRECTORY}/{model}_all.mat"
            if not (directory / member).exists():
                opened.extract(member, directory)
