"""Rebuilding a merged daily tree under the servable feature contract."""

import csv
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from src.feature_contract.definitions import SIDECAR_FILE
from src.feature_contract.rebuild_daily import CONTRACT_NAME, rebuild_tree

LATS = np.array([52.0, 51.9, 51.8])
LONS = np.array([10.0, 10.1, 10.2])
FAM = "volumetric_soil_water_layer_1"
RAW = [0.30, 0.35, 0.40]
# The cube stores float32; the rebuild divides by what it reads back.
S = float(np.float32(0.05))
M = float(np.float32(0.30))
HOURS = 24
# Decimal strings pandas' DEFAULT float parser misreads by 1-2 ulp (job 46722293, pandas 2.3.3).
MISREAD_BY_DEFAULT_PARSER = ["206.45833333333334", "-1.0985725869735084e-07", "0.009523809523809525"]


def _write_cube(path):
    sigma = np.full((1, 3, 3), 0.05, dtype=np.float32)
    mean = np.full((1, 3, 3), 0.30, dtype=np.float32)
    xr.Dataset(
        {"swvl_sigma": (("layer", "lat", "lon"), sigma), "swvl_mean": (("layer", "lat", "lon"), mean)},
        coords={"layer": np.array([1], dtype=np.int8), "lat": LATS, "lon": LONS},
    ).to_netcdf(path)
    return path


def _site_frame(lat=51.9, lon=10.1):
    raw = np.array(RAW)
    site_sigma = 0.01  # the legacy per-site sigma the rebuild must replace
    return pd.DataFrame(
        {
            "TIMESTAMP": ["2012-04-07 00:00:00+00:00", "2012-04-08 00:00:00+00:00", "2012-04-09 00:00:00+00:00"],
            "latitude_x": [lat] * 3,
            "longitude_x": [lon] * 3,
            "sap_velocity": [1.5, np.nan, 2.25],
            "vpd": [0.123456789012345, 0.5, 0.75],
            FAM: raw / site_sigma,
            FAM + "_min": (raw - 0.01) / site_sigma,
            FAM + "_max": (raw + 0.01) / site_sigma,
            FAM + "_sum": raw * HOURS / site_sigma,
            FAM + "_raw": raw,
            FAM + "_raw_min": raw - 0.01,
            FAM + "_raw_max": raw + 0.01,
            FAM + "_raw_sum": raw * HOURS,
            FAM + "_zscore": [9.0] * 3,
            FAM + "_zscore_min": [9.0] * 3,
            FAM + "_zscore_max": [9.0] * 3,
            FAM + "_zscore_sum": [9.0] * 3,
        }
    )


def _make_src_tree(root, frame):
    config_dir = root / "src" / "fullyear_fullday_tx"
    (config_dir / "daily").mkdir(parents=True)
    frame.to_csv(config_dir / "daily" / "AAA_SITE_daily.csv", index=False)
    frame.to_csv(config_dir / "daily" / "all_biomes_merged_daily.csv", index=False)
    provenance = {"config_name": "fullyear_fullday_tx", "flags": {"apply_treatment_filter": True}}
    (config_dir / "RUN_PROVENANCE.json").write_text(json.dumps(provenance))
    return config_dir


def _csv_rows(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


def _write_rows(path, rows):
    with open(path, "w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)


@pytest.fixture
def cube_path(tmp_path):
    return _write_cube(tmp_path / "cube.nc")


@pytest.fixture
def src_tree(tmp_path):
    return _make_src_tree(tmp_path, _site_frame())


@pytest.fixture
def rebuilt(src_tree, cube_path, tmp_path):
    dst = tmp_path / "dst" / "fullyear_fullday_tx"
    manifest = rebuild_tree(src_tree, dst, cube_path, layers=(1,))
    return dst, manifest


def _read_out(dst):
    return pd.read_csv(dst / "daily" / "AAA_SITE_daily.csv", float_precision="round_trip")


def test_bare_family_is_raw_over_the_climatology_sigma(rebuilt):
    out, src = _read_out(rebuilt[0]), _site_frame()
    for suffix in ["", "_min", "_max", "_sum"]:
        np.testing.assert_allclose(out[FAM + suffix], src[FAM + "_raw" + suffix] / S, rtol=1e-12)


def test_zscore_family_is_the_climatology_anomaly(rebuilt):
    out, src = _read_out(rebuilt[0]), _site_frame()
    for suffix in ["", "_min", "_max"]:
        expected = (src[FAM + "_raw" + suffix] - M) / S
        np.testing.assert_allclose(out[FAM + "_zscore" + suffix], expected, rtol=1e-9, atol=1e-9)
    expected_sum = (src[FAM + "_raw_sum"] - HOURS * M) / S
    np.testing.assert_allclose(out[FAM + "_zscore_sum"], expected_sum, rtol=1e-9, atol=1e-9)


def test_rewritten_cells_are_the_shortest_exact_repr(rebuilt):
    rows = _csv_rows(rebuilt[0] / "daily" / "AAA_SITE_daily.csv")
    j = rows[0].index(FAM)
    assert [rows[k][j] for k in (1, 2, 3)] == [repr(r / S) for r in RAW]


def test_every_untouched_cell_is_identical_text(rebuilt, src_tree):
    src_rows = _csv_rows(src_tree / "daily" / "AAA_SITE_daily.csv")
    out_rows = _csv_rows(rebuilt[0] / "daily" / "AAA_SITE_daily.csv")
    assert out_rows[0] == src_rows[0]
    assert len(out_rows) == len(src_rows)
    header = src_rows[0]
    keep = [j for j, name in enumerate(header) if not (name.startswith(FAM) and not name.startswith(FAM + "_raw"))]
    for k in range(1, len(src_rows)):
        assert [src_rows[k][j] for j in keep] == [out_rows[k][j] for j in keep]


def test_decimals_the_default_parser_misreads_survive_unchanged(tmp_path, cube_path):
    """A read_csv/to_csv cycle would move these by 1-2 ulp; the rebuild must not touch them."""
    src = _make_src_tree(tmp_path, _site_frame())
    path = src / "daily" / "AAA_SITE_daily.csv"
    rows = _csv_rows(path)
    j = rows[0].index("vpd")
    for k, text in enumerate(MISREAD_BY_DEFAULT_PARSER, start=1):
        rows[k][j] = text
    _write_rows(path, rows)
    dst = tmp_path / "dst"
    rebuild_tree(src, dst, cube_path, layers=(1,))
    out = _csv_rows(dst / "daily" / "AAA_SITE_daily.csv")
    assert [out[k][j] for k in (1, 2, 3)] == MISREAD_BY_DEFAULT_PARSER


def test_a_ragged_row_fails_loudly(tmp_path, cube_path):
    src = _make_src_tree(tmp_path, _site_frame())
    path = src / "daily" / "AAA_SITE_daily.csv"
    rows = _csv_rows(path)
    rows[2] = rows[2][:-1]
    _write_rows(path, rows)
    with pytest.raises(ValueError, match="fields"):
        rebuild_tree(src, tmp_path / "dst", cube_path, layers=(1,))


def test_the_aggregate_file_is_not_copied(rebuilt):
    assert not (rebuilt[0] / "daily" / "all_biomes_merged_daily.csv").exists()


def test_provenance_keeps_the_source_flags_and_adds_the_contract(rebuilt, src_tree):
    provenance = json.loads((rebuilt[0] / "RUN_PROVENANCE.json").read_text())
    assert provenance["config_name"] == "fullyear_fullday_tx"
    assert provenance["flags"] == {"apply_treatment_filter": True}
    block = provenance["feature_contract"]
    assert block["name"] == CONTRACT_NAME == "servable"
    assert len(block["swvl"]["cube_sha256"]) == 64
    assert block["swvl"]["clip"] is None
    assert block["source_tree"] == str(src_tree)
    assert "precip_memory" in block


def test_cell_table_records_each_site_and_layer(rebuilt):
    cells = pd.read_csv(rebuilt[0] / "feature_contract_cells.csv")
    assert cells[["site", "layer"]].values.tolist() == [["AAA_SITE", 1]]
    assert cells["fallback"].tolist() == [False]
    assert cells["sigma"].iloc[0] == pytest.approx(S)


def test_manifest_counts(rebuilt):
    manifest = rebuilt[1]
    assert manifest["n_sites"] == 1
    assert manifest["n_rows"] == 3


def test_the_source_tree_is_not_modified(src_tree, cube_path, tmp_path):
    before = (src_tree / "daily" / "AAA_SITE_daily.csv").read_bytes()
    rebuild_tree(src_tree, tmp_path / "dst2", cube_path, layers=(1,))
    assert (src_tree / "daily" / "AAA_SITE_daily.csv").read_bytes() == before


def test_refuses_to_overwrite_an_existing_destination(src_tree, cube_path, tmp_path):
    dst = tmp_path / "exists"
    dst.mkdir()
    with pytest.raises(FileExistsError):
        rebuild_tree(src_tree, dst, cube_path, layers=(1,))


def test_refuses_a_tree_the_merge_already_built_servable(src_tree, cube_path, tmp_path):
    """Rebuilding a merge-native tree would divide by sigma twice, and the result still looks like a
    number, so nothing downstream could catch it.

    This fixture's RUN_PROVENANCE.json carries no feature_contract block -- exactly the state between
    the merge and the sweep that folds the sidecar in -- so the block check alone lets it through.
    """
    (src_tree / SIDECAR_FILE).write_text(json.dumps({"name": CONTRACT_NAME}))
    with pytest.raises(ValueError, match=SIDECAR_FILE):
        rebuild_tree(src_tree, tmp_path / "dst_guard" / "fullyear_fullday_tx", cube_path, layers=(1,))


def test_a_missing_raw_column_fails_loudly(tmp_path, cube_path):
    src = _make_src_tree(tmp_path, _site_frame().drop(columns=[FAM + "_raw"]))
    with pytest.raises(ValueError, match="_raw"):
        rebuild_tree(src, tmp_path / "dst", cube_path, layers=(1,))


def test_a_family_column_without_its_raw_companion_fails_loudly(tmp_path, cube_path):
    """Otherwise the column keeps its legacy per-site value inside a tree labelled servable."""
    src = _make_src_tree(tmp_path, _site_frame().drop(columns=[FAM + "_raw_min"]))
    with pytest.raises(ValueError, match=FAM + "_min"):
        rebuild_tree(src, tmp_path / "dst", cube_path, layers=(1,))


def test_subdirectories_other_than_daily_are_reported_not_copied(tmp_path, cube_path):
    src = _make_src_tree(tmp_path, _site_frame())
    (src / "hourly").mkdir()
    dst = tmp_path / "dst"
    manifest = rebuild_tree(src, dst, cube_path, layers=(1,))
    assert manifest["not_copied_dirs"] == ["hourly"]
    assert not (dst / "hourly").exists()


def test_a_site_without_a_cell_fails_loudly_and_names_it(tmp_path, cube_path):
    src = _make_src_tree(tmp_path, _site_frame(lat=-30.0, lon=150.0))
    with pytest.raises(LookupError, match="AAA_SITE"):
        rebuild_tree(src, tmp_path / "dst", cube_path, layers=(1,))


def test_zero_raw_mean_leaves_the_zscore_sum_undefined(tmp_path, cube_path):
    frame = _site_frame()
    frame.loc[0, [FAM + "_raw", FAM + "_raw_sum"]] = 0.0
    src = _make_src_tree(tmp_path, frame)
    dst = tmp_path / "dst"
    rebuild_tree(src, dst, cube_path, layers=(1,))
    rows = _csv_rows(dst / "daily" / "AAA_SITE_daily.csv")
    j = rows[0].index(FAM + "_zscore_sum")
    assert rows[1][j] == ""
    assert np.isfinite(float(rows[2][j]))
