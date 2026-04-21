"""Tests for per_site_sm_shap."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.Analyzers.per_site_sm_shap import (
    FEATURE_COLS,
    SKIP_STATUSES,
    SM_COL_NAME,
    TARGET_COL,
    build_feature_matrix,
    load_site_data,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "src" / "Analyzers" / "per_site_sm_shap.py"


def test_cli_help_runs_without_error() -> None:
    """--help must exit 0 and print the --sm-variant flag."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--sm-variant" in result.stdout
    assert "--n-jobs" in result.stdout
    assert "--min-rows" in result.stdout


# ── load_site_data tests (Task 3) ──────────────────────────────────────────

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_load_site_data_drops_nans_and_nonphysical(tmp_path: Path) -> None:
    """Rows with NaN in any selected col or sap_velocity<=0 are removed."""
    site_csv = FIXTURE_DIR / "fake_site_daily.csv"
    assert site_csv.exists(), "Fixture missing"

    df, status = load_site_data(
        site_csv=site_csv,
        sm_variant="raw",
        min_rows=1,
    )
    assert status == "OK"
    assert len(df) == 8  # 12 - 4 dropped rows
    assert df[["sap_velocity", "vpd"]].notna().all().all()
    assert (df["sap_velocity"] > 0).all()
    assert "sm" in df.columns
    assert df["sm"].between(0.20, 0.30).all()


def test_load_site_data_selects_zscore_variant() -> None:
    df, status = load_site_data(
        site_csv=FIXTURE_DIR / "fake_site_daily.csv",
        sm_variant="zscore",
        min_rows=1,
    )
    assert status == "OK"
    assert df["sm"].between(-1.5, 0.0).all()


def test_load_site_data_skips_when_too_few_rows() -> None:
    df, status = load_site_data(
        site_csv=FIXTURE_DIR / "fake_site_daily.csv",
        sm_variant="raw",
        min_rows=100,
    )
    assert df is None
    assert status == "TOO_FEW_ROWS"
    assert status in SKIP_STATUSES


def test_load_site_data_missing_file(tmp_path: Path) -> None:
    df, status = load_site_data(
        site_csv=tmp_path / "nope.csv",
        sm_variant="raw",
        min_rows=1,
    )
    assert df is None
    assert status == "MISSING_FILE"


def test_load_site_data_missing_sm_variant(tmp_path: Path) -> None:
    df_in = pd.read_csv(FIXTURE_DIR / "fake_site_daily.csv")
    df_in.drop(columns=["volumetric_soil_water_layer_1_zscore"], inplace=True)
    path = tmp_path / "no_zscore.csv"
    df_in.to_csv(path, index=False)

    df, status = load_site_data(site_csv=path, sm_variant="zscore", min_rows=1)
    assert df is None
    assert status == "MISSING_SM_VARIANT"


# ── build_feature_matrix tests (Task 4) ────────────────────────────────────


def _make_clean_df(n: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "TIMESTAMP": pd.date_range("2020-06-01", periods=n, freq="D"),
            TARGET_COL: rng.uniform(0.5, 3.0, n),
            "vpd": rng.uniform(0.5, 2.5, n),
            "ta": rng.uniform(10, 30, n),
            "ws": rng.uniform(0.5, 3.0, n),
            "sw_in": rng.uniform(100, 300, n),
            "precip_sum": rng.uniform(0, 5, n),
            SM_COL_NAME: rng.uniform(0.15, 0.35, n),
        }
    )


def test_build_feature_matrix_shape():
    df = _make_clean_df(n=30)
    X, y = build_feature_matrix(df)
    assert X.shape == (30, 6)
    assert y.shape == (30,)


def test_build_feature_matrix_column_order():
    df = _make_clean_df(n=10)
    X, _ = build_feature_matrix(df)
    assert list(X.columns) == list(FEATURE_COLS)


def test_build_feature_matrix_preserves_row_order():
    df = _make_clean_df(n=10)
    X, y = build_feature_matrix(df)
    np.testing.assert_array_equal(y.values, df[TARGET_COL].values)
