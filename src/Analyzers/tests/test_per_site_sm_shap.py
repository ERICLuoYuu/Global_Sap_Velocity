"""Tests for per_site_sm_shap."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.Analyzers.per_site_sm_shap import SKIP_STATUSES, load_site_data

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
