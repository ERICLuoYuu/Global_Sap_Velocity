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
    PARAM_DIST,
    SKIP_STATUSES,
    SM_COL_NAME,
    TARGET_COL,
    SiteConfig,
    build_feature_matrix,
    compute_shap,
    fit_final_model,
    load_site_data,
    load_site_metadata,
    lookup_site_meta,
    plot_dependence_pair,
    process_one_site,
    save_model,
    save_shap_parquet,
    tune_site_hp,
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


# ── site metadata tests (Task 5) ───────────────────────────────────────────


def test_load_site_metadata_returns_frame_with_site_code_column(tmp_path):
    csv = tmp_path / "meta.csv"
    csv.write_text("site_code,PFT,biome\nARG_MAZ,ENF,Temperate forest\n")
    meta = load_site_metadata(csv)
    assert "site_code" in meta.columns
    assert meta.loc[meta.site_code == "ARG_MAZ", "PFT"].iloc[0] == "ENF"


def test_lookup_site_meta_unknown_site_returns_sentinels(tmp_path):
    csv = tmp_path / "meta.csv"
    csv.write_text("site_code,PFT,biome\nARG_MAZ,ENF,Temperate forest\n")
    meta = load_site_metadata(csv)
    info = lookup_site_meta(meta, "NOT_IN_FILE")
    assert info == {"PFT": "unknown", "biome": "unknown"}


def test_lookup_site_meta_known_site():
    import io

    meta = pd.read_csv(io.StringIO("site_code,PFT,biome\nFIN_HYY,ENF,Boreal forest\n"))
    info = lookup_site_meta(meta, "FIN_HYY")
    assert info == {"PFT": "ENF", "biome": "Boreal forest"}


def test_load_site_metadata_accepts_site_biome_only_file(tmp_path):
    """Real site_biome_mapping.csv header is `site,biome` with no PFT column."""
    csv = tmp_path / "meta.csv"
    csv.write_text("site,biome\nAUT_PAT_KRU,Boreal forest\n")
    meta = load_site_metadata(csv)
    assert "site_code" in meta.columns
    assert "PFT" in meta.columns
    row = meta.loc[meta.site_code == "AUT_PAT_KRU"].iloc[0]
    assert row["biome"] == "Boreal forest"
    assert row["PFT"] == "unknown"


# ── HP tuning tests (Task 6) ───────────────────────────────────────────────


def _make_synthetic_xy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "vpd": rng.uniform(0.5, 2.5, n),
            "ta": rng.uniform(10, 30, n),
            "ws": rng.uniform(0.5, 3.0, n),
            "sw_in": rng.uniform(100, 300, n),
            "precip_sum": rng.uniform(0, 5, n),
            SM_COL_NAME: rng.uniform(0.15, 0.35, n),
        }
    )
    y = 0.002 * X["sw_in"] + 5.0 * X["sm"] + 0.05 * X["vpd"] + rng.normal(0, 0.05, n)
    return X, pd.Series(y)


def test_tune_site_hp_returns_expected_keys():
    X, y = _make_synthetic_xy(n=200)
    result = tune_site_hp(X, y, random_state=42)
    assert set(PARAM_DIST.keys()).issubset(result.best_params.keys())


def test_tune_site_hp_cv_r2_is_positive_on_learnable_data():
    X, y = _make_synthetic_xy(n=200)
    result = tune_site_hp(X, y, random_state=42)
    assert result.cv_r2_mean > 0.5, f"Expected learnable synthetic data; got R2={result.cv_r2_mean}"


# ── final model + SHAP tests (Task 7) ──────────────────────────────────────


def test_fit_final_model_predicts_reasonably():
    X, y = _make_synthetic_xy(n=200)
    best_params = {
        "max_depth": 4,
        "min_child_weight": 3,
        "n_estimators": 300,
        "subsample": 1.0,
        "gamma": 0.0,
    }
    fit = fit_final_model(X, y, best_params, random_state=42)
    assert fit.in_sample_r2 > 0.6


def test_compute_shap_shapes_and_identity():
    X, y = _make_synthetic_xy(n=120)
    best_params = {
        "max_depth": 3,
        "min_child_weight": 3,
        "n_estimators": 200,
        "subsample": 1.0,
        "gamma": 0.0,
    }
    fit = fit_final_model(X, y, best_params, random_state=42)
    shap_res = compute_shap(fit.model, X)

    n, p = X.shape
    assert shap_res.shap_values.shape == (n, p)
    assert shap_res.shap_interaction is not None
    assert shap_res.shap_interaction.shape == (n, p, p)
    recovered = shap_res.shap_interaction.sum(axis=2)
    np.testing.assert_allclose(recovered, shap_res.shap_values, atol=1e-3)


def test_compute_shap_main_effect_length_matches_x():
    X, y = _make_synthetic_xy(n=150)
    best_params = {
        "max_depth": 3,
        "min_child_weight": 3,
        "n_estimators": 200,
        "subsample": 1.0,
        "gamma": 0.0,
    }
    fit = fit_final_model(X, y, best_params, random_state=42)
    shap_res = compute_shap(fit.model, X)
    assert shap_res.main_effect_sm.shape == (150,)


# ── plot tests (Task 8) ────────────────────────────────────────────────────


def test_plot_dependence_pair_writes_png(tmp_path):
    X, y = _make_synthetic_xy(n=120)
    best_params = {
        "max_depth": 3,
        "min_child_weight": 3,
        "n_estimators": 200,
        "subsample": 1.0,
        "gamma": 0.0,
    }
    fit = fit_final_model(X, y, best_params, random_state=42)
    shap_res = compute_shap(fit.model, X)

    out_png = tmp_path / "test_plot.png"
    plot_dependence_pair(
        X=X,
        shap_result=shap_res,
        sm_variant="raw",
        site_meta={
            "site_code": "FAKE_SITE",
            "PFT": "ENF",
            "biome": "Temperate forest",
            "n_rows": len(X),
            "cv_r2_mean": 0.78,
            "cv_r2_std": 0.04,
            "in_sample_r2": 0.91,
            "best_params": best_params,
        },
        output_path=out_png,
    )
    assert out_png.exists()
    assert out_png.stat().st_size > 10_000


# ── writer tests (Task 9) ──────────────────────────────────────────────────


def test_save_model_roundtrip(tmp_path):
    import joblib

    X, y = _make_synthetic_xy(n=120)
    best_params = {
        "max_depth": 3,
        "min_child_weight": 3,
        "n_estimators": 200,
        "subsample": 1.0,
        "gamma": 0.0,
    }
    fit = fit_final_model(X, y, best_params, random_state=42)
    path = tmp_path / "m.joblib"
    save_model(fit.model, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.predict(X), fit.model.predict(X))


def test_save_shap_parquet_has_expected_columns(tmp_path):
    X, y = _make_synthetic_xy(n=50)
    best_params = {
        "max_depth": 3,
        "min_child_weight": 3,
        "n_estimators": 200,
        "subsample": 1.0,
        "gamma": 0.0,
    }
    fit = fit_final_model(X, y, best_params, random_state=42)
    shap_res = compute_shap(fit.model, X)
    timestamps = pd.date_range("2020-06-01", periods=50, freq="D")
    path = tmp_path / "s.parquet"
    save_shap_parquet(
        shap_result=shap_res,
        X=X,
        timestamps=timestamps,
        output_path=path,
    )
    loaded = pd.read_parquet(path)
    assert {
        "TIMESTAMP",
        "sm",
        "shap_sm",
        "main_effect_sm",
        "shap_vpd",
        "shap_ta",
        "shap_ws",
        "shap_sw_in",
        "shap_precip_sum",
    }.issubset(loaded.columns)
    assert len(loaded) == 50


# ── process_one_site tests (Task 10) ───────────────────────────────────────


def test_process_one_site_missing_file(tmp_path):
    cfg = SiteConfig(
        site_code="NO_SUCH_SITE",
        site_csv=tmp_path / "missing.csv",
        sm_variant="raw",
        min_rows=100,
        output_root=tmp_path / "out",
        site_meta={"PFT": "unknown", "biome": "unknown"},
        random_state=42,
    )
    row = process_one_site(cfg)
    assert row["status"] == "MISSING_FILE"
    assert row["site_code"] == "NO_SUCH_SITE"


def test_process_one_site_too_few_rows(tmp_path):
    site_csv = FIXTURE_DIR / "fake_site_daily.csv"
    cfg = SiteConfig(
        site_code="FAKE_SITE",
        site_csv=site_csv,
        sm_variant="raw",
        min_rows=1_000,
        output_root=tmp_path / "out",
        site_meta={"PFT": "ENF", "biome": "Temperate forest"},
        random_state=42,
    )
    row = process_one_site(cfg)
    assert row["status"] == "TOO_FEW_ROWS"


def _write_synthetic_site_csv(path: Path, n: int = 200, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    sm_raw = rng.uniform(0.15, 0.38, n)
    df = pd.DataFrame(
        {
            "TIMESTAMP": pd.date_range("2020-06-01", periods=n, freq="D"),
            "sap_velocity": (
                0.002 * rng.uniform(100, 300, n)
                + 5.0 * sm_raw
                + 0.05 * rng.uniform(0.5, 2.5, n)
                + rng.normal(0, 0.05, n)
            ),
            "vpd": rng.uniform(0.5, 2.5, n),
            "ta": rng.uniform(10, 30, n),
            "ws": rng.uniform(0.5, 3.0, n),
            "sw_in": rng.uniform(100, 300, n),
            "precip_sum": rng.uniform(0, 5, n),
            "volumetric_soil_water_layer_1_raw": sm_raw,
            "volumetric_soil_water_layer_1_zscore": (sm_raw - sm_raw.mean()) / sm_raw.std(),
        }
    )
    df["sap_velocity"] = df["sap_velocity"].clip(lower=0.01)
    df.to_csv(path, index=False)


@pytest.mark.slow
def test_process_one_site_end_to_end(tmp_path):
    """End-to-end: synthetic 200-row site, full HP search + SHAP + plot. ~60 s."""
    site_csv = tmp_path / "FAKE_SITE_daily.csv"
    _write_synthetic_site_csv(site_csv, n=200, seed=0)

    cfg = SiteConfig(
        site_code="FAKE_SITE",
        site_csv=site_csv,
        sm_variant="raw",
        min_rows=100,
        output_root=tmp_path / "out",
        site_meta={"PFT": "ENF", "biome": "Temperate forest"},
        random_state=42,
    )
    row = process_one_site(cfg)
    assert row["status"] in {"OK", "OK_NO_INTERACTION"}, row["status"]
    assert row["n_rows"] == 200
    assert row["cv_r2_mean"] > 0.3
    assert (tmp_path / "out" / "plots" / "FAKE_SITE_SM_dependence.png").exists()
    assert (tmp_path / "out" / "shap_values" / "FAKE_SITE_shap.parquet").exists()
    assert (tmp_path / "out" / "models" / "FAKE_SITE.joblib").exists()


# ── main CLI e2e (Task 11) ─────────────────────────────────────────────────


@pytest.mark.slow
def test_main_writes_summary_and_runs_on_fixture_only(tmp_path):
    """End-to-end CLI using a private data dir containing synthetic 200-row site."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_site_csv(data_dir / "FAKE_SITE_daily.csv", n=200, seed=0)

    meta_csv = tmp_path / "meta.csv"
    meta_csv.write_text("site_code,PFT,biome\nFAKE_SITE,ENF,Temperate forest\n")
    out_root = tmp_path / "out"

    from src.Analyzers import per_site_sm_shap as m

    rc = m.main(
        [
            "--sm-variant",
            "raw",
            "--n-jobs",
            "1",
            "--min-rows",
            "100",
            "--sites",
            "FAKE_SITE",
            "--output-dir",
            str(out_root),
            "--data-dir",
            str(data_dir),
            "--site-meta-csv",
            str(meta_csv),
        ]
    )
    assert rc == 0
    summary_csv = out_root / "raw" / "summary.csv"
    assert summary_csv.exists()
    summary = pd.read_csv(summary_csv)
    assert "FAKE_SITE" in summary["site_code"].values
    status = summary.loc[summary.site_code == "FAKE_SITE", "status"].iloc[0]
    assert status in {"OK", "OK_NO_INTERACTION"}, status


# ── pool figure tests (Task 12) ────────────────────────────────────────────


def test_make_pool_figure_writes_png(tmp_path):
    from src.Analyzers.per_site_sm_shap import make_pool_figure

    shap_dir = tmp_path / "shap_values"
    shap_dir.mkdir()
    rng = np.random.default_rng(0)
    for site in ["A", "B", "C"]:
        n = 50
        df = pd.DataFrame(
            {
                "TIMESTAMP": pd.date_range("2020-06-01", periods=n),
                "sm": rng.uniform(0.1, 0.4, n),
                "main_effect_sm": rng.normal(0, 0.05, n),
            }
        )
        df.to_parquet(shap_dir / f"{site}_shap.parquet")

    summary = pd.DataFrame(
        {
            "site_code": ["A", "B", "C"],
            "status": ["OK"] * 3,
            "biome": ["Boreal forest"] * 3,
            "PFT": ["ENF"] * 3,
            "n_rows": [50, 50, 50],
        }
    )
    summary.to_csv(tmp_path / "summary.csv", index=False)

    out_png = tmp_path / "pool_by_biome.png"
    make_pool_figure(
        output_root=tmp_path,
        facet_by="biome",
        output_path=out_png,
        sm_variant="raw",
    )
    assert out_png.exists()
