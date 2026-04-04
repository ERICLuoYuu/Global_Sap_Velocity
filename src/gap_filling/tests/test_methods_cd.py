import numpy as np
import pandas as pd
import pytest


def _has_torch():
    try:
        import torch

        return True
    except ImportError:
        return False


def _make_series_with_env(n=500, gap_size=6, freq="h"):
    """Create sap + env series with a gap."""
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    sap = np.abs(np.sin(np.linspace(0, 10 * np.pi, n))) * 5 + 1
    sap_s = pd.Series(sap, index=idx)
    mid = n // 2
    sap_s.iloc[mid : mid + gap_size] = np.nan
    rng = np.random.default_rng(42)
    env = pd.DataFrame(
        {
            "ta": rng.normal(20, 5, n),
            "vpd": rng.uniform(0.5, 3.0, n),
            "sw_in": np.abs(np.sin(np.linspace(0, 10 * np.pi, n))) * 800,
        },
        index=idx,
    )
    return sap_s, env, mid, gap_size


class TestGroupC:
    def test_rf_fit_predict(self):
        from src.gap_filling.methods.ml import fit_rf, predict_rf

        s, _, _, _ = _make_series_with_env()
        model = fit_rf(s)
        assert model is not None
        filled = predict_rf(s, model)
        assert filled.isna().sum() == 0

    def test_xgb_fit_predict(self):
        from src.gap_filling.methods.ml import fit_xgb, predict_xgb

        s, _, _, _ = _make_series_with_env()
        model = fit_xgb(s)
        assert model is not None
        filled = predict_xgb(s, model)
        assert filled.isna().sum() == 0

    def test_knn_fit_predict(self):
        from src.gap_filling.methods.ml import fit_knn, predict_knn

        s, _, _, _ = _make_series_with_env()
        model = fit_knn(s)
        assert model is not None
        filled = predict_knn(s, model)
        assert filled.isna().sum() == 0

    def test_rf_env_fit_predict(self):
        from src.gap_filling.methods.ml import fit_rf, predict_rf

        s, env, _, _ = _make_series_with_env()
        model = fit_rf(s, env_df=env)
        filled = predict_rf(s, model, env_df=env)
        assert filled.isna().sum() == 0

    def test_fit_returns_none_insufficient_data(self):
        from src.gap_filling.methods.ml import fit_rf

        idx = pd.date_range("2020-01-01", periods=10, freq="h")
        s = pd.Series(np.ones(10), index=idx)
        s.iloc[5:] = np.nan
        model = fit_rf(s)
        assert model is None

    def test_predict_none_model_fallback(self):
        from src.gap_filling.methods.ml import predict_rf

        s, _, _, _ = _make_series_with_env()
        filled = predict_rf(s, None)  # None model -> linear fallback
        assert filled.isna().sum() == 0

    def test_non_negative_output(self):
        from src.gap_filling.methods.ml import fit_rf, predict_rf

        s, _, _, _ = _make_series_with_env()
        model = fit_rf(s)
        filled = predict_rf(s, model)
        assert (filled >= 0).all()


class TestMethodRegistry:
    def test_methods_dict_has_groups_ab(self):
        from src.gap_filling.methods import METHOD_GROUPS, METHODS

        assert len(METHOD_GROUPS["A"]) == 4
        assert len(METHOD_GROUPS["B"]) == 3
        assert len(METHOD_GROUPS["C"]) == 3
        assert len(METHOD_GROUPS["Ce"]) == 3
        # Total non-DL methods: 13
        non_dl = [k for k, v in METHODS.items() if v["group"] not in ("D", "De")]
        assert len(non_dl) == 13

    def test_group_a_methods_have_fill(self):
        from src.gap_filling.methods import METHODS

        for name in ["A_linear", "A_cubic", "A_akima", "A_nearest"]:
            assert "fill" in METHODS[name]
            assert callable(METHODS[name]["fill"])

    def test_group_c_methods_have_fit_predict(self):
        from src.gap_filling.methods import METHODS

        for name in ["C_rf", "C_xgb", "C_knn"]:
            assert "fit" in METHODS[name]
            assert "predict" in METHODS[name]

    def test_env_methods_flagged(self):
        from src.gap_filling.methods import METHODS

        for name in ["Ce_rf_env", "Ce_xgb_env", "Ce_knn_env"]:
            assert METHODS[name].get("env") is True


@pytest.mark.skipif(not _has_torch(), reason="PyTorch not available")
class TestGroupD:
    def test_lstm_fit_predict(self):
        from src.gap_filling.methods.dl import fit_dl, predict_dl

        s, _, _, _ = _make_series_with_env(n=200, gap_size=3)
        model = fit_dl(s, model_name="lstm")
        assert model is not None
        filled = predict_dl(s, model, model_name="lstm")
        assert filled.isna().sum() == 0

    def test_dl_env_fit_predict(self):
        from src.gap_filling.methods.dl import fit_dl, predict_dl

        s, env, _, _ = _make_series_with_env(n=200, gap_size=3)
        model = fit_dl(s, model_name="lstm", env_df=env)
        filled = predict_dl(s, model, model_name="lstm", env_df=env)
        assert filled.isna().sum() == 0

    def test_dl_registry_has_all_models(self):
        from src.gap_filling.methods import METHOD_GROUPS

        assert len(METHOD_GROUPS["D"]) == 6
        assert len(METHOD_GROUPS["De"]) == 6
