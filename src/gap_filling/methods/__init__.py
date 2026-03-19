"""Method registry mapping names to fit/predict/fill callables."""

from src.gap_filling.methods.interpolation import (
    fill_akima,
    fill_cubic,
    fill_linear,
    fill_nearest,
)
from src.gap_filling.methods.ml import (
    fit_knn,
    fit_rf,
    fit_xgb,
    predict_knn,
    predict_rf,
    predict_xgb,
)
from src.gap_filling.methods.statistical import (
    fill_mdv,
    fill_rolling_mean,
    fill_stl,
)

METHODS: dict[str, dict] = {
    # Group A — no training
    "A_linear": {"group": "A", "fill": fill_linear},
    "A_cubic": {"group": "A", "fill": fill_cubic},
    "A_akima": {"group": "A", "fill": fill_akima},
    "A_nearest": {"group": "A", "fill": fill_nearest},
    # Group B — no training
    "B_mdv": {"group": "B", "fill": fill_mdv},
    "B_rolling": {"group": "B", "fill": fill_rolling_mean},
    "B_stl": {"group": "B", "fill": fill_stl},
    # Group C — ML (fit/predict)
    "C_rf": {"group": "C", "fit": fit_rf, "predict": predict_rf},
    "C_xgb": {"group": "C", "fit": fit_xgb, "predict": predict_xgb},
    "C_knn": {"group": "C", "fit": fit_knn, "predict": predict_knn},
    # Group Ce — ML + env
    "Ce_rf_env": {"group": "Ce", "fit": fit_rf, "predict": predict_rf, "env": True},
    "Ce_xgb_env": {"group": "Ce", "fit": fit_xgb, "predict": predict_xgb, "env": True},
    "Ce_knn_env": {"group": "Ce", "fit": fit_knn, "predict": predict_knn, "env": True},
}

# DL methods added conditionally
try:
    from src.gap_filling.methods.dl import MODEL_CLASSES, fit_dl, predict_dl

    for _name, _cls_name in [
        ("D_lstm", "lstm"),
        ("D_cnn", "cnn"),
        ("D_cnn_lstm", "cnn_lstm"),
        ("D_transformer", "transformer"),
        ("D_bilstm", "bilstm"),
        ("D_gru", "gru"),
    ]:
        METHODS[_name] = {
            "group": "D",
            "fit": fit_dl,
            "predict": predict_dl,
            "model_name": _cls_name,
        }

    for _name, _cls_name in [
        ("De_lstm_env", "lstm"),
        ("De_cnn_env", "cnn"),
        ("De_cnn_lstm_env", "cnn_lstm"),
        ("De_transformer_env", "transformer"),
        ("De_bilstm_env", "bilstm"),
        ("De_gru_env", "gru"),
    ]:
        METHODS[_name] = {
            "group": "De",
            "fit": fit_dl,
            "predict": predict_dl,
            "model_name": _cls_name,
            "env": True,
        }
except ImportError:
    pass  # PyTorch not available — DL methods excluded

METHOD_GROUPS = {
    "A": [k for k, v in METHODS.items() if v["group"] == "A"],
    "B": [k for k, v in METHODS.items() if v["group"] == "B"],
    "C": [k for k, v in METHODS.items() if v["group"] == "C"],
    "Ce": [k for k, v in METHODS.items() if v["group"] == "Ce"],
    "D": [k for k, v in METHODS.items() if v["group"] == "D"],
    "De": [k for k, v in METHODS.items() if v["group"] == "De"],
}
