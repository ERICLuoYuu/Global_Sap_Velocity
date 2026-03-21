"""Group D/De: Deep Learning gap-filling methods (PyTorch).

Extracted from notebooks/gap_benchmark.py. Key change: _current_env_df global
removed; env_df passed explicitly. fit/predict separated for GapFiller pattern.
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.gap_filling.methods.interpolation import fill_linear

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
WINDOW = 48
DL_EPOCHS = 50
DL_PATIENCE = 5
DL_BATCH = 32
DL_VAL_SPLIT = 0.1
ENV_FEATURE_COLS = ["ta", "vpd", "sw_in", "ppfd_in", "ws"]

_HAS_TORCH = False
try:
    import torch
    import torch.nn as _nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    _HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
except ImportError:
    torch = None
    _nn = None
    _DEVICE = None
    _N_GPUS = 0


# ---------------------------------------------------------------------------
# PyTorch model classes
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class LSTMModel(_nn.Module):
        """2-layer LSTM (64 units each)."""

        def __init__(self, n_features: int = 1):
            super().__init__()
            self.lstm1 = _nn.LSTM(n_features, 64, batch_first=True)
            self.drop1 = _nn.Dropout(0.1)
            self.lstm2 = _nn.LSTM(64, 64, batch_first=True)
            self.drop2 = _nn.Dropout(0.1)
            self.fc = _nn.Linear(64, 1)

        def forward(self, x):
            out, _ = self.lstm1(x)
            out = self.drop1(out)
            out, _ = self.lstm2(out)
            out = self.drop2(out[:, -1, :])
            return self.fc(out)

    class BiLSTMModel(_nn.Module):
        """Bidirectional LSTM — uses both past and future context."""

        def __init__(self, n_features: int = 1):
            super().__init__()
            self.lstm = _nn.LSTM(n_features, 64, batch_first=True, bidirectional=True)
            self.drop = _nn.Dropout(0.1)
            self.fc = _nn.Linear(128, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.drop(out[:, -1, :])
            return self.fc(out)

    class GRUModel(_nn.Module):
        """2-layer GRU — simpler LSTM variant."""

        def __init__(self, n_features: int = 1):
            super().__init__()
            self.gru = _nn.GRU(n_features, 64, num_layers=2, dropout=0.1, batch_first=True)
            self.fc = _nn.Linear(64, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])

    class CNNModel(_nn.Module):
        """3-layer 1D-CNN (32->64->32, kernel=3)."""

        def __init__(self, n_features: int = 1):
            super().__init__()
            self.conv1 = _nn.Conv1d(n_features, 32, 3, padding=1)
            self.conv2 = _nn.Conv1d(32, 64, 3, padding=1)
            self.conv3 = _nn.Conv1d(64, 32, 3, padding=1)
            self.fc1 = _nn.Linear(32, 32)
            self.fc2 = _nn.Linear(32, 1)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = x.mean(dim=2)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    class CNNLSTMModel(_nn.Module):
        """CNN-LSTM hybrid: Conv1D(32->64) -> LSTM(64) -> Dense(1)."""

        def __init__(self, n_features: int = 1):
            super().__init__()
            self.conv1 = _nn.Conv1d(n_features, 32, 3, padding=1)
            self.conv2 = _nn.Conv1d(32, 64, 3, padding=1)
            self.lstm = _nn.LSTM(64, 64, batch_first=True)
            self.drop = _nn.Dropout(0.1)
            self.fc = _nn.Linear(64, 1)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.permute(0, 2, 1)
            out, _ = self.lstm(x)
            out = self.drop(out[:, -1, :])
            return self.fc(out)

    class PositionalEmbedding(_nn.Module):
        """Learned positional embedding for Transformer encoder."""

        def __init__(self, seq_len: int, d_model: int):
            super().__init__()
            self.emb = _nn.Embedding(seq_len, d_model)
            self.seq_len = seq_len

        def forward(self, x):
            pos = torch.arange(self.seq_len, device=x.device)
            return x + self.emb(pos)

    class TransformerModel(_nn.Module):
        """Transformer encoder (2 blocks, 2 heads, d=64)."""

        def __init__(self, n_features: int = 1):
            super().__init__()
            self.input_proj = _nn.Linear(n_features, 64)
            self.pos_emb = PositionalEmbedding(WINDOW, 64)
            encoder_layer = _nn.TransformerEncoderLayer(
                d_model=64,
                nhead=2,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True,
            )
            self.encoder = _nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.fc = _nn.Linear(64, 1)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_emb(x)
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.fc(x)

    MODEL_CLASSES = {
        "lstm": LSTMModel,
        "bilstm": BiLSTMModel,
        "gru": GRUModel,
        "cnn": CNNModel,
        "cnn_lstm": CNNLSTMModel,
        "transformer": TransformerModel,
    }

else:
    LSTMModel = BiLSTMModel = GRUModel = CNNModel = CNNLSTMModel = TransformerModel = None
    PositionalEmbedding = None
    MODEL_CLASSES = {}


# ---------------------------------------------------------------------------
# Core DL functions
# ---------------------------------------------------------------------------


def _build_dl_sequences(s: pd.Series, window: int = WINDOW, env_df: pd.DataFrame | None = None):
    """Build supervised (X, y) arrays from contiguous, fully-observed windows."""
    vals = s.values.astype(np.float32)
    env_vals = None
    n_features = 1
    if env_df is not None:
        env_aligned = env_df.reindex(s.index).ffill().bfill().fillna(0)
        env_vals = env_aligned.values.astype(np.float32)
        n_features = 1 + env_vals.shape[1]
    X_list, y_list = [], []
    for i in range(window, len(vals)):
        ctx_sap = vals[i - window : i]
        tgt = vals[i]
        if np.isnan(tgt) or np.any(np.isnan(ctx_sap)):
            continue
        if env_vals is not None:
            ctx_env = env_vals[i - window : i]
            if np.any(np.isnan(ctx_env)):
                continue
            ctx = np.column_stack([ctx_sap, ctx_env])
        else:
            ctx = ctx_sap[:, np.newaxis]
        X_list.append(ctx)
        y_list.append(tgt)
    if not X_list:
        return (
            np.empty((0, window, n_features), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def _train_dl_model(
    s: pd.Series,
    model,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    env_df: pd.DataFrame | None = None,
    device=None,
):
    """Fit a PyTorch model on observed data with early stopping.

    Returns the fitted model or None when too few training samples.
    """
    X, y = _build_dl_sequences(s, env_df=env_df)
    if len(X) < 100:
        return None
    if device is None:
        device = _DEVICE if _DEVICE is not None else torch.device("cpu")

    n_samples, window, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    X_scaled = scaler_X.fit_transform(X_flat).reshape(X.shape)
    ys = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    n_val = max(1, int(n_samples * DL_VAL_SPLIT))
    n_train = n_samples - n_val
    X_train_t = torch.tensor(X_scaled[:n_train], dtype=torch.float32)
    y_train_t = torch.tensor(ys[:n_train], dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_scaled[n_train:], dtype=torch.float32).to(device)
    y_val_t = torch.tensor(ys[n_train:], dtype=torch.float32).unsqueeze(1).to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = _nn.MSELoss()

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=DL_BATCH, shuffle=True, pin_memory=False)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for _epoch in range(DL_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= DL_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    return model


def _predict_at_gaps(
    s: pd.Series,
    model,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    window: int = WINDOW,
    env_df: pd.DataFrame | None = None,
    device=None,
) -> pd.Series:
    """Autoregressively predict at gap positions using a trained model."""
    if device is None:
        device = next(model.parameters()).device if _HAS_TORCH else torch.device("cpu")
    filled = s.copy()
    vals = s.values.copy().astype(np.float32)
    env_vals = None
    n_features = 1
    if env_df is not None:
        env_aligned = env_df.reindex(s.index).ffill().bfill().fillna(0)
        env_vals = env_aligned.values.astype(np.float32)
        n_features = 1 + env_vals.shape[1]

    _prefix_end = min(window, len(vals))
    _any_prefix_nan = any(np.isnan(vals[i]) for i in range(_prefix_end))
    if _any_prefix_nan:
        _full_interp = filled.interpolate(method="linear", limit_direction="both")
        for i in range(_prefix_end):
            if np.isnan(vals[i]):
                _v = _full_interp.iloc[i]
                _v = max(0.0, float(_v)) if not np.isnan(_v) else 0.0
                vals[i] = _v
                filled.iloc[i] = _v

    model.eval()
    _gap_count = 0
    with torch.no_grad():
        for i in range(window, len(vals)):
            if np.isnan(vals[i]):
                ctx_sap = vals[i - window : i].copy()
                ctx_sap = pd.Series(ctx_sap).ffill().bfill().fillna(0).values.astype(np.float32)
                if env_vals is not None:
                    ctx_env = env_vals[i - window : i]
                    ctx = np.column_stack([ctx_sap, ctx_env])
                else:
                    ctx = ctx_sap[:, np.newaxis]
                ctx_flat = ctx.reshape(-1, n_features)
                ctx_sc = scaler_X.transform(ctx_flat).reshape(1, window, n_features)
                ctx_t = torch.tensor(ctx_sc, dtype=torch.float32).to(device)
                pred_sc = model(ctx_t).cpu().numpy()
                pred = float(scaler_y.inverse_transform(pred_sc)[0, 0])
                pred = max(0.0, pred)
                vals[i] = pred
                filled.iloc[i] = pred
                _gap_count += 1
                if _gap_count % 1000 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return filled.clip(lower=0)


# ---------------------------------------------------------------------------
# Combined fill (benchmark compatibility)
# ---------------------------------------------------------------------------


def _dl_fill(s: pd.Series, model_cls, env_df: pd.DataFrame | None = None) -> pd.Series:
    """Generic DL gap-filling pipeline."""
    if not _HAS_TORCH:
        return fill_linear(s)
    n_features = 1
    if env_df is not None:
        env_cols = [c for c in env_df.columns if c in ENV_FEATURE_COLS]
        if env_cols:
            env_df = env_df[env_cols]
            n_features = 1 + len(env_cols)
        else:
            env_df = None
    model = model_cls(n_features=n_features)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    fitted = _train_dl_model(s, model, scaler_X, scaler_y, env_df=env_df)
    if fitted is None:
        return fill_linear(s)
    return _predict_at_gaps(s, fitted, scaler_X, scaler_y, env_df=env_df)


# ---------------------------------------------------------------------------
# Public API: fit_dl / predict_dl
# ---------------------------------------------------------------------------


def _fit_dl_on_ground_truth(gt_series: pd.Series, model_cls, env_df: pd.DataFrame | None = None):
    """Pre-train a DL model on full ground truth. Returns (model_cpu, scaler_X, scaler_y) or None."""
    if not _HAS_TORCH:
        return None
    n_features = 1
    if env_df is not None:
        env_cols = [c for c in env_df.columns if c in ENV_FEATURE_COLS]
        if env_cols:
            env_df = env_df[env_cols]
            n_features = 1 + len(env_cols)
        else:
            env_df = None
    model = model_cls(n_features=n_features)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    fitted = _train_dl_model(gt_series, model, scaler_X, scaler_y, env_df=env_df)
    if fitted is None:
        return None
    fitted = fitted.cpu()
    gc.collect()
    return fitted, scaler_X, scaler_y


def _predict_dl_at_gaps_cached(s: pd.Series, cached, env_df: pd.DataFrame | None = None) -> pd.Series:
    """Predict at gap positions using a pre-trained DL model."""
    if cached is None:
        return fill_linear(s)
    model, scaler_X, scaler_y = cached
    if env_df is not None:
        env_cols = [c for c in env_df.columns if c in ENV_FEATURE_COLS]
        env_df = env_df[env_cols] if env_cols else None
    if _HAS_TORCH:
        device = _DEVICE if _DEVICE is not None else torch.device("cpu")
        model = model.to(device)
    return _predict_at_gaps(s, model, scaler_X, scaler_y, env_df=env_df)


def fit_dl(
    s: pd.Series,
    model_name: str = "lstm",
    env_df: pd.DataFrame | None = None,
):
    """Fit a DL model. Returns (model_cpu, scaler_X, scaler_y) or None."""
    if not _HAS_TORCH or model_name not in MODEL_CLASSES:
        return None
    return _fit_dl_on_ground_truth(s, MODEL_CLASSES[model_name], env_df=env_df)


def predict_dl(
    s: pd.Series,
    cached,
    model_name: str = "lstm",
    env_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Predict gaps using a pre-trained DL model."""
    return _predict_dl_at_gaps_cached(s, cached, env_df=env_df)
