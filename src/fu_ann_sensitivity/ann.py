# src/fu_ann_sensitivity/ann.py
"""Per-site feed-forward ANN ensemble (Fu et al. 2022 method).

One hidden layer (~10 nodes), trained 5x with different seeds on a 60/20/20
train/val/test split, LBFGS optimizer (PyTorch has no Levenberg-Marquardt; LBFGS
is the closest full-batch second-order optimizer). Ensemble prediction is the
per-row median across repeats. Used as a smooth nonlinear response surface whose
local partial derivatives give the SWC/VPD sensitivities.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

HIDDEN_DEFAULT = 10
LBFGS_ITER = 10  # LBFGS iterations per outer step
MAX_OUTER = 50  # outer steps (early stopping usually halts sooner)
PATIENCE = 6  # validation-fail tolerance (matches MATLAB trainlm max_fail)


class _MLP(nn.Module):
    """Single-hidden-layer tanh regressor."""

    def __init__(self, n_in: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _split(n: int, seed: int):
    """Deterministic 60/20/20 train/val/test index split."""
    g = np.random.default_rng(seed)
    idx = g.permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return idx[:a], idx[a:b], idx[b:]


def _train_one(X: np.ndarray, y: np.ndarray, hidden: int, seed: int):
    """Train one network with validation early stopping; return (model, test_r).

    Mirrors MATLAB ``trainlm``: optimize on the 60% train split, monitor the 20%
    validation split each outer step, keep the best-validation weights, and stop
    after ``PATIENCE`` consecutive non-improving steps. Test r is on the held-out
    20% test split.
    """
    torch.manual_seed(seed)
    tr, va, te = _split(len(X), seed)
    x_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    model = _MLP(X.shape[1], hidden)
    opt = torch.optim.LBFGS(model.parameters(), max_iter=LBFGS_ITER, line_search_fn="strong_wolfe")
    loss_fn = nn.MSELoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(model(x_t[tr]), y_t[tr])
        loss.backward()
        return loss

    ref = va if len(va) > 0 else tr  # fall back to train loss if val empty
    best_val = float("inf")
    best_state = None
    fails = 0
    for _ in range(MAX_OUTER):
        model.train()
        opt.step(closure)
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(x_t[ref]), y_t[ref]))
        if vloss < best_val - 1e-7:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            fails = 0
        else:
            fails += 1
            if fails >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_te = model(x_t[te]).numpy()
    r = float(np.corrcoef(pred_te, y[te])[0, 1]) if len(te) > 2 and pred_te.std() > 0 else 0.0
    return model, r


def train_ensemble(X, y, n_repeats: int = 5, hidden: int = HIDDEN_DEFAULT, seed: int = 0):
    """Train an ensemble; return (list_of_models, median_test_r).

    X: (n, n_pred) z-scored predictors; y: (n,) z-scored response.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    models = []
    rs = []
    for k in range(n_repeats):
        m, r = _train_one(X, y, hidden, seed + k)
        models.append(m)
        rs.append(r)
    return models, float(np.median(rs))


def ensemble_predict(models, X) -> np.ndarray:
    """Per-row median prediction across the ensemble."""
    x_t = torch.tensor(np.asarray(X, dtype=np.float64), dtype=torch.float32)
    with torch.no_grad():
        preds = np.stack([m(x_t).numpy() for m in models], axis=0)
    return np.median(preds, axis=0)
