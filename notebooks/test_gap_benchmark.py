"""Quick smoke test for gap_benchmark.py functions on synthetic data."""
import sys, os
sys.path.insert(0, '..')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
from scipy import interpolate as scipy_interp
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from statsmodels.tsa.seasonal import STL

RANDOM_SEED = 42
WINDOW = 48
N_LAGS = 24

# --- Replicate key functions from gap_benchmark.py ---

def detect_gaps_in_series(series):
    if len(series) == 0:
        return []
    is_nan = pd.isna(series).values.astype(np.int8)
    idx = series.index
    n = len(is_nan)
    padded = np.concatenate([[0], is_nan.astype(int), [0]])
    diff = np.diff(padded)
    gap_starts = np.where(diff == 1)[0]
    gap_ends = np.where(diff == -1)[0]
    gaps = []
    for s, e in zip(gap_starts, gap_ends):
        gaps.append({'start': idx[s], 'end': idx[min(e-1, n-1)], 'duration_steps': int(e-s)})
    return gaps


def inject_gaps_replicated(gt_series, gap_size, n_reps, rng):
    n = len(gt_series)
    if n < gap_size * 3:
        return []
    max_start = n - gap_size
    min_spacing = 2 * gap_size
    results = []
    used_starts = []
    max_attempts = n_reps * 10
    attempts = 0
    while len(results) < n_reps and attempts < max_attempts:
        attempts += 1
        start = int(rng.integers(0, max_start))
        too_close = any(abs(start - prev) < min_spacing for prev in used_starts)
        if too_close:
            continue
        used_starts.append(start)
        idx = list(range(start, start + gap_size))
        masked = gt_series.copy()
        masked.iloc[idx] = np.nan
        results.append((masked, idx))
    for _ in range(n_reps - len(results)):
        start = int(rng.integers(0, max_start))
        idx = list(range(start, start + gap_size))
        masked = gt_series.copy()
        masked.iloc[idx] = np.nan
        results.append((masked, idx))
    return results


def compute_metrics(true_vals, pred_vals):
    m = ~(np.isnan(true_vals) | np.isnan(pred_vals))
    t, p = true_vals[m], pred_vals[m]
    if len(t) == 0:
        return {k: np.nan for k in ['rmse', 'mae', 'r2', 'mape', 'nse']}
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    mae = float(np.mean(np.abs(t - p)))
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    if ss_tot > 1e-12:
        r2 = float(1.0 - ss_res / ss_tot)
    else:
        r2 = 1.0 if ss_res < 1e-12 else 0.0
    nse = r2  # same formula
    nz = np.abs(t) > 1e-10
    mape = float(np.mean(np.abs((t[nz] - p[nz]) / t[nz])) * 100) if nz.sum() > 0 else np.nan
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape, 'nse': nse}


def _is_hourly_series(s):
    if len(s) < 2:
        return True
    return bool(pd.Series(s.index).diff().dropna().median() <= pd.Timedelta('2h'))


def fill_linear(s):
    return s.interpolate(method='linear', limit_direction='both').clip(lower=0)


def fill_cubic(s):
    try:
        return s.interpolate(method='spline', order=3, limit_direction='both').clip(lower=0)
    except Exception:
        return fill_linear(s)


def fill_akima(s):
    if s.notna().sum() < 4:
        return fill_linear(s)
    try:
        valid_idx = np.where(s.notna())[0]
        valid_vals = s.values[valid_idx]
        akima = scipy_interp.Akima1DInterpolator(valid_idx, valid_vals)
        filled = s.copy()
        null_pos = np.where(s.isna())[0]
        in_range = (null_pos >= valid_idx[0]) & (null_pos <= valid_idx[-1])
        if in_range.any():
            filled.iloc[null_pos[in_range]] = akima(null_pos[in_range])
        if (~in_range).any():
            filled = filled.interpolate(method='nearest', limit_direction='both')
        return filled.clip(lower=0)
    except Exception:
        return fill_linear(s)


def fill_nearest(s):
    return s.interpolate(method='nearest', limit_direction='both').clip(lower=0)


def fill_mdv(s, window_days=7):
    filled = s.copy()
    null_mask = s.isna()
    if not null_mask.any():
        return filled.clip(lower=0)
    hourly = _is_hourly_series(s)
    period_key = s.index.hour if hourly else s.index.dayofweek
    for period_val in pd.unique(period_key[null_mask]):
        missing_at_period = null_mask & (period_key == period_val)
        if not missing_at_period.any():
            continue
        for ts in s.index[missing_at_period]:
            w_start = ts - pd.Timedelta(days=window_days)
            w_end = ts + pd.Timedelta(days=window_days)
            ctx = s[(s.index >= w_start) & (s.index <= w_end) & (period_key == period_val) & s.notna()]
            if len(ctx) > 0:
                filled[ts] = max(0.0, float(ctx.mean()))
    if filled.isna().any():
        filled = fill_linear(filled)
    return filled.clip(lower=0)


def fill_rolling_mean(s, window=None):
    if window is None:
        window = 48 if _is_hourly_series(s) else 7
    rolling = s.rolling(window=window, min_periods=max(1, window // 4), center=True).mean()
    filled = s.fillna(rolling)
    if filled.isna().any():
        filled = fill_linear(filled)
    return filled.clip(lower=0)


def fill_stl(s):
    try:
        period = 24 if _is_hourly_series(s) else 7
        if len(s) < period * 2:
            return fill_rolling_mean(s)
        s_pre = s.interpolate(method='linear', limit_direction='both')
        if s_pre.isna().any():
            s_pre = s_pre.ffill().bfill()
        if s_pre.isna().any():
            s_pre = s_pre.fillna(s_pre.mean() if not s_pre.isna().all() else 0.0)
        res = STL(s_pre, period=period, robust=True).fit()
        residual = pd.Series(res.resid, index=s.index)
        residual[s.isna()] = np.nan
        residual_filled = residual.interpolate(method='linear', limit_direction='both').fillna(0.0)
        return pd.Series(res.trend + res.seasonal + residual_filled.values, index=s.index).clip(lower=0)
    except Exception:
        return fill_rolling_mean(s)


def build_ml_features(s, n_lags=N_LAGS, env_df=None):
    df = pd.DataFrame({'y': s.values}, index=s.index)
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['hour_of_day'] = df.index.hour
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['rolling_mean_24'] = df['y'].rolling(24, min_periods=1).mean()
    df['rolling_std_24'] = df['y'].rolling(24, min_periods=1).std().fillna(0)
    df['is_daytime'] = ((df.index.hour >= 6) & (df.index.hour <= 20)).astype(int)
    return df.drop(columns=['y'])


def _fit_and_predict_ml(s, model_cls, model_kwargs, env_df=None):
    features = build_ml_features(s, env_df=env_df)
    train_mask = s.notna() & features.notna().all(axis=1)
    test_mask = s.isna() & features.notna().all(axis=1)
    if train_mask.sum() < 50:
        return fill_linear(s)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(features[train_mask].values)
    model = model_cls(**model_kwargs)
    model.fit(X_train_s, s[train_mask].values)
    filled = s.copy()
    if test_mask.sum() > 0:
        X_test = scaler.transform(features[test_mask].values)
        filled[test_mask] = np.clip(model.predict(X_test), 0, None)
    if filled.isna().any():
        filled = fill_linear(filled)
    return filled.clip(lower=0)


def fill_rf(s):
    return _fit_and_predict_ml(s, RandomForestRegressor,
                               {'n_estimators': 100, 'n_jobs': -1, 'random_state': RANDOM_SEED})


def fill_xgb_model(s):
    return _fit_and_predict_ml(s, xgb.XGBRegressor,
                               {'n_estimators': 100, 'learning_rate': 0.1,
                                'max_depth': 5, 'random_state': RANDOM_SEED,
                                'verbosity': 0, 'tree_method': 'hist'})


def fill_knn(s, k=10):
    k_actual = min(k, max(1, int(s.notna().sum() * 0.1)))
    return _fit_and_predict_ml(s, KNeighborsRegressor, {'n_neighbors': k_actual})


# --- Tests ---
print('='*60)
print('  GAP BENCHMARK SMOKE TEST')
print('='*60)

# Synthetic sap flow
np.random.seed(42)
idx = pd.date_range('2020-01-01', periods=500, freq='h')
vals = np.maximum(0, 5 * np.sin(2 * np.pi * np.arange(500) / 24) + np.random.randn(500) * 0.5 + 3)
s = pd.Series(vals, index=idx)
s_gap = s.copy()
s_gap.iloc[100:112] = np.nan

# Test 1: Gap detection
print('\n[1] Gap detection')
gaps = detect_gaps_in_series(s_gap)
assert len(gaps) == 1
assert gaps[0]['duration_steps'] == 12
print('  PASS: 1 gap, 12 steps')

# Test 2: Gap injection
print('\n[2] Gap injection with spacing')
rng = np.random.default_rng(42)
reps = inject_gaps_replicated(s, 24, 5, rng)
assert len(reps) == 5
for i, (ms, gidx) in enumerate(reps):
    assert ms.isna().sum() == 24
print('  PASS: 5 replicates, 24h gaps, all correctly sized')

# Test 3: All Group A-C methods
print('\n[3] Method testing (Groups A-C)')
methods = [
    ('A_linear', fill_linear), ('A_cubic', fill_cubic),
    ('A_akima', fill_akima), ('A_nearest', fill_nearest),
    ('B_mdv', fill_mdv), ('B_rolling', fill_rolling_mean),
    ('B_stl', fill_stl),
    ('C_rf', fill_rf), ('C_xgb', fill_xgb_model), ('C_knn', fill_knn),
]

for name, fn in methods:
    r = fn(s_gap.copy())
    assert r.notna().all(), f'{name} left NaNs!'
    m = compute_metrics(s.iloc[100:112].values, r.iloc[100:112].values)
    print(f'  {name:>12}: R2={m["r2"]:.4f}  RMSE={m["rmse"]:.4f}  NSE={m["nse"]:.4f}  PASS')

# Test 4: Metrics edge cases
print('\n[4] Metrics edge cases')
m1 = compute_metrics(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
assert m1['r2'] == 1.0, f'Perfect constant: expected R2=1.0, got {m1["r2"]}'
print('  Constant actual = constant pred -> R2=1.0: PASS')

m2 = compute_metrics(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.5, 0.5]))
assert m2['r2'] == 0.0, f'Constant actual: expected R2=0.0, got {m2["r2"]}'
print('  Constant actual != pred -> R2=0.0: PASS')

m3 = compute_metrics(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]))
assert not np.isnan(m3['r2'])
print('  NaN handling -> valid R2: PASS')

# Test 5: Large gap (72h)
print('\n[5] Large gap test (72h)')
s_big = s.copy()
s_big.iloc[200:272] = np.nan
for name, fn in [('linear', fill_linear), ('mdv', fill_mdv), ('rf', fill_rf)]:
    r = fn(s_big.copy())
    assert r.notna().all(), f'{name} left NaNs on 72h gap!'
    m = compute_metrics(s.iloc[200:272].values, r.iloc[200:272].values)
    print(f'  {name:>8} on 72h gap: R2={m["r2"]:.4f}  RMSE={m["rmse"]:.4f}')

print('\n' + '='*60)
print('  ALL TESTS PASSED')
print('='*60)

