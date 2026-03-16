# Gap Benchmark Optimization Plan — v2

## Executive Summary

The `gap_benchmark.py` script had **3 critical correctness bugs** (now fixed in v1) and **5 performance bottlenecks**. ML model caching was added in v1, cutting C-group time by ~10×. But DL methods (D/De groups = 8 methods) still retrain per replicate, and no parallelization exists. On the 192-core zen4 node, only 1 core is used.

**v1 fixes (already applied):**
- `_infer_site_name` truncation → biome, CSV, dedup bugs
- ML model caching (`_fit_ml_on_ground_truth` / `_predict_ml_at_gaps`)
- MDV tz-aware fix (`.index` instead of `.index.values`)

**v2 remaining issues, ranked by impact:**

| # | Severity | Issue | Expected Speedup |
|---|----------|-------|------------------|
| 1 | CRITICAL | DL models retrain per replicate (D/De: 8 methods) | ~50× for D/De groups |
| 2 | HIGH | No parallelization (192 cores idle) | ~16-48× overall |
| 3 | HIGH | Cubic spline O(n³) on full series | ~50× for A_cubic |
| 4 | HIGH | RF/XGB `n_jobs=-1` oversubscription risk | Prevents deadlock |
| 5 | HIGH | TF retracing from `clear_session()` per call | ~5× for D/De |
| 6 | MEDIUM | MDV per-timestamp Python loop | ~5× for B_mdv |
| 7 | MEDIUM | Resume logic not in canonical codebase | Enables restartability |
| 8 | MEDIUM | `_current_env_df` global — thread-unsafe | Prevents race conditions |
| 9 | LOW | Memory accumulation in `all_results` | Reduces GC pressure |

**Expected total runtime after v2:** ~1-2h on 48 cores (conservative), down from >24h incomplete.

## Bug Fixes (v1 — already applied)

### Fix 1: `_infer_site_name` — strip suffix
```python
# BEFORE: truncated to 2 parts → FIN_HYY_SME → FIN_HYY
# AFTER:  strip suffix → FIN_HYY_SME
def _infer_site_name(stem: str) -> str:
    return stem.replace("_sapf_data_outliers_removed", "")
```

### Fix 2: CSV glob pattern — exact match
```python
# BEFORE: f"{_site}_*_sapf_data_outliers_removed.csv"
# AFTER:  f"{_site}_sapf_data_outliers_removed.csv"
```

### Fix 3: Deduplication — `.drop_duplicates(subset='site')`

## Performance Optimizations (v2 — to implement)

### Opt 1: DL model caching (CRITICAL — ~50× speedup for D/De)

**Problem:** `_dl_fill()` builds, compiles, and trains a new Keras model for every
(segment × gap_size × replicate) call. For LSTM: 81 segments × 11 gap_sizes × 50 reps
= 44,550 model trains at ~5-10s each.

**Solution:** Mirror the ML caching pattern:

```python
def _fit_dl_on_ground_truth(gt_series, build_model_fn, env_df=None):
    """Train DL model once on full ground truth. Returns (model, scaler_X, scaler_y)."""
    keras.backend.clear_session()  # once per segment, not per replicate
    n_features = 1 + (len(env_df.columns) if env_df is not None else 0)
    model = build_model_fn(n_features=n_features)
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    fitted = _train_dl_model(gt_series, model, scaler_X, scaler_y, env_df=env_df)
    if fitted is None:
        return None
    return fitted, scaler_X, scaler_y

def _predict_dl_at_gaps_cached(s, cached, env_df=None):
    """Predict at gap positions using pre-trained DL model."""
    if cached is None:
        return fill_linear(s)
    model, scaler_X, scaler_y = cached
    return _predict_at_gaps(s, model, scaler_X, scaler_y, env_df=env_df)
```

**Register DL methods in `_DL_CACHE_CONFIGS`** analogous to `_ML_CACHE_CONFIGS`.

### Opt 2: Cubic spline local window (~50× speedup for A_cubic)

**Problem:** `s.interpolate(method="spline", order=3)` on 8000-point series is O(n³).

**Solution:**
```python
def fill_cubic(s: pd.Series) -> pd.Series:
    filled = s.copy()
    gaps = _find_gap_ranges(s)  # list of (start_idx, end_idx)
    for start, end in gaps:
        margin = max((end - start) * 3, 72)
        lo = max(0, start - margin)
        hi = min(len(s), end + margin)
        window = s.iloc[lo:hi]
        try:
            filled.iloc[start:end] = window.interpolate(
                method="spline", order=3, limit_direction="both"
            ).iloc[start - lo : end - lo]
        except Exception:
            filled.iloc[start:end] = window.interpolate(
                method="linear", limit_direction="both"
            ).iloc[start - lo : end - lo]
    if filled.isna().any():
        filled = fill_linear(filled)
    return filled.clip(lower=0)
```

### Opt 3: Method-group parallel executor with thread controls

**Problem:** 192 cores idle; nested threading risk.

**Solution:** Run segments in parallel within each method, with group-specific resource allocation:

| Group | Methods | Workers | Threads/Worker | Total Cores |
|-------|---------|---------|----------------|-------------|
| A (interp) | linear, cubic, akima, nearest | 48 | 1 | 48 |
| B (stats) | mdv, rolling, stl | 32 | 2 | 64 |
| C (ML) | rf, xgb, knn | 16 | 8 | 128 |
| D (DL) | lstm, cnn, cnn_lstm, transformer | 8 | 16 | 128 |
| Ce (env-ML) | rf_env, xgb_env, knn_env | 16 | 8 | 128 |
| De (env-DL) | lstm_env, cnn_env, ... | 8 | 16 | 128 |

**Thread control:** Set environment variables per worker process:
```python
import os
os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads_per_worker)
```

**Use `joblib.Parallel(backend="loky")`** with `forkserver`-safe imports. TF/sklearn
imported at module level is OK with loky (which uses pickle, not fork).

### Opt 4: Fix RF `n_jobs` for parallel safety

**Problem:** `n_jobs=-1` in RF/XGB means "use all cores" — catastrophic in parallel.

**Solution:** Set `n_jobs` per method group:
- C/Ce group workers: `n_jobs=8` (matches threads_per_worker)
- Or derive from `os.cpu_count() // n_workers`

### Opt 5: Eliminate `_current_env_df` global state

**Problem:** Global mutable `_current_env_df` is thread-unsafe.

**Solution:** Pass `env_df` explicitly to all fill functions. For `fill_*_env()` variants
called via the METHODS dict, wrap them in closures or use `functools.partial`.

### Opt 6: Resume logic in canonical code

**Problem:** Resume logic exists only on Palma (patched directly), not in local git.

**Solution:** Merge the Palma resume code into the local `run_phase5()`.

### Opt 7: Chunked CSV writes

**Problem:** `all_results` list grows to ~1.5M dicts.

**Solution:** Write per-method results to temp CSVs, merge at end using `pd.concat`.

## Parallelization Design

### Architecture

```
main()
  └─ run_phase5(ground_truth_store)
       ├─ For each (scale, method):
       │    ├─ Determine group → (n_workers, threads_per_worker)
       │    ├─ Pre-fit model on ground truth (if cacheable) — sequential
       │    └─ joblib.Parallel(n_jobs=n_workers):
       │         └─ _benchmark_one_segment(key, sdata, cached_model, ...)
       │              ├─ For each gap_size:
       │              │    ├─ inject_gaps_replicated()
       │              │    └─ For each replicate: predict + metrics
       │              └─ Return list[dict] of results
       ├─ Collect results, write partial CSV
       └─ Resume logic: skip completed (scale, method) pairs
```

### Worker function (thread-safe)

```python
def _benchmark_one_segment(
    key, sdata, scale, method_name, method_fn, gap_sizes,
    n_reps, rng_seed, cached_model=None, env_df=None
) -> list:
    """Pure function — no global state, no side effects."""
    rng = np.random.default_rng(rng_seed)
    gt = sdata.get(scale)
    if gt is None or len(gt) < 50:
        return []
    results = []
    for gsize in gap_sizes:
        if gsize >= len(gt) * 0.5:
            continue
        reps = inject_gaps_replicated(gt, gsize, n_reps, rng)
        for ri, (gs, gidx) in enumerate(reps):
            if not gidx:
                continue
            try:
                if cached_model is not None:
                    filled = _predict_at_gaps_cached(gs.copy(), cached_model, env_df)
                else:
                    filled = method_fn(gs.copy())
                tv = gt.iloc[gidx].values
                pv = filled.iloc[gidx].values
                met = compute_metrics(tv, pv)
            except Exception:
                met = {k: np.nan for k in ["rmse", "mae", "r2", "mape", "nse"]}
            results.append({
                "time_scale": scale, "site": sdata["site"],
                "method": method_name, "group": method_name.split("_")[0],
                "gap_size": gsize, "replicate": ri,
                "env_features": method_name.endswith("_env"), **met,
            })
    return results
```

### SLURM script

```bash
#!/bin/bash
#SBATCH --job-name=gap-bench-v2
#SBATCH --output=/scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark/gap_v2_%j.log
#SBATCH --time=06:00:00
#SBATCH --mem=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=zen4
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yu.luo@uni-muenster.de

source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark
cd /scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark

# Global thread controls (overridden per method group in Python)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u -c "
import sys; sys.path.insert(0, '.')
from notebooks.gap_benchmark import main
main()
"
```

## Testing Strategy

### Unit/Integration tests (pytest)
Already written in `notebooks/test_gap_benchmark.py`:
- `TestInferSiteName` (4 tests) — site code extraction
- `TestBiomeMapping` (3 tests) — metadata loading
- `TestCSVPathResolution` (2 tests) — file roundtrip
- `TestSegmentDeduplication` (1 test) — no duplicates
- `TestGapFillingRegression` (5 tests) — fill correctness
- `TestEdgeCases` (3 tests) — boundary conditions
- `TestPerformance` (2 tests) — cubic/MDV speed

### New tests needed:
- `TestDLCaching` — verify DL cached prediction matches non-cached
- `TestParallelSafety` — no global state leaks between workers
- `TestResumeLogic` — partial CSV loads correctly, completed pairs skipped

### HPC smoke test:
3 methods × 5 segments × 2 gap sizes × 5 replicates → should complete in <10min.

## Phased Implementation Order

| Step | Change | Risk | Dependencies |
|------|--------|------|--------------|
| 1 | Add resume logic to local code | Low | None |
| 2 | DL model caching | Medium | None |
| 3 | Cubic local-window scoping | Medium | None |
| 4 | Eliminate `_current_env_df` global | Medium | Needed before parallelization |
| 5 | Fix RF `n_jobs` | Low | Needed before parallelization |
| 6 | Method-group parallel executor | High | Steps 4, 5 |
| 7 | Thread controls in worker init | Medium | Step 6 |
| 8 | Chunked CSV writes | Low | Step 6 |
| 9 | SLURM job script | Low | Steps 6-8 |

## HPC Resource Estimate

### After v2 optimization (zen4 node)

| Resource | Value | Justification |
|----------|-------|---------------|
| Partition | zen4 | 192 cores, 763GB RAM |
| CPUs | 192 | Full node, `--exclusive` |
| Memory | all (`--mem=0`) | Prevents OOM from any group |
| Wall time | 06:00:00 | Conservative; expect ~2-3h |
| Storage | ~500 MB | CSV results + PNG figures |

### Runtime estimate by method group

| Group | Methods | Time (v1 sequential) | Time (v2 parallel) |
|-------|---------|---------------------|-------------------|
| A (4) | linear, cubic, akima, nearest | ~40min | ~2min |
| B (3) | mdv, rolling, stl | ~3h | ~10min |
| C (3) | rf, xgb, knn | ~2h (cached) | ~15min |
| D (4) | lstm, cnn, cnn_lstm, transformer | >24h (uncached) | ~45min |
| Ce (3) | rf_env, xgb_env, knn_env | ~2h (cached) | ~15min |
| De (4) | lstm_env, ..., transformer_env | >24h (uncached) | ~45min |
| **Total** | **21 methods × 2 scales** | **>50h** | **~2-3h** |
