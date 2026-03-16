# Gap Benchmark Optimization Plan

## Executive Summary

The `gap_benchmark.py` script has **3 critical correctness bugs** and **5 performance bottlenecks** that caused a 24h+ run to fail before completion. All three correctness bugs trace to a single root cause: `_infer_site_name()` truncates multi-part SAPFLUXNET site codes to 2 parts, causing cascading failures in biome lookup, CSV file resolution, and site deduplication.

**Impact ranking (by severity):**

| # | Type | Issue | Impact |
|---|------|-------|--------|
| 1 | Correctness | Site name truncation → biome "Unknown" | ~80% of sites misclassified |
| 2 | Correctness | CSV glob pattern fails for 2-part site codes | 8+ sites silently dropped |
| 3 | Correctness | Duplicate site entries in summary | Sites processed up to 8× |
| 4 | Performance | ML models retrained per replicate | ~8h wasted (RF+XGB+KNN) |
| 5 | Performance | Cubic spline fits entire series | 72× slower than necessary |
| 6 | Performance | STL decomposition repeated per replicate | ~2.5h wasted |
| 7 | Performance | MDV per-timestamp Python loop | ~24min wasted |
| 8 | Performance | No parallelization | Single-threaded on HPC |

**Expected improvement:** From >24h (incomplete) to ~1-2h (complete) on a single zen4 node with 16 cores.

## Bug Fixes

### Fix 1: `_infer_site_name` — strip suffix instead of truncating

**Before:**
```python
def _infer_site_name(stem: str) -> str:
    parts = stem.replace('_sapf_data_outliers_removed', '').split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else parts[0]
```

**After:**
```python
def _infer_site_name(stem: str) -> str:
    return stem.replace('_sapf_data_outliers_removed', '')
```

**Expected behavior:** `FIN_HYY_SME_sapf_data_outliers_removed` → `FIN_HYY_SME` (was: `FIN_HYY`). This fixes:
- Biome lookup: `FIN_HYY_SME_site_md.csv` now found
- CSV glob: Correct full site code used in glob pattern
- Deduplication: Each file maps to a unique site name

### Fix 2: CSV glob pattern in Phase 3

**Before:**
```python
_candidates = sorted(SAP_DIR.glob(f'{_site}_*_sapf_data_outliers_removed.csv'))
```

**After:**
```python
_candidates = sorted(SAP_DIR.glob(f'{_site}_sapf_data_outliers_removed.csv'))
```

With Fix 1, `_site` is now the full site code (e.g., `ARG_TRE`), so we can match the exact filename without wildcards. This is both correct and faster.

### Fix 3: Deduplicate site_summary before Phase 3

**Before:** `selected_sites` may contain duplicate rows for the same site (from Phase 1 appending one row per file, which after Fix 1 should be one row per unique site). As a safety net:

**After:** Add `.drop_duplicates(subset='site', keep='first')` before iterating in Phase 3.

## Performance Optimizations

### Opt 1: Cache ML/DL models per segment (expected: ~10× speedup for C/D groups)

**Problem:** For each `(segment, method, gap_size, replicate)`, the code calls `fill_rf(gapped_series)` which:
1. Builds lag features from the gapped series
2. Trains a fresh RF/XGB/KNN on observed values
3. Predicts at gap positions

The training data is essentially the same across replicates and gap sizes for a given segment — only the gap position differs.

**Solution:** Restructure Phase 5 to:
1. For each (segment, method): pre-train the model once on the full ground truth
2. For each (gap_size, replicate): inject gap, build features only at gap positions, predict using cached model

**Complexity:** Medium — requires refactoring `_fit_and_predict_ml` into separate fit/predict phases.

### Opt 2: Scope cubic interpolation to local window (expected: ~50× speedup for A_cubic)

**Problem:** `fill_cubic` calls `s.interpolate(method='spline', order=3)` on the entire series.

**Solution:** Create a local-window wrapper:
1. Identify gap positions in the series
2. Extract a window of ±`max(gap_size * 3, 72)` points around the gap
3. Interpolate only within this window
4. Splice result back into the original series

### Opt 3: Cache STL decomposition per segment (expected: ~10× speedup for B_stl)

**Problem:** STL decomposition (trend + seasonal extraction) is repeated for every replicate.

**Solution:** Pre-compute STL on the full ground truth once per segment. For each replicate, only re-interpolate the residual at gap positions.

### Opt 4: Vectorize MDV gap-filling (expected: ~5× speedup for B_mdv)

**Problem:** Python-level loop over each missing timestamp with per-timestamp windowed lookup.

**Solution:** Pre-compute a lookup table of `{(hour_of_day, day_bin): mean_value}` for each segment, then vectorized fill using `map()` or `merge()`.

### Opt 5: Add joblib parallelization (expected: ~8-16× speedup on HPC)

**Problem:** Entire benchmark runs single-threaded.

**Solution:** Parallelize at the segment level within each method:
```python
from joblib import Parallel, delayed

def _benchmark_one_segment(key, sdata, method_fn, gap_sizes, n_reps, rng_seed):
    ...  # returns list of result dicts

results = Parallel(n_jobs=n_cpus)(
    delayed(_benchmark_one_segment)(key, sdata, mfn, gsizes, N_REPLICATES, seed)
    for key, sdata in ground_truth_store.items()
)
```

**Why segment-level, not method-level:** Methods within a segment share data locality. Segment-level parallelism avoids serializing large DataFrames and works well with joblib's `loky` backend.

### Opt 6: Chunked CSV writes (expected: reduced memory from ~4GB to ~100MB peak)

**Problem:** `all_results` list accumulates ~900K dicts in memory.

**Solution:** Write results to CSV incrementally per-method using `mode='a'` (append) with header only on first write.

## Parallelization Strategy

### SLURM Design for Palma zen4

**Single-node parallel (recommended for this workload):**
```bash
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
```

Use `joblib.Parallel(n_jobs=16)` for segment-level parallelism within the Python process. This avoids the overhead of SLURM array jobs for a computation that fits comfortably on one node after optimization.

**Alternative: SLURM array jobs (if single-node insufficient):**
- Split by method group: 6 array tasks (A, B, C, D, Ce, De)
- Each task runs all segments for its method group
- Final aggregation job combines results
- Risk: need shared filesystem for partial results

## Testing Strategy

### Correctness Tests (pytest)

1. **test_infer_site_name**: Assert correct extraction for 2-part, 3-part, 4-part site codes
2. **test_biome_not_unknown**: Load all 165 sites, assert no "Unknown" biomes (requires data access)
3. **test_csv_path_resolution**: For all sites passing completeness threshold, assert CSV file exists
4. **test_no_duplicate_segments**: Assert no duplicate (site, column) pairs in ground_truth_store
5. **test_gap_filling_regression**: For 2 sites × 2 methods × 2 gap sizes × 3 replicates, assert metrics match reference values (rtol=1e-5)
6. **test_edge_cases**: Empty segments, segments shorter than gap size, all-NaN columns

### Performance Benchmarks

1. **bench_cubic_vs_linear**: Assert A_cubic on 1 segment with 5 reps completes in <30s
2. **bench_rf_cached**: Assert C_rf on 1 segment with 5 reps and 3 gap sizes completes in <60s

### HPC Smoke Test

Run the full script with reduced parameters:
- 3 methods (A_linear, B_mdv, C_rf)
- 2 gap sizes (24, 168)
- 5 replicates
- All qualifying segments

Expected: completes in <30 min on zen4 with 16 cores.

## Phased Implementation

| Phase | Changes | Risk | Rollback |
|-------|---------|------|----------|
| 1 | Fix `_infer_site_name` | Low — purely corrective | Revert single function |
| 2 | Fix CSV glob pattern | Low — follows from Phase 1 | Revert one line |
| 3 | Add deduplication safety net | Low — additive | Remove one line |
| 4 | Scope cubic to local window | Medium — must verify numerical equivalence | Revert to original `fill_cubic` |
| 5 | Cache ML models per segment | Medium — refactor required | Revert to original `_fit_and_predict_ml` |
| 6 | Cache STL decomposition | Low — isolated change | Revert to original `fill_stl` |
| 7 | Vectorize MDV | Low — isolated change | Revert to original `fill_mdv` |
| 8 | Add joblib parallelization | Medium — concurrency risks | Set `n_jobs=1` to disable |
| 9 | Chunked CSV writes | Low — additive | Revert to in-memory accumulation |

Dependencies: Phase 1 must precede 2 and 3. Phases 4-9 are independent.

## HPC Resource Estimate

### After optimization (single zen4 node)

| Resource | Value | Justification |
|----------|-------|---------------|
| Partition | zen4 | AMD EPYC, 128 cores/node |
| CPUs | 16 | Sufficient for joblib parallelism |
| Memory | 64 GB | Peak ~4GB data + 16 × ~2GB per worker |
| Wall time | 4:00:00 | Conservative; expect ~1-2h |
| Storage | ~500 MB | CSV results + PNG figures |

### Smoke test (reduced parameters)

| Resource | Value |
|----------|-------|
| CPUs | 8 |
| Memory | 32 GB |
| Wall time | 01:00:00 |

### Full run estimate

With all 21 methods, 11+7 gap sizes, 50 replicates, ~47 segments:
- Group A (4 methods): ~5 min
- Group B (3 methods): ~15 min
- Group C (3 methods): ~20 min (with model caching)
- Group D (4 methods): ~30 min (with model caching)
- Groups Ce+De (7 methods): ~30 min
- **Total estimate: ~1.5-2h** on 16 cores

## Recommendations (non-mandatory)

1. **Reduce replicates from 50 to 30** with bootstrap confidence intervals — would save ~40% compute while maintaining statistical power. The current 50 replicates with fixed-seed RNG are deterministic, so bootstrap CI from 30 replicates would give equivalent precision.

2. **Add CLI arguments** for `--n-replicates`, `--methods`, `--gap-sizes` to enable quick debugging runs without editing the script.

3. **Add progress bars** using `tqdm` for better monitoring of long-running HPC jobs.
