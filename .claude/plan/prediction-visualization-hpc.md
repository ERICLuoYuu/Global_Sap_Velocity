# Prediction Visualization HPC Deployment Plan

**Status:** IN PROGRESS
**Target:** Palma II HPC `map-viz` worktree (`/scratch/tmp/yluo2/gsv-wt/map-viz`)
**Branch:** `main` (detached in worktree)

---

## 1. Problem Statement

The existing `prediction_visualization.py` is a monolithic script hardcoded for a single CSV file,
producing a single mean-aggregated GeoTIFF. It needs to be extended to:

1. Accept CLI parameters for batch processing on HPC
2. Process multiple prediction CSVs (multi-GB files, total ~51 GB)
3. Generate per-timestamp GeoTIFFs and PNGs
4. Compute temporal composite maps (mean, median, max, min, std, count)
5. Run as a SLURM job on the zen4 partition

## 2. Exploration Findings

### 2.1 Existing Script Analysis (1079 lines)

| Function | Purpose | Reusable? |
|----------|---------|-----------|
| `get_valid_block_size()` | Block size calculation for rasterio | ✅ Yes |
| `inspect_large_csv()` | CSV statistics/inspection | ✅ Optional utility |
| `_sort_using_sqlite/dask/memory()` | Sorting by `name` column | ❌ Not needed for per-timestamp flow |
| `_create_tiled_output()` + `_process_tile()` | Tiled GeoTIFF for large grids | ✅ Yes, for global 0.1° grid |
| `plot_sap_velocity_map()` | Matplotlib visualization with shapefile mask | ✅ Yes, refactor for Cartopy |
| `create_sap_velocity_tif()` | Main pipeline: read → filter → aggregate → rasterize | ⚠️ Core logic reusable, needs refactoring |

### 2.2 HPC Venv Dependencies

**Available:** numpy 2.0.2, pandas 2.3.3, matplotlib 3.9.4, scipy 1.13.1, rasterio 1.4.3,
scikit-learn 1.6.1, xgboost 2.1.4, tqdm 4.67.3, cartopy 0.23.0, psutil, joblib 1.5.3,
pillow 11.3.0, shapely 2.0.7, netCDF4 1.7.2

**⚠️ MISSING (must install):**
- `dask[dataframe]` — required for out-of-core processing of 29 GB CSV files
  - Install: `pip install "dask[dataframe]"` (pulls dask + toolz + fsspec + cloudpickle [already installed])
  - Alternative: pandas chunked reading (viable but more complex, less maintainable)

**NOT available (acceptable):**
- `geopandas` — shapefile masking can use rasterio.features + shapely directly
- `dask.distributed` — not strictly needed; `dask.threaded` or `synchronous` schedulers suffice

### 2.3 Prediction CSV Structure

```
Columns: [unnamed_index], timestamp, timestamp.1, latitude, longitude,
         [features...], name, sap_velocity_cnn_lstm, sap_velocity_ensemble
```

- Primary target column: `sap_velocity_cnn_lstm` (also `sap_velocity_ensemble`)
- Column should be configurable via CLI `--value-column`
- Files: 29 GB (multi-day), 9–11 GB (single day), 41–135 MB (smaller subsets)
- Grid resolution: 0.1° (inferred from `Grid_65.0_100.0` naming pattern)
- Multiple timestamps per file; daytime filter uses `sw_in > 15`

### 2.4 Python 3.9 Compatibility

- ✅ No `X | Y` union syntax found in existing code
- ✅ No `match` statements found
- ⚠️ Will add `from __future__ import annotations` to all new/modified files as precaution

### 2.5 SLURM Template (from job_aoa_train_zen4.sh)

```bash
#SBATCH --partition=zen4
#SBATCH --cpus-per-task=128
#SBATCH --mem=240G
```

For visualization: increase to `--mem=480G` (29 GB CSV → ~50 GB in memory + raster buffers).

## 3. Architecture Decision

### Approach: Extend existing module (not new file)

**Rationale:** The existing `prediction_visualization.py` has well-tested rasterization logic
(`create_sap_velocity_tif`, `_create_tiled_output`, `get_valid_block_size`). Creating a new
file would duplicate this. Instead:

1. **Refactor** existing functions to accept column name as parameter
2. **Add** new functions for per-timestamp processing and composite generation
3. **Add** `argparse` CLI interface in a new `main()` function
4. **Replace** hardcoded `__main__` block with CLI dispatch

### Processing Pipeline

```
CLI args → discover_timestamps() → [per-timestamp loop]:
  → filter_daytime() → groupby(lat, lon).mean()
  → rasterize_to_geotiff() → plot_map() (optional)
→ compute_composites() → write composite GeoTIFFs
```

### Parallelization Strategy

- **Dask** for CSV reading (out-of-core chunked I/O)
- **Dask scheduler:** `synchronous` for single-timestamp processing (memory-safe),
  `threads` for I/O-bound GeoTIFF writing
- **No distributed cluster** needed — single zen4 node with 128 cores and 480 GB RAM
- **ThreadPoolExecutor** for parallel tile writing (existing pattern)
- **Composites:** Stack GeoTIFFs with rasterio, compute pixel-wise statistics with numpy

### Memory Budget (480 GB available)

| Component | Estimated Memory |
|-----------|-----------------|
| Dask CSV partition (256 MB blocksize) | ~2 GB working set |
| Per-timestamp aggregated DataFrame | ~500 MB (worst case) |
| Global raster grid (0.1°, 1380×3600) | ~19 MB (float32) |
| Composite stack (6 stats × grid) | ~114 MB |
| Overhead + pandas/numpy buffers | ~5 GB |
| **Total estimated peak** | **~8 GB** |

Highly conservative — 480 GB provides >50× headroom.

### Temporal Aggregation Approach

**Two-pass strategy:**
1. **Pass 1:** Generate per-timestamp GeoTIFFs (individual maps)
2. **Pass 2:** Stack GeoTIFFs → compute composites (mean, median, max, min, std, count)

**Rationale:** Two-pass is safer (per-timestamp TIFFs are checkpointable),
enables incremental processing, and keeps memory bounded.

## 4. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Dask not installed | HIGH | Install `dask[dataframe]` before running; fallback to pandas chunks |
| 29 GB CSV I/O bottleneck on /scratch | MEDIUM | Use large Dask blocksize (512 MB), minimize re-reads |
| Empty `sap_velocity_cnn_lstm` column | HIGH | Detect and skip empty timestamps; log warnings |
| Memory spike during composite stacking | LOW | Stack only rasterio bands (19 MB each), not full DataFrames |
| Job exceeds 48h walltime | LOW | 51 GB total, ~10 min per timestamp; estimate <6h for full pipeline |
| Windows line endings in SLURM script | LOW | Self-fixing `sed -i 's/\r$//'` in job script header |

## 5. Task Breakdown

| ID | Title | Dependencies | Description |
|----|-------|-------------|-------------|
| install-dask | Install dask in HPC venv | — | `pip install "dask[dataframe]"` in shared venv |
| refactor-core | Refactor core rasterization | — | Make `create_sap_velocity_tif` accept column name param, extract reusable functions |
| add-timestamp-discovery | Add timestamp discovery | refactor-core | Function to scan CSV and return unique timestamps |
| add-per-timestamp-maps | Per-timestamp map generation | add-timestamp-discovery | Loop over timestamps, generate individual GeoTIFFs |
| add-composites | Composite map generation | add-per-timestamp-maps | Stack GeoTIFFs, compute mean/median/max/min/std/count |
| add-cartopy-plotting | Cartopy-based map plotting | refactor-core | Replace plain matplotlib with Cartopy coastlines/borders |
| add-cli | Add argparse CLI | add-per-timestamp-maps, add-composites | Input dir, output dir, run_id, value column, stats, etc. |
| create-slurm-job | Create job_map_viz.sh | add-cli | SLURM script mirroring AOA zen4 templates |
| write-tests | Write unit tests | add-cli | Test CLI parsing, gridding, aggregation, composites |
| code-review | Run code reviewer | write-tests | Automated review of all changes |
| security-review | Run security reviewer | create-slurm-job | Review SLURM script for security |
| smoke-test | Local smoke test | write-tests | Test with 1000-row sample CSV |
| commit | Git commit | code-review, security-review, smoke-test | Stage and commit all changes |

## 6. File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/make_prediction/prediction_visualization.py` | MODIFY | Add CLI, batch processing, composites, Cartopy plotting |
| `job_map_viz.sh` | CREATE | SLURM job script for zen4 partition |
| `src/make_prediction/tests/test_prediction_visualization.py` | CREATE | Unit tests for new functions |

## 7. Deviations and Notes

- Will be updated as implementation progresses
