# Forward Feature Selection — Deviations Log

## Deviation 1: Precomputed CV splits for PFT stratification
- **Phase**: Phase 4 (Review)
- **Planned**: Use `StratifiedGroupKFold` directly as `cv` in SFS
- **Actual**: Precompute splits via `list(cv_splitter.split(x_all, pfts_encoded, groups))` and pass the list to SFS
- **Reason**: CRITICAL BUG — mlxtend passes regression target `y` to `cv.split()`, but StratifiedGroupKFold needs PFT categories for stratification. Precomputed splits decouple stratification (by PFT) from model training (on sap velocity).
- **Impact on acceptance criteria**: None changed — same CV behavior as training script
- **Approved**: auto (bug fix, no scope change)

## Deviation 2: Skip sites with missing columns instead of using available_cols
- **Phase**: Phase 4 (Review, round 2)
- **Planned**: Use `available_cols` to gracefully handle sites with missing features
- **Actual**: Skip sites with ANY missing column, matching the training script's strict behavior
- **Reason**: HIGH BUG — sites with different column counts produce matrices of different widths, crashing `np.vstack`. The training script skips such sites (L459-462). All surviving sites must have identical column sets.
- **Impact on acceptance criteria**: None changed — sites missing features are excluded (same as training script)
- **Approved**: auto (bug fix, no scope change)

## Deviation 3: Replace fake pooled scorers with true post-hoc pooled R2/RMSE
- **Phase**: Phase 4 (Review, round 3)
- **Planned**: 4 scoring modes (mean_r2, pooled_r2, neg_rmse, pooled_rmse) as separate SLURM jobs
- **Actual**: 2 scoring modes (mean_r2, neg_rmse) for SFS selection. True pooled R2/RMSE computed post-hoc via cross_val_predict for every step. Deleted job_ffs_pooled_r2.sh and job_ffs_pooled_rmse.sh.
- **Reason**: make_scorer + cross_val_score evaluates per fold — "pooled" scorers were identical to mean-fold. True pooled metrics require concatenating all OOF predictions before scoring, which needs cross_val_predict outside the SFS loop.
- **Impact on acceptance criteria**: 4 SLURM jobs → 2 jobs. Each job now produces mean-fold + pooled metrics.
- **Approved**: user-requested (explicit instruction to compute true pooled R2)

## Deviation 4: Lazy imports for heavy dependencies
- **Phase**: Phase 5 (Test expansion)
- **Planned**: Top-level imports of xgboost, sklearn, training script helpers
- **Actual**: Heavy imports moved inside functions that need them (run_forward_selection, _compute_pooled_metrics, load_and_cache_features). Light functions (_extract_results, _save_results, _add_rh_rolling_and_lag, load_cache) importable without heavy deps.
- **Reason**: Importing data_loader.py triggered the full training script (cartopy, tensorflow, shap ~1GB), making local tests fail. Lazy imports enable the full 90-test suite to run locally without HPC packages.
- **Impact on acceptance criteria**: None changed — same runtime behavior on HPC
- **Approved**: auto (test infrastructure, no functional change)

---

## Status Summary
- **Total deviations**: 4
- **Unapproved**: 0
- **Phase**: Phase 5 (Test expansion) COMPLETE — 90/90 tests pass
