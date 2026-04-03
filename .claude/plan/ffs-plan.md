# Forward Feature Selection — Plan

See approved plan: `C:\Users\Ab Leo\.claude\plans\mutable-herding-shell.md`

## Acceptance Criteria

### Local (verified — 90/90 tests pass)
- [x] `feature_registry.py` lists all ~107 candidates + 6 mandatory, PFT grouped, rh stats included
- [x] `data_loader.py` loads CSVs, computes all features (incl. rh rolling/lag), caches to .npz
- [x] `scorers.py` returns correct sklearn scorer strings for mean_r2 and neg_rmse
- [x] `selector.py` extracts results and saves JSON+npz correctly (mock SFS tests)
- [x] `diagnostics.py` loads results, plots curves/rankings, compares scorers, exports JSON
- [x] True pooled R²/RMSE computed post-hoc via cross_val_predict (Deviation 3)
- [x] Precomputed CV splits decouple PFT stratification from regression target (Deviation 1)
- [x] Lazy imports enable full test suite without HPC packages (Deviation 4)

### HPC (pending deployment)
- [ ] `pip install mlxtend` on HPC venv
- [ ] Cache builder SLURM job completes, .npz file valid
- [ ] 2 selection SLURM jobs complete without error (mean_r2 + neg_rmse; Deviation 3: 4→2 jobs)
- [ ] Performance curves plotted for each scoring mode
- [ ] Selected features exported to JSON
- [ ] Comparison across 2 scoring modes generated
