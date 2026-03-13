# AOA Implementation Plan

## Decision: Python-native implementation

**Rationale:** Project is 100% Python. R CAST adds rpy2/subprocess complexity, same chunking
needed regardless. Algorithm is well-specified; scipy provides all distance primitives.

## Architecture

```
src/aoa/
  __init__.py        # Public API: compute_aoa()
  dissimilarity.py   # Steps 1-4: standardize, weight, compute DI
  threshold.py       # Steps 5-6: CV-based training DI, threshold
  aoa.py             # Step 7: orchestrator, chunked processing
  plotting.py        # DI/AOA visualization
  tests/
    __init__.py
    test_dissimilarity.py
    test_threshold.py
    test_aoa.py
    test_integration.py
```

## Algorithm (Meyer & Pebesma 2021)

1. Standardize: `X_s = (X - mean_train) / std_train`
2. Weight: `X_sw = w * X_s` (variable importance)
3. d_bar = mean pairwise Euclidean distance in weighted training space
4. DI_k = min_i(dist(new_k, train_i)) / d_bar
5. Training DI: for each training point, DI using only out-of-fold points
6. Threshold = Q75 + 1.5 * IQR of training DI
7. AOA = DI <= threshold

## Integration Points

- **Model:** `outputs/models/xgb/{run_id}/FINAL_xgb_{run_id}.joblib`
- **Scaler:** `outputs/models/xgb/{run_id}/FINAL_scaler_{run_id}_feature.pkl`
- **Config:** `outputs/models/xgb/{run_id}/FINAL_config_{run_id}.json`
  - Contains: feature_names, shap_feature_names, preprocessing info
- **CV folds:** Reconstruct via StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
  - Groups: create_spatial_groups(lat, lon, method='kmeans') from src/tools.py
  - Stratification: PFT encoded
- **Feature importance:** `model.feature_importances_` (XGBoost gain-based)
  - SHAP fallback from saved SHAP values if available
- **Prediction pipeline:** integrate alongside predict_sap_velocity_sequantial.py

## Scalability

- Chunk prediction grid (default 10,000 points/batch)
- Cache d_bar + weighted training data (constant across chunks)
- KD-tree for nearest-neighbor lookup (O(n log n) vs O(n*m))
- joblib Parallel for chunk processing
- Peak memory target: <16 GB

## Phases

1. **Tests first** (RED): all test files in src/aoa/tests/
2. **Implement** (GREEN): dissimilarity.py -> threshold.py -> aoa.py -> plotting.py
3. **Refactor** (IMPROVE): optimize, clean up
4. **Review**: code-reviewer + security-reviewer
5. **Validate**: small tile (10x10 deg, 7 days), then full scale
