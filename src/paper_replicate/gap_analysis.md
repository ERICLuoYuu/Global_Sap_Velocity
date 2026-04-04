# Phase 2: Gap Analysis

## A. Fully Specified (can implement directly)

1. LSTM architecture: 1 layer, 256 hidden, sequence=24h, dropout=0.4
2. Optimizer: Adam, lr=0.0005, weight_decay=0.001
3. Loss: MSE on z-score normalized target
4. Batch size: 64, epochs: 20
5. Forget gate bias init: 3.0
6. Gradient clipping: max_norm=1
7. Input noise: 5% Gaussian on X and y during training
8. Data split: 50/10/40 (train/val/test) for gauged, 33/31 stands for ungauged
9. Growing season: April-September, daytime hours 7-21
10. Feature list: 6 meteo + 2 LAI + 4 static numerical + 12 one-hot = 24 total
11. Standardization: z-score per feature over training data, one-hot untouched
12. Target: z-score using non-zero sap flow mean/std
13. Evaluation: KGE (hydroeval package), NSE, MAE
14. Monte Carlo: 10 random splits, seeds 1-10
15. Zero sap flow: excluded from training loss

## B. Underspecified in Paper (resolved from code)

| Detail | Paper says | Code reveals |
|--------|-----------|-------------|
| LAI features | Not mentioned | 2 LAI features from eo_data/ used |
| Loss function | Not stated | MSELoss(reduction='mean') |
| Weight decay | Not stated | 0.001 |
| Gradient clipping | Not stated | max_norm=1 |
| Input noise | Not stated | 5% Gaussian on X and y |
| Forget gate bias | Not stated | Initialized to 3.0 |
| LR schedule | Not stated | None (LambdaLR lambda=1.0) |
| Early stopping | Not stated | NOT used, trains all 20 epochs |
| Model selection | Not stated | Saves every epoch (unclear which used) |
| Target normalization | Not stated | z-score of non-zero sap flow values |
| Daytime filtering | Not stated | Hours 7-21 only |
| NaN handling | Not stated | Filled with 0 |
| Negative sap flow | Not stated | Replaced with 0 |
| Validation set | Not clearly stated | 10% of data used for validation |
| Random seeds | Not stated | torch seed=1, split seeds 1-10 |

## C. Still Ambiguous (even after reading code)

1. **Which epoch's model is used for evaluation?** Code saves all 20 models but
   doesn't show which one is loaded for testing. Likely the last epoch (epoch 19),
   but could be best-validation-loss epoch.

2. **LAI data source:** eo_data/ directory contains LAI - variable names match
   ERA5-Land naming but source not confirmed. Could be Copernicus Climate Data
   Store or similar. Need to check data.zip to confirm.

3. **Exact site list:** The 64 site codes are not listed in the paper or supplement.
   We need data.zip to get the exact list. Our 65 EU sites may include one extra
   or they may have excluded one of ours.

4. **Single-stand and single-genus model details:** Code for these variants is NOT
   in the Zenodo archive (only Baseline, Gauged, Ungauged notebooks). We must
   infer implementation from the gauged model code.

5. **Ungauged stratification details:** Code shows IGBP-stratified random split of
   stands, but exact implementation of "proportional representation" constraint
   may differ from our interpretation.

6. **How models are ensembled:** Paper shows uncertainty bands from 10 models but
   doesn't clarify if final KGE is computed per-model-then-averaged, or
   predictions-averaged-then-KGE.

## D. Adaptation Needs for Our Setup

### Data
- We have SAPFLUXNET v0.1.5 - same version, near-complete overlap (65 vs 64 EU sites)
- We need the LAI/eo_data - must either download data.zip or source LAI ourselves
- Our existing data at data/raw/0.1.5/0.1.5/csv/sapwood/ has env_data, sapf_data,
  plant_md, site_md but likely NOT eo_data (need to verify)

### Compute
- PyTorch available on Palma2 HPC
- GPU needed (LSTM training with 2279 time series, 24h sequences)
- Each model trains 20 epochs - relatively fast
- 10 Monte Carlo runs per variant = manageable

### Framework
- Paper uses PyTorch (confirmed from code)
- We should use PyTorch (we have TensorFlow in our main pipeline but PyTorch
  matches their code exactly)
- hydroeval package needed for KGE computation
