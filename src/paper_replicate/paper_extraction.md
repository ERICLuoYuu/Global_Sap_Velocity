# Phase 1: Paper Extraction — Loritz et al. (2024)

**Title:** Generalizing Tree-Level Sap Flow Across the European Continent
**Journal:** Geophysical Research Letters, 51, e2023GL107350
**DOI:** 10.1029/2023GL107350
**Authors:** Loritz R, Wu CH, Klotz D, Gauch M, Kratzert F, Bassiouni M

---

## A. Data

### Source
- SAPFLUXNET database v0.1.5 (Poyatos et al., 2021)
- European subset only

### Sites
- **64 forest stands** out of 202 globally
- **738 individual trees**
- Selected for European focus: "high density of measurements and strong overlap
  of tree genera found within this region"

### Genera (6, with >20 individual tree measurements each)
| Genus | N trees |
|-------|---------|
| Pinus | 282 |
| Picea | 159 |
| Quercus | 144 |
| Fagus | 94 |
| Larix | 30 |
| Pseudotsuga | 29 |

### Forest Types (6, IGBP classification)
| Type | N stands |
|------|----------|
| ENF (Evergreen Needleleaf) | 34 |
| MF (Mixed Forest) | 11 |
| DBF (Deciduous Broadleaf) | 8 |
| DNF (Deciduous Needleleaf) | 5 |
| EBF (Evergreen Broadleaf) | 4 |
| SAV (Savannas) | 2 |

### Target Variable
- **Sap flow** in cm3 h-1 (NOT sap flux density, NOT sap velocity)
- **Tree-level** (individual plant), NOT site-averaged
- **Standardized** during training: z-score using mean and std of non-zero values

### Temporal Structure
- Hourly resolution
- Growing season only: April-September (months 4-9)
- **Daytime only: hours 7-21** (from code: hours where hour < 22 and hour > 6)
- Winter excluded (Oct-Mar)
- Each tree's seasonal time series treated as separate sample
- Total: **2,279 individual growing-season time series** ("years")

### Quality Control
- Paper: not explicitly described beyond SAPFLUXNET's own QC
- Code reveals:
  - Sap flow < 0 replaced with 0
  - NaN values filled with 0
  - Zero sap flow excluded from training loss (min_obs filter)
  - Resampled to hourly mean

### Gap Handling
- NaN filled with 0 (from code)
- No explicit gap-filling algorithm mentioned

---

## B. Input Features

### CRITICAL FINDING: Code has MORE features than paper describes

The paper states "six dynamic features" and "six static features", but the
code reveals **8 dynamic features** (2 additional LAI features from Earth
Observation data not mentioned in the paper text).

### Dynamic Features (8 total, hourly)
| Variable | Unit | Source | In paper? |
|----------|------|--------|-----------|
| ta (air temperature) | C | SAPFLUXNET env_data | Yes |
| rh (relative humidity) | % | SAPFLUXNET env_data | Yes |
| vpd (vapor pressure deficit) | kPa | SAPFLUXNET env_data | Yes |
| sw_in (shortwave radiation) | W m-2 | SAPFLUXNET env_data | Yes |
| precip (precipitation) | mm | SAPFLUXNET env_data | Yes |
| ws (wind speed) | m s-1 | SAPFLUXNET env_data | Yes |
| leaf_area_index_high_vegetation | - | eo_data (ERA5?) | **NO** |
| leaf_area_index_low_vegetation | - | eo_data (ERA5?) | **NO** |

**Source of environmental data:** SAPFLUXNET on-site measurements (NOT ERA5)
**Source of LAI:** Earth Observation (eo_data/ directory) - likely ERA5-Land or
Copernicus, as the variable names match ERA5-Land naming conventions

### Static Features (6 total)
| Variable | Unit | Category |
|----------|------|----------|
| si_elev (elevation) | m | Site |
| si_mat (mean annual temperature) | C | Site |
| si_map (mean annual precipitation) | mm | Site |
| si_igbp (IGBP forest type) | categorical | Site |
| pl_dbh (diameter at breast height) | cm | Tree |
| pl_genus (tree genus) | categorical | Tree |

### Encoding
- **One-hot encoding** for si_igbp (6 classes: DBF, DNF, EBF, ENF, MF, SAV)
- **One-hot encoding** for pl_genus (6 classes: Fagus, Larix, Picea, Pinus, Pseudotsuga, Quercus)

### Total Feature Count
- 6 meteorological + 2 LAI + 4 numerical static + 6 IGBP one-hot + 6 genus one-hot = **24 features**

### Preprocessing
- All numerical features (non-one-hot) standardized: (x - mean) / std
- Mean and std computed over TRAINING data only
- One-hot features left as 0/1

### Missing Feature Handling
- NaN filled with 0

---

## C. LSTM Architecture

### From paper (Section 2.2.1)
| Parameter | Value |
|-----------|-------|
| Hidden layers | 1 |
| Hidden neurons | 256 |
| Sequence length | 24 hours |
| Dropout rate | 0.4 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Batch size | 64 |
| Epochs | 20 |

Based on Mai et al. (2022) with reduced sequence length and epochs.
"LSTM performance was relatively stable with a wide range of hyperparameters."

### From code (additional details NOT in paper)
| Parameter | Value | Source |
|-----------|-------|--------|
| Forget gate bias init | 3.0 | Cell 15 |
| Gradient clipping | Yes, max_norm=1 | network_params |
| Weight decay | 0.001 | optimizer setup |
| LR scheduler | LambdaLR(lambda=1.0) (no decay) | Cell 16 |
| Input noise | 5% Gaussian noise on both X and y | Cell 18 |
| Loss function | MSELoss(reduction='mean') | Cell 16 |
| Target standardization | z-score (non-zero sap flow) | Cell 7 |
| Early stopping | NOT used (trains all 20 epochs) | Cell 18 |
| Model saves | Every epoch | Cell 18 |

### Architecture (from code, Cell 12)
```
Input (batch, seq_len=24, features=24)
  -> nn.LSTM(input_size=24, hidden_size=256, num_layers=1, batch_first=True)
  -> take last timestep output[:, -1, :]
  -> Dropout(0.4)
  -> nn.Linear(256, 1)
  -> Output: single sap flow prediction
```

- Standard LSTM, NOT Entity-Aware LSTM (EA-LSTM)
- Static features fed as constant values across all 24 timesteps in the sequence
  (i.e., static features are repeated for each hourly timestep)
- No embedding layers
- No attention mechanism
- No batch normalization
- h0 and c0 initialized to zeros

---

## D. Training Details

### Optimization
- Loss: MSE (on z-score normalized target)
- Optimizer: Adam(lr=0.0005, weight_decay=0.001)
- LR schedule: None (LambdaLR with lambda=1.0)
- Gradient clipping: max_norm=1
- Input noise: 5% Gaussian added to X and y during training
- Forget gate bias: initialized to 3.0 (helps learn long-term dependencies)

### Training Loop
- 20 epochs, no early stopping
- Model saved after every epoch
- Zero sap flow samples excluded from loss computation
- Validation loss computed but not used for stopping/selection
- drop_last=True for DataLoader

### Random Seeds
- torch.manual_seed(1) and torch.cuda.manual_seed(1)
- random.seed(seed) where seed varies 1-10 for Monte Carlo runs

### Hardware
- Not specified in paper
- Code designed for GPU (CUDA) with CPU fallback

---

## E. Model Variants

### 1. Gauged-Continental Model
- **Training:** 50% of 2,279 time series (1,140 years) across ALL 64 stands
- **Validation:** 10% (227 years)
- **Testing:** 40% (912 years)
- Split: random shuffle by individual tree-year time series (not by site)
- 10 Monte Carlo random splits (seeds 1-10)
- Tests temporal generalization (unseen time periods at seen stands)

### 2. Ungauged-Continental Model
- **Training:** 33 forest stands (randomly selected)
- **Testing:** remaining 31 stands
- Split stratified by IGBP forest type (proportional representation)
- 10 random subsets
- Tests spatial generalization (unseen locations)

### 3. Single Forest Stand Models
- One LSTM per forest stand
- Trained only on data from that stand (all genera at site)
- Tests if local models beat continental models

### 4. Single Tree Genera Models
- One LSTM per genus (6 total)
- Trained across all stands where that genus exists
- Tests if genus-specific models beat continental models

### 5. Gauged Baseline (statistical)
- Monthly-averaged hourly diurnal sap flow cycle per stand per genus
- Built 10 times on same training data as gauged-continental
- No ML model, pure statistical reference

### 6. Ungauged Baseline (statistical)
- Monthly-averaged hourly diurnal sap flow cycle per genus across Europe
- Built on same 33-stand training data as ungauged-continental

### Total models trained
- 10 gauged-continental + 10 ungauged-continental + 64 single-stand + 6 single-genus + 20 baselines = ~110 models

---

## F. Evaluation

### Metrics
- **Primary:** KGE (Kling-Gupta Efficiency, Gupta et al. 2009)
  - Components: Pearson correlation (r), bias ratio (alpha), variability ratio (beta)
- **Secondary:** NSE (Nash-Sutcliffe Efficiency), MAE (Mean Absolute Error)

### Computation
- Per-tree evaluation, then aggregated per-genus and overall
- KGE formula: standard KGE (not modified KGE)
- Uses `hydroeval` Python package for KGE/NSE computation

---

## G. Key Results

### Headline Numbers
| Model | KGE (mean +/- std) |
|-------|--------------------|
| Gauged-continental | **0.77 +/- 0.04** |
| Gauged baseline | 0.64 +/- 0.05 |
| Ungauged-continental | **0.52 +/- 0.16** |
| Ungauged baseline | -0.11 +/- 0.15 |
| Ungauged-continental (70% training) | ~0.65 +/- 0.08 |

### Per-Genus (Gauged-Continental)
| Genus | KGE | Note |
|-------|-----|------|
| Quercus | High (not exact) | Consistent high performance |
| Fagus | High (not exact) | Consistent high performance |
| Pseudotsuga | 0.76 +/- 0.07 | Despite small dataset (80 years) |
| Picea | 0.55 +/- 0.06 | Weakest performer |
| Pinus | Not reported separately | |
| Larix | Not reported separately | |

### Per-Genus Baseline (Gauged)
| Genus | KGE |
|-------|-----|
| Pseudotsuga | 0.34 +/- 0.07 |
| Picea | 0.28 +/- 0.03 |

### Key Findings
1. Continental models ALWAYS matched or outperformed specialized models
2. No single-stand model exceeded gauged-continental performance
3. Single-genus models more sensitive to data amount
4. Forest type (IGBP) more important than specific genus for model performance
5. Continental models could predict sap flow for genera not in training (KGE 0.2-0.4)
6. Adding total stand density/basal area: marginal improvement (KGE 0.79 -> 0.81)
7. Sensor type information did not improve models

---

## H. Data Availability Statement

"The data on which this article is based are available in Poyatos et al. (2021).
All python codes to run the models are available in Loritz and Wu (2023)."

- Data: SAPFLUXNET v0.1.5
- Code: Zenodo DOI 10.5281/zenodo.10118262
