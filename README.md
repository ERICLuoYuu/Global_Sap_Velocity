# Predicting Global Patterns of Tree Water Use
### A Machine Learning Approach Using SAPFLUXNET

**Authors:** Yu Luo, Mana Gharun
**Affiliation:** Biosphere-Atmosphere Interaction Research Group, Institute of Landscape Ecology, Department of Geosciences, University of Münster, Germany

---

## Overview

This project develops a machine learning pipeline to predict global tree sap velocity (a proxy for tree water use) at daily temporal resolution and global spatial scale. By combining sap flow observations from the SAPFLUXNET database with ERA5-Land climate reanalysis data and satellite-derived spatial features, trained models are used to generate spatially and temporally continuous maps of tree water use worldwide.

**Why it matters:**
- Understanding global tree water use is critical for predicting forest resilience under climate change
- Quantifying ecosystem responses to environmental shifts
- Assessing the potential of forest restoration at global scale

## Objectives

1. Produce a daily, global time-series map of tree water use (sap velocity)
2. Reveal spatial and temporal patterns of how environmental predictors and their interactions drive tree water use
3. Identify systematic variations of tree water use across forest types and biomes
4. Quantify global forest water demand to guide forest restoration strategies

---

## Data Sources

### SAPFLUXNET Database
- **Description:** Global network of standardized sap flow measurements
- **Coverage:** 202 sites spanning multiple biomes and climate zones worldwide
- **Temporal resolution:** Half-hourly, hourly, or two-hourly (site-dependent)
- **Temporal range:** 1995–2018
- **Reference:** [sapfluxnet.pfeiffer.es](https://sapfluxnet.pfeiffer.es)

### ERA5-Land Reanalysis (ECMWF)
- **Description:** ECMWF's global land surface climate reanalysis dataset
- **Spatial resolution:** 0.1° grid (~9 km at the equator)
- **Temporal range:** 1950–2025, hourly data
- **Access:** [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu)

### Auxiliary Spatial Data
- Biome classification and plant functional type (PFT) maps
- Leaf Area Index (LAI) from GLOBMAP
- Canopy height from global datasets
- Digital Elevation Model (DEM)
- Soil properties from SoilGrids
- Plant trait data (iNaturalist)
- Stand age datasets

---

## Model Architecture

The pipeline uses time-windowed input sequences: **8 consecutive daily timesteps** of environmental variables are used to predict sap velocity at the **9th timestep**.

### Trained Models
| Model | Type | Notes |
|---|---|---|
| XGBoost | Gradient Boosted Trees | Primary model, fast inference |
| Random Forest | Ensemble Tree | Baseline comparison |
| CNN-LSTM | Deep Learning | Captures temporal dependencies |
| Feedforward ANN | Neural Network | Various architectures tested |
| SVM | Support Vector Machine | Baseline comparison |

### Cross-Validation Strategies
- **Spatial Block CV** — prevents spatial data leakage between folds
- **Temporal CV** — respects time ordering of observations
- **Group K-Fold** — groups by site to test generalization to unseen locations

---

## Project Structure

```
global-sap-velocity/
├── src/                                # Core source code
│   ├── Analyzers/                      # Data loading, cleaning, and QC
│   │   ├── sap_analyzer.py             # SapFlowAnalyzer class
│   │   ├── env_analyzer.py             # EnvironmentalAnalyzer class
│   │   └── mannual_removal_processor.py# Manual QC record handling
│   ├── Extractors/                     # Feature extraction from remote sources
│   │   ├── extract_climatedata_gee.py  # ERA5 extraction via Google Earth Engine
│   │   ├── spatial_extractor_gee.py    # Spatial feature extraction (GEE)
│   │   ├── spatial_features_extractor.py
│   │   ├── extract_lai.py              # LAI extraction
│   │   ├── extract_canopyheight.py
│   │   ├── extract_soilgrid.py
│   │   └── extract_pft.py
│   ├── hyperparameter_optimization/    # Model training and tuning
│   │   ├── hyper_tuner.py              # Base optimizer (GridSearch / RandomSearch)
│   │   ├── blockcv.py                  # Spatial block cross-validation
│   │   ├── timeseries_processor.py     # Time-windowed feature generation
│   │   ├── test_hyperparameter_tuning_ann_blockcv.py
│   │   ├── test_hyperparameter_tuning_ann_spatial.py
│   │   └── test_hyperparameter_tuning_ann_temporal_seg.py
│   ├── make_prediction/                # Inference pipeline
│   │   ├── config.py                   # Global configuration and paths
│   │   ├── predict_sap_velocity_sequantial.py  # Main prediction script
│   │   └── process_era5land_gee_opt_fix.py     # ERA5 preprocessing
│   ├── models/                         # Model class implementations
│   ├── features/                       # Feature engineering utilities
│   ├── cross_validation/               # CV strategy implementations
│   ├── visualization/                  # Plotting and mapping
│   ├── derive_climate_data/            # Derived climate variable computation
│   ├── tools.py                        # Shared utilities (merging, timezone, etc.)
│   └── sapf_plotter.py                 # Sap flow visualization
│
├── notebooks/                          # Analysis and exploration scripts
│   ├── initial_exploration.py          # Data loading and QC workflow
│   ├── merge_gap_filled_hourly.py      # Hourly gap-filling and aggregation
│   ├── add_era5_data.py                # Attaching ERA5 variables to site data
│   ├── download_DEM.py                 # Terrain data download
│   ├── process_extracted_era5.py       # ERA5 post-processing
│   └── VPD_mismatch_analysis.py        # VPD quality analysis
│
├── data/
│   ├── raw/                            # Immutable input data
│   │   ├── grided/                     # Gridded ERA5-Land and spatial datasets
│   │   ├── era5land_site_data/         # ERA5 extracted at SAPFLUXNET sites
│   │   └── extracted_data/             # Other extracted site-level data
│   └── processed/                      # Cleaned and merged datasets
│       ├── sap/                        # Processed sap flow data
│       └── env/                        # Processed environmental data
│
├── models/                             # Trained model artifacts
│   ├── xgb/                            # XGBoost model files (.joblib, .json)
│   ├── CNN-LSTM/                       # Deep learning checkpoints (.h5)
│   └── automl/                         # AutoML results
│
├── outputs/                            # Generated results
│   ├── predictions/                    # Model prediction outputs
│   ├── figures/                        # Visualization outputs
│   ├── hyper_tuning/                   # Hyperparameter tuning logs and results
│   ├── scalers/                        # Fitted preprocessing scalers (.pkl)
│   └── statistics/                     # Evaluation metrics and summaries
│
├── plots/                              # Additional plot outputs by model type
├── tests/                              # Unit tests
├── docs/                               # Extended documentation
├── environment.yml                     # Conda environment specification
├── requirements.txt                    # pip dependencies
└── setup.py                            # Package installation
```

---

## Setup

### Prerequisites
- Python 3.9
- CUDA 11.8 (for GPU-accelerated training)
- Google Earth Engine account (for data extraction)
- Copernicus CDS API key (for ERA5-Land download)

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate mvenv
pip install -r requirements.txt
```

### Option 2: pip + virtualenv

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Google Earth Engine Authentication

```bash
earthengine authenticate
```

---

## Workflow

### 1. Data Preparation

Load, clean, and quality-control sap flow and environmental data:

```bash
python notebooks/initial_exploration.py
```

Aggregate hourly sap flow to daily resolution and gap-fill:

```bash
python notebooks/merge_gap_filled_hourly.py
```

### 2. Feature Extraction

Extract ERA5-Land climate variables at SAPFLUXNET site locations:

```bash
python src/Extractors/extract_climatedata_gee.py
```

Extract spatial features (LAI, canopy height, biome, soil properties):

```bash
python src/Extractors/spatial_features_extractor.py
```

Attach ERA5 data to the merged dataset:

```bash
python notebooks/add_era5_data.py
```

### 3. Model Training & Hyperparameter Optimization

Train and tune models using spatial block cross-validation:

```bash
# ANN with spatial block CV
python src/hyperparameter_optimization/test_hyperparameter_tuning_ann_blockcv.py

# ANN with temporal segment CV
python src/hyperparameter_optimization/test_hyperparameter_tuning_ann_temporal_seg.py

# ML models (XGBoost, RF, SVM)
python src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial.py
```

### 4. Global Prediction

Run inference over global ERA5-Land grids using a trained model:

```bash
python src/make_prediction/predict_sap_velocity_sequantial.py \
    --model xgb \
    --start_date 2010-01-01 \
    --end_date 2010-12-31
```

---

## Key Configuration

Edit [src/make_prediction/config.py](src/make_prediction/config.py) to adjust:

| Parameter | Description | Default |
|---|---|---|
| `GEE_PROJECT` | Google Earth Engine project ID | `'ee-yuluo-2'` |
| `DEFAULT_MODEL` | Path to model file | XGBoost default |
| `PREDICTION_WINDOW_SIZE` | Number of input timesteps | `8` |
| `DEFAULT_LAT_MIN/MAX` | Latitude bounds | `-60` to `78` |
| `DEFAULT_LON_MIN/MAX` | Longitude bounds | `-180` to `180` |
| `DASK_MEMORY_LIMIT` | Dask memory cap | `"8GB"` |

---

## Explanatory Variables

Climate variables derived from ERA5-Land:
- Air temperature (`ta`)
- Precipitation (`precip`)
- Wind speed (`ws`)
- Vapor Pressure Deficit (`vpd`)
- Shortwave incoming radiation (`sw_in`)
- Soil moisture

Spatial / ecological variables:
- Biome classification (16 categories)
- Plant Functional Type (PFT)
- Leaf Area Index (LAI)
- Canopy height
- Terrain (elevation, slope)
- Soil properties (texture, depth)
- Plant traits

Time-derived features:
- Day of year (cyclical encoding)
- Season indicators

---

## Outputs

- **Global sap velocity maps:** Daily rasters at 0.1° resolution
- **Performance metrics:** R², RMSE, MAE per model and cross-validation fold
- **Feature importance:** Variable contribution ranking (XGBoost, RF)
- **Validation plots:** Predicted vs. observed, error distributions, spatial residual maps

---

## Citation

If you use this code or data pipeline in your research, please cite:

> Yu Luo, Mana Gharun (2024). *Predicting Global Patterns of Tree Water Use: A Machine Learning Approach Using SAPFLUXNET.* Biosphere-Atmosphere Interaction Research Group, University of Münster, Germany.

---

## License

This project is developed for academic research purposes. Please contact the authors before reusing or redistributing any part of this codebase.

---

## Contact

**Yu Luo** — Institute of Landscape Ecology, University of Münster
**Mana Gharun** — Biosphere-Atmosphere Interaction Research Group, University of Münster
