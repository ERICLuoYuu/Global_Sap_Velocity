# Phase 0: Code & Data Search Results

## Published Code

**Found on Zenodo:** DOI 10.5281/zenodo.10118262 (Loritz & Wu, 2023)
- License: CC-BY 4.0
- Published: 2023-11-13

### Files
| File | Size | Description |
|------|------|-------------|
| Baseline model.ipynb | 15.7 kB | Monthly-averaged diurnal cycle baseline |
| Gauged continental model.ipynb | 37.7 kB | LSTM trained on all 64 stands, tested on unseen time |
| Ungauged continental model.ipynb | 37.8 kB | LSTM trained on 33 stands, tested on 31 unseen stands |
| data.zip | 390.5 MB | Preprocessed SAPFLUXNET European subset |

### data.zip structure (inferred from code)
```
data/
  plant_md/     {si_code}_plant_md.csv   (pl_code, pl_dbh, pl_species)
  site_md/      {si_code}_site_md.csv    (si_elev, si_mat, si_map, si_igbp)
  env_data/     {si_code}_env_data.csv   (TIMESTAMP, ta, precip, rh, sw_in, ws, vpd)
  eo_data/      {si_code}_eo_data.csv    (date, LAI_high_veg, LAI_low_veg)
  sapf_data/    {si_code}_sapf_data.csv  (TIMESTAMP, per-plant sap flow columns)
```

### Code quality assessment
- Language: Python, PyTorch
- Framework: Custom PyTorch LSTM (does NOT use NeuralHydrology package)
- Structure: 3 Jupyter notebooks with inline code, no modular library
- Reusability: HIGH - code is straightforward, well-commented, self-contained
- Key concern: Heavy use of dynamic variable creation, but logic is clear
- Can run in: Google Colab or local with path adjustment

### Recommendation
Use their code as reference implementation and their data.zip directly for
faithful replication. Rewrite into modular Python for our pipeline. Download
data.zip (390 MB) to get the exact preprocessed European subset. Their data
includes an eo_data/ directory with LAI features NOT mentioned in the paper.

## GitHub Search Results

| Query | Result |
|-------|--------|
| Loritz + sap flow + LSTM on GitHub | No standalone repo found |
| NeuralHydrology package | Exists but NOT used by this paper |
| Co-author repos (Kratzert, Klotz, Gauch) | NeuralHydrology group, no sap flow code |

## SAPFLUXNET Data

- Paper uses: SAPFLUXNET v0.1.5
- We have: SAPFLUXNET v0.1.5 at data/raw/0.1.5/0.1.5/csv/sapwood/
- Our total sites: 165 (they report 202 stands globally)
- Our European sites: 65 (they used 64)
- Overlap is near-complete

## Notebooks Downloaded To
```
src/paper_replicate/
  baseline_model.ipynb
  gauged_continental_model.ipynb
  ungauged_continental_model.ipynb
```
