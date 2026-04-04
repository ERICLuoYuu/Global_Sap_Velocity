# Implementation Plan: Loritz et al. (2024) Replication

## 1. Problem Space

Replicate the LSTM-based sap flow prediction from Loritz et al. (2024, GRL)
using their published code + data. Rewrite their 3 Jupyter notebooks into a
clean, modular Python package. Train on Palma2 HPC GPU nodes. Compare our
results against their reported KGE values to validate the replication.

**Scope:** 4 model variants (gauged baseline, gauged LSTM, ungauged LSTM,
ungauged baseline). 10 Monte Carlo runs each for LSTM variants.

**Out of scope:** Single-stand models, single-genus models (code not published,
paper shows they underperform continental models anyway).

**Target numbers to reproduce:**
| Model | Paper KGE |
|-------|-----------|
| Gauged-continental LSTM | 0.77 +/- 0.04 |
| Gauged baseline | 0.64 +/- 0.05 |
| Ungauged-continental LSTM | 0.52 +/- 0.16 |
| Ungauged baseline | -0.11 +/- 0.15 |

## 2. Approach Analysis

### Approach A: Run their notebooks directly on Colab/HPC
- Pros: Fastest path to results, zero rewrite risk
- Cons: Colab-oriented code, no integration with our project, uses dynamic
  variable creation, hard to extend, no tests
- Verdict: REJECTED - doesn't integrate, doesn't teach us

### Approach B: Rewrite into modular Python, use their data.zip (CHOSEN)
- Pros: Clean code, testable, integrates with our project structure, faithful
  to their implementation but maintainable
- Cons: Risk of introducing bugs during rewrite (mitigated by comparing
  intermediate outputs against their code)
- Verdict: CHOSEN - best balance of fidelity and code quality

### Approach C: Reimplement from paper description alone
- Pros: Tests our understanding of the paper
- Cons: Would miss LAI features, noise injection, forget gate bias, etc.
  (underspecified in paper). Would NOT reproduce their results.
- Verdict: REJECTED - too many hidden details only in code

## 3. Failure Modes

| Risk | Severity | Mitigation |
|------|----------|------------|
| Data mismatch (our SAPFLUXNET vs theirs) | HIGH | Use their data.zip directly |
| PyTorch version differences | MEDIUM | Pin versions, test on HPC before full run |
| GPU memory issues | LOW | Dataset is moderate (~4.5M samples), 256 hidden |
| Random seed non-reproducibility across platforms | MEDIUM | Compare intermediate outputs per-seed |
| Numerical precision (CPU vs GPU, CUDA versions) | LOW | Accept small KGE differences (<0.01) |
| Missing eo_data (LAI) in their data.zip | MEDIUM | Verify after download, source from ERA5-Land if needed |
| hydroeval package version differences | LOW | Pin version, verify KGE formula matches |
| SLURM partition unavailability | LOW | Fallback to zen1-zen4 |
| data.zip download fails on HPC | LOW | Download locally, SCP to HPC |

## 4. Dependency Chain

```
Phase 0: Environment Setup
  ├── Download data.zip to HPC
  ├── Install PyTorch + dependencies
  └── Verify GPU access
      │
Phase 1: Core Modules (no dependencies between files)
  ├── config.py (standalone)
  ├── model.py (depends on config)
  ├── dataset.py (depends on config)
  ├── evaluator.py (standalone)
  └── data_loader.py (depends on config)
      │
Phase 2: Tests for Core Modules
  ├── test_model.py (verify forward pass shapes)
  ├── test_dataset.py (verify sequence windowing)
  └── test_evaluator.py (verify KGE against known values)
      │
Phase 3: Training Infrastructure
  ├── trainer.py (depends on model, dataset, config)
  └── SLURM job scripts
      │
Phase 4: Experiment Scripts (sequential)
  ├── run_baseline.py (validates data pipeline, fastest)
  ├── run_gauged.py (main result, depends on trainer)
  └── run_ungauged.py (second result, depends on trainer)
      │
Phase 5: Analysis
  ├── compare_results.py
  └── plots.py
```

## 5. Test Strategy

### Unit Tests (before implementation)
- **test_model.py:**
  - LSTM forward pass: input (64, 24, 24) -> output (64, 1)
  - Forget gate bias initialization = 3.0
  - Dropout applied during training, not eval
  - h0/c0 initialized to zeros

- **test_dataset.py:**
  - Sequence windowing: item[i] returns (24, 24) tensor for features, scalar for target
  - Padding for sequences shorter than seq_len
  - Standardization applied correctly (non-one-hot only)
  - Target standardization uses non-zero sap flow stats
  - Zero-NaN handling

- **test_evaluator.py:**
  - KGE computation matches hydroeval for known inputs
  - Per-genus aggregation logic
  - Edge cases: all-zero predictions, single sample

### Integration Tests
- **Baseline sanity check:** Run gauged baseline on small data subset, verify
  KGE computation end-to-end
- **Single-seed gauged LSTM:** Train 1 model (seed=1) for 2 epochs, verify
  loss decreases and predictions are reasonable
- **Shape consistency:** Full pipeline from data load to KGE output

### Acceptance Gate
- Gauged baseline KGE matches paper (0.64 +/- 0.05) — this validates the
  data pipeline and evaluation code
- Gauged LSTM KGE within 0.05 of paper (0.72-0.82 acceptable range)
- Ungauged LSTM KGE within 0.10 of paper (0.42-0.62 acceptable range,
  higher variance expected)

## 6. Acceptance Criteria

- [ ] data.zip downloaded and extracted on HPC
- [ ] All 24 features loaded correctly from data, matching their code
- [ ] SequenceDataset produces correct tensor shapes (batch, 24, 24)
- [ ] LSTM forward pass produces correct output shape
- [ ] Forget gate bias = 3.0 verified
- [ ] Unit tests pass (model, dataset, evaluator)
- [ ] Gauged baseline: 10 runs complete, KGE ~0.64
- [ ] Gauged LSTM: 10 Monte Carlo runs complete, KGE ~0.77
- [ ] Ungauged LSTM: 10 Monte Carlo runs complete, KGE ~0.52
- [ ] Ungauged baseline: 10 runs complete, KGE ~-0.11
- [ ] Per-genus KGE breakdown computed
- [ ] Results comparison document written
- [ ] Key figures reproduced
- [ ] All code in src/paper_replicate/, outputs in outputs/paper_replicate/

---

## Implementation Steps

### Step 0: Environment Setup on HPC [~30 min]
0a. SSH to Palma2, check GPU partition availability (zen5 > zen4 > zen3 > zen2)
0b. Download data.zip from Zenodo to HPC scratch
0c. Extract and verify contents (64 sites, env_data, eo_data, sapf_data, etc.)
0d. List site codes, compare with paper's 64 European stands
0e. Create conda/venv with: torch, numpy, pandas, hydroeval, scikit-learn, matplotlib
0f. Verify GPU access with a simple torch.cuda test via sbatch

### Step 1: config.py [~15 min]
- Dataclass with all hyperparameters matching their network_params dict
- Path configuration for data and output directories
- Feature lists (dynamic, static, one-hot)
- Plant genera list, IGBP types list

### Step 2: data_loader.py [~1 hr]
- load_all_data(data_dir) -> returns dict of per-tree-year DataFrames
- Replicates their Cell 6 logic: iterate plant_md/, load env, eo, site, plant,
  sapf data per site, split into per-tree per-year DataFrames
- One-hot encode genus and IGBP
- Filter: April-September, hours 7-21, negative sap flow -> 0, NaN -> 0
- compute_standardization_stats(train_data) -> mean, std for features and target
- split_gauged(all_data, seed) -> train, val, test (50/10/40)
- split_ungauged(all_data, seed, igbp_class) -> train_stands, test_stands

### Step 3: dataset.py [~30 min]
- SequenceDataset(df, target, features, seq_len, standardize, mean, std)
- Faithful port of their Cell 11
- Windowed sequences, padding for short sequences

### Step 4: model.py [~20 min]
- RNN_network(input_size, hidden_size, num_layers, dropout)
- Faithful port of their Cell 12
- init_forget_gate_bias(model, value=3.0) utility function
- Forward: LSTM -> last timestep -> dropout -> linear(1)

### Step 5: evaluator.py [~30 min]
- compute_kge(pred, obs) using hydroeval
- compute_nse(pred, obs)
- compute_mae(pred, obs)
- evaluate_per_genus(pred_dict, obs_dict, plant_list)
- evaluate_overall(pred_dict, obs_dict)
- Format results as DataFrame for easy comparison

### Step 6: Tests [~45 min]
- test_model.py: shapes, forget gate, dropout behavior
- test_dataset.py: windowing, standardization, edge cases
- test_evaluator.py: KGE against known values

### Step 7: trainer.py [~45 min]
- train_one_epoch(model, loader, optimizer, criterion, device, noise_std=0.05)
- validate_one_epoch(model, loader, criterion, device)
- train_model(config) -> trained model, training history
- Implements: noise injection, gradient clipping, min_obs filter, model saving
- predict(model, test_loader, device, sapf_mean, sapf_std) -> pred_dict, obs_dict

### Step 8: run_baseline.py [~30 min]
- Gauged baseline: monthly-averaged hourly diurnal cycle per stand per genus
- Ungauged baseline: monthly-averaged hourly diurnal cycle per genus
- 10 seeds, evaluate KGE
- This validates the entire data pipeline and evaluation code

### Step 9: SLURM job scripts [~20 min]
- job_baseline.sh: CPU job for baselines
- job_gauged.sh: GPU job for 10 gauged LSTM runs
- job_ungauged.sh: GPU job for 10 ungauged LSTM runs
- Partition selection based on availability

### Step 10: run_gauged.py [~30 min]
- 10 Monte Carlo runs (seeds 1-10)
- For each: split data, build loaders, train 20 epochs, predict, evaluate
- Save: model checkpoints, predictions, per-genus KGE, overall KGE
- Output to outputs/paper_replicate/gauged/

### Step 11: run_ungauged.py [~30 min]
- 10 Monte Carlo runs with IGBP-stratified stand splits
- Same training loop as gauged but different data splits
- Output to outputs/paper_replicate/ungauged/

### Step 12: compare_results.py [~30 min]
- Load all results
- Compare against paper's reported numbers
- Per-genus breakdown
- Statistical summary (mean, std across Monte Carlo runs)
- Output: results_comparison.md

### Step 13: plots.py [~30 min]
- Reproduce Figure 1: KGE per genus for gauged and ungauged
- Time series examples: observed vs predicted for select trees
- Uncertainty bands from 10 model ensemble

### Milestone Gates

**Gate 1 (after Step 6):** Unit tests pass -> proceed to training
**Gate 2 (after Step 8):** Baseline KGE matches paper -> data pipeline validated
**Gate 3 (after Step 10):** Gauged LSTM KGE ~0.77 -> core replication achieved
**Gate 4 (after Step 13):** All results documented -> replication complete
