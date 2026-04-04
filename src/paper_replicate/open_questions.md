# Phase 2: Open Questions

## Q1. Use their data.zip or our own SAPFLUXNET data?

**Options:**
- (a) Download their data.zip (390 MB) and use it directly for exact replication
- (b) Use our SAPFLUXNET v0.1.5 data with our own preprocessing

**Recommendation:** Option (a) for faithful replication. Their data.zip contains
the preprocessed European subset with the exact 64 sites, the eo_data/ LAI
features, and all preprocessing already done. This eliminates site selection
ambiguity and preprocessing differences as sources of discrepancy.

**Follow-up:** We can later compare "their data + our code" vs "our data + our code"
to isolate the effect of data differences.

## Q2. Scope: Which model variants to replicate?

**Paper has 6 variants:**
1. Gauged-continental LSTM (code available)
2. Ungauged-continental LSTM (code available)
3. Gauged baseline (code available)
4. Ungauged baseline (code available)
5. Single forest stand LSTMs (code NOT available)
6. Single genus LSTMs (code NOT available)

**Recommendation:** Start with variants 1-4 (all have published code). Add 5-6
later if needed, since they require implementation from scratch and the paper's
main finding is that they DON'T outperform continental models.

## Q3. Priority order of implementation?

**Recommendation:**
1. Gauged baseline (simplest, validates data pipeline)
2. Gauged-continental LSTM (main result, code available)
3. Ungauged-continental LSTM (second main result)
4. Ungauged baseline (simple)
5. Single-genus LSTM (if time permits)
6. Single-stand LSTM (if time permits)

## Q4. How close does the replication need to be?

**Options:**
- (a) Exact numerical reproduction (same KGE to 2nd decimal)
- (b) Qualitative reproduction (same trends, similar magnitudes)
- (c) Methodological reproduction (same approach, our implementation)

**Recommendation:** Aim for (a) using their data.zip, since we have their code
and data. If we can't match exactly, document discrepancies and hypothesize why.

## Q5. Use their notebook code directly, or rewrite?

**Options:**
- (a) Run their notebooks as-is (minimal effort, but Colab-oriented)
- (b) Rewrite into modular Python package (more effort, but integrates with our pipeline)
- (c) Hybrid: extract core logic, wrap in our module structure

**Recommendation:** Option (c). Their code is clear but uses non-idiomatic patterns.
Rewrite into clean modular code that follows our project structure, using their
code as the reference to verify correctness.

## Q6. Model selection: which epoch to use?

Their code saves models at every epoch but doesn't show which is loaded for
evaluation. Most likely it's the final epoch (20). Should we:
- (a) Use epoch 20 (last epoch, matching likely behavior)
- (b) Use best validation loss epoch
- (c) Try both and report

**Recommendation:** (a) first, then (c) if results don't match.

## Q7. The missing LAI features

The paper says 6 dynamic features but the code uses 8 (adding 2 LAI features).
Should we:
- (a) Include LAI (matching the code)
- (b) Exclude LAI (matching the paper text)
- (c) Test both

**Recommendation:** (a) - the code is the ground truth for what they actually did.
The paper text appears to have omitted these features from the description.

## Q8. GPU availability on Palma2?

Need to check:
- Which GPU partitions are available
- How much VRAM needed (dataset is moderate, 256 hidden units)
- Estimated training time per model

## Q9. Integration with our pipeline

After replication, how does this relate to our existing work?
- Our pipeline predicts sap velocity (cm3/cm2/h), theirs predicts sap flow (cm3/h)
- Our pipeline uses ERA5, theirs uses SAPFLUXNET on-site met data
- Our target is daily, theirs is hourly
- Our approach is XGBoost, theirs is LSTM

This is a comparison/benchmark study, not a replacement for our pipeline.
