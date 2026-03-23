#!/usr/bin/env Rscript
# Generate CAST reference fixtures for Python AOA cross-validation (AC-14).
#
# Usage (on HPC):
#   module load palma/2024a GCC/13.3.0 R/4.4.2
#   Rscript src/aoa/tests/fixtures/generate_cast_fixtures.R
#
# Output: src/aoa/tests/fixtures/cast_reference.json
#
# This script creates a small synthetic dataset, runs CAST::trainDI()
# and CAST::aoa(), and saves all intermediate values so the Python
# implementation can compare within 1e-10 tolerance.

library(CAST)
library(jsonlite)

set.seed(42)

# --- Synthetic dataset: 30 samples, 4 features, 3 CV folds ---
n <- 30L
p <- 4L
feature_names <- c("f1", "f2", "f3", "f4")

# Training data (reproducible via set.seed)
X_train <- matrix(rnorm(n * p), nrow = n, ncol = p)
colnames(X_train) <- feature_names
train_df <- as.data.frame(X_train)

# Response (not used by AOA, but needed for model fit)
y_train <- rnorm(n)

# Feature importance weights (simulating mean |SHAP|)
# CAST expects weight as a 1-row data.frame with column names matching features
set.seed(99)
weights_vec <- abs(rnorm(p)) + 0.01
names(weights_vec) <- feature_names
weights_df <- as.data.frame(t(weights_vec))

# CV fold assignments (balanced 3 folds)
fold_labels <- rep(1:3, length.out = n)
# Create CV index lists as CAST expects: CVtest = list of test indices per fold
CVtest <- list()
CVtrain <- list()
for (k in 1:3) {
    test_idx <- which(fold_labels == k)
    train_idx <- which(fold_labels != k)
    CVtest[[k]] <- test_idx
    CVtrain[[k]] <- train_idx
}

# --- Run CAST::trainDI ---
# trainDI computes: standardize → weight → d_bar → cross-fold DI → threshold
train_result <- trainDI(
    train = train_df,
    weight = weights_df,
    CVtest = CVtest,
    CVtrain = CVtrain
)

# --- New data for prediction DI ---
set.seed(777)
n_new <- 10L
X_new <- matrix(rnorm(n_new * p), nrow = n_new, ncol = p)
colnames(X_new) <- feature_names
new_df <- as.data.frame(X_new)

# Compute prediction DI
aoa_result <- aoa(
    newdata = new_df,
    trainDI = train_result
)

# --- Extract all values ---
# CAST internals:
#   trainDist = pairwise distances in weighted space (= our d_bar source)
#   threshold = outlier-removed max(training DI)
#   trainDI = per-sample cross-fold DI

# Standardization parameters (from scale())
means <- attr(scale(train_df), "scaled:center")
sds <- attr(scale(train_df), "scaled:scale")  # This is sd() = ddof=1

fixtures <- list(
    # Input data
    X_train = as.list(as.data.frame(X_train)),
    X_new = as.list(as.data.frame(X_new)),
    feature_names = feature_names,
    weights = as.numeric(weights_vec),
    fold_labels = as.integer(fold_labels),
    n_train = n,
    n_features = p,
    n_new = n_new,

    # Standardization
    feature_means = as.numeric(means),
    feature_stds = as.numeric(sds),

    # CAST outputs (field names: trainDist_avrgmean, threshold)
    d_bar = train_result$trainDist_avrgmean,
    threshold = train_result$threshold,
    training_di = as.numeric(train_result$trainDI),

    # Prediction DI
    prediction_di = as.numeric(aoa_result$DI),
    prediction_aoa = as.logical(aoa_result$AOA == 1)
)

# Save
output_path <- "src/aoa/tests/fixtures/cast_reference.json"
write_json(fixtures, output_path, pretty = TRUE, digits = 17, auto_unbox = TRUE)
cat("CAST fixtures saved to", output_path, "\n")
cat("d_bar:", fixtures$d_bar, "\n")
cat("threshold:", fixtures$threshold, "\n")
cat("n training DI values:", length(fixtures$training_di), "\n")
cat("n prediction DI values:", length(fixtures$prediction_di), "\n")
