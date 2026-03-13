"""Phase 4 validation: realistic AOA pipeline test.

Simulates a realistic scenario with spatially-structured training data,
multiple biomes, 10-fold stratified spatial CV, and mixed in/out-of-distribution
prediction points. Validates memory usage, correctness, and output consistency.
"""

import json
import os

import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from src.aoa.aoa import compute_aoa
from src.aoa.model_bridge import (
    load_aoa_arrays,
    load_model_config,
    load_shap_weights,
    reconstruct_fold_indices,
    save_aoa_arrays,
)
from src.aoa.plotting import plot_aoa_map, plot_di_histogram


def _make_realistic_data(rng, n_sites=50, n_features=27, records_per_site=200):
    """Generate realistic spatially-structured training data.

    Mimics 50 sites with 200 records each (10,000 total), 27 features,
    8 PFT classes, and spatial groups from grid-based binning.
    """
    # Simulate site locations across global range
    lats = rng.uniform(-60, 78, n_sites)
    lons = rng.uniform(-180, 180, n_sites)

    # Grid-based spatial grouping (0.05 degree grid, then unique-ify)
    lat_bins = np.arange(-60, 79, 0.05)
    lon_bins = np.arange(-180, 181, 0.05)
    lat_idx = np.digitize(lats, lat_bins) - 1
    lon_idx = np.digitize(lons, lon_bins) - 1
    grid_ids = lat_idx * len(lon_bins) + lon_idx
    unique_grids = np.unique(grid_ids)
    grid_map = {g: i for i, g in enumerate(unique_grids)}
    site_groups = np.array([grid_map[g] for g in grid_ids])

    # 8 PFT classes (matching real model)
    pft_names = ["MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"]
    site_pfts = rng.choice(pft_names, n_sites)

    # Build record-level arrays
    X_list, group_list, pft_list = [], [], []
    for i in range(n_sites):
        # Each site has slightly different feature distributions (realistic)
        site_mean = rng.normal(0, 1, n_features)
        site_std = rng.uniform(0.5, 2.0, n_features)
        X_site = rng.normal(site_mean, site_std, (records_per_site, n_features))
        X_list.append(X_site)
        group_list.append(np.full(records_per_site, site_groups[i]))
        pft_list.append(np.full(records_per_site, site_pfts[i]))

    X_train = np.vstack(X_list)
    groups = np.concatenate(group_list)
    pfts_str = np.concatenate(pft_list)
    pfts_encoded = np.unique(pfts_str, return_inverse=True)[1]

    return X_train, groups, pfts_encoded, pfts_str


class TestRealisticAOAPipeline:
    """Full pipeline validation with realistic synthetic data."""

    @pytest.fixture
    def realistic_data(self):
        rng = np.random.default_rng(42)
        X_train, groups, pfts_encoded, pfts_str = _make_realistic_data(rng)
        n_folds = 10
        seed = 42

        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        X_dummy = np.zeros((X_train.shape[0], 1))
        fold_indices = [test_idx for _, test_idx in cv.split(X_dummy, pfts_encoded, groups)]

        # SHAP-like weights (normalized)
        raw_weights = rng.uniform(0.01, 1.0, X_train.shape[1])
        weights = raw_weights / raw_weights.sum()

        return {
            "X_train": X_train,
            "groups": groups,
            "pfts_encoded": pfts_encoded,
            "fold_indices": fold_indices,
            "weights": weights,
            "n_folds": n_folds,
            "seed": seed,
        }

    def test_full_pipeline_runs(self, realistic_data):
        """AOA computation completes on 10k training + 5k new points."""
        rng = np.random.default_rng(99)
        X_train = realistic_data["X_train"]

        # 2500 in-distribution + 2500 out-of-distribution
        in_dist = X_train[rng.choice(X_train.shape[0], 2500, replace=False)]
        out_dist = rng.normal(
            loc=X_train.mean(axis=0) + 3 * X_train.std(axis=0),
            scale=X_train.std(axis=0),
            size=(2500, X_train.shape[1]),
        )
        new_X = np.vstack([in_dist, out_dist])

        result = compute_aoa(
            train_X=X_train,
            new_X=new_X,
            weights=realistic_data["weights"],
            fold_indices=realistic_data["fold_indices"],
            chunk_size=2000,
        )

        assert result["di"].shape == (5000,)
        assert result["aoa"].shape == (5000,)
        assert result["di_train"].shape == (X_train.shape[0],)
        assert isinstance(result["threshold"], float)
        assert isinstance(result["d_bar"], float)
        assert result["threshold"] > 0
        assert result["d_bar"] > 0

    def test_in_dist_mostly_inside_aoa(self, realistic_data):
        """In-distribution points should mostly be inside AOA."""
        rng = np.random.default_rng(99)
        X_train = realistic_data["X_train"]
        in_dist = X_train[rng.choice(X_train.shape[0], 1000, replace=False)]

        result = compute_aoa(
            train_X=X_train,
            new_X=in_dist,
            weights=realistic_data["weights"],
            fold_indices=realistic_data["fold_indices"],
        )

        pct_inside = result["aoa"].sum() / len(result["aoa"])
        assert pct_inside > 0.7, f"Expected >70% inside AOA, got {pct_inside:.1%}"

    def test_out_dist_mostly_outside_aoa(self, realistic_data):
        """Out-of-distribution points should mostly be outside AOA."""
        rng = np.random.default_rng(99)
        X_train = realistic_data["X_train"]
        out_dist = rng.normal(
            loc=X_train.mean(axis=0) + 5 * X_train.std(axis=0),
            scale=X_train.std(axis=0),
            size=(1000, X_train.shape[1]),
        )

        result = compute_aoa(
            train_X=X_train,
            new_X=out_dist,
            weights=realistic_data["weights"],
            fold_indices=realistic_data["fold_indices"],
        )

        pct_outside = 1.0 - result["aoa"].sum() / len(result["aoa"])
        assert pct_outside > 0.7, f"Expected >70% outside AOA, got {pct_outside:.1%}"

    def test_chunked_matches_unchunked(self, realistic_data):
        """Chunked and non-chunked DI values should be identical."""
        rng = np.random.default_rng(99)
        X_train = realistic_data["X_train"]
        new_X = X_train[rng.choice(X_train.shape[0], 500, replace=False)]

        result_chunked = compute_aoa(
            train_X=X_train,
            new_X=new_X,
            weights=realistic_data["weights"],
            fold_indices=realistic_data["fold_indices"],
            chunk_size=100,
        )
        result_full = compute_aoa(
            train_X=X_train,
            new_X=new_X,
            weights=realistic_data["weights"],
            fold_indices=realistic_data["fold_indices"],
            chunk_size=0,
        )

        np.testing.assert_allclose(result_chunked["di"], result_full["di"])
        np.testing.assert_array_equal(result_chunked["aoa"], result_full["aoa"])

    def test_fold_reconstruction_matches(self, realistic_data):
        """Reconstructed fold indices should match original."""
        X_train = realistic_data["X_train"]
        groups = realistic_data["groups"]
        pfts_encoded = realistic_data["pfts_encoded"]

        reconstructed = reconstruct_fold_indices(
            n_samples=X_train.shape[0],
            groups=groups,
            pfts_encoded=pfts_encoded,
            n_folds=realistic_data["n_folds"],
            random_seed=realistic_data["seed"],
            split_type="spatial_stratified",
        )

        original = realistic_data["fold_indices"]
        assert len(reconstructed) == len(original)
        for orig, recon in zip(original, reconstructed):
            np.testing.assert_array_equal(np.sort(orig), np.sort(recon))


class TestArtifactRoundtrip:
    """Test saving/loading model artifacts through the full pipeline."""

    def test_save_load_run_aoa(self, tmp_path):
        """Save artifacts, load them, and run AOA end-to-end."""
        rng = np.random.default_rng(42)
        X_train, groups, pfts_encoded, _ = _make_realistic_data(
            rng,
            n_sites=20,
            records_per_site=100,
        )

        run_id = "test_run"
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Save AOA arrays
        save_aoa_arrays(model_dir, run_id, X_train, groups, pfts_encoded)

        # Save config
        feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]
        config = {
            "feature_names": feature_names,
            "random_seed": 42,
            "split_type": "spatial_stratified",
            "cv_results": {"n_folds": 5},
        }
        with open(model_dir / f"FINAL_config_{run_id}.json", "w") as f:
            json.dump(config, f)

        # Save SHAP CSV
        import pandas as pd

        shap_weights = rng.uniform(0.01, 1.0, len(feature_names))
        shap_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": shap_weights,
            }
        )
        shap_csv = tmp_path / "shap.csv"
        shap_df.to_csv(shap_csv, index=False)

        # Load everything back
        loaded_config = load_model_config(model_dir / f"FINAL_config_{run_id}.json")
        loaded_weights = load_shap_weights(shap_csv, loaded_config["feature_names"])
        loaded_X, loaded_groups, loaded_pfts = load_aoa_arrays(model_dir, run_id)

        np.testing.assert_array_equal(loaded_X, X_train)
        np.testing.assert_array_equal(loaded_groups, groups)
        np.testing.assert_array_equal(loaded_pfts, pfts_encoded)

        # Reconstruct folds and compute AOA
        fold_indices = reconstruct_fold_indices(
            n_samples=loaded_X.shape[0],
            groups=loaded_groups,
            pfts_encoded=loaded_pfts,
            n_folds=loaded_config["cv_results"]["n_folds"],
            random_seed=loaded_config["random_seed"],
            split_type=loaded_config["split_type"],
        )

        new_X = rng.normal(size=(200, X_train.shape[1]))
        result = compute_aoa(
            train_X=loaded_X,
            new_X=new_X,
            weights=loaded_weights,
            fold_indices=fold_indices,
        )

        assert result["di"].shape == (200,)
        assert result["threshold"] > 0


class TestPlottingIntegration:
    """Test plotting with realistic AOA outputs."""

    def test_di_histogram_with_real_data(self, tmp_path):
        """Plot DI histogram from a full AOA run."""
        rng = np.random.default_rng(42)
        X_train, groups, pfts_encoded, _ = _make_realistic_data(
            rng,
            n_sites=15,
            records_per_site=50,
        )

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        X_dummy = np.zeros((X_train.shape[0], 1))
        fold_indices = [t for _, t in cv.split(X_dummy, pfts_encoded, groups)]
        weights = np.ones(X_train.shape[1]) / X_train.shape[1]

        new_X = rng.normal(size=(200, X_train.shape[1]))
        result = compute_aoa(
            train_X=X_train,
            new_X=new_X,
            weights=weights,
            fold_indices=fold_indices,
        )

        fig, ax = plot_di_histogram(result["di"], result["threshold"], di_train=result["di_train"])
        fig.savefig(tmp_path / "di_hist.png", dpi=72)
        import matplotlib.pyplot as plt

        plt.close(fig)

        assert (tmp_path / "di_hist.png").exists()
        assert (tmp_path / "di_hist.png").stat().st_size > 0

    def test_aoa_map_with_synthetic_grid(self, tmp_path):
        """Plot AOA map on a 10x10 degree synthetic grid."""
        rng = np.random.default_rng(42)

        # 10x10 degree tile at 0.5 degree resolution = 20x20 = 400 points
        lats = np.arange(40, 50, 0.5)
        lons = np.arange(0, 10, 0.5)

        # Simulated DI values — shape must be (len(lats), len(lons))
        di_grid = rng.exponential(scale=0.5, size=(len(lats), len(lons)))
        threshold = 1.0

        fig, ax = plot_aoa_map(
            lats,
            lons,
            di_grid,
            threshold,
        )
        fig.savefig(tmp_path / "aoa_map.png", dpi=72)
        import matplotlib.pyplot as plt

        plt.close(fig)

        assert (tmp_path / "aoa_map.png").exists()
