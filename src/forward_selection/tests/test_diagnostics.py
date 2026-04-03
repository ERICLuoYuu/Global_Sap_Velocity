"""Tests for diagnostics.py — JSON roundtrip, plotting, export."""

import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # non-interactive backend for CI

from src.forward_selection.diagnostics import (
    compare_scorers,
    export_selected_features,
    load_results,
    plot_feature_ranking,
    plot_performance_curve,
)


def _make_results(scoring: str = "mean_r2") -> dict[str, Any]:
    """Build a synthetic results dict matching selector output."""
    return {
        "scoring": scoring,
        "best_score": 0.72,
        "n_selected": 10,
        "selected_features": ["sw_in", "vpd", "ta", "LAI"],
        "selected_indices": [0, 1, 2, 3],
        "step_scores": {7: 0.55, 8: 0.62, 9: 0.68, 10: 0.72, 11: 0.71},
        "step_pooled_r2": {7: 0.53, 8: 0.60, 9: 0.66, 10: 0.70, 11: 0.69},
        "step_pooled_rmse": {7: 0.45, 8: 0.40, 9: 0.35, 10: 0.32, 11: 0.33},
        "step_features": {
            7: ["sw_in", "vpd", "ta", "ext_rad", "ppfd_in", "ws", "LAI"],
            8: ["sw_in", "vpd", "ta", "ext_rad", "ppfd_in", "ws", "LAI", "precip"],
            9: ["sw_in", "vpd", "ta", "ext_rad", "ppfd_in", "ws", "LAI", "precip", "elevation"],
            10: ["sw_in", "vpd", "ta", "ext_rad", "ppfd_in", "ws", "LAI", "precip", "elevation", "Year sin"],
            11: [
                "sw_in",
                "vpd",
                "ta",
                "ext_rad",
                "ppfd_in",
                "ws",
                "LAI",
                "precip",
                "elevation",
                "Year sin",
                "canopy_height",
            ],
        },
        "elapsed_minutes": 5.5,
        "config": {"n_splits": 10, "random_seed": 42, "k_features": "best", "forward": True, "floating": False},
    }


def _write_results_json(results: dict, path: Path) -> None:
    """Write results to JSON with string keys (matching _save_results)."""
    serializable = dict(results)
    for key in ("step_scores", "step_pooled_r2", "step_pooled_rmse", "step_features"):
        serializable[key] = {str(k): v for k, v in results[key].items()}
    with open(path, "w") as f:
        json.dump(serializable, f)


class TestLoadResults:
    def test_converts_step_keys_to_int(self, tmp_path: Path) -> None:
        results = _make_results()
        jpath = tmp_path / "ffs_mean_r2_results.json"
        _write_results_json(results, jpath)
        loaded = load_results(jpath)
        assert all(isinstance(k, int) for k in loaded["step_scores"])

    def test_converts_pooled_keys_to_int(self, tmp_path: Path) -> None:
        results = _make_results()
        jpath = tmp_path / "ffs_mean_r2_results.json"
        _write_results_json(results, jpath)
        loaded = load_results(jpath)
        assert all(isinstance(k, int) for k in loaded["step_pooled_r2"])
        assert all(isinstance(k, int) for k in loaded["step_pooled_rmse"])

    def test_roundtrip_preserves_values(self, tmp_path: Path) -> None:
        results = _make_results()
        jpath = tmp_path / "ffs_mean_r2_results.json"
        _write_results_json(results, jpath)
        loaded = load_results(jpath)
        assert loaded["best_score"] == pytest.approx(0.72)
        assert loaded["step_scores"][10] == pytest.approx(0.72)

    def test_handles_missing_pooled_keys(self, tmp_path: Path) -> None:
        results = _make_results()
        del results["step_pooled_r2"]
        del results["step_pooled_rmse"]
        jpath = tmp_path / "ffs_mean_r2_results.json"
        with open(jpath, "w") as f:
            serializable = dict(results)
            for key in ("step_scores", "step_features"):
                serializable[key] = {str(k): v for k, v in results[key].items()}
            json.dump(serializable, f)
        loaded = load_results(jpath)
        assert "step_pooled_r2" not in loaded


class TestPlotPerformanceCurve:
    def test_returns_figure(self) -> None:
        fig = plot_performance_curve(_make_results())
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        out = tmp_path / "curve.png"
        plot_performance_curve(_make_results(), output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_with_neg_rmse_scoring(self) -> None:
        results = _make_results(scoring="neg_rmse")
        # Negate scores to simulate neg_rmse
        results["step_scores"] = {k: -v for k, v in results["step_scores"].items()}
        fig = plot_performance_curve(results)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_without_pooled_data(self) -> None:
        results = _make_results()
        del results["step_pooled_r2"]
        del results["step_pooled_rmse"]
        fig = plot_performance_curve(results)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotFeatureRanking:
    def test_returns_figure(self) -> None:
        fig = plot_feature_ranking(_make_results())
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_saves_to_file(self, tmp_path: Path) -> None:
        out = tmp_path / "ranking.png"
        plot_feature_ranking(_make_results(), output_path=out)
        assert out.exists()
        import matplotlib.pyplot as plt

        plt.close("all")


class TestExportSelectedFeatures:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "selected.json"
        export_selected_features(_make_results(), out)
        with open(out) as f:
            data = json.load(f)
        assert data["scoring"] == "mean_r2"
        assert data["best_score"] == pytest.approx(0.72)
        assert isinstance(data["selected_features"], list)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "dir" / "selected.json"
        export_selected_features(_make_results(), out)
        assert out.exists()


class TestCompareScorers:
    def test_empty_directory(self, tmp_path: Path) -> None:
        result = compare_scorers(tmp_path)
        assert result == {}

    def test_single_scorer(self, tmp_path: Path) -> None:
        results = _make_results()
        _write_results_json(results, tmp_path / "ffs_mean_r2_results.json")
        comparison = compare_scorers(tmp_path)
        assert "mean_r2" in comparison
        assert comparison["mean_r2"]["best_score"] == pytest.approx(0.72)

    def test_two_scorers_consensus(self, tmp_path: Path) -> None:
        r1 = _make_results("mean_r2")
        r1["selected_features"] = ["sw_in", "vpd", "LAI"]
        _write_results_json(r1, tmp_path / "ffs_mean_r2_results.json")

        r2 = _make_results("neg_rmse")
        r2["selected_features"] = ["sw_in", "vpd", "precip"]
        _write_results_json(r2, tmp_path / "ffs_neg_rmse_results.json")

        comparison = compare_scorers(tmp_path)
        assert set(comparison["_consensus"]) == {"sw_in", "vpd"}

    def test_saves_comparison_json(self, tmp_path: Path) -> None:
        _write_results_json(_make_results(), tmp_path / "ffs_mean_r2_results.json")
        compare_scorers(tmp_path)
        assert (tmp_path / "ffs_comparison.json").exists()


class TestEdgeCases:
    """Boundary values, negative paths, regression tests."""

    def test_single_step_performance_curve(self) -> None:
        results = _make_results()
        results["step_scores"] = {7: 0.55}
        results["step_pooled_r2"] = {7: 0.53}
        results["step_pooled_rmse"] = {7: 0.45}
        results["step_features"] = {7: ["sw_in", "vpd"]}
        fig = plot_performance_curve(results)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_single_step_feature_ranking(self) -> None:
        results = _make_results()
        results["step_scores"] = {7: 0.55}
        results["step_features"] = {7: ["sw_in", "vpd"]}
        fig = plot_feature_ranking(results)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_save_load_plot_roundtrip(self, tmp_path: Path) -> None:
        """Full integration: save → load → plot without error."""
        results = _make_results()
        jpath = tmp_path / "ffs_mean_r2_results.json"
        _write_results_json(results, jpath)
        loaded = load_results(jpath)
        fig = plot_performance_curve(loaded, output_path=tmp_path / "curve.png")
        assert (tmp_path / "curve.png").exists()
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_neg_rmse_uses_pooled_rmse_overlay(self, tmp_path: Path) -> None:
        """Regression: neg_rmse scoring should overlay pooled RMSE, not pooled R2."""
        results = _make_results(scoring="neg_rmse")
        results["step_scores"] = {k: -0.5 + 0.01 * k for k in results["step_scores"]}
        fig = plot_performance_curve(results, output_path=tmp_path / "rmse_curve.png")
        assert (tmp_path / "rmse_curve.png").exists()
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_export_includes_all_required_keys(self, tmp_path: Path) -> None:
        out = tmp_path / "sel.json"
        export_selected_features(_make_results(), out)
        with open(out) as f:
            data = json.load(f)
        required = {"scoring", "best_score", "selected_features", "n_selected"}
        assert required.issubset(set(data.keys()))
