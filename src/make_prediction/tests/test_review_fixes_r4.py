"""
Test Round 4: Tests for review fixes applied in the 2026-03-19 review session.

Covers:
- C1: file_start_time initialization
- H1: sys.exit(1) on unhandled errors
- H2: Duplicate target index averaging in map_predictions_to_df_improved
- H3: Configurable log path via SAP_LOG_DIR
- M1: No duplicate comments (code hygiene)
"""

import ast
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from predict_sap_velocity_sequential import (
    DataTransformer,
    ModelConfig,
    _get_log_path,
    map_predictions_to_df_improved,
    prepare_flat_features,
)

# =============================================================================
# C1: file_start_time initialization
# =============================================================================


class TestFileStartTime:
    """Verify that file_start_time is defined before use in main()."""

    def test_file_start_time_defined_before_use(self):
        """Parse the AST to verify file_start_time assignment precedes its usage."""
        src_path = Path(__file__).parent.parent / "predict_sap_velocity_sequential.py"
        with open(src_path) as f:
            tree = ast.parse(f.read())

        # Find the main() function
        main_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                main_func = node
                break

        assert main_func is not None, "main() function not found"

        # Find file_start_time assignment and usage lines
        assign_line = None
        use_line = None
        for node in ast.walk(main_func):
            # Assignment: file_start_time = time.time()
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "file_start_time":
                        assign_line = node.lineno
                        break
            # Usage: end_time - file_start_time
            if isinstance(node, ast.Name) and node.id == "file_start_time":
                if use_line is None or node.lineno > (assign_line or 0):
                    use_line = node.lineno

        assert assign_line is not None, "file_start_time is never assigned in main()"
        assert use_line is not None, "file_start_time is never used in main()"
        assert assign_line < use_line, f"file_start_time assigned at line {assign_line} but used at line {use_line}"


# =============================================================================
# H1: sys.exit(1) on unhandled errors
# =============================================================================


class TestUnhandledErrorExit:
    """Verify that the top-level except block calls sys.exit(1)."""

    def test_main_guard_exits_on_exception(self):
        """Parse AST to verify sys.exit(1) in the except Exception block."""
        src_path = Path(__file__).parent.parent / "predict_sap_velocity_sequential.py"
        with open(src_path) as f:
            source = f.read()

        # Find the if __name__ == "__main__" block
        tree = ast.parse(source)
        found_exit = False

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                # Check if this handler catches Exception (general)
                if node.type and isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    # Check body for sys.exit(1)
                    for stmt in ast.walk(node):
                        if isinstance(stmt, ast.Call):
                            func = stmt.func
                            if (
                                isinstance(func, ast.Attribute)
                                and isinstance(func.value, ast.Name)
                                and func.value.id == "sys"
                                and func.attr == "exit"
                            ):
                                # Check the argument is 1
                                if stmt.args and isinstance(stmt.args[0], ast.Constant):
                                    if stmt.args[0].value == 1:
                                        found_exit = True

        assert found_exit, "No sys.exit(1) found in except Exception handler"


# =============================================================================
# H2: Duplicate target index averaging
# =============================================================================


class TestDuplicateIndexAveraging:
    """Verify that overlapping predictions are averaged, not overwritten."""

    def _make_test_data(self, n_rows=20):
        """Create a simple test DataFrame."""
        return pd.DataFrame(
            {
                "ta": np.random.randn(n_rows),
                "vpd": np.random.randn(n_rows),
            }
        )

    def test_no_duplicates_preserves_all_predictions(self):
        """When no duplicate indices, all predictions are kept as-is."""
        df = self._make_test_data(10)
        predictions = np.array([1.0, 2.0, 3.0])
        metadata = [
            {
                "prediction_target_idx": 0,
                "location_id": "A",
                "window_start_idx": 0,
                "window_end_idx": 2,
                "window_position": 0,
            },
            {
                "prediction_target_idx": 3,
                "location_id": "A",
                "window_start_idx": 1,
                "window_end_idx": 3,
                "window_position": 1,
            },
            {
                "prediction_target_idx": 6,
                "location_id": "A",
                "window_start_idx": 4,
                "window_end_idx": 6,
                "window_position": 2,
            },
        ]

        result = map_predictions_to_df_improved(df, predictions, metadata, "xgb")
        pred_col = "sap_velocity_xgb"

        assert result[pred_col].iloc[0] == pytest.approx(1.0)
        assert result[pred_col].iloc[3] == pytest.approx(2.0)
        assert result[pred_col].iloc[6] == pytest.approx(3.0)

    def test_duplicate_indices_are_averaged(self):
        """When multiple windows predict the same index, results should be averaged."""
        df = self._make_test_data(10)
        # Two predictions for index 5: values 4.0 and 6.0 -> average should be 5.0
        predictions = np.array([4.0, 6.0, 10.0])
        metadata = [
            {
                "prediction_target_idx": 5,
                "location_id": "A",
                "window_start_idx": 0,
                "window_end_idx": 4,
                "window_position": 0,
            },
            {
                "prediction_target_idx": 5,
                "location_id": "A",
                "window_start_idx": 1,
                "window_end_idx": 5,
                "window_position": 1,
            },
            {
                "prediction_target_idx": 8,
                "location_id": "A",
                "window_start_idx": 3,
                "window_end_idx": 7,
                "window_position": 2,
            },
        ]

        result = map_predictions_to_df_improved(df, predictions, metadata, "xgb")
        pred_col = "sap_velocity_xgb"

        # Index 5 should be average of 4.0 and 6.0 = 5.0
        assert result[pred_col].iloc[5] == pytest.approx(5.0), f"Expected 5.0 but got {result[pred_col].iloc[5]}"
        # Index 8 should be 10.0 (no overlap)
        assert result[pred_col].iloc[8] == pytest.approx(10.0)

    def test_triple_duplicate_averaging(self):
        """Three predictions for the same index should all be averaged."""
        df = self._make_test_data(10)
        predictions = np.array([3.0, 6.0, 9.0])
        metadata = [
            {
                "prediction_target_idx": 5,
                "location_id": "A",
                "window_start_idx": 0,
                "window_end_idx": 4,
                "window_position": 0,
            },
            {
                "prediction_target_idx": 5,
                "location_id": "A",
                "window_start_idx": 1,
                "window_end_idx": 5,
                "window_position": 1,
            },
            {
                "prediction_target_idx": 5,
                "location_id": "A",
                "window_start_idx": 2,
                "window_end_idx": 6,
                "window_position": 2,
            },
        ]

        result = map_predictions_to_df_improved(df, predictions, metadata, "xgb")
        pred_col = "sap_velocity_xgb"

        # Average of 3, 6, 9 = 6.0
        assert result[pred_col].iloc[5] == pytest.approx(6.0)

    def test_none_target_indices_skipped(self):
        """Predictions with None target_idx should be excluded."""
        df = self._make_test_data(10)
        predictions = np.array([1.0, 2.0, 3.0])
        metadata = [
            {
                "prediction_target_idx": 2,
                "location_id": "A",
                "window_start_idx": 0,
                "window_end_idx": 1,
                "window_position": 0,
            },
            {
                "prediction_target_idx": None,
                "location_id": "A",
                "window_start_idx": 1,
                "window_end_idx": 2,
                "window_position": 1,
            },
            {
                "prediction_target_idx": 7,
                "location_id": "A",
                "window_start_idx": 5,
                "window_end_idx": 6,
                "window_position": 2,
            },
        ]

        result = map_predictions_to_df_improved(df, predictions, metadata, "xgb")
        pred_col = "sap_velocity_xgb"

        assert result[pred_col].notna().sum() == 2  # Only 2 valid predictions


# =============================================================================
# H3: Configurable log path
# =============================================================================


class TestLogPath:
    """Verify _get_log_path respects SAP_LOG_DIR env var."""

    def test_default_log_path_in_cwd(self):
        """Without SAP_LOG_DIR, log goes to cwd."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove SAP_LOG_DIR if present
            os.environ.pop("SAP_LOG_DIR", None)
            path = _get_log_path()
            assert path.name == "sap_velocity_prediction.log"

    def test_log_path_with_sap_log_dir(self, tmp_path):
        """With SAP_LOG_DIR set to valid dir, log goes there."""
        with patch.dict(os.environ, {"SAP_LOG_DIR": str(tmp_path)}):
            path = _get_log_path()
            assert path.parent == tmp_path
            assert path.name == "sap_velocity_prediction.log"

    def test_log_path_with_invalid_sap_log_dir(self):
        """With SAP_LOG_DIR set to non-existent dir, falls back to cwd."""
        with patch.dict(os.environ, {"SAP_LOG_DIR": "/nonexistent/dir/xyz"}):
            path = _get_log_path()
            # Should fall back to cwd
            assert path.name == "sap_velocity_prediction.log"
            assert str(path.parent) in (".", "")

    def test_log_path_with_empty_sap_log_dir(self):
        """Empty SAP_LOG_DIR should fall back to cwd."""
        with patch.dict(os.environ, {"SAP_LOG_DIR": ""}):
            path = _get_log_path()
            assert path.name == "sap_velocity_prediction.log"


# =============================================================================
# Edge cases for existing functions affected by fixes
# =============================================================================


class TestEdgeCases:
    """Edge cases for functions that interact with the fixed code."""

    def test_map_predictions_empty_metadata(self):
        """Empty metadata should return DataFrame with NaN predictions."""
        df = pd.DataFrame({"ta": [1, 2, 3]})
        result = map_predictions_to_df_improved(df, np.array([]), [], "xgb")
        assert "sap_velocity_xgb" not in result.columns or result["sap_velocity_xgb"].isna().all()

    def test_map_predictions_mismatched_lengths(self):
        """More predictions than metadata should only map available metadata."""
        df = pd.DataFrame({"ta": range(10)})
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metadata = [
            {
                "prediction_target_idx": i,
                "location_id": "A",
                "window_start_idx": 0,
                "window_end_idx": i,
                "window_position": i,
            }
            for i in range(3)  # Only 3 metadata entries
        ]

        result = map_predictions_to_df_improved(df, predictions, metadata, "xgb")
        pred_col = "sap_velocity_xgb"
        assert result[pred_col].notna().sum() == 3

    def test_prepare_flat_features_all_nan_column(self):
        """Column with all NaN should result in zero valid samples."""
        df = pd.DataFrame(
            {
                "feat1": [1.0, 2.0, 3.0],
                "feat2": [np.nan, np.nan, np.nan],  # All NaN
            }
        )
        X, indices = prepare_flat_features(df, ["feat1", "feat2"])
        assert len(X) == 0
        assert len(indices) == 0

    def test_prepare_flat_features_partial_nan(self):
        """Rows with any NaN should be excluded."""
        df = pd.DataFrame(
            {
                "feat1": [1.0, 2.0, 3.0, 4.0],
                "feat2": [10.0, np.nan, 30.0, 40.0],
            }
        )
        X, indices = prepare_flat_features(df, ["feat1", "feat2"])
        assert len(X) == 3  # Row 1 excluded
        assert 1 not in indices

    def test_data_transformer_log1p_roundtrip(self):
        """log1p -> expm1 roundtrip should preserve values."""
        config = ModelConfig(model_type="xgb", target_transform="log1p")
        transformer = DataTransformer(config)

        original = np.array([0.0, 1.0, 10.0, 100.0])
        transformed = transformer.transform_target(original, inverse=False)
        recovered = transformer.transform_target(transformed, inverse=True)

        np.testing.assert_allclose(recovered, original, rtol=1e-10)

    def test_data_transformer_negative_clamp_log1p(self):
        """Negative values should be clamped to 0 before log1p."""
        config = ModelConfig(model_type="xgb", target_transform="log1p")
        transformer = DataTransformer(config)

        values = np.array([-5.0, -1.0, 0.0, 1.0])
        result = transformer.transform_target(values, inverse=False)

        # Negative values clamped to 0, log1p(0) = 0
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 0.0
        assert result[3] > 0.0


# =============================================================================
# M1: Code hygiene verification
# =============================================================================


class TestCodeHygiene:
    """Verify no duplicate comments or other code hygiene issues."""

    def test_no_duplicate_adjacent_comments(self):
        """No two adjacent lines should be identical comments."""
        src_path = Path(__file__).parent.parent / "predict_sap_velocity_sequential.py"
        with open(src_path) as f:
            lines = f.readlines()

        duplicates = []
        for i in range(len(lines) - 1):
            line = lines[i].strip()
            next_line = lines[i + 1].strip()
            if line.startswith("#") and line == next_line:
                duplicates.append((i + 1, line))

        assert not duplicates, f"Found duplicate adjacent comments: {duplicates}"
