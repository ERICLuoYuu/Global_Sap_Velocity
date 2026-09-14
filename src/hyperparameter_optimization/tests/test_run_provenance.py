"""``FINAL_config_<run_id>.json`` must say how the run was configured.

Every trainer recorded ``feature_names`` -- the FINAL, post-engineering,
post-one-hot list -- but never the inputs that produced it. Two runs differing
only in ``--spatial_split_method`` (one group per SITE versus a 0.05 deg grid,
worth +0.145 pooled R2 through co-location leakage) wrote configs identical
apart from their scores. A finished run could not be identified from its own
config, which is how a comparison between two runs stopped being checkable.
"""

import argparse
import ast
import json
from pathlib import Path

import pytest

from src.hyperparameter_optimization.training_utils import build_run_provenance

_HPO = Path(__file__).resolve().parents[1]
_SRC = _HPO.parent


def _args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


class TestBuildRunProvenance:
    """The helper the six config writers share."""

    def test_records_the_selected_feature_list_and_its_source(self) -> None:
        out = build_run_provenance(
            _args(selected_features="/path/to/ffs_cze_site.json"),
            ["sap_velocity", "vpd", "ta"],
        )
        assert out["selected_features"] == ["sap_velocity", "vpd", "ta"]  # nosec B101
        assert out["selected_features_path"] == "/path/to/ffs_cze_site.json"  # nosec B101

    def test_records_the_split_method_that_distinguishes_two_runs(self) -> None:
        out = build_run_provenance(
            _args(spatial_split_method="grid", n_groups=10, grid_size=0.05, BALANCED=False),
            [],
        )
        assert out["split_config"]["spatial_split_method"] == "grid"  # nosec B101
        assert out["split_config"]["n_groups"] == 10  # nosec B101
        assert out["split_config"]["grid_size"] == 0.05  # nosec B101
        assert out["split_config"]["balanced"] is False  # nosec B101

    def test_a_flag_the_parser_never_defined_records_none_instead_of_raising(self) -> None:
        """The property that keeps this from destroying a finished run.

        Five of the six callers build their own ArgumentParser. The DL and
        autoML parsers define no ``--grid_size``; ``train_from_cache`` defines
        none of the spatial flags. This runs AFTER the model is fitted, so an
        AttributeError here would throw away hours of training to save a field.
        """
        out = build_run_provenance(_args(), None)
        assert out["split_config"]["grid_size"] is None  # nosec B101
        assert out["split_config"]["spatial_split_method"] is None  # nosec B101
        assert out["selected_features"] is None  # nosec B101
        assert out["selected_features_path"] is None  # nosec B101

    def test_n_groups_falls_back_to_the_ffs_cache_spelling(self) -> None:
        """``train_from_cache`` calls the same quantity ``--n_splits``."""
        assert build_run_provenance(_args(n_splits=10), [])["split_config"]["n_groups"] == 10  # nosec B101

    def test_a_zero_valued_flag_is_kept_not_treated_as_unset(self) -> None:
        """``getattr(...) or next`` would silently fall through on 0/False."""
        out = build_run_provenance(_args(n_groups=0, n_splits=99, BALANCED=False, IS_STRATIFIED=False), [])
        assert out["split_config"]["n_groups"] == 0  # nosec B101
        assert out["split_config"]["balanced"] is False  # nosec B101
        assert out["split_config"]["is_stratified"] is False  # nosec B101

    def test_paths_survive_a_json_dump_without_default_str(self) -> None:
        """``train_from_cache.py`` dumps with no ``default=str``; a raw Path raises."""
        out = build_run_provenance(
            _args(selected_features=Path("/a/features.json")),
            ["vpd"],
            data_dir=Path("/scratch/merged/daily"),
        )
        json.dumps(out)  # must not raise
        assert out["selected_features_path"] == str(Path("/a/features.json"))  # nosec B101
        assert out["split_config"]["data_dir"] == str(Path("/scratch/merged/daily"))  # nosec B101

    def test_the_returned_list_is_a_copy(self) -> None:
        """The config must not alias a list the trainer keeps mutating."""
        cols = ["vpd"]
        out = build_run_provenance(_args(), cols)
        cols.append("ta")
        assert out["selected_features"] == ["vpd"]  # nosec B101

    def test_it_splats_into_a_config_literal(self) -> None:
        """How all six callers use it: ``{..., **build_run_provenance(...), ...}``."""
        config = {"model_type": "xgb", **build_run_provenance(_args(n_groups=10), ["vpd"]), "random_seed": 42}
        assert config["model_type"] == "xgb"  # nosec B101
        assert config["random_seed"] == 42  # nosec B101
        assert config["split_config"]["n_groups"] == 10  # nosec B101


#: Every module that writes a ``FINAL_config_*.json``. Found with
#: ``grep -rn 'FINAL_config_' --include=*.py src/``; the five trainers plus the
#: FFS-cache trainer. Adding a seventh without provenance should fail here.
WRITERS = (
    ("ML_spatial_stratified", _HPO / "test_hyperparameter_tuning_ML_spatial_stratified.py"),
    ("ML_spatial_stratified_DA", _HPO / "test_hyperparameter_tuning_ML_spatial_stratified_DA.py"),
    ("ML_spatial_stratified_prediction", _HPO / "test_hyperparameter_tuning_ML_spatial_stratified_prediction.py"),
    ("DL_spatial_stratified", _HPO / "test_hyperparameter_tuning_DL_spatial_stratified.py"),
    ("autoML_spatial_stratified", _HPO / "test_hyperparameter_tuning_autoML_spatial_stratified.py"),
    ("forward_selection/train_from_cache", _SRC / "forward_selection" / "train_from_cache.py"),
)


def _calls_build_run_provenance(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "build_run_provenance":
                return True
    return False


class TestEveryConfigWriterRecordsProvenance:
    """A helper nobody calls fixes nothing."""

    @pytest.mark.parametrize("label,path", WRITERS, ids=[w[0] for w in WRITERS])
    def test_writer_calls_the_helper(self, label: str, path: Path) -> None:
        if not path.exists():
            pytest.skip(f"{label} not present in this checkout")
        assert _calls_build_run_provenance(path.read_text(encoding="utf-8")), (  # nosec B101
            f"{label} writes a FINAL_config but never calls build_run_provenance(), "
            "so its config cannot say which features or split method produced it."
        )

    def test_every_named_writer_actually_exists(self) -> None:
        """The list is an allowlist; a typo would skip its check silently."""
        missing = [label for label, path in WRITERS if not path.exists()]
        assert missing == [], f"WRITERS names files that are not here: {missing}"  # nosec B101

    def test_detector_is_not_vacuous(self) -> None:
        assert _calls_build_run_provenance("cfg = {**build_run_provenance(args, cols)}")  # nosec B101
        assert not _calls_build_run_provenance("cfg = {'feature_names': names}")  # nosec B101
