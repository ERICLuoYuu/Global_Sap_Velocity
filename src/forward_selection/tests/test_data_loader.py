"""Tests for data_loader.py — cache roundtrip."""

import ast
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forward_selection import data_loader
from src.forward_selection.data_loader import PFT_COLUMN_PRIORITY, load_cache
from src.forward_selection.run_selection import _parse_k_features


class TestLoadCache:
    def _make_cache(self, tmp_path: Path) -> Path:
        cache_path = tmp_path / "test_cache.npz"
        rng = np.random.RandomState(42)
        np.savez_compressed(
            cache_path,
            X=rng.randn(100, 5).astype(np.float32),
            y=rng.rand(100).astype(np.float32),
            groups=np.repeat([0, 1, 2, 3, 4], 20),
            pfts_encoded=np.repeat([0, 1, 2], [40, 30, 30]),
            feature_names=np.array(["f1", "f2", "f3", "f4", "f5"]),
            pft_categories=np.array(["ENF", "DBF", "EBF"]),
        )
        return cache_path

    def test_loads_all_keys(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        expected_keys = {"X", "y", "groups", "pfts_encoded", "feature_names", "pft_categories"}
        assert set(data.keys()) == expected_keys

    def test_shapes_match(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        assert data["X"].shape == (100, 5)
        assert data["y"].shape == (100,)
        assert data["groups"].shape == (100,)
        assert data["pfts_encoded"].shape == (100,)

    def test_feature_names_are_list(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        assert isinstance(data["feature_names"], list)
        assert data["feature_names"] == ["f1", "f2", "f3", "f4", "f5"]

    def test_pft_categories_are_list(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        assert isinstance(data["pft_categories"], list)
        assert data["pft_categories"] == ["ENF", "DBF", "EBF"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_cache(tmp_path / "nonexistent.npz")

    def test_dtypes_preserved(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        assert data["X"].dtype == np.float32
        assert data["y"].dtype == np.float32


class TestPftColumnPriority:
    """The vegetation class must come from `pft`, never from `biome` by accident.

    The original resolver iterated `df.columns` and matched each against a list
    of acceptable names, which hands precedence to whichever name appears first
    in the CSV. In the merged daily files `biome` is column 39 and `pft` is 40,
    so every run stratified by biome while logging "PFT". Nothing failed, and
    the cache reported 7 classes where 8 were expected -- the only visible
    symptom.
    """

    @staticmethod
    def _resolve(columns):
        """The shipped resolution rule, exercised on a bare column list."""
        by_lower = {c.lower(): c for c in columns}
        return next(
            (by_lower[name] for name in PFT_COLUMN_PRIORITY if name in by_lower),
            None,
        )

    @staticmethod
    def _resolve_old(columns):
        """The behaviour being fixed, kept as a NEGATIVE CONTROL.

        Without it the test below passes for any column order and proves
        nothing: it must be shown that the old rule really did pick `biome`.
        """
        return next(
            (c for c in columns if c.lower() in ["pft", "plant_functional_type", "biome"]),
            None,
        )

    # Column order as the merge actually writes it: biome first, then pft.
    MERGED_ORDER = ["TIMESTAMP", "sap_velocity", "ext_rad", "biome", "pft", "site_name"]

    def test_pft_wins_over_biome_in_merged_column_order(self) -> None:
        assert self._resolve(self.MERGED_ORDER) == "pft"

    def test_negative_control_old_rule_picked_biome(self) -> None:
        # If this ever returns "pft", the bug was not what we think it was and
        # the test above is not testing anything.
        assert self._resolve_old(self.MERGED_ORDER) == "biome"

    def test_order_in_the_file_does_not_matter(self) -> None:
        # The whole point: precedence comes from the list, not the CSV.
        assert self._resolve(["pft", "biome"]) == "pft"
        assert self._resolve(["biome", "pft"]) == "pft"

    def test_biome_used_only_when_pft_absent(self) -> None:
        assert self._resolve(["biome", "site_name"]) == "biome"

    def test_middle_priority_beats_biome(self) -> None:
        assert self._resolve(["biome", "plant_functional_type"]) == "plant_functional_type"

    def test_case_insensitive_match_is_preserved(self) -> None:
        # The old rule compared c.lower(); losing that would silently drop a
        # site whose header is capitalised.
        assert self._resolve(["BIOME", "PFT"]) == "PFT"

    def test_none_when_nothing_matches(self) -> None:
        assert self._resolve(["TIMESTAMP", "sap_velocity"]) is None


class TestParseKFeatures:
    """--k_features must reach mlxtend as a type it accepts.

    mlxtend takes an int, a (min, max) tuple, or exactly the strings "best" and
    "parsimonious". argparse declares the flag as str so the keywords work,
    which meant "40" was forwarded verbatim and raised AttributeError -- but
    only inside sfs.fit(), after the cache load and CV split construction had
    already succeeded. A job would burn several minutes before failing.
    """

    def test_numeric_string_becomes_int(self) -> None:
        assert _parse_k_features("40") == 40
        assert isinstance(_parse_k_features("40"), int)

    def test_regression_forty_is_not_left_as_a_string(self) -> None:
        # The exact failure: mlxtend rejects any string that is not one of its
        # two keywords, so a str here is the bug, not a cosmetic detail.
        assert not isinstance(_parse_k_features("40"), str)

    def test_keywords_pass_through_untouched(self) -> None:
        assert _parse_k_features("best") == "best"
        assert _parse_k_features("parsimonious") == "parsimonious"

    def test_range_becomes_tuple(self) -> None:
        assert _parse_k_features("10,40") == (10, 40)

    def test_whitespace_tolerated(self) -> None:
        assert _parse_k_features("  25  ") == 25
        assert _parse_k_features(" 10 , 40 ") == (10, 40)

    def test_unknown_string_is_left_alone_for_mlxtend_to_reject(self) -> None:
        # Silently rewriting a typo into a number would be worse than failing.
        assert _parse_k_features("bset") == "bset"


_EMPTY_CONSTRUCTORS = {"dict", "list", "set", "tuple", "frozenset"}


def _returns_nothing_useful(value) -> bool:
    """True if a ``return`` hands back an empty result instead of a site.

    ``return {}, {}, set()`` -- the shape the DL trainer used -- discards the
    site exactly as ``continue`` does. ``return True`` from a predicate does
    not, and must not trip the gate.
    """
    if value is None:  # bare `return`
        return True
    if isinstance(value, ast.Constant):
        return value.value is None
    if isinstance(value, ast.Dict):
        return not value.keys
    if isinstance(value, (ast.List, ast.Set, ast.Tuple)):
        return all(_returns_nothing_useful(el) for el in value.elts)
    if isinstance(value, ast.Call):
        return (
            isinstance(value.func, ast.Name)
            and value.func.id in _EMPTY_CONSTRUCTORS
            and not value.args
            and not value.keywords
        )
    return False


def _discards_the_site(stmt) -> bool:
    """True if *stmt* abandons the current site rather than processing it."""
    if isinstance(stmt, ast.Continue):
        return True
    if isinstance(stmt, ast.Return):
        return _returns_nothing_useful(stmt.value)
    return False


class TestNoSiteIdPrefixExclusion:
    """The loader must not silently drop sites by name prefix.

    A ``startswith`` guard ending in ``continue`` removed all 16 CZE sites from
    every FFS run between 2026-04-03 and 2026-08-25. It logged at INFO, where the
    column-gate skips log at WARNING, so it never showed up in a warning scan --
    and no plan, deviation note, test, or comment ever recorded why it was there.

    A future exclusion may well be justified. It does not get to be invisible:
    put it in a named registry or a documented CLI filter, where a reader can
    see the site list and the reason, not in an undocumented prefix test.

    Two fixes were not enough either. When this gate covered only the FFS loader
    and the Stage-5 trainer, the same block was still live in six more modules,
    and one of them spelled the skip ``return {}, {}, set()`` rather than
    ``continue`` -- so a detector looking only for ``continue`` would have passed
    it vacuously. Both the module list and the skip shapes are part of the gate.
    """

    @staticmethod
    def _prefix_skips(source: str) -> list:
        """Locate every ``if <expr>.startswith(...)`` that then drops the site.

        The ``startswith`` is searched for anywhere inside the test, not just as
        the whole of it. ``aoa/backfill.py`` spelled its guard
        ``if site_col and df[site_col].iloc[0].startswith("CZE")`` -- a BoolOp,
        not a Call -- and a detector that required ``node.test`` to *be* the call
        reported that file clean while the exclusion was still live in it.
        """
        found = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.If):
                continue
            if not any(_discards_the_site(stmt) for stmt in node.body):
                continue
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "startswith":
                    found.append(ast.dump(sub.args[0]) if sub.args else "<no-arg>")
                    break
        return found

    #: Every module that walks the per-site corpus and could silently drop part
    #: of it. The FFS loader was fixed first; the Stage-5 trainer carried its own
    #: copy of the same guard, so the selection ran on 167 sites including 11 CZE
    #: while the MODEL was fitted on 143 with none of them. Six further copies
    #: surfaced on 2026-08-25, in the DA, prediction, autoML, DL, evaluation and
    #: AOA-backfill paths. One fix was not enough and neither was one test.
    _SRC = Path(data_loader.__file__).parents[1]
    _HPO = _SRC / "hyperparameter_optimization"

    GUARDED = (
        ("forward_selection/data_loader.py", Path(data_loader.__file__)),
        (
            "hyperparameter_optimization/*_ML_spatial_stratified.py",
            _HPO / "test_hyperparameter_tuning_ML_spatial_stratified.py",
        ),
        (
            "hyperparameter_optimization/*_ML_spatial_stratified_DA.py",
            _HPO / "test_hyperparameter_tuning_ML_spatial_stratified_DA.py",
        ),
        (
            "hyperparameter_optimization/*_ML_spatial_stratified_prediction.py",
            _HPO / "test_hyperparameter_tuning_ML_spatial_stratified_prediction.py",
        ),
        (
            "hyperparameter_optimization/*_autoML_spatial_stratified.py",
            _HPO / "test_hyperparameter_tuning_autoML_spatial_stratified.py",
        ),
        (
            "hyperparameter_optimization/*_DL_spatial_stratified.py",
            _HPO / "test_hyperparameter_tuning_DL_spatial_stratified.py",
        ),
        (
            "hyperparameter_optimization/evaluate_trained_model.py",
            _HPO / "evaluate_trained_model.py",
        ),
        ("aoa/backfill.py", _SRC / "aoa" / "backfill.py"),
    )

    @pytest.mark.parametrize("label,path", GUARDED, ids=[g[0] for g in GUARDED])
    def test_carries_no_prefix_exclusion(self, label: str, path: Path) -> None:
        if not path.exists():
            pytest.skip(f"{label} not present in this checkout")
        source = path.read_text(encoding="utf-8")
        assert self._prefix_skips(source) == [], (  # nosec B101
            f"{label} drops sites by name prefix. If an exclusion is warranted, "
            "put it in a named registry or a documented CLI filter where the site "
            "list and the reason are visible."
        )

    def test_every_guarded_module_actually_exists(self) -> None:
        """The gate is an allowlist; a typo in a path would skip it silently."""
        missing = [label for label, path in self.GUARDED if not path.exists()]
        assert missing == [], f"GUARDED names files that are not here: {missing}"  # nosec B101

    def test_detector_fires_on_the_removed_block(self) -> None:
        """Negative control: without this, the test above could pass vacuously."""
        removed = textwrap.dedent(
            """
            for site_id in sites:
                if site_id.startswith("CZE"):
                    logger.info("Skipping site %s", site_id)
                    continue
                keep(site_id)
            """
        )
        assert len(self._prefix_skips(removed)) == 1  # nosec B101

    def test_detector_fires_on_the_return_shaped_skip(self) -> None:
        """The DL trainer spelled it this way. A continue-only detector missed it."""
        removed = textwrap.dedent(
            """
            def load(site_id):
                if site_id.startswith("CZE"):
                    logging.info("Skipping site")
                    return {}, {}, set()
                return build(site_id)
            """
        )
        assert len(self._prefix_skips(removed)) == 1  # nosec B101

    def test_detector_fires_on_a_guard_hidden_behind_and(self) -> None:
        """The AOA backfill spelling: the call is buried in a BoolOp.

        A detector that inspects only ``node.test`` itself reports this clean.
        """
        removed = textwrap.dedent(
            """
            for csv_path in paths:
                site_col = "site_name" if "site_name" in df.columns else None
                if site_col and df[site_col].iloc[0].startswith("CZE"):
                    logger.info("excluded")
                    continue
                keep(csv_path)
            """
        )
        assert len(self._prefix_skips(removed)) == 1  # nosec B101

    def test_detector_ignores_a_startswith_that_does_not_skip(self) -> None:
        """A startswith used for anything but a silent skip must not trip it."""
        benign = textwrap.dedent(
            """
            for site_id in sites:
                if site_id.startswith("CZE"):
                    country = "Czechia"
                keep(site_id)
            """
        )
        assert self._prefix_skips(benign) == []  # nosec B101

    def test_detector_ignores_a_predicate_returning_a_real_value(self) -> None:
        """A return of True answers a question; it does not discard a site."""
        predicate = textwrap.dedent(
            """
            def is_czech(site_id):
                if site_id.startswith("CZE"):
                    return True
                return False
            """
        )
        assert self._prefix_skips(predicate) == []  # nosec B101
