"""Tests for the DL trainer's parity path (--data_dir / --selected_features /
--IS_TRANSFORM) and the ScaledTargetPipeline that carries the target transform.

Each test is written so that reverting the change it guards makes it FAIL, not
merely stop covering something -- the linter hook in this repo has quietly
rewritten code meaning before, so a gate that cannot fail is worse than no gate.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path("/scratch/tmp/yluo2/gsv")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.preprocessing import StandardScaler  # noqa: E402
from src.feature_contract.definitions import (  # noqa: E402
    ERA5_DAILY_PRECIP_COL,
    RETIRED_SWC_FEATURES,
    assert_dataset_contract,
)
from src.hyperparameter_optimization.target_transformer import (  # noqa: E402
    ScaledTargetPipeline,
)

DATA_DIR = ROOT / "outputs" / "processed_data" / "sapwood_combined" / "merged" / "daily"
FFS_GRID = ROOT / "configs" / "selected_features_ffs_grid005_cze.json"
FFS_SITE = ROOT / "configs" / "selected_features_ffs_site_cze.json"
#: configs/ is not tracked, so a checkout without it skips the tests that read these two selections.
HAVE_FFS_CONFIGS = FFS_GRID.exists() and FFS_SITE.exists()
PFT_ONEHOTS = ["MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"]


# ───────────────────────────────────────────────── ScaledTargetPipeline
class TestScaledTargetPipeline:
    @staticmethod
    def _skewed(n=500, seed=0):
        rng = np.random.default_rng(seed)
        return rng.lognormal(mean=1.0, sigma=1.2, size=(n, 1))

    def test_round_trip_log1p_recovers_original(self):
        y = self._skewed()
        p = ScaledTargetPipeline("log1p")
        back = p.inverse_transform(p.fit_transform(y))
        assert np.allclose(back, y, rtol=1e-8, atol=1e-8)  # nosec B101

    def test_round_trip_none_recovers_original(self):
        y = self._skewed()
        p = ScaledTargetPipeline("none")
        back = p.inverse_transform(p.fit_transform(y))
        assert np.allclose(back, y, rtol=1e-8, atol=1e-8)  # nosec B101

    def test_shape_is_preserved(self):
        """TargetTransformer flattens; the pipeline must restore the caller's shape.

        Every DL call site passes 2-D and immediately reshapes the result, so a
        silently flattened return would broadcast wrongly rather than raise.
        """
        y = self._skewed(n=120)
        p = ScaledTargetPipeline("log1p")
        assert p.fit_transform(y).shape == y.shape  # nosec B101
        assert p.inverse_transform(p.transform(y)).shape == y.shape  # nosec B101

    def test_none_is_exactly_a_standard_scaler(self):
        """method='none' must be a drop-in for the StandardScaler it replaces."""
        y = self._skewed()
        mine = ScaledTargetPipeline("none").fit_transform(y)
        theirs = StandardScaler().fit_transform(y)
        assert np.allclose(mine, theirs)  # nosec B101

    def test_log1p_actually_changes_the_objective(self):
        """Guards against the transform being wired up but inert.

        On a lognormal target, standardising log1p(y) leaves far less skew than
        standardising y.  If someone re-points this at a bare StandardScaler the
        two skews become equal and this fails.
        """
        from scipy.stats import skew

        y = self._skewed()
        skew_plain = abs(float(skew(ScaledTargetPipeline("none").fit_transform(y).ravel())))
        skew_log = abs(float(skew(ScaledTargetPipeline("log1p").fit_transform(y).ravel())))
        assert skew_log < skew_plain / 2, (  # nosec B101
            f"log1p skew {skew_log:.3f} not much below plain {skew_plain:.3f}"
        )

    def test_transform_after_fit_matches_fit_transform(self):
        y = self._skewed()
        p = ScaledTargetPipeline("log1p").fit(y)
        assert np.allclose(p.transform(y), ScaledTargetPipeline("log1p").fit_transform(y))  # nosec B101

    def test_is_picklable(self):
        """The trainer joblib.dumps the label scaler every fold."""
        import pickle  # nosec B403

        y = self._skewed()
        p = ScaledTargetPipeline("log1p").fit(y)
        rt = pickle.loads(pickle.dumps(p))  # nosec B301
        assert np.allclose(rt.transform(y), p.transform(y))  # nosec B101


# ───────────────────────────────────────────────── CLI parity
class TestParityCLI:
    @staticmethod
    def _parse(argv):
        import src.hyperparameter_optimization.test_hyperparameter_tuning_DL_spatial_stratified as dl

        old = sys.argv
        try:
            sys.argv = ["prog"] + argv
            return dl.parse_args()
        finally:
            sys.argv = old

    def test_new_arguments_exist(self):
        a = self._parse(["--hyperparameters", "x.json"])
        for name in ("data_dir", "selected_features", "IS_TRANSFORM", "TRANSFORM_METHOD", "grid_size"):
            assert hasattr(a, name), f"--{name} missing from the DL parser"  # nosec B101

    def test_boolean_flags_accept_false(self):
        """Under the original type=bool, '--IS_CV False' evaluated True.

        bool('False') is True, so every boolean flag was stuck on. Passing False
        must now actually mean False, or the job scripts silently lie.
        """
        a = self._parse(["--hyperparameters", "x.json", "--IS_CV", "False", "--IS_STRATIFIED", "False"])
        assert a.IS_CV is False  # nosec B101
        assert a.IS_STRATIFIED is False  # nosec B101

    def test_boolean_flags_still_accept_true(self):
        a = self._parse(["--hyperparameters", "x.json", "--IS_CV", "True"])
        assert a.IS_CV is True  # nosec B101

    def test_defaults_preserve_legacy_behaviour(self):
        a = self._parse(["--hyperparameters", "x.json"])
        assert a.data_dir is None  # nosec B101
        assert a.selected_features is None  # nosec B101
        assert a.IS_TRANSFORM is False, (  # nosec B101
            "legacy runs must not silently gain a target transform"
        )

    def test_the_feature_contract_defaults_to_servable(self):
        """As in the ML trainer and every other consumer: a tree from today's merge needs no flag."""
        assert self._parse(["--hyperparameters", "x.json"]).FEATURE_CONTRACT == "servable"  # nosec B101

    def test_site_is_still_accepted(self):
        a = self._parse(["--hyperparameters", "x.json", "--FEATURE_CONTRACT", "site"])
        assert a.FEATURE_CONTRACT == "site"  # nosec B101

    def test_an_unknown_contract_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            self._parse(["--hyperparameters", "x.json", "--FEATURE_CONTRACT", "grid"])


# ───────────────────────────────────────────────── the loader
# ───────────────────────────────────────────────── frozen feature list
class TestFrozenFeatureList:
    """--selected_features is refused where it cannot mean what it says, as in the ML trainer."""

    @staticmethod
    def _write(tmp_path, names):
        path = tmp_path / "features.json"
        path.write_text(json.dumps({"selected_features": names}))
        return path

    @staticmethod
    def _load(path, contract):
        import src.hyperparameter_optimization.test_hyperparameter_tuning_DL_spatial_stratified as dl

        return dl.load_frozen_feature_list(path, contract)

    def test_a_clean_list_comes_back_as_written(self, tmp_path):
        names = ["sw_in", "vpd", "precip"] + PFT_ONEHOTS
        assert self._load(self._write(tmp_path, names), "servable") == names  # nosec B101

    def test_a_bare_list_is_accepted_too(self, tmp_path):
        path = tmp_path / "features.json"
        path.write_text(json.dumps(["sw_in", "vpd"]))
        assert self._load(path, "servable") == ["sw_in", "vpd"]  # nosec B101

    def test_a_retired_soil_moisture_name_is_refused(self, tmp_path):
        """The bare column is still in the merged CSVs: it would load, and train on the retired definition."""
        with pytest.raises(ValueError, match="volumetric_soil_water_layer_1"):
            self._load(self._write(tmp_path, ["sw_in", "volumetric_soil_water_layer_1"]), "site")

    def test_the_refusal_names_the_file(self, tmp_path):
        with pytest.raises(ValueError, match="features.json"):
            self._load(self._write(tmp_path, ["swc_layer3_norm"]), "site")

    def test_the_era5_metres_column_is_refused_under_servable(self, tmp_path):
        with pytest.raises(ValueError, match=ERA5_DAILY_PRECIP_COL):
            self._load(self._write(tmp_path, ["precip", ERA5_DAILY_PRECIP_COL]), "servable")

    def test_the_same_list_is_accepted_under_site(self, tmp_path):
        """Under site it is the reanalysis beside the rain gauge: two measurements, both allowed."""
        names = ["precip", ERA5_DAILY_PRECIP_COL]
        assert self._load(self._write(tmp_path, names), "site") == names  # nosec B101

    def test_the_raw_pft_column_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="one-hot"):
            self._load(self._write(tmp_path, ["sw_in", "pft"]), "servable")

    def test_a_json_that_holds_no_list_is_refused(self, tmp_path):
        path = tmp_path / "features.json"
        path.write_text(json.dumps({"selected_features": "sw_in"}))
        with pytest.raises(ValueError, match="list"):
            self._load(path, "servable")

    @pytest.mark.skipif(not HAVE_FFS_CONFIGS, reason="configs/ selections not present")
    @pytest.mark.parametrize("config", [FFS_GRID, FFS_SITE])
    def test_the_historical_cze_selections_are_refused(self, config):
        """Both predate the 2026-09-14 retirement -- the reason the loader test below strips them."""
        with pytest.raises(ValueError, match="retired"):
            self._load(config, "site")


# ───────────────────────────────────────────────── contract reaches feature engineering
class TestParityContract:
    """process_date_file engineers under the run's contract, not under the library default."""

    @staticmethod
    def _site_csv(tmp_path):
        days = pd.date_range("2020-06-01", periods=8, freq="D")
        frame = pd.DataFrame(
            {
                "TIMESTAMP": days,
                "solar_TIMESTAMP": days,
                "site_name": "SITE_A",
                "lat": 50.0,
                "lon": 8.0,
                "pft": "ENF",
                "sap_velocity": np.linspace(1.0, 2.0, len(days)),
                "sw_in": np.linspace(100.0, 200.0, len(days)),
            }
        )
        path = tmp_path / "SITE_A_daily.csv"
        frame.to_csv(path, index=False)
        return path

    @staticmethod
    def _args(**extra):
        args = {
            "PARITY": True,
            "USED_COLS": ["sap_velocity", "sw_in"],
            "ALL_POSSIBLE_PFT_TYPES": PFT_ONEHOTS,
            "INPUT_WIDTH": 2,
            "SHIFT": 1,
            "ALL_PFT_TYPES": set(),
            "TUNING_WORKERS": 1,
            "TIME_SCALE": "daily",
        }
        args.update(extra)
        return args

    @pytest.mark.parametrize("contract", ["servable", "site"])
    def test_feature_engineering_receives_the_run_contract(self, tmp_path, monkeypatch, contract):
        import src.hyperparameter_optimization.test_hyperparameter_tuning_DL_spatial_stratified as dl

        seen = []

        def spy(df, time_scale="daily", verbose=False, contract=None):
            seen.append(contract)
            return df, []

        monkeypatch.setattr(dl, "apply_all_feature_engineering", spy)
        dl.process_date_file(self._site_csv(tmp_path), self._args(FEATURE_CONTRACT=contract))
        assert seen == [contract]  # nosec B101

    def test_a_parity_run_without_a_contract_stops(self, tmp_path):
        """process_date_file turns every error inside its try into a skipped site; a lost contract must not."""
        import src.hyperparameter_optimization.test_hyperparameter_tuning_DL_spatial_stratified as dl

        with pytest.raises(KeyError, match="FEATURE_CONTRACT"):
            dl.process_date_file(self._site_csv(tmp_path), self._args())

    def test_the_legacy_path_needs_no_contract(self, tmp_path, monkeypatch):
        """Without --selected_features nothing is engineered, so there is nothing for a contract to reach."""
        import src.hyperparameter_optimization.test_hyperparameter_tuning_DL_spatial_stratified as dl

        called = []

        def spy(df, *args, **kwargs):
            called.append(kwargs)
            return df, []

        monkeypatch.setattr(dl, "apply_all_feature_engineering", spy)
        dl.process_date_file(self._site_csv(tmp_path), self._args(PARITY=False))
        assert called == []  # nosec B101


@pytest.mark.skipif(not DATA_DIR.exists(), reason="sapwood_combined corpus not present")
class TestParityLoader:
    @staticmethod
    def _site_files():
        files = sorted(DATA_DIR.glob("*daily.csv"))
        return [f for f in files if "all_biomes_merged" not in f.name]

    def test_all_biomes_aggregate_is_present_and_excluded(self):
        """The trap this filter exists for must actually be in the directory.

        If the aggregate ever stops being written, this test should fail loudly
        rather than pass vacuously and let the filter rot untested.
        """
        raw = sorted(DATA_DIR.glob("*daily.csv"))
        agg = [f for f in raw if "all_biomes_merged" in f.name]
        assert agg, (  # nosec B101
            "no all_biomes_merged file found - the filter is now untested, re-check the trap"
        )
        assert all("all_biomes_merged" not in f.name for f in self._site_files())  # nosec B101
        assert len(self._site_files()) == len(raw) - len(agg)  # nosec B101

    @pytest.mark.skipif(not HAVE_FFS_CONFIGS, reason="configs/ selections not present")
    @pytest.mark.parametrize("config", [FFS_GRID, FFS_SITE])
    def test_parity_path_resolves_every_ffs_column_on_a_real_site(self, config):
        """The whole point of calling apply_all_feature_engineering in the loader.

        Without it the engineered FFS columns (height_x_vpd, priestley_taylor_pet,
        vpd_x_swc ...) are absent and every site fails the gate.
        """
        import src.hyperparameter_optimization.test_hyperparameter_tuning_DL_spatial_stratified as dl

        # DATA_DIR is a legacy tree, so the parity path engineers it under site. Were it ever rebuilt
        # servable, this line says so -- rather than a puzzling "accepted no site" further down.
        assert_dataset_contract(DATA_DIR.parent, "site")
        feats = json.loads(config.read_text())["selected_features"]
        # Both selections predate the 2026-09-14 soil-moisture retirement, and the trainer now refuses
        # them whole (TestFrozenFeatureList). What this test guards is the engineering of the rest, so
        # the retired names are removed -- and must still be there to remove, or the filter has outlived
        # its reason and should go.
        retired = sorted(set(feats) & set(RETIRED_SWC_FEATURES))
        assert retired, f"{config.name} names no retired feature any more: drop this filter"  # nosec B101
        used = sorted(f for f in feats if f not in RETIRED_SWC_FEATURES)
        if "sap_velocity" not in used:
            used = ["sap_velocity"] + used

        args = {
            "PARITY": True,
            "FEATURE_CONTRACT": "site",
            "USED_COLS": used,
            "ALL_POSSIBLE_PFT_TYPES": ["MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"],
            "INPUT_WIDTH": 2,
            "SHIFT": 1,
            "ALL_PFT_TYPES": set(),
            "TUNING_WORKERS": 1,
            "TIME_SCALE": "daily",
        }

        accepted = 0
        for f in self._site_files()[:40]:
            site_data, site_info, _ = dl.process_date_file(f, args)
            if site_data:
                accepted += 1
        assert accepted > 0, f"parity loader accepted no site at all for {config.name}"  # nosec B101

    def test_parity_path_is_off_by_default(self):
        """PARITY absent must leave the legacy branch running, untouched."""
        import src.hyperparameter_optimization.test_hyperparameter_tuning_DL_spatial_stratified as dl

        legacy_cols = [
            "sap_velocity",
            "sw_in",
            "ppfd_in",
            "ext_rad",
            "ws",
            "ta",
            "vpd",
            "LAI",
            "prcip/PET",
            "pft",
            "canopy_height",
            "elevation",
            "latitude",
            "longitude",
            "Day sin",
            "Year sin",
        ]
        args = {
            "USED_COLS": legacy_cols,
            "ALL_POSSIBLE_PFT_TYPES": ["MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"],
            "INPUT_WIDTH": 2,
            "SHIFT": 1,
            "ALL_PFT_TYPES": set(),
            "TUNING_WORKERS": 1,
            "TIME_SCALE": "daily",
        }
        accepted = 0
        for f in self._site_files()[:40]:
            site_data, _, _ = dl.process_date_file(f, args)
            if site_data:
                accepted += 1
        assert accepted > 0, "legacy loader path regressed - it accepted no site"  # nosec B101


# ───────────────────────────────────────────────── configs
@pytest.mark.skipif(not HAVE_FFS_CONFIGS, reason="configs/ selections not present")
class TestFFSConfigs:
    @pytest.mark.parametrize("config,method", [(FFS_GRID, "grid"), (FFS_SITE, "default")])
    def test_config_is_well_formed(self, config, method):
        d = json.loads(config.read_text())
        assert d["n_features"] == 47
        assert len(d["selected_features"]) == 47
        assert d["matched_spatial_split_method"] == method
        assert "pft" not in d["selected_features"], "must name one-hot columns, not raw 'pft'"
        for c in ("MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"):
            assert c in d["selected_features"], f"PFT one-hot {c} missing"

    def test_the_two_configs_actually_differ(self):
        """Guards against both jobs accidentally training the same thing."""
        a = set(json.loads(FFS_GRID.read_text())["selected_features"])
        b = set(json.loads(FFS_SITE.read_text())["selected_features"])
        assert a != b, "grid and site FFS selections are identical - wrong source files"
