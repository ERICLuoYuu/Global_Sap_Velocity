"""One design-matrix column order for the FFS cache, whatever the interpreter's hash seed.

``load_and_cache_features`` built its column list as ``list(set(base + ADDITIONAL_FEATURES))``. A set
of strings iterates in an order drawn from the process's hash seed, so each cache build laid the design
matrix out differently. That is not cosmetic. Forward selection breaks ties by column position, and so
do XGBoost's column subsampling and split finding, so the same rows and the same random seeds could
select a different feature set. The trainer's job pinned PYTHONHASHSEED; the cache and selection jobs,
which build and read that matrix, did not.

Each order comes from a fresh interpreter, because the hash seed is fixed when the interpreter starts
and cannot be varied inside one process.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forward_selection.feature_registry import PFT_ONEHOT_COLS

REPO = Path(__file__).resolve().parents[3]

#: The loader's base column list, in its declared order, minus the target and ``pft``.
BASE = [
    "sw_in", "ws", "precip", "ta", "ta_max", "ta_min", "vpd", "vpd_max", "vpd_min", "ext_rad",
    "ppfd_in", "canopy_height", "elevation", "LAI", "prcip/PET", "soil_temperature_level_1", "day_length",
]  # fmt: skip
TIME = ["Day sin", "Day cos", "Week sin", "Week cos", "Month sin", "Month cos", "Year sin", "Year cos"]
#: Stands in for ADDITIONAL_FEATURES. Declared neither alphabetically nor in base order, and naming a
#: base column a second time, so an order that sorts, or that lets the later duplicate win, fails too.
EXTRA = ["zz_extra", "precip", "aa_extra", "mm_extra"]
HASH_SEEDS = ("0", "1", "2", "3")

#: Run in a fresh interpreter per hash seed. The training helpers are replaced by stand-ins, as in
#: test_ffs_feature_contract, so only the loader's own column handling is exercised.
_BUILD = """
import json
import sys
from pathlib import Path

import numpy as np

from src.forward_selection import data_loader

daily, cache = Path(sys.argv[1]), Path(sys.argv[2])
time_names, extra = json.loads(sys.argv[3]), json.loads(sys.argv[4])


def feature_engineering(df, time_scale="daily", verbose=False, contract="site"):
    return df, []


def add_time_features(df, datetime_column=None):
    return df.assign(**{name: 0.5 for name in time_names})


def create_spatial_groups(lat, lon, method="default", lat_grid_size=0.05, lon_grid_size=0.05, **kwargs):
    return np.arange(len(lat)), {}


data_loader._import_training_helpers = lambda: (feature_engineering, add_time_features, create_spatial_groups)
data_loader.ADDITIONAL_FEATURES = extra
built = data_loader.load_and_cache_features(daily, cache, contract="servable")
print(json.dumps(built["feature_names"]))
"""


def _tree(root):
    """A servable merged config directory with two ``daily/`` site files carrying every column."""
    config_dir = root / "fullyear_fullday_tx"
    daily = config_dir / "daily"
    daily.mkdir(parents=True)
    (config_dir / "RUN_PROVENANCE.json").write_text(json.dumps({"feature_contract": {"name": "servable"}}))
    rng = np.random.RandomState(0)
    days = pd.date_range("2020-06-01", periods=6, freq="D")
    columns = list(dict.fromkeys(BASE + EXTRA + ["sap_velocity"]))
    for i in range(2):
        frame = pd.DataFrame({name: rng.uniform(0.1, 1.0, len(days)) for name in columns})
        frame.insert(0, "TIMESTAMP", days)
        frame["solar_TIMESTAMP"] = days
        frame["site_name"] = "SITE_%d" % i
        frame["lat"] = 50.0 + i
        frame["lon"] = 8.0 + i
        frame["pft"] = "ENF"
        frame.to_csv(daily / ("SITE_%d_daily.csv" % i), index=False)
    return daily


def _feature_names(daily, cache, hash_seed):
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH", "")) if p)
    proc = subprocess.run(
        [sys.executable, "-c", _BUILD, str(daily), str(cache), json.dumps(TIME), json.dumps(EXTRA)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, "cache build failed under PYTHONHASHSEED=%s: %s" % (hash_seed, proc.stderr[-3000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def names_by_seed(tmp_path_factory):
    root = tmp_path_factory.mktemp("column_order")
    daily = _tree(root)
    return {seed: _feature_names(daily, root / ("cache_%s.npz" % seed), seed) for seed in HASH_SEEDS}


class TestOneColumnOrder:
    def test_every_hash_seed_lays_the_matrix_out_the_same_way(self, names_by_seed):
        first = names_by_seed[HASH_SEEDS[0]]
        moved = {seed: names[:6] for seed, names in names_by_seed.items() if names != first}
        assert not moved, "column order moved with the hash seed: seed %s starts %s, but %s" % (
            HASH_SEEDS[0],
            first[:6],
            moved,
        )

    def test_the_order_is_the_order_the_loader_declares(self, names_by_seed):
        names = names_by_seed[HASH_SEEDS[0]]
        declared = list(dict.fromkeys(BASE + EXTRA)) + TIME
        assert [n for n in names if n not in PFT_ONEHOT_COLS] == declared

    def test_a_column_named_twice_is_kept_once_where_it_first_appears(self, names_by_seed):
        names = names_by_seed[HASH_SEEDS[0]]
        assert names.count("precip") == 1
        assert names.index("precip") == BASE.index("precip")

    def test_the_pft_indicators_follow_in_registry_order(self, names_by_seed):
        names = names_by_seed[HASH_SEEDS[0]]
        assert [n for n in names if n in PFT_ONEHOT_COLS] == list(PFT_ONEHOT_COLS)


class TestTheSweepStagesShareOneHashSeed:
    def test_sourcing_the_sweep_config_exports_a_fixed_seed(self):
        """Cache, selection and training all source sweep5_config.sh; only the trainer's job pinned it."""
        env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
        proc = subprocess.run(
            ["bash", "-c", "source ./sweep5_config.sh >/dev/null && printenv PYTHONHASHSEED"],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, "sweep5_config.sh exports no PYTHONHASHSEED: %s" % proc.stderr
        assert proc.stdout.strip() == "0"
