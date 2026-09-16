"""The ERA5 daily total is loaded under every contract, and is a FEATURE only under ``site``.

Feature engineering's ``cwd`` group reads the metres column, so it cannot simply go unread. It is
dropped on the way into the design matrix instead -- the same treatment the Saxton-Rawls parameters
get -- and only when ``precip`` has become that same total, which is the servable contract.
"""

import numpy as np
import pytest

from src.feature_contract.definitions import ERA5_DAILY_PRECIP_COL
from src.forward_selection.data_loader import _intermediate_columns, load_cache
from src.forward_selection.feature_registry import (
    ADDITIONAL_FEATURES,
    CANDIDATE_FEATURES,
    INTERMEDIATE_ONLY,
)


def _cache(tmp_path, names, contract):
    """A feature cache shaped exactly as build_feature_cache writes one."""
    path = tmp_path / "feature_cache.npz"
    np.savez(
        path,
        X=np.zeros((3, len(names)), dtype=np.float32),
        y=np.zeros(3, dtype=np.float32),
        groups=np.zeros(3, dtype=np.int64),
        pfts_encoded=np.zeros(3, dtype=np.int64),
        feature_names=np.array(names),
        pft_categories=np.array(["ENF"]),
        feature_contract=np.array(contract),
    )
    return path


class TestLoadedEitherWayFeatureOnlyUnderSite:
    def test_it_is_loaded_under_both_contracts(self):
        """Dropping it from the load would take the cwd feature with it."""
        assert ERA5_DAILY_PRECIP_COL in ADDITIONAL_FEATURES

    def test_servable_keeps_it_out_of_the_design_matrix(self):
        assert ERA5_DAILY_PRECIP_COL in _intermediate_columns("servable")

    def test_site_keeps_it_as_a_feature(self):
        assert ERA5_DAILY_PRECIP_COL not in _intermediate_columns("site")

    def test_the_existing_intermediates_survive_under_both(self):
        for contract in ("site", "servable"):
            assert set(INTERMEDIATE_ONLY) <= set(_intermediate_columns(contract))

    def test_it_stays_a_registered_candidate(self):
        """The registry is contract-free; the loader decides. Under site it is still selectable."""
        assert ERA5_DAILY_PRECIP_COL in CANDIDATE_FEATURES


class TestACacheBuiltBeforeThisRule:
    def test_a_servable_cache_naming_both_is_refused(self, tmp_path):
        path = _cache(tmp_path, ["precip", ERA5_DAILY_PRECIP_COL], "servable")
        with pytest.raises(ValueError, match=ERA5_DAILY_PRECIP_COL):
            load_cache(path)

    def test_the_same_cache_loads_under_site(self, tmp_path):
        path = _cache(tmp_path, ["precip", ERA5_DAILY_PRECIP_COL], "site")
        assert load_cache(path)["feature_contract"] == "site"

    def test_a_clean_servable_cache_loads(self, tmp_path):
        path = _cache(tmp_path, ["precip", "ta"], "servable")
        assert load_cache(path)["feature_contract"] == "servable"
