"""The QC configuration is data, not literals scattered through two analyzers.

The threshold in particular: nobody could confirm the FLUXNET wording for QC
levels 0/1/2 from documentation (fluxnet.org 403, the Pastorello paper behind
an auth redirect, the ONEFlux README silent).  What the data shows is that
level 1 is 10-20x smoother than level 0 at the same site -- the signature of a
gap-fill, not a measurement.  The default follows the evidence; the config
exists so the decision can be revisited without touching the algorithm.
"""

import dataclasses

import pytest

from src.qc_flags.config import DERIVED_FROM, QcFlagConfig


class TestDefaults:
    def test_threshold_defaults_to_one(self):
        # Decision D1: keep only level 0.
        assert QcFlagConfig().qc_threshold == 1

    def test_actionable_tokens_default_to_range_warn_only(self):
        assert QcFlagConfig().actionable_tokens == frozenset({"RANGE_WARN"})

    def test_is_frozen(self):
        config = QcFlagConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.qc_threshold = 2

    def test_threshold_is_configurable(self):
        assert QcFlagConfig(qc_threshold=2).qc_threshold == 2

    def test_a_threshold_of_zero_is_rejected(self):
        # qc >= 0 masks everything, including the level the data calls good.
        # Almost certainly a typo for "keep only 0", so refuse it rather than
        # silently blank the corpus.
        with pytest.raises(ValueError):
            QcFlagConfig(qc_threshold=0)

    def test_a_negative_threshold_is_rejected(self):
        with pytest.raises(ValueError):
            QcFlagConfig(qc_threshold=-1)


class TestDerivedVariableMap:
    def test_ppfd_in_comes_from_sw_in(self):
        assert DERIVED_FROM["ppfd_in"] == ("sw_in",)

    def test_rh_comes_from_ta_and_vpd(self):
        assert set(DERIVED_FROM["rh"]) == {"ta", "vpd"}

    def test_no_variable_is_derived_from_itself(self):
        for derived, sources in DERIVED_FROM.items():
            assert derived not in sources

    def test_the_map_is_acyclic(self):
        # A cycle would make propagation order-dependent.
        for derived, sources in DERIVED_FROM.items():
            for source in sources:
                assert derived not in DERIVED_FROM.get(source, ())

    def test_config_carries_the_map_by_default(self):
        assert QcFlagConfig().derived_from == DERIVED_FROM
