"""The SAPFLUXNET flag vocabulary, and the one token that is never applied.

Acceptance criterion 15(a).  `OUT_WARN` is not "not yet supported" -- it is
ruled out permanently (decision D2), because this pipeline runs its own outlier
detection and applying the archive's would stack two different definitions of
"outlier" on the same series.  A constant is easy to edit, so the rule is
pinned by a test that names the token explicitly.
"""

import pytest

from src.qc_flags import vocabulary as v


class TestActionableTokens:
    def test_range_warn_is_the_only_actionable_token(self):
        # Criterion 15(a): exact equality, not membership. If a token is added
        # to the actionable set this fails, which is the point.
        assert frozenset({"RANGE_WARN"}) == v.ACTIONABLE_TOKENS

    def test_out_warn_is_never_actionable(self):
        assert "OUT_WARN" in v.NEVER_ACTIONABLE
        assert "OUT_WARN" not in v.ACTIONABLE_TOKENS

    def test_never_actionable_and_actionable_are_disjoint(self):
        assert not (v.ACTIONABLE_TOKENS & v.NEVER_ACTIONABLE)

    def test_every_known_token_is_classified(self):
        # No token may sit outside all three sets -- an unclassified token would
        # be silently ignored, which is how OUT_WARN would creep back in.
        classified = v.ACTIONABLE_TOKENS | v.NEVER_ACTIONABLE | v.NO_OP_TOKENS
        assert classified == v.KNOWN_TOKENS


class TestIsActionable:
    def test_bare_range_warn(self):
        assert v.is_actionable("RANGE_WARN") is True

    def test_bare_out_warn_is_not(self):
        assert v.is_actionable("OUT_WARN") is False

    def test_composite_containing_range_warn(self):
        # Tokens arrive `;`-joined: "NA_PRESENT;RANGE_WARN".
        assert v.is_actionable("NA_PRESENT;RANGE_WARN") is True

    def test_composite_of_only_non_actionable_tokens(self):
        assert v.is_actionable("NA_PRESENT;OUT_WARN") is False

    @pytest.mark.parametrize("empty", ["", None, float("nan")])
    def test_empty_values_are_not_actionable(self, empty):
        assert v.is_actionable(empty) is False

    def test_range_removed_is_not_range_warn(self):
        # Substring matching would make RANGE_REMOVED look actionable. It is
        # not: the value is already gone, so acting on it is a no-op at best.
        assert v.is_actionable("RANGE_REMOVED") is False

    def test_unknown_token_is_not_actionable(self):
        assert v.is_actionable("SOMETHING_NEW") is False


class TestSeriesLevel:
    def test_actionable_mask_over_a_series(self):
        import pandas as pd

        s = pd.Series(["RANGE_WARN", "OUT_WARN", "NA_PRESENT;RANGE_WARN", None, "RANGE_REMOVED"])
        mask = v.actionable_mask(s)
        assert list(mask) == [True, False, True, False, False]

    def test_actionable_mask_preserves_index(self):
        import pandas as pd

        s = pd.Series(["RANGE_WARN", "OUT_WARN"], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
        mask = v.actionable_mask(s)
        assert list(mask.index) == list(s.index)
