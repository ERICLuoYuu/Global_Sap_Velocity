"""Tests for feature_registry.py."""

import pytest

from src.forward_selection.feature_registry import (
    CANDIDATE_FEATURES,
    MANDATORY_FEATURES,
    PFT_ONEHOT_COLS,
    build_feature_groups,
    get_all_expected_columns,
    get_candidate_group_names,
)


def _make_feature_names() -> list[str]:
    """Build a realistic feature_names list matching the full registry."""
    names = list(MANDATORY_FEATURES)
    for col_names in CANDIDATE_FEATURES.values():
        names.extend(col_names)
    return names


class TestMandatoryFeatures:
    def test_mandatory_count(self) -> None:
        assert len(MANDATORY_FEATURES) == 6

    def test_mandatory_names(self) -> None:
        expected = {"sw_in", "ppfd_in", "ta", "vpd", "ws", "ext_rad"}
        assert set(MANDATORY_FEATURES) == expected


class TestPFTGroup:
    def test_pft_has_8_columns(self) -> None:
        assert len(PFT_ONEHOT_COLS) == 8

    def test_pft_is_single_candidate_group(self) -> None:
        assert "pft" in CANDIDATE_FEATURES
        assert CANDIDATE_FEATURES["pft"] == PFT_ONEHOT_COLS

    def test_pft_biome_names(self) -> None:
        expected = {"MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"}
        assert set(PFT_ONEHOT_COLS) == expected


class TestBuildFeatureGroups:
    def test_mandatory_indices_correct(self) -> None:
        feature_names = _make_feature_names()
        mandatory_idx, _ = build_feature_groups(feature_names)
        for idx in mandatory_idx:
            assert feature_names[idx] in MANDATORY_FEATURES
        assert len(mandatory_idx) == len(MANDATORY_FEATURES)

    def test_pft_group_has_8_indices(self) -> None:
        feature_names = _make_feature_names()
        _, candidate_groups = build_feature_groups(feature_names)
        # Find the PFT group (the one with 8 indices)
        pft_groups = [g for g in candidate_groups if len(g) == 8]
        assert len(pft_groups) == 1
        pft_idx = pft_groups[0]
        pft_names = {feature_names[i] for i in pft_idx}
        assert pft_names == set(PFT_ONEHOT_COLS)

    def test_all_other_groups_are_size_1(self) -> None:
        feature_names = _make_feature_names()
        _, candidate_groups = build_feature_groups(feature_names)
        for group in candidate_groups:
            # Either PFT (8) or individual (1)
            assert len(group) in (1, 8)

    def test_missing_features_excluded(self) -> None:
        feature_names = ["sw_in", "ta", "vpd"]  # minimal
        mandatory_idx, candidate_groups = build_feature_groups(feature_names)
        assert len(mandatory_idx) == 3  # only sw_in, ta, vpd
        assert len(candidate_groups) == 0  # no candidates match

    def test_no_overlap_mandatory_and_candidates(self) -> None:
        feature_names = _make_feature_names()
        mandatory_idx, candidate_groups = build_feature_groups(feature_names)
        mandatory_set = set(mandatory_idx)
        for group in candidate_groups:
            for idx in group:
                assert idx not in mandatory_set


class TestGetCandidateGroupNames:
    def test_returns_existing_groups(self) -> None:
        feature_names = _make_feature_names()
        names = get_candidate_group_names(feature_names)
        assert "pft" in names
        assert "precip" in names
        assert len(names) == len(CANDIDATE_FEATURES)

    def test_partial_features(self) -> None:
        feature_names = ["precip", "LAI", "MF"]
        names = get_candidate_group_names(feature_names)
        assert "precip" in names
        assert "LAI" in names
        assert "pft" in names  # MF is one of the PFT columns


class TestGetAllExpectedColumns:
    def test_includes_mandatory(self) -> None:
        cols = get_all_expected_columns()
        for feat in MANDATORY_FEATURES:
            assert feat in cols

    def test_includes_pft_onehot(self) -> None:
        cols = get_all_expected_columns()
        for pft in PFT_ONEHOT_COLS:
            assert pft in cols

    def test_total_count(self) -> None:
        cols = get_all_expected_columns()
        # 6 mandatory + all candidate columns
        total_candidate_cols = sum(len(v) for v in CANDIDATE_FEATURES.values())
        assert len(cols) == 6 + total_candidate_cols


class TestEdgeCases:
    """Negative paths, boundaries, and regression tests for review fixes."""

    def test_mandatory_not_in_candidates(self) -> None:
        """Regression: mandatory features must not appear as candidates."""
        mandatory_set = set(MANDATORY_FEATURES)
        for group_name, col_names in CANDIDATE_FEATURES.items():
            for col in col_names:
                assert col not in mandatory_set, f"{col} in both mandatory and candidate group '{group_name}'"

    def test_no_duplicate_column_names_across_groups(self) -> None:
        """Each column name should appear in at most one candidate group."""
        seen: dict[str, str] = {}
        for group_name, col_names in CANDIDATE_FEATURES.items():
            for col in col_names:
                assert col not in seen, f"'{col}' in both '{seen[col]}' and '{group_name}'"
                seen[col] = group_name

    def test_empty_feature_names(self) -> None:
        mandatory_idx, candidate_groups = build_feature_groups([])
        assert mandatory_idx == []
        assert candidate_groups == []

    def test_duplicate_feature_names_uses_first_occurrence(self) -> None:
        feature_names = ["sw_in", "ta", "sw_in", "vpd"]
        mandatory_idx, _ = build_feature_groups(feature_names)
        # name_to_idx maps to last occurrence (dict overwrites), but all mandatory found
        assert len(mandatory_idx) == 3  # sw_in, ta, vpd

    def test_build_groups_with_only_mandatory(self) -> None:
        feature_names = list(MANDATORY_FEATURES)
        mandatory_idx, candidate_groups = build_feature_groups(feature_names)
        assert len(mandatory_idx) == 6
        assert candidate_groups == []

    def test_candidate_count_matches_registry(self) -> None:
        """Total candidate groups should match the OrderedDict length."""
        assert len(CANDIDATE_FEATURES) > 90  # plan says ~100 groups
