"""Tests for training_utils extracted module."""

import numpy as np
import pandas as pd
import pytest

from src.hyperparameter_optimization.training_utils import (
    add_time_features,
    build_pft_onehot,
    convert_windows_to_numpy,
    create_spatial_groups,
)


class TestAddTimeFeatures:
    """Test cyclical time feature creation."""

    @pytest.fixture
    def time_df(self):
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        return pd.DataFrame({"solar_TIMESTAMP": dates, "ta": np.random.randn(365)})

    def test_adds_time_columns(self, time_df):
        result = add_time_features(time_df)
        for col in ["Day sin", "Day cos", "Year sin", "Year cos"]:
            assert col in result.columns, f"Missing {col}"

    def test_cyclical_range(self, time_df):
        result = add_time_features(time_df)
        # Sin/cos should be in [-1, 1]
        assert result["Year sin"].between(-1, 1).all()
        assert result["Year cos"].between(-1, 1).all()


class TestConvertWindowsToNumpy:
    """Test window list to numpy conversion."""

    def test_basic_conversion(self):
        rng = np.random.RandomState(42)
        windows = [
            (rng.randn(5, 3), rng.randn(1)),
            (rng.randn(5, 3), rng.randn(1)),
        ]
        X, y = convert_windows_to_numpy(windows)
        # Reshapes to 2D: (n_windows, input_width * n_features)
        assert X.shape == (2, 15)
        assert y.shape == (2,)

    def test_empty_input(self):
        X, y = convert_windows_to_numpy([])
        assert len(X) == 0
        assert len(y) == 0


class TestCreateSpatialGroups:
    """Test spatial group creation."""

    def test_returns_tuple(self):
        lat = np.array([51.0] * 50 + [52.0] * 50)
        lon = np.array([7.0] * 50 + [8.0] * 50)
        groups, info = create_spatial_groups(lat, lon, method="grid", n_groups=2)
        assert isinstance(groups, np.ndarray)
        assert len(groups) == 100

    def test_group_count(self):
        lat = np.array([51.0] * 50 + [52.0] * 50)
        lon = np.array([7.0] * 50 + [8.0] * 50)
        groups, _ = create_spatial_groups(lat, lon, method="grid", n_groups=4)
        # Should have at most n_groups unique groups
        assert len(np.unique(groups)) <= 4

    def test_balanced_folds_hold_when_inert_sites_are_appended(self):
        """A site carrying ZERO rows must not move a real site between folds.

        `np.argsort(data_counts)[::-1]` defaults to quicksort, which is not
        stable, so lengthening the array permutes the order of TIED counts and
        the greedy packer sends them to different folds. Measured on the real
        177-site population: 16 of 167 real sites moved while the fold
        row-totals stayed identical to the row -- proof it is sort order, not
        capacity.
        """
        rng = np.random.RandomState(0)
        n_real = 60
        # Ties are the trigger; all-distinct counts give a total order.
        counts = np.repeat([300.0, 200.0, 148.0, 89.0, 88.0, 42.0], 10)
        lat = rng.uniform(-50.0, 60.0, n_real)
        lon = rng.uniform(-150.0, 170.0, n_real)

        alone, _ = create_spatial_groups(
            lat=lat, lon=lon, data_counts=counts, n_groups=10, balanced=True, method="default"
        )
        padded, _ = create_spatial_groups(
            lat=np.concatenate([lat, rng.uniform(-50.0, 60.0, 10)]),
            lon=np.concatenate([lon, rng.uniform(-150.0, 170.0, 10)]),
            data_counts=np.concatenate([counts, np.zeros(10)]),
            n_groups=10,
            balanced=True,
            method="default",
        )
        moved = int(np.sum(alone != padded[:n_real]))
        assert moved == 0, f"{moved} of {n_real} real sites changed fold"

    def test_balanced_grid_folds_hold_when_inert_sites_are_appended(self):
        """Same instability, other branch: np.argsort(cell_data_counts)."""
        rng = np.random.RandomState(1)
        n_real = 60
        counts = np.repeat([300.0, 200.0, 148.0, 89.0, 88.0, 42.0], 10)
        lat = rng.uniform(-50.0, 60.0, n_real)
        lon = rng.uniform(-150.0, 170.0, n_real)
        # Keep the inert points INSIDE the existing extent: the real empty
        # sites did not widen it either (n_lon_bins 6481 with and without),
        # and widening it would change the bin count for an unrelated reason.
        inert_lat = rng.uniform(lat.min(), lat.max(), 10)
        inert_lon = rng.uniform(lon.min(), lon.max(), 10)

        alone, _ = create_spatial_groups(
            lat=lat, lon=lon, data_counts=counts, n_groups=10, balanced=True, method="grid"
        )
        padded, _ = create_spatial_groups(
            lat=np.concatenate([lat, inert_lat]),
            lon=np.concatenate([lon, inert_lon]),
            data_counts=np.concatenate([counts, np.zeros(10)]),
            n_groups=10,
            balanced=True,
            method="grid",
        )
        moved = int(np.sum(alone != padded[:n_real]))
        assert moved == 0, f"{moved} of {n_real} real sites changed fold"

    def test_tied_counts_are_assigned_deterministically(self):
        """All-equal counts must give one fixed, reproducible round robin.

        Note the direction. The greedy consumes
        ``np.argsort(counts, kind="stable")[::-1]``: the stable sort orders ties
        by ascending index, then ``[::-1]`` reverses the lot, so tied sites are
        consumed LAST-index-first. Deterministic, which is the property that
        matters -- but not input order, so it is worth pinning. A later switch
        to ``np.argsort(-counts, kind="stable")`` would read as tidier and
        would silently reshuffle every fold; this test makes that visible.
        """
        lat = np.arange(6, dtype=float) + 50.0
        lon = np.arange(6, dtype=float) + 7.0
        counts = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        groups, _ = create_spatial_groups(
            lat=lat, lon=lon, data_counts=counts, n_groups=3, balanced=True, method="default"
        )
        assert list(groups) == [2, 1, 0, 2, 1, 0]

        # And it must not depend on how many inert sites trail the array.
        padded, _ = create_spatial_groups(
            lat=np.concatenate([lat, lat[:4] + 0.5]),
            lon=np.concatenate([lon, lon[:4] + 0.5]),
            data_counts=np.concatenate([counts, np.zeros(4)]),
            n_groups=3,
            balanced=True,
            method="default",
        )
        assert list(padded[:6]) == list(groups)

    def test_the_tie_scenario_is_actually_order_sensitive(self):
        """Control: prove the two tests above can detect an unstable sort.

        If fold assignment did not depend on the order tied counts are consumed
        in, they would pass vacuously against any implementation. Asserting
        numpy's quicksort is unstable would be testing numpy; asserting the
        SCENARIO is order-sensitive is version-independent.
        """
        counts = np.repeat([300.0, 200.0, 148.0, 89.0, 88.0, 42.0], 10)

        def greedy(order):
            groups = np.zeros(len(counts), dtype=int) - 1
            load = np.zeros(10)
            for idx in order:
                target = int(np.argmin(load))
                groups[idx] = target
                load[target] += counts[idx]
            return groups

        order = list(np.argsort(counts, kind="stable")[::-1])
        assert counts[order[0]] == counts[order[1]], "first two must be tied for this control"
        swapped = list(order)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        assert not np.array_equal(greedy(order), greedy(swapped)), (
            "swapping two TIED sites did not change the folds -- this scenario "
            "cannot detect an unstable sort, so the stability tests are vacuous"
        )


class TestBoundaryValues:
    """Round 1: Edge cases and boundary values."""

    def test_single_timestamp(self):
        """Single date produces valid time features."""
        df = pd.DataFrame({"solar_TIMESTAMP": pd.to_datetime(["2020-06-15"]), "ta": [25.0]})
        result = add_time_features(df)
        assert result["Year sin"].notna().all()
        assert result["Day cos"].notna().all()

    def test_windows_single_element(self):
        """Single window element."""
        windows = [(np.array([[1.0, 2.0]]), np.array([3.0]))]
        X, y = convert_windows_to_numpy(windows)
        assert X.shape == (1, 2)
        assert y.shape == (1,)

    def test_spatial_groups_single_point(self):
        """Single coordinate."""
        groups, info = create_spatial_groups(np.array([51.0]), np.array([7.0]), method="grid")
        assert len(groups) == 1
        assert len(info) == 1

    def test_spatial_groups_invalid_method(self):
        """Unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_spatial_groups(np.array([51.0]), np.array([7.0]), method="invalid")


class TestNegativePaths:
    """Round 2: Negative paths and error handling."""

    def test_add_time_features_missing_column(self):
        """Missing datetime column raises ValueError."""
        df = pd.DataFrame({"ta": [1.0, 2.0]})
        with pytest.raises(ValueError, match="not found"):
            add_time_features(df, datetime_column="solar_TIMESTAMP")

    def test_add_time_features_nan_timestamps(self):
        """NaN timestamps are dropped with warning."""
        df = pd.DataFrame(
            {
                "solar_TIMESTAMP": [pd.Timestamp("2020-01-01"), pd.NaT, pd.Timestamp("2020-01-03")],
                "ta": [1.0, 2.0, 3.0],
            }
        )
        result = add_time_features(df)
        assert len(result) == 2  # NaT row dropped

    def test_add_time_features_non_datetime_index(self):
        """Non-convertible index raises ValueError."""
        df = pd.DataFrame({"ta": [1.0]}, index=["not_a_date"])
        with pytest.raises(ValueError, match="cannot be converted"):
            add_time_features(df, datetime_column=None)

    def test_spatial_groups_default_method(self):
        """Default method with balanced groups."""
        lat = np.array([51.0, 52.0, 53.0, 54.0])
        lon = np.array([7.0, 8.0, 9.0, 10.0])
        counts = np.array([100, 50, 75, 25])
        groups, info = create_spatial_groups(lat, lon, data_counts=counts, n_groups=2, method="default")
        assert len(groups) == 4
        assert len(np.unique(groups)) <= 2


PFT_COLUMNS = ["MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"]


class TestBuildPftOnehot:
    """A missing PFT and a WRONG PFT must not be treated the same way.

    Both produce an all-zero one-hot row, but only one is a defect. Conflating
    them cost 27 sites -- FRA_HES and the four largest CZE stands among them --
    because a single blank calendar row rejected the whole site.
    """

    @staticmethod
    def _frame(values):
        return pd.DataFrame({"pft": values, "sap_velocity": [1.0] * len(values)})

    def test_rows_with_no_pft_are_dropped_not_fatal(self):
        """The real corpus shape since the 2026-08-24 re-merge: ['ENF', nan]."""
        df = self._frame(["ENF", None, "ENF", None])
        out = build_pft_onehot(df, "pft", "CZE_BK1", PFT_COLUMNS)
        assert len(out) == 2
        assert out["ENF"].tolist() == [1, 1]

    def test_still_raises_on_an_unrecognised_value(self):
        """NEGATIVE CONTROL -- the guard's original purpose must survive.

        A long IGBP name where a short code is expected is silently NaN-coerced
        by pd.Categorical; XGBoost then never splits on PFT and SHAP reports 0
        for every PFT column. If this stopped raising, the fix would have
        removed the protection instead of narrowing it.
        """
        df = self._frame(["Mixed Forest", "Mixed Forest"])
        with pytest.raises(ValueError, match="Unrecognised value"):
            build_pft_onehot(df, "pft", "SITE_X", PFT_COLUMNS)

    def test_raises_when_a_wrong_value_hides_among_missing_ones(self):
        """The mixed case must not be masked by dropping the NaNs first."""
        df = self._frame(["ENF", None, "Mixed Forest"])
        with pytest.raises(ValueError, match="Unrecognised value"):
            build_pft_onehot(df, "pft", "SITE_X", PFT_COLUMNS)

    def test_raises_when_no_row_has_a_pft(self):
        df = self._frame([None, None])
        with pytest.raises(ValueError, match="no row carries"):
            build_pft_onehot(df, "pft", "SITE_X", PFT_COLUMNS)

    def test_emits_every_pft_column_not_just_the_present_one(self):
        """XGBoost needs a stable feature order across sites."""
        df = self._frame(["DBF", "DBF"])
        out = build_pft_onehot(df, "pft", "SITE_X", PFT_COLUMNS)
        for col in PFT_COLUMNS:
            assert col in out.columns, f"missing {col}"
        assert out["DBF"].tolist() == [1, 1]
        assert out["ENF"].tolist() == [0, 0]

    def test_does_not_drop_rows_when_nothing_is_missing(self):
        df = self._frame(["ENF", "DBF", "MF"])
        out = build_pft_onehot(df, "pft", "SITE_X", PFT_COLUMNS)
        assert len(out) == 3
