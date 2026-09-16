"""The CLI must be able to name a servable run, its period and its own output.

Before, the only knob was --base-dir (which moved nothing but `base`), the
period was the hard-coded 2016-2018 TimePeriod, and every run wrote over the
previous run's figures.
"""

from __future__ import annotations

import pytest
from src.transpiration_comparison.products.serving_sap_velocity import (
    ServingSapVelocityProduct,
)

from src.transpiration_comparison.cli import build_parser, build_pipeline


def _args(argv):
    return build_parser().parse_args(argv)


class TestPeriod:
    def test_the_period_flags_reach_the_pipeline(self, tmp_path):
        args = _args(
            [
                "--base-dir",
                str(tmp_path),
                "--period-start",
                "2015-01-01",
                "--period-end",
                "2015-12-31",
                "preprocess",
            ]
        )
        pipeline = build_pipeline(args)
        assert pipeline.period.start == "2015-01-01"  # nosec B101
        assert pipeline.period.end == "2015-12-31"  # nosec B101

    def test_the_default_period_is_unchanged(self, tmp_path):
        pipeline = build_pipeline(_args(["--base-dir", str(tmp_path), "preprocess"]))
        assert pipeline.period.start == "2016-01-01"  # nosec B101
        assert pipeline.period.end == "2018-12-31"  # nosec B101


class TestSapSource:
    def test_serving_and_run_id_reach_the_pipeline(self, tmp_path):
        args = _args(
            [
                "--base-dir",
                str(tmp_path),
                "--sap-source",
                "serving",
                "--run-id",
                "sweep5_smzp",
                "preprocess",
            ]
        )
        pipeline = build_pipeline(args)
        assert isinstance(pipeline._sap_handler(), ServingSapVelocityProduct)  # nosec B101

    def test_serving_without_a_run_id_is_refused(self, tmp_path):
        args = _args(["--base-dir", str(tmp_path), "--sap-source", "serving", "preprocess"])
        with pytest.raises(SystemExit):
            build_pipeline(args)

    def test_the_default_is_still_the_parquet_dump(self, tmp_path):
        pipeline = build_pipeline(_args(["--base-dir", str(tmp_path), "preprocess"]))
        assert pipeline.sap_source == "parquet"  # nosec B101


class TestOutputIsolation:
    def test_an_out_tag_gives_the_run_its_own_directories(self, tmp_path):
        args = _args(["--base-dir", str(tmp_path), "--out-tag", "servable_smzp", "preprocess"])
        pipeline = build_pipeline(args)
        assert pipeline.paths.figures_dir.name == "servable_smzp"  # nosec B101
        assert pipeline.paths.preprocessed_dir.name == "servable_smzp"  # nosec B101

    def test_base_dir_moves_every_path(self, tmp_path):
        pipeline = build_pipeline(_args(["--base-dir", str(tmp_path), "preprocess"]))
        assert str(pipeline.paths.products_dir).startswith(str(tmp_path))  # nosec B101
        assert str(pipeline.paths.serving_store).startswith(str(tmp_path))  # nosec B101


class TestAlignment:
    def test_the_alignment_method_is_selectable(self, tmp_path):
        args = _args(["--base-dir", str(tmp_path), "--alignment", "linear", "preprocess"])
        assert build_pipeline(args).alignment == "linear"  # nosec B101

    def test_the_default_alignment_reproduces_the_published_run(self, tmp_path):
        assert build_pipeline(_args(["--base-dir", str(tmp_path), "preprocess"])).alignment == "nearest"  # nosec B101


class TestAllowMissing:
    def test_allow_missing_reaches_the_pipeline(self, tmp_path):
        args = _args(["--base-dir", str(tmp_path), "--allow-missing", "gldas", "preprocess"])
        assert build_pipeline(args).allow_missing == ["gldas"]  # nosec B101

    def test_several_products_may_be_allowed(self, tmp_path):
        args = _args(["--base-dir", str(tmp_path), "--allow-missing", "gldas,pmlv2", "preprocess"])
        assert build_pipeline(args).allow_missing == ["gldas", "pmlv2"]  # nosec B101

    def test_the_subcommand_survives_the_flag(self, tmp_path):
        """nargs='+' here swallowed `preprocess` and argparse then said the
        subcommand was missing -- caught before it cost a queue slot."""
        args = _args(["--base-dir", str(tmp_path), "--allow-missing", "gldas", "compare", "--phase", "spatial"])
        assert args.command == "compare"  # nosec B101

    def test_nothing_allowed_by_default(self, tmp_path):
        assert build_pipeline(_args(["--base-dir", str(tmp_path), "preprocess"])).allow_missing == []  # nosec B101
