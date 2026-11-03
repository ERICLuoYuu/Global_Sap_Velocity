from __future__ import annotations

import importlib
import textwrap

import pandas as pd
import pytest
from src.static_features import results, run
from src.static_features.results import ExtractionFailure, RunContext, read_part

GOOD_PRODUCT = """
    from __future__ import annotations

    import pandas as pd

    from src.static_features.results import OK, ProductResult


    def extract(context):
        index = pd.Index(context.sites["site_name"], name="site_name")
        values = pd.DataFrame({"x": 1.5}, index=index)
        return ProductResult(
            product="fake_good",
            values=values,
            status=pd.DataFrame({"x": OK}, index=index),
            coverage=pd.DataFrame(index=index),
            by_year=None,
            extras={},
            provenance={"source": "fake"},
        )


    def fetch_inputs(context):
        (context.raw_root / "fetched.txt").write_text("done", encoding="utf-8")
"""

FAILING_PRODUCT = """
    from __future__ import annotations

    from src.static_features.results import ExtractionFailure


    def extract(context):
        raise ExtractionFailure("fake_failing failed at site 200: A")
"""

INVALID_PRODUCT = """
    from __future__ import annotations

    import numpy as np
    import pandas as pd

    from src.static_features.results import OK, ProductResult


    def extract(context):
        index = pd.Index(context.sites["site_name"], name="site_name")
        return ProductResult(
            product="fake_invalid",
            values=pd.DataFrame({"x": np.nan}, index=index),
            status=pd.DataFrame({"x": OK}, index=index),
            coverage=pd.DataFrame(index=index),
            by_year=None,
            extras={},
            provenance={},
        )
"""


@pytest.fixture
def registry(tmp_path, monkeypatch):
    package = tmp_path / "pkg"
    package.mkdir()
    for name, source in [
        ("fake_good", GOOD_PRODUCT),
        ("fake_failing", FAILING_PRODUCT),
        ("fake_invalid", INVALID_PRODUCT),
    ]:
        (package / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.syspath_prepend(str(package))
    return {name: name for name in ("fake_good", "fake_failing", "fake_invalid")}


@pytest.fixture
def context(tmp_path):
    sites = pd.DataFrame({"site_name": ["A", "B"], "lat": [1.0, 2.0], "lon": [3.0, 4.0]})
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    return RunContext(sites=sites, raw_root=raw_root, output_root=tmp_path / "out", project="test-project")


def test_parts_live_under_the_output_root(tmp_path):
    context = RunContext(sites=pd.DataFrame(), raw_root=tmp_path, output_root=tmp_path / "out", project="p")
    assert context.parts_root == tmp_path / "out" / "parts"


def test_run_product_writes_a_part_with_the_site_hash(context, registry):
    path = run.run_product("fake_good", context, registry=registry)
    back, site_hash = read_part(context.parts_root, "fake_good")

    assert path == context.parts_root / "fake_good"
    assert back.values.loc["B", "x"] == 1.5
    assert site_hash == run.site_list_hash(context.sites)


# Review Focus 2
def test_failed_product_writes_no_part(context, registry):
    with pytest.raises(ExtractionFailure, match="site 200"):
        run.run_product("fake_failing", context, registry=registry)
    assert not (context.parts_root / "fake_failing").exists()


def test_failed_rerun_keeps_the_earlier_good_part(context, registry):
    run.run_product("fake_good", context, registry=registry)
    mixed = {"fake_good": "fake_failing"}
    with pytest.raises(ExtractionFailure):
        run.run_product("fake_good", context, registry=mixed)

    back, _ = read_part(context.parts_root, "fake_good")
    assert back.values.loc["A", "x"] == 1.5


def test_run_product_refuses_invalid_result(context, registry):
    with pytest.raises(ExtractionFailure, match="fake_invalid"):
        run.run_product("fake_invalid", context, registry=registry)
    assert not (context.parts_root / "fake_invalid").exists()


def test_registry_is_imported_only_on_use():
    assert run.run_product.__defaults__[0] is run.PRODUCTS
    assert len(run.PRODUCTS) == 16
    assert all(path.startswith("src.static_features.products.") for path in run.PRODUCTS.values())


def _patch_context(monkeypatch, context, registry):
    calls = []

    def for_scale(scale, project):
        calls.append((scale, project))
        return context

    monkeypatch.setattr(results.RunContext, "for_scale", staticmethod(for_scale))
    monkeypatch.setattr(run, "PRODUCTS", registry)
    return calls


def test_main_runs_the_named_product_and_returns_zero(context, registry, monkeypatch):
    calls = _patch_context(monkeypatch, context, registry)
    code = run.main(["--product", "fake_good", "--scale", "sapwood_combined", "--project", "other-project"])

    assert code == 0
    assert calls == [("sapwood_combined", "other-project")]
    assert (context.parts_root / "fake_good" / "values.csv").exists()


def test_main_defaults_to_the_verified_earth_engine_project(context, registry, monkeypatch):
    calls = _patch_context(monkeypatch, context, registry)
    run.main(["--product", "fake_good", "--scale", "sapwood_combined"])
    assert calls[0][1] == "era5download-447713"


def test_main_returns_nonzero_on_failure(context, registry, monkeypatch):
    _patch_context(monkeypatch, context, registry)
    assert run.main(["--product", "fake_failing", "--scale", "sapwood_combined"]) != 0
    assert not (context.parts_root / "fake_failing").exists()


def test_main_returns_nonzero_on_invalid_result(context, registry, monkeypatch):
    _patch_context(monkeypatch, context, registry)
    assert run.main(["--product", "fake_invalid", "--scale", "sapwood_combined"]) != 0


def test_fetch_only_fetches_and_writes_no_part(context, registry, monkeypatch):
    _patch_context(monkeypatch, context, registry)
    code = run.main(["--product", "fake_good", "--scale", "sapwood_combined", "--fetch-only"])

    assert code == 0
    assert (context.raw_root / "fetched.txt").read_text(encoding="utf-8") == "done"
    assert not (context.parts_root / "fake_good").exists()


def test_fetch_only_on_a_product_with_nothing_to_fetch_fails(context, registry, monkeypatch):
    _patch_context(monkeypatch, context, registry)
    assert run.main(["--product", "fake_failing", "--scale", "sapwood_combined", "--fetch-only"]) != 0


def test_main_refuses_an_unknown_product(context, registry, monkeypatch):
    _patch_context(monkeypatch, context, registry)
    with pytest.raises(SystemExit):
        run.main(["--product", "nonesuch", "--scale", "sapwood_combined"])


def test_for_scale_reads_the_real_corpus():
    context = RunContext.for_scale("sapwood_combined", "some-project")

    assert len(context.sites) == 226
    assert context.project == "some-project"
    assert context.raw_root.name == "grided" and context.raw_root.is_dir()
    assert context.output_root.parts[-2:] == ("static_1km", "sapwood_combined")
    dev = context.sites.set_index("site_name").loc["GBR_DEV_CON"]
    # The corrected point: the published one, 4076 and 21153, is water in the Firth of Forth.
    assert (dev["row"], dev["col"]) == (4070, 21160)


@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
def test_every_registered_product_imports_and_can_extract():
    for name, module_path in run.PRODUCTS.items():
        module = importlib.import_module(module_path)
        assert callable(module.extract), name


def test_a_part_records_the_code_it_was_built_from(context, registry, monkeypatch):
    monkeypatch.setattr(run, "code_state", lambda: {"commit": "abc", "package_clean": True})
    run.run_product("fake_good", context, registry=registry)
    back, _ = read_part(context.parts_root, "fake_good")

    assert back.provenance == {"source": "fake", "code": {"commit": "abc", "package_clean": True}}


def test_the_code_state_is_read_before_the_work_starts(context, registry, monkeypatch):
    read = []
    monkeypatch.setattr(run, "code_state", lambda: read.append(True) or {"commit": "abc", "package_clean": True})

    with pytest.raises(ExtractionFailure):
        run.run_product("fake_failing", context, registry=registry)

    assert read == [True]
