"""Integration test for the file-level run() orchestration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.sapwood_imputation.impute_sapwood_area import (
    SOURCE_IMPUTED,
    SOURCE_MEASURED,
    _build_parser,
    run,
)


def _write_site(csv_dir, site: str, df: pd.DataFrame) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_dir / f"{site}_plant_md.csv", index=False)


def _make_data_dir(tmp_path):
    """Build a minimal data/raw/<v>/<v>/csv/plant tree and return base_data_dir."""
    base = tmp_path / "data"
    plant_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant"

    rng = np.random.default_rng(7)
    n = 60
    # Lognormal scatter, NOT a bare power law. A noise-free ASW = a * DBH^b is
    # exactly the signature provenance.py flags as provider-derived, so a
    # frictionless fixture would be excluded from its own fit and the run would
    # fail with "no group has >= min_n trees". Real measurements scatter; so
    # must the fixture. sigma 0.15 is well inside the archive's own ~0.56.
    dbh = rng.uniform(8, 55, size=n)
    pine = pd.DataFrame(
        {
            "pl_code": [f"S1_p{i}" for i in range(n)],
            "pl_species": "Pinus sylvestris",
            "pl_dbh": dbh,
            "pl_sapw_area": 0.6 * dbh**1.9 * np.exp(rng.normal(0.0, 0.15, size=n)),
        }
    )
    dbh2 = rng.uniform(8, 55, size=n)
    oak = pd.DataFrame(
        {
            "pl_code": [f"S2_o{i}" for i in range(n)],
            "pl_species": "Quercus ilex",
            "pl_dbh": dbh2,
            "pl_sapw_area": 1.4 * dbh2**1.6 * np.exp(rng.normal(0.0, 0.15, size=n)),
        }
    )
    # A site with missing ASW to be imputed + one missing-DBH (unresolved).
    target = pd.DataFrame(
        {
            "pl_code": ["S3_t0", "S3_t1"],
            "pl_species": ["Pinus sylvestris", "Quercus ilex"],
            "pl_dbh": [30.0, np.nan],
            "pl_sapw_area": [np.nan, np.nan],
        }
    )
    _write_site(plant_dir, "S1", pine)
    _write_site(plant_dir, "S2", oak)
    _write_site(plant_dir, "S3", target)
    return base


@pytest.mark.integration
def test_run_writes_filled_files_and_report(tmp_path) -> None:
    base = _make_data_dir(tmp_path)
    fit = run(scale="plant", base_data_dir=str(base), dry_run=False)

    assert set(fit.groups) == {"angiosperm", "gymnosperm"}

    out_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant_asw_filled"
    assert (out_dir / "S3_plant_md.csv").exists()
    report = json.loads((out_dir / "asw_allometry_fit_report.json").read_text())
    assert report["scale"] == "plant"
    assert report["counts"][SOURCE_IMPUTED] == 1  # S3_t0 imputed
    assert report["counts"]["unresolved"] == 1  # S3_t1 missing DBH

    # The artifact must be self-describing: a methods section should be writable
    # from the report alone, without opening the code to learn the model form.
    written_fit = report["fit"]
    assert "offset[functional_type]" in written_fit["form"]
    assert written_fit["offsets"][written_fit["reference"]] == 0.0
    assert set(written_fit["n_by_group"]) == {"angiosperm", "gymnosperm"}
    # Nested model: adding the functional type can never lower R2.
    assert written_fit["r2"] >= written_fit["r2_without_functional_type"]

    # The imputed file: original column preserved, new columns present.
    s3 = pd.read_csv(out_dir / "S3_plant_md.csv")
    assert s3["pl_sapw_area"].isna().all()  # original untouched (was all NaN)
    imp = s3[s3["pl_sapw_area_source"] == SOURCE_IMPUTED].iloc[0]
    assert imp["pl_sapw_area_filled"] == pytest.approx(0.6 * 30.0**1.9, rel=0.2)

    # A measured site keeps its values.
    s1 = pd.read_csv(out_dir / "S1_plant_md.csv")
    assert (s1["pl_sapw_area_source"] == SOURCE_MEASURED).all()


@pytest.mark.integration
def test_run_dry_run_writes_nothing(tmp_path) -> None:
    base = _make_data_dir(tmp_path)
    run(scale="plant", base_data_dir=str(base), dry_run=True)
    out_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant_asw_filled"
    assert not out_dir.exists()


@pytest.mark.integration
def test_degraded_site_is_still_written(tmp_path) -> None:
    """A file missing pl_dbh must appear in the output, not silently vanish.

    Consumers glob the output directory; a missing site there is invisible.
    """
    base = _make_data_dir(tmp_path)
    plant_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant"
    _write_site(plant_dir, "S4", pd.DataFrame({"pl_code": ["S4_x0"], "pl_species": ["Fagus sylvatica"]}))

    run(scale="plant", base_data_dir=str(base), dry_run=False)
    out_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant_asw_filled"

    # Output mirrors the input site set exactly.
    written = {p.name for p in out_dir.glob("*_plant_md.csv")}
    assert written == {f"S{i}_plant_md.csv" for i in (1, 2, 3, 4)}

    # The degraded site carries the full schema, flagged unresolved.
    s4 = pd.read_csv(out_dir / "S4_plant_md.csv")
    assert s4["pl_sapw_area_source"].tolist() == ["unresolved"]
    assert s4["pl_sapw_area_filled"].isna().all()

    report = json.loads((out_dir / "asw_allometry_fit_report.json").read_text())
    assert report["n_input_files"] == 4
    assert report["degraded_files"] == [
        {"file": "S4_plant_md.csv", "origin": "external", "missing_columns": ["pl_dbh"]}
    ]


@pytest.mark.integration
def test_output_subdir_may_not_overwrite_the_input_dir(tmp_path) -> None:
    base = _make_data_dir(tmp_path)
    with pytest.raises(ValueError, match="overwriting the source plant metadata"):
        run(scale="plant", base_data_dir=str(base), output_subdir="plant", dry_run=False)


# ---------------------------------------------------------------------------
# The internal corpus
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_internal_sites_are_processed_and_vote(tmp_path) -> None:
    """--include-internal must both widen the fit and produce output files."""
    base = _make_data_dir(tmp_path)
    internal = tmp_path / "internal" / "plant"
    dbh = np.linspace(12.0, 48.0, 12)
    _write_site(
        internal,
        "AUT_MMG",
        pd.DataFrame(
            {
                "pl_code": [f"AUT_MMG_p{i}" for i in range(12)],
                "pl_species": "Pinus sylvestris",
                "pl_dbh": dbh,
                # scatter, or provenance.py reads the fixture as a formula
                "pl_sapw_area": 0.45 * dbh**1.9 * np.exp(np.random.default_rng(3).normal(0, 0.15, len(dbh))),
            }
        ),
    )

    without = run(scale="plant", base_data_dir=str(base), dry_run=True)
    with_internal = run(
        scale="plant", base_data_dir=str(base), dry_run=False, include_internal=True, internal_dir=str(internal)
    )

    assert with_internal.n == without.n + 12

    out_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant_asw_filled"
    assert (out_dir / "AUT_MMG_plant_md.csv").exists()

    report = json.loads((out_dir / "asw_allometry_fit_report.json").read_text())
    assert report["include_internal"] is True
    assert report["sources"]["files_by_origin"] == {"external": 3, "internal": 1}
    assert report["counts_by_origin"]["internal"][SOURCE_MEASURED] == 12


@pytest.mark.integration
def test_duplicate_internal_site_is_written_but_does_not_vote(tmp_path) -> None:
    """SWE_NOR's stems are already in the archive; it must not vote twice."""
    base = _make_data_dir(tmp_path)
    internal = tmp_path / "internal" / "plant"
    dbh = np.linspace(15.0, 45.0, 10)
    _write_site(
        internal,
        "SWE_NOR",
        pd.DataFrame(
            {
                "pl_code": [f"SWE_NOR_p{i}" for i in range(10)],
                "pl_species": "Picea abies",
                "pl_dbh": dbh,
                "pl_sapw_area": 0.5 * dbh**1.9 * np.exp(np.random.default_rng(4).normal(0, 0.15, len(dbh))),
            }
        ),
    )

    without = run(scale="plant", base_data_dir=str(base), dry_run=True)
    with_internal = run(
        scale="plant", base_data_dir=str(base), dry_run=False, include_internal=True, internal_dir=str(internal)
    )

    # Excluded from the fit ...
    assert with_internal.n == without.n
    out_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant_asw_filled"
    # ... but still written, with its measured values intact.
    written = pd.read_csv(out_dir / "SWE_NOR_plant_md.csv")
    assert (written["pl_sapw_area_source"] == SOURCE_MEASURED).all()

    report = json.loads((out_dir / "asw_allometry_fit_report.json").read_text())
    assert report["sources"]["duplicate_sites_excluded_from_fit"] == {"SWE_NOR": "SWE_NOR_ST1_AF1 + SWE_NOR_ST1_AF2"}


@pytest.mark.integration
def test_one_fit_pooled_across_scales(tmp_path) -> None:
    """The fit is a property of stems, so both scales must return the SAME one.

    Also pins that a site living at only one scale still votes: reading a single
    directory silently drops ~37 archive sites, and the resulting two fits then
    disagree by up to ~1%.
    """
    base = _make_data_dir(tmp_path)  # writes csv/plant/{S1,S2,S3}
    sapwood_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "sapwood"
    rng = np.random.default_rng(21)
    d = rng.uniform(10, 50, size=40)
    # A site that exists ONLY at sapwood scale.
    _write_site(
        sapwood_dir,
        "S9",
        pd.DataFrame(
            {
                "pl_code": [f"S9_g{i}" for i in range(40)],
                "pl_species": "Picea abies",
                "pl_dbh": d,
                "pl_sapw_area": 0.5 * d**1.85 * np.exp(rng.normal(0, 0.15, 40)),
            }
        ),
    )

    plant_fit = run(scale="plant", base_data_dir=str(base), dry_run=True)
    sapwood_fit = run(scale="sapwood", base_data_dir=str(base), dry_run=True)

    assert plant_fit.n == sapwood_fit.n
    assert plant_fit.slope == sapwood_fit.slope
    assert plant_fit.intercept == sapwood_fit.intercept
    assert dict(plant_fit.offsets) == dict(sapwood_fit.offsets)
    # S9 is sapwood-only; the plant run must still have counted its 40 trees.
    assert plant_fit.n == 160


@pytest.mark.integration
def test_cross_scale_duplicate_keeps_the_richer_copy(tmp_path) -> None:
    """One set of stems votes once -- and via the copy that actually has ASW.

    Modelled on ITA_CP2, whose plant copy carries DBH but no pl_sapw_area at
    all; preferring a fixed scale would discard eight measured trees.
    """
    base = _make_data_dir(tmp_path)
    csv = base / "raw" / "0.1.5" / "0.1.5" / "csv"
    d = np.linspace(12.0, 44.0, 9)
    asw = 0.62 * d**1.88 * np.exp(np.random.default_rng(5).normal(0, 0.15, 9))
    _write_site(  # plant copy: DBH only, ASW entirely absent
        csv / "plant",
        "S7",
        pd.DataFrame(
            {
                "pl_code": [f"S7_t{i}" for i in range(9)],
                "pl_species": "Fagus sylvatica",
                "pl_dbh": d,
                "pl_sapw_area": [np.nan] * 9,
            }
        ),
    )
    _write_site(  # sapwood copy: same stems, with ASW
        csv / "sapwood",
        "S7",
        pd.DataFrame(
            {
                "pl_code": [f"S7_t{i}" for i in range(9)],
                "pl_species": "Fagus sylvatica",
                "pl_dbh": d,
                "pl_sapw_area": asw,
            }
        ),
    )

    fit = run(scale="plant", base_data_dir=str(base), dry_run=False)
    assert fit.n == 129  # 120 fixture + S7's 9, counted exactly once

    out = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant_asw_filled"
    report = json.loads((out / "asw_allometry_fit_report.json").read_text())
    assert report["fit_pool"]["scales_pooled"] == ["plant", "sapwood"]
    dropped = {d["site"]: d for d in report["fit_pool"]["cross_scale_duplicates_dropped"]}
    assert dropped["S7"]["kept_scale"] == "sapwood"
    assert dropped["S7"]["fittable_kept"] == 9
    assert dropped["S7"]["fittable_dropped"] == 0


@pytest.mark.integration
def test_default_policy_bars_only_the_documented_site(tmp_path) -> None:
    """End-to-end: a documented site is dropped, a merely-detected one is not.

    Guards the whole point of the 'declared' default -- that the statistical
    detector reports without acting, so nobody has to accept a chosen R2 cut.
    """
    base = _make_data_dir(tmp_path)
    plant_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant"

    # A noise-free power law: DETECTED as derived, but not documented.
    d = np.linspace(12.0, 50.0, 10)
    _write_site(
        plant_dir,
        "USA_TNP",
        pd.DataFrame(
            {
                "pl_code": [f"USA_TNP_p{i}" for i in range(10)],
                "pl_species": "Pinus taeda",
                "pl_dbh": d,
                "pl_sapw_area": 0.55 * d**1.9,
            }
        ),
    )
    # Documented in provenance.ESTIMATED_SITES, and scattered enough to pass
    # every statistical test -- only the registry can bar it.
    d2 = np.linspace(14.0, 46.0, 8)
    _write_site(
        plant_dir,
        "AUT_ZOE",
        pd.DataFrame(
            {
                "pl_code": [f"AUT_ZOE_p{i}" for i in range(8)],
                "pl_species": "Fagus sylvatica",
                "pl_dbh": d2,
                "pl_sapw_area": 0.7 * d2**1.8 * np.exp(np.random.default_rng(11).normal(0, 0.2, 8)),
            }
        ),
    )

    fit = run(scale="plant", base_data_dir=str(base), dry_run=False)
    strict = run(scale="plant", base_data_dir=str(base), dry_run=True, exclusion_policy="all")

    # Default drops AUT_ZOE's 8 only; 'all' drops USA_TNP's 10 as well.
    assert fit.n - strict.n == 10

    out_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant_asw_filled"
    report = json.loads((out_dir / "asw_allometry_fit_report.json").read_text())
    assert report["estimated_exclusion_policy"] == "declared"
    assert report["estimated_verdicts_excluded"] == ["provider_declared_estimate"]
    # The detector still REPORTS the undocumented site it did not act on.
    assert "USA_TNP" in report["asw_provenance"]["estimated_sites"]["allometry_derived"]

    # Only the barred site is branded in the delivered data.
    assert (pd.read_csv(out_dir / "AUT_ZOE_plant_md.csv")["pl_sapw_area_source"] == "provider_estimate").all()
    assert (pd.read_csv(out_dir / "USA_TNP_plant_md.csv")["pl_sapw_area_source"] == SOURCE_MEASURED).all()


@pytest.mark.integration
def test_implausible_internal_tree_is_flagged_in_the_report(tmp_path) -> None:
    """A mis-united tree neither votes nor reaches pl_sapw_area_filled."""
    base = _make_data_dir(tmp_path)
    internal = tmp_path / "internal" / "plant"
    _write_site(
        internal,
        "NOR_HUR",
        pd.DataFrame(
            {
                "pl_code": ["NOR_HUR_p0", "NOR_HUR_p1"],
                "pl_species": ["Picea abies", "Picea abies"],
                "pl_dbh": [293.0, 30.0],  # first is mm, second is cm
                "pl_sapw_area": [242.2, 250.0],
            }
        ),
    )
    run(scale="plant", base_data_dir=str(base), dry_run=False, include_internal=True, internal_dir=str(internal))

    out_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant_asw_filled"
    written = pd.read_csv(out_dir / "NOR_HUR_plant_md.csv")
    assert written["pl_sapw_area_source"].tolist() == ["implausible", SOURCE_MEASURED]
    assert pd.isna(written["pl_sapw_area_filled"].iloc[0])

    report = json.loads((out_dir / "asw_allometry_fit_report.json").read_text())
    assert report["geometry_gate"]["internal"]["implausible"] == 1
    assert report["geometry_gate"]["external"]["implausible"] == 0


def _multi_site_data_dir(tmp_path):
    """Six sites, each holding both functional types.

    Leave-one-site-out can only score a site when the remaining sites still
    carry every fitted group, so the two-site fixture above cannot exercise it.
    """
    base = tmp_path / "data"
    plant_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant"
    rng = np.random.default_rng(23)
    for s in range(6):
        frames = []
        for species, coefficient in (("Pinus sylvestris", 0.7), ("Quercus ilex", 1.3)):
            dbh = rng.uniform(10, 55, size=20)
            frames.append(
                pd.DataFrame(
                    {
                        "pl_code": [f"S{s}_{species[:3]}{i}" for i in range(20)],
                        "pl_species": species,
                        "pl_dbh": dbh,
                        "pl_sapw_area": coefficient * dbh**1.75 * np.exp(rng.normal(0.0, 0.2, size=20)),
                    }
                )
            )
        _write_site(plant_dir, f"SITE{s}", pd.concat(frames, ignore_index=True))
    return base


@pytest.mark.integration
def test_report_carries_real_out_of_sample_validation(tmp_path) -> None:
    """The artifact must state what the fit is worth on a site it never saw.

    Guards a regression this module has actually suffered: the validation call
    being replaced by an empty dict, which leaves the report structurally valid
    while silently reducing it to an in-sample R2. Asserting the keys is not
    enough -- the numbers have to have been computed.
    """
    base = _multi_site_data_dir(tmp_path)
    fit = run(scale="plant", base_data_dir=str(base), dry_run=False)

    out_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant_asw_filled"
    report = json.loads((out_dir / "asw_allometry_fit_report.json").read_text())
    validation = report["validation"]
    assert set(validation) == {
        "leave_one_site_out",
        "residual_spread_by_size",
        "bias_by_genus",
        "site_leverage",
    }

    loso = validation["leave_one_site_out"]
    assert loso["scored_sites"] == 6
    assert loso["unweighted"]["trees_scored"] == fit.n
    # Held-out skill is real but never flatters the in-sample fit.
    assert 0.0 < loso["unweighted"]["r2"] <= fit.r2
    assert validation["site_leverage"]["sites"] == 6
    assert validation["site_leverage"]["trees"] == fit.n


@pytest.mark.integration
def test_report_states_which_back_transform_was_applied(tmp_path) -> None:
    """Both factors travel with the artifact, so the choice can be audited."""
    base = _multi_site_data_dir(tmp_path)
    fit = run(scale="plant", base_data_dir=str(base), dry_run=False, back_transform="smearing")

    out_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant_asw_filled"
    written = json.loads((out_dir / "asw_allometry_fit_report.json").read_text())["fit"]["back_transform"]

    assert written["applied"] == "smearing"
    assert written["smearing_factor"] == pytest.approx(fit.correction_factor)
    assert written["baskerville_factor"] != pytest.approx(fit.correction_factor)
    assert "residual_skew" in written and "residuals_look_normal" in written


@pytest.mark.integration
def test_taxon_refinement_is_on_by_default(tmp_path) -> None:
    """Forgetting the flag must not silently produce the worse model.

    The refinement used to be opt-in so the published Poyatos-form fit stayed the
    reproducible baseline. That trade stopped being worth it once the imputed
    area began feeding a *division*: sap velocity = flow / area, so an area error
    lands undiluted in the training target, and none of the run's own integrity
    checks can see it -- row count, coverage, source flags and every statistic in
    the fit report are identical either way, because the refinement is an
    additive offset applied at prediction time and never touches the fit. The
    baseline is still reachable, but now it is the deviation that has to be
    typed out.
    """
    base = _multi_site_data_dir(tmp_path)
    run(scale="plant", base_data_dir=str(base), dry_run=False)
    out_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant_asw_filled"
    report = json.loads((out_dir / "asw_allometry_fit_report.json").read_text())
    assert report["taxon_offsets"] is not None


@pytest.mark.integration
def test_the_unrefined_baseline_is_still_reachable(tmp_path) -> None:
    """Reproducing the published Poyatos-form fit stays possible, just explicit."""
    base = _multi_site_data_dir(tmp_path)
    run(scale="plant", base_data_dir=str(base), dry_run=False, taxon_refinement=False)
    out_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant_asw_filled"
    report = json.loads((out_dir / "asw_allometry_fit_report.json").read_text())
    assert report["taxon_offsets"] is None


@pytest.mark.unit
def test_cli_defaults_to_the_refined_model() -> None:
    parser = _build_parser()
    assert parser.parse_args([]).taxon_refinement is True


@pytest.mark.unit
def test_cli_can_opt_out_of_the_refinement() -> None:
    parser = _build_parser()
    assert parser.parse_args(["--no-taxon-refinement"]).taxon_refinement is False


@pytest.mark.unit
def test_the_old_explicit_flag_still_parses() -> None:
    """Job scripts and documented commands pass --taxon-refinement; they must keep working."""
    parser = _build_parser()
    assert parser.parse_args(["--taxon-refinement"]).taxon_refinement is True


@pytest.mark.integration
def test_taxon_refinement_touches_imputed_values_only(tmp_path) -> None:
    """A measured ASW is never nudged toward the model, only imputed ones."""
    base = _multi_site_data_dir(tmp_path)
    plant_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant"
    # A site with one measured and one unmeasured tree of the same species.
    _write_site(
        plant_dir,
        "TARGET",
        pd.DataFrame(
            {
                "pl_code": ["T_m", "T_i"],
                "pl_species": ["Quercus ilex", "Quercus ilex"],
                "pl_dbh": [30.0, 30.0],
                "pl_sapw_area": [123.456, np.nan],
            }
        ),
    )
    out_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant_asw_filled"

    run(scale="plant", base_data_dir=str(base), dry_run=False, taxon_refinement=False)
    plain = pd.read_csv(out_dir / "TARGET_plant_md.csv")
    run(scale="plant", base_data_dir=str(base), dry_run=False, taxon_refinement=True)
    refined = pd.read_csv(out_dir / "TARGET_plant_md.csv")

    for frame in (plain, refined):
        # The provider's own number is passed through untouched either way.
        assert frame.loc[0, "pl_sapw_area_filled"] == pytest.approx(123.456)
        assert frame.loc[0, "pl_sapw_area_source"] == SOURCE_MEASURED
        assert frame.loc[1, "pl_sapw_area_source"] == SOURCE_IMPUTED
    # The source column never changes meaning; only the imputed magnitude may.
    assert (plain["pl_sapw_area_source"] == refined["pl_sapw_area_source"]).all()


@pytest.mark.integration
def test_taxon_refinement_reports_the_evidence_for_each_offset(tmp_path) -> None:
    """Every earned offset must carry the evidence that earned it."""
    base = _multi_site_data_dir(tmp_path)
    run(scale="plant", base_data_dir=str(base), dry_run=False, taxon_refinement=True)
    out_dir = base / "raw" / "0.1.5" / "0.1.5" / "csv" / "plant_asw_filled"
    report = json.loads((out_dir / "asw_allometry_fit_report.json").read_text())["taxon_offsets"]

    assert report is not None
    assert report["z_min"] > 0
    for entry in report["offsets"]:
        assert entry["n_sites"] >= 2  # never a single site's effect
        assert abs(entry["offset_log"]) <= abs(entry["raw_offset_log"]) + 1e-12  # shrunk
