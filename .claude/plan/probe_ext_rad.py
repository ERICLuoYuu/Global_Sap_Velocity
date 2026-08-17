"""Determine empirically what SAPFLUXNET's `ext_rad` column actually is.

`ext_rad` is a declared training feature, present at all 202 external sites and
absent at all 24 internal ones. Before synthesising it for the internal sites we
have to know which quantity it is, because the codebase implies two mutually
exclusive definitions:

  * `src/gap_filling/tests/test_e2e_real_site.py:189` treats `ext_rad == 0` as
    night -- that only holds for a geometry-dependent quantity.
  * `scripts/audit_data_quality.py:762` exempts `ext_rad` from the "night mean
    should be low" check -- that only makes sense for a day-of-year-only
    quantity, which never goes to zero.

Both cannot be true. This script decides it from the delivered data, then scores
candidate formulas against the sites that already ship the column. A candidate
that reproduces the 202 external sites is safe to apply to the 24 internal ones;
one that does not would inject a site-dependent bias exactly aligned with the
external/internal split, which is worse than the column being absent.

Candidates
----------
A  E0(doy) only, REddyProc/solartime `fCalcExtRadiation` -- never zero.
B  TOA horizontal, pvlib E0 x cos(zenith), clipped at 0 -- zero at night.
C  REddyProc E0 x cos(zenith), clipped at 0 -- separates the E0 model from the
   solar geometry, so a near-miss on B can be attributed to one or the other.

The timestamp convention is scored, not assumed: for a naive TIMESTAMP the
script sweeps UTC offsets and reports the best-fitting one. If the winning
offset is not 0 for a site whose metadata says UTC, that is itself a finding.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CORPUS = Path("/scratch/tmp/yluo2/Global_Sap_Velocity/data/raw/0.1.5/0.1.5/csv/sapwood_combined")
N_SITES = 6
SOLAR_CONSTANT = 1366.1


def has_ext_rad(path: Path) -> bool:
    with path.open() as handle:
        return "ext_rad" in handle.readline().strip().split(",")


def site_coords(site: str):
    """Return (lat, lon, elev) from the site metadata, or None if unusable."""
    md_path = CORPUS / (site + "_site_md.csv")
    if not md_path.exists():
        return None
    md = pd.read_csv(md_path)
    if md.empty:
        return None
    row = md.iloc[0]

    def num(*names):
        for name in names:
            if name in md.columns:
                value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
                if pd.notna(value):
                    return float(value)
        return None

    lat = num("si_lat", "latitude", "lat")
    lon = num("si_long", "si_lon", "longitude", "lon")
    elev = num("si_elev", "elevation", "elev")
    if lat is None or lon is None:
        return None
    return lat, lon, 0.0 if elev is None else elev


def reddyproc_e0(doy: np.ndarray) -> np.ndarray:
    """Extraterrestrial irradiance at normal incidence, REddyProc's series."""
    gamma = 2.0 * np.pi * (doy - 1.0) / 365.24
    return SOLAR_CONSTANT * (
        1.00011
        + 0.034221 * np.cos(gamma)
        + 0.00128 * np.sin(gamma)
        + 0.000719 * np.cos(2.0 * gamma)
        + 0.000077 * np.sin(2.0 * gamma)
    )


def score(observed: np.ndarray, predicted: np.ndarray) -> dict:
    ok = np.isfinite(observed) & np.isfinite(predicted)
    if ok.sum() < 100:
        return {"n": int(ok.sum()), "rmse": float("nan"), "r": float("nan"), "bias": float("nan")}
    obs, pred = observed[ok], predicted[ok]
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    bias = float(np.mean(pred - obs))
    denom = obs.std() * pred.std()
    r = float(np.mean((obs - obs.mean()) * (pred - pred.mean())) / denom) if denom > 0 else float("nan")
    return {"n": int(ok.sum()), "rmse": rmse, "r": r, "bias": bias, "maxerr": float(np.max(np.abs(obs - pred)))}


def main() -> None:
    env_files = sorted(CORPUS.glob("*_env_data.csv"))
    external = []
    for path in env_files:
        if has_ext_rad(path):
            external.append(path.name.replace("_env_data.csv", ""))
    print(f"external sites carrying ext_rad: {len(external)}")

    # Spread the probe across latitude -- a day-of-year-only column is identical
    # everywhere, so a latitude spread is what makes the two candidates separable.
    coords = {}
    for site in external:
        got = site_coords(site)
        if got is not None:
            coords[site] = got
    by_lat = sorted(coords.items(), key=lambda kv: kv[1][0])
    if len(by_lat) < N_SITES:
        print(f"FATAL: only {len(by_lat)} sites have usable coordinates")
        sys.exit(1)
    picks = [by_lat[int(round(i * (len(by_lat) - 1) / (N_SITES - 1)))] for i in range(N_SITES)]

    print("\n=== probe sites (latitude spread) ===")
    for site, (lat, lon, elev) in picks:
        print(f"  {site:<18} lat={lat:8.3f} lon={lon:9.3f} elev={elev:7.1f}")

    try:
        import pvlib

        print(f"\npvlib {pvlib.__version__}")
    except ImportError:
        print("\nFATAL: pvlib unavailable in this environment")
        sys.exit(1)

    # ---- part 1: what does the delivered column look like? ------------------
    print("\n=== part 1: shape of the delivered ext_rad column ===")
    print(f"{'site':<18} {'min':>10} {'max':>10} {'mean':>10} {'frac==0':>10} {'n':>10}")
    frames = {}
    for site, _ in picks:
        df = pd.read_csv(CORPUS / (site + "_env_data.csv"), usecols=["TIMESTAMP", "ext_rad"])
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce", utc=True)
        df = df.dropna(subset=["TIMESTAMP"])
        values = pd.to_numeric(df["ext_rad"], errors="coerce")
        frames[site] = df.assign(ext_rad=values).dropna(subset=["ext_rad"])
        finite = values.dropna()
        if finite.empty:
            print(f"{site:<18} {'ALL NaN':>10}")
            continue
        print(
            f"{site:<18} {finite.min():10.2f} {finite.max():10.2f} {finite.mean():10.2f} "
            f"{float((finite == 0).mean()):10.4f} {len(finite):10d}"
        )

    # ---- part 2: same instant, different latitude ---------------------------
    # A day-of-year-only column is identical across sites at the same UTC
    # instant. A geometry-dependent one is not. This is the cleanest separator.
    print("\n=== part 2: same UTC instant, different latitude ===")
    usable = [(s, f) for s, f in frames.items() if not f.empty]
    if len(usable) >= 2:
        (site_a, frame_a), (site_b, frame_b) = usable[0], usable[-1]
        merged = frame_a.merge(frame_b, on="TIMESTAMP", suffixes=("_a", "_b"))
        if merged.empty:
            print(f"  no overlapping timestamps between {site_a} and {site_b}")
        else:
            diff = (merged["ext_rad_a"] - merged["ext_rad_b"]).abs()
            print(
                f"  {site_a} (lat {coords[site_a][0]:.2f}) vs {site_b} (lat {coords[site_b][0]:.2f}), "
                f"{len(merged)} shared instants"
            )
            print(f"  mean |difference| = {diff.mean():.3f} W/m2, max = {diff.max():.3f}")
            verdict = (
                "IDENTICAL: day-of-year-only quantity" if diff.max() < 1.0 else "DIFFERS: geometry-dependent quantity"
            )
            print(f"  -> {verdict}")
    else:
        print("  not enough usable sites")

    # ---- part 3: score the candidate formulas -------------------------------
    print("\n=== part 3: candidate formulas vs the delivered column ===")
    for site, (lat, lon, elev) in picks:
        frame = frames.get(site)
        if frame is None or frame.empty:
            continue
        # Subsample: the formulas are deterministic, so a slice is as decisive
        # as the whole series and keeps this inside a small memory footprint.
        sample = frame.iloc[:: max(1, len(frame) // 20000)].copy()
        times = pd.DatetimeIndex(sample["TIMESTAMP"])
        observed = sample["ext_rad"].to_numpy(dtype=float)
        doy = times.dayofyear.to_numpy(dtype=float)

        cand_a = reddyproc_e0(doy)
        solpos = pvlib.solarposition.get_solarposition(times, lat, lon, altitude=elev)
        cos_z = np.cos(np.radians(solpos["apparent_zenith"].to_numpy(dtype=float)))
        e0_pvlib = pvlib.irradiance.get_extra_radiation(times).to_numpy(dtype=float)
        cand_b = np.clip(e0_pvlib * cos_z, 0.0, None)
        cand_c = np.clip(cand_a * cos_z, 0.0, None)

        print(f"\n  {site} (lat {lat:.3f})")
        for label, predicted in (
            ("A E0(doy) only", cand_a),
            ("B pvlib E0 x cos(z)", cand_b),
            ("C RP E0 x cos(z)", cand_c),
        ):
            s = score(observed, predicted)
            print(
                f"    {label:<22} n={s['n']:<7d} rmse={s['rmse']:9.3f}  r={s['r']:7.4f}  "
                f"bias={s['bias']:9.3f}  maxerr={s.get('maxerr', float('nan')):9.3f}"
            )

        # Timestamp convention: sweep whole-hour offsets and report the best.
        # A non-zero winner means TIMESTAMP is not the UTC it was parsed as.
        best = None
        for offset in range(-12, 15):
            shifted = times + pd.Timedelta(hours=offset)
            sp = pvlib.solarposition.get_solarposition(shifted, lat, lon, altitude=elev)
            cz = np.cos(np.radians(sp["apparent_zenith"].to_numpy(dtype=float)))
            pred = np.clip(pvlib.irradiance.get_extra_radiation(shifted).to_numpy(dtype=float) * cz, 0.0, None)
            rmse = score(observed, pred)["rmse"]
            if best is None or (np.isfinite(rmse) and rmse < best[1]):
                best = (offset, rmse)
        print(f"    best whole-hour offset for candidate B: {best[0]:+d} h (rmse {best[1]:.3f})")


if __name__ == "__main__":
    main()
