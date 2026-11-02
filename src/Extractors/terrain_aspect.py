"""Slope and aspect at a site: the pixel the site is in, not a 250 m average.

``extract_canopyheight.py`` used to average aspect over a 250 m buffer with
``ee.Reducer.mean()``.  Aspect is a CIRCULAR quantity, and an arithmetic mean of
degrees is meaningless across the 0/360 seam: a north-facing slope whose pixels
read 355, 0 and 5 degrees averages to 120 -- east-south-east.  Measured over the
226-site corpus the artefact was unmistakable: 106 sites landed in the "S" sector
and **not one** in "N".

Averaging is also the wrong operation on its own terms.  The median vector
strength over those 250 m windows is 0.277, and 162 of 224 windows fall below
0.5, so most sites have no coherent mean direction to report even if the circular
mean were used.  A single pixel is a statement about one place; a mean over
incoherent directions is a statement about nowhere.

So: read the pixel containing the site, and read it at the DEM's own scale.

TWO TRAPS, both measured rather than assumed:

1.  THE SCALE IS NOT A LITERAL.  Earth Engine computes terrain products *in the
    projection you ask for*, so ``scale`` does not select a pixel -- it defines
    the grid the elevation surface is resampled onto before it is
    differentiated.  ``extract_canopyheight.py`` hardcoded ``scale=30`` while
    ASTER GDEM is 1 arc-second, or 30.922080776 m: over the 226 sites that moved
    aspect at 110 (max 160.6 deg) and slope at 113 (max 13.7 deg).

    Rounding is not safe either.  Asking for 30.922081 m -- the same number to
    six decimals -- instead of the DEM's own 30.922080776 m returns slope 6.61
    deg / aspect 192.16 deg at FRA_FON where the exact scale gives 3.96 / 200.66.
    A difference of 2e-7 m in the request, 8.5 degrees in the answer.  So the
    scale is taken from the image itself (:func:`native_scale`), as a server-side
    ``ee.Number`` that never round-trips through a Python float.
    :data:`DEM_NOMINAL_SCALE_M` records the expected value for review and
    :func:`assert_dem_scale` alarms if the product moves under it, but neither is
    what gets sent.

2.  FLAT GROUND.  Earth Engine answers ``aspect = 0`` where the gradient
    vanishes, and 0 reads as due north.  ASTER GDEM is integer metres, so this is
    not hypothetical: three corpus sites (GBR_DEV_CON, GBR_DEV_DRO, USA_WVF) have
    a dead-flat centre pixel.  The flat guard therefore has to be judged on the
    SAME pixel the aspect came from -- USA_WVF's 250 m buffered slope is 11.5
    degrees, so a guard fed the buffered slope waves all three through and
    invents three north-facing sites.  That would be the bug being fixed here,
    mirrored.

``aspect_sin``/``aspect_cos`` are trained model features, so this module's output
reaches the model; ``aspect_deg`` and ``aspect_category`` are reporting only.
"""

from __future__ import annotations

import numpy as np

#: The scale Earth Engine reports for ``projects/sat-io/open-datasets/ASTER/GDEM``
#: (1 arc-second on EPSG:4326).  Recorded so the expected value is reviewable and
#: so :func:`assert_dem_scale` can alarm if the asset is regridded.  It is NOT
#: what gets sent: see trap 1 in the module docstring -- a rounded copy of a
#: native scale is a different request, not the same one.
DEM_NOMINAL_SCALE_M = 30.922081

#: How far the asset's reported scale may sit from the constant above before it
#: is treated as stale rather than merely rounded.
DEM_SCALE_TOLERANCE_M = 1e-4

#: Below this slope the surface has no meaningful direction and Earth Engine's
#: aspect is an artefact of a vanishing gradient, not a bearing.  Kept at the
#: value the original extractor used.
FLAT_SLOPE_DEG = 0.1

#: Compass sectors in 45-degree bins, N centred on 0.
SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def compass_sector(aspect_deg):
    """Sector name for an aspect in degrees, or None if there is no aspect.

    N spans 337.5..22.5, so the bins are offset by half a sector before binning.
    """
    if aspect_deg is None:
        return None
    value = float(aspect_deg)
    if np.isnan(value):
        return None
    return SECTORS[int(((value % 360) + 22.5) // 45) % 8]


def standardize_aspect(aspect_value, slope_value):
    """Aspect as a bearing, a sector, and a unit vector.

    Parameters
    ----------
    aspect_value:
        Aspect in degrees clockwise from north, as Earth Engine reports it.
    slope_value:
        Slope in degrees **of the same pixel the aspect came from**.  Passing a
        slope from a different footprint defeats the flat-terrain guard; see the
        module docstring.

    Returns
    -------
    dict with ``aspect_standardized``, ``aspect_category``, ``aspect_sin`` and
    ``aspect_cos``.

    On flat ground the sin/cos pair is the ZERO VECTOR rather than null.  That is
    deliberate and it is the right encoding: on the unit circle a zero-magnitude
    vector carries no direction, which is exactly the claim being made, and it
    keeps the feature numeric so the site is not dropped from training for having
    level ground.  A missing aspect *reading*, by contrast, stays null -- that is
    absence of data, not absence of direction.
    """
    if slope_value is None or np.isnan(slope_value) or abs(slope_value) < FLAT_SLOPE_DEG:
        return {
            "aspect_standardized": None,
            "aspect_category": None,
            "aspect_sin": 0.0,
            "aspect_cos": 0.0,
        }

    if aspect_value is None or np.isnan(aspect_value):
        return {
            "aspect_standardized": None,
            "aspect_category": None,
            "aspect_sin": None,
            "aspect_cos": None,
        }

    standardized = float(aspect_value) % 360
    return {
        "aspect_standardized": standardized,
        "aspect_category": compass_sector(standardized),
        # sin/cos of the RAW reading: the modulo above only renames 360 as 0, and
        # sin/cos are already periodic, so both give the same vector.
        "aspect_sin": float(np.sin(np.radians(aspect_value))),
        "aspect_cos": float(np.cos(np.radians(aspect_value))),
    }


def native_scale(image):
    """The image's own pixel size, as a server-side ``ee.Number``.

    Deliberately NOT a Python float.  ``nominalScale().getInfo()`` hands back a
    number that has to be written down to be reused, and a written-down copy of a
    native scale is a different request: at FRA_FON, 30.922081 m instead of the
    DEM's 30.922080776 m returns slope 6.61 deg / aspect 192.16 deg where the
    exact value gives 3.96 / 200.66.  Keeping the number server-side removes the
    opportunity to round it.
    """
    return image.projection().nominalScale()


def assert_dem_scale(elevation_image):
    """Alarm if the DEM's scale has moved away from :data:`DEM_NOMINAL_SCALE_M`.

    Returns the scale Earth Engine reports.  This is a staleness check on the
    documented constant, not the source of the sampling scale -- requests use
    :func:`native_scale`.  Raises ``ValueError`` on a real difference, because
    sampling off a product's own grid does not fail, it silently resamples the
    elevation surface and differentiates the resample.
    """
    reported = elevation_image.projection().nominalScale().getInfo()
    if reported is None:
        raise ValueError("the DEM asset would not report a nominal scale")
    reported = float(reported)
    if abs(reported - DEM_NOMINAL_SCALE_M) > DEM_SCALE_TOLERANCE_M:
        raise ValueError(
            f"the DEM reports a nominal scale of {reported:.9f} m, but DEM_NOMINAL_SCALE_M records "
            f"{DEM_NOMINAL_SCALE_M:.9f} m. The asset appears to have been regridded; update the constant."
        )
    return reported


def terrain_bands(elevation_image):
    """Slope and aspect of ``elevation_image`` as one two-band image.

    One image means one request: :func:`centre_pixel_terrain` reduces both bands
    together, so slope and aspect are guaranteed to describe the same pixel.
    Reducing them separately is what allowed the flat guard to be judged on a
    different footprint from the aspect it was guarding.
    """
    import ee

    return ee.Terrain.slope(elevation_image).addBands(ee.Terrain.aspect(elevation_image))


def centre_pixel_terrain(terrain, points, scale, key="site_name"):
    """Slope and aspect of the pixel containing each point, in ONE request.

    Parameters
    ----------
    terrain:
        Two-band image from :func:`terrain_bands`.
    points:
        Iterable of ``(key_value, lon, lat)``.
    scale:
        Required, and it should be :func:`native_scale` of the source elevation
        image.  There is no default on purpose: a default invites a rounded
        literal, and a rounded scale is a different grid (module docstring, trap
        1).
    key:
        Property name used to label each point in the request.

    Returns
    -------
    dict mapping each key value to ``(slope_deg, aspect_deg)``, with ``None`` for
    either where Earth Engine returned nothing.

    ``Reducer.first()`` over an unbuffered point is the single pixel containing
    it.  Note that ``first()`` names its outputs after the BANDS only while the
    image has more than one band -- on a single-band image the property comes
    back as ``"first"`` instead, which is a trap worth knowing: a probe that
    selected one band and then looked for it by name reported null for every site
    and very nearly retired a working call shape.
    """
    import ee

    features = [
        ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {key: key_value}) for key_value, lon, lat in points
    ]
    if not features:
        return {}

    reduced = terrain.reduceRegions(
        collection=ee.FeatureCollection(features),
        reducer=ee.Reducer.first(),
        scale=scale,
    ).getInfo()

    out = {}
    for feature in reduced.get("features", []):
        props = feature.get("properties", {})
        out[props.get(key)] = (props.get("slope"), props.get("aspect"))
    return out


def terrain_attributes_at_point(terrain, key_value, lon, lat, scale):
    """:func:`centre_pixel_terrain` for a single point, plus the standardisation.

    Returns ``aspect_deg``, ``slope_deg_centre`` and the four standardised
    fields.  The per-site and batched callers share one implementation on
    purpose: the last time this repository grew a second private copy of a
    mapping, the copy kept a unit bug alive after the original was fixed.
    """
    got = centre_pixel_terrain(terrain, [(key_value, lon, lat)], scale)
    slope_deg, aspect_deg = got.get(key_value, (None, None))
    result = {"aspect_deg": aspect_deg, "slope_deg_centre": slope_deg}
    result.update(standardize_aspect(aspect_deg, slope_deg))
    return result
