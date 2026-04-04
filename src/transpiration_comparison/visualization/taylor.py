from __future__ import annotations

"""Taylor diagram visualization using SkillMetrics."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def taylor_diagram(
    reference: xr.DataArray,
    products: dict[str, xr.DataArray],
    title: str,
    output_path: Path,
) -> None:
    """Create a Taylor diagram comparing multiple products against a reference.

    Parameters
    ----------
    reference : xr.DataArray
        Reference field (flattened over time, space as needed).
    products : dict
        product_name -> DataArray of same shape as reference.
    title : str
        Plot title.
    output_path : Path
        Where to save the figure.
    """
    try:
        import skill_metrics as sm
    except ImportError:
        logger.warning("SkillMetrics not installed. Using manual Taylor diagram.")
        _manual_taylor(reference, products, title, output_path)
        return

    # Compute statistics for each product
    ref_flat = reference.values.ravel()
    ref_valid = np.isfinite(ref_flat)

    stats_list = []
    labels = ["Reference"]

    for name, prod_da in products.items():
        prod_flat = prod_da.values.ravel()
        valid = ref_valid & np.isfinite(prod_flat)
        if valid.sum() < 100:
            continue

        r = ref_flat[valid]
        p = prod_flat[valid]
        stats_list.append(
            {
                "name": name,
                "std": np.std(p),
                "corr": np.corrcoef(r, p)[0, 1],
                "crmsd": np.sqrt(np.mean(((p - np.mean(p)) - (r - np.mean(r))) ** 2)),
            }
        )
        labels.append(name)

    if not stats_list:
        logger.warning("No valid product data for Taylor diagram")
        return

    ref_std = np.std(ref_flat[ref_valid])

    # Build arrays for SkillMetrics
    sdevs = [ref_std] + [s["std"] for s in stats_list]
    crmsd_arr = [0.0] + [s["crmsd"] for s in stats_list]
    corrs = [1.0] + [s["corr"] for s in stats_list]

    fig = plt.figure(figsize=(10, 10))
    sm.taylor_diagram(
        np.array(sdevs),
        np.array(crmsd_arr),
        np.array(corrs),
        markerLabel=labels,
        markerLabelColor="black",
        markerColor="red",
        styleOBS="-",
        colOBS="black",
        titleOBS="Reference",
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def _manual_taylor(
    reference: xr.DataArray,
    products: dict[str, xr.DataArray],
    title: str,
    output_path: Path,
) -> None:
    """Simplified Taylor diagram without SkillMetrics dependency."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    ref_flat = reference.values.ravel()
    ref_valid = np.isfinite(ref_flat)
    ref_std = np.std(ref_flat[ref_valid])

    colors = plt.cm.Set1(np.linspace(0, 1, len(products)))

    for idx, (name, prod_da) in enumerate(products.items()):
        prod_flat = prod_da.values.ravel()
        valid = ref_valid & np.isfinite(prod_flat)
        if valid.sum() < 100:
            continue

        r = ref_flat[valid]
        p = prod_flat[valid]
        corr = np.corrcoef(r, p)[0, 1]
        std = np.std(p)

        # Taylor diagram: angle = arccos(correlation), radius = std
        theta = np.arccos(max(-1, min(1, corr)))
        ax.plot(theta, std, "o", color=colors[idx], markersize=10, label=name)

    # Reference point
    ax.plot(0, ref_std, "k*", markersize=15, label="Reference")

    ax.set_thetamax(180)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")
