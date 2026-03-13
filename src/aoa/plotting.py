"""AOA visualization utilities: DI histograms and spatial AOA maps."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_di_histogram(
    di: np.ndarray,
    threshold: float,
    *,
    di_train: np.ndarray | None = None,
    bins: int = 30,
    figsize: tuple[float, float] = (8, 5),
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot histogram of Dissimilarity Index values with threshold line.

    Parameters
    ----------
    di : (n,) DI values for new/prediction points
    threshold : AOA threshold value
    di_train : optional (n_train,) training DI values to overlay
    bins : number of histogram bins
    figsize : figure size in inches

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    if di_train is not None:
        ax.hist(di_train, bins=bins, alpha=0.5, label="Training DI", color="steelblue")
        ax.hist(di, bins=bins, alpha=0.5, label="Prediction DI", color="coral")
        ax.legend()
    else:
        ax.hist(di, bins=bins, alpha=0.7, label="DI", color="steelblue")

    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label="Threshold")
    ax.set_xlabel("Dissimilarity Index")
    ax.set_ylabel("Count")
    ax.set_title("DI Distribution")
    ax.legend()

    return fig, ax


def plot_aoa_map(
    lats: np.ndarray,
    lons: np.ndarray,
    di_grid: np.ndarray,
    threshold: float,
    *,
    figsize: tuple[float, float] = (12, 6),
    cmap: str = "RdYlGn_r",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot spatial map of DI values with AOA boundary.

    Parameters
    ----------
    lats : (n_lat,) latitude values
    lons : (n_lon,) longitude values
    di_grid : (n_lat, n_lon) DI values on the grid
    threshold : AOA threshold for contour overlay
    figsize : figure size in inches
    cmap : colormap name

    Returns
    -------
    fig, ax : matplotlib Figure and Axes

    Raises
    ------
    ValueError if di_grid shape doesn't match (len(lats), len(lons)).
    """
    if di_grid.shape != (len(lats), len(lons)):
        raise ValueError(
            f"di_grid shape {di_grid.shape} does not match (len(lats), len(lons)) = ({len(lats)}, {len(lons)})"
        )

    fig, ax = plt.subplots(figsize=figsize)

    # Use masked array to handle NaN gracefully
    masked_di = np.ma.masked_invalid(di_grid)

    im = ax.pcolormesh(
        lons,
        lats,
        masked_di,
        cmap=cmap,
        shading="auto",
    )
    fig.colorbar(im, ax=ax, label="Dissimilarity Index")

    # Overlay AOA boundary contour where data exists
    valid_mask = ~np.isnan(di_grid)
    if np.any(valid_mask):
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        # Use filled contour at threshold level to show AOA boundary
        aoa_binary = np.where(np.isnan(di_grid), np.nan, (di_grid <= threshold).astype(float))
        masked_aoa = np.ma.masked_invalid(aoa_binary)
        ax.contour(lon_grid, lat_grid, masked_aoa, levels=[0.5], colors="black", linewidths=1.0)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Area of Applicability")

    return fig, ax
