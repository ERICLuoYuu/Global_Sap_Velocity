# src/fu_ann_sensitivity/plotting.py
"""Fu et al. 2022-style figures for the ANN sensitivity results.

Fig 2 -> ``plot_sensitivity_heatmap``: a 2-D SM-pct x VPD-pct heatmap of one
         sensitivity leg (diverging cmap centred at 0, '*' on significant cells).
Fig 3 -> ``plot_dual_legs``: 1-D dual-leg curves -- both sensitivities (to SM and
         to VPD) vs SM bin (panel a) and vs VPD bin (panel b), median + 25-75 band.
``plot_pft_panels`` -> small-multiples of the Fig-3a dual-leg curves by PFT.

The ``response`` passed in is the data column name (``E``/``Gc``); axis/title text
maps it to a consistent label via ``src.plot_labels`` (``E`` -> "sap flow").

Backend forced to Agg so the figures render headless on Palma.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.plot_labels import response_label

_DIVERGING = "RdBu_r"


def _symmetric_limit(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    lim = float(np.max(np.abs(finite)))
    return lim if lim > 0 else 1.0


def plot_sensitivity_heatmap(map2d, sig_mask, title: str, out_path) -> None:
    """2-D sensitivity heatmap (Fig 2). SM percentile on y, VPD percentile on x."""
    map2d = np.asarray(map2d, dtype=float)
    n = map2d.shape[0]
    lim = _symmetric_limit(map2d)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(map2d, origin="lower", cmap=_DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    if sig_mask is not None:
        for i in range(n):
            for j in range(n):
                if bool(np.asarray(sig_mask)[i, j]) and np.isfinite(map2d[i, j]):
                    ax.text(j, i, "*", ha="center", va="center", color="k", fontsize=9)
    ax.set_xlabel("VPD percentile bin")
    ax.set_ylabel("SM percentile bin")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="sensitivity (z-score units)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_leg_panel(ax, tbl, axis_name: str) -> None:
    x = tbl[axis_name].to_numpy()
    for leg, color, label in [("d_sm", "tab:brown", "to SM (drying)"), ("d_vpd", "tab:blue", "to VPD (rising)")]:
        med = tbl[f"{leg}_median"].to_numpy()
        ax.plot(x, med, "-o", color=color, label=label, markersize=4)
        ax.fill_between(x, tbl[f"{leg}_q25"].to_numpy(), tbl[f"{leg}_q75"].to_numpy(), color=color, alpha=0.18)
        sig = tbl[f"{leg}_sig"].to_numpy().astype(bool)
        if sig.any():
            ax.plot(x[sig], med[sig], "o", color=color, markersize=8, markerfacecolor="none")
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")


def plot_dual_legs(by_sm_bin, by_vpd_bin, response: str, out_path) -> None:
    """Fig 3: both sensitivity legs vs SM bin (a) and vs VPD bin (b)."""
    rlab = response_label(response)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    _plot_leg_panel(ax_a, by_sm_bin, "sm_bin")
    ax_a.set_xlabel("SM percentile bin")
    ax_a.set_ylabel(f"sensitivity of {rlab}")
    ax_a.set_title("(a) at each SM bin")
    _plot_leg_panel(ax_b, by_vpd_bin, "vpd_bin")
    ax_b.set_xlabel("VPD percentile bin")
    ax_b.set_title("(b) at each VPD bin")
    ax_b.legend(loc="best", fontsize=8)
    fig.suptitle(f"Disentangled SM vs VPD sensitivity — {rlab}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pft_panels(per_pft_by_sm_bin: dict, response: str, out_path) -> None:
    """Small-multiples of the Fig-3a dual-leg curves, one panel per PFT."""
    pfts = list(per_pft_by_sm_bin)
    n = max(len(pfts), 1)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow), squeeze=False)
    flat = axes.ravel()
    for idx, pft in enumerate(pfts):  # index-based, not zip() -> py39/Palma safe
        ax = flat[idx]
        _plot_leg_panel(ax, per_pft_by_sm_bin[pft], "sm_bin")
        ax.set_title(pft)
        ax.set_xlabel("SM percentile bin")
    for ax in flat[len(pfts) :]:
        ax.axis("off")
    rlab = response_label(response)
    if pfts:  # avoid an empty-axis legend warning when there are no PFT panels
        flat[0].set_ylabel(f"sensitivity of {rlab}")
        flat[0].legend(loc="best", fontsize=7)
    fig.suptitle(f"Sensitivity by PFT (at each SM bin) — {rlab}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
