from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

STYLE = {
    "cloud": dict(color="#4c78a8", lw=2.0, ls="-"),
    "updraft": dict(color="#f58518", lw=1.6, ls="-"),
    "downdraft": dict(color="#2ca02c", lw=1.6, ls="--"),
    "core": dict(color="#d62728", lw=1.4, ls="-"),
}

def footprint(obj, shape):
    mask = np.zeros(shape, dtype=bool)
    idx = np.asarray(obj["indices_3d"])
    mask[idx[:, 1], idx[:, 2]] = True
    return mask


def plot_scene(frame, y_section=30):
    w = frame["W"].values
    hydro = frame["MRC"].values + frame["MRI"].values
    z_km = frame["level"].values / 1000.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    wmax = np.abs(w).max()

    # signed W extreme of each column (strongest updraft or downdraft)
    w_col = np.where(w.max(axis=0) >= -w.min(axis=0), w.max(axis=0), w.min(axis=0))
    im = ax1.imshow(w_col, origin="lower", cmap="RdBu_r", vmin=-wmax, vmax=wmax)
    ax1.contour(hydro.max(axis=0), levels=[0.05], colors="k", linewidths=1.2)
    ax1.axhline(y_section, color="k", ls=":", lw=0.8)
    for name, (yy, xx) in {"cloud A": (45, 28), "cloud B": (42, 84)}.items():
        ax1.annotate(name, (xx, yy), ha="center", fontsize=9)
    fig.colorbar(im, ax=ax1, label="W (m/s)")

    im2 = ax2.pcolormesh(np.arange(w.shape[2]), z_km, w[:, y_section, :], cmap="RdBu_r",
                         vmin=-wmax, vmax=wmax, shading="nearest")
    ax2.contour(np.arange(w.shape[2]), z_km, hydro[:, y_section, :], levels=[0.05], colors="k", linewidths=1.2)
    fig.colorbar(im2, ax=ax2, label="W (m/s)")
    fig.tight_layout()
    plt.show()


def draw_objects(ax, objects_by_level, shape, styles=STYLE, label_ids=False):
    for level_name, objects in objects_by_level.items():
        style = styles.get(level_name, dict(color="gray", lw=1.0, ls="-"))
        for obj in objects:
            ax.contour(footprint(obj, shape).astype(float), levels=[0.5], colors=[style["color"]],
                       linewidths=style["lw"], linestyles=style["ls"])
            if label_ids:
                ax.annotate(obj["object_id"].split("_", 1)[1], (obj["centroid_x"], obj["centroid_y"]),
                            color=style["color"], fontsize=8, ha="center", va="center", weight="bold")


def legend(ax, level_names, styles=STYLE, **kwargs):
    handles = [Line2D([], [], color=styles[n]["color"], lw=styles[n]["lw"], ls=styles[n]["ls"], label=n)
               for n in level_names if n in styles]
    ax.legend(handles=handles, **kwargs)


def plot_step(frame, step, title=None, label_ids=False, ax=None):
    hydro = (frame["MRC"].values + frame["MRI"].values).max(axis=0)
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.4))
    ax.imshow(hydro, origin="lower", cmap="Greys", vmin=0, vmax=1.5)
    draw_objects(ax, step["levels"], hydro.shape, label_ids=label_ids)
    legend(ax, step["levels"], loc="lower left", fontsize=8)
    if show:
        plt.show()
    return ax
