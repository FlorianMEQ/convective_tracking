# Plotting examples for convective tracking results
# Used in convective_cores_example.ipynb

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from skimage.segmentation import find_boundaries

VIEW = np.s_[0:500, 0:500]

COLORS = {
    "mcs": "lime",
    "core": "red",
    "downdraft": "blue",
}

def plot_all_detections(t, frames, levels, results, PARAMS):
    # Tb
    tb_da = frames[t]["MSG2_IR108"]

    if tb_da.ndim > 2:
        tb_da = tb_da.isel({
            PARAMS.get("vertical_dim", "level"): 0
        })

    tb_full = np.asarray(tb_da.values).squeeze()

    y_start = VIEW[0].start or 0
    x_start = VIEW[1].start or 0

    tb = tb_full[VIEW]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Infrared
    im = ax.imshow(
        tb,
        origin="lower",
        cmap="gray_r",
        alpha=0.9,
        interpolation="nearest",
    )

    for lvl in levels:
        level_name = lvl.name.lower()
        color = COLORS.get(level_name)

        if color is None:
            continue

        objs = results[t]["levels"][lvl.name]
        labels = np.zeros(tb.shape, dtype=int)

        for label, obj in enumerate(objs, start=1):
            indices_2d = obj.get("indices_2d")
            indices_3d = obj.get("indices_3d")

            if indices_2d is not None and len(indices_2d) > 0:
                coords = np.asarray(indices_2d, dtype=int)
            elif indices_3d is not None and len(indices_3d) > 0:
                coords_3d = np.asarray(indices_3d, dtype=int)
                coords = coords_3d[:, 1:3]
            else:
                continue

            coords = coords.copy()
            coords[:, 0] -= y_start
            coords[:, 1] -= x_start

            valid = (
                (coords[:, 0] >= 0)
                & (coords[:, 0] < tb.shape[0])
                & (coords[:, 1] >= 0)
                & (coords[:, 1] < tb.shape[1])
            )

            coords = coords[valid]

            if len(coords) > 0:
                labels[coords[:, 0], coords[:, 1]] = label
        if labels.max() == 0:
            print('no objects for', lvl.name)
            continue
        mask = labels > 0

        overlay = np.ma.masked_where(
            ~mask,
            np.ones(labels.shape),
        )
        ax.imshow(
            overlay,
            origin="lower",
            cmap=ListedColormap([color]),
            alpha=0.35,
            interpolation="nearest",
        )
        boundaries = find_boundaries(labels, mode="inner")

        ax.contour(
            boundaries.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=0.8,
            origin="lower",
        )
    legend_handles = [
        Patch(facecolor=COLORS["mcs"], edgecolor=COLORS["mcs"],
              alpha=0.5, label="MCS"),
        Patch(facecolor=COLORS["core"], edgecolor=COLORS["core"],
              alpha=0.5, label="Core"),
        Patch(facecolor=COLORS["downdraft"], edgecolor=COLORS["downdraft"],
              alpha=0.5, label="Downdraft"),
    ]

    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()

def plot_object_3d(t, level_name, results, object_index=0):
    objs = results[t]["levels"][level_name]

    if object_index < 0 or object_index >= len(objs):
        raise IndexError(
            f"Index {object_index} is out of range for {level_name}"
        )

    obj = objs[object_index]
    if "indices_3d" not in obj or len(obj["indices_3d"]) == 0:
        print("No 3D coords")
        return

    coords = np.asarray(obj["indices_3d"], dtype=int)

    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]

    color = COLORS.get(level_name.lower(), "gray")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        x, y, z,
        c=color,
        s=20,
        alpha=0.6,
        depthshade=False,
    )
    object_id = obj.get("object_id", object_index)

    ax.set_title(
        f"{level_name.upper()} — objet {object_id}\n"
        f"{len(coords)}"
    )

    ax.set_xlabel("X ")
    ax.set_ylabel("Y ")
    ax.set_zlabel("Z")
    ax.set_xlim(x.min() - 2, x.max() + 2)
    ax.set_ylim(y.min() - 2, y.max() + 2)
    ax.set_zlim(z.min() - 1, z.max() + 1)

    plt.tight_layout()
    plt.show()

COLORS = {
    "mcs": "lime",
    "core": "red",
    "downdraft": "blue",
}

VIEW = np.s_[:, :]


def get_object_coords(obj):
    indices_2d = obj.get("indices_2d")
    indices_3d = obj.get("indices_3d")

    if indices_2d is not None and len(indices_2d) > 0:
        return np.asarray(indices_2d, dtype=int)

    if indices_3d is not None and len(indices_3d) > 0:
        return np.asarray(indices_3d, dtype=int)[:, 1:3]

    return np.empty((0, 2), dtype=int)


def plot_frame(t, ax, frames, results_by_time):
    ax.clear()
    tb_da = frames[t]["MSG2_IR108"]

    if tb_da.ndim > 2:
        tb_da = tb_da.isel({
            "level": 0
        })

    tb_full = np.asarray(tb_da.values).squeeze()

    y_start = VIEW[0].start or 0
    x_start = VIEW[1].start or 0
    tb = tb_full[VIEW]

    ax.imshow(
        tb,
        origin="lower",
        cmap="gray_r",
        interpolation="nearest",
    )
    frame_result = results_by_time[t]
    levels_result = frame_result["levels"]

    for level_name, color in COLORS.items():

        if level_name not in levels_result:
            continue

        objects = levels_result[level_name]
        labels = np.zeros(tb.shape, dtype=int)

        for label, obj in enumerate(objects, start=1):
            coords = get_object_coords(obj)

            if len(coords) == 0:
                continue

            coords = coords.copy()
            coords[:, 0] -= y_start
            coords[:, 1] -= x_start

            valid = (
                (coords[:, 0] >= 0)
                & (coords[:, 0] < tb.shape[0])
                & (coords[:, 1] >= 0)
                & (coords[:, 1] < tb.shape[1])
            )

            coords = coords[valid]

            if len(coords) > 0:
                labels[coords[:, 0], coords[:, 1]] = label

        if labels.max() == 0:
            continue

        mask = labels > 0

        overlay = np.ma.masked_where(
            ~mask,
            np.ones(labels.shape),
        )

        ax.imshow(
            overlay,
            origin="lower",
            cmap=ListedColormap([color]),
            alpha=0.35,
            interpolation="nearest",
        )

        boundaries = find_boundaries(labels, mode="inner")

        ax.contour(
            boundaries.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=0.8,
            origin="lower",
        )

        ax.set_title(f"time index {t}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")