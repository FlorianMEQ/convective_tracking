from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import label as cc_label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


def z_indices_between(level_values, zmin_m, zmax_m):
    """Return indices where level values are inside [zmin_m, zmax_m]."""
    level_values = level_values.astype(float)
    return np.nonzero((level_values >= zmin_m) & (level_values <= zmax_m))[0]


def get_structure_2d(connectivity = 8):
    """Build a 2D connectivity structure for 4 or 8-neighborhood."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be one of: 4, 8")
    return generate_binary_structure(2, 1 if connectivity == 4 else 2)


def get_structure_3d(connectivity = 18):
    """Build a 3D connectivity structure for 6, 18 or 26-neighborhood."""
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be one of: 6, 18, 26")
    return generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])


def connected_components_2d(mask, connectivity = 8):
    """Label 2D connected components on a boolean mask."""
    return cc_label(mask.astype(bool), structure=get_structure_2d(connectivity))


def connected_components_3d(mask, connectivity = 18):
    """Label 3D connected components on a boolean mask."""
    return cc_label(mask.astype(bool), structure=get_structure_3d(connectivity))


_METRIC_KEYS = (
    "voxels", "area_px", "area_km2", "width_px", "width_km", "height_levels",
    "height_km", "volume_km3", "width_median_px", "width_median_km",
)


def _axis_extent(occupied):
    """Extent (first to last True, inclusive) along axis 1 for each row of a 2D boolean array."""
    return occupied.shape[1] - occupied[:, ::-1].argmax(axis=1) - occupied.argmax(axis=1)


def compute_component_metrics(
    mask3d,
    level_vals = None,
    dx_km = 1.0,
):
    """Compute geometric metrics for one 3D component (axes: z, y, x)."""
    proj = mask3d.any(axis=0)
    area_px = int(proj.sum())
    if area_px == 0:
        return dict.fromkeys(_METRIC_KEYS, 0)

    ys, xs = np.where(proj)
    width_px = int(max(np.ptp(xs), np.ptp(ys)) + 1)

    z_idx = np.where(mask3d.any(axis=(1, 2)))[0]
    height_levels = int(z_idx[-1] - z_idx[0] + 1)
    width_x = _axis_extent(mask3d.any(axis=1)[z_idx])
    width_y = _axis_extent(mask3d.any(axis=2)[z_idx])
    width_median_px = float(np.median(np.maximum(width_x, width_y)))

    height_km = volume_km3 = 0.0
    if level_vals is not None and len(level_vals) > 1:
        dz_m = np.diff(level_vals)
        dz_km = np.append(dz_m, dz_m[-1]) / 1000.0
        height_km = float(dz_km[z_idx].sum())
        volume_km3 = float((mask3d.sum(axis=(1, 2)) * dz_km).sum()) * dx_km**2

    return {
        "voxels": int(mask3d.sum()),
        "area_px": area_px,
        "area_km2": area_px * dx_km**2,
        "width_px": width_px,
        "width_km": width_px * dx_km,
        "height_levels": height_levels,
        "height_km": height_km,
        "volume_km3": volume_km3,
        "width_median_px": width_median_px,
        "width_median_km": width_median_px * dx_km,
    }


def iou_from_sets(a, b):
    """IoU between two arrays of unique integer keys."""
    inter = len(np.intersect1d(a, b, assume_unique=False))
    union = len(a) + len(b) - inter
    return float(inter / union) if union > 0 else 0.0


def pack_indices(idx, grid_size=10000):
    if idx.size == 0:
        return np.array([], dtype=np.int64)
    idx = idx.astype(np.int64)
    if idx.shape[1] == 2:
        return idx[:, 0] * grid_size + idx[:, 1]
    elif idx.shape[1] == 3:
        return idx[:, 0] * (grid_size ** 2) + idx[:, 1] * grid_size + idx[:, 2]
    else:
        raise ValueError("pack_indices only supports 2 or 3 columns")

def advect_indices(idx, dy_pixels, dx_pixels):
    """Shift the last two (spatial y, x) coordinate columns of an index array."""
    shifted = idx.copy()
    shifted[:, -2] += int(round(dy_pixels))
    shifted[:, -1] += int(round(dx_pixels))
    return shifted


def merge_stacked_seeds(
    markers: np.ndarray,
    field: np.ndarray,
    xy_radius = 3,
):
    """Merge vertically-stacked watershed seeds whose horizontal (axis 0) projections overlap.

    Two seeds are considered stacked if their projections along axis 0 overlap
    after a disk dilation of radius ``xy_radius``. The most intense seed
    (highest value in ``field``) absorbs the others. Fully generic: works on
    any 3D labeled marker array, independent of the physical meaning of axis 0.
    """
    labels = np.unique(markers)
    labels = labels[labels > 0]
    if labels.size <= 1:
        return markers, int(labels.size)

    r = max(0, int(xy_radius))
    cy, cx = np.ogrid[-r : r + 1, -r : r + 1]
    disk = (cy**2 + cx**2) <= r * r

    projections = {int(lbl): np.any(markers == lbl, axis=0) for lbl in labels}
    peaks = {int(lbl): field[markers == lbl].max() for lbl in labels}
    order = sorted(projections, key=peaks.get, reverse=True)

    parent = {lbl: lbl for lbl in order}
    for i, strong in enumerate(order):
        if parent[strong] != strong:
            continue
        dilated = binary_dilation(projections[strong], structure=disk)
        for weak in order[i + 1 :]:
            if parent[weak] == weak and np.any(dilated & projections[weak]):
                parent[weak] = strong

    roots = sorted(set(parent.values()))
    new_id = {root: i for i, root in enumerate(roots, start=1)}
    merged = np.zeros_like(markers)
    for lbl, root in parent.items():
        merged[markers == lbl] = new_id[root]
    return merged, len(roots)


def recursive_watershed_segmentation(
    indices: np.ndarray,
    values: np.ndarray,
    *,
    compute_adaptive_h: Callable[[np.ndarray], float],
    stop_condition: Callable[[np.ndarray], bool],
    connectivity = 18,
    merge_stacked = True,
    stacked_seeds_xy_radius = 3,
    id_prefix = "root",
    _depth = 0,
    max_depth=3,
):
    """Generic recursive H-maxima watershed segmentation of a labeled 3D point set.
    Parameters
    ----------
    indices : ndarray (n, 3)
        Global integer coordinates (a0, a1, a2) of the point set to segment.
    values : ndarray (n,)
        Intensity value at each point (used to build the local watershed field).
    compute_adaptive_h : callable
        Given the array of local values, returns the H-maxima height threshold.
    stop_condition : callable
        Given a local boolean mask, returns True if recursion should stop.
    connectivity : int
        3D connectivity used for both H-maxima footprint and watershed labeling.
    merge_stacked : bool
        Whether to merge vertically-stacked seeds before watershed.
    stacked_seeds_xy_radius : int
        Disk radius (pixels) used for stacked-seed merging.
    id_prefix : str
        Identifier prefix propagated to (and suffixed by) child subgroups.

    Returns
    -------
    list of dict
        Each dict has keys ``"id"``, ``"indices"`` (list of (a0, a1, a2)
        tuples) and ``"values"`` (list of floats).
    """
    if indices.size == 0 or _depth >= max_depth:
        return [{"id": id_prefix, "indices": indices.tolist(), "values": values.tolist()}]

    origin = indices.min(axis=0)
    grid = np.zeros(indices.max(axis=0) - origin + 1, dtype=float)
    grid[tuple((indices - origin).T)] = values

    mask = grid > 0
    if stop_condition(mask):
        return [{"id": id_prefix, "indices": indices.tolist(), "values": values.tolist()}]

    structure = get_structure_3d(connectivity)
    local_maxima = h_maxima(grid, compute_adaptive_h(values), footprint=structure)
    markers, n_markers = cc_label(local_maxima, structure=structure)

    if merge_stacked and n_markers > 1:
        markers, n_markers = merge_stacked_seeds(markers, grid, xy_radius=stacked_seeds_xy_radius)

    segmentation = watershed(-grid, markers, mask=mask)
    unique_labels = np.unique(segmentation)
    unique_labels = unique_labels[unique_labels > 0]
    if unique_labels.size <= 1:
        return [{"id": id_prefix, "indices": indices.tolist(), "values": values.tolist()}]

    subgroups: List[Dict[str, Any]] = []
    for label_id in unique_labels:
        comp = segmentation == label_id
        subgroups.extend(
            recursive_watershed_segmentation(
                np.argwhere(comp) + origin,
                grid[comp],
                compute_adaptive_h=compute_adaptive_h,
                stop_condition=stop_condition,
                connectivity=connectivity,
                merge_stacked=merge_stacked,
                stacked_seeds_xy_radius=stacked_seeds_xy_radius,
                id_prefix=f"{id_prefix}_d{_depth}_s{int(label_id)}",
                _depth=_depth + 1,
            )
        )
    return subgroups
