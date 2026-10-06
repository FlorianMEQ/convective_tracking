"""Generic object detection, driven by a declarative rule.

:func:`detect_objects` is the single detector used by every
:class:`~.level.TrackingLevel`. What an "object" is (MCS, convective core, ...)
is decided only by the rule stored in ``level.params``; this module knows
nothing about the physical meaning of the variables.

Rule keys
---------
Grid
    ``dx_km`` (1.0)                horizontal grid spacing.
    ``vertical_dim`` ("level")     vertical dimension of 3D variables.
    ``vertical_dim_w`` ("level_w") vertical coordinate holding layer heights (m),
                                   needed for ``height_km`` and ``volume_km3``.
    ``min_alt_km`` / ``max_alt_km`` optional vertical clipping (km).

Mask (required)
    ``conditions``: list of ``{"variables": name | [names], "op": ">", "value": x}``,
    combined with AND. Several variables are summed before the comparison;
    variables absent from the dataset are ignored. The mask is 2D or 3D
    depending on the variables. With a parent object, the mask is restricted to
    the parent's footprint (``indices_2d`` / ``indices_3d``).

Objects
    ``connectivity``   2D: 4 or 8 (default 8); 3D: 6, 18 or 26 (default 26).
    ``split``          optional, one of:
        ``{"method": "seeded_watershed", "variable": v, "mode": "min" | "max",
        "seed_h" | "seed_percentile" | "seed_threshold": x, "outside_value": y}``
        Seeds are the extrema of ``v`` and objects the watershed basins within
        the mask (components without seed are dropped). ``outside_value``
        (default: the extremum of ``v`` in the mask, plus 1) fills the
        pixels outside the mask before the ``seed_h`` selection.
        ``{"method": "recursive_h_maxima", "variable": v, "h_slope", "h_min",
        "h_max", "h_reference", "min_width_km", "connectivity",
        "stacked_seeds_xy_radius"}``
        Components are recursively split until their median width is at most
        ``min_width_km``; ``v`` must be positive inside the mask.
    ``min_voxels``, ``min_area_px``, ``min_width_px``, ``min_area_km2``
                       size filters (default 0).
    ``stats``          list of ``{"name": key, "variable": v, "stat": s}`` with ``s`` in
                       mean, min, max, percentile (with ``"q"``) or layer_mean
                       (mean within ``half_depth`` (2) levels around the vertical
                       centroid of a 3D object, over the whole column for a 2D one).
                       A variable absent from the dataset gives 0.0.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import find_objects
from skimage.morphology import h_minima
from skimage.segmentation import watershed

from .functions import (
    compute_component_metrics,
    connected_components_2d,
    connected_components_3d,
    recursive_watershed_segmentation,
    z_indices_between,
)
from .level import DetectionContext, TrackingLevel

Rule = Dict[str, Any]


def _cached(context, key, compute):
    """Compute once per timestep: ``context`` is fresh for every timestep."""
    cache = context.metadata.setdefault("_fields_cache", {})
    if key not in cache:
        cache[key] = compute()
    return cache[key]


def _vertical_indices(context, rule):
    coords = np.asarray(context.dataset.coords[rule.get("vertical_dim", "level")].values, dtype=float)
    low, high = rule.get("min_alt_km"), rule.get("max_alt_km")
    return z_indices_between(
        coords,
        -np.inf if low is None else low * 1000.0,
        np.inf if high is None else high * 1000.0,
    )


def _field(context, rule, name):
    """Values of a variable (vertically clipped if 3D), or None if absent from the dataset."""
    ds = context.dataset
    if name not in ds:
        return None

    vertical_dim = rule.get("vertical_dim", "level")
    if vertical_dim not in ds[name].dims:
        return _cached(context, (name,), lambda: ds[name].values)

    clip = (rule.get("min_alt_km"), rule.get("max_alt_km"))
    return _cached(
        context,
        (name, vertical_dim, clip),
        lambda: ds[name].isel({vertical_dim: _vertical_indices(context, rule)}).values,
    )


def _required_field(context, rule, name):
    field = _field(context, rule, name)
    if field is None:
        raise KeyError(f"Variable '{name}' of the split rule is not in the dataset")
    return field


def _layer_heights(context, rule):
    dim_w = rule.get("vertical_dim_w", "level_w")
    if dim_w not in context.dataset.coords:
        return None
    clip = (rule.get("min_alt_km"), rule.get("max_alt_km"))
    return _cached(
        context,
        ("_heights", dim_w, clip),
        lambda: np.asarray(
            context.dataset[dim_w].isel({dim_w: _vertical_indices(context, rule)}).values, dtype=float
        ),
    )


def _build_mask(context, rule):
    mask = None
    for condition in rule["conditions"]:
        names = condition["variables"]
        names = [names] if isinstance(names, str) else names
        fields = [f for f in (_field(context, rule, name) for name in names) if f is not None]
        if not fields:
            return None
        op = condition.get("op", ">")
        if op == ">":
            passed = sum(fields[1:], fields[0]) > float(condition["value"])
        elif op == ">=":
            passed = sum(fields[1:], fields[0]) >= float(condition["value"])
        elif op == "<":
            passed = sum(fields[1:], fields[0]) < float(condition["value"])
        elif op == "<=":
            passed = sum(fields[1:], fields[0]) <= float(condition["value"])
        else:
            raise ValueError(f"Unsupported comparison operator: {op}")
        mask = passed if mask is None else mask & passed
    return mask


def _restrict_to_parent(mask, parent_object):
    """Keep the mask inside the parent's footprint (broadcast over axes the parent does not have)."""
    parent_idx = next(
        (parent_object[key] for key in ("indices_3d", "indices_2d") if parent_object.get(key)), None
    )
    if parent_idx is None:
        return None
    parent_idx = np.asarray(parent_idx, dtype=np.int64)
    ndim = min(parent_idx.shape[1], mask.ndim)
    footprint = np.zeros(mask.shape[-ndim:], dtype=bool)
    footprint[tuple(parent_idx[:, -ndim:].T)] = True
    return mask & footprint


def _connected_components(mask, connectivity):
    if mask.ndim == 2:
        return connected_components_2d(mask, connectivity or 8)
    return connected_components_3d(mask, connectivity or 26)


def _seeded_watershed(
    context, rule, mask, split, connectivity
):
    """Watershed basins of ``split["variable"]`` within the mask, one per seed component."""
    sign = -1.0 if split.get("mode", "min") == "max" else 1.0
    field = sign * _required_field(context, rule, split["variable"])

    if "seed_h" in split:
        outside = sign * split["outside_value"] if "outside_value" in split else field[mask].max() + 1.0
        seeds = h_minima(np.where(mask, field, outside), float(split["seed_h"])).astype(bool)
    elif "seed_percentile" in split:
        inside = field[mask]
        seeds = field <= float(np.percentile(inside, split["seed_percentile"])) if inside.size else np.zeros_like(mask)
    else:
        seeds = field <= sign * float(split["seed_threshold"])

    seed_labels, n_seeds = _connected_components(seeds & mask, connectivity)
    return watershed(field, seed_labels, mask=mask) if n_seeds else np.zeros(mask.shape, dtype=np.int32)


def _label_groups(labels):
    """Global coordinates (n, ndim) of each labeled object, by increasing label."""
    groups = []
    for label, box in enumerate(find_objects(labels), start=1):
        if box is not None:
            offset = np.array([axis.start for axis in box])
            groups.append(np.argwhere(labels[box] == label) + offset)
    return groups


def _bbox_mask(idx):
    """Boolean mask cropped to the bounding box of the coordinates, and the box origin."""
    origin = idx.min(axis=0)
    mask = np.zeros(idx.max(axis=0) - origin + 1, dtype=bool)
    mask[tuple((idx - origin).T)] = True
    return mask, origin


def _measure(idx, heights, dx_km):
    mask, origin = _bbox_mask(idx)
    if mask.ndim == 2:
        return compute_component_metrics(mask[None], dx_km=dx_km)
    layers = None if heights is None else heights[origin[0] : origin[0] + mask.shape[0]]
    return compute_component_metrics(mask, level_vals=layers, dx_km=dx_km)


def _is_large_enough(metrics, rule):
    return (
        metrics["voxels"] >= rule.get("min_voxels", 0)
        and metrics["area_px"] >= rule.get("min_area_px", 0)
        and metrics["width_px"] >= rule.get("min_width_px", 0)
        and metrics["area_km2"] >= rule.get("min_area_km2", 0.0)
    )


def _recursive_split(
    context, rule, split, idx, dx_km
):
    field = _required_field(context, rule, split["variable"])
    reference = float(split.get("h_reference", rule["conditions"][0]["value"]))
    h_slope = float(split.get("h_slope", 0.3))
    h_min = float(split.get("h_min", 1.0))
    h_max = float(split.get("h_max", 10.0))
    min_width_km = float(split.get("min_width_km", 1.0))

    def compute_adaptive_h(values: np.ndarray) -> float:
        return float(min(max(h_slope * (float(values.max()) - reference), h_min), h_max))

    def stop_condition(mask: np.ndarray) -> bool:
        return compute_component_metrics(mask, dx_km=dx_km)["width_median_km"] <= min_width_km

    subgroups = recursive_watershed_segmentation(
        idx,
        field[tuple(idx.T)],
        compute_adaptive_h=compute_adaptive_h,
        stop_condition=stop_condition,
        connectivity=int(split.get("connectivity", 18)),
        merge_stacked=True,
        stacked_seeds_xy_radius=int(split.get("stacked_seeds_xy_radius", 3)),
    )
    return [np.array(group["indices"], dtype=np.int64) for group in subgroups]


def _values_at(field, idx):
    """Field values at the object's coordinates; axes the object lacks are kept whole."""
    ndim = idx.shape[1]
    if field.ndim <= ndim:
        return field[tuple(idx[:, ndim - field.ndim :].T)]
    return field[(slice(None),) * (field.ndim - ndim) + tuple(idx.T)]


def _layer_mean(field, idx, half_depth):
    if idx.shape[1] != 3 or field.ndim != 3:
        return float(np.nanmean(_values_at(field, idx)))

    z = idx[:, 0]
    center = int(round(float(z.mean())))
    in_layer = (z >= max(int(z.min()), center - half_depth)) & (z <= min(int(z.max()), center + half_depth))
    if not in_layer.any():
        in_layer[:] = True
    return float(np.nanmean(field[tuple(idx[in_layer].T)]))


def _statistic(stat, field, idx):
    kind = stat["stat"]
    if kind == "layer_mean":
        return _layer_mean(field, idx, int(stat.get("half_depth", 2)))

    values = _values_at(field, idx)
    if kind == "percentile":
        return float(np.nanpercentile(values, stat["q"]))
    return float({"mean": np.nanmean, "min": np.nanmin, "max": np.nanmax}[kind](values))


def _describe(
    context, rule, idx, metrics
):
    ndim = idx.shape[1]
    description: Dict[str, Any] = {
        "area_km2": metrics["area_km2"],
        "width_km": metrics["width_km"],
        "centroid_x": float(idx[:, -1].mean()),
        "centroid_y": float(idx[:, -2].mean()),
    }
    if ndim == 3:
        description.update(
            volume_km3=metrics["volume_km3"],
            height_km=metrics["height_km"],
            centroid_z=float(idx[:, 0].mean()),
        )
    for stat in rule.get("stats", []):
        field = _field(context, rule, stat["variable"])
        description[stat["name"]] = 0.0 if field is None else _statistic(stat, field, idx)
    description[f"indices_{ndim}d"] = list(map(tuple, idx.tolist()))
    return description


def detect_objects(
    context,
    parent_object,
    level,
):
    """Detect the objects of one level in the current timestep (see the module docstring for the rule)."""
    rule = level.params
    mask = _build_mask(context, rule)
    if mask is not None and parent_object is not None:
        mask = _restrict_to_parent(mask, parent_object)
    if mask is None or not mask.any():
        return []

    dx_km = float(rule.get("dx_km", 1.0))
    heights = _layer_heights(context, rule)
    split = rule.get("split") or {}
    method = split.get("method")

    if method == "seeded_watershed":
        labels = _seeded_watershed(context, rule, mask, split, rule.get("connectivity"))
    else:
        labels, _ = _connected_components(mask, rule.get("connectivity"))
    groups = _label_groups(labels)

    def keep(group: np.ndarray) -> Optional[Dict[str, float]]:
        if len(group) < rule.get("min_voxels", 0):
            return None
        metrics = _measure(group, heights, dx_km)
        return metrics if _is_large_enough(metrics, rule) else None

    if method == "recursive_h_maxima":
        groups = [
            part
            for group in groups
            if keep(group) is not None
            for part in _recursive_split(context, rule, split, group, dx_km)
        ]

    objects = []
    for group in groups:
        metrics = keep(group)
        if metrics is not None:
            objects.append(_describe(context, rule, group, metrics))
    return objects
