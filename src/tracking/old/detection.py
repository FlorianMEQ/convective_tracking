from __future__ import annotations
"""Generic 3D object detection, driven by a declarative rule.

Uses:
    from .level import DetectionContext, TrackingLevel

:func:`detect_objects` is the single detector used by every
:class:`~.level.TrackingLevel`. What an "object" is (MCS, convective core, ...)
is decided only by the rule stored in ``level.params``; this module knows
nothing about the physical meaning of the variables. All detected objects are
3D: a 2D variable (e.g. a column quantity such as brightness temperature) can
be used in a condition, it is simply broadcast over the vertical dimension.

The caller gets back the object's geometry (``indices_3d``, extent, centroid)
and is expected to compute and analyze whatever physical quantities it needs
from those voxel coordinates; this module does not compute them.

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
    variables absent from the dataset are ignored. With a parent object, the
    mask is restricted to the parent's footprint (``indices_3d``).

Objects
    ``connectivity``   3D connected-component connectivity: 6, 18 or 26 (default 26).
    ``split``          optional name of a variable. When given, each connected
                       component is further split by a recursive H-maxima
                       watershed on that variable (``v`` must be positive
                       inside the mask); this is the only splitting method.
    ``min_voxels``, ``min_area_px``, ``min_width_px``, ``min_area_km2``
                       size filters (default 0).
"""
from typing import Optional, List, Dict, Any
import numpy as np
from .level import DetectionContext, TrackingLevel
from .functions import (
    compute_component_metrics,
    connected_components_3d,
    recursive_watershed_segmentation,
    z_indices_between,
)
import os
Rule = None

_COMPARATORS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}

_SPLIT_H_SLOPE = 0.3
_SPLIT_H_MIN = 1.0
_SPLIT_H_MAX = 10.0
_SPLIT_MIN_WIDTH_KM = 1.0
_SPLIT_CONNECTIVITY = 18
_SPLIT_STACKED_SEEDS_RADIUS = 3


def _cached(context, key, compute):
    """Compute once per timestep: ``context`` is fresh for every timestep."""
    cache = context.metadata.setdefault("_fields_cache", {})
    if key not in cache:
        cache[key] = compute()
    return cache[key]


def _vertical_indices(context, rule):
    cache = context.metadata.setdefault("_vertical_cache", {})
    key = (
        rule.get("vertical_dim", "level"),
        rule.get("min_alt_km"),
        rule.get("max_alt_km"),
    )
    if key not in cache:
        coords = np.asarray(
            context.dataset.coords[key[0]].values, 
            dtype=np.float32, 
        )
        cache[key] = z_indices_between(
            coords,
            -np.inf if key[1] is None else key[1] * 1000.0,
            np.inf if key[2] is None else key[2] * 1000.0,
        )
    return cache[key]


def _field(context, rule, name):
    ds = context.dataset
    if name not in ds:
        return None
    vertical_dim = rule.get("vertical_dim", "level")
    if vertical_dim not in ds[name].dims:
        return _cached(context, (name,), lambda: np.asarray(ds[name].data, dtype=float))

    clip = (rule.get("min_alt_km"), rule.get("max_alt_km"))
    return _cached(
        context,
        (name, vertical_dim, clip),
        lambda: np.asarray(ds[name].isel({vertical_dim: _vertical_indices(context, rule)}).data, dtype=np.float32),
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

def _slice_field(field, sl):
    if field.ndim == 3:
        return field[sl]
    if field.ndim == 2:
        return field[sl[1], sl[2]]
    return field

def _build_mask(context, rule, preloaded=None, sl=None):
    if preloaded is None:
        preloaded = _preload_fields(context, rule)
    if sl is not None:
        preloaded = {
            name: _slice_field(field, sl)
            for name, field in preloaded.items()
        }
    vertical_size = None
    for f in preloaded.values():
        if f.ndim == 3:
            vertical_size = f.shape[0]
            break
    if vertical_size is None:
        vertical_size = len(context.dataset.coords[rule.get("vertical_dim", "level")])

    mask = None
    for condition in rule["conditions"]:
        names = condition["variables"]
        names = [names] if isinstance(names, str) else names
        fields = [preloaded[name] for name in names if name in preloaded]
        if not fields:
            return None 

        fields_2d = [f for f in fields if f.ndim == 2]
        fields_3d = [f for f in fields if f.ndim == 3]

        if fields_3d:
            combined_3d = fields_3d[0]
            for f in fields_3d[1:]:
                combined_3d += f  
        else:
            combined_3d = None

        if fields_2d:
            for f_2d in fields_2d:
                if combined_3d is None:
                    combined_3d = np.zeros((vertical_size, *f_2d.shape), dtype=f_2d.dtype)
                combined_3d += f_2d

        if combined_3d is None:
            continue 

        op = condition.get("op", ">")
        if op == ">":
            passed = combined_3d > float(condition["value"])
        elif op == ">=":
            passed = combined_3d >= float(condition["value"])
        elif op == "<":
            passed = combined_3d < float(condition["value"])
        elif op == "<=":
            passed = combined_3d <= float(condition["value"])
        else:
            passed = combined_3d > float(condition["value"])  # Default

        if mask is None:
            mask = passed
        else: 
            mask &= passed

    return mask


def _restrict_to_parent(mask, parent_object):
    parent_idx = parent_object.get("indices_3d")
    if not parent_idx:
        return None
    footprint = np.zeros(mask.shape, dtype=bool)
    idx_array = np.array(parent_idx, dtype=np.int64)
    footprint[tuple(idx_array.T)] = True
    np.logical_and(mask, footprint, out=mask)
    return mask


def _label_groups(labels):
    flat = labels.ravel()
    nz = np.flatnonzero(flat)
    if nz.size == 0:
        return []
    order = nz[np.argsort(flat[nz], kind="stable")]
    sorted_labels = flat[order]
    splits = np.flatnonzero(np.diff(sorted_labels)) + 1
    coords = np.column_stack(np.unravel_index(order, labels.shape))
    return np.split(coords, splits)


def _bbox_mask(idx):
    """Boolean mask cropped to the bounding box of the coordinates, and the box origin."""
    origin = idx.min(axis=0)
    mask = np.zeros(idx.max(axis=0) - origin + 1, dtype=bool)
    mask[tuple((idx - origin).T)] = True
    return mask, origin


def _measure(idx, heights, dx_km):
    mask, origin = _bbox_mask(idx)
    layers = None if heights is None else heights[origin[0] : origin[0] + mask.shape[0]]
    return compute_component_metrics(mask, level_vals=layers, dx_km=dx_km)


def _is_large_enough(metrics, rule):
    return (
        metrics["voxels"] >= rule.get("min_voxels", 0)
        and metrics["area_px"] >= rule.get("min_area_px", 0)
        and metrics["width_px"] >= rule.get("min_width_px", 0)
        and metrics["area_km2"] >= rule.get("min_area_km2", 0.0)
    )


def _condition_threshold(rule, variable):
    """Threshold applied to ``variable`` in the mask conditions, 0.0 if not found."""
    for condition in rule["conditions"]:
        names = condition["variables"]
        names = [names] if isinstance(names, str) else names
        if variable in names:
            return float(condition["value"])
    return 0.0

def _describe(idx, metrics):
    return {
        "area_km2": metrics["area_km2"],
        "volume_km3": metrics["volume_km3"],
        "width_km": metrics["width_km"],
        "height_km": metrics["height_km"],
        "centroid_x": float(idx[:, 2].mean()),
        "centroid_y": float(idx[:, 1].mean()),
        "centroid_z": float(idx[:, 0].mean()),
        "indices_3d": np.asarray(idx, dtype=np.int64),
    }

def _preload_fields(context, rule):
    ds = context.dataset
    vertical_dim = rule.get("vertical_dim", "level")
    z_idx = _vertical_indices(context, rule)

    preloaded = {}
    for condition in rule["conditions"]:
        names = condition["variables"]
        names = [names] if isinstance(names, str) else names
        for name in names:
            if name not in preloaded and name in ds:
                if vertical_dim in ds[name].dims:
                    preloaded[name] = ds[name].isel({vertical_dim: z_idx}).values.astype(np.float32)
                else:
                    preloaded[name] = ds[name].values.astype(np.float32)
    return preloaded


def detect_objects(context, parent_object, level):
    rule = level.params
    field = _cached(context, ("fields", level.name), lambda: _preload_fields(context, rule))

    if parent_object is None:
        sl, origin, pidx = (slice(None),) * 3, np.zeros(3, dtype=np.int64), None
    else:
        pidx = np.asarray(parent_object["indices_3d"], dtype=np.int64)
        origin = pidx.min(axis=0)
        sl = tuple(slice(a, b) for a, b in zip(origin, pidx.max(axis=0) + 1))

    mask = _build_mask(context, rule, field, sl)
    
    if pidx is not None:
        fp = np.zeros(mask.shape, dtype=bool)
        fp[tuple((pidx - origin).T)] = True
        mask &= fp
    if not mask.any():
        return []
    
    labels, _ = connected_components_3d(mask, rule.get("connectivity") or 26)
    groups = [g + origin for g in _label_groups(labels)]

    dx_km = float(rule.get("dx_km", 1.0))
    heights = _layer_heights(context, rule)
    split_variable = rule.get("split")
    objects = []
    for group in groups:
        if len(group) < rule.get("min_voxels", 0):
            continue

        metrics = _measure(group, heights, dx_km)
        if not _is_large_enough(metrics, rule):
            continue

        if split_variable and field is not None:
            parts = _recursive_split(
                field[split_variable],
                rule,
                split_variable,
                group,
                dx_km,
            )
        else:
            parts = [group]

        for part in parts:
            if part is group:
                part_metrics = metrics
            else:
                part_metrics = _measure(part, heights, dx_km)
            if _is_large_enough(part_metrics, rule):
                objects.append(_describe(part, part_metrics))

    return objects

def _recursive_split_fast(field, rule, variable, idx, dx_km, reference, max_depth=1):
    if len(idx) == 0 or max_depth <= 0:
        return [idx]

    values = field[tuple(idx.T)]
    if values.size == 0:
        return [idx]

    threshold = reference + (float(values.max()) - reference) * _SPLIT_H_SLOPE
    threshold = min(max(threshold, _SPLIT_H_MIN), _SPLIT_H_MAX)
    origin = idx.min(axis=0)
    shape = idx.max(axis=0) - origin + 1
    local_field = np.zeros(shape, dtype=np.float32)
    local_field[tuple((idx - origin).T)] = values
    split_mask = local_field > threshold
    labels, _ = connected_components_3d(split_mask, _SPLIT_CONNECTIVITY)

    parts = []
    for label in np.unique(labels):
        if label == 0:
            continue
        local_coords = np.argwhere(labels == label)
        global_coords = local_coords + origin
        if len(global_coords) >= rule.get("min_voxels", 5):
            parts.append(global_coords)

    return parts if parts else [idx]

def _recursive_split(
    field, rule, variable, idx, dx_km, max_depth=3
):
    """Split one component by recursive H-maxima watershed on ``variable``."""
    reference = _condition_threshold(rule, variable)

    def compute_adaptive_h(values):
        h = _SPLIT_H_SLOPE * (float(values.max()) - reference)
        return float(min(max(h, _SPLIT_H_MIN), _SPLIT_H_MAX))

    def stop_condition(mask):
        return compute_component_metrics(mask, dx_km=dx_km)["width_median_km"] <= _SPLIT_MIN_WIDTH_KM

    subgroups = recursive_watershed_segmentation(
        idx,
        field[tuple(idx.T)],
        compute_adaptive_h=compute_adaptive_h,
        stop_condition=stop_condition,
        connectivity=_SPLIT_CONNECTIVITY,
        merge_stacked=True,
        stacked_seeds_xy_radius=_SPLIT_STACKED_SEEDS_RADIUS,
        max_depth=max_depth,
    )
    return [np.array(group["indices"], dtype=np.int64) for group in subgroups]