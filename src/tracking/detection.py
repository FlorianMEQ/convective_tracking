from __future__ import annotations

import numpy as np

from .functions import (
    compute_component_metrics,
    connected_components_2d,
    connected_components_3d,
    recursive_watershed_segmentation,
    z_indices_between,
)

_COMPARATORS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}

_SPLIT_DEFAULTS = {
    "h_slope": 0.3,
    "h_min": 1.0,
    "h_max": 10.0,
    "min_width_km": 1.0,
    "connectivity": 18,
    "stacked_seeds_xy_radius": 3,
    "max_depth": 3,
}


def _names(condition):
    names = condition["variables"]
    return [names] if isinstance(names, str) else list(names)


def _split_spec(rule):
    split = rule.get("split")
    if not split:
        return None
    spec = {"variable": split} if isinstance(split, str) else dict(split)
    return {**_SPLIT_DEFAULTS, **spec}


def _cache(context, name):
    return context.metadata.setdefault(name, {})


def _vertical_indices(context, rule):
    key = (rule.get("vertical_dim", "level"), rule.get("min_alt_km"), rule.get("max_alt_km"))
    cache = _cache(context, "_vertical_cache")
    if key not in cache:
        coords = np.asarray(context.dataset.coords[key[0]].values, dtype=float)
        cache[key] = z_indices_between(
            coords,
            -np.inf if key[1] is None else key[1] * 1000.0,
            np.inf if key[2] is None else key[2] * 1000.0,
        )
    return cache[key]


def _load(context, rule, name):
    ds = context.dataset
    if name not in ds:
        return None
    vertical_dim = rule.get("vertical_dim", "level")
    is_3d = vertical_dim in ds[name].dims
    key = (name, vertical_dim, rule.get("min_alt_km"), rule.get("max_alt_km")) if is_3d else (name,)
    cache = _cache(context, "_fields_cache")
    if key not in cache:
        da = ds[name].isel({vertical_dim: _vertical_indices(context, rule)}) if is_3d else ds[name]
        arr = np.asarray(da.values, dtype=np.float32)
        arr.setflags(write=False)
        cache[key] = arr
    return cache[key]


def _rule_fields(context, rule):
    names = {n for c in rule["conditions"] for n in _names(c)}
    split = _split_spec(rule)
    if split:
        names.add(split["variable"])
    names.update(s["variable"] for s in rule.get("stats", ()))
    fields = {}
    for name in names:
        arr = _load(context, rule, name)
        if arr is not None:
            fields[name] = arr
    return fields


def _layer_heights(context, rule):
    dim_w = rule.get("vertical_dim_w", "level_w")
    if dim_w not in context.dataset.coords:
        return None
    key = ("_heights", dim_w, rule.get("min_alt_km"), rule.get("max_alt_km"))
    cache = _cache(context, "_fields_cache")
    if key not in cache:
        cache[key] = np.asarray(
            context.dataset[dim_w].isel({dim_w: _vertical_indices(context, rule)}).values, dtype=float
        )
    return cache[key]



def _slice(field, sl):
    return field[sl] if field.ndim == 3 else field[sl[1:]]


def _build_mask(rule, fields, sl):
    """2D mask if every condition uses 2D variables, 3D mask otherwise; None if a condition has no variable."""
    mask = None
    for condition in rule["conditions"]:
        present = [_slice(fields[n], sl) for n in _names(condition) if n in fields]
        if not present:
            return None
        total = present[0]
        for f in present[1:]:
            total = total + f
        op = condition.get("op", ">")
        if op not in _COMPARATORS:
            raise ValueError(f"Unsupported operator '{op}' (use one of {sorted(_COMPARATORS)})")
        passed = _COMPARATORS[op](total, float(condition["value"]))
        mask = passed if mask is None else mask & passed
    return None if mask is None else np.array(mask, dtype=bool)


def _label_groups(labels):
    """Coordinates of each labeled component, in label order."""
    flat = labels.ravel()
    nz = np.flatnonzero(flat)
    if nz.size == 0:
        return []
    order = nz[np.argsort(flat[nz], kind="stable")]
    splits = np.flatnonzero(np.diff(flat[order])) + 1
    coords = np.column_stack(np.unravel_index(order, labels.shape))
    return np.split(coords, splits)


def _bbox_mask(idx):
    origin = idx.min(axis=0)
    mask = np.zeros(idx.max(axis=0) - origin + 1, dtype=bool)
    mask[tuple((idx - origin).T)] = True
    return mask, origin


def _measure(idx, heights, dx_km):
    if idx.shape[1] == 2:
        width_px = int(max(np.ptp(idx[:, 0]), np.ptp(idx[:, 1])) + 1)
        return {"voxels": len(idx), "area_px": len(idx), "area_km2": len(idx) * dx_km**2,
                "width_px": width_px, "width_km": width_px * dx_km, "height_km": 0.0, "volume_km3": 0.0}
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


def _split_orientation(rule, variable):
    """(sign, reference): sign = -1 when the variable is thresholded from above (``<``, ``<=``)."""
    for condition in rule["conditions"]:
        if variable in _names(condition):
            sign = -1.0 if condition.get("op", ">") in ("<", "<=") else 1.0
            return sign, sign * float(condition["value"])
    return 1.0, 0.0


def _split(field, rule, spec, idx, dx_km):
    sign, reference = _split_orientation(rule, spec["variable"])
    flat = idx.shape[1] == 2
    if flat and field.ndim == 3:
        raise ValueError(f"A 2D object cannot be split on the 3D variable '{spec['variable']}'")
    points = np.column_stack([np.zeros(len(idx), dtype=idx.dtype), idx]) if flat else idx
    values = field[idx[:, 0], idx[:, 1]] if flat else field[tuple(idx.T)]

    def compute_adaptive_h(v):
        h = spec["h_slope"] * (float(v.max()) - reference)
        return float(min(max(h, spec["h_min"]), spec["h_max"]))

    def stop_condition(mask):
        return compute_component_metrics(mask, dx_km=dx_km)["width_median_km"] <= spec["min_width_km"]

    groups = recursive_watershed_segmentation(
        points,
        sign * values,
        compute_adaptive_h=compute_adaptive_h,
        stop_condition=stop_condition,
        connectivity=spec["connectivity"],
        merge_stacked=True,
        stacked_seeds_xy_radius=spec["stacked_seeds_xy_radius"],
        max_depth=spec["max_depth"],
    )
    return [g["indices"][:, 1:] if flat else g["indices"] for g in groups]


def _stat_values(field, idx, kind, half_depth):
    if idx.shape[1] == 2:
        return field[:, idx[:, 0], idx[:, 1]] if field.ndim == 3 else field[idx[:, 0], idx[:, 1]]
    if field.ndim == 2:
        return field[idx[:, 1], idx[:, 2]]
    if kind == "layer_mean":
        zc = int(round(float(idx[:, 0].mean())))
        near = np.abs(idx[:, 0] - zc) <= half_depth
        idx = idx[near] if near.any() else idx
    return field[tuple(idx.T)]


def _compute_stats(rule, fields, idx):
    out = {}
    for stat in rule.get("stats", ()):
        field = fields.get(stat["variable"])
        if field is None:
            continue
        kind = stat.get("stat", "mean")
        values = _stat_values(field, idx, kind, stat.get("half_depth", 2))
        if kind in ("mean", "layer_mean"):
            out[stat["name"]] = float(np.nanmean(values))
        elif kind in ("min", "max", "sum"):
            out[stat["name"]] = float(getattr(np, f"nan{kind}")(values))
        elif kind == "percentile":
            out[stat["name"]] = float(np.nanpercentile(values, stat["q"]))
        else:
            raise ValueError(f"Unsupported stat '{kind}'")
    return out


def _footprint_2d(idx):
    width = int(idx[:, 2].max()) + 1
    keys = np.unique(idx[:, 1].astype(np.int64) * width + idx[:, 2])
    return np.column_stack(np.divmod(keys, width))


def _describe(idx, metrics, stats):
    obj = {
        "area_km2": metrics["area_km2"],
        "volume_km3": metrics["volume_km3"],
        "width_km": metrics["width_km"],
        "height_km": metrics["height_km"],
        "centroid_x": float(idx[:, -1].mean()),
        "centroid_y": float(idx[:, -2].mean()),
    }
    if idx.shape[1] == 3:
        obj["centroid_z"] = float(idx[:, 0].mean())
    obj.update(stats)
    if idx.shape[1] == 3:
        obj["indices_3d"] = idx
        obj["indices_2d"] = _footprint_2d(idx)
    else:
        obj["indices_2d"] = idx
    return obj


def _parent_region(parent_object):
    """(parent indices, origin (z, y, x), slices) of the parent's bounding box."""
    idx = parent_object.get("indices_3d")
    if idx is not None and len(idx):
        idx = np.asarray(idx, dtype=np.int64)
        origin = idx.min(axis=0)
        return idx, origin, tuple(slice(a, b) for a, b in zip(origin, idx.max(axis=0) + 1))
    idx = parent_object.get("indices_2d")
    if idx is None or not len(idx):
        return None, None, None
    idx = np.asarray(idx, dtype=np.int64)
    lo, hi = idx.min(axis=0), idx.max(axis=0) + 1
    return idx, np.array([0, lo[0], lo[1]]), (slice(None), slice(lo[0], hi[0]), slice(lo[1], hi[1]))


def _restrict(mask, pidx, origin):
    if pidx.shape[1] == 3 and mask.ndim == 3:
        footprint = np.zeros(mask.shape, dtype=bool)
        footprint[tuple((pidx - origin).T)] = True
    else:
        yx = pidx[:, -2:] - origin[1:]
        footprint = np.zeros(mask.shape[-2:], dtype=bool)
        footprint[yx[:, 0], yx[:, 1]] = True
    mask &= footprint
    return mask


def detect_objects(context, parent_object, level):
    rule = level.params
    fields = _rule_fields(context, rule)

    if parent_object is None:
        origin, sl, pidx = np.zeros(3, dtype=np.int64), (slice(None),) * 3, None
    else:
        pidx, origin, sl = _parent_region(parent_object)
        if pidx is None:
            return []

    mask = _build_mask(rule, fields, sl)
    if mask is None:
        return []
    if pidx is not None:
        mask = _restrict(mask, pidx, origin)
    if not mask.any():
        return []

    connectivity = rule.get("connectivity")
    if mask.ndim == 3:
        labels, _ = connected_components_3d(mask, connectivity or 26)
        offset = origin
    else:
        labels, _ = connected_components_2d(mask, 4 if connectivity in (4, 6) else 8)
        offset = origin[1:]

    dx_km = float(rule.get("dx_km", 1.0))
    heights = _layer_heights(context, rule) if mask.ndim == 3 else None
    split = _split_spec(rule)
    split_field = fields.get(split["variable"]) if split else None
    if split and split_field is None:
        raise KeyError(f"Split variable '{split['variable']}' is not in the dataset")
    if split_field is not None and mask.ndim == 3 and split_field.ndim == 2:
        split_field = np.broadcast_to(split_field, (len(_vertical_indices(context, rule)), *split_field.shape))

    objects = []
    for group in _label_groups(labels):
        group += offset
        if len(group) < rule.get("min_voxels", 0):
            continue
        metrics = _measure(group, heights, dx_km)
        if not _is_large_enough(metrics, rule):
            continue

        parts = _split(split_field, rule, split, group, dx_km) if split_field is not None else [group]
        for part in parts:
            part_metrics = metrics if part is group else _measure(part, heights, dx_km)
            if _is_large_enough(part_metrics, rule):
                objects.append(_describe(part, part_metrics, _compute_stats(rule, fields, part)))
    return objects
