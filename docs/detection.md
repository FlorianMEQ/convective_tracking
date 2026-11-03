# Detection: writing a rule

`detect_objects(context, parent_object, level)` is the single detector of the library. It reads the rule in `level.params` and returns a list of objects. It is called once per timestep for a root level, and once per parent object for a child level.

## Steps

1. **Mask.** Each condition compares a variable (or the sum of several) to a threshold; conditions are combined with AND. With a parent, the mask is restricted to the parent: its voxels for a 3D parent, the column below its footprint for a 2D parent.
2. **Components.** Connected components of the mask.
3. **Size filters.** Components that are too small are dropped.
4. **Split** (optional). Each remaining component is cut along the maxima of a variable; size filters are applied again to each piece.
5. **Description.** Geometry and requested statistics of each object.

## 2D and 3D objects

The dimension of an object follows from its conditions:

- if **every** condition uses 2D variables, the mask and the objects are **2D**: footprints, with `indices_2d` only;
- as soon as **one** condition uses a 3D variable, the mask is **3D**; 2D variables in other conditions are broadcast over the column.

## Rule keys

### Grid

| Key | Default | Description |
|---|---|---|
| `dx_km` | 1.0 | horizontal grid spacing (km) |
| `vertical_dim` | `"level"` | vertical dimension of 3D variables; its coordinate holds altitudes (m) |
| `min_alt_km`, `max_alt_km` | none | keep only the levels between these altitudes |

### Mask (required)

```python
"conditions": [
    {"variables": "W", "op": ">", "value": 1.0},
    {"variables": ["MRI", "MRC", "MRS"], "op": ">", "value": 0.01},   # summed before the comparison
]
```

| Field | Description |
|---|---|
| `variables` | one name or a list of names; several variables are summed |
| `op` | `>` (default), `>=`, `<`, `<=`; any other value raises `ValueError` |
| `value` | threshold |

A variable missing from the dataset is ignored within its condition. If **no** variable of a condition exists, the level detects nothing at that timestep (no error).

### Components and size filters

| Key | Default | Description |
|---|---|---|
| `connectivity` | 3D: 26; 2D: 8 | 3D: 6, 18 or 26. 2D: 4 or 8 (a 3D value is accepted: 6 means 4, 18 and 26 mean 8) |
| `min_voxels` | 0 | minimum number of voxels (pixels for a 2D object) |
| `min_area_px` | 0 | minimum footprint area, in pixels |
| `min_area_km2` | 0 | minimum footprint area, in km² |
| `min_width_px` | 0 | minimum largest horizontal extent, in pixels |

### Splitting: `split`

Without `split`, one connected component is one object. With `split`, each component is cut by a recursive H-maxima watershed: the seeds are the maxima of the variable that stand out by more than `h`; each piece is split again while its median width exceeds `min_width_km`.

```python
"split": "W"                                                   # default settings
"split": {"variable": "W", "h_slope": 0.3, "min_width_km": 2.0}  # override some of them
```

| Key | Default | Description |
|---|---|---|
| `variable` | required | variable to split on; it does not need to appear in `conditions` |
| `h_slope`, `h_min`, `h_max` | 0.3, 1.0, 10.0 | `h = h_slope · (max − reference)`, bounded to `[h_min, h_max]`; the reference is the threshold applied to `variable` in `conditions` (0 if it has none) |
| `min_width_km` | 1.0 | stop splitting a piece whose median width (over its levels) is at most this |
| `connectivity` | 18 | 3D connectivity of the watershed |
| `stacked_seeds_xy_radius` | 3 | seeds whose horizontal projections are closer than this (px) are merged: vertically stacked maxima give one object |
| `max_depth` | 3 | maximum number of recursive splits |

When `variable` is thresholded from above (`op` is `<` or `<=`, e.g. a downdraft `W < -2`), the split follows its **minima**. A missing `variable` raises `KeyError`. A 2D object can only be split on a 2D variable.

## Produced object

| Key | 3D object | 2D object |
|---|---|---|
| `indices_3d` | `(n, 3)` int array of `(z, y, x)` | absent |
| `indices_2d` | `(m, 2)` footprint `(y, x)` | `(m, 2)` pixels |
| `area_km2`, `width_km` | footprint area, largest horizontal extent | same |
| `height_km`, `volume_km3` | from `level_w` (0 without it) | 0 |
| `centroid_x`, `centroid_y` | mean grid index | mean grid index |
| `centroid_z` | mean level index | absent |
| one key per `stats` entry | yes | yes |

The engine then adds `level`, `object_id`, `time_value` and, for a child, `parent_object_id`.

Indices are global grid indices (not relative to the parent). The `z` index refers to the vertically clipped grid when `min_alt_km` / `max_alt_km` are set.

## Writing your own detector

`TrackingLevel.detector` can be any callable `(context, parent_object, level) -> list[dict]`. To be tracked, each object needs the index key read by its tracker (`indices_3d` or `indices_2d`); everything else is free. `context.dataset` is the timestep's dataset, `level.params` the rule.

```python
def largest_only(context, parent_object, level):
    objects = detect_objects(context, parent_object, level)
    return sorted(objects, key=lambda o: o["area_km2"])[-1:]
```
