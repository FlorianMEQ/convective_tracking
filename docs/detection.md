# Detection: rules

`detect_objects(context, parent_object, level)` reads the rule from `level.params`. Steps:

1. **Mask**: conjunction (AND) of the `conditions`. With a parent, the mask is restricted to the parent footprint (`indices_3d` or `indices_2d`).
2. **Labeling**: connected components, or splitting (`split`).
3. **Size filters**: `min_voxels`, `min_area_px`, `min_width_px`, `min_area_km2`.
4. **Description**: geometric metrics, statistics, indices.

## Rule keys

### Grid

| Key | Default | Description |
|---|---|---|
| `dx_km` | 1.0 | horizontal grid spacing |
| `vertical_dim` | `"level"` | vertical dimension of 3D variables |
| `vertical_dim_w` | `"level_w"` | coordinate of the layer heights (m) |
| `min_alt_km`, `max_alt_km` | none | vertical clipping |

### Mask (required)

```python
"conditions": [
    {"variables": "WT", "op": ">", "value": 1.0},
    {"variables": ["MRI", "MRC", "MRS"], "value": 0.01},   # summed before the comparison
]
```

Operators: `>` (default), `>=`, `<`, `<=`. The mask is 2D or 3D depending on the variables. If none of the variables of a condition exists, the level detects nothing.

### Objects

| Key | Default | Description |
|---|---|---|
| `connectivity` | 2D: 8 (or 4); 3D: 26 (6, 18, 26) | connectivity |
| `split` | none | splitting, see below |
| `min_voxels`, `min_area_px`, `min_width_px`, `min_area_km2` | 0 | filters |
| `stats` | `[]` | statistics |

### `split`: `seeded_watershed`

Seeds are the extrema of `variable`; objects are the watershed basins within the mask (a component without a seed is dropped).

```python
{"method": "seeded_watershed", "variable": "MSG2_IR108", "mode": "min",
 "seed_h": 5.0, "outside_value": 231.0}
```

- `mode`: `"min"` (default) or `"max"`.
- One seed method: `seed_h` (h-minima: minimum depth), `seed_percentile` (percentile within the mask) or `seed_threshold` (fixed threshold). Priority in this order.
- `outside_value`: value outside the mask for `seed_h` (default: extremum within the mask + 1).

### `split`: `recursive_h_maxima`

Each component is recursively split (H-maxima seeded watershed) while its median width exceeds `min_width_km`. `variable` must be positive within the mask.

| Key | Default | Description |
|---|---|---|
| `h_slope` | 0.3 | `h = h_slope · (max − h_reference)`, bounded |
| `h_min`, `h_max` | 1.0, 10.0 | bounds of `h` |
| `h_reference` | value of the 1st condition | reference threshold |
| `min_width_km` | 1.0 | median width at which splitting stops |
| `connectivity` | 18 | watershed connectivity |
| `stacked_seeds_xy_radius` | 3 | merging of vertically stacked seeds (px) |

### `stats`

```python
{"name": "tb_min_k", "variable": "MSG2_IR108", "stat": "min"}
{"name": "speed_99pct", "variable": "WT", "stat": "percentile", "q": 99}
{"name": "u_steering", "variable": "UT", "stat": "layer_mean"}
```

`stat` ∈ `mean`, `min`, `max`, `percentile` (with `q`), `layer_mean` (mean within ±`half_depth` (2) levels around the vertical centroid of a 3D object, over the whole column for a 2D object).

## Produced object

| Field | Description |
|---|---|
| `object_id`, `level`, `parent_object_id` | added by the engine |
| `area_km2`, `width_km` | horizontal extent |
| `centroid_x`, `centroid_y` | centroid in pixels |
| `volume_km3`, `height_km`, `centroid_z` | 3D objects only |
| *stats* | one key per entry of `stats` |
| `indices_2d` or `indices_3d` | list of `(y, x)` or `(z, y, x)` tuples |

## Provided convective rules

`convective_rules(params)` returns:

| Level | Mask | Splitting | Filters | Statistics |
|---|---|---|---|---|
| `mcs` | `Tb ≤ tb_env_k` | `seeded_watershed` on Tb (min) | `min_area_km2 = mcs_min_area_km2` (100) | `tb_min_k`, `tb_mean_k`, `u_steering`, `v_steering` (means) |
| `convective_object` | `WT > w_thresh` AND Σ hydro `> hydro_min_gkg` | none | `min_voxels` 5, `min_width_px` 5, `min_area_px` 15 | `max_speed`, `speed_99pct`, winds (`layer_mean`) |
| `core` | `WT > core_w_thresh` | `recursive_h_maxima` on `WT` | `min_voxels` 5, `min_width_px` 3, `min_area_px` 5 | speed, `mri_mean`, `mrg_mean`, winds |

A statistic whose variable is not defined (e.g. no `ut_var`) is simply omitted.
