# API reference

Everything is importable from `src.tracking`.

## `TrackingLevel`

```python
TrackingLevel(name, detector, parent_name=None, params={}, tracker=None)
```

`detector(context, parent_object, level) -> list[dict]`. `level.detect(context, parent_object)` calls the detector and returns `[]` if it returns `None`. `tracker` is the level's tracker (optional): `level.track(time_index, time_value, timestamp_s, objects)` feeds it, `level.finalize_tracks()` returns its tracks (`{}` without a tracker).

## `DetectionContext`

Dataclass: `dataset`, `time_index`, `time_value` (str), `timestamp_s`, `metadata` (dict, including the caches).

## `HierarchicalTrackingEngine(levels)`

| Method | Description |
|---|---|
| `run_time_step(dataset, time_index, time_value=None, timestamp_s=None, metadata=None)` | processes one step; returns `{"time_index", "time", "levels"}` |
| `run(datasets, time_values=None, timestamps_s=None, metadata=None)` | processes a sequence; returns `engine.results_by_time` |
| `finalize_tracks()` | tracks per level: `{level: tracks}` |
| `run_and_track(...)` | `run` + `finalize_tracks`; returns `{"results_by_time", "tracks_by_level"}` |

Raises `ValueError`: duplicate level name, unknown parent, parent declared after its child.

## `detect_objects(context, parent_object, level)`

Generic detector, see [detection.md](detection.md). Raises `KeyError` if the variable of a `split` is missing from the Dataset and `ValueError` for an unsupported operator.

## `AdvectedIoUTracker`

`update(time_index, time_value, timestamp_s, objects)` and `finalize()`. See [tracking.md](tracking.md).

## `src.tracking.functions`

| Function | Description |
|---|---|
| `z_indices_between(level_values, zmin_m, zmax_m)` | indices of the levels within an interval |
| `connected_components_2d(mask, connectivity=8)` / `_3d(mask, connectivity=18)` | labeling |
| `compute_component_metrics(mask3d, level_vals=None, dx_km=1.0)` | voxels, area, width, height, volume, median width |
| `iou_from_sets(a, b)` | IoU between two arrays of unique keys |
| `pack_indices(idx, grid_size=10000)` | encodes `(n, 2|3)` into `int64` keys |
| `advect_indices(idx, dy_pixels, dx_pixels)` | shifts the last two columns |
| `merge_stacked_seeds(markers, field, xy_radius=3)` | merges vertically stacked seeds |
| `recursive_watershed_segmentation(indices, values, compute_adaptive_h, stop_condition, connectivity, merge_stacked, stacked_seeds_xy_radius, ...)` | recursive H-maxima watershed |

## `src.tracking.legacy_adapter`

`LegacyLevelSpec`, `create_legacy_levels`, `extract_time_values_and_timestamps`, `to_legacy_results_by_time`, `run_legacy_pipeline`: run the new engine with the old detection / tracking functions (non-regression).
