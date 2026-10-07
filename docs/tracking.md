# Temporal tracking: `AdvectedIoUTracker`

Links the objects of one level from one timestep to the next.

## Algorithm

At each `update(time_index, time_value, timestamp_s, objects)`:

1. **Items**: for each object carrying `index_key`, compute centroid, voxel keys, steering wind (`u_steering`, `v_steering`) and, if `feature_fn` is set, the feature vector.
2. **Advection**: each active track is shifted by its wind over `dt` (seconds → pixels through `dx_meters`).
3. **Spatial pre-filter**: a candidate is considered only if its centroid is within `radius` of the predicted centroid:
   - with `max_speed_ms`: `max_speed_ms · dt / dx_meters`;
   - otherwise: `min(pre_filter_pixels, advection_error_ms · dt / dx_meters + shape_buffer_pixels)`.
4. **Score**:
   - without `feature_fn`: IoU between the advected footprint and the object;
   - with `feature_fn`: `w_overlap · IoU + w_morph · morphological_similarity` if IoU > 0; otherwise, if `dt > gap_bridge_seconds` and distance ≤ `gap_bridge_max_dist_pixels`, `0.8 · w_morph · similarity` (bridging a temporal gap).
5. **Assignment**: Hungarian algorithm on `1 − score`; pairs with score ≥ `min_score` are kept.
6. **Merges (`merge`)**: an unassigned track whose score with an already taken object is ≥ `min_score` is ended, with the event `{"type": "merge", "time", "into": track_id}`.
7. **Splits (`split`)**: an unassigned object overlapping (score ≥ `min_score`) an assigned track starts a new track with `parent_id`; a `split` event is added to the parent track. The child inherits the parent's wind.
8. **New tracks**: every unassigned object starts a track `"{track_prefix}_{NNNN}"`.
9. **Patience**: a track not seen for more than `patience` steps is dropped.

Morphological similarity is the weighted mean (normalized `feature_weights`) of the per-feature `min/max` ratios; the features of a track are smoothed by an exponential moving average (`feat_ema_alpha`).

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `track_prefix` | required | prefix of the track identifiers |
| `index_key` | `"indices_3d"` | `"indices_2d"` for MCS |
| `min_score` | 0.1 | minimum association score |
| `patience` | 0 | timesteps without detection tolerated |
| `dx_meters` | 1000 | grid spacing |
| `grid_size` | 10000 | index encoding factor (maximum grid size) |
| `pre_filter_pixels` | 30 | maximum search radius |
| `advection_error_ms` | 5 | assumed wind error (m/s) |
| `shape_buffer_pixels` | 3 | margin related to the object size |
| `max_speed_ms` | `None` | if set, replaces the radius computation above |
| `gap_bridge_seconds`, `gap_bridge_max_dist_pixels` | 120, 10 | temporal gap bridging |
| `feature_fn`, `feature_weights` | `None` | morphological similarity (`feature_weights` is required with `feature_fn`) |
| `feat_ema_alpha` | 0.7 | feature smoothing |
| `w_overlap`, `w_morph` | 0.6, 0.4 | IoU / morphology weighting |
| `steering_fields` | `("u_steering", "v_steering")` | wind fields in the objects |
| `detect_splits_merges` | `True` | record split/merge events and link split children to parent tracks; disable to match trackers that only continue or start tracks |

Without a timestamp (`timestamp_s=None`), the step is 30 s (`DEFAULT_STEP_SECONDS`).

## Example with morphological features

```python
import numpy as np
from src.tracking import AdvectedIoUTracker

tracker = AdvectedIoUTracker(
    "core", index_key="indices_3d", dx_meters=1000.0, min_score=0.1,
    feature_fn=lambda o: np.array([o["volume_km3"], o["height_km"], o["max_speed"]]),
    feature_weights=np.array([1.0, 1.0, 1.0]),
)
```

## Output: `tracker.finalize()`

```python
{
  "core_0001": {
    "track_id": "core_0001",
    "start_time": "...", "end_time": "...",
    "history": [obj_t0, obj_t1, ...],      # objects from the detector
    "parent_id": "core_0007",              # if issued from a split
    "events": [{"type": "split", "time": "...", "child_centroid": (y, x)},
               {"type": "merge", "time": "...", "into": "core_0003"}],
  },
  ...
}
```

## Settings used in the notebook

| Level | `index_key` | `min_score` | `patience` | Others |
|---|---|---|---|---|
| `mcs` | `indices_2d` | 0.01 | 4 | `max_speed_ms=30` |
| `convective_object` | `indices_3d` | 0.1 | 2 | `pre_filter_pixels=60` |
| `core` | `indices_3d` | 0.1 | 0 | `pre_filter_pixels=30` |

## Link with the levels

A `TrackingLevel` carries a tracker (`TrackingLevel(..., tracker=AdvectedIoUTracker(...))`). At each timestep, after the level detection, the engine calls `level.track(...)`, which runs `tracker.update(...)` with all the objects of the level. `engine.finalize_tracks()` (and `run_and_track`) returns `{level: tracker.finalize()}`.

The legacy convective-object tracker does not infer split or merge relationships: it only continues matched objects and starts new tracks for unmatched objects. Configure `detect_splits_merges=False` when comparing that legacy behavior. The MCS and core trackers do record splits and merges, so their setting remains enabled.

The tracker can also be used on its own:

```python
for step in engine.results_by_time:
    tracker.update(step["time_index"], step["time"], None, step["levels"]["core"])
tracks = tracker.finalize()
```
