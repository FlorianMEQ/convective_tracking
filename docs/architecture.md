# Architecture

## Overview

```
 Datasets (one per timestep)
          │
          ▼
 HierarchicalTrackingEngine ── ordered list of ──► TrackingLevel
   run / run_time_step                                name, parent_name, params (rule), tracker
          │                                           detector = detect_objects
          │  DetectionContext (dataset, time_index,        │
          │  time_value, timestamp_s, metadata)            ▼
          │                                         detect_objects(context, parent, level)
          ▼                                                │
 results_by_time  ◄────────────── objects (dict) ◄─────────┘
          │
          ▼
 AdvectedIoUTracker (one per level, level.tracker) ──► tracks {track_id: {history, events, ...}}
```

Principle: **the library code knows nothing about physics**. The convective meaning lives in the rules ([examples/convective_rules.py](../examples/convective_rules.py)).

## Modules of `src/tracking`

| Module | Content |
|---|---|
| [level.py](../src/tracking/level.py) | `DetectionContext` (data of one timestep), `TrackingLevel` (name, parent, detector, `params` rule, tracker) |
| [engine.py](../src/tracking/engine.py) | `HierarchicalTrackingEngine`: level validation, time loop, parent → child propagation |
| [detection.py](../src/tracking/detection.py) | `detect_objects`: generic rule-driven detector (see [detection.md](detection.md)) |
| [trackers.py](../src/tracking/trackers.py) | `AdvectedIoUTracker` (see [tracking.md](tracking.md)) |
| [functions.py](../src/tracking/functions.py) | Numerical building blocks: connected components, metrics, IoU, advection, recursive watershed |
| [legacy_adapter.py](../src/tracking/legacy_adapter.py) | Bridge to the old code, used for non-regression |

## Engine

For each timestep, `run_time_step`:

1. creates a fresh `DetectionContext` (copy of `metadata`, so per-timestep caches do not leak);
2. walks the levels in order; a root level is called once (`parent=None`), a child level once **per parent object**;
3. completes each object: `level`, `object_id` (`t{time}_{level}_{NNNN}`) and, for a child, `parent_object_id`;
4. feeds the level's tracker (`level.track`) with all the detected objects;
5. appends the step to `engine.results_by_time`.

`TrackingLevel.detector` is any callable `(context, parent_object, level) -> list[dict]`: `detect_objects` can therefore be replaced by a custom detector.

## Field cache

`detect_objects` caches in `context.metadata["_fields_cache"]` the arrays extracted from the Dataset (with vertical clipping), so they are read only once per timestep, even with many parent objects.

## Old code and migration

[src/old/fct_core_tracking.py](../src/old/fct_core_tracking.py) contains the initial implementation (`find_convective_cores_3d`, `track_mcs`, `track_conv_objects`, `track_convective_cores`). It is kept only as a non-regression reference. [legacy_adapter.py](../src/tracking/legacy_adapter.py) allows running the new detector with the old tracking functions (`run_legacy_pipeline`, `create_legacy_levels`, `to_legacy_results_by_time`).

The class diagram is in [Diagramme de classe.drawio.svg](<../Diagramme de classe.drawio.svg>).
