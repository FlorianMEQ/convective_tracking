# HODT — hierarchical object detection and tracking

HODT detects nested objects in gridded atmospheric model output and tracks each kind of object through time.

What an object *is* (a cloud, a downdraft, a convective core etc ...) is set by a **rule**: a small dictionary of thresholds and size filters. Where it is *searched* is set by the **hierarchy**: a child level is only searched inside the objects of its parent level.

Example :

```
mcs                       2D   Tb ≤ 240 K                     (root, no parent level)
└──convective_object      3D   W > 1 m/s
   └── core               3D   W > 4 m/s, split along the maxima of W
```

## Installation

```bash
pip install numpy scipy scikit-image xarray pandas netcdf4
pip install matplotlib jupyter     # notebooks
```

## Quick start

From the example in `examples/` :

```python
from examples.synthetic_data import make_frames
from src.tracking import AdvectedIoUTracker, HierarchicalTrackingEngine, TrackingLevel, detect_objects

frames, times, seconds = make_frames(8)

rules = {
    "cloud": {
      "conditions": [
            {
               "variables": ["MRC", "MRI"], "op": ">", "value": 0.05
            }
         ], "min_voxels": 50
      },
    "core": {
      "conditions": [
            {
               "variables": "W", "op": ">", "value": 3.0
            }
         ], 
      "split": "W", 
      "min_voxels": 10
      },
}

levels = [
    TrackingLevel("cloud", detect_objects, parent_name=None, params=rules["cloud"], tracker=AdvectedIoUTracker("cloud")),
    TrackingLevel("core", detect_objects, parent_name="cloud", params=rules["core"], tracker=AdvectedIoUTracker("core")),
]

out = HierarchicalTrackingEngine(levels).run_and_track(frames, time_values=times, timestamps_s=seconds)

for tid, track in out["tracks_by_level"]["core"].items():
    print(tid, track["start_time"], "->", track["end_time"], track.get("events", []))
```

```
core_0001 12:00 -> 12:10 [{'type': 'merge', 'time': '12:15', 'into': 'core_0002'}]
core_0002 12:00 -> 12:35 [{'type': 'absorbed', 'time': '12:15', 'from': 'core_0001'}]
core_0003 12:00 -> 12:35 []
```

## Documentation

| Page | Read it to |
|---|---|
| [docs/user-guide.md](docs/user-guide.md) | User guide |
| [examples/synthetic_hierarchy_example.ipynd](examples/synthetic_hierarchy_example.ipynd) | Jupyter Notebook : First step with HOTD |