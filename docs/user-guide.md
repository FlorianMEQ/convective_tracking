# User guide

This guide goes from a simulation file to tracks. The two notebooks follow the same steps: [synthetic_hierarchy_example.ipynb](../examples/synthetic_hierarchy_example.ipynb) on a synthetic scene, [convective_cores_example.ipynb](../examples/convective_cores_example.ipynb) on a MesoNH file.

## 1. Input data

The engine takes a **list of `xarray.Dataset`, one per timestep** (for instance `ds.isel(time=i)` for each `i`).

| Element | Requirement |
|---|---|
| 3D variables | dimensions `(level, y, x)` in this order; any names for `y` and `x` |
| 2D variables | dimensions `(y, x)` (e.g. a brightness temperature) |
| `level` coordinate | altitude (m) of each level; required, used for `min_alt_km` / `max_alt_km` |
| `level_w` coordinate | altitude (m) of the layer interfaces; optional, needed for `height_km` and `volume_km3` (both are 0 without it) |

```python
import pandas as pd
import xarray as xr

with xr.open_dataset("simulation.nc") as ds:
    frames = [ds.isel(time=i).load() for i in range(ds.sizes["time"])]
times = [str(f["time"].values).split("T")[-1].split(".")[0] for f in frames]     # "12:05:00"
seconds = [float(pd.Timestamp(f["time"].values).timestamp()) for f in frames]
```

## 2. One rule per level

A rule is a `dict`. The minimum is a list of `conditions`, combined with AND:

```python
updraft = {
    "conditions": [
        {"variables": "WT", "op": ">", "value": 1.0},
        {"variables": ["MRI", "MRC"], "op": ">", "value": 0.01},   # summed, then compared
    ],
    "min_voxels": 5,
}
```

Objects are the connected components of the mask. Optional keys cut them along the maxima of a variable (`split`), filter them by size (`min_voxels`, `min_area_km2`, …) and attach statistics to them (`stats`). If every condition uses 2D variables, objects are 2D (a footprint); otherwise they are 3D. All the keys are in [detection.md](detection.md).

## 3. Declaring the hierarchy

A level joins a name, a detector, a rule, a parent and, optionally, a tracker:

```python
from src.tracking import AdvectedIoUTracker, HierarchicalTrackingEngine, TrackingLevel, detect_objects

levels = [
    TrackingLevel("mcs", detect_objects, parent_name=None, params=rules["mcs"],
                  tracker=AdvectedIoUTracker("mcs", index_key="indices_2d", patience=4, max_speed_ms=30.0)),
    TrackingLevel("convective_object", detect_objects, parent_name="mcs", params=rules["convective_object"],
                  tracker=AdvectedIoUTracker("co", patience=2)),
    TrackingLevel("core", detect_objects, parent_name="convective_object", params=rules["core"],
                  tracker=AdvectedIoUTracker("core", patience=1)),
]
engine = HierarchicalTrackingEngine(levels)
```

- A **root** level (`parent_name=None`) is searched once, over the whole domain.
- A **child** level is searched once per parent object, inside that parent only: inside its voxels for a 3D parent, in the column below its footprint for a 2D parent.
- Siblings (two levels with the same parent) are searched independently of each other.
- Levels are listed parents first. The engine raises `ValueError` for a duplicate name, an unknown parent, or a parent declared after its child.
