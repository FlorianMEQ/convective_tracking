# convective_tracking

Hierarchical 3D detection and tracking of convective structures (MesoNH model output, synthetic satellite brightness temperature).

Structures are detected on several **nested levels**, then tracked in time level by level:

```
mcs                  (root)     mesoscale convective system, detected on Tb (2D)
└─ convective_object            3D convective objects (WT and hydrometeors)
   └─ core                      3D convective cores (recursive watershed on WT)
```

The algorithm is **generic**: a single detector (`detect_objects`) serves every level. What an "object" is (MCS, core, ...) is defined only by a **declarative rule** (a dictionary) attached to the level, and each level is bound to its own tracker.

## Repository content

| Path | Role |
|---|---|
| [src/tracking/](src/tracking) | Library: engine, generic detector, tracker, utility functions |
| [examples/convective_rules.py](examples/convective_rules.py) | "Convective" detection rules (MCS, objects, cores) built from physical parameters |
| [examples/convective_cores_example.ipynb](examples/convective_cores_example.ipynb) | Example notebook on 3 timesteps of a MesoNH file |
| [scripts/compare_legacy_new.py](scripts/compare_legacy_new.py) | Non-regression: detections and timestep-to-timestep tracking links |
| [scripts/diagnose_cores.py](scripts/diagnose_cores.py) | Voxel-by-voxel diagnosis of core detection differences |
| [src/old/fct_core_tracking.py](src/old/fct_core_tracking.py) | Old monolithic implementation (reference) |
| [Diagramme de classe.drawio.svg](<Diagramme de classe.drawio.svg>) | Class diagram |
| [docs/](docs) | Detailed documentation |

## Installation

Python ≥ 3.9 (tested with 3.12). Dependencies:

```bash
pip install numpy scipy scikit-image pandas xarray netcdf4
# for the notebook
pip install matplotlib jupyter
```

The repository is not an installable package: the scripts add the repository root to `sys.path`, and the notebook can be run from the root or from `examples/`. Import with `from src.tracking import ...` from the repository root.

## Quick start

```python
import xarray as xr
from examples.convective_rules import convective_rules
from src.tracking import AdvectedIoUTracker, HierarchicalTrackingEngine, TrackingLevel, detect_objects

PARAMS = {
    "dx_km": 1.0, "min_alt_km": 0.0, "max_alt_km": 20.0,
    "wt_var": "WT", "ut_var": "UT", "vt_var": "VT",
    "mri_var": "MRI", "mrc_var": "MRC", "mrs_var": "MRS", "mrg_var": "MRG", "mrr_var": "MRR",
    "tb_var": "MSG2_IR108", "tb_env_k": 230.0, "tb_seed_pct": 0.1, "mcs_min_area_km2": 100.0,
    "w_thresh": 1.0, "hydro_min_gkg": 0.01, "core_w_thresh": 4.0,
}

rules = convective_rules(PARAMS)
levels = [
    TrackingLevel("mcs", detect_objects, None, rules["mcs"],
                  tracker=AdvectedIoUTracker("mcs", "indices_2d", min_score=0.01, patience=4, max_speed_ms=30.0)),
    TrackingLevel("convective_object", detect_objects, "mcs", rules["convective_object"],
                  tracker=AdvectedIoUTracker("co", "indices_3d", patience=2)),
    TrackingLevel("core", detect_objects, "convective_object", rules["core"],
                  tracker=AdvectedIoUTracker("core", "indices_3d")),
]

ds = xr.open_dataset("simulation.nc")
frames = [ds.isel(time=i) for i in range(ds.sizes["time"])]

engine = HierarchicalTrackingEngine(levels)
out = engine.run_and_track(frames)                    # detection + tracking
cores = out["results_by_time"][0]["levels"]["core"]   # objects detected at the first timestep
tracks = out["tracks_by_level"]["core"]               # core tracks
```

## Documentation

- [User guide](docs/user-guide.md): data format, parameters, execution, scripts
- [Architecture](docs/architecture.md): modules, classes, data flow
- [Detection](docs/detection.md): reference of the detection rules
- [Temporal tracking](docs/tracking.md): `AdvectedIoUTracker`, scores, merges and splits
- [API reference](docs/api.md): public functions and classes

## Notes

- [scripts/compare_legacy_new.py](scripts/compare_legacy_new.py) and [scripts/diagnose_cores.py](scripts/diagnose_cores.py) have a default NetCDF path specific to one machine (`DEFAULT_NETCDF`): pass `--netcdf` explicitly.
- `*.nc`, `*.json`, `*.png`, `*.gif` files and the `DATA/` folder are ignored by git.
