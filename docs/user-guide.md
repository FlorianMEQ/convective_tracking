# User guide

## 1. Input data

The engine processes a **list of `xarray.Dataset`, one per timestep** (e.g. `ds.isel(time=i)`).

| Element | Description |
|---|---|
| Dimensions | 3D variables: `(level, y, x)`; 2D variables (Tb): `(y, x)` |
| `level` coordinate | altitudes (m) of the grid levels; used for the `min_alt_km` / `max_alt_km` clipping |
| `level_w` coordinate | layer heights (m), required for `height_km` and `volume_km3` |
| `WT` | vertical velocity (m/s) |
| `UT`, `VT` | steering wind (m/s); used by the tracker for advection |
| `MRI`, `MRC`, `MRS`, `MRG`, `MRR` | hydrometeor mixing ratios (g/kg) |
| `MSG2_IR108` | synthetic brightness temperature (K) |
| `time` (scalar coord., optional) | timestep timestamp (pass it through `timestamps_s` to `engine.run`) |

Variable names are configurable (`wt_var`, `tb_var`, ...). A variable missing from the Dataset is ignored in a mask, and a statistic on a missing variable is `0.0`.

## 2. Physical parameters

They are passed to [`convective_rules`](../examples/convective_rules.py), which builds one rule per level.

| Parameter | Role |
|---|---|
| `dx_km` | horizontal grid spacing (km) |
| `min_alt_km`, `max_alt_km` | vertical clipping |
| `wt_var`, `ut_var`, `vt_var`, `mri_var`...`mrr_var`, `tb_var` | variable names |
| `tb_env_k` | **MCS**: envelope `Tb ≤ tb_env_k` |
| `tb_seed_h_k` | MCS: minimum depth (K) of a Tb minimum to be a seed |
| `tb_seed_pct` | MCS: seeds = percentile (%) of Tb within the envelope |
| `tb_seed_k` | MCS: seeds = `Tb ≤ tb_seed_k` (190 K by default) |
| `mcs_min_area_km2` | MCS: minimum area (100 km² by default) |
| `w_thresh` | **convective objects**: `WT` threshold (m/s) |
| `hydro_min_gkg` | convective objects: minimum sum of hydrometeors (0.01 g/kg by default) |
| `connectivity` | 3D connectivity (6, 18 or 26) |
| `core_w_thresh` | **cores**: `WT` threshold (m/s) |
| `h_adapt_slope`, `h_adapt_min`, `h_adapt_max` | cores: adaptive H-maxima height `h` |
| `min_width_threshold_km` | cores: median width below which splitting stops |
| `segmentation_connectivity` | cores: watershed connectivity (18 by default) |
| `stacked_seeds_xy_radius` | cores: radius (px) for merging vertically stacked seeds |

MCS seed priority: `tb_seed_h_k` > `tb_seed_pct` > `tb_seed_k`.

## 3. Build the hierarchy

```python
levels = [
    TrackingLevel("mcs", detect_objects, None, rules["mcs"],
                  tracker=AdvectedIoUTracker("mcs", "indices_2d", min_score=0.01, patience=4, max_speed_ms=30.0)),
    TrackingLevel("convective_object", detect_objects, "mcs", rules["convective_object"],
                  tracker=AdvectedIoUTracker("co", "indices_3d", patience=2)),
    TrackingLevel("core", detect_objects, "convective_object", rules["core"],
                  tracker=AdvectedIoUTracker("core", "indices_3d")),
]
engine = HierarchicalTrackingEngine(levels)
```

Each level is bound to **its own** tracker (`tracker=`, optional: without a tracker, the level is detected but not tracked).

Engine validation rules: unique level names, parent must exist and be declared **before** its children.

A child level detects its objects **within the footprint of each parent object**; each object receives a `parent_object_id`.

Other hierarchies are possible (e.g. `convective_object` as an independent root, not tied to the MCS: see [compare_legacy_new.py](../scripts/compare_legacy_new.py)).

## 4. Run

```python
results = engine.run(frames, time_values=labels, timestamps_s=seconds)
# or step by step:
step = engine.run_time_step(frames[0], time_index=0, time_value="12:00:00", timestamp_s=0.0)
```

Each step returns:

```python
{"time_index": 0, "time": "12:00:00",
 "levels": {"mcs": [obj, ...], "convective_object": [...], "core": [...]}}
```

See [detection.md](detection.md) for the content of an object.

## 5. Track

At each timestep, the engine feeds the tracker of each level with **all** the objects of the level (all parents combined). Tracks are retrieved at the end:

```python
out = engine.run_and_track(frames, time_values=labels, timestamps_s=seconds)
out["results_by_time"]       # detections
out["tracks_by_level"]       # {"mcs": {...}, "convective_object": {...}, "core": {...}}
# or, after engine.run(...): engine.finalize_tracks()
```

A level without a tracker returns `{}`. Details in [tracking.md](tracking.md).

## 6. Scripts

### `compare_legacy_new.py`: non-regression

Compares the old pipeline ([src/old](../src/old/fct_core_tracking.py)) to the new generic detector on a synthetic dataset (one Gaussian cluster, several timesteps) or a real file. It reports per-timestep detection counts and also runs the legacy tracker and `AdvectedIoUTracker` on the **same new-detected objects**. For each adjacent timestep pair, it compares exact object-footprint continuation, split, and merge links, independently of track ID names. The command exits non-zero if detection counts or links differ.

```bash
python scripts/compare_legacy_new.py                                   # synthetic
python scripts/compare_legacy_new.py --netcdf file.nc                  # real
python scripts/compare_legacy_new.py -i file.nc --max-steps 5 --start-step 10
```

### `diagnose_cores.py`: core detection differences

Lists the voxels that differ between the old and the new core detection.

```bash
python scripts/diagnose_cores.py --netcdf file.nc -t 0 1 2 --tb-env-k 235 --max-voxels 12
```

| Option | Default |
|---|---|
| `--netcdf`, `-i` | hard-coded path (override it) |
| `--time-index`, `-t` | `0 ... 87` |
| `--tb-env-k` | `PARAMS["tb_env_k"]` |
| `--max-voxels` | 12 |

> The scripts import `src.old.*` and `examples.*`: run them from the repository root.

## 7. Notebook

[examples/convective_cores_example.ipynb](../examples/convective_cores_example.ipynb) loads 3 timesteps of a MesoNH file, builds the three levels, runs detection and tracking, and plots the objects. Set `NETCDF_PATH` (environment variable or first data cell).
