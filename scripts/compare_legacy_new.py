from __future__ import annotations

from time import perf_counter
import argparse
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import xarray as xr
xr.set_options(warn_for_unclosed_files=False)
from examples.convective_rules import convective_rules
from src.tracking import AdvectedIoUTracker, HierarchicalTrackingEngine, TrackingLevel, detect_objects
from src.old.fct_core_tracking import (
    _UC_FEAT_WEIGHTS,
    _FEAT_WEIGHTS,
    _uc_features,
    _core_features,
    find_convective_cores_3d,
    track_uc,
    track_cc,
    track_mcs,
)

NJ = NI = 64
NLEV = 100
CENTER0 = (20, 20)  # (y, x) at timestep 0
RADIUS_ENVELOPE = 12  # Tb <= tb_env_k envelope (Level A)
RADIUS_SEED = 7       # Tb <= tb_seed_k seed (Level A)
RADIUS_UPDRAFT = 3    # WT/hydrometeor disk (Levels B & C)
Z_LOW, Z_HIGH = 5, 15  # vertical slab of the updraft
N_STEPS = 6
SHIFT_PER_STEP = 1  # x displacement (pixels) between timesteps


def _disk_mask(ny: int, nx: int, cy: float, cx: float, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0:ny, 0:nx]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2

def extract_time_values_and_timestamps(
    datasets,
    fallback_step_seconds = 30.0,
):
    dataset_list = list(datasets)
    time_values = []
    timestamps_s = []

    for idx, ds in enumerate(dataset_list):
        if hasattr(ds, "coords") and "time" in ds.coords:
            raw_time = ds["time"].values
            time_values.append(str(raw_time))
            timestamps_s.append(float(pd.Timestamp(raw_time).timestamp()))
        else:
            time_values.append(str(idx))
            timestamps_s.append(float(idx * fallback_step_seconds))

    return {"datasets": dataset_list, "time_values": time_values, "timestamps_s": timestamps_s}

def to_legacy_results_by_time(
    engine_results,
    *,
    result_key_by_level,
    timestamps_s = None,
):
    legacy = []

    for step in engine_results:
        idx = int(step.get("time_index", 0))
        has_timestamp = timestamps_s is not None and idx < len(timestamps_s)
        entry = {
            "time": step.get("time", str(idx)),
            "timestamp_s": float(timestamps_s[idx]) if has_timestamp else idx * 30.0,
        }

        for level_name, result_key in result_key_by_level.items():
            objs = step.get("levels", {}).get(level_name, [])
            entry[result_key] = objs
            entry[f"n_{result_key}"] = len(objs)

        legacy.append(entry)

    return legacy

def make_synthetic_dataset(t_idx: int) -> xr.Dataset:
    """One timestep with a single convective cluster, shifted in x according to t_idx."""
    cy, cx = CENTER0[0], CENTER0[1] + t_idx * SHIFT_PER_STEP

    level = np.linspace(0.0, 60000.0, NLEV)
    updraft_xy = _disk_mask(NJ, NI, cy, cx, RADIUS_UPDRAFT)

    wt = np.zeros((NLEV, NJ, NI), dtype=np.float32)
    wt[Z_LOW:Z_HIGH, updraft_xy] = 6.0

    mri = np.zeros_like(wt)
    mri[Z_LOW:Z_HIGH, updraft_xy] = 0.05

    zeros3d = np.zeros_like(wt)
    ut = np.zeros_like(wt)
    vt = np.zeros_like(wt)

    tb = np.full((NJ, NI), 250.0, dtype=np.float32)
    tb[_disk_mask(NJ, NI, cy, cx, RADIUS_ENVELOPE)] = 220.0
    tb[_disk_mask(NJ, NI, cy, cx, RADIUS_SEED)] = 185.0

    time_val = (
        np.datetime64("2024-01-01T00:00:00") + np.timedelta64(t_idx * 5, "m")
    ).astype("datetime64[ns]")

    return xr.Dataset(
        {
            "WT": (("level", "nj", "ni"), wt),
            "UT": (("level", "nj", "ni"), ut),
            "VT": (("level", "nj", "ni"), vt),
            "MRI": (("level", "nj", "ni"), mri),
            "MRC": (("level", "nj", "ni"), zeros3d),
            "MRS": (("level", "nj", "ni"), zeros3d),
            "MRG": (("level", "nj", "ni"), zeros3d),
            "MRR": (("level", "nj", "ni"), zeros3d),
            "MSG2_IR108": (("nj", "ni"), tb),
        },
        coords={
            "level": level,
            "nj": np.arange(NJ),
            "ni": np.arange(NI),
            "time": time_val,
        },
    )

def load_datasets_from_netcdf(
    path: Path,
    max_steps=None,
    start_step: int = 0,
) -> list[xr.Dataset]:
    """Load a NetCDF file with **minimal memory usage**."""
    # Liste des variables **strictement nécessaires** pour ton script
    REQUIRED_VARS = {
        "WT", "UT", "VT", "MRI", "MRC", "MRS", "MRG", "MRR",
        "MSG2_IR108", "level", "nj", "ni", "time"
    }

    # Ouvre le fichier avec **chunking optimal** (1 timestep à la fois)
    ds = xr.open_dataset(
        path,
        chunks={'time': 1, 'level': 20},  # Chunks par timestep + tranches verticales
        engine='h5netcdf'  # Plus stable que 'netcdf4' pour les gros fichiers
    )

    # Supprime les variables inutiles **avant chargement**
    vars_to_drop = [v for v in ds.data_vars if v not in REQUIRED_VARS]
    if vars_to_drop:
        ds = ds.drop_vars(vars_to_drop)

    if "time" in ds.dims:
        n_time = int(ds.sizes["time"])
        end = n_time if max_steps is None else min(n_time, start_step + max_steps)
        if start_step >= n_time:
            raise ValueError(f"start_step={start_step} >= number of available timesteps ({n_time})")

        frames = []
        for i in range(start_step, end):
            # Charge **UNIQUEMENT** le timestep i (pas tout le fichier !)
            frame = ds.isel(time=i).load()
            # Convertit en **float32** (50% de mémoire en moins vs float64)
            frame = frame.astype(np.float32)
            frames.append(frame)
    else:
        frames = [ds.astype(np.float32).load()]

    ds.close()  # Ferme explicitement le fichier
    return frames


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--netcdf",
        "-i",
        type=Path,
        default=None,
        help=(
            "Path to a real NetCDF file to use instead of the synthetic "
            "dataset (must expose the variables/coords expected by "
            "find_convective_cores_3d: WT, MRI, MRC, level/level coords, etc.)."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of timesteps to load from the NetCDF (default: all).",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Index of the first timestep to load from the NetCDF (default: 0).",
    )
    return parser.parse_args(argv)


def run_old_pipeline(datasets: list[xr.Dataset]) -> dict:
    results_by_time: list[dict] = []

    print("\n=== OLD PIPELINE ===")
    start_old = perf_counter()
    for ds in datasets:
        results_by_time.extend(find_convective_cores_3d(ds))
    old_time = perf_counter() - start_old
    print(f"Old pipeline: {old_time:.3f} seconds")
    return {
        "results_by_time": results_by_time,
        "tracks_by_level": {
            "mcs": track_mcs(
                results_by_time,
                mcs_min_area_bt_km2=None,  
                mcs_bt_overshoot_k=None,    
                mcs_pr_max_mmh=None,       
                mcs_min_duration_minutes=None
            ),
            "conv_objects": track_uc(results_by_time),
            "cores": track_cc(results_by_time),
        },
    }


# Parameters of the new detectors (scientific values of the old code).
DETECTION_PARAMS = {
    "vertical_dim": "level",
    "dx_km": 1.0,
    "min_alt_km": 0.0,
    "max_alt_km": 20.0,
    "wt_var": "WT",
    "ut_var": "UT",
    "vt_var": "VT",
    "mri_var": "MRI",
    "mrc_var": "MRC",
    "mrs_var": "MRS",
    "mrg_var": "MRG",
    "mrr_var": "MRR",
    "tb_var": "MSG2_IR108",
    "tb_seed_pct": 0.1,
    "tb_seed_k": 190.0,
    "tb_env_k": 240.0,
    "mcs_min_area_km2": 1000.0,
    "w_thresh": 1.0,
    "hydro_min_gkg": 0.01,
    "connectivity": 26,
    "core_w_thresh": 4.0,
    "segmentation_connectivity": 18,
    "h_adapt_slope": 0.3,
    "h_adapt_min": 1.0,
    "h_adapt_max": 10.0,
    "min_width_threshold_km": 1.0,
    "stacked_seeds_xy_radius": 3,
}


# Dans /workspace/scripts/compare_legacy_new.py

def build_detection_levels():
    """Build detection levels, each bound to its generic tracker."""
    rules = convective_rules(DETECTION_PARAMS)
    return [
        TrackingLevel(
            "mcs",
            detect_objects,
            None,
            rules["mcs"],
            tracker=AdvectedIoUTracker(
                "mcs",
                index_key="indices_2d",
                min_score=0.05,
                patience=4,
                dx_meters=1000.0,
                max_speed_ms=30.0,
                steering_fields=("ut_steering", "vt_steering"),
            ),
        ),
        TrackingLevel(
            "convective_object",
            detect_objects,
            "mcs",
            rules["convective_object"],
            tracker=AdvectedIoUTracker(
                "co",
                index_key="indices_3d",
                min_score=0.1,
                patience=2,
                dx_meters=1000.0,
                min_iou=0.05,
                pre_filter_pixels=60.0,
                advection_error_ms=5.0,
                shape_buffer_pixels=5.0,
                gap_bridge_seconds=120.0,
                gap_bridge_max_dist_pixels=20.0,
                feature_fn=_uc_features,
                feature_weights=_UC_FEAT_WEIGHTS,
                feat_ema_alpha=0.7,
                w_overlap=0.7,
                w_morph=0.3,
                steering_fields=("ut_steering", "vt_steering"),
                detect_splits_merges=True,
            ),
        ),
        TrackingLevel(
            "core",
            detect_objects,
            "convective_object",
            rules["core"],
            tracker=AdvectedIoUTracker(
                "core",
                index_key="indices_3d",
                min_score=0.1,
                patience=1,
                min_iou=0.02,
                dx_meters=1000.0,
                pre_filter_pixels=30.0,
                advection_error_ms=5.0,
                shape_buffer_pixels=3.0,
                gap_bridge_seconds=120.0,
                gap_bridge_max_dist_pixels=10.0,
                feature_fn=_core_features,
                feature_weights=_FEAT_WEIGHTS,
                feat_ema_alpha=0.7,
                w_overlap=0.6,
                w_morph=0.4,
                steering_fields=("ut_steering", "vt_steering"),
                detect_splits_merges=True,
            ),
        ),
    ]

def run_new_pipeline(datasets: list[xr.Dataset]) -> dict:
    levels = build_detection_levels()
    time_info = extract_time_values_and_timestamps(datasets)
    engine = HierarchicalTrackingEngine(levels)

    print("\n=== NEW PIPELINE ===")
    start_new = perf_counter()
    generic_result = engine.run_and_track(
        datasets=time_info["datasets"],
        time_values=time_info["time_values"],
        timestamps_s=time_info["timestamps_s"],
    )
    new_time = perf_counter() - start_new
    print(f"New pipeline: {new_time:.3f} seconds")

    result_key_by_level = {
        "mcs": "mcs",          
        "convective_object": "uc",  
        "core": "cc", 
    }
    legacy_results = to_legacy_results_by_time(
        generic_result["results_by_time"],
        result_key_by_level=result_key_by_level,
        timestamps_s=time_info["timestamps_s"],
    )
    return {
        "results_by_time": legacy_results,
        "tracks_by_level": {
            "mcs": track_mcs(legacy_results),
            "uc": track_uc(legacy_results),
            "cc": track_cc(legacy_results),
        },
        "generic_tracks_by_level": generic_result["tracks_by_level"],
    }


def _footprint(obj, index_key):
    indices = obj.get(index_key, [])
    if len(indices) == 0:
        return frozenset()
    return frozenset(tuple(idx) for idx in indices)

def _transition_edges(tracks, results_by_time, result_key, index_key):
    """Return continuation, split, and merge links between adjacent steps."""
    time_by_label = {}
    for time_index, step in enumerate(results_by_time):
        time_by_label[step["time"]] = time_index

    # Map each object to its time index and footprint
    obj_to_info = {}
    for time_index, step in enumerate(results_by_time):
        for obj in step.get(result_key, []):
            footprint = _footprint(obj, index_key)
            obj_to_info[id(obj)] = (time_index, footprint)

    observations_by_track = {}
    for track_id, track in tracks.items():
        observations_by_track[track_id] = [
            obj_to_info[id(obj)]
            for obj in track.get("history", [])
            if id(obj) in obj_to_info
        ]

    edges = set()
    for observations in observations_by_track.values():
        for (previous_time, previous_object), (current_time, current_object) in zip(
            observations, observations[1:]
        ):
            if current_time == previous_time + 1:
                edges.add(("continue", previous_time, previous_object, current_object))

    time_by_object = {}
    time_by_label = {}
    for time_index, step in enumerate(results_by_time):
        time_by_label[step["time"]] = time_index
        for obj in step.get(result_key, []):
            time_by_object[id(obj)] = time_index

    observations_by_track = {}
    for track_id, track in tracks.items():
        observations_by_track[track_id] = [
            (time_by_object[id(obj)], _footprint(obj, index_key))
            for obj in track.get("history", [])
            if id(obj) in time_by_object
        ]

    edges = set()
    for observations in observations_by_track.values():
        for (previous_time, previous_object), (current_time, current_object) in zip(
            observations, observations[1:]
        ):
            if current_time == previous_time + 1:
                edges.add(("continue", previous_time, previous_object, current_object))

    for child_id, child in tracks.items():
        parent_id = child.get("parent_id")
        child_observations = observations_by_track[child_id]
        if parent_id not in tracks or not child_observations:
            continue
        child_time, child_object = child_observations[0]
        parent = tracks[parent_id]
        split_at_child_time = any(
            event.get("type") == "split"
            and time_by_label.get(event.get("time")) == child_time
            for event in parent.get("events", [])
        )
        if not split_at_child_time:
            continue
        previous = [
            observation
            for observation in observations_by_track[parent_id]
            if observation[0] == child_time - 1
        ]
        if previous:
            edges.add(("split", child_time - 1, previous[-1][1], child_object))

    for source_id, source in tracks.items():
        source_observations = observations_by_track[source_id]
        for event in source.get("events", []):
            target_id = event.get("into")
            event_time = time_by_label.get(event.get("time"))
            if event.get("type") != "merge" or target_id not in tracks or event_time is None:
                continue
            previous = [
                observation
                for observation in source_observations
                if observation[0] == event_time - 1
            ]
            target = [
                observation
                for observation in observations_by_track[target_id]
                if observation[0] == event_time
            ]
            if previous and target:
                edges.add(("merge", event_time - 1, previous[-1][1], target[0][1]))
    return edges


def compare_tracking_transitions(old_tracks, new_tracks, detections):
    """Compare both tracker implementations on the exact same detected objects."""
    rows = []
    for result_key, new_level_name, index_key in (
        ("mcs", "mcs", "indices_3d"),
        ("uc", "convective_object", "indices_3d"),
        ("cc", "core", "indices_3d"),
    ):
        old_edges = _transition_edges(
            old_tracks[result_key], detections["results_by_time"], result_key, index_key
        )
        new_edges = _transition_edges(
            new_tracks[new_level_name], detections["results_by_time"], result_key, index_key
        )
        for time_index in range(max(0, len(detections["results_by_time"]) - 1)):
            transition = (
                f"{detections['results_by_time'][time_index]['time']} -> "
                f"{detections['results_by_time'][time_index + 1]['time']}"
            )
            for link_type in ("continue", "split", "merge"):
                old_at_time = {
                    edge for edge in old_edges if edge[1] == time_index and edge[0] == link_type
                }
                new_at_time = {
                    edge for edge in new_edges if edge[1] == time_index and edge[0] == link_type
                }
                rows.append(
                    {
                        "level": result_key,
                        "link_type": link_type,
                        "transition": transition,
                        "old_links": len(old_at_time),
                        "new_links": len(new_at_time),
                        "same_links": len(old_at_time & new_at_time),
                        "old_only": len(old_at_time - new_at_time),
                        "new_only": len(new_at_time - old_at_time),
                    }
                )
    return pd.DataFrame(rows)


def summarize_counts(result: dict) -> pd.DataFrame:
    rows = []
    for step in result["results_by_time"]:
        rows.append(
            {
                "time": step.get("time"),
                "n_mcs": len(step.get("mcs", [])),
                "n_conv_objects": len(step.get("conv_objects", [])),
                "n_cores": len(step.get("cores", [])),
            }
        )
    return pd.DataFrame(rows)


def summarize_tracks(result: dict) -> dict:
    return {name: len(tracks) for name, tracks in result["tracks_by_level"].items()}


def main(argv = None) -> int:
    args = parse_args(argv)

    if args.netcdf is not None:
        print(f"Loading real dataset: {args.netcdf}")
        datasets = load_datasets_from_netcdf(
            args.netcdf, max_steps=args.max_steps, start_step=args.start_step
        )
        print(f"{len(datasets)} timesteps loaded")
    else:
        datasets = [make_synthetic_dataset(t) for t in range(N_STEPS)]

    old_result = run_old_pipeline(datasets)

    new_result = run_new_pipeline(datasets)
    
    old_counts = summarize_counts(old_result)
    new_counts = summarize_counts(new_result)
    count_cols = ["n_mcs", "n_conv_objects", "n_cores"]

    print("=== Object counts per timestep ===")
    print("--- Old pipeline ---")
    print(old_counts.to_string(index=False))
    print("--- New detector ---")
    print(new_counts.to_string(index=False))

    counts_diff = old_counts[count_cols] - new_counts[count_cols]
    print("--- Difference (old - new) ---")
    print(counts_diff.to_string(index=False))

    print()
    print("=== Number of tracks on the new detections ===")
    print("Legacy tracker:", summarize_tracks(new_result))
    generic_track_counts = {
        name: len(tracks) for name, tracks in new_result["generic_tracks_by_level"].items()
    }
    print("Generic tracker:", generic_track_counts)

    transitions = compare_tracking_transitions(
        new_result["tracks_by_level"], new_result["generic_tracks_by_level"], new_result
    )
    print()
    print("=== Adjacent-timestep tracking comparison on identical detected objects ===")
    print(transitions.to_string(index=False))

    detections_match = bool((counts_diff == 0).all().all())
    tracking_matches = bool(
        transitions.empty or ((transitions["old_only"] == 0) & (transitions["new_only"] == 0)).all()
    )
    print()
    print(
        "RESULT:",
        "OK - detection counts and timestep-to-timestep links match"
        if detections_match and tracking_matches
        else "DIVERGENCE detected",
    )
    return 0 if detections_match and tracking_matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
