"""Non-regression comparison: old pipeline vs new pipeline.

This script runs two variants on the same synthetic dataset (a single
Gaussian/disk "convective cluster", over several timesteps):

  1. "Old"  : ``src.old.fct_core_tracking.find_convective_cores_3d`` +
     ``track_mcs`` / ``track_conv_objects`` / ``track_convective_cores``.

  2. "New detection + legacy tracking" : the generic engine
     (``src.tracking``) with the generic detector ``src.tracking.detection``
     configured by the rules of ``examples.convective_rules`` (through
     ``examples.convective_rules``), using both the legacy tracking functions
     and the new ``AdvectedIoUTracker``.

     This bridge isolates the question: "does the new detection produce the
     same objects as the old one?" independently of the tracker. It also
     compares the legacy trackers and ``AdvectedIoUTracker`` on the exact same
     new detections, checking whether each object is linked to the same object
     at the next timestep.

Usage
-----
    # Synthetic dataset (default)
    conda run -n mesonh python scripts/compare_legacy_new.py

    # Real NetCDF file (e.g. MesoNH output), all timesteps
    conda run -n mesonh python scripts/compare_legacy_new.py --netcdf /path/to/file.nc

    # Limit the number of loaded timesteps (real files are often large)
    conda run -n mesonh python scripts/compare_legacy_new.py --netcdf file.nc --max-steps 5 --start-step 10

The script prints the object counts per timestep and per level for both
pipelines, as well as the number of tracks obtained, and returns a non-zero
exit code if a divergence is detected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

# Allow direct execution (`python scripts/compare_legacy_new.py`) in addition
# to `python -m scripts.compare_legacy_new`: without this, only the scripts/
# folder is added to sys.path and `examples`/`src` cannot be found.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import xarray as xr

from examples.convective_rules import convective_rules
from src.tracking import AdvectedIoUTracker, HierarchicalTrackingEngine, TrackingLevel, detect_objects
from src.old.fct_core_tracking import (
    _CO_FEAT_WEIGHTS,
    _FEAT_WEIGHTS,
    _conv_obj_features,
    _core_features,
    find_convective_cores_3d,
    track_conv_objects,
    track_convective_cores,
    track_mcs,
)
from src.tracking.legacy_adapter import (
    extract_time_values_and_timestamps,
    to_legacy_results_by_time,
)

# --- Geometry of the synthetic dataset ---
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


def make_synthetic_dataset(t_idx: int) -> xr.Dataset:
    """One timestep with a single convective cluster, shifted in x according to t_idx."""
    cy, cx = CENTER0[0], CENTER0[1] + t_idx * SHIFT_PER_STEP

    level = np.linspace(0.0, 20000.0, NLEV)
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
        coords={"level": level, "level": level, "time": time_val},
    )


def load_datasets_from_netcdf(
    path: Path,
    max_steps: Optional[int] = None,
    start_step: int = 0,
) -> list[xr.Dataset]:
    """Load a real NetCDF file and split it into a list of per-timestep Datasets
    (a single element if the ``time`` dimension is absent).
    """
    with xr.open_dataset(path) as ds:
        if "time" in ds.dims:
            n_time = int(ds.sizes["time"])
            end = n_time if max_steps is None else min(n_time, start_step + max_steps)
            if start_step >= n_time:
                raise ValueError(
                    f"start_step={start_step} >= number of available timesteps ({n_time})"
                )
            frames = [ds.isel(time=i).load() for i in range(start_step, end)]
        else:
            frames = [ds.load()]
    return frames


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
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
    for ds in datasets:
        results_by_time.extend(find_convective_cores_3d(ds))

    return {
        "results_by_time": results_by_time,
        "tracks_by_level": {
            "mcs": track_mcs(results_by_time),
            "conv_objects": track_conv_objects(results_by_time),
            "cores": track_convective_cores(results_by_time),
        },
    }


# Parameters of the new detectors (scientific values of the old code).
DETECTION_PARAMS = {
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
    "tb_env_k": 235.0,
    "mcs_min_area_km2": 100.0,
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


def build_detection_levels() -> list[TrackingLevel]:
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
            None,
            rules["convective_object"],
            tracker=AdvectedIoUTracker(
                "co",
                index_key="indices_3d",
                min_score=0.1,
                patience=2,
                dx_meters=1000.0,
                pre_filter_pixels=60.0,
                advection_error_ms=5.0,
                shape_buffer_pixels=5.0,
                gap_bridge_seconds=120.0,
                gap_bridge_max_dist_pixels=20.0,
                feature_fn=_conv_obj_features,
                feature_weights=_CO_FEAT_WEIGHTS,
                feat_ema_alpha=0.7,
                w_overlap=0.7,
                w_morph=0.3,
                steering_fields=("ut_steering", "vt_steering"),
                # The legacy convective-object tracker does not model these events.
                detect_splits_merges=False,
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
                patience=0,
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
            ),
        ),
    ]


def run_new_pipeline(datasets: list[xr.Dataset]) -> dict:
    levels = build_detection_levels()
    time_info = extract_time_values_and_timestamps(datasets)
    engine = HierarchicalTrackingEngine(levels)
    generic_result = engine.run_and_track(
        datasets=time_info["datasets"],
        time_values=time_info["time_values"],
        timestamps_s=time_info["timestamps_s"],
    )
    result_key_by_level = {
        "mcs": "mcs",
        "convective_object": "conv_objects",
        "core": "cores",
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
            "conv_objects": track_conv_objects(legacy_results),
            "cores": track_convective_cores(legacy_results),
        },
        "generic_tracks_by_level": generic_result["tracks_by_level"],
    }


def _footprint(obj, index_key):
    return frozenset(map(tuple, obj.get(index_key, [])))


def _transition_edges(tracks, results_by_time, result_key, index_key):
    """Return continuation, split, and merge links between adjacent steps."""
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
        ("mcs", "mcs", "indices_2d"),
        ("conv_objects", "convective_object", "indices_3d"),
        ("cores", "core", "indices_3d"),
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


def main(argv: Optional[Sequence[str]] = None) -> int:
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
