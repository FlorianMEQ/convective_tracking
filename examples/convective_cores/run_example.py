from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import xarray as xr

from lib.fct_core_tracking import track_conv_objects, track_convective_cores, track_mcs
from lib.tracking import HierarchicalTrackingEngine
from .convective_levels import build_convective_levels


def _to_legacy_results(
    engine_results: List[Dict[str, Any]],
    timestamps_s: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Convert generic engine output to the legacy results_by_time schema."""
    legacy: List[Dict[str, Any]] = []
    for step in engine_results:
        idx = int(step.get("time_index", 0))
        if timestamps_s is not None and idx < len(timestamps_s):
            timestamp_s = float(timestamps_s[idx])
        else:
            timestamp_s = float(idx * 30.0)

        levels = step.get("levels", {})
        mcs = levels.get("mcs", [])
        conv_objects = levels.get("convective_object", [])
        cores = levels.get("core", [])
        legacy.append(
            {
                "time": step.get("time", str(step.get("time_index", 0))),
                "timestamp_s": timestamp_s,
                "mcs": mcs,
                "n_mcs": len(mcs),
                "conv_objects": conv_objects,
                "n_conv_objects": len(conv_objects),
                "cores": cores,
                "n_cores": len(cores),
            }
        )
    return legacy


def run_convective_example(
    datasets: Iterable[Any],
    legacy_detection_kwargs: Optional[Dict[str, Any]] = None,
    mcs_track_kwargs: Optional[Dict[str, Any]] = None,
    conv_object_track_kwargs: Optional[Dict[str, Any]] = None,
    core_track_kwargs: Optional[Dict[str, Any]] = None,
):
    """Run the new hierarchy with legacy-equivalent convective behavior."""
    levels = build_convective_levels(legacy_detection_kwargs=legacy_detection_kwargs)
    dataset_list = list(datasets)
    time_values: List[str] = []
    timestamps_s: List[float] = []

    for idx, ds in enumerate(dataset_list):
        if hasattr(ds, "coords") and "time" in ds.coords:
            raw_time = ds["time"].values
            time_values.append(str(raw_time))
            timestamps_s.append(float(pd.Timestamp(raw_time).timestamp()))
        else:
            time_values.append(str(idx))
            timestamps_s.append(float(idx * 30.0))

    engine = HierarchicalTrackingEngine(levels=levels)
    engine_results = engine.run(datasets=dataset_list, time_values=time_values)
    legacy_results = _to_legacy_results(engine_results, timestamps_s=timestamps_s)

    return {
        "results_by_time": legacy_results,
        "tracks_by_level": {
            "mcs": track_mcs(legacy_results, **(mcs_track_kwargs or {})),
            "convective_object": track_conv_objects(
                legacy_results,
                **(conv_object_track_kwargs or {}),
            ),
            "core": track_convective_cores(legacy_results, **(core_track_kwargs or {})),
        },
    }


def run_from_netcdf(
    path: str,
    legacy_detection_kwargs: Optional[Dict[str, Any]] = None,
    mcs_track_kwargs: Optional[Dict[str, Any]] = None,
    conv_object_track_kwargs: Optional[Dict[str, Any]] = None,
    core_track_kwargs: Optional[Dict[str, Any]] = None,
) -> dict:
    ds = xr.open_dataset(path)
    if "time" in ds.dims and int(ds.dims["time"]) > 1:
        frames: List[Any] = [ds.isel(time=i) for i in range(int(ds.dims["time"]))]
    else:
        frames = [ds]
    return run_convective_example(
        frames,
        legacy_detection_kwargs=legacy_detection_kwargs,
        mcs_track_kwargs=mcs_track_kwargs,
        conv_object_track_kwargs=conv_object_track_kwargs,
        core_track_kwargs=core_track_kwargs,
    )
