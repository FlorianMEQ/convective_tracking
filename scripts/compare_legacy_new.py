from __future__ import annotations

import argparse
import gc
import statistics
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xarray as xr

from examples.convective_rules import convective_rules
from src.old.fct_core_tracking import (
    _FEAT_WEIGHTS,
    _UC_FEAT_WEIGHTS,
    _core_features,
    _uc_features,
    find_convective_cores_3d,
    track_cc,
    track_mcs,
    track_uc,
)
from src.tracking import AdvectedIoUTracker, HierarchicalTrackingEngine, TrackingLevel, detect_objects

PARAMS = {
    "dx_km": 1.0,
    "min_alt_km": 0.0,
    "max_alt_km": 20.0,
    "tb_env_k": 240.0,
    "hydro_min_gkg": 0.01,
    "mcs_min_area_km2": 1000.0,
    "w_thresh": 1.0,
    "core_w_thresh": 4.0,
}

LEGACY_KWARGS = {k: PARAMS[k] for k in ("min_alt_km", "max_alt_km", "tb_env_k", "hydro_min_gkg", "mcs_min_area_km2", "w_thresh")}

# legacy result key -> (new level, legacy steering keys)
LEVELS = {
    "mcs": ("mcs", ("ut_steering", "vt_steering")),
    "uc": ("convective_object", ("ut_steering_ms", "vt_steering_ms")),
    "cc": ("core", ("ut_steering_ms", "vt_steering_ms")),
}

NO_FRAGMENT_COLLAPSE = -1.0
VARIABLES = ("WT", "UT", "VT", "MRI", "MRC", "MRS", "MRG", "MRR", "MSG2_IR108")


def synthetic_frames(n_steps=6, ny=80, nx=100, nz=40):
    """One cold shield over two separate updraft towers, drifting 1 px per 5 min."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    level = np.arange(nz) * 500.0
    u = 1000.0 / 300.0
    frames = []
    for t in range(n_steps):
        cy, cx = 40, 35 + t
        wt = np.zeros((nz, ny, nx), dtype=np.float32)
        for dx in (-9, 9):
            wt[5:15, (yy - cy) ** 2 + (xx - cx - dx) ** 2 <= 9] = 6.0
        tb = np.where((yy - cy) ** 2 + (xx - cx) ** 2 <= 20**2, 220.0, 250.0).astype(np.float32)
        variables = {
            "WT": wt,
            "MRI": np.where(wt > 0, 0.05, 0.0).astype(np.float32),
            "UT": np.full_like(wt, u),
            "VT": np.zeros_like(wt),
            "MRC": np.zeros_like(wt),
        }
        data = {k: (("level", "nj", "ni"), v) for k, v in variables.items()}
        data["MSG2_IR108"] = (("nj", "ni"), tb)
        time = np.datetime64("2024-01-01T00:00:00") + np.timedelta64(5 * t, "m")
        frames.append(xr.Dataset(data, coords={"level": level, "time": time.astype("datetime64[ns]")}))
    return frames


def netcdf_frames(path, max_steps=None, start_step=0):
    with xr.open_dataset(path) as ds:
        keep = [v for v in VARIABLES if v in ds.data_vars]
        ds = ds[keep]
        if "time" not in ds.dims:
            return [ds.load().astype(np.float32)]
        stop = ds.sizes["time"] if max_steps is None else min(ds.sizes["time"], start_step + max_steps)
        if start_step >= stop:
            raise ValueError(f"start_step={start_step} leaves no timestep (file has {ds.sizes['time']})")
        return [ds.isel(time=i).load().astype(np.float32) for i in range(start_step, stop)]


def time_axis(frames):
    labels = [str(f["time"].values).split("T")[-1].split(".")[0] for f in frames]
    stamps = [float(pd.Timestamp(f["time"].values).timestamp()) for f in frames]
    return labels, stamps


def build_levels():
    rules = convective_rules(PARAMS)
    common = {"dx_meters": PARAMS["dx_km"] * 1000.0}
    return [
        TrackingLevel("mcs", detect_objects, None, rules["mcs"], tracker=AdvectedIoUTracker(
            "mcs", index_key="indices_2d", min_score=0.05, patience=4, max_speed_ms=30.0, **common)),
        TrackingLevel("mcs_body", detect_objects, "mcs", rules["mcs_body"]),
        TrackingLevel("convective_object", detect_objects, None, rules["convective_object"], tracker=AdvectedIoUTracker(
            "co", min_score=0.1, patience=2, min_iou=0.05, pre_filter_pixels=60.0, shape_buffer_pixels=5.0,
            gap_bridge_max_dist_pixels=20.0, feature_fn=_uc_features, feature_weights=_UC_FEAT_WEIGHTS,
            w_overlap=0.7, w_morph=0.3, **common)),
        TrackingLevel("core", detect_objects, "convective_object", rules["core"], tracker=AdvectedIoUTracker(
            "core", min_score=0.1, patience=1, min_iou=0.02, feature_fn=_core_features,
            feature_weights=_FEAT_WEIGHTS, **common)),
    ]


def legacy_detect(frames):
    return [find_convective_cores_3d(ds, **LEGACY_KWARGS)[0] for ds in frames]


def legacy_track(steps):
    return {
        "mcs": track_mcs(steps, mcs_min_area_bt_km2=None, mcs_bt_overshoot_k=None, mcs_pr_max_mmh=None,
                         mcs_min_duration_minutes=None, max_fragment_span_s=NO_FRAGMENT_COLLAPSE),
        "uc": track_uc(steps, max_fragment_span_s=NO_FRAGMENT_COLLAPSE),
        "cc": track_cc(steps, max_fragment_span_s=NO_FRAGMENT_COLLAPSE),
    }


def new_detect(engine, frames, labels, stamps):
    return [engine.detect_time_step(ds, i, labels[i], stamps[i]) for i, ds in enumerate(frames)]


def new_track(engine, steps):
    engine.reset()
    for step in steps:
        for level in engine.levels:
            level.track(step["time_index"], step["time"], step["timestamp_s"], step["levels"][level.name])
    return engine.finalize_tracks()


def as_legacy(steps):
    """New detections in the legacy result format, for the legacy trackers."""
    out = []
    for step in steps:
        entry = {"time": step["time"], "timestamp_s": step["timestamp_s"]}
        for key, (name, (u_key, v_key)) in LEVELS.items():
            objects = []
            for obj in step["levels"][name]:
                obj = dict(obj, **{f"{key}_id": obj["object_id"], u_key: obj.get("u_steering", 0.0), v_key: obj.get("v_steering", 0.0)})
                obj["indices_2d"] = [tuple(p) for p in obj["indices_2d"].tolist()]
                objects.append(obj)
            entry[key] = objects
        out.append(entry)
    return out


def timed(fn, repeat):
    fn()
    times = []
    for _ in range(repeat):
        gc.collect()
        start = perf_counter()
        result = fn()
        times.append(perf_counter() - start)
    return result, times


def links(tracks, steps, level_name):
    """Set of (kind, time index, source object id, target object id) between adjacent timesteps."""
    when = {o["object_id"]: s["time_index"] for s in steps for o in s["levels"][level_name]}
    index_of = {s["time"]: s["time_index"] for s in steps}
    observed = {tid: [(when[o["object_id"]], o["object_id"]) for o in tr["history"]] for tid, tr in tracks.items()}

    def at(tid, t):
        return next((oid for ti, oid in observed.get(tid, ()) if ti == t), None)

    edges = set()
    for obs in observed.values():
        edges |= {("continue", t0, a, b) for (t0, a), (t1, b) in zip(obs, obs[1:]) if t1 == t0 + 1}
    for tid, tr in tracks.items():
        parent = tr.get("parent_id")
        if parent in tracks:
            t, child = observed[tid][0]
            source = at(parent, t - 1)
            if source:
                edges.add(("split", t - 1, source, child))
        for event in tr.get("events", ()):
            t = index_of.get(event.get("time"))
            if event.get("type") == "merge" and t is not None:
                source, target = at(tid, t - 1), at(event.get("into"), t)
                if source and target:
                    edges.add(("merge", t - 1, source, target))
    return edges


def compare_links(legacy_tracks, new_tracks, steps):
    rows = []
    for key, (name, _) in LEVELS.items():
        old, new = links(legacy_tracks[key], steps, name), links(new_tracks[name], steps, name)
        for kind in ("continue", "split", "merge"):
            o = {e for e in old if e[0] == kind}
            n = {e for e in new if e[0] == kind}
            rows.append({"level": key, "link": kind, "legacy": len(o), "new": len(n), "common": len(o & n),
                         "legacy_only": len(o - n), "new_only": len(n - o)})
    return pd.DataFrame(rows)


def compare_counts(legacy_steps, new_steps, labels):
    data = {}
    for key, (name, _) in LEVELS.items():
        data[(key, "legacy")] = [len(s[key]) for s in legacy_steps]
        data[(key, "new")] = [len(s["levels"][name]) for s in new_steps]
    return pd.DataFrame(data, index=pd.Index(labels, name="time"))


def timing_table(times, n_steps):
    rows = []
    for stage in ("detection", "tracking", "total"):
        old, new = times["legacy"][stage], times["new"][stage]
        rows.append({
            "stage": stage,
            "legacy best (s)": min(old), "legacy median (s)": statistics.median(old),
            "new best (s)": min(new), "new median (s)": statistics.median(new),
            "legacy ms/step": 1000 * min(old) / n_steps, "new ms/step": 1000 * min(new) / n_steps,
            "speedup": min(old) / min(new) if min(new) > 0 else float("inf"),
        })
    return pd.DataFrame(rows).set_index("stage")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--netcdf", "-i", type=Path, help="real NetCDF file (default: synthetic scene)")
    parser.add_argument("--max-steps", type=int, help="number of timesteps to load")
    parser.add_argument("--start-step", type=int, default=0, help="first timestep to load")
    parser.add_argument("--repeat", type=int, default=3, help="timed runs per stage, after one warm-up (default: 3)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frames = netcdf_frames(args.netcdf, args.max_steps, args.start_step) if args.netcdf else synthetic_frames()
    labels, stamps = time_axis(frames)
    engine = HierarchicalTrackingEngine(build_levels())
    print(f"{len(frames)} timesteps, {args.repeat} timed runs per stage")

    legacy_steps, t_old_det = timed(lambda: legacy_detect(frames), args.repeat)
    _, t_old_trk = timed(lambda: legacy_track(legacy_steps), args.repeat)
    new_steps, t_new_det = timed(lambda: new_detect(engine, frames, labels, stamps), args.repeat)
    new_tracks, t_new_trk = timed(lambda: new_track(engine, new_steps), args.repeat)
    times = {
        "legacy": {"detection": t_old_det, "tracking": t_old_trk, "total": [a + b for a, b in zip(t_old_det, t_old_trk)]},
        "new": {"detection": t_new_det, "tracking": t_new_trk, "total": [a + b for a, b in zip(t_new_det, t_new_trk)]},
    }

    pd.set_option("display.width", 160)
    counts = compare_counts(legacy_steps, new_steps, labels)
    print("\n=== Objects per timestep ===")
    print(counts.to_string())
    body = Counter(o.get("parent_object_id") for s in new_steps for o in s["levels"]["mcs_body"])
    print(f"(mcs_body, not compared: {sum(body.values())} objects below {len(body)} shields)")

    transitions = compare_links(legacy_track(as_legacy(new_steps)), new_tracks, new_steps)
    print("\n=== Tracking links on the same new detections ===")
    print(transitions.to_string(index=False))

    print("\n=== Timing ===")
    print(timing_table(times, len(frames)).round(3).to_string())

    same_counts = all((counts[(k, "legacy")] == counts[(k, "new")]).all() for k in LEVELS)
    same_links = bool((transitions[["legacy_only", "new_only"]] == 0).all().all())
    print("\nRESULT:", "OK" if same_counts and same_links else "DIVERGENCE",
          f"(counts {'identical' if same_counts else 'differ'}, links {'identical' if same_links else 'differ'})")
    return 0 if same_counts and same_links else 1


if __name__ == "__main__":
    raise SystemExit(main())
