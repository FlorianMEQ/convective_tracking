from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import xarray as xr

from src.old.fct_core_tracking import _compute_component_metrics, find_convective_cores_3d
from examples.convective_rules import convective_rules
from src.tracking import HierarchicalTrackingEngine, TrackingLevel, detect_objects
from src.tracking.functions import compute_component_metrics

DEFAULT_NETCDF = "/home/ricj/Documents/Meso-nh/mesochain/learning/cnn/DATA/MesoNH-ice3_CADDIWAF7_1km.nc"

PARAMS: Dict[str, Any] = {
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
    "tb_env_k": 230.0,
    "tb_seed_pct": 0.1,
    "mcs_min_area_km2": 100.0,
    "w_thresh": 1.0,
    "hydro_min_gkg": 0.01,
    "core_w_thresh": 4.0,
}

# (new level, key of the old result, key of the indices)
LEVELS = (
    ("mcs", "mcs", "indices_2d"),
    ("convective_object", "conv_objects", "indices_3d"),
    ("core", "cores", "indices_3d"),
)


def voxels(obj: Dict[str, Any], key: str) -> frozenset:
    return frozenset(map(tuple, obj[key]))


def mask_of(vox: frozenset) -> np.ndarray:
    """Local 3D mask of a set of voxels (z, y, x)."""
    arr = np.array(sorted(vox))
    return _mask(arr)


def _mask(arr: np.ndarray) -> np.ndarray:
    lo = arr.min(axis=0)
    mask = np.zeros(tuple(arr.max(axis=0) - lo + 1), dtype=bool)
    mask[arr[:, 0] - lo[0], arr[:, 1] - lo[1], arr[:, 2] - lo[2]] = True
    return mask


def stop_criteria(vox: frozenset, dx_km: float) -> str:
    """Recursion stopping criteria evaluated on the ``vox`` region."""
    mask = mask_of(vox)
    new = compute_component_metrics(mask, dx_km=dx_km)
    old = _compute_component_metrics(mask)
    return (
        f"max footprint width={new['width_km']:.1f} km, "
        f"per-level median width={old['width_median_km']:.1f} km"
    )


def describe_voxels(vox: frozenset, max_voxels: int) -> str:
    if not vox:
        return "none"
    arr = np.array(sorted(vox))
    levels = sorted(set(arr[:, 0].tolist()))
    shown = ", ".join(str(tuple(v)) for v in arr[:max_voxels].tolist())
    more = f" … (+{len(arr) - max_voxels})" if len(arr) > max_voxels else ""
    return f"{len(arr)} voxels, z levels={levels}\n        (z, y, x) : {shown}{more}"


def run_new(frame: xr.Dataset, time_value: str) -> Dict[str, List[Dict[str, Any]]]:
    rules = convective_rules(PARAMS)
    levels = [
        TrackingLevel("mcs", detect_objects, None, rules["mcs"]),
        TrackingLevel("convective_object", detect_objects, None, rules["convective_object"]),
        TrackingLevel("core", detect_objects, "convective_object", rules["core"]),
    ]
    return HierarchicalTrackingEngine(levels).run([frame], [time_value])[0]["levels"]


def run_old(frame: xr.Dataset) -> Dict[str, Any]:
    p = PARAMS
    return find_convective_cores_3d(
        frame,
        wt_var=p["wt_var"], ut_var=p["ut_var"], vt_var=p["vt_var"],
        mri_var=p["mri_var"], mrc_var=p["mrc_var"], mrs_var=p["mrs_var"],
        mrg_var=p["mrg_var"], mrr_var=p["mrr_var"], tb_var=p["tb_var"],
        w_thresh=p["w_thresh"], min_alt_km=p["min_alt_km"], max_alt_km=p["max_alt_km"],
        tb_seed_pct=p["tb_seed_pct"], tb_env_k=p["tb_env_k"],
        hydro_min_gkg=p["hydro_min_gkg"], mcs_min_area_km2=p["mcs_min_area_km2"],
    )[0]


def compare_levels(old: Dict[str, Any], new: Dict[str, Any]) -> bool:
    print("  Level                  old      new          old only           new only")
    all_same = True
    for name, old_key, idx_key in LEVELS:
        so = {voxels(o, idx_key) for o in old[old_key]}
        sn = {voxels(n, idx_key) for n in new[name]}
        same = so == sn
        all_same &= same
        print(
            f"  {name:<19} {len(old[old_key]):>6} {len(new[name]):>8} "
            f"{len(so - sn):>17} {len(sn - so):>18}   {'OK' if same else 'DIFFERENT'}"
        )
    return all_same


def diagnose_cores(old: Dict[str, Any], new: Dict[str, Any], max_voxels: int) -> None:
    dx_km = PARAMS["dx_km"]
    old_objs = {voxels(o, "indices_3d"): o["conv_obj_id"] for o in old["conv_objects"]}
    new_obj_vox = {o["object_id"]: voxels(o, "indices_3d") for o in new["convective_object"]}

    old_by_parent: Dict[str, List[frozenset]] = defaultdict(list)
    for core in old["cores"]:
        old_by_parent[core["conv_obj_id"]].append(voxels(core, "indices_3d"))

    new_by_parent: Dict[str, List[frozenset]] = defaultdict(list)
    unmatched_parent = 0
    for core in new["core"]:
        parent_vox = new_obj_vox.get(core["parent_object_id"])
        parent_old_id = old_objs.get(parent_vox)
        if parent_old_id is None:
            unmatched_parent += 1
            continue
        new_by_parent[parent_old_id].append(voxels(core, "indices_3d"))

    if unmatched_parent:
        print(f"  {unmatched_parent} new core(s) whose parent object does not exist in the old code.")

    n_bad = 0
    for parent_id in sorted(set(old_by_parent) | set(new_by_parent)):
        so, sn = set(old_by_parent[parent_id]), set(new_by_parent[parent_id])
        if so == sn:
            continue
        n_bad += 1
        only_old, only_new = sorted(so - sn, key=len), sorted(sn - so, key=len)
        print(
            f"\n  Convective object {parent_id}: {len(old_by_parent[parent_id])} cores (old) "
            f"vs {len(new_by_parent[parent_id])} (new)"
        )

        # Pair the disagreeing cores by maximum overlap
        used = set()
        for o in only_old:
            cand = [n for n in only_new if n not in used]
            best: Optional[frozenset] = max(cand, key=lambda n: len(n & o), default=None)
            print(f"    old : {len(o)} voxels ({stop_criteria(o, dx_km)})")
            if best is None or not (best & o):
                print("      -> no new core overlaps it")
                continue
            used.add(best)
            print(f"    new: {len(best)} voxels ({stop_criteria(best, dx_km)})")
            print(f"      in common : {len(best & o)} voxels")
            print(f"      old only  : {describe_voxels(o - best, max_voxels)}")
            print(f"      new only : {describe_voxels(best - o, max_voxels)}")
        for n in only_new:
            if n not in used:
                print(f"    new without equivalent : {len(n)} voxels ({stop_criteria(n, dx_km)})")

    if n_bad == 0:
        print("  Cores: no convective object shows any difference.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--netcdf", "-i", type=Path, default=Path(DEFAULT_NETCDF))
    parser.add_argument("--time-index", "-t", type=int, nargs="+", default=list(range(0, 88)),
                        help="Timestep indices to diagnose (default: 0 to 87).")
    parser.add_argument("--tb-env-k", type=float, default=PARAMS["tb_env_k"])
    parser.add_argument("--max-voxels", type=int, default=12,
                        help="Maximum number of voxels listed per difference (default: 12).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    PARAMS["tb_env_k"] = args.tb_env_k
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    ds = xr.open_dataset(args.netcdf)
    ok = True
    for t in args.time_index:
        frame = ds.isel(time=t)
        time_value = str(frame["time"].values).split("T")[-1].split(".")[0]
        print(f"\n=== timestep {t} ({time_value}) ===")
        old, new = run_old(frame), run_new(frame, time_value)
        same = compare_levels(old, new)
        ok &= same
        if not same:
            diagnose_cores(old, new, args.max_voxels)
    ds.close()

    print("\nResult:", "identical detections." if ok else "differences detected.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
