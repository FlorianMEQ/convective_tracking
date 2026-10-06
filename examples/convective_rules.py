"""Detection rules of the MCS > convective object > convective core hierarchy.

This is where the convective meaning lives: ``src.tracking`` only provides the
generic detector (``detect_objects``) and each level is configured by a rule.
"""
from __future__ import annotations

from typing import Any, Dict


def convective_rules(params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build one detection rule per level ("mcs", "convective_object", "core").

    ``params`` holds the physical thresholds and variable names (``wt_var``,
    ``tb_env_k``, ``w_thresh``, ``core_w_thresh``, ...).
    """
    p = params
    grid = {key: p[key] for key in ("dx_km", "min_alt_km", "max_alt_km") if key in p}
    connectivity_3d = {"connectivity": p["connectivity"]} if "connectivity" in p else {}
    winds = [
        {"name": "u_steering", "variable": p.get("ut_var"), "stat": "layer_mean"},
        {"name": "v_steering", "variable": p.get("vt_var"), "stat": "layer_mean"},
    ]
    hydro_vars = [p[key] for key in ("mri_var", "mrc_var", "mrs_var", "mrg_var", "mrr_var") if p.get(key)]
    speed = [
        {"name": "max_speed", "variable": p["wt_var"], "stat": "max"},
        {"name": "speed_99pct", "variable": p["wt_var"], "stat": "percentile", "q": 99},
    ]

    split_keys = {  # parameter name -> key of the recursive_h_maxima split
        "h_adapt_slope": "h_slope",
        "h_adapt_min": "h_min",
        "h_adapt_max": "h_max",
        "min_width_threshold_km": "min_width_km",
        "segmentation_connectivity": "connectivity",
        "stacked_seeds_xy_radius": "stacked_seeds_xy_radius",
    }
    core_split = {new: p[old] for old, new in split_keys.items() if old in p}

    # Seeds of an MCS: coldest Tb. Priority: minima depth > percentile > fixed threshold.
    if p.get("tb_seed_h_k") is not None:
        seed = {"seed_h": p["tb_seed_h_k"], "outside_value": p["tb_env_k"] + 1.0}
    elif p.get("tb_seed_pct") is not None:
        seed = {"seed_percentile": p["tb_seed_pct"]}
    else:
        seed = {"seed_threshold": p.get("tb_seed_k", 190.0)}

    def stats(*entries):
        return [entry for entry in entries if entry["variable"]]

    return {
        "mcs": {
            **grid,
            "conditions": [{"variables": p["tb_var"], "op": "<=", "value": p["tb_env_k"]}],
            "split": {"method": "seeded_watershed", "variable": p["tb_var"], **seed},
            "min_area_km2": p.get("mcs_min_area_km2", 100.0),
            "stats": stats(
                {"name": "tb_min_k", "variable": p["tb_var"], "stat": "min"},
                {"name": "tb_mean_k", "variable": p["tb_var"], "stat": "mean"},
                {"name": "u_steering", "variable": p.get("ut_var"), "stat": "mean"},
                {"name": "v_steering", "variable": p.get("vt_var"), "stat": "mean"},
            ),
        },
        "convective_object": {
            **grid,
            **connectivity_3d,
            "conditions": [
                {"variables": p["wt_var"], "value": p["w_thresh"]},
                {"variables": hydro_vars, "value": p.get("hydro_min_gkg", 0.01)},
            ],
            "min_voxels": 5,
            "min_width_px": 5,
            "min_area_px": 15,
            "stats": stats(*speed, *winds),
        },
        "core": {
            **grid,
            **connectivity_3d,
            "conditions": [{"variables": p["wt_var"], "value": p["core_w_thresh"]}],
            "split": {"method": "recursive_h_maxima", "variable": p["wt_var"], **core_split},
            "min_voxels": 5,
            "min_width_px": 3,
            "min_area_px": 5,
            "stats": stats(
                *speed,
                {"name": "mri_mean", "variable": p.get("mri_var"), "stat": "mean"},
                {"name": "mrg_mean", "variable": p.get("mrg_var"), "stat": "mean"},
                *winds,
            ),
        },
    }
