def convective_rules(params):
    p = params
    grid = {
        "dx_km": p.get("dx_km", 1.0),
        "min_alt_km": p.get("min_alt_km", 0.0),
        "max_alt_km": p.get("max_alt_km", 20.0),
        "vertical_dim": "level",
    }
    wt, ut, vt, tb = (p.get(k, d) for k, d in (("wt_var", "WT"), ("ut_var", "UT"), ("vt_var", "VT"), ("tb_var", "MSG2_IR108")))
    hydro = {"variables": [p.get("mri_var", "MRI"), p.get("mrc_var", "MRC")], "op": ">", "value": p.get("hydro_min_gkg", 0.01)}

    def steering(stat):
        return [{"name": "u_steering", "variable": ut, "stat": stat}, {"name": "v_steering", "variable": vt, "stat": stat}]

    updraft_stats = steering("layer_mean") + [
        {"name": "wt_max_ms", "variable": wt, "stat": "max"},
        {"name": "wt_99pct_ms", "variable": wt, "stat": "percentile", "q": 99},
    ]

    return {
        "mcs": {
            **grid,
            "conditions": [{"variables": tb, "op": "<=", "value": p.get("tb_env_k", 240.0)}],
            "connectivity": 8,
            "min_area_km2": p.get("mcs_min_area_km2", 1000.0),
            "stats": steering("mean") + [
                {"name": "tb_min_k", "variable": tb, "stat": "min"},
                {"name": "tb_mean_k", "variable": tb, "stat": "mean"},
            ],
        },
        "mcs_body": {
            **grid,
            "conditions": [hydro],
            "connectivity": 26,
            "min_voxels": p.get("mcs_min_voxels", 5),
        },
        "convective_object": {
            **grid,
            "conditions": [{"variables": wt, "op": ">", "value": p.get("w_thresh", 1.0)}, hydro],
            "connectivity": 26,
            "min_voxels": p.get("uc_min_voxels", 5),
            "min_width_px": p.get("uc_min_width_px", 5),
            "min_area_px": p.get("uc_min_area_px", 15),
            "stats": updraft_stats,
        },
        "core": {
            **grid,
            "conditions": [{"variables": wt, "op": ">", "value": p.get("core_w_thresh", 4.0)}],
            "split": wt,
            "connectivity": 26,
            "min_voxels": p.get("core_min_voxels", 5),
            "min_width_px": p.get("core_min_width_px", 3),
            "min_area_px": p.get("core_min_area_px", 5),
            "stats": updraft_stats,
        },
    }
