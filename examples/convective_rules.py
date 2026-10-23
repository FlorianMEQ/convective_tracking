def convective_rules(params):
    p = params
    grid = {
        "dx_km": p.get("dx_km", 1.0),
        "min_alt_km": p.get("min_alt_km", 0.0),
        "max_alt_km": p.get("max_alt_km", 20.0),
        "vertical_dim": "level",
    }

    return {
        "mcs": {
            **grid,
            "conditions": [
                {
                    "variables": [p.get("mri_var", "MRI"), p.get("mrc_var", "MRC")],
                    "op": ">",
                    "value": p.get("hydro_min_gkg", 0.01),
                },
                {
                    "variables": p.get("tb_var", "MSG2_IR108"),
                    "op": "<=",
                    "value": p.get("tb_env_k", 240.0),
                },
            ],
            "connectivity": 26, 
            "min_voxels": p.get("mcs_min_voxels", 5),
            "min_area_px": p.get("mcs_min_area_px", 5),
            "min_width_px": p.get("mcs_min_width_px", 3),
            "min_area_km2": params.get("mcs_min_area_km2", 1000.0),
            "take_largest": True,
            "use_2d_for_mcs": True
        },
        "convective_object": {
            **grid,
            "conditions": [
                # WT > seuil ET hydro > seuil
                {"variables": p.get("wt_var", "WT"), "op": ">", "value": p.get("w_thresh", 1.0)},
                {
                    "variables": [p.get("mri_var", "MRI"), p.get("mrc_var", "MRC")],
                    "op": ">",
                    "value": p.get("hydro_min_gkg", 0.01),
                },
            ],
            "connectivity": 26,
            "min_voxels": p.get("uc_min_voxels", 5),
            "min_width_px": p.get("uc_min_width_px", 5),
            "min_area_px": p.get("uc_min_area_px", 15),
        },
        "core": {
            **grid,
            "conditions": [
                {"variables": p.get("wt_var", "WT"), "op": ">", "value": p.get("core_w_thresh", 4.0)},
            ],
            "split": p.get("wt_var", "WT"),  # Variable pour le watershed
            "connectivity": 18,
            "min_voxels": p.get("core_min_voxels", 5),
            "min_width_px": p.get("core_min_width_px", 3),
            "min_area_px": p.get("core_min_area_px", 5),
        },
    }