"""
Hierarchical 3D detection and tracking of convective systems (MesoNH).
=====================================================================

Three nested levels detected on a single xarray time step, then linked
through time by three independent tracking functions:

  Level A - MCS
    Detection  : Tb <= 240 K thresholding + 2D connected components (adapted
                 MOAAP approach) + 3D spatial connected extension on the
                 total hydrometeor field.
    Qualification: thresholds applied on the full track in track_mcs
                 (peak_area >= 10,000 km2, tb_min <= 225 K, pr_max >= 10 mm/h),
                 which preserves the genesis and decay phases.
    Tracking   : 2D IoU, patience = 4 time steps.

  Level B - Convective Objects (UC)
    Detection  : 3D connected components on a WT >= W_MIN_THRESHOLD_B AND
                 hydro > hydro_min_gkg mask.
    Tracking   : advected 3D IoU + morphological similarity (Hungarian),
                 patience = 2 time steps.

  Level C - Convective Cores (CC)
    Detection  : recursive 3D watershed with adaptive H-Maxima seeds
                 (W_MIN_THRESHOLD_C threshold) inside each UC.
    Tracking   : advected 3D IoU + morphological EMA + splits/merges
                 (Hungarian), patience = 1 time step.

Public API
----------
    find_convective_cores_3d(ds, ...)      -> per-time-step results
    track_mcs(results_by_time, ...) -> dict of Level A tracks
    track_uc(results_by_time, ...)  -> dict of Level B (UC) tracks
    track_cc(results_by_time, ...)  -> dict of Level C (CC) tracks
    segment_core_recursive(core_data, ...) -> standalone Level C segmentation

Minimal usage
-------------
    import xarray as xr
    from lib.fct_core_tracking import (
        find_convective_cores_3d,
        track_mcs, track_uc, track_cc,
    )

    results = []
    for ds_t in xr.open_dataset("simulation.nc").groupby("time"):
        results.extend(find_convective_cores_3d(ds_t))

    mcs_tracks = track_mcs(results)
    uc_tracks  = track_uc(results)
    cc_tracks  = track_cc(results)

Dependencies
------------
    numpy, pandas, scipy (ndimage, optimize), scikit-image (morphology, segmentation)
"""

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from scipy.ndimage import label as cc_label
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

# ================================================================
# GLOBAL CONSTANTS
# ================================================================

# --- Horizontal resolution (same in x and y) ---
DX_METERS = 1000.0          # m/pixel
DX_KM     = DX_METERS / 1000.0  # km/pixel

# --- Vertical velocity thresholds ---
W_MIN_THRESHOLD_B = 1.0   # Level B: binary WT mask (m/s)
W_MIN_THRESHOLD_C = 4.0   # Level C: watershed + adaptive h-maxima (m/s)

# --- Adaptive H-Maxima (Level C) ---
# h = SLOPE * (WT_max - W_MIN_THRESHOLD_C), clamped to [H_ADAPT_MIN, H_ADAPT_MAX]
H_ADAPT_SLOPE = 0.3       # slope of the h(WT_max) relation
H_ADAPT_MIN   = 1.0       # floor of h (m/s)
H_ADAPT_MAX   = 10.0      # ceiling of h (m/s)

# --- 3D connectivity ---
CONNECTIVITY_3D = 18      # 6=face, 18=face+edge, 26=face+edge+corner

# --- Merging of vertically stacked seeds (Level C) ---
MERGE_VERTICAL_STACKED_SEEDS = True
STACKED_SEEDS_XY_RADIUS      = 3   # 2D dilation radius used to detect stacking (pixels)

# --- MCS vertical extension: "connected to cloud top" tolerance (Level A) ---
# A per-column top_z check alone lets a shallow, spatially isolated patch
# through: if a column's only hydro content is a low, disconnected pocket, that
# pocket IS "the top of its own column" and would otherwise qualify. Requiring
# a label's own highest point to be within this tolerance of the tallest point
# reached anywhere in the footprint filters those out, while still tolerating
# the few-hundred-meter dips expected across a genuinely anvil-connected top.
MCS_TOP_CONNECT_TOL_M = 2000.0


# ================================================================
# INTERNAL UTILITIES
# ================================================================

def _z_indices_between(ds, zmin_m, zmax_m):
    """Return the vertical level indices between zmin_m and zmax_m."""
    z_vals = ds.coords["level"].values.astype(float)
    mask = (z_vals >= zmin_m) & (z_vals <= zmax_m)
    return np.nonzero(mask)[0]


def _compute_component_metrics(mask3d, level_vals=None):
    """
    Geometric metrics of a 3D component (Level B or C).

    Parameters
    ----------
    mask3d : ndarray bool, shape (nz, ny, nx)
        Local mask of the component.
    level_vals : ndarray float, optional
        Level altitudes in meters. If provided, height and volume are
        computed in physical units (km, km3). Otherwise dz = 1 km is assumed.

    Returns
    -------
    dict
        Widths, areas, volume and height in pixels (without level_vals) or
        in km / km2 / km3 (with level_vals). Horizontal resolution: DX_KM.
    """
    voxels = int(mask3d.sum())
    proj   = mask3d.any(axis=0)
    area   = int(proj.sum())

    if area == 0:
        return {
            "width": 0, "area": 0, "voxels": 0,
            "width_x": 0, "width_y": 0, "height": 0,
            "width_km": 0.0, "width_x_km": 0.0, "width_y_km": 0.0, "height_km": 0.0,
            "area_km2": 0.0, "volume_km3": 0.0,
            "width_max_dz": 0,
            "area_max_dz_km2": 0.0, "area_mean_km2": 0.0, "area_median_km2": 0.0,
            "width_mean_km": 0.0, "width_median_km": 0.0,
        }

    ys, xs  = np.where(proj)
    width_x = int(xs.max() - xs.min() + 1)
    width_y = int(ys.max() - ys.min() + 1)
    width   = max(width_x, width_y)

    z_proj    = mask3d.any(axis=(1, 2))
    z_indices = np.where(z_proj)[0]
    height    = int(z_indices.max() - z_indices.min() + 1) if len(z_indices) > 0 else 0

    count_z = mask3d.sum(axis=(1, 2))
    active  = count_z > 0
    areas_arr = count_z[active]

    nz, ny, nx = mask3d.shape
    x_proj = mask3d.any(axis=1)
    y_proj = mask3d.any(axis=2)
    x_min_k  = x_proj.argmax(axis=1)
    x_max_k  = nx - 1 - x_proj[:, ::-1].argmax(axis=1)
    y_min_k  = y_proj.argmax(axis=1)
    y_max_k  = ny - 1 - y_proj[:, ::-1].argmax(axis=1)
    widths_arr = np.maximum(x_max_k - x_min_k + 1, y_max_k - y_min_k + 1)[active]

    if areas_arr.size > 0:
        area_max_dz  = int(areas_arr.max())
        area_mean    = float(areas_arr.mean())
        area_median  = float(np.median(areas_arr))
        width_max_dz = int(widths_arr.max())
        width_mean   = float(widths_arr.mean())
        width_median = float(np.median(widths_arr))
    else:
        area_max_dz = width_max_dz = 0
        area_mean = area_median = width_mean = width_median = 0.0

    width_km   = float(width)   * DX_KM
    width_x_km = float(width_x) * DX_KM
    width_y_km = float(width_y) * DX_KM
    area_km2   = float(area)    * DX_KM ** 2
    volume_km3 = float(voxels)  * DX_KM ** 2   # dz assumed 1 km if level_vals is absent
    height_km  = float(height)

    area_max_dz_km2 = float(area_max_dz)  * DX_KM ** 2
    area_mean_km2   = float(area_mean)    * DX_KM ** 2
    area_median_km2 = float(area_median)  * DX_KM ** 2
    width_mean_km   = float(width_mean)   * DX_KM
    width_median_km = float(width_median) * DX_KM

    if level_vals is not None and len(level_vals) > 1:
        dz_m  = np.diff(level_vals)
        dz_km = np.append(dz_m, dz_m[-1]) / 1000.0
        if len(z_indices) > 0:
            height_km = float(dz_km[z_indices].sum())
        volume_km3 = float((count_z * dz_km).sum()) * DX_KM ** 2

    return {
        "width": width, "area": area, "voxels": voxels,
        "width_x": width_x, "width_y": width_y, "height": height,
        "width_km": width_km, "width_x_km": width_x_km, "width_y_km": width_y_km,
        "height_km": height_km,
        "area_km2": area_km2, "volume_km3": volume_km3,
        "width_max_dz": width_max_dz,
        "area_max_dz_km2": area_max_dz_km2, "area_mean_km2": area_mean_km2,
        "area_median_km2": area_median_km2,
        "width_mean_km": width_mean_km, "width_median_km": width_median_km,
    }


def _get_structure_3d(connectivity=CONNECTIVITY_3D):
    """3D structuring element with 6, 18 or 26 connectivity."""
    if connectivity not in (6, 18, 26):
        raise ValueError(f"Invalid 3D connectivity: {connectivity}. Use 6, 18 or 26.")
    s = np.zeros((3, 3, 3), dtype=bool)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    s[1, 1, 1] = True
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and manhattan == 1:
                    s[dz + 1, dy + 1, dx + 1] = True
                elif connectivity == 18 and not (abs(dx) == 1 and abs(dy) == 1 and abs(dz) == 1):
                    s[dz + 1, dy + 1, dx + 1] = True
                elif connectivity == 26:
                    s[dz + 1, dy + 1, dx + 1] = True
    return s


# Precomputed structuring elements - these were previously rebuilt on every
# call (a 27-iteration Python loop for _get_structure_3d, or a fresh np.ones
# allocation) even though CONNECTIVITY_3D and the 2D/26-connectivity cases
# used here never change at runtime. 26-connectivity is a full 3x3x3 True
# array by construction, identical to np.ones((3, 3, 3), dtype=bool).
_STRUCT_2D_8CONN        = np.ones((3, 3), dtype=bool)
_STRUCT_3D_26CONN       = np.ones((3, 3, 3), dtype=bool)
_STRUCTURE_3D_RECURSIVE = _get_structure_3d(CONNECTIVITY_3D)


def _merge_stacked_markers(markers, grid_3d, xy_radius=STACKED_SEEDS_XY_RADIUS):
    """
    Merge vertically stacked H-Maxima seeds.

    Two seeds are considered stacked if their 2D (XY) projections overlap
    after a disk dilation of radius ``xy_radius`` pixels. The most intense
    seed (WT_max) absorbs the other.

    Parameters
    ----------
    markers : ndarray int
        Seed labels (output of cc_label on h_maxima).
    grid_3d : ndarray float
        Local WT field (same shape as markers).
    xy_radius : int
        Dilation disk radius in pixels.

    Returns
    -------
    merged : ndarray int
        Markers after merging.
    n_after : int
        Number of seeds after merging.
    n_merged : int
        Number of merges performed.
    """
    labels = np.unique(markers)
    labels = labels[labels > 0]
    if len(labels) <= 1:
        return markers, len(labels), 0

    r = int(max(0, xy_radius))
    cy_d, cx_d = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (cy_d ** 2 + cx_d ** 2) <= r * r

    label_info    = []
    proj_by_label = {}

    for lbl in labels:
        lbl  = int(lbl)
        mask = (markers == lbl)
        proj = np.any(mask, axis=0)
        if not np.any(proj):
            continue
        max_wt = np.max(grid_3d[mask])
        dil    = binary_dilation(proj, structure=disk)
        proj_by_label[lbl] = {"proj": proj, "dil": dil}
        label_info.append({"label": lbl, "wt_max": max_wt})

    label_info.sort(key=lambda x: x["wt_max"], reverse=True)
    parent = {info["label"]: info["label"] for info in label_info}

    for i in range(len(label_info)):
        li = label_info[i]["label"]
        if parent[li] != li:
            continue
        dil_i = proj_by_label[li]["dil"]
        for j in range(i + 1, len(label_info)):
            lj = label_info[j]["label"]
            if parent[lj] != lj:
                continue
            if np.any(dil_i & proj_by_label[lj]["proj"]):
                parent[lj] = li

    roots      = sorted(list(set(parent.values())))
    root_to_new = {r: i + 1 for i, r in enumerate(roots)}
    old_to_new  = {lbl: root_to_new[parent[lbl]] for lbl in labels}

    merged = np.zeros_like(markers)
    for old_lbl, new_lbl in old_to_new.items():
        merged[markers == old_lbl] = new_lbl

    n_before = len(labels)
    n_after  = len(np.unique(merged[merged > 0]))
    return merged, n_after, max(0, n_before - n_after)


def _detect_level_a(tb_2d, inprr_2d, hydro_3d, time_hms,
                    tb_env_k, hydro_min_gkg, mcs_min_area_km2,
                    wt_3d=None, level_vals=None, ut_3d=None, vt_3d=None):
    """
    Detection of candidate convective shields (Level A) - adapted MOAAP approach.

    Steps:
      1. Tb <= tb_env_k thresholding -> binary cold shield mask.
      2. 2D connected components (8-connectivity) -> candidate objects.
      3. Noise filter by minimum area (mcs_min_area_km2).
      4. Per-object metrics: tb_min_k, pr_max_mmh (if inprr_2d is provided).
      5. Connected 3D spatial extension on the total hydro field (unchanged).
      6. Geometric and physical metrics on this 3D structure via
         ``_compute_component_metrics`` (volume, per-z-level areas/widths,
         WT statistics), symmetric with what is done at Levels B/C.

    MCS qualification (peak area thresholds, Tb overshoot, precipitation) is
    performed a posteriori in track_mcs on the full track, which preserves
    the genesis and decay phases.

    Parameters
    ----------
    tb_2d : ndarray float, shape (ny, nx)
        Synthetic brightness temperature (K).
    inprr_2d : ndarray float or None, shape (ny, nx)
        Instantaneous surface precipitation rate (mm/h).
        None if the variable is not available in the dataset.
    hydro_3d : ndarray float, shape (nz, ny, nx)
        Total hydrometeor field (g/kg).
    time_hms : str
        Time step in HH:MM:SS format.
    tb_env_k : float
        Cold shield threshold (K). Recommended default: 240 K (adapted from MOAAP's 241 K).
    hydro_min_gkg : float
        Minimum hydrometeor threshold for the 3D extension (g/kg).
    mcs_min_area_km2 : float
        Minimum noise-filter area (km2). Filters out small objects before
        tracking; must stay well below track_mcs's MCS qualification threshold.
    wt_3d : ndarray float or None, shape (nz, ny, nx)
        Total vertical velocity (m/s). If provided, WT statistics
        (max/mean/99pct) are computed on the connected 3D structure (the raw
        per-voxel array is not kept in the output dict - too large at MCS
        scale). None disables these fields.
    level_vals : ndarray float or None
        W-level altitudes (m). If provided, ``volume_km3``/``height_km`` are
        weighted by the actual ``dz`` instead of the dz=1 km assumption.
    ut_3d, vt_3d : ndarray float or None, shape (nz, ny, nx)
        Wind components (m/s). If provided, averaged over the connected 3D
        structure (``ut_ms``/``vt_ms``, distinct from the steering wind).

    Returns
    -------
    mcs_list : list of dict
        One dict per detected candidate object.
    obj_labels : ndarray int, shape (ny, nx)
        2D label map (used to associate Level B objects with a parent A).
    """
    env_mask = tb_2d <= tb_env_k

    obj_labels, n_objs = cc_label(env_mask, structure=_STRUCT_2D_8CONN)

    if n_objs == 0:
        return [], np.zeros_like(tb_2d, dtype=np.int32)

    mcs_list = []
    for lab in range(1, n_objs + 1):
        cc_mask_2d = obj_labels == lab
        area_km2   = float(cc_mask_2d.sum())
        if area_km2 < mcs_min_area_km2:
            continue
        ys, xs = np.where(cc_mask_2d)
        nz_loc = hydro_3d.shape[0]

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        loc_ys = ys - y_min
        loc_xs = xs - x_min
        local_vol = np.zeros((nz_loc, y_max - y_min + 1, x_max - x_min + 1), dtype=bool)
        local_vol[:, loc_ys, loc_xs] = hydro_3d[:, ys, xs] > hydro_min_gkg

        # 26-connectivity: needed to link voxels through diagonal neighbors
        # in columns that have empty intermediate levels.
        labeled_local, _ = cc_label(local_vol, structure=_STRUCT_3D_26CONN)

        col_bool  = local_vol[:, loc_ys, loc_xs]
        has_any   = col_bool.any(axis=0)
        top_z     = np.where(has_any,
                             nz_loc - 1 - col_bool[::-1].argmax(axis=0),
                             -1)
        valid      = top_z >= 0
        top_labels = (set(labeled_local[top_z[valid], loc_ys[valid], loc_xs[valid]].tolist())
                     - {0})

        # A label reaching "the top of its own column" is not enough: a
        # column whose only content is a shallow, spatially isolated pocket
        # (no connectivity path to the tall structure at all) trivially
        # satisfies that check too. Keep only labels whose own highest point
        # is within MCS_TOP_CONNECT_TOL_M of the tallest point reached
        # anywhere in this footprint - i.e. genuinely connected to the real
        # cloud top, not just locally topping out low.
        if top_labels and valid.any():
            global_top_z = int(top_z[valid].max())
            if level_vals is not None:
                above_tol = np.nonzero(
                    level_vals[:global_top_z + 1] >= level_vals[global_top_z] - MCS_TOP_CONNECT_TOL_M
                )[0]
                min_top_z = int(above_tol.min()) if above_tol.size else global_top_z
            else:
                min_top_z = max(0, global_top_z - 3)
            label_reaches_top = set()
            for lbl in top_labels:
                lbl_top_z = np.nonzero(np.any(labeled_local == lbl, axis=(1, 2)))[0].max()
                if lbl_top_z >= min_top_z:
                    label_reaches_top.add(lbl)
            top_labels = label_reaches_top

        if top_labels:
            conn_local     = np.isin(labeled_local, list(top_labels))
            connected_mask = conn_local[:, loc_ys, loc_xs]
        else:
            connected_mask = np.zeros((nz_loc, len(ys)), dtype=bool)

        # Dense local mask (nz_loc, ny_bbox, nx_bbox) - rebuilt from the
        # "columns" representation above so _compute_component_metrics can be
        # reused just like at Levels B/C (volume, per-z-level areas and
        # widths, symmetric with what is done at B/C).
        dense_mask = np.zeros((nz_loc, y_max - y_min + 1, x_max - x_min + 1), dtype=bool)
        dense_mask[:, loc_ys, loc_xs] = connected_mask
        metrics_a = _compute_component_metrics(dense_mask, level_vals)

        z3, y3_loc, x3_loc = np.where(dense_mask)
        y3, x3   = y3_loc + y_min, x3_loc + x_min
        z_active = np.where(dense_mask.any(axis=(1, 2)))[0]

        # WT statistics computed on the fly over the voxels (z3,y3,x3) - the
        # raw per-voxel array and the 3D indices are not kept in the output
        # dict: at Level A (MCS footprints, often thousands of voxels), that
        # would make every history snapshot excessively heavy.
        if wt_3d is not None and z3.size > 0:
            wt_vals_a = wt_3d[z3, y3, x3]
            wt_max_a  = float(np.nanmax(wt_vals_a))
            wt_mean_a = float(np.nanmean(wt_vals_a))
            wt_99_a   = float(np.nanpercentile(wt_vals_a, 99))
        else:
            wt_max_a = wt_mean_a = wt_99_a = None

        ut_mean_a = float(np.nanmean(ut_3d[z3, y3, x3])) if ut_3d is not None and z3.size > 0 else None
        vt_mean_a = float(np.nanmean(vt_3d[z3, y3, x3])) if vt_3d is not None and z3.size > 0 else None

        if level_vals is not None and z_active.size > 0:
            height_min_km_a = float(level_vals[z_active.min()]) / 1000.0
            height_max_km_a = float(level_vals[z_active.max()]) / 1000.0
        else:
            height_min_km_a = height_max_km_a = None

        n_vox_3d = int(dense_mask.sum())
        tb_cc     = tb_2d[cc_mask_2d]
        pr_max    = (float(inprr_2d[cc_mask_2d].max())
                     if inprr_2d is not None else None)
        mcs_list.append({
            "mcs_id":      f"t{time_hms}_mcs{lab}",
            "time_value":  time_hms,
            "area_2d_km2": area_km2,
            "n_voxels_3d": n_vox_3d,
            "volume_km3":       metrics_a["volume_km3"],
            "area_max_dz_km2":  metrics_a["area_max_dz_km2"],
            "area_mean_km2":    metrics_a["area_mean_km2"],
            "area_median_km2":  metrics_a["area_median_km2"],
            "width_max_km":     float(metrics_a["width_max_dz"]),
            "width_mean_km":    metrics_a["width_mean_km"],
            "width_median_km":  metrics_a["width_median_km"],
            "height_km":        metrics_a["height_km"],
            "height_min_km":    height_min_km_a,
            "height_max_km":    height_max_km_a,
            "wt_max_ms":        wt_max_a,
            "wt_mean_ms":       wt_mean_a,
            "wt_99pct_ms":      wt_99_a,
            "ut_ms":            ut_mean_a,
            "vt_ms":            vt_mean_a,
            "tb_min_k":    float(tb_cc.min()),
            "tb_mean_k":   float(tb_cc.mean()),
            "pr_max_mmh":  pr_max,
            "z_top_idx":   int(z_active.max()) if z_active.size > 0 else 0,
            "z_base_idx":  int(z_active.min()) if z_active.size > 0 else 0,
            "centroid_x":  float(xs.mean()),
            "centroid_y":  float(ys.mean()),
            "indices_2d":  list(zip(ys.tolist(), xs.tolist())),
            "_label":      int(lab),
        })

    return mcs_list, obj_labels


# ================================================================
# MAIN DETECTION (public entry point)
# ================================================================

def find_convective_cores_3d(
    ds,
    wt_var="WT",
    ut_var="UT",
    vt_var="VT",
    mri_var="MRI",
    mrc_var="MRC",
    mrv_var="MRV",
    mrs_var="MRS",
    mrg_var="MRG",
    mrr_var="MRR",
    temp_var="TEMP",
    pres_var="PRES",
    iwp_var="IWP",
    inprr_var="INPRR",
    wice_var="WICE",
    vim_var="VIM",
    tb_var="MSG2_IR108",
    w_thresh=W_MIN_THRESHOLD_B,
    min_alt_km=0,
    max_alt_km=20,
    tb_env_k=240.0,
    hydro_min_gkg=0.01,
    mcs_min_area_km2=1000.0,
):
    """
    Hierarchical 3D detection for a single time step.

    Chains the three detection levels (A -> B -> C) on the xarray dataset
    ``ds`` selected for one time step. Results are returned as a
    single-element list, a format compatible with the tracking functions
    (``track_mcs``, ``track_uc``, ``track_cc``).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset for one time step containing the atmospheric variables.
        Must expose a ``"level"`` coordinate (altitudes in meters) and a
        ``"level"`` coordinate (W-level altitudes in meters).
    wt_var : str
        Total vertical velocity (m/s). Required.
    ut_var, vt_var : str
        Zonal and meridional wind components (m/s). Optional (steering wind
        set to 0 if absent).
    mri_var, mrc_var : str
        Primary ice and cloud liquid water mixing ratio (g/kg).
        Required (used for hydro_3d).
    mrv_var, mrs_var, mrg_var, mrr_var : str
        Vapor, snow, graupel, rain (g/kg). Optional.
    temp_var, pres_var : str
        Temperature (K) and pressure (Pa). Optional.
    iwp_var, inprr_var, wice_var, vim_var : str
        Integrated 2D variables (IWP kg/m2, precip mm/h, etc.). Optional.
    tb_var : str
        Synthetic brightness temperature (K). If absent from the dataset,
        Level A detection is disabled.
    w_thresh : float
        WT threshold for the Level B binary mask (m/s). Default: W_MIN_THRESHOLD_B.
    min_alt_km, max_alt_km : float
        Lower and upper bound of the analyzed column (km).
    tb_env_k : float
        Level A cold shield threshold (K). Default: 240 K (adapted from MOAAP's 241 K).
    hydro_min_gkg : float
        Minimum hydrometeor content threshold (g/kg). Default: 0.01 g/kg.
    mcs_min_area_km2 : float
        Level A noise-filter minimum area (km2). Default: 1000 km2. Must stay
        well below track_mcs's MCS qualification threshold (10,000 km2).

    Returns
    -------
    list of dict
        Single-element list containing a dict with the keys:
        ``"time"``, ``"timestamp_s"``,
        ``"mcs"`` (list), ``"n_mcs"``,
        ``"uc"`` (list), ``"n_uc"``,
        ``"cc"`` (list), ``"n_cc"``,
        ``"n_raw_cores"``, ``"n_segmented_cores"``,
        ``"n_subgroups_from_watershed"``.
    """
    results_by_time = []

    # 1. Select vertical levels
    z_idx      = _z_indices_between(ds, min_alt_km * 1000.0, max_alt_km * 1000.0)
    level_vals = ds["level"].isel(level=z_idx).values.astype(float)

    # 2. Extract numpy arrays
    wt_np    = ds[wt_var].isel(level=z_idx).values
    ut_np    = ds[ut_var].isel(level=z_idx).values if ut_var    and ut_var    in ds else None
    vt_np    = ds[vt_var].isel(level=z_idx).values if vt_var    and vt_var    in ds else None
    mri_np   = ds[mri_var].isel(level=z_idx).values
    mrc_np   = ds[mrc_var].isel(level=z_idx).values
    mrv_np   = ds[mrv_var].isel(level=z_idx).values if mrv_var  and mrv_var  in ds else None
    mrs_np   = ds[mrs_var].isel(level=z_idx).values if mrs_var  and mrs_var  in ds else None
    mrg_np   = ds[mrg_var].isel(level=z_idx).values if mrg_var  and mrg_var  in ds else None
    mrr_np   = ds[mrr_var].isel(level=z_idx).values if mrr_var  and mrr_var  in ds else None
    temp_np  = ds[temp_var].isel(level=z_idx).values if temp_var and temp_var in ds else None
    pres_np  = ds[pres_var].isel(level=z_idx).values if pres_var and pres_var in ds else None
    iwp_np   = ds[iwp_var].values   if iwp_var   and iwp_var   in ds else None
    inprr_np = ds[inprr_var].values if inprr_var and inprr_var in ds else None
    wice_np  = ds[wice_var].values  if wice_var  and wice_var  in ds else None
    vim_np   = ds[vim_var].values   if vim_var   and vim_var   in ds else None
    tb_np = ds[tb_var].values if tb_var and tb_var in ds else None

    # 3. Total hydro field (UC mask + 3D MCS extension)
    hydro_3d = mri_np + mrc_np

    # Binary UC mask (Level B): WT > threshold AND hydrometeors present
    mask_np = (wt_np > w_thresh) & (hydro_3d > hydro_min_gkg)

    raw_time    = ds["time"].values
    time_hms    = str(raw_time).split("T")[-1].split(".")[0]
    timestamp_s = float(pd.Timestamp(raw_time).timestamp())

    # 3b. Level A (MCS) detection
    mcs_list   = []
    env_labels = None
    _label_map = {}
    if tb_np is not None:
        mcs_list, env_labels = _detect_level_a(
            tb_np, inprr_np, hydro_3d, time_hms,
            tb_env_k, hydro_min_gkg, mcs_min_area_km2,
            wt_3d=wt_np, level_vals=level_vals, ut_3d=ut_np, vt_3d=vt_np,
        )
        _label_map = {m.pop("_label"): m["mcs_id"] for m in mcs_list}

        # Steering wind of each MCS: column average over the 2D footprint
        for m in mcs_list:
            yx = m.get("indices_2d", [])
            if yx and ut_np is not None and vt_np is not None:
                ys_m = np.array([p[0] for p in yx])
                xs_m = np.array([p[1] for p in yx])
                m["ut_steering"] = float(ut_np[:, ys_m, xs_m].mean())
                m["vt_steering"] = float(vt_np[:, ys_m, xs_m].mean())
            else:
                m["ut_steering"] = 0.0
                m["vt_steering"] = 0.0

    if not mask_np.any():
        results_by_time.append({
            "time": time_hms, "timestamp_s": timestamp_s,
            "mcs": mcs_list, "uc": [], "n_cc": 0, "cc": [],
        })
        return results_by_time

    # 4. 3D connected components (UC - Level B)
    labels, nlabels = cc_label(mask_np, structure=_STRUCT_3D_26CONN)
    if nlabels == 0:
        results_by_time.append({
            "time": time_hms, "timestamp_s": timestamp_s,
            "mcs": mcs_list, "uc": [], "n_cc": 0, "cc": [],
        })
        return results_by_time

    # 5. Bounding boxes
    slices = find_objects(labels)
    cc_list        = []
    uc_list = []
    nb_raw_cores = nb_segmented_cores = nb_subgroups_from_watershed = 0

    # 6. Loop over each UC (Level B)
    for lab, sl in enumerate(slices, 1):
        if sl is None:
            continue
        comp = (labels[sl] == lab)
        if comp.sum() < 5:
            continue

        z_coords, y_coords, x_coords = np.where(comp)
        z_coords += sl[0].start
        y_coords += sl[1].start
        x_coords += sl[2].start

        zmin_co, zmax_co = int(z_coords.min()), int(z_coords.max())
        level_vals_co    = level_vals[zmin_co : zmax_co + 1] if level_vals is not None else None
        # width/area do not depend on level_vals (see _compute_component_metrics),
        # so a single call serves both the noise filter below and the full metrics.
        metrics_co       = _compute_component_metrics(comp, level_vals_co)
        if metrics_co["width"] < 5 or metrics_co["area"] < 15:
            continue
        unique_2d_co     = list(set(zip(y_coords.tolist(), x_coords.tolist())))
        cx_co = float(np.mean(x_coords))
        cy_co = float(np.mean(y_coords))

        # Parent MCS via centroid
        mcs_id_co = None
        if env_labels is not None:
            cy_i    = int(np.clip(round(cy_co), 0, env_labels.shape[0] - 1))
            cx_i    = int(np.clip(round(cx_co), 0, env_labels.shape[1] - 1))
            lbl_at  = int(env_labels[cy_i, cx_i])
            if lbl_at in _label_map:
                mcs_id_co = _label_map[lbl_at]

        # Steering wind: vertical centroid +/- 2 levels
        z_mean_co = float(np.mean(z_coords))
        zc_co     = int(round(z_mean_co))
        z_lo_co   = max(zmin_co, zc_co - 2)
        z_hi_co   = min(zmax_co, zc_co + 2)
        mask_mid_co = (z_coords >= z_lo_co) & (z_coords <= z_hi_co)
        if ut_np is not None:
            ut_mean_co = float(np.nanmean(ut_np[z_coords, y_coords, x_coords]))
            vt_mean_co = float(np.nanmean(vt_np[z_coords, y_coords, x_coords]))
            if mask_mid_co.any():
                ut_steering_co = float(np.nanmean(
                    ut_np[z_coords[mask_mid_co], y_coords[mask_mid_co], x_coords[mask_mid_co]]))
                vt_steering_co = float(np.nanmean(
                    vt_np[z_coords[mask_mid_co], y_coords[mask_mid_co], x_coords[mask_mid_co]]))
            else:
                ut_steering_co, vt_steering_co = ut_mean_co, vt_mean_co
        else:
            ut_mean_co = vt_mean_co = 0.0
            ut_steering_co = vt_steering_co = 0.0

        mask_base_co      = (z_coords == zmin_co)
        y_base_co, x_base_co = y_coords[mask_base_co], x_coords[mask_base_co]
        width_base_km_co  = (float(max(int(x_base_co.max() - x_base_co.min() + 1),
                                       int(y_base_co.max() - y_base_co.min() + 1)))
                             if y_base_co.size > 0 else 0.0)
        wt_vals_co        = wt_np[z_coords, y_coords, x_coords]

        uc_id  = f"t{time_hms}_uc{lab}"
        indices_3d   = list(zip(z_coords.tolist(), y_coords.tolist(), x_coords.tolist()))
        uc_list.append({
            "uc_id":     uc_id,
            "time_value":      time_hms,
            "mcs_id":          mcs_id_co,
            "area_max_dz_km2":  metrics_co["area_max_dz_km2"],
            "area_mean_km2":    metrics_co["area_mean_km2"],
            "area_median_km2":  metrics_co["area_median_km2"],
            "volume_km3":       metrics_co["volume_km3"],
            "width_max_km":     float(metrics_co["width_max_dz"]),
            "width_base_km":    width_base_km_co,
            "width_mean_km":    metrics_co["width_mean_km"],
            "width_median_km":  metrics_co["width_median_km"],
            "height_km":        metrics_co["height_km"],
            "height_min_km":    float(level_vals[zmin_co]) / 1000.0,
            "height_max_km":    float(level_vals[zmax_co]) / 1000.0,
            "centroid_x":       cx_co,
            "centroid_y":       cy_co,
            "centroid_z":       z_mean_co,
            "wt_max_ms":        float(np.nanmax(wt_vals_co)),
            "wt_mean_ms":       float(np.nanmean(wt_vals_co)),
            "wt_99pct_ms":      float(np.nanpercentile(wt_vals_co, 99)),
            "ut_ms":            ut_mean_co,
            "vt_ms":            vt_mean_co,
            "ut_steering_ms":   ut_steering_co,
            "vt_steering_ms":   vt_steering_co,
            "indices_2d":       unique_2d_co,
            "indices_3d":       indices_3d,
            "n_cc":          0,
        })

        # 7. Pre-filter WT > W_MIN_THRESHOLD_C + CC3D for Level C (CC)
        # Restrict to the WT > 4 m/s sub-volume then re-label the connected
        # components (26-connectivity) so each isolated cluster is processed
        # and filtered independently - same behavior as the former
        # monolithic 4 m/s code.
        wt_vals_b = wt_np[z_coords, y_coords, x_coords]
        mask_c    = wt_vals_b > W_MIN_THRESHOLD_C
        if not mask_c.any():
            uc_list[-1]["n_cc"] = 0
            continue

        zc = z_coords[mask_c]; yc = y_coords[mask_c]; xc = x_coords[mask_c]
        wt_c = wt_vals_b[mask_c]

        # CC3D on the 4 m/s sub-volume (26-connectivity)
        zmin_c, zmax_c = int(zc.min()), int(zc.max())
        ymin_c, ymax_c = int(yc.min()), int(yc.max())
        xmin_c, xmax_c = int(xc.min()), int(xc.max())
        shape_c = (zmax_c - zmin_c + 1, ymax_c - ymin_c + 1, xmax_c - xmin_c + 1)
        vol_c   = np.zeros(shape_c, dtype=bool)
        vol_c[zc - zmin_c, yc - ymin_c, xc - xmin_c] = True
        labels_c, n_clusters_c = cc_label(vol_c, structure=_STRUCT_3D_26CONN)

        subgroups_indices = []
        for clab in range(1, n_clusters_c + 1):
            cl_mask = (labels_c == clab)
            if cl_mask.sum() < 5:
                continue
            metrics_cl = _compute_component_metrics(cl_mask)
            if metrics_cl["width"] < 3 or metrics_cl["area"] < 5:
                continue
            cl_z, cl_y, cl_x = np.where(cl_mask)
            cl_z_g = cl_z + zmin_c; cl_y_g = cl_y + ymin_c; cl_x_g = cl_x + xmin_c
            cl_wt  = wt_np[cl_z_g, cl_y_g, cl_x_g]
            item_cl = {
                "cc_id":     f"{uc_id}_cc{clab}",
                "indices_3d":   list(zip(cl_z_g.tolist(), cl_y_g.tolist(), cl_x_g.tolist())),
                "wt_3d_values": cl_wt.tolist(),
            }
            nb_raw_cores += 1
            sg = segment_core_recursive(item_cl, level_vals=level_vals, min_width_threshold=1)
            if len(sg) > 1:
                nb_segmented_cores          += 1
                nb_subgroups_from_watershed += len(sg)
            subgroups_indices.extend(sg)

        n_cc_this_uc = 0
        for sg_data in subgroups_indices:
            sg_indices = np.array(sg_data["indices_3d"])
            if len(sg_indices) == 0:
                continue
            z_vals = sg_indices[:, 0]
            y_vals = sg_indices[:, 1]
            x_vals = sg_indices[:, 2]
            zmin, zmax = z_vals.min(), z_vals.max()
            ymin, ymax = y_vals.min(), y_vals.max()
            xmin, xmax = x_vals.min(), x_vals.max()
            shape_local = (zmax - zmin + 1, ymax - ymin + 1, xmax - xmin + 1)
            mask_local  = np.zeros(shape_local, dtype=bool)
            mask_local[z_vals - zmin, y_vals - ymin, x_vals - xmin] = True
            level_vals_local = level_vals[zmin:zmax + 1] if level_vals is not None else None
            metrics = _compute_component_metrics(mask_local, level_vals_local)
            if metrics["width"] < 3 or metrics["area"] < 5 or metrics["voxels"] < 5:
                continue

            wt_vals      = wt_np[z_vals, y_vals, x_vals]
            vmax         = float(np.nanmax(wt_vals))
            vmean        = float(np.nanmean(wt_vals))
            v99          = float(np.nanpercentile(wt_vals, 99)) if wt_vals.size > 0 else 0.0
            mri_mean_val = float(np.nanmean(mri_np[z_vals, y_vals, x_vals])) if mri_np is not None else 0.0
            mrc_mean_val = float(np.nanmean(mrc_np[z_vals, y_vals, x_vals])) if mrc_np is not None else 0.0
            mrv_mean_val = float(np.nanmean(mrv_np[z_vals, y_vals, x_vals])) if mrv_np is not None else 0.0
            mrs_mean_val = float(np.nanmean(mrs_np[z_vals, y_vals, x_vals])) if mrs_np is not None else 0.0
            mrg_mean_val = float(np.nanmean(mrg_np[z_vals, y_vals, x_vals])) if mrg_np is not None else 0.0
            mrr_mean_val = float(np.nanmean(mrr_np[z_vals, y_vals, x_vals])) if mrr_np is not None else 0.0
            ut_mean_val  = float(np.nanmean(ut_np[z_vals, y_vals, x_vals]))  if ut_np  is not None else 0.0
            vt_mean_val  = float(np.nanmean(vt_np[z_vals, y_vals, x_vals]))  if vt_np  is not None else 0.0
            x_mean = float(np.mean(x_vals))
            y_mean = float(np.mean(y_vals))
            z_mean = float(np.mean(z_vals))

            # Steering wind: vertical centroid +/- 2 levels
            zc     = int(round(z_mean))
            z_lo   = max(int(zmin), zc - 2)
            z_hi   = min(int(zmax), zc + 2)
            mask_mid = (z_vals >= z_lo) & (z_vals <= z_hi)
            if mask_mid.any() and ut_np is not None:
                ut_steering = float(np.nanmean(
                    ut_np[z_vals[mask_mid], y_vals[mask_mid], x_vals[mask_mid]]))
                vt_steering = float(np.nanmean(
                    vt_np[z_vals[mask_mid], y_vals[mask_mid], x_vals[mask_mid]]))
            else:
                ut_steering, vt_steering = ut_mean_val, vt_mean_val

            height_min_km = float(level_vals[zmin]) / 1000.0
            height_max_km = float(level_vals[zmax]) / 1000.0
            mask_base     = (z_vals == zmin)
            y_base, x_base = y_vals[mask_base], x_vals[mask_base]
            width_km_base = (float(max(int(x_base.max() - x_base.min() + 1),
                                       int(y_base.max() - y_base.min() + 1)))
                             if y_base.size > 0 else 0.0)

            unique_yx = list(set(zip(y_vals.tolist(), x_vals.tolist())))
            mri_vals  = mri_np[z_vals, y_vals, x_vals]
            mrc_vals  = mrc_np[z_vals, y_vals, x_vals]
            mrv_vals  = mrv_np[z_vals, y_vals, x_vals].tolist() if mrv_np is not None else []
            mrs_vals  = mrs_np[z_vals, y_vals, x_vals].tolist() if mrs_np is not None else []
            mrg_vals  = mrg_np[z_vals, y_vals, x_vals].tolist() if mrg_np is not None else []
            mrr_vals  = mrr_np[z_vals, y_vals, x_vals].tolist() if mrr_np is not None else []
            temp_vals = temp_np[z_vals, y_vals, x_vals].tolist() if temp_np is not None else []
            pres_vals = pres_np[z_vals, y_vals, x_vals].tolist() if pres_np is not None else []

            item = {
                "cc_id":        sg_data["cc_id"],
                "uc_id":     uc_id,
                "mcs_id":          mcs_id_co,
                "time_value":      time_hms,
                "area_max_dz_km2": metrics["area_max_dz_km2"],
                "area_mean_km2":   metrics["area_mean_km2"],
                "area_median_km2": metrics["area_median_km2"],
                "volume_km3":      metrics["volume_km3"],
                "height_min_km":   height_min_km,
                "height_max_km":   height_max_km,
                "height_km":       metrics["height_km"],
                "width_base_km":   width_km_base,
                "width_max_km":    float(metrics["width_max_dz"]),
                "width_mean_km":   metrics["width_mean_km"],
                "width_median_km": metrics["width_median_km"],
                "wt_max_ms":       vmax,
                "wt_mean_ms":      vmean,
                "wt_99pct_ms":     v99,
                "centroid_x":      x_mean,
                "centroid_y":      y_mean,
                "centroid_z":      z_mean,
                "ut_ms":           ut_mean_val,
                "vt_ms":           vt_mean_val,
                "ut_steering_ms":  ut_steering,
                "vt_steering_ms":  vt_steering,
                "mri_3d_values":  mri_vals.tolist(),
                "mri_mean":       mri_mean_val,
                "mrc_3d_values":  mrc_vals.tolist(),
                "mrc_mean":       mrc_mean_val,
                "mrv_3d_values":  mrv_vals, "mrv_mean": mrv_mean_val,
                "mrs_3d_values":  mrs_vals, "mrs_mean": mrs_mean_val,
                "mrg_3d_values":  mrg_vals, "mrg_mean": mrg_mean_val,
                "mrr_3d_values":  mrr_vals, "mrr_mean": mrr_mean_val,
                "indices_3d":      sg_data["indices_3d"],
                "indices_2d":      unique_yx,
                "wt_3d_values":    wt_vals.tolist(),
                "temp_3d_values":  temp_vals,
                "pres_3d_values":  pres_vals,
            }
            if temp_np is not None:
                item["temp_mean"] = float(np.nanmean(temp_np[z_vals, y_vals, x_vals]))
            if pres_np is not None:
                item["pres_mean"] = float(np.nanmean(pres_np[z_vals, y_vals, x_vals]))
            if unique_yx:
                ys_u, xs_u = zip(*unique_yx)
                if iwp_np is not None:
                    iwp_vals = iwp_np[ys_u, xs_u]
                    item["iwp_2d_values"] = iwp_vals.tolist()
                    item["iwp_mean"]      = float(np.nanmean(iwp_vals))
                    item["iwp_max"]       = float(np.nanmax(iwp_vals))
                if inprr_np is not None:
                    inprr_vals = inprr_np[ys_u, xs_u]
                    item["inprr_2d_values"] = inprr_vals.tolist()
                    item["inprr_mean"]      = float(np.nanmean(inprr_vals))
                    item["inprr_max"]       = float(np.nanmax(inprr_vals))
                if wice_np is not None:
                    wice_vals = wice_np[ys_u, xs_u]
                    item["wice_2d_values"] = wice_vals.tolist()
                    item["wice_mean"]      = float(np.nanmean(wice_vals))
                    item["wice_max"]       = float(np.nanmax(wice_vals))
                if vim_np is not None:
                    vim_vals = vim_np[ys_u, xs_u]
                    item["vim_2d_values"] = vim_vals.tolist()
                    item["vim_mean"]      = float(np.nanmean(vim_vals))
                    item["vim_max"]       = float(np.nanmax(vim_vals))

            cc_list.append(item)
            n_cc_this_uc += 1

        uc_list[-1]["n_cc"] = n_cc_this_uc

    # Top-down hierarchical constraint:
    #   a UC must contain >=1 CC AND belong to an MCS
    # uc_list = [
    #     uc for uc in uc_list
    #     if uc["n_cc"] > 0 and uc["mcs_id"] is not None
    # ]

    # # CC whose parent UC was filtered out are excluded (B->C cascade)
    # surviving_uc_ids = {uc["uc_id"] for uc in uc_list}
    # cc_list = [c for c in cc_list if c["uc_id"] in surviving_uc_ids]

    count = len(cc_list)
    results_by_time.append({
        "time":                       time_hms,
        "timestamp_s":                timestamp_s,
        "mcs":                        mcs_list,
        "n_mcs":                      len(mcs_list),
        "uc":               uc_list,
        "n_uc":             len(uc_list),
        "n_cc":                    count,
        "cc":                      cc_list,
        "n_raw_cores":                nb_raw_cores,
        "n_segmented_cores":          nb_segmented_cores,
        "n_subgroups_from_watershed": nb_subgroups_from_watershed,
    })
    return results_by_time


# ================================================================
# LEVEL C RECURSIVE SEGMENTATION (usable standalone)
# ================================================================

def segment_core_recursive(core_data, level_vals, min_width_threshold=1, depth=0):
    """
    Recursive segmentation of a convective object via adaptive H-Maxima watershed (Level C).

    At each recursion level, seeds are detected via H-Maxima on the local WT
    field, then the watershed separates the distinct cores. Recursion stops
    when a subgroup is small enough (``width_median <= min_width_threshold``)
    or when the watershed produces only a single region.

    This function can be called directly on any object to extract its
    convective cores without going through ``find_convective_cores_3d``.

    Parameters
    ----------
    core_data : dict
        Must contain:
       - ``"cc_id"``    : parent object identifier (str)
       - ``"indices_3d"``  : list of (z, y, x) tuples in global coordinates
       - ``"wt_3d_values"``: list of matching WT values (m/s)
    level_vals : ndarray float
        Level altitudes in meters (global coordinate, not the sub-domain).
    min_width_threshold : float
        Median width (km) below which recursion stops.
    depth : int
        Current recursion level (0 = root). Do not set manually.

    Returns
    -------
    list of dict
        One dict per final core with the keys:
        ``"cc_id"``, ``"indices_3d"``, ``"wt_3d_values"``.
    """
    indices_3d   = np.array(core_data["indices_3d"])
    wt_3d_values = np.array(core_data["wt_3d_values"])

    if len(indices_3d) == 0:
        return []

    z_vals = indices_3d[:, 0]
    y_vals = indices_3d[:, 1]
    x_vals = indices_3d[:, 2]

    zmin, zmax = z_vals.min(), z_vals.max()
    ymin, ymax = y_vals.min(), y_vals.max()
    xmin, xmax = x_vals.min(), x_vals.max()

    shape  = (zmax - zmin + 1, ymax - ymin + 1, xmax - xmin + 1)
    grid_3d = np.zeros(shape, dtype=float)
    grid_3d[z_vals - zmin, y_vals - ymin, x_vals - xmin] = wt_3d_values

    level_vals_local = level_vals[zmin : zmax + 1] if level_vals is not None else None

    mask_current = (grid_3d > 0)
    metrics      = _compute_component_metrics(mask_current, level_vals=level_vals_local)

    if metrics["width_median_km"] <= min_width_threshold:
        return [{
            "indices_3d":    indices_3d.tolist(),
            "wt_3d_values":  wt_3d_values.tolist(),
            "cc_id":      core_data.get("cc_subid", core_data.get("cc_id", "root")),
        }]

    structure_3d = _STRUCTURE_3D_RECURSIVE
    wt_local_max = np.max(grid_3d)

    h_dynamique = H_ADAPT_SLOPE * (wt_local_max - W_MIN_THRESHOLD_C)
    h_dynamique = float(min(h_dynamique, H_ADAPT_MAX))
    h_dynamique = max(H_ADAPT_MIN, h_dynamique)

    local_maxima      = h_maxima(grid_3d, h_dynamique, footprint=structure_3d)
    markers, num_markers = cc_label(local_maxima, structure=structure_3d)

    if MERGE_VERTICAL_STACKED_SEEDS and num_markers > 1:
        markers, num_markers, _ = _merge_stacked_markers(
            markers, grid_3d, xy_radius=STACKED_SEEDS_XY_RADIUS)

    mask_global    = grid_3d > 0   # input is already pre-filtered at W_MIN_THRESHOLD_C
    segmentation_3d = watershed(-grid_3d, markers, mask=mask_global)

    unique_labels = np.unique(segmentation_3d)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) <= 1:
        return [{
            "indices_3d":    indices_3d.tolist(),
            "wt_3d_values":  wt_3d_values.tolist(),
            "cc_id":      core_data.get("cc_subid", core_data.get("cc_id", "root")),
        }]

    subgroups = []
    for label_id in unique_labels:
        mask     = (segmentation_3d == label_id)
        z_local, y_local, x_local = np.where(mask)
        wt_vals_sub = grid_3d[mask]

        z_global = z_local + zmin
        y_global = y_local + ymin
        x_global = x_local + xmin

        parent_id       = core_data.get("cc_subid", core_data.get("cc_id"))
        new_subgroup_id = f"{parent_id}_d{depth}_s{int(label_id)}"

        subgroup = {
            "cc_subid":  new_subgroup_id,
            "indices_3d":   np.column_stack((z_global, y_global, x_global)).tolist(),
            "wt_3d_values": wt_vals_sub.tolist(),
        }
        subgroups.extend(
            segment_core_recursive(subgroup, level_vals, min_width_threshold, depth + 1)
        )

    return subgroups


# ================================================================
# TRACKING - COMMON UTILITIES
# ================================================================

_GRID_SIZE = np.int64(600)  # > max grid size (500 in CADDIWAF)


def _pack_indices(idx3d):
    """
    Encode (z, y, x) into a unique int64 key for fast set comparison.

    Returned sorted: every caller only ever uses these keys for order-
    independent IoU comparisons (see ``_iou``), and ``_advect_shift`` only
    adds a constant offset to y/x before re-packing, which preserves sort
    order. Sorting once here (instead of on every pairwise IoU call) is what
    lets ``_iou`` skip re-sorting its inputs.
    """
    G = _GRID_SIZE
    keys = (idx3d[:, 0].astype(np.int64) * G * G
            + idx3d[:, 1].astype(np.int64) * G
            + idx3d[:, 2].astype(np.int64))
    keys.sort()
    return keys


def _pack_indices_2d(yx_list):
    """Encode a list of (y, x) into unique int64 keys, sorted (see ``_pack_indices``)."""
    arr = np.array(yx_list, dtype=np.int64)
    keys = arr[:, 0] * _GRID_SIZE + arr[:, 1]
    keys.sort()
    return keys


def _advect_shift(idx3d, u_pix, v_pix):
    """Horizontally shift an object's voxels by advection (pixels)."""
    shifted = idx3d.copy()
    shifted[:, 1] += int(round(v_pix))   # y (meridional)
    shifted[:, 2] += int(round(u_pix))   # x (zonal)
    return shifted


def _advect_shift_2d(keys, u_pix, v_pix):
    """
    Shift packed 2D keys (y*G+x) by advection without unpacking them.

    Adding the same constant offset to every element of a sorted array keeps
    it sorted, so a sorted ``keys`` input stays sorted here - no re-sort needed.
    """
    return keys + int(round(v_pix)) * _GRID_SIZE + int(round(u_pix))


def _rebuild_geometry_3d(indices_a, indices_b):
    """
    Rebuild a local mask from the union of two disjoint voxel sets and
    return the ``_compute_component_metrics`` metrics of the recombined
    object, along with the (deduplicated) union of indices.

    Horizontal metrics (areas, per-z-level widths) do not depend on ``dz``
    and are therefore exact. ``height_km`` assumes a uniform ``dz`` (this
    pipeline stage has no access to the run's ``level_vals``) - the only
    approximation in this reconstruction.
    """
    idx = np.unique(np.array(list(indices_a) + list(indices_b), dtype=np.int64), axis=0)
    mins  = idx.min(axis=0)
    shape = tuple((idx.max(axis=0) - mins + 1).tolist())
    mask  = np.zeros(shape, dtype=bool)
    local = idx - mins
    mask[local[:, 0], local[:, 1], local[:, 2]] = True
    return _compute_component_metrics(mask), idx


def _union_indices(a, b):
    """Deduplicated union of two index lists (tuples (y,x) or (z,y,x))."""
    return [tuple(v) for v in ({tuple(v) for v in a} | {tuple(v) for v in b})]


def _merge_extra_ids(a, b, id_key):
    """
    Track the raw detection id(s) absorbed into a fused entry.

    ``fused = dict(a)`` keeps only ``a``'s own id (e.g. ``uc_id``) and
    silently drops ``b``'s - but ``b``'s raw id still exists as a genuinely
    separate object in the per-time-step detection output (``results.pkl``,
    since fusing the track's history does not and cannot retroactively merge
    the two blobs' raw pixels). Downstream consumers keyed on the raw id
    (e.g. the gif scripts, which color a track's pixels by looking up its
    raw id) need this list to attribute the fragment's own pixels to the
    surviving track too, instead of falling back to an "untracked" color.
    """
    extra_key = f"{id_key}_extra"
    return list(a.get(extra_key, [])) + [b[id_key]]


def _wavg(va, vb, wa, wb):
    """Exact weighted average of two means over disjoint sets."""
    if va is None or vb is None:
        return va if va is not None else vb
    wsum = wa + wb
    return (va * wa + vb * wb) / wsum if wsum > 0 else 0.5 * (va + vb)


def _vmax(va, vb):
    vals = [v for v in (va, vb) if v is not None]
    return max(vals) if vals else None


def _vmin(va, vb):
    vals = [v for v in (va, vb) if v is not None]
    return min(vals) if vals else None


def _fused_stat(a, b, raw_key, kind):
    """Exact statistic recomputed over the union of two raw per-voxel arrays."""
    ra, rb = a.get(raw_key), b.get(raw_key)
    if ra is None or rb is None:
        return None
    arr = np.asarray(list(ra) + list(rb), dtype=np.float64)
    if arr.size == 0:
        return None
    if kind == "mean":
        return float(np.nanmean(arr))
    if kind == "max":
        return float(np.nanmax(arr))
    if kind == "p99":
        return float(np.nanpercentile(arr, 99))
    raise ValueError(kind)


def _fuse_cc_cores(a, b):
    """
    Merge two disjoint CC (Level C) detections originating from the same
    physical object, fragmented over one time step. All intensity
    statistics are recomputed exactly from the raw per-voxel arrays
    (``wt_3d_values``, ``mri/mrc/mrv/mrs/mrg/mrr_3d_values``, ...), which
    are kept at this level.
    """
    geom, idx = _rebuild_geometry_3d(a["indices_3d"], b["indices_3d"])
    n_a, n_b  = len(a["indices_3d"]), len(b["indices_3d"])

    fused = dict(a)
    fused["cc_id_extra"]     = _merge_extra_ids(a, b, "cc_id")
    fused["indices_3d"]      = idx.tolist()
    fused["indices_2d"]      = _union_indices(a.get("indices_2d", []), b.get("indices_2d", []))
    fused["volume_km3"]      = a.get("volume_km3", 0.0) + b.get("volume_km3", 0.0)
    fused["height_min_km"]   = _vmin(a.get("height_min_km"), b.get("height_min_km"))
    fused["height_max_km"]   = _vmax(a.get("height_max_km"), b.get("height_max_km"))
    fused["area_max_dz_km2"] = geom["area_max_dz_km2"]
    fused["area_mean_km2"]   = geom["area_mean_km2"]
    fused["area_median_km2"] = geom["area_median_km2"]
    fused["width_max_km"]    = float(geom["width_max_dz"])
    fused["width_mean_km"]   = geom["width_mean_km"]
    fused["width_median_km"] = geom["width_median_km"]
    fused["height_km"]       = geom["height_km"]
    fused["centroid_x"] = _wavg(a.get("centroid_x"), b.get("centroid_x"), n_a, n_b)
    fused["centroid_y"] = _wavg(a.get("centroid_y"), b.get("centroid_y"), n_a, n_b)
    fused["centroid_z"] = _wavg(a.get("centroid_z"), b.get("centroid_z"), n_a, n_b)
    fused["ut_ms"]           = _wavg(a.get("ut_ms"), b.get("ut_ms"), n_a, n_b)
    fused["vt_ms"]           = _wavg(a.get("vt_ms"), b.get("vt_ms"), n_a, n_b)
    fused["ut_steering_ms"]  = _wavg(a.get("ut_steering_ms"), b.get("ut_steering_ms"), n_a, n_b)
    fused["vt_steering_ms"]  = _wavg(a.get("vt_steering_ms"), b.get("vt_steering_ms"), n_a, n_b)

    for raw_key, mean_key, max_key, p99_key in [
        ("wt_3d_values",   "wt_mean_ms", "wt_max_ms", "wt_99pct_ms"),
        ("mri_3d_values",  "mri_mean",   None,        None),
        ("mrc_3d_values",  "mrc_mean",   None,        None),
        ("mrv_3d_values",  "mrv_mean",   None,        None),
        ("mrs_3d_values",  "mrs_mean",   None,        None),
        ("mrg_3d_values",  "mrg_mean",   None,        None),
        ("mrr_3d_values",  "mrr_mean",   None,        None),
        ("temp_3d_values", "temp_mean",  None,        None),
        ("pres_3d_values", "pres_mean",  None,        None),
    ]:
        if raw_key in a and raw_key in b:
            fused[raw_key] = list(a[raw_key]) + list(b[raw_key])
            if mean_key: fused[mean_key] = _fused_stat(a, b, raw_key, "mean")
            if max_key:  fused[max_key]  = _fused_stat(a, b, raw_key, "max")
            if p99_key:  fused[p99_key]  = _fused_stat(a, b, raw_key, "p99")

    if "iwp_2d_values" in a and "iwp_2d_values" in b:
        fused["iwp_2d_values"] = list(a["iwp_2d_values"]) + list(b["iwp_2d_values"])
        fused["iwp_mean"] = _fused_stat(a, b, "iwp_2d_values", "mean")
        fused["iwp_max"]  = _fused_stat(a, b, "iwp_2d_values", "max")

    return fused


def _fuse_uc_cores(a, b):
    """
    Merge two disjoint UC (Level B) detections. No raw per-voxel array is
    kept at this level: ``wt_max_ms`` (max, exact) and ``wt_mean_ms``
    (voxel-count-weighted average, exact) are reconstructed, but
    ``wt_99pct_ms`` is a percentile - not reconstructible without the raw
    per-voxel values - and is removed from the merged dict rather than kept
    with a wrong value.
    """
    geom, idx = _rebuild_geometry_3d(a["indices_3d"], b["indices_3d"])
    n_a, n_b  = len(a["indices_3d"]), len(b["indices_3d"])

    fused = dict(a)
    fused["uc_id_extra"]     = _merge_extra_ids(a, b, "uc_id")
    fused["indices_3d"]      = idx.tolist()
    fused["indices_2d"]      = _union_indices(a.get("indices_2d", []), b.get("indices_2d", []))
    fused["volume_km3"]      = a.get("volume_km3", 0.0) + b.get("volume_km3", 0.0)
    fused["height_min_km"]   = _vmin(a.get("height_min_km"), b.get("height_min_km"))
    fused["height_max_km"]   = _vmax(a.get("height_max_km"), b.get("height_max_km"))
    fused["area_max_dz_km2"] = geom["area_max_dz_km2"]
    fused["area_mean_km2"]   = geom["area_mean_km2"]
    fused["area_median_km2"] = geom["area_median_km2"]
    fused["width_max_km"]    = float(geom["width_max_dz"])
    fused["width_mean_km"]   = geom["width_mean_km"]
    fused["width_median_km"] = geom["width_median_km"]
    fused["height_km"]       = geom["height_km"]
    fused["centroid_x"] = _wavg(a.get("centroid_x"), b.get("centroid_x"), n_a, n_b)
    fused["centroid_y"] = _wavg(a.get("centroid_y"), b.get("centroid_y"), n_a, n_b)
    fused["centroid_z"] = _wavg(a.get("centroid_z"), b.get("centroid_z"), n_a, n_b)
    fused["wt_max_ms"]  = _vmax(a.get("wt_max_ms"), b.get("wt_max_ms"))
    fused["wt_mean_ms"] = _wavg(a.get("wt_mean_ms"), b.get("wt_mean_ms"), n_a, n_b)
    fused["ut_ms"]           = _wavg(a.get("ut_ms"), b.get("ut_ms"), n_a, n_b)
    fused["vt_ms"]           = _wavg(a.get("vt_ms"), b.get("vt_ms"), n_a, n_b)
    fused["ut_steering_ms"]  = _wavg(a.get("ut_steering_ms"), b.get("ut_steering_ms"), n_a, n_b)
    fused["vt_steering_ms"]  = _wavg(a.get("vt_steering_ms"), b.get("vt_steering_ms"), n_a, n_b)
    fused["n_cc"] = a.get("n_cc", 0) + b.get("n_cc", 0)
    fused.pop("wt_99pct_ms", None)

    return fused


def _fuse_mcs_cores(a, b):
    """
    Merge two disjoint MCS (Level A) detections originating from the same
    physical object, fragmented over one time step.

    Unlike B/C, no 3D indices or raw per-voxel array are kept at Level A
    (MCS footprints are too large for that). Per-z-level metrics
    (``area_max_dz_km2``, ``width_max_km``, ...) therefore cannot be
    recomputed exactly via mask reconstruction: they are approximated by the
    extremum of the two fragments, means/medians by a ``n_voxels_3d``-
    weighted average. ``wt_99pct_ms`` (a percentile, not reconstructible
    without the raw array) is removed rather than kept wrong - same
    principle as at Level B.
    """
    n_a = len(a.get("indices_2d", [])) or 1
    n_b = len(b.get("indices_2d", [])) or 1
    v_a = a.get("n_voxels_3d", 0) or 1
    v_b = b.get("n_voxels_3d", 0) or 1

    fused = dict(a)
    fused["mcs_id_extra"] = _merge_extra_ids(a, b, "mcs_id")
    fused["indices_2d"]  = _union_indices(a.get("indices_2d", []), b.get("indices_2d", []))
    fused["area_2d_km2"] = a.get("area_2d_km2", 0.0) + b.get("area_2d_km2", 0.0)
    fused["n_voxels_3d"] = a.get("n_voxels_3d", 0) + b.get("n_voxels_3d", 0)
    fused["tb_min_k"]    = _vmin(a.get("tb_min_k"), b.get("tb_min_k"))
    fused["tb_mean_k"]   = _wavg(a.get("tb_mean_k"), b.get("tb_mean_k"), n_a, n_b)
    fused["pr_max_mmh"]  = _vmax(a.get("pr_max_mmh"), b.get("pr_max_mmh"))
    fused["z_top_idx"]   = _vmax(a.get("z_top_idx"), b.get("z_top_idx"))
    fused["z_base_idx"]  = _vmin(a.get("z_base_idx"), b.get("z_base_idx"))
    fused["centroid_x"]  = _wavg(a.get("centroid_x"), b.get("centroid_x"), n_a, n_b)
    fused["centroid_y"]  = _wavg(a.get("centroid_y"), b.get("centroid_y"), n_a, n_b)

    fused["volume_km3"]      = a.get("volume_km3", 0.0) + b.get("volume_km3", 0.0)
    fused["height_min_km"]   = _vmin(a.get("height_min_km"), b.get("height_min_km"))
    fused["height_max_km"]   = _vmax(a.get("height_max_km"), b.get("height_max_km"))
    fused["height_km"]       = _wavg(a.get("height_km"), b.get("height_km"), v_a, v_b)
    fused["area_max_dz_km2"] = _vmax(a.get("area_max_dz_km2"), b.get("area_max_dz_km2"))
    fused["area_mean_km2"]   = _wavg(a.get("area_mean_km2"), b.get("area_mean_km2"), v_a, v_b)
    fused["area_median_km2"] = _wavg(a.get("area_median_km2"), b.get("area_median_km2"), v_a, v_b)
    fused["width_max_km"]    = _vmax(a.get("width_max_km"), b.get("width_max_km"))
    fused["width_mean_km"]   = _wavg(a.get("width_mean_km"), b.get("width_mean_km"), v_a, v_b)
    fused["width_median_km"] = _wavg(a.get("width_median_km"), b.get("width_median_km"), v_a, v_b)
    fused["wt_max_ms"]  = _vmax(a.get("wt_max_ms"), b.get("wt_max_ms"))
    fused["wt_mean_ms"] = _wavg(a.get("wt_mean_ms"), b.get("wt_mean_ms"), v_a, v_b)
    fused["ut_ms"] = _wavg(a.get("ut_ms"), b.get("ut_ms"), v_a, v_b)
    fused["vt_ms"] = _wavg(a.get("vt_ms"), b.get("vt_ms"), v_a, v_b)
    fused.pop("wt_99pct_ms", None)

    return fused


def _hms_to_seconds(hms):
    """Parse a time string into seconds. Handles both 'HH:MM:SS' and ISO timestamps."""
    hms = str(hms)
    if "T" in hms:  # Format ISO (ex: "2024-01-01T00:05:00")
        hms = hms.split("T")[-1].split(".")[0]  # Extrait "00:05:00"
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


def _collapse_transient_fragments(all_tracks, fuse_fn, max_fragment_span_s=120.0):
    """
    Retroactively merge fragments born from a split that is reabsorbed by
    their own parent, provided the fragment's own lifetime fits within
    ``max_fragment_span_s``.

    The dataset's time steps are not uniform here (e.g. bursts of a few fine
    30s/60s steps separated by a much larger gap, ~780s in CADDIWAF7): a
    fragment observed only within one such burst (its own history spans at
    most ``max_fragment_span_s``) and that merges back into the very track it
    split from corresponds to a transient segmentation fragmentation (e.g. an
    unstable watershed for that burst) rather than a real physical
    split/merge - regardless of whether it was observed once or several
    times within that burst. A fragment that instead survives across a
    burst-to-burst gap before remerging is kept as a genuine split (its own
    span exceeds ``max_fragment_span_s``).

    Each of the fragment's history entries is recombined, via ``fuse_fn``
    (specific to level A/B/C), into the parent's own entry for the matching
    time step - instead of being silently lost. The fragment's track is then
    deleted and the corresponding split/absorbed events are removed from the
    parent, replaced by a single ``"transient_fragment"`` event. The parent
    keeps its unique ``track_id`` throughout the whole sequence.
    """
    to_collapse = []
    for tid, tr in all_tracks.items():
        parent_id = tr.get("parent_id")
        history   = tr.get("history", [])
        if parent_id is None or not history or parent_id not in all_tracks:
            continue
        span_s = (_hms_to_seconds(history[-1]["time_value"])
                 - _hms_to_seconds(history[0]["time_value"]))
        if span_s > max_fragment_span_s:
            continue
        merge_events = [e for e in tr.get("events", []) if e.get("type") == "merge"]
        if len(merge_events) != 1 or merge_events[0].get("into") != parent_id:
            continue
        to_collapse.append((tid, parent_id, history, tr.get("start_time")))

    for tid, parent_id, frag_history, frag_start_time in to_collapse:
        parent = all_tracks[parent_id]
        for frag_entry in frag_history:
            frag_time_value = frag_entry.get("time_value")
            for i, h in enumerate(parent["history"]):
                if h.get("time_value") == frag_time_value:
                    parent["history"][i] = fuse_fn(h, frag_entry)
                    break

        parent["events"] = [
            e for e in parent.get("events", [])
            if not (e.get("type") == "split"    and e.get("child_id") == tid)
            and not (e.get("type") == "absorbed" and e.get("from")     == tid)
        ]
        parent["events"].append(
            {"type": "transient_fragment", "time": frag_start_time}
        )
        del all_tracks[tid]

    return all_tracks


def _count_common_sorted(sorted_a, sorted_b):
    """
    Number of elements common to two sorted, unique int64 arrays.

    Equivalent to ``len(np.intersect1d(sorted_a, sorted_b, assume_unique=True))``
    but avoids re-sorting either array (``np.intersect1d`` always sorts both
    of its inputs internally, even when they're already sorted) - a real cost
    here since the matching loops call this for every (track, item) pair with
    the same ``shifted_keys``/``ci["keys"]`` arrays reused across many pairs.
    """
    if sorted_a.size == 0 or sorted_b.size == 0:
        return 0
    insert_at = np.searchsorted(sorted_a, sorted_b).clip(max=sorted_a.size - 1)
    return int(np.count_nonzero(sorted_a[insert_at] == sorted_b))


def _iou(keys_a, keys_b):
    """3D IoU between two sets of voxel keys (numpy int64, already sorted - see ``_pack_indices``)."""
    n_inter = _count_common_sorted(keys_a, keys_b)
    n_union = keys_a.size + keys_b.size - n_inter
    return n_inter / n_union if n_union > 0 else 0.0


def _iou_2d(keys_a, keys_b):
    """2D IoU between two sets of footprint keys (numpy int64, already sorted - see ``_pack_indices_2d``)."""
    n_inter = _count_common_sorted(keys_a, keys_b)
    n_union = keys_a.size + keys_b.size - n_inter
    return n_inter / n_union if n_union > 0 else 0.0


# ================================================================
# TRACKING - LEVEL C (Convective Cores - CC)
# ================================================================

def _core_features(core):
    """
    Feature vector [7] for the morphological similarity of cores.
    All values are positive; similarity is computed via min/max ratio
    (scale-invariant).
    """
    wt_arr = np.array(core.get("wt_3d_values", [0.0]), dtype=np.float32)
    return np.array([
        core.get("area_max_dz_km2", 0.0),
        core.get("volume_km3",      0.0),
        core.get("wt_99pct_ms",     core.get("wt_max_ms", 0.0)),
        core.get("iwp_mean",        0.0),
        core.get("mrg_mean",        0.0),
        float(np.percentile(wt_arr, 50)),
        float(np.percentile(wt_arr, 75)),
    ], dtype=np.float32)


# Per-feature weights (same order as _core_features):
#   area_max_dz_km2  volume_km3  wt_99pct_ms  iwp_mean  mrg_mean  wt_p50  wt_p75
_FEAT_WEIGHTS        = np.array([1.0, 1.0, 1.5, 2.0, 0.5, 2.0, 1.5], dtype=np.float32)
_FEAT_WEIGHTS       /= _FEAT_WEIGHTS.sum()


def _morph_sim(feat_a, feat_b):
    """Weighted similarity [0, 1] between two feature vectors (min/max ratio)."""
    mx    = np.maximum(feat_a, feat_b)
    valid = mx > 1e-9
    if not valid.any():
        return 0.5
    safe_mx = np.where(valid, mx, 1.0)
    ratios  = np.where(valid, np.minimum(feat_a, feat_b) / safe_mx, 1.0)
    w       = np.where(valid, _FEAT_WEIGHTS, 0.0)
    w_sum   = w.sum()
    return float((ratios * w).sum() / w_sum) if w_sum > 0 else 0.5


def track_cc(
    results_by_time,
    min_overlap=0.1,
    patience=1,
    dx_meters=DX_METERS,
    w_overlap=0.6,
    w_morph=0.4,
    min_iou=0.02,
    feat_ema_alpha=0.7,
    max_fragment_span_s=120.0,
):
    """
    Tracking of convective cores (Level C) via advected 3D IoU + Hungarian algorithm.

    Per-time-step strategy:
      1. Shift the track's voxels by advection (u_steering * dt, v_steering * dt).
      2. Compute the 3D IoU between advected voxels and voxels detected at T+1.
      3. Filter out candidates with IoU < min_iou (no geometric distance filter).
      4. Combine with morphological similarity (EMA over the track's history).
      5. Optimal assignment via the Hungarian algorithm.
      6. Detect splits and merges from the residual scores.

    Parameters
    ----------
    results_by_time : list of dict
        Output of ``find_convective_cores_3d`` concatenated over several time steps.
    min_overlap : float
        Minimum composite score to validate a match.
    patience : int
        Number of time steps without an observation before the track is dropped.
    dx_meters : float
        Grid resolution (m/pixel).
    w_overlap, w_morph : float
        Weights of the 3D IoU and morphological similarity in the composite score.
    min_iou : float
        Minimum advected 3D IoU for a candidate to enter the score matrix.
        Replaces the geometric centroid-distance filter.
    feat_ema_alpha : float
        Memory coefficient of the feature EMA (0 = no memory, 1 = frozen).
    max_fragment_span_s : float
        Maximum lifetime (s) of a split-off fragment, measured over its own
        history, for it to be collapsed back into its parent as a transient
        segmentation artifact (see ``_collapse_transient_fragments``). A
        fragment that survives longer than this before remerging into its
        own parent is kept as a genuine split.

    Returns
    -------
    dict
        Keys: track identifiers (``"Track_XXXX"``).
        Values: dict with ``"track_id"``, ``"start_time"``, ``"end_time"``,
        ``"duration_minutes"``, ``"history"`` (list of core snapshots), and
        optionally ``"events"`` (splits/merges) and ``"parent_id"``.
    """
    all_tracks    = {}
    active_tracks = {}
    track_counter = 0

    for t_idx, time_data in enumerate(results_by_time):
        time_val   = time_data["time"]
        current_ts = time_data.get("timestamp_s", t_idx * 30.0)
        cc_detections = time_data["cc"]

        core_items = []
        for c in cc_detections:
            cy    = c["centroid_y"]
            cx    = c["centroid_x"]
            idx3d = np.array(c["indices_3d"], dtype=np.int32)
            core_items.append({
                "core": c,
                "cy": cy, "cx": cx,
                "idx3d": idx3d,
                "keys": _pack_indices(idx3d),
                "feat": _core_features(c),
            })

        matched_tracks  = set()
        matched_cores   = set()
        split_children  = {}

        if active_tracks and core_items:
            track_ids    = list(active_tracks.keys())
            score_matrix = np.zeros((len(track_ids), len(core_items)), dtype=np.float32)

            for r, tid in enumerate(track_ids):
                st           = active_tracks[tid]
                dt           = current_ts - st["last_timestamp_s"]
                u_pix        = st["u"] * dt / dx_meters
                v_pix        = st["v"] * dt / dx_meters
                shifted_keys = _pack_indices(_advect_shift(st["idx3d"], u_pix, v_pix))

                for c_idx, ci in enumerate(core_items):
                    iou = _iou(shifted_keys, ci["keys"])
                    if iou < min_iou:
                        continue
                    morph = _morph_sim(st["feat_ema"], ci["feat"])
                    score_matrix[r, c_idx] = w_overlap * iou + w_morph * morph

            row_ind, col_ind = linear_sum_assignment(1.0 - score_matrix)
            matched_col_to_row = {}

            for r, c_idx in zip(row_ind, col_ind):
                if score_matrix[r, c_idx] < min_overlap:
                    continue
                tid  = track_ids[r]
                ci   = core_items[c_idx]
                core = ci["core"]
                matched_tracks.add(tid)
                matched_cores.add(c_idx)
                matched_col_to_row[c_idx] = r

                st = active_tracks[tid]
                st["last_time_idx"]    = t_idx
                st["last_timestamp_s"] = current_ts
                st["cy"]     = ci["cy"]
                st["cx"]     = ci["cx"]
                st["idx3d"]  = ci["idx3d"]
                st["last_core"] = core
                st["last_feat"] = ci["feat"]
                st["feat_ema"]  = (feat_ema_alpha * st["feat_ema"]
                                   + (1.0 - feat_ema_alpha) * ci["feat"])
                st["u"] = core.get("ut_steering_ms", core.get("ut_ms", st["u"]))
                st["v"] = core.get("vt_steering_ms", core.get("vt_ms", st["v"]))
                all_tracks[tid]["history"].append(core)

            # Merges: unmatched track that overlaps an already-taken core -> dies immediately
            for r, tid in enumerate(track_ids):
                if tid in matched_tracks:
                    continue
                for c_idx in matched_cores:
                    if score_matrix[r, c_idx] >= min_overlap:
                        winner_tid = track_ids[matched_col_to_row[c_idx]]
                        all_tracks[tid].setdefault("events", []).append(
                            {"type": "merge", "time": time_val, "into": winner_tid}
                        )
                        all_tracks[winner_tid].setdefault("events", []).append(
                            {"type": "absorbed", "time": time_val, "from": tid}
                        )
                        if tid in active_tracks:
                            all_tracks[tid]["end_time"]         = time_val
                            all_tracks[tid]["end_timestamp_s"]  = current_ts
                            all_tracks[tid]["duration_minutes"] = (
                                current_ts - all_tracks[tid]["start_timestamp_s"]) / 60.0
                            del active_tracks[tid]
                        break

            # Splits: unmatched core that overlaps an already-taken track
            # (the "split" event is created further below, in the "new
            # cores" loop, once the child's ID is known - see child_id)
            split_children = {}
            for c_idx, ci in enumerate(core_items):
                if c_idx in matched_cores:
                    continue
                best_score, best_parent = -1.0, None
                for r, tid in enumerate(track_ids):
                    if tid not in matched_tracks:
                        continue
                    s = score_matrix[r, c_idx]
                    if s >= min_overlap and s > best_score:
                        best_score, best_parent = s, tid
                if best_parent is not None:
                    split_children[c_idx] = best_parent

        # Unmatched new cores -> new tracks
        for c_idx, ci in enumerate(core_items):
            if c_idx in matched_cores:
                continue
            track_counter += 1
            tid  = f"CCTrack_{track_counter:04d}"
            core = ci["core"]

            parent_tid = split_children.get(c_idx)
            if parent_tid and parent_tid in active_tracks:
                parent_st = active_tracks[parent_tid]
                u_init = core.get("ut_steering_ms", core.get("ut_ms", parent_st["u"]))
                v_init = core.get("vt_steering_ms", core.get("vt_ms", parent_st["v"]))
            else:
                u_init = core.get("ut_steering_ms", core.get("ut_ms", 0.0))
                v_init = core.get("vt_steering_ms", core.get("vt_ms", 0.0))

            track_data = {
                "track_id":          tid,
                "start_time":        time_val,
                "start_timestamp_s": current_ts,
                "history":           [core],
            }
            if parent_tid:
                track_data["parent_id"] = parent_tid
                all_tracks[parent_tid].setdefault("events", []).append(
                    {"type": "split", "time": time_val,
                     "child_id": tid, "child_centroid": (ci["cy"], ci["cx"])}
                )
            all_tracks[tid]    = track_data
            active_tracks[tid] = {
                "last_time_idx":    t_idx,
                "last_timestamp_s": current_ts,
                "cy":       ci["cy"], "cx": ci["cx"],
                "idx3d":    ci["idx3d"],
                "last_core": core,
                "last_feat": ci["feat"],
                "feat_ema": ci["feat"].copy(),
                "u": u_init, "v": v_init,
            }

        # Patience
        dead = [tid for tid, st in active_tracks.items()
                if t_idx - st["last_time_idx"] > patience]
        for tid in dead:
            st       = active_tracks[tid]
            last_idx = st["last_time_idx"]
            all_tracks[tid]["end_time"]         = results_by_time[last_idx]["time"]
            all_tracks[tid]["end_timestamp_s"]  = results_by_time[last_idx].get(
                "timestamp_s", st["last_timestamp_s"])
            all_tracks[tid]["duration_minutes"] = (
                all_tracks[tid]["end_timestamp_s"]
               - all_tracks[tid]["start_timestamp_s"]) / 60.0
            del active_tracks[tid]

    # Finalize tracks that are still active
    for tid, st in active_tracks.items():
        last_idx = st["last_time_idx"]
        all_tracks[tid]["end_time"]         = results_by_time[last_idx]["time"]
        all_tracks[tid]["end_timestamp_s"]  = results_by_time[last_idx].get(
            "timestamp_s", st["last_timestamp_s"])
        all_tracks[tid]["duration_minutes"] = (
            all_tracks[tid]["end_timestamp_s"]
           - all_tracks[tid]["start_timestamp_s"]) / 60.0

    return _collapse_transient_fragments(all_tracks, _fuse_cc_cores, max_fragment_span_s)


# ================================================================
# TRACKING - LEVEL B (Convective Objects - UC)
# ================================================================

def _uc_features(co):
    """Feature vector [4] for the morphological similarity of convective objects."""
    return np.array([
        co.get("area_max_dz_km2", 0.0),
        co.get("volume_km3",      0.0),
        co.get("width_max_km",    0.0),
        co.get("wt_99pct_ms",     co.get("wt_max_ms", 0.0)),
    ], dtype=np.float32)


_UC_FEAT_WEIGHTS  = np.array([1.5, 1.0, 1.5, 2.0], dtype=np.float32)
_UC_FEAT_WEIGHTS /= _UC_FEAT_WEIGHTS.sum()


def _morph_sim_uc(feat_a, feat_b):
    """Weighted morphological similarity for convective objects."""
    mx    = np.maximum(feat_a, feat_b)
    valid = mx > 1e-9
    if not valid.any():
        return 0.5
    safe_mx = np.where(valid, mx, 1.0)
    ratios  = np.where(valid, np.minimum(feat_a, feat_b) / safe_mx, 1.0)
    w       = np.where(valid, _UC_FEAT_WEIGHTS, 0.0)
    w_sum   = w.sum()
    return float((ratios * w).sum() / w_sum) if w_sum > 0 else 0.5


def track_uc(
    results_by_time,
    min_overlap=0.1,
    patience=2,
    dx_meters=DX_METERS,
    w_overlap=0.7,
    w_morph=0.3,
    min_iou=0.05,
    feat_ema_alpha=0.7,
    max_fragment_span_s=120.0,
):
    """
    Tracking of convective objects (Level B) via advected 3D IoU + Hungarian algorithm.

    Same logic as ``track_cc`` (Level C), including full split/merge/absorbed
    detection. Patience is higher than at Level C (2 vs 1): convective
    objects are more stable and can temporarily fall outside the WT mask
    without physically disappearing.

    Parameters
    ----------
    results_by_time : list of dict
        Output of ``find_convective_cores_3d`` concatenated over several time steps.
    min_overlap : float
        Minimum composite score to validate a match.
    patience : int
        Number of time steps without an observation before the track is dropped.
    dx_meters : float
        Grid resolution (m/pixel).
    w_overlap, w_morph : float
        Weights of the 3D IoU and morphological similarity.
    min_iou : float
        Minimum advected 3D IoU for a candidate to enter the score matrix.
        Replaces the geometric centroid-distance filter.
    feat_ema_alpha : float
        Memory coefficient of the feature EMA.
    max_fragment_span_s : float
        Maximum lifetime (s) of a split-off fragment for it to be collapsed
        back into its parent as a transient segmentation artifact - see
        ``track_cc`` / ``_collapse_transient_fragments``.

    Returns
    -------
    dict
        Keys: track identifiers (``"UCTrack_XXXX"``).
        Values: dict with ``"track_id"``, ``"start_time"``, ``"end_time"``,
        ``"duration_minutes"``, ``"history"``.
    """
    all_tracks    = {}
    active_tracks = {}
    track_counter = 0

    for t_idx, time_data in enumerate(results_by_time):
        time_val   = time_data["time"]
        current_ts = time_data.get("timestamp_s", t_idx * 30.0)
        uc_detections = time_data.get("uc", [])

        items = []
        for co in uc_detections:
            yx3d = co.get("indices_3d", [])
            if not yx3d:
                continue
            idx3d = np.array(yx3d, dtype=np.int32)
            items.append({
                "co":    co,
                "cy":    co["centroid_y"],
                "cx":    co["centroid_x"],
                "idx3d": idx3d,
                "keys":  _pack_indices(idx3d),
                "feat":  _uc_features(co),
            })

        matched_tracks     = set()
        matched_items      = set()
        matched_col_to_row = {}
        split_children     = {}

        if active_tracks and items:
            track_ids    = list(active_tracks.keys())
            score_matrix = np.zeros((len(track_ids), len(items)), dtype=np.float32)

            for r, tid in enumerate(track_ids):
                st           = active_tracks[tid]
                dt           = current_ts - st["last_timestamp_s"]
                u_pix        = st["u"] * dt / dx_meters
                v_pix        = st["v"] * dt / dx_meters
                shifted_keys = _pack_indices(_advect_shift(st["idx3d"], u_pix, v_pix))

                for c_idx, ci in enumerate(items):
                    iou = _iou(shifted_keys, ci["keys"])
                    if iou < min_iou:
                        continue
                    morph = _morph_sim_uc(st["feat_ema"], ci["feat"])
                    score_matrix[r, c_idx] = w_overlap * iou + w_morph * morph

            row_ind, col_ind = linear_sum_assignment(1.0 - score_matrix)
            for r, c_idx in zip(row_ind, col_ind):
                if score_matrix[r, c_idx] < min_overlap:
                    continue
                tid = track_ids[r]
                ci  = items[c_idx]
                co  = ci["co"]
                matched_tracks.add(tid)
                matched_items.add(c_idx)
                matched_col_to_row[c_idx] = r

                st = active_tracks[tid]
                st["last_time_idx"]    = t_idx
                st["last_timestamp_s"] = current_ts
                st["cy"]      = ci["cy"]
                st["cx"]      = ci["cx"]
                st["idx3d"]   = ci["idx3d"]
                st["feat_ema"] = (feat_ema_alpha * st["feat_ema"]
                                  + (1.0 - feat_ema_alpha) * ci["feat"])
                st["u"] = co.get("ut_steering_ms", co.get("ut_ms", st["u"]))
                st["v"] = co.get("vt_steering_ms", co.get("vt_ms", st["v"]))
                all_tracks[tid]["history"].append(co)

            # Merges: unmatched track that overlaps an already-taken UC -> dies immediately
            for r, tid in enumerate(track_ids):
                if tid in matched_tracks:
                    continue
                for c_idx in matched_items:
                    if score_matrix[r, c_idx] >= min_overlap:
                        winner_tid = track_ids[matched_col_to_row[c_idx]]
                        all_tracks[tid].setdefault("events", []).append(
                            {"type": "merge", "time": time_val, "into": winner_tid}
                        )
                        all_tracks[winner_tid].setdefault("events", []).append(
                            {"type": "absorbed", "time": time_val, "from": tid}
                        )
                        if tid in active_tracks:
                            all_tracks[tid]["end_time"]         = time_val
                            all_tracks[tid]["end_timestamp_s"]  = current_ts
                            all_tracks[tid]["duration_minutes"] = (
                                current_ts - all_tracks[tid]["start_timestamp_s"]) / 60.0
                            del active_tracks[tid]
                        break

            # Splits: unmatched UC that overlaps an already-taken track
            # (the "split" event is created further below, once the child's ID is known)
            for c_idx, ci in enumerate(items):
                if c_idx in matched_items:
                    continue
                best_score, best_parent = -1.0, None
                for r, tid in enumerate(track_ids):
                    if tid not in matched_tracks:
                        continue
                    s = score_matrix[r, c_idx]
                    if s >= min_overlap and s > best_score:
                        best_score, best_parent = s, tid
                if best_parent is not None:
                    split_children[c_idx] = best_parent

        for c_idx, ci in enumerate(items):
            if c_idx in matched_items:
                continue
            track_counter += 1
            tid        = f"UCTrack_{track_counter:04d}"
            co         = ci["co"]
            parent_tid = split_children.get(c_idx)
            track_data = {
                "track_id":          tid,
                "start_time":        time_val,
                "start_timestamp_s": current_ts,
                "history":           [co],
            }
            if parent_tid:
                track_data["parent_id"] = parent_tid
                all_tracks[parent_tid].setdefault("events", []).append(
                    {"type": "split", "time": time_val,
                     "child_id": tid, "child_centroid": (ci["cy"], ci["cx"])}
                )
            all_tracks[tid] = track_data
            active_tracks[tid] = {
                "last_time_idx":    t_idx,
                "last_timestamp_s": current_ts,
                "cy":      ci["cy"],
                "cx":      ci["cx"],
                "idx3d":   ci["idx3d"],
                "feat_ema": ci["feat"].copy(),
                "u": co.get("ut_steering_ms", co.get("ut_ms", 0.0)),
                "v": co.get("vt_steering_ms", co.get("vt_ms", 0.0)),
                "patience": 0 if parent_tid else patience,
            }

        dead = [tid for tid, st in active_tracks.items()
                if t_idx - st["last_time_idx"] > st.get("patience", patience)]
        for tid in dead:
            st       = active_tracks[tid]
            last_idx = st["last_time_idx"]
            all_tracks[tid]["end_time"]         = results_by_time[last_idx]["time"]
            all_tracks[tid]["end_timestamp_s"]  = results_by_time[last_idx].get(
                "timestamp_s", st["last_timestamp_s"])
            all_tracks[tid]["duration_minutes"] = (
                all_tracks[tid]["end_timestamp_s"]
               - all_tracks[tid]["start_timestamp_s"]) / 60.0
            del active_tracks[tid]

    for tid, st in active_tracks.items():
        last_idx = st["last_time_idx"]
        all_tracks[tid]["end_time"]         = results_by_time[last_idx]["time"]
        all_tracks[tid]["end_timestamp_s"]  = results_by_time[last_idx].get(
            "timestamp_s", st["last_timestamp_s"])
        all_tracks[tid]["duration_minutes"] = (
            all_tracks[tid]["end_timestamp_s"]
           - all_tracks[tid]["start_timestamp_s"]) / 60.0

    return _collapse_transient_fragments(all_tracks, _fuse_uc_cores, max_fragment_span_s)


# ================================================================
# TRACKING - LEVEL A (MCS)
# ================================================================

def track_mcs(
    results_by_time,
    min_iou=0.05,
    patience=4,
    mcs_min_area_bt_km2=10000.0,
    mcs_bt_overshoot_k=225.0,
    mcs_pr_max_mmh=10.0,
    mcs_min_duration_minutes=30.0,
    max_fragment_span_s=120.0,
):
    """
    Tracking of MCSs (Level A) via advected 2D IoU + Hungarian algorithm.

    Each track's footprint is shifted by its steering wind (``ut_steering``,
    ``vt_steering`` stored in the MCS dict) before computing the 2D IoU.

    Splits and merges are detected and recorded in ``"events"``:
   - Split : an MCS gives rise to several children. The parent track keeps
              its own ``track_id`` and continues under the branch the
              Hungarian assignment matches it to; each child inherits
              ``"parent_id"``.
   - Merge : several MCSs converge into one. The absorbed tracks end
              immediately with a ``"merge"`` event.

    Parameters
    ----------
    results_by_time : list of dict
        Output of ``find_convective_cores_3d`` concatenated over several time steps.
    min_iou : float
        Minimum 2D IoU to validate a match.
    patience : int
        Number of time steps without an observation before the track is dropped.
    mcs_min_area_bt_km2 : float or None
        Maximum Tb shield area over the track's lifetime (km2).
        None to disable this criterion.
    mcs_bt_overshoot_k : float or None
        Cold overshoot threshold: min(tb_min_k) over the track must be <= this
        value (K). None to disable.
    mcs_pr_max_mmh : float or None
        Maximum precipitation under the shield over the track (mm/h). None to
        disable, or if INPRR is absent from the dataset.
    max_fragment_span_s : float
        Maximum lifetime (s) of a split-off fragment for it to be collapsed
        back into its parent as a transient segmentation artifact - see
        ``track_cc`` / ``_collapse_transient_fragments``.

    Returns
    -------
    dict
        Keys: track identifiers (``"MCSTrack_XXXX"``).
        Values: dict with ``"track_id"``, ``"start_time"``, ``"end_time"``,
        ``"duration_minutes"``, ``"history"``, and optionally ``"events"``
        and ``"parent_id"``.
        Only tracks whose peak validates the MOAAP criteria are returned;
        each track keeps all of its time steps (genesis + maturity + decay).
    """
    all_tracks    = {}
    active_tracks = {}
    track_counter = 0

    for t_idx, time_data in enumerate(results_by_time):
        time_val   = time_data["time"]
        current_ts = time_data.get("timestamp_s", t_idx * 30.0)
        mcs_list   = time_data.get("mcs", [])

        items = []
        for mcs in mcs_list:
            yx = mcs.get("indices_2d", [])
            if not yx:
                continue
            items.append({
                "mcs":  mcs,
                "cy":   mcs["centroid_y"],
                "cx":   mcs["centroid_x"],
                "u":    mcs.get("ut_steering", 0.0),
                "v":    mcs.get("vt_steering", 0.0),
                "keys": _pack_indices_2d(yx),
            })

        matched_tracks     = set()
        matched_items      = set()
        split_children     = {}
        matched_col_to_row = {}

        if active_tracks and items:
            track_ids    = list(active_tracks.keys())
            score_matrix = np.zeros((len(track_ids), len(items)), dtype=np.float32)

            for r, tid in enumerate(track_ids):
                st = active_tracks[tid]
                dt = current_ts - st["last_timestamp_s"]
                if dt <= 0:
                    dt = 30.0

                # Footprint advection
                u_pix    = st["u"] * dt / DX_METERS
                v_pix    = st["v"] * dt / DX_METERS
                adv_keys = _advect_shift_2d(st["keys"], u_pix, v_pix)

                for c_idx, ci in enumerate(items):
                    score_matrix[r, c_idx] = _iou_2d(adv_keys, ci["keys"])

            row_ind, col_ind = linear_sum_assignment(1.0 - score_matrix)
            for r, c_idx in zip(row_ind, col_ind):
                if score_matrix[r, c_idx] < min_iou:
                    continue
                tid = track_ids[r]
                ci  = items[c_idx]
                matched_tracks.add(tid)
                matched_items.add(c_idx)
                matched_col_to_row[c_idx] = r

                st = active_tracks[tid]
                st["last_time_idx"]    = t_idx
                st["last_timestamp_s"] = current_ts
                st["cy"]   = ci["cy"]
                st["cx"]   = ci["cx"]
                st["u"]    = ci["u"]
                st["v"]    = ci["v"]
                st["keys"] = ci["keys"]
                all_tracks[tid]["history"].append(ci["mcs"])

            # Merges: unmatched track that overlaps an already-taken MCS -> dies immediately
            for r, tid in enumerate(track_ids):
                if tid in matched_tracks:
                    continue
                for c_idx in matched_items:
                    if score_matrix[r, c_idx] >= min_iou:
                        winner_tid = track_ids[matched_col_to_row[c_idx]]
                        all_tracks[tid].setdefault("events", []).append(
                            {"type": "merge", "time": time_val, "into": winner_tid}
                        )
                        all_tracks[winner_tid].setdefault("events", []).append(
                            {"type": "absorbed", "time": time_val, "from": tid}
                        )
                        if tid in active_tracks:
                            all_tracks[tid]["end_time"]         = time_val
                            all_tracks[tid]["end_timestamp_s"]  = current_ts
                            all_tracks[tid]["duration_minutes"] = (
                                current_ts - all_tracks[tid]["start_timestamp_s"]) / 60.0
                            del active_tracks[tid]
                        break

            # Splits: unmatched MCS that overlaps an already-taken track
            # (the "split" event is created further below, once the child's ID is known)
            for c_idx, ci in enumerate(items):
                if c_idx in matched_items:
                    continue
                best_score, best_parent = -1.0, None
                for r, tid in enumerate(track_ids):
                    if tid not in matched_tracks:
                        continue
                    s = score_matrix[r, c_idx]
                    if s >= min_iou and s > best_score:
                        best_score, best_parent = s, tid
                if best_parent is not None:
                    split_children[c_idx] = best_parent

        # Unmatched new MCSs -> new tracks
        for c_idx, ci in enumerate(items):
            if c_idx in matched_items:
                continue
            track_counter += 1
            tid = f"MCSTrack_{track_counter:04d}"
            mcs = ci["mcs"]

            track_data = {
                "track_id":          tid,
                "start_time":        time_val,
                "start_timestamp_s": current_ts,
                "history":           [mcs],
            }
            parent_tid = split_children.get(c_idx)
            if parent_tid:
                track_data["parent_id"] = parent_tid
                all_tracks[parent_tid].setdefault("events", []).append(
                    {"type": "split", "time": time_val,
                     "child_id": tid, "child_centroid": (ci["cy"], ci["cx"])}
                )

            all_tracks[tid] = track_data
            active_tracks[tid] = {
                "last_time_idx":    t_idx,
                "last_timestamp_s": current_ts,
                "cy":   ci["cy"], "cx": ci["cx"],
                "u":    ci["u"],  "v":  ci["v"],
                "keys": ci["keys"],
            }

        dead = [tid for tid, st in active_tracks.items()
                if t_idx - st["last_time_idx"] > patience]
        for tid in dead:
            st       = active_tracks[tid]
            last_idx = st["last_time_idx"]
            all_tracks[tid]["end_time"]         = results_by_time[last_idx]["time"]
            all_tracks[tid]["end_timestamp_s"]  = results_by_time[last_idx].get(
                "timestamp_s", st["last_timestamp_s"])
            all_tracks[tid]["duration_minutes"] = (
                all_tracks[tid]["end_timestamp_s"]
               - all_tracks[tid]["start_timestamp_s"]) / 60.0
            del active_tracks[tid]

    for tid, st in active_tracks.items():
        last_idx = st["last_time_idx"]
        all_tracks[tid]["end_time"]         = results_by_time[last_idx]["time"]
        all_tracks[tid]["end_timestamp_s"]  = results_by_time[last_idx].get(
            "timestamp_s", st["last_timestamp_s"])
        all_tracks[tid]["duration_minutes"] = (
            all_tracks[tid]["end_timestamp_s"]
           - all_tracks[tid]["start_timestamp_s"]) / 60.0

    all_tracks = _collapse_transient_fragments(all_tracks, _fuse_mcs_cores, max_fragment_span_s)

    # A posteriori MOAAP qualification: keep only tracks whose peak validates
    # the MCS criteria. The full track is kept (genesis + decay).
    filtered = {}
    for tid, tr in all_tracks.items():
        hist = tr.get("history", [])
        if not hist:
            continue
        peak_area = max(h.get("area_2d_km2", 0.0) for h in hist)
        min_tb    = min(h.get("tb_min_k", 999.0) for h in hist)
        pr_vals   = [h["pr_max_mmh"] for h in hist if h.get("pr_max_mmh") is not None]
        peak_pr   = max(pr_vals) if pr_vals else None

        duration  = tr.get("duration_minutes", 0.0)
        area_ok   = mcs_min_area_bt_km2      is None or peak_area >= mcs_min_area_bt_km2
        tb_ok     = mcs_bt_overshoot_k       is None or min_tb    <= mcs_bt_overshoot_k
        pr_ok     = mcs_pr_max_mmh           is None or peak_pr   is None or peak_pr >= mcs_pr_max_mmh
        dur_ok    = mcs_min_duration_minutes is None or duration  >= mcs_min_duration_minutes

        if area_ok and tb_ok and pr_ok and dur_ok:
            filtered[tid] = tr

    return filtered