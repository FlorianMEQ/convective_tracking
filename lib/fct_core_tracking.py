"""
Détection et suivi hiérarchique 3D des systèmes convectifs (MesoNH).
=====================================================================

Trois niveaux emboîtés détectés sur un unique pas de temps xarray,
puis reliés dans le temps par trois fonctions de tracking indépendantes :

  Niveau A — MCS
    Détection  : watershed 2D sur Tb synthétique (approche TOOCAN adaptée)
                 + extension 3D connexe sur le champ hydro total.
    Tracking   : IoU 2D, patience = 4 pas de temps.

  Niveau B — Objets convectifs
    Détection  : composantes connexes 3D sur masque WT ≥ W_MIN_THRESHOLD_B
                 ET hydro > hydro_min_gkg.
    Tracking   : IoU 3D advecté + similarité morphologique (hongrois),
                 patience = 2 pas de temps.

  Niveau C — Cœurs convectifs
    Détection  : watershed 3D récursif avec graines H-Maxima adaptatives
                 (seuil W_MIN_THRESHOLD_C) à l'intérieur de chaque objet B.
    Tracking   : IoU 3D advecté + EMA morphologique + splits/merges (hongrois),
                 patience = 0 pas de temps.

API publique
------------
    find_convective_cores_3d(ds, ...)       → résultats par pas de temps
    track_mcs(results_by_time, ...)         → dict de traces Niveau A
    track_conv_objects(results_by_time, ...) → dict de traces Niveau B
    track_convective_cores(results_by_time, ...) → dict de traces Niveau C
    segment_core_recursive(core_data, ...)  → segmentation standalone Niveau C

Usage minimal
-------------
    import xarray as xr
    from lib.fct_core_tracking import (
        find_convective_cores_3d,
        track_mcs, track_conv_objects, track_convective_cores,
    )

    results = []
    for ds_t in xr.open_dataset("simulation.nc").groupby("time"):
        results.extend(find_convective_cores_3d(ds_t))

    mcs_tracks  = track_mcs(results)
    co_tracks   = track_conv_objects(results)
    core_tracks = track_convective_cores(results)

Dépendances
-----------
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
# CONSTANTES GLOBALES
# ================================================================

# --- Résolution horizontale (identique en x et y) ---
DX_METERS = 1000.0          # m/pixel
DX_KM     = DX_METERS / 1000.0  # km/pixel

# --- Seuils de vitesse verticale ---
W_MIN_THRESHOLD_B = 1.0   # Niveau B : masque binaire WT (m/s)
W_MIN_THRESHOLD_C = 4.0   # Niveau C : watershed + h-maxima adaptatif (m/s)

# --- H-Maxima adaptatif (Niveau C) ---
# h = SLOPE * (WT_max - W_MIN_THRESHOLD_C), borné dans [H_ADAPT_MIN, H_ADAPT_MAX]
H_ADAPT_SLOPE = 0.3       # pente de la relation h(WT_max)
H_ADAPT_MIN   = 1.0       # plancher de h (m/s)
H_ADAPT_MAX   = 10.0      # plafond de h (m/s)

# --- Connectivité 3D ---
CONNECTIVITY_3D = 18      # 6=face, 18=face+arête, 26=face+arête+coin

# --- Fusion de graines verticalement empilées (Niveau C) ---
MERGE_VERTICAL_STACKED_SEEDS = True
STACKED_SEEDS_XY_RADIUS      = 3   # rayon de dilatation 2D pour détecter l'empilement (pixels)


# ================================================================
# UTILITAIRES INTERNES
# ================================================================

def _z_indices_between(ds, zmin_m, zmax_m):
    """Retourne les indices des niveaux verticaux compris entre zmin_m et zmax_m."""
    z_vals = ds.coords["level"].values.astype(float)
    mask = (z_vals >= zmin_m) & (z_vals <= zmax_m)
    return np.nonzero(mask)[0]


def _compute_component_metrics(mask3d, level_vals=None):
    """
    Métriques géométriques d'un composant 3D (Niveau B ou C).

    Parameters
    ----------
    mask3d : ndarray bool, shape (nz, ny, nx)
        Masque local du composant.
    level_vals : ndarray float, optionnel
        Altitudes des niveaux en mètres. Si fourni, hauteur et volume sont
        calculés en unités physiques (km, km³). Sinon, on suppose dz = 1 km.

    Returns
    -------
    dict
        Largeurs, aires, volume et hauteur en pixels (sans level_vals)
        ou en km / km² / km³ (avec level_vals). Résolution horizontale : DX_KM.
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
    volume_km3 = float(voxels)  * DX_KM ** 2   # dz supposé 1 km si level_vals absent
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
    """Élément structurant 3D de connectivité 6, 18 ou 26."""
    if connectivity not in (6, 18, 26):
        raise ValueError(f"Connectivité 3D invalide : {connectivity}. Utiliser 6, 18 ou 26.")
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


def _merge_stacked_markers(markers, grid_3d, xy_radius=STACKED_SEEDS_XY_RADIUS):
    """
    Fusionne les graines H-Maxima verticalement empilées.

    Deux graines sont considérées empilées si leurs projections 2D (XY) se
    recouvrent après dilatation disque de rayon ``xy_radius`` pixels. La
    graine la plus intense (WT_max) absorbe l'autre.

    Parameters
    ----------
    markers : ndarray int
        Étiquettes des graines (sortie de cc_label sur h_maxima).
    grid_3d : ndarray float
        Champ WT local (même shape que markers).
    xy_radius : int
        Rayon du disque de dilatation en pixels.

    Returns
    -------
    merged : ndarray int
        Marqueurs après fusion.
    n_after : int
        Nombre de graines après fusion.
    n_merged : int
        Nombre de fusions effectuées.
    """
    labels = np.unique(markers)
    labels = labels[labels > 0]
    if len(labels) <= 1:
        return markers, len(labels), 0

    r   = int(max(0, xy_radius))
    d   = 2 * r + 1
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


def _detect_level_a(tb_2d, hydro_3d, time_hms, tb_seed_pct, tb_seed_k,
                    tb_env_k, hydro_min_gkg, mcs_min_area_km2):
    """
    Détection des MCS (Niveau A) — approche TOOCAN adaptée aux données modèle.

    Étapes :
      1. Définir l'enveloppe convective : pixels Tb ≤ tb_env_k.
      2. Définir les germes :
         - si tb_seed_pct fourni : tb_seed_pct % des pixels les plus froids
           dans l'enveloppe (adaptatif, recommandé pour données modèle) ;
         - sinon : seuil fixe Tb ≤ tb_seed_k (TOOCAN satellite, 190 K).
      3. Watershed 2D sur Tb (croissance vers les pixels plus chauds).
      4. Filtrage par aire minimale (mcs_min_area_km2).
      5. Extension 3D connexe : pour chaque MCS 2D, extraction du volume
         hydro connecté depuis le sommet des colonnes dans l'empreinte 2D.

    Tous les paramètres sont transmis sans valeurs par défaut depuis
    find_convective_cores_3d — cette fonction n'est pas destinée à être
    appelée directement.

    Parameters
    ----------
    tb_2d : ndarray float, shape (ny, nx)
        Température de brillance synthétique (K).
    hydro_3d : ndarray float, shape (nz, ny, nx)
        Champ d'hydrométéores total (g/kg).
    time_hms : str
        Heure du pas de temps au format HH:MM:SS.
    tb_seed_pct : float or None
        Percentile (dans [0, 100]) des pixels les plus froids utilisés comme
        germes. None pour basculer sur le seuil fixe tb_seed_k.
    tb_seed_k : float
        Seuil fixe de température des germes (K). Utilisé si tb_seed_pct=None.
    tb_env_k : float
        Seuil d'enveloppe convective (K).
    hydro_min_gkg : float
        Seuil minimal d'hydrométéores pour l'extension 3D (g/kg).
    mcs_min_area_km2 : float
        Aire minimale d'un MCS (km²).

    Returns
    -------
    mcs_list : list of dict
        Un dict par MCS détecté (métriques 2D + 3D + champ "_label" interne).
    ws_labels : ndarray int, shape (ny, nx)
        Labels du watershed 2D (utilisés pour associer les objets B à un MCS).
    """
    env_mask = tb_2d <= tb_env_k

    if tb_seed_pct is not None:
        cold_pixels = tb_2d[env_mask]
        seed_mask = (tb_2d <= float(np.percentile(cold_pixels, tb_seed_pct))
                     if cold_pixels.size > 0
                     else np.zeros_like(tb_2d, dtype=bool))
    else:
        seed_mask = tb_2d <= tb_seed_k

    struct_2d    = np.ones((3, 3), dtype=bool)
    seed_labels, n_seeds = cc_label(seed_mask, structure=struct_2d)

    if n_seeds == 0:
        return [], np.zeros_like(tb_2d, dtype=np.int32)

    ws_labels = watershed(tb_2d, seed_labels, mask=env_mask)

    mcs_list = []
    for lab in range(1, n_seeds + 1):
        cc_mask_2d = ws_labels == lab
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

        # Connectivité 26 : nécessaire pour relier des voxels via des voisins diagonaux
        # dans des colonnes présentant des niveaux vides intermédiaires.
        struct3d        = np.ones((3, 3, 3), dtype=bool)
        labeled_local, _ = cc_label(local_vol, structure=struct3d)

        col_bool  = local_vol[:, loc_ys, loc_xs]
        has_any   = col_bool.any(axis=0)
        top_z     = np.where(has_any,
                             nz_loc - 1 - col_bool[::-1].argmax(axis=0),
                             -1)
        valid      = top_z >= 0
        top_labels = (set(labeled_local[top_z[valid], loc_ys[valid], loc_xs[valid]].tolist())
                      - {0})

        if top_labels:
            conn_local     = np.isin(labeled_local, list(top_labels))
            connected_mask = conn_local[:, loc_ys, loc_xs]
        else:
            connected_mask = np.zeros((nz_loc, len(ys)), dtype=bool)

        n_vox_3d = int(connected_mask.sum())
        z_active  = np.where(connected_mask.any(axis=1))[0]
        tb_cc     = tb_2d[cc_mask_2d]
        mcs_list.append({
            "mcs_id":           f"t{time_hms}_mcs{lab}",
            "time_value":       time_hms,
            "area_2d_km2":      area_km2,
            "seed_area_2d_km2": float((seed_mask & cc_mask_2d).sum()),
            "n_voxels_3d":      n_vox_3d,
            "tb_min_k":         float(tb_cc.min()),
            "tb_mean_k":        float(tb_cc.mean()),
            "z_top_idx":        int(z_active.max()) if z_active.size > 0 else 0,
            "z_base_idx":       int(z_active.min()) if z_active.size > 0 else 0,
            "centroid_x":       float(xs.mean()),
            "centroid_y":       float(ys.mean()),
            "indices_2d":       list(zip(ys.tolist(), xs.tolist())),
            "_label":           int(lab),
        })

    return mcs_list, ws_labels


# ================================================================
# DÉTECTION PRINCIPALE (entrée publique)
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
    tb_np_override=None,
    w_thresh=W_MIN_THRESHOLD_B,
    min_alt_km=0,
    max_alt_km=20,
    tb_seed_k=190.0,
    tb_seed_pct=0.1,
    tb_env_k=235.0,
    hydro_min_gkg=0.01,
    mcs_min_area_km2=100.0,
):
    """
    Détection hiérarchique 3D pour un unique pas de temps.

    Enchaîne les trois niveaux de détection (A → B → C) sur le dataset
    xarray ``ds`` sélectionné pour un pas de temps. Les résultats sont
    renvoyés dans une liste à un élément, format compatible avec les
    fonctions de tracking (``track_mcs``, ``track_conv_objects``,
    ``track_convective_cores``).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset d'un pas de temps contenant les variables atmosphériques.
        Doit exposer une coordonnée ``"level"`` (altitudes en mètres) et
        une coordonnée ``"level_w"`` (altitudes des niveaux W en mètres).
    wt_var : str
        Vitesse verticale totale (m/s). Obligatoire.
    ut_var, vt_var : str
        Composantes zonale et méridionale du vent (m/s). Optionnelles
        (vent steering mis à 0 si absentes).
    mri_var, mrc_var : str
        Rapport de mélange glace primaire et eau liquide nuageuse (g/kg).
        Obligatoires (utilisées pour hydro_3d).
    mrv_var, mrs_var, mrg_var, mrr_var : str
        Vapeur, neige, graupel, pluie (g/kg). Optionnelles.
    temp_var, pres_var : str
        Température (K) et pression (Pa). Optionnelles.
    iwp_var, inprr_var, wice_var, vim_var : str
        Variables 2D intégrées (IWP kg/m², précip mm/h, etc.). Optionnelles.
    tb_var : str
        Température de brillance synthétique (K). Si absente du dataset,
        la détection Niveau A est désactivée.
    w_thresh : float
        Seuil WT pour le masque binaire Niveau B (m/s). Défaut : W_MIN_THRESHOLD_B.
    min_alt_km, max_alt_km : float
        Borne inférieure et supérieure de la colonne analysée (km).
    tb_seed_k : float
        Seuil fixe de germe Tb (K) — utilisé si tb_seed_pct=None.
    tb_seed_pct : float or None
        Percentile des pixels Tb les plus froids utilisés comme germes
        dans l'enveloppe tb_env_k (0.1 = 0.1% des pixels les plus froids).
        Mettre à None pour basculer sur tb_seed_k.
    tb_env_k : float
        Seuil d'enveloppe convective (K). Défaut : 235 K.
    hydro_min_gkg : float
        Seuil minimal de contenu en hydrométéores (g/kg). Défaut : 0.01 g/kg.
    mcs_min_area_km2 : float
        Aire minimale d'un MCS Niveau A (km²). Défaut : 100 km².

    Returns
    -------
    list of dict
        Liste à un élément contenant un dictionnaire avec les clés :
        ``"time"``, ``"timestamp_s"``,
        ``"mcs"`` (list), ``"n_mcs"``,
        ``"conv_objects"`` (list), ``"n_conv_objects"``,
        ``"cores"`` (list), ``"n_cores"``,
        ``"n_raw_cores"``, ``"n_segmented_cores"``,
        ``"n_subgroups_from_watershed"``.
    """
    results_by_time = []

    # 1. Sélection des niveaux verticaux
    z_idx      = _z_indices_between(ds, min_alt_km * 1000.0, max_alt_km * 1000.0)
    level_vals = ds["level_w"].isel(level_w=z_idx).values.astype(float)

    # 2. Extraction numpy
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
    if tb_np_override is not None:
        tb_np = tb_np_override
    else:
        tb_np = ds[tb_var].values if tb_var and tb_var in ds else None

    # 3. Champ hydro total (masque Niveau B + extension 3D Niveau A)
    hydro_3d = mri_np + mrc_np
    if mrs_np is not None: hydro_3d = hydro_3d + mrs_np
    if mrg_np is not None: hydro_3d = hydro_3d + mrg_np
    if mrr_np is not None: hydro_3d = hydro_3d + mrr_np

    # Masque binaire Niveau B : WT > seuil ET présence d'hydrométéores
    mask_np = (wt_np > w_thresh) & (hydro_3d > hydro_min_gkg)

    raw_time    = ds["time"].values
    time_hms    = str(raw_time).split("T")[-1].split(".")[0]
    timestamp_s = float(pd.Timestamp(raw_time).timestamp())

    # 3b. Détection MCS Niveau A
    mcs_list   = []
    env_labels = None
    _label_map = {}
    if tb_np is not None:
        mcs_list, env_labels = _detect_level_a(
            tb_np, hydro_3d, time_hms,
            tb_seed_pct, tb_seed_k, tb_env_k,
            hydro_min_gkg, mcs_min_area_km2,
        )
        _label_map = {m.pop("_label"): m["mcs_id"] for m in mcs_list}

        # Vent steering de chaque MCS : moyenne de colonne sur le footprint 2D
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
            "mcs": mcs_list, "conv_objects": [], "n_cores": 0, "cores": [],
        })
        return results_by_time

    # 4. Composantes connexes 3D (Niveau B)
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, nlabels = cc_label(mask_np, structure=structure)
    if nlabels == 0:
        results_by_time.append({
            "time": time_hms, "timestamp_s": timestamp_s,
            "mcs": mcs_list, "conv_objects": [], "n_cores": 0, "cores": [],
        })
        return results_by_time

    # 5. Boîtes englobantes
    slices = find_objects(labels)
    corps_list        = []
    conv_objects_list = []
    nb_raw_cores = nb_segmented_cores = nb_subgroups_from_watershed = 0

    # 6. Boucle sur chaque objet convectif (Niveau B)
    for lab, sl in enumerate(slices, 1):
        if sl is None:
            continue
        comp = (labels[sl] == lab)
        if comp.sum() < 5:
            continue
        metrics_pre = _compute_component_metrics(comp)
        if metrics_pre["width"] < 5 or metrics_pre["area"] < 15:
            continue

        z_coords, y_coords, x_coords = np.where(comp)
        z_coords += sl[0].start
        y_coords += sl[1].start
        x_coords += sl[2].start

        zmin_co, zmax_co = int(z_coords.min()), int(z_coords.max())
        level_vals_co    = level_vals[zmin_co : zmax_co + 1] if level_vals is not None else None
        metrics_co       = _compute_component_metrics(comp, level_vals_co)
        unique_2d_co     = list(set(zip(y_coords.tolist(), x_coords.tolist())))
        cx_co = float(np.mean(x_coords))
        cy_co = float(np.mean(y_coords))

        # MCS parent via centroïde
        mcs_id_co = None
        if env_labels is not None:
            cy_i    = int(np.clip(round(cy_co), 0, env_labels.shape[0] - 1))
            cx_i    = int(np.clip(round(cx_co), 0, env_labels.shape[1] - 1))
            lbl_at  = int(env_labels[cy_i, cx_i])
            if lbl_at in _label_map:
                mcs_id_co = _label_map[lbl_at]

        # Vent steering : centroïde vertical ±2 niveaux
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

        conv_obj_id  = f"t{time_hms}_co{lab}"
        indices_3d   = list(zip(z_coords.tolist(), y_coords.tolist(), x_coords.tolist()))
        conv_objects_list.append({
            "conv_obj_id":     conv_obj_id,
            "time_value":      time_hms,
            "mcs_id":          mcs_id_co,
            "area_max_dz_km2": metrics_co["area_max_dz_km2"],
            "volume_km3":      metrics_co["volume_km3"],
            "width_max_km":    float(metrics_co["width_max_dz"]),
            "width_median_km": metrics_co["width_median_km"],
            "height_km":       metrics_co["height_km"],
            "height_min_km":   float(level_vals[zmin_co]) / 1000.0,
            "height_max_km":   float(level_vals[zmax_co]) / 1000.0,
            "centroid_x":      cx_co,
            "centroid_y":      cy_co,
            "centroid_z":      z_mean_co,
            "max_speed":       float(np.nanmax(wt_np[z_coords, y_coords, x_coords])),
            "speed_99pct":     float(np.nanpercentile(wt_np[z_coords, y_coords, x_coords], 99)),
            "ut_mean":         ut_mean_co,
            "vt_mean":         vt_mean_co,
            "ut_steering":     ut_steering_co,
            "vt_steering":     vt_steering_co,
            "indices_2d":      unique_2d_co,
            "indices_3d":      indices_3d,
            "n_cores":         0,
        })

        # 7. Pré-filtrage WT > W_MIN_THRESHOLD_C + CC3D pour le Niveau C
        # On restreint au sous-volume WT > 4 m/s puis on re-étiquette les composantes
        # connexes (connectivité 26) pour que chaque cluster isolé soit traité et filtré
        # indépendamment — même comportement que l'ancien code monolithique à 4 m/s.
        wt_vals_b = wt_np[z_coords, y_coords, x_coords]
        mask_c    = wt_vals_b > W_MIN_THRESHOLD_C
        if not mask_c.any():
            conv_objects_list[-1]["n_cores"] = 0
            continue

        zc = z_coords[mask_c]; yc = y_coords[mask_c]; xc = x_coords[mask_c]
        wt_c = wt_vals_b[mask_c]

        # CC3D sur le sous-volume à 4 m/s (connectivité 26)
        zmin_c, zmax_c = int(zc.min()), int(zc.max())
        ymin_c, ymax_c = int(yc.min()), int(yc.max())
        xmin_c, xmax_c = int(xc.min()), int(xc.max())
        shape_c = (zmax_c - zmin_c + 1, ymax_c - ymin_c + 1, xmax_c - xmin_c + 1)
        vol_c   = np.zeros(shape_c, dtype=bool)
        vol_c[zc - zmin_c, yc - ymin_c, xc - xmin_c] = True
        struct26 = np.ones((3, 3, 3), dtype=bool)
        labels_c, n_clusters_c = cc_label(vol_c, structure=struct26)

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
                "group_id":     f"{conv_obj_id}_c{clab}",
                "indices_3d":   list(zip(cl_z_g.tolist(), cl_y_g.tolist(), cl_x_g.tolist())),
                "wt_3d_values": cl_wt.tolist(),
            }
            nb_raw_cores += 1
            sg = segment_core_recursive(item_cl, level_vals=level_vals, min_width_threshold=1)
            if len(sg) > 1:
                nb_segmented_cores          += 1
                nb_subgroups_from_watershed += len(sg)
            subgroups_indices.extend(sg)

        n_cores_this_obj = 0
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

            # Vent steering : centroïde vertical ±2 niveaux
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
                "group_id":       sg_data["group_id"],
                "conv_obj_id":    conv_obj_id,
                "mcs_id":         mcs_id_co,
                "time_value":     time_hms,
                "area_max_dz":    metrics["area_max_dz_km2"],
                "area_mean":      metrics["area_mean_km2"],
                "area_median":    metrics["area_median_km2"],
                "volume":         metrics["volume_km3"],
                "height_min":     height_min_km,
                "height_max":     height_max_km,
                "height":         metrics["height_km"],
                "width_base":     width_km_base,
                "width_max":      float(metrics["width_max_dz"]),
                "width_mean":     metrics["width_mean_km"],
                "width_median":   metrics["width_median_km"],
                "max_speed":      vmax,
                "mean_speed":     vmean,
                "speed_99pct":    v99,
                "mri_3d_values":  mri_vals.tolist(),
                "mri_mean":       mri_mean_val,
                "mrc_3d_values":  mrc_vals.tolist(),
                "mrc_mean":       mrc_mean_val,
                "mrv_3d_values":  mrv_vals, "mrv_mean": mrv_mean_val,
                "mrs_3d_values":  mrs_vals, "mrs_mean": mrs_mean_val,
                "mrg_3d_values":  mrg_vals, "mrg_mean": mrg_mean_val,
                "mrr_3d_values":  mrr_vals, "mrr_mean": mrr_mean_val,
                "localization": {
                    "ut": ut_mean_val, "vt": vt_mean_val,
                    "ut_steering": ut_steering, "vt_steering": vt_steering,
                    "x": x_mean, "y": y_mean, "z": z_mean,
                },
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

            corps_list.append(item)
            n_cores_this_obj += 1

        conv_objects_list[-1]["n_cores"] = n_cores_this_obj

    count = len(corps_list)
    results_by_time.append({
        "time":                       time_hms,
        "timestamp_s":                timestamp_s,
        "mcs":                        mcs_list,
        "n_mcs":                      len(mcs_list),
        "conv_objects":               conv_objects_list,
        "n_conv_objects":             len(conv_objects_list),
        "n_cores":                    count,
        "cores":                      corps_list,
        "n_raw_cores":                nb_raw_cores,
        "n_segmented_cores":          nb_segmented_cores,
        "n_subgroups_from_watershed": nb_subgroups_from_watershed,
    })
    return results_by_time


# ================================================================
# SEGMENTATION RÉCURSIVE NIVEAU C (utilisable en standalone)
# ================================================================

def segment_core_recursive(core_data, level_vals, min_width_threshold=1, depth=0):
    """
    Segmentation récursive d'un objet convectif par watershed H-Maxima adaptatif (Niveau C).

    À chaque niveau de récursion, les graines sont détectées par H-Maxima sur
    le champ WT local, puis le watershed sépare les cœurs distincts. L'appel
    s'arrête quand un sous-groupe est suffisamment petit
    (``width_median ≤ min_width_threshold``) ou quand le watershed ne produit
    qu'une seule région.

    Cette fonction peut être appelée directement sur un objet quelconque pour
    en extraire les cœurs convectifs sans passer par ``find_convective_cores_3d``.

    Parameters
    ----------
    core_data : dict
        Doit contenir :
        - ``"group_id"``    : identifiant de l'objet parent (str)
        - ``"indices_3d"``  : liste de tuples (z, y, x) en coordonnées globales
        - ``"wt_3d_values"``: liste de valeurs WT correspondantes (m/s)
    level_vals : ndarray float
        Altitudes des niveaux en mètres (coordonnée globale, pas le sous-domaine).
    min_width_threshold : float
        Largeur médiane (km) en dessous de laquelle la récursion s'arrête.
    depth : int
        Niveau de récursion courant (0 = racine). Ne pas modifier manuellement.

    Returns
    -------
    list of dict
        Un dict par cœur final avec les clés :
        ``"group_id"``, ``"indices_3d"``, ``"wt_3d_values"``.
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
            "group_id":      core_data.get("subgroup_id", core_data.get("group_id", "root")),
        }]

    structure_3d = _get_structure_3d(CONNECTIVITY_3D)
    wt_local_max = np.max(grid_3d)

    h_dynamique = H_ADAPT_SLOPE * (wt_local_max - W_MIN_THRESHOLD_C)
    h_dynamique = float(min(h_dynamique, H_ADAPT_MAX))
    h_dynamique = max(H_ADAPT_MIN, h_dynamique)

    local_maxima      = h_maxima(grid_3d, h_dynamique, footprint=structure_3d)
    markers, num_markers = cc_label(local_maxima, structure=structure_3d)

    if MERGE_VERTICAL_STACKED_SEEDS and num_markers > 1:
        markers, num_markers, _ = _merge_stacked_markers(
            markers, grid_3d, xy_radius=STACKED_SEEDS_XY_RADIUS)

    mask_global    = grid_3d > 0   # l'entrée est déjà pré-filtrée à W_MIN_THRESHOLD_C
    segmentation_3d = watershed(-grid_3d, markers, mask=mask_global)

    unique_labels = np.unique(segmentation_3d)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) <= 1:
        return [{
            "indices_3d":    indices_3d.tolist(),
            "wt_3d_values":  wt_3d_values.tolist(),
            "group_id":      core_data.get("subgroup_id", core_data.get("group_id", "root")),
        }]

    subgroups = []
    for label_id in unique_labels:
        mask     = (segmentation_3d == label_id)
        z_local, y_local, x_local = np.where(mask)
        wt_vals_sub = grid_3d[mask]

        z_global = z_local + zmin
        y_global = y_local + ymin
        x_global = x_local + xmin

        parent_id       = core_data.get("subgroup_id", core_data.get("group_id"))
        new_subgroup_id = f"{parent_id}_d{depth}_s{int(label_id)}"

        subgroup = {
            "subgroup_id":  new_subgroup_id,
            "indices_3d":   np.column_stack((z_global, y_global, x_global)).tolist(),
            "wt_3d_values": wt_vals_sub.tolist(),
        }
        subgroups.extend(
            segment_core_recursive(subgroup, level_vals, min_width_threshold, depth + 1)
        )

    return subgroups


# ================================================================
# TRACKING — UTILITAIRES COMMUNS
# ================================================================

_GRID_SIZE = np.int64(600)  # > taille max de la grille (500 dans CADDIWAF)


def _pack_indices(idx3d):
    """Encode (z, y, x) en clé int64 unique pour comparaison rapide d'ensembles."""
    G = _GRID_SIZE
    return (idx3d[:, 0].astype(np.int64) * G * G
            + idx3d[:, 1].astype(np.int64) * G
            + idx3d[:, 2].astype(np.int64))


def _pack_indices_2d(yx_list):
    """Encode une liste de (y, x) en clés int64 uniques."""
    arr = np.array(yx_list, dtype=np.int64)
    return arr[:, 0] * _GRID_SIZE + arr[:, 1]


def _advect_shift(idx3d, u_pix, v_pix):
    """Décale horizontalement les voxels d'un objet par advection (pixels)."""
    shifted = idx3d.copy()
    shifted[:, 1] += int(round(v_pix))   # y (méridional)
    shifted[:, 2] += int(round(u_pix))   # x (zonal)
    return shifted


def _advect_shift_2d(keys, u_pix, v_pix):
    """Décale des clés 2D packées (y*G+x) par advection sans dépaquetage."""
    return keys + int(round(v_pix)) * _GRID_SIZE + int(round(u_pix))


def _iou(keys_a, keys_b):
    """IoU 3D entre deux ensembles de clés voxels (numpy int64)."""
    n_inter = len(np.intersect1d(keys_a, keys_b, assume_unique=True))
    n_union = len(keys_a) + len(keys_b) - n_inter
    return n_inter / n_union if n_union > 0 else 0.0


def _iou_2d(keys_a, keys_b):
    """IoU 2D entre deux ensembles de clés footprint (numpy int64)."""
    n_inter = len(np.intersect1d(keys_a, keys_b, assume_unique=True))
    n_union = len(keys_a) + len(keys_b) - n_inter
    return n_inter / n_union if n_union > 0 else 0.0


# ================================================================
# TRACKING — NIVEAU C (cœurs convectifs)
# ================================================================

def _core_features(core):
    """
    Vecteur de features [7] pour la similarité morphologique des cœurs.
    Toutes les valeurs sont positives ; la similarité est calculée par ratio min/max
    (invariante à l'échelle).
    """
    wt_arr = np.array(core.get("wt_3d_values", [0.0]), dtype=np.float32)
    return np.array([
        core.get("area_max_dz",  0.0),
        core.get("volume",       0.0),
        core.get("speed_99pct",  core.get("max_speed", 0.0)),
        core.get("iwp_mean",     0.0),
        core.get("mrg_mean",     0.0),
        float(np.percentile(wt_arr, 50)),
        float(np.percentile(wt_arr, 75)),
    ], dtype=np.float32)


# Poids par feature (même ordre que _core_features) :
#   area_max_dz  volume  speed_99pct  iwp_mean  mrg_mean  wt_p50  wt_p75
_FEAT_WEIGHTS        = np.array([1.0, 1.0, 1.5, 2.0, 0.5, 2.0, 1.5], dtype=np.float32)
_FEAT_WEIGHTS       /= _FEAT_WEIGHTS.sum()


def _morph_sim(feat_a, feat_b):
    """Similarité pondérée [0, 1] entre deux vecteurs de features (ratio min/max)."""
    mx    = np.maximum(feat_a, feat_b)
    valid = mx > 1e-9
    if not valid.any():
        return 0.5
    safe_mx = np.where(valid, mx, 1.0)
    ratios  = np.where(valid, np.minimum(feat_a, feat_b) / safe_mx, 1.0)
    w       = np.where(valid, _FEAT_WEIGHTS, 0.0)
    w_sum   = w.sum()
    return float((ratios * w).sum() / w_sum) if w_sum > 0 else 0.5


def track_convective_cores(
    results_by_time,
    min_overlap=0.1,
    patience=0,
    dx_meters=DX_METERS,
    pre_filter_pixels=30,
    w_overlap=0.6,
    w_morph=0.4,
    gap_bridge_seconds=120.0,
    gap_bridge_max_dist_pixels=10,
    advection_error_ms=5.0,
    shape_buffer_pixels=3,
    feat_ema_alpha=0.7,
):
    """
    Tracking des cœurs convectifs (Niveau C) par IoU 3D advecté + algorithme hongrois.

    Stratégie par pas de temps :
      1. Décaler les voxels du track par advection (u_steering * dt, v_steering * dt).
      2. Calculer l'IoU 3D entre voxels advectés et voxels détectés à T+1.
      3. Combiner avec la similarité morphologique (EMA sur l'historique du track).
      4. Appariement optimal via algorithme hongrois.
      5. Détecter splits et merges sur les scores résiduels.

    Parameters
    ----------
    results_by_time : list of dict
        Sortie de ``find_convective_cores_3d`` concaténée sur plusieurs pas de temps.
    min_overlap : float
        Score composite minimum pour valider un appariement.
    patience : int
        Nombre de pas de temps sans observation avant suppression de la trace.
    dx_meters : float
        Résolution de la grille (m/pixel).
    pre_filter_pixels : int
        Cap absolu du résidu centroïde après advection (pixels).
    w_overlap, w_morph : float
        Poids de l'IoU 3D et de la similarité morphologique dans le score composite.
    gap_bridge_seconds : float
        Durée minimale d'un gap pour activer le bridge morphologique sans overlap.
    gap_bridge_max_dist_pixels : int
        Rayon centroïde max pour le bridge inter-bloc (pixels).
    advection_error_ms : float
        Erreur typique du vent steering (m/s) — détermine la limite adaptative
        du filtre de résidu : limit = min(pre_filter_pixels, err*dt/dx + shape_buffer).
    shape_buffer_pixels : int
        Marge pour le changement de forme + erreur sub-pixel (pixels).
    feat_ema_alpha : float
        Coefficient de mémoire de l'EMA des features (0 = pas de mémoire, 1 = figée).

    Returns
    -------
    dict
        Clés : identifiants de trace (``"Track_XXXX"``).
        Valeurs : dict avec ``"track_id"``, ``"start_time"``, ``"end_time"``,
        ``"duration_minutes"``, ``"history"`` (list de snapshots core),
        et optionnellement ``"events"`` (splits/merges) et ``"parent_id"``.
    """
    all_tracks    = {}
    active_tracks = {}
    track_counter = 0

    for t_idx, time_data in enumerate(results_by_time):
        time_val   = time_data["time"]
        current_ts = time_data.get("timestamp_s", t_idx * 30.0)
        cores      = time_data["cores"]

        core_items = []
        for c in cores:
            if "indices_2d" in c and c["indices_2d"]:
                ys, xs = zip(*c["indices_2d"])
                cy, cx = float(np.mean(ys)), float(np.mean(xs))
            else:
                zs, ys, xs = zip(*c["indices_3d"])
                cy, cx = float(np.mean(ys)), float(np.mean(xs))
            c["centroid_y"] = cy
            c["centroid_x"] = cx
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
                st      = active_tracks[tid]
                dt      = current_ts - st["last_timestamp_s"]
                u_pix   = st["u"] * dt / dx_meters
                v_pix   = st["v"] * dt / dx_meters
                pred_cy = st["cy"] + v_pix
                pred_cx = st["cx"] + u_pix
                shifted_keys = _pack_indices(_advect_shift(st["idx3d"], u_pix, v_pix))

                adaptive_limit    = min(pre_filter_pixels,
                                        advection_error_ms * dt / dx_meters + shape_buffer_pixels)
                adaptive_limit_sq = adaptive_limit ** 2

                for c_idx, ci in enumerate(core_items):
                    residual_sq = (ci["cy"] - pred_cy) ** 2 + (ci["cx"] - pred_cx) ** 2
                    if residual_sq > adaptive_limit_sq:
                        continue
                    iou   = _iou(shifted_keys, ci["keys"])
                    morph = _morph_sim(st["feat_ema"], ci["feat"])
                    if iou > 0.0:
                        score_matrix[r, c_idx] = w_overlap * iou + w_morph * morph
                    elif dt > gap_bridge_seconds:
                        if residual_sq <= gap_bridge_max_dist_pixels ** 2:
                            score_matrix[r, c_idx] = w_morph * morph * 0.8

            row_ind, col_ind = linear_sum_assignment(1.0 - score_matrix)
            matched_col_to_row = {}

            for r, c_idx in zip(row_ind, col_ind):
                if score_matrix[r, c_idx] < min_overlap:
                    continue
                tid  = track_ids[r]
                ci   = core_items[c_idx]
                core = ci["core"]
                loc  = core.get("localization", {})
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
                st["u"] = loc.get("ut_steering", loc.get("ut", st["u"]))
                st["v"] = loc.get("vt_steering", loc.get("vt", st["v"]))
                all_tracks[tid]["history"].append(core)

            # Merges : track non appariée qui overlap un cœur déjà pris → mort immédiate
            for r, tid in enumerate(track_ids):
                if tid in matched_tracks:
                    continue
                for c_idx in matched_cores:
                    if score_matrix[r, c_idx] >= min_overlap:
                        winner_tid = track_ids[matched_col_to_row[c_idx]]
                        all_tracks[tid].setdefault("events", []).append(
                            {"type": "merge", "time": time_val, "into": winner_tid}
                        )
                        if tid in active_tracks:
                            all_tracks[tid]["end_time"]         = time_val
                            all_tracks[tid]["end_timestamp_s"]  = current_ts
                            all_tracks[tid]["duration_minutes"] = (
                                current_ts - all_tracks[tid]["start_timestamp_s"]) / 60.0
                            del active_tracks[tid]
                        break

            # Splits : cœur non apparié qui overlap un track déjà pris
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
                    all_tracks[best_parent].setdefault("events", []).append(
                        {"type": "split", "time": time_val,
                         "child_centroid": (ci["cy"], ci["cx"])}
                    )
                    split_children[c_idx] = best_parent

        # Nouveaux cœurs non appariés → nouvelles traces
        for c_idx, ci in enumerate(core_items):
            if c_idx in matched_cores:
                continue
            track_counter += 1
            tid  = f"Track_{track_counter:04d}"
            core = ci["core"]
            loc  = core.get("localization", {})

            parent_tid = split_children.get(c_idx)
            if parent_tid and parent_tid in active_tracks:
                parent_st = active_tracks[parent_tid]
                u_init = loc.get("ut_steering", loc.get("ut", parent_st["u"]))
                v_init = loc.get("vt_steering", loc.get("vt", parent_st["v"]))
            else:
                u_init = loc.get("ut_steering", loc.get("ut", 0.0))
                v_init = loc.get("vt_steering", loc.get("vt", 0.0))

            track_data = {
                "track_id":          tid,
                "start_time":        time_val,
                "start_timestamp_s": current_ts,
                "history":           [core],
            }
            if parent_tid:
                track_data["parent_id"] = parent_tid
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

    # Finaliser les traces encore actives
    for tid, st in active_tracks.items():
        last_idx = st["last_time_idx"]
        all_tracks[tid]["end_time"]         = results_by_time[last_idx]["time"]
        all_tracks[tid]["end_timestamp_s"]  = results_by_time[last_idx].get(
            "timestamp_s", st["last_timestamp_s"])
        all_tracks[tid]["duration_minutes"] = (
            all_tracks[tid]["end_timestamp_s"]
            - all_tracks[tid]["start_timestamp_s"]) / 60.0

    return all_tracks


# ================================================================
# TRACKING — NIVEAU B (objets convectifs)
# ================================================================

def _conv_obj_features(co):
    """Vecteur de features [4] pour la similarité morphologique des objets convectifs."""
    return np.array([
        co.get("area_max_dz_km2", 0.0),
        co.get("volume_km3",      0.0),
        co.get("width_max_km",    0.0),
        co.get("speed_99pct",     co.get("max_speed", 0.0)),
    ], dtype=np.float32)


_CO_FEAT_WEIGHTS  = np.array([1.5, 1.0, 1.5, 2.0], dtype=np.float32)
_CO_FEAT_WEIGHTS /= _CO_FEAT_WEIGHTS.sum()


def _morph_sim_co(feat_a, feat_b):
    """Similarité morphologique pondérée pour les objets convectifs."""
    mx    = np.maximum(feat_a, feat_b)
    valid = mx > 1e-9
    if not valid.any():
        return 0.5
    safe_mx = np.where(valid, mx, 1.0)
    ratios  = np.where(valid, np.minimum(feat_a, feat_b) / safe_mx, 1.0)
    w       = np.where(valid, _CO_FEAT_WEIGHTS, 0.0)
    w_sum   = w.sum()
    return float((ratios * w).sum() / w_sum) if w_sum > 0 else 0.5


def track_conv_objects(
    results_by_time,
    min_overlap=0.1,
    patience=2,
    dx_meters=DX_METERS,
    pre_filter_pixels=60,
    w_overlap=0.7,
    w_morph=0.3,
    advection_error_ms=5.0,
    shape_buffer_pixels=5,
    gap_bridge_seconds=120.0,
    gap_bridge_max_dist_pixels=20,
    feat_ema_alpha=0.7,
):
    """
    Tracking des objets convectifs (Niveau B) par IoU 3D advecté + algorithme hongrois.

    Même logique que ``track_convective_cores`` (Niveau C), sans détection de
    splits/merges. Patience plus élevée (2 vs 0) : les objets convectifs sont plus
    stables et peuvent temporairement sortir du masque WT sans disparaître physiquement.

    Parameters
    ----------
    results_by_time : list of dict
        Sortie de ``find_convective_cores_3d`` concaténée sur plusieurs pas de temps.
    min_overlap : float
        Score composite minimum pour valider un appariement.
    patience : int
        Nombre de pas de temps sans observation avant suppression de la trace.
    dx_meters : float
        Résolution de la grille (m/pixel).
    pre_filter_pixels : int
        Cap absolu du résidu centroïde après advection (pixels).
    w_overlap, w_morph : float
        Poids de l'IoU 3D et de la similarité morphologique.
    advection_error_ms : float
        Erreur typique du vent steering (m/s).
    shape_buffer_pixels : int
        Marge pour le changement de forme + erreur sub-pixel (pixels).
    gap_bridge_seconds : float
        Durée minimale d'un gap pour activer le bridge morphologique.
    gap_bridge_max_dist_pixels : int
        Rayon centroïde max pour le bridge inter-bloc (pixels).
    feat_ema_alpha : float
        Coefficient de mémoire de l'EMA des features.

    Returns
    -------
    dict
        Clés : identifiants de trace (``"COTrack_XXXX"``).
        Valeurs : dict avec ``"track_id"``, ``"start_time"``, ``"end_time"``,
        ``"duration_minutes"``, ``"history"``.
    """
    all_tracks    = {}
    active_tracks = {}
    track_counter = 0

    for t_idx, time_data in enumerate(results_by_time):
        time_val   = time_data["time"]
        current_ts = time_data.get("timestamp_s", t_idx * 30.0)
        conv_objs  = time_data.get("conv_objects", [])

        items = []
        for co in conv_objs:
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
                "feat":  _conv_obj_features(co),
            })

        matched_tracks = set()
        matched_items  = set()

        if active_tracks and items:
            track_ids    = list(active_tracks.keys())
            score_matrix = np.zeros((len(track_ids), len(items)), dtype=np.float32)

            for r, tid in enumerate(track_ids):
                st    = active_tracks[tid]
                dt    = current_ts - st["last_timestamp_s"]
                u_pix = st["u"] * dt / dx_meters
                v_pix = st["v"] * dt / dx_meters
                pred_cy = st["cy"] + v_pix
                pred_cx = st["cx"] + u_pix

                adaptive_limit    = min(pre_filter_pixels,
                                        advection_error_ms * dt / dx_meters + shape_buffer_pixels)
                adaptive_limit_sq = adaptive_limit ** 2
                shifted_keys      = _pack_indices(_advect_shift(st["idx3d"], u_pix, v_pix))

                for c_idx, ci in enumerate(items):
                    residual_sq = (ci["cy"] - pred_cy) ** 2 + (ci["cx"] - pred_cx) ** 2
                    if residual_sq > adaptive_limit_sq:
                        continue
                    iou   = _iou(shifted_keys, ci["keys"])
                    morph = _morph_sim_co(st["feat_ema"], ci["feat"])
                    if iou > 0.0:
                        score_matrix[r, c_idx] = w_overlap * iou + w_morph * morph
                    elif dt > gap_bridge_seconds:
                        if residual_sq <= gap_bridge_max_dist_pixels ** 2:
                            score_matrix[r, c_idx] = w_morph * morph * 0.8

            row_ind, col_ind = linear_sum_assignment(1.0 - score_matrix)
            for r, c_idx in zip(row_ind, col_ind):
                if score_matrix[r, c_idx] < min_overlap:
                    continue
                tid = track_ids[r]
                ci  = items[c_idx]
                co  = ci["co"]
                matched_tracks.add(tid)
                matched_items.add(c_idx)

                st = active_tracks[tid]
                st["last_time_idx"]    = t_idx
                st["last_timestamp_s"] = current_ts
                st["cy"]      = ci["cy"]
                st["cx"]      = ci["cx"]
                st["idx3d"]   = ci["idx3d"]
                st["feat_ema"] = (feat_ema_alpha * st["feat_ema"]
                                  + (1.0 - feat_ema_alpha) * ci["feat"])
                st["u"] = co.get("ut_steering", co.get("ut_mean", st["u"]))
                st["v"] = co.get("vt_steering", co.get("vt_mean", st["v"]))
                all_tracks[tid]["history"].append(co)

        for c_idx, ci in enumerate(items):
            if c_idx in matched_items:
                continue
            track_counter += 1
            tid = f"COTrack_{track_counter:04d}"
            co  = ci["co"]
            all_tracks[tid] = {
                "track_id":          tid,
                "start_time":        time_val,
                "start_timestamp_s": current_ts,
                "history":           [co],
            }
            active_tracks[tid] = {
                "last_time_idx":    t_idx,
                "last_timestamp_s": current_ts,
                "cy":      ci["cy"],
                "cx":      ci["cx"],
                "idx3d":   ci["idx3d"],
                "feat_ema": ci["feat"].copy(),
                "u": co.get("ut_steering", co.get("ut_mean", 0.0)),
                "v": co.get("vt_steering", co.get("vt_mean", 0.0)),
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

    return all_tracks


# ================================================================
# TRACKING — NIVEAU A (MCS)
# ================================================================

def track_mcs(
    results_by_time,
    min_iou=0.05,
    patience=4,
    max_speed_ms=30.0,
):
    """
    Tracking des MCS (Niveau A) par IoU 2D advecté + algorithme hongrois.

    Le footprint de chaque track est décalé par son vent steering
    (``ut_steering``, ``vt_steering`` stockés dans le dict MCS) avant le
    calcul de l'IoU. Un appariement est rejeté si la vitesse de déplacement
    du centroïde dépasse ``max_speed_ms`` (contrainte physique).

    Splits et merges sont détectés et enregistrés dans ``"events"`` :
    - Split  : un MCS donne naissance à plusieurs enfants. La track parent se
               termine au moment du split ; chaque enfant hérite ``"parent_id"``.
    - Merge  : plusieurs MCS convergent en un seul. Les tracks absorbées se
               terminent immédiatement avec un event ``"merge"``.

    Parameters
    ----------
    results_by_time : list of dict
        Sortie de ``find_convective_cores_3d`` concaténée sur plusieurs pas de temps.
    min_iou : float
        IoU 2D minimum pour valider un appariement.
    patience : int
        Nombre de pas de temps sans observation avant suppression de la trace.
    max_speed_ms : float
        Vitesse maximale physique du centroïde MCS (m/s). Tout appariement
        impliquant une vitesse supérieure est rejeté (score forcé à 0).

    Returns
    -------
    dict
        Clés : identifiants de trace (``"MCSTrack_XXXX"``).
        Valeurs : dict avec ``"track_id"``, ``"start_time"``, ``"end_time"``,
        ``"duration_minutes"``, ``"history"``,
        et optionnellement ``"events"`` et ``"parent_id"``.
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

                # Advection du footprint et du centroïde de la track
                u_pix    = st["u"] * dt / DX_METERS
                v_pix    = st["v"] * dt / DX_METERS
                adv_keys = _advect_shift_2d(st["keys"], u_pix, v_pix)
                adv_cx   = st["cx"] + u_pix
                adv_cy   = st["cy"] + v_pix

                # Distance résiduelle max depuis le centroïde advecté
                max_dist_px = max_speed_ms * dt / DX_METERS

                for c_idx, ci in enumerate(items):
                    dist_px = np.hypot(ci["cx"] - adv_cx, ci["cy"] - adv_cy)
                    if dist_px > max_dist_px:
                        continue
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

            # Merges : track non appariée qui overlap un MCS déjà pris → mort immédiate
            for r, tid in enumerate(track_ids):
                if tid in matched_tracks:
                    continue
                for c_idx in matched_items:
                    if score_matrix[r, c_idx] >= min_iou:
                        winner_tid = track_ids[matched_col_to_row[c_idx]]
                        all_tracks[tid].setdefault("events", []).append(
                            {"type": "merge", "time": time_val, "into": winner_tid}
                        )
                        if tid in active_tracks:
                            all_tracks[tid]["end_time"]         = time_val
                            all_tracks[tid]["end_timestamp_s"]  = current_ts
                            all_tracks[tid]["duration_minutes"] = (
                                current_ts - all_tracks[tid]["start_timestamp_s"]) / 60.0
                            del active_tracks[tid]
                        break

            # Splits : MCS non apparié qui overlap une track déjà prise
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
                    all_tracks[best_parent].setdefault("events", []).append(
                        {"type": "split", "time": time_val,
                         "child_centroid": (ci["cy"], ci["cx"])}
                    )
                    split_children[c_idx] = best_parent

        # Nouveaux MCS non appariés → nouvelles traces
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

    return all_tracks
test2