"""Comparaison non-régression : ancien pipeline vs nouveau pipeline.

Ce script exécute deux variantes sur un jeu de données synthétique identique
(un seul "amas convectif" gaussien/disque, sur 3 pas de temps) :

  1. "Ancien"  : ``src.old.fct_core_tracking.find_convective_cores_3d`` +
     ``track_mcs`` / ``track_conv_objects`` / ``track_convective_cores``.

  2. "Nouveau détection + tracking legacy" : le moteur générique
     (``src.tracking``) avec le détecteur générique ``src.tracking.detection``
     configuré par les règles de ``examples.convective_rules`` (via
     ``src.tracking.legacy_adapter.run_legacy_pipeline``), mais en
     réutilisant les MÊMES fonctions de tracking legacy que (1).

     Ce pont isole la question : "la nouvelle détection produit-elle les
     mêmes objets que l'ancienne ?" indépendamment du tracker. Une fois
     cette équivalence validée, on peut comparer séparément
     ``AdvectedIoUTracker`` (nouveau) aux fonctions ``track_*`` (ancien).

Usage
-----
    # Jeu de données synthétique (par défaut)
    conda run -n mesonh python scripts/compare_legacy_new.py

    # Fichier NetCDF réel (ex: sortie MesoNH), tous les pas de temps
    conda run -n mesonh python scripts/compare_legacy_new.py --netcdf /chemin/vers/fichier.nc

    # Limiter le nombre de pas de temps chargés (fichiers réels souvent volumineux)
    conda run -n mesonh python scripts/compare_legacy_new.py --netcdf fichier.nc --max-steps 5 --start-step 10

Le script imprime les comptes d'objets par pas de temps et par niveau pour
les deux pipelines, ainsi que le nombre de traces obtenues, et retourne un
code de sortie non nul si une divergence est détectée.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

# Permet l'exécution directe (`python scripts/compare_legacy_new.py`) en plus
# de `python -m scripts.compare_legacy_new` : sans ça, seul le dossier
# scripts/ est ajouté à sys.path et `examples`/`src` restent introuvables.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import xarray as xr

from examples.convective_rules import convective_rules
from src.tracking import TrackingLevel, detect_objects
from src.old.fct_core_tracking import (
    find_convective_cores_3d,
    track_conv_objects,
    track_convective_cores,
    track_mcs,
)
from src.tracking.legacy_adapter import run_legacy_pipeline

# --- Géométrie du jeu de données synthétique ---
NJ = NI = 500
NLEV = 100
CENTER0 = (20, 20)  # (y, x) au pas de temps 0
RADIUS_ENVELOPE = 12  # enveloppe Tb <= tb_env_k (Niveau A)
RADIUS_SEED = 7       # germe Tb <= tb_seed_k (Niveau A)
RADIUS_UPDRAFT = 3    # disque WT/hydrométéores (Niveaux B & C)
Z_LOW, Z_HIGH = 5, 15  # tranche verticale de l'ascendance
N_STEPS = 6
SHIFT_PER_STEP = 1  # déplacement en x (pixels) entre pas de temps


def _disk_mask(ny: int, nx: int, cy: float, cx: float, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0:ny, 0:nx]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def make_synthetic_dataset(t_idx: int) -> xr.Dataset:
    """Un pas de temps avec un unique amas convectif, décalé en x selon t_idx."""
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
        coords={"level": level, "level_w": level, "time": time_val},
    )


def load_datasets_from_netcdf(
    path: Path,
    max_steps: Optional[int] = None,
    start_step: int = 0,
) -> list[xr.Dataset]:
    """Charge un fichier NetCDF réel et le découpe en une liste de Dataset
    par pas de temps (un seul élément si la dimension ``time`` est absente).
    """
    with xr.open_dataset(path) as ds:
        if "time" in ds.dims:
            n_time = int(ds.sizes["time"])
            end = n_time if max_steps is None else min(n_time, start_step + max_steps)
            if start_step >= n_time:
                raise ValueError(
                    f"start_step={start_step} >= nombre de pas de temps disponibles ({n_time})"
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
            "Chemin vers un fichier NetCDF réel à utiliser à la place du jeu "
            "de données synthétique (doit exposer les variables/coords "
            "attendues par find_convective_cores_3d : WT, MRI, MRC, coords "
            "level/level_w, etc.)."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Nombre maximum de pas de temps à charger depuis le NetCDF (défaut: tous).",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Index du premier pas de temps à charger depuis le NetCDF (défaut: 0).",
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


# Paramètres des nouveaux détecteurs (valeurs scientifiques de l'ancien code).
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
    """Niveaux de détection seuls : le suivi est fait par l'ancien code."""
    rules = convective_rules(DETECTION_PARAMS)
    return [
        TrackingLevel("mcs", detect_objects, None, rules["mcs"]),
        TrackingLevel("convective_object", detect_objects, None, rules["convective_object"]),
        TrackingLevel("core", detect_objects, "convective_object", rules["core"]),
    ]


def run_new_detection_old_tracking(datasets: list[xr.Dataset]) -> dict:
    levels = build_detection_levels()
    return run_legacy_pipeline(
        datasets=datasets,
        levels=levels,
        result_key_by_level={
            "mcs": "mcs",
            "convective_object": "conv_objects",
            "core": "cores",
        },
        track_fns_by_level={
            "mcs": track_mcs,
            "conv_objects": track_conv_objects,
            "cores": track_convective_cores,
        },
    )


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
        print(f"Chargement du dataset réel : {args.netcdf}")
        datasets = load_datasets_from_netcdf(
            args.netcdf, max_steps=args.max_steps, start_step=args.start_step
        )
        print(f"{len(datasets)} pas de temps chargés")
    else:
        datasets = [make_synthetic_dataset(t) for t in range(N_STEPS)]

    old_result = run_old_pipeline(datasets)
    new_result = run_new_detection_old_tracking(datasets)

    old_counts = summarize_counts(old_result)
    new_counts = summarize_counts(new_result)
    count_cols = ["n_mcs", "n_conv_objects", "n_cores"]

    print("=== Comptes d'objets par pas de temps ===")
    print("--- Ancien pipeline ---")
    print(old_counts.to_string(index=False))
    print("--- Nouveau (détection) + tracking legacy ---")
    print(new_counts.to_string(index=False))

    counts_diff = old_counts[count_cols] - new_counts[count_cols]
    print("--- Différence (ancien - nouveau) ---")
    print(counts_diff.to_string(index=False))

    print()
    print("=== Nombre de traces par niveau ===")
    print("Ancien :", summarize_tracks(old_result))
    print("Nouveau:", summarize_tracks(new_result))

    ok = bool((counts_diff == 0).all().all())
    print()
    print("RESULTAT:", "OK - comptes identiques" if ok else "DIVERGENCE détectée")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

