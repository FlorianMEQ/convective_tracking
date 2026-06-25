from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.ndimage import label as cc_label


def z_indices_between(level_values: np.ndarray, zmin_m: float, zmax_m: float) -> np.ndarray:
    """Return indices where level values are inside [zmin_m, zmax_m]."""
    level_values = level_values.astype(float)
    return np.nonzero((level_values >= zmin_m) & (level_values <= zmax_m))[0]


def get_structure_3d(connectivity: int = 18) -> np.ndarray:
    """Build a 3D connectivity structure for 6, 18 or 26-neighborhood."""
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be one of: 6, 18, 26")

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


def connected_components_3d(mask: np.ndarray, connectivity: int = 18):
    """Label 3D connected components on a boolean mask."""
    structure = get_structure_3d(connectivity)
    return cc_label(mask.astype(bool), structure=structure)


def compute_component_metrics(
    mask3d: np.ndarray,
    level_vals: Optional[np.ndarray] = None,
    dx_km: float = 1.0,
) -> Dict[str, float]:
    """Compute geometric metrics for one 3D component."""
    voxels = int(mask3d.sum())
    proj = mask3d.any(axis=0)
    area_px = int(proj.sum())

    if area_px == 0:
        return {
            "voxels": 0,
            "area_px": 0,
            "area_km2": 0.0,
            "width_px": 0,
            "width_km": 0.0,
            "height_levels": 0,
            "height_km": 0.0,
            "volume_km3": 0.0,
        }

    ys, xs = np.where(proj)
    width_px = int(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))

    z_proj = mask3d.any(axis=(1, 2))
    z_idx = np.where(z_proj)[0]
    height_levels = int(z_idx.max() - z_idx.min() + 1) if z_idx.size else 0

    area_km2 = area_px * dx_km * dx_km
    width_km = width_px * dx_km

    if level_vals is not None and len(level_vals) > 1 and z_idx.size:
        dz_km = np.diff(level_vals, append=level_vals[-1]) / 1000.0
        height_km = float(dz_km[z_idx].sum())
        count_z = mask3d.sum(axis=(1, 2))
        volume_km3 = float((count_z * dz_km).sum()) * dx_km * dx_km
    else:
        height_km = float(height_levels)
        volume_km3 = voxels * dx_km * dx_km

    return {
        "voxels": float(voxels),
        "area_px": float(area_px),
        "area_km2": float(area_km2),
        "width_px": float(width_px),
        "width_km": float(width_km),
        "height_levels": float(height_levels),
        "height_km": float(height_km),
        "volume_km3": float(volume_km3),
    }


def iou_from_sets(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two arrays of unique integer keys."""
    inter = len(np.intersect1d(a, b, assume_unique=False))
    union = len(a) + len(b) - inter
    return float(inter / union) if union > 0 else 0.0
