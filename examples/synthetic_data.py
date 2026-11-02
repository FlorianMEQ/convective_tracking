import numpy as np
import xarray as xr

NZ, NY, NX = 20, 60, 100  
DZ_M = 500.0                
DX_KM = 1.0                 
DT_S = 300.0               
DRIFT_PX = 2                


def _blob(z, y, x, center, radius, amplitude):
    cz, cy, cx = center
    rz, ry, rx = radius
    r2 = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
    return amplitude * np.exp(-r2)


def make_frame(t):
    z, y, x = np.meshgrid(np.arange(NZ), np.arange(NY), np.arange(NX), indexing="ij")
    shift = DRIFT_PX * t

    w = np.zeros((NZ, NY, NX))
    hydro = np.zeros((NZ, NY, NX))

    # --- MCS A:------------------------------
    ax, ay = 28 + shift, 30
    hydro += _blob(z, y, x, (10, ay, ax), (6, 11, 14), 1.0)
    w += _blob(z, y, x, (9, ay, ax), (5, 5, 11), 2.0)
    gap = max(12 - 2 * t, 0)
    w += _blob(z, y, x, (9, ay, ax - gap / 2), (4, 3, 3), 9.0)
    w += _blob(z, y, x, (9, ay, ax + gap / 2), (4, 3, 3), 7.0)
    w += _blob(z, y, x, (5, ay + 2, ax - 10), (3, 3, 3), -4.0)
    hydro += _blob(z, y, x, (10, 32, 84), (5, 6, 6), 0.8)
    w += _blob(z, y, x, (10, 32, 84), (3, 2.5, 2.5), 6.0)
    w += _blob(z, y, x, (6, 10, 84), (3, 2.5, 2.5), 6.0)

    levels = (np.arange(NZ) + 0.5) * DZ_M
    levels_w = np.arange(NZ) * DZ_M
    return xr.Dataset(
        {
            "W": (("level", "y", "x"), w.astype(np.float32)),
            "MRC": (("level", "y", "x"), np.where(z < 10, hydro, 0.0).astype(np.float32)),
            "MRI": (("level", "y", "x"), np.where(z >= 10, hydro, 0.0).astype(np.float32)),
        },
        coords={"level": levels, "level_w": levels_w},
    )


def make_frames(n_times: int = 8):
    frames = [make_frame(t) for t in range(n_times)]
    labels = [f"12:{5 * t:02d}" for t in range(n_times)]  # 12:00, 12:05, ...
    timestamps = [t * DT_S for t in range(n_times)]
    return frames, labels, timestamps
