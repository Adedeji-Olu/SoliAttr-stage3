"""Common representation: the Doppler x time map ("dt-map") and pooled block features.

dt-map = mean over the 4 receive channels and over the range axis of the 32x32 range-Doppler frame,
giving [T, D]. The Doppler axis is selected by config.DOPPLER_AXIS. All attributions in this project
live on the pooled 8x8 (time block x Doppler band) grid of this map, so CNN and SVM explanations
share the same physical coordinates.
"""
import numpy as np
from . import config as C


def dt_maps(rdm, idx, doppler_axis=C.DOPPLER_AXIS, chunk=256):
    """rdm: [N,4,T,32,32] (memmap ok). Returns float32 [len(idx), T, D]."""
    out = []
    range_axis = 3 + (1 - doppler_axis)      # absolute axis of the range dimension in a chunk [n,4,T,32,32]
    for s in range(0, len(idx), chunk):
        sel = np.sort(idx[s:s + chunk])
        order = np.argsort(np.argsort(idx[s:s + chunk]))
        a = np.asarray(rdm[sel], dtype=np.float32)
        a = a.mean(axis=1)                    # channels -> [n,T,32,32]
        a = a.mean(axis=range_axis - 1)       # range    -> [n,T,D]
        out.append(a[order])
    return np.concatenate(out, 0)


def block_features(dtn, grid=C.GRID):
    """dtn: normalised dt-map [N,T,D]. Returns [N, 2*grid*grid] (block mean and block std)."""
    n, t, d = dtn.shape
    b = dtn.reshape(n, grid, t // grid, grid, d // grid)
    return np.concatenate([b.mean((2, 4)).reshape(n, -1), b.std((2, 4)).reshape(n, -1)], 1)


def pool_cells(a, grid=C.GRID):
    """Mean-pool [..., T, D] to [..., grid, grid]."""
    t, d = a.shape[-2:]
    b = a.reshape(*a.shape[:-2], grid, t // grid, grid, d // grid)
    return b.mean((-3, -1))


class Norm:
    """Scalar mean/std fitted on the training split only."""
    def __init__(self):
        self.mu, self.sd = 0.0, 1.0

    def fit(self, x):
        flat = np.asarray(x[:: max(1, len(x) // 400)], dtype=np.float64)
        self.mu, self.sd = float(flat.mean()), float(flat.std() + 1e-8)
        return self

    def __call__(self, x):
        return (np.asarray(x, dtype=np.float32) - self.mu) / self.sd
