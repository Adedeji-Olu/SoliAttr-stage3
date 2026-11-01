"""Frequency-domain perturbations on the dt-map [N, T, D] (Doppler is the last axis).

p in [0,1] is a strength parameter. Strength is NOT comparable across kinds, so each kind is
calibrated to a target mean SSIM between clean and perturbed maps ("severity matching"),
using validation data only.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.metrics import structural_similarity

KINDS = ("band_mask", "smooth", "floor_shift")


def apply(X, kind, p):
    X = np.asarray(X, dtype=np.float32)
    if p <= 0:
        return X.copy()
    n, t, d = X.shape
    if kind == "band_mask":                       # attenuate a centred Doppler band (notch / DC-removal style filter)
        # width grows from 2 to 16 bins and gain falls from 1 to 0 as p goes 0 -> 1, so severity is continuous
        # (a hard blank of whole bins jumped straight past the mild SSIM targets: energy sits in the central bins).
        w = 2 + int(round(14 * p)); lo = (d - w) // 2
        out = X.copy()
        out[:, :, lo:lo + w] *= (1.0 - p)
        return out
    if kind == "smooth":                          # magnitude smoothing across Doppler (coarser Doppler resolution)
        return gaussian_filter1d(X, sigma=6.0 * p, axis=2, mode="nearest").astype(np.float32)
    if kind == "floor_shift":                     # raised noise floor: clip everything below a fraction of each map's peak
        # (a quantile-based floor was a no-op: most of each map is ~0, so the quantile was ~0.)
        peak = X.reshape(n, -1).max(axis=1)[:, None, None]
        return np.maximum(X, 0.5 * p * peak).astype(np.float32)
    raise ValueError(kind)


def mean_ssim(A, B):
    vals = []
    for a, b in zip(A, B):
        rng = float(a.max() - a.min()) or 1.0
        vals.append(structural_similarity(a, b, data_range=rng))
    return float(np.mean(vals))


def calibrate(kind, X_ref, target_ssim, n=48, iters=14):
    """Bisection for the p whose mean SSIM(clean, perturbed) is closest to target_ssim.
    Returns (p, achieved_ssim). If even p=1 stays above the target, returns p=1."""
    ref = X_ref[:: max(1, len(X_ref) // n)][:n]
    f = lambda p: mean_ssim(ref, apply(ref, kind, p))
    if f(1.0) > target_ssim:
        return 1.0, f(1.0)
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > target_ssim:
            lo = mid
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    return p, f(p)


def calibrate_acc(kind, model, X, y, drop_pts, iters=10):
    """Bisection for the p at which `model`'s accuracy on (X, y) falls by drop_pts points (accuracy matching).
    Returns (p, achieved_drop_pts). If even p=1 drops less than drop_pts, returns p=1."""
    base = float((model.predict(X) == y).mean())
    f = lambda p: (base - float((model.predict(apply(X, kind, p)) == y).mean())) * 100
    if f(1.0) < drop_pts:
        return 1.0, f(1.0)
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < drop_pts:
            lo = mid
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    return p, f(p)
