"""Attributions on the shared 8x8 (time x Doppler) grid, map normalisation and similarity metrics."""
import numpy as np
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

from . import config as C
from .features import pool_cells


def gradcam(model, X, target):
    """Grad-CAM at the last conv layer of SpecCNN (already 8x8). Returns [n,8,8]."""
    import torch
    net, out = model.net.eval(), []
    for s in range(0, len(X), 128):
        x = model._t(X[s:s + 128]); t = torch.as_tensor(target[s:s + 128]).long().to(model.device)
        logits, f = net(x, feat=True)
        g = torch.autograd.grad(logits.gather(1, t[:, None]).sum(), f)[0]
        cam = torch.relu((g.mean((2, 3), keepdim=True) * f).sum(1))
        out.append(cam.detach().cpu().numpy())
    return np.concatenate(out, 0)


def occlusion(model, X, target, grid=C.GRID, chunk=2048):
    """Occlusion on the grid: replace one cell by the training mean and record the drop in the
    target-class score. Works for any model with .scores and .mu, so CNN and SVM share a definition."""
    n, t, d = X.shape; bt, bd = t // grid, d // grid
    base = model.scores(X)[np.arange(n), target]
    out = np.zeros((n, grid, grid), np.float32)
    for i in range(grid):
        for j in range(grid):
            Xo = X.copy(); Xo[:, i * bt:(i + 1) * bt, j * bd:(j + 1) * bd] = model.mu
            sc = np.concatenate([model.scores(Xo[s:s + chunk]) for s in range(0, n, chunk)])
            out[:, i, j] = base - sc[np.arange(n), target]
    return out


def norm_map(m, mode="pos_l1"):
    """Per-sample normalisation. pos_l1: keep positive evidence, scale to sum 1 (uniform if empty)."""
    m = np.asarray(m, np.float64)
    if mode == "none":
        return m
    m = np.maximum(m, 0); s = m.sum((-2, -1), keepdims=True)
    uni = np.full_like(m, 1.0 / (m.shape[-1] * m.shape[-2]))
    return np.where(s > 1e-12, m / np.maximum(s, 1e-12), uni)


def _rank_sim(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(spearmanr(a.ravel(), b.ravel())[0])


def _topk(a, b, k=10):
    ta, tb = set(np.argsort(-a.ravel(), kind="stable")[:k]), set(np.argsort(-b.ravel(), kind="stable")[:k])
    return len(ta & tb) / k


def _ssim(a, b):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi - lo < 1e-12: return 1.0
    return float(structural_similarity(a, b, data_range=hi - lo, win_size=3))


def similarity(A, B):
    """Per-sample Spearman / top-10 overlap / SSIM between two stacks of normalised maps."""
    return {"spearman": np.array([_rank_sim(a, b) for a, b in zip(A, B)]),
            "topk": np.array([_topk(a, b) for a, b in zip(A, B)]),
            "ssim": np.array([_ssim(a, b) for a, b in zip(A, B)])}
