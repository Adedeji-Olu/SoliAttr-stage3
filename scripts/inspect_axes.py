"""Help decide which axis of the 32x32 map is Doppler: print energy profiles along both axes.
Zero-Doppler energy normally concentrates near the middle bins of the Doppler axis (static clutter)."""
import argparse, numpy as np
from soli_attr.data import load_cache

ap = argparse.ArgumentParser(); ap.add_argument("--cache", default="cache"); a = ap.parse_args()
rdm, meta = load_cache(a.cache)
sub = np.asarray(rdm[::max(1, len(meta['gesture']) // 200)], dtype=np.float32)
e = np.abs(sub).mean((0, 1, 2))            # [32,32]
np.set_printoptions(precision=3, suppress=True, linewidth=200)
print("mean |value| per ROW   (axis 0):", e.mean(1)); print("mean |value| per COLUMN(axis 1):", e.mean(0))
print("Static clutter piles up at the zero-Doppler bins. The axis whose profile has a sharp central peak/trough is Doppler.")
print("Set DOPPLER_AXIS in src/soli_attr/config.py accordingly, then re-run.")
