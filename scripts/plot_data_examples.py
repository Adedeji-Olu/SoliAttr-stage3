"""Data figure: all 32 range-Doppler frames of one sequence (channel 0), the four channels at one frame, and the
Doppler-time map the models read. Usage: python scripts/plot_data_examples.py --cache cache --gesture 5 --out results/figs/data_overview.png"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from soli_attr.data import load_cache
from soli_attr.features import dt_maps

ap = argparse.ArgumentParser(); ap.add_argument("--cache", default="cache"); ap.add_argument("--gesture", type=int, default=5)
ap.add_argument("--frame", type=int, default=16); ap.add_argument("--out", default="results/figs/data_overview.png"); a = ap.parse_args()
rdm, meta = load_cache(a.cache); y = meta["gesture"].astype(int)
i = int(np.where(y == a.gesture)[0][0]); seq = np.asarray(rdm[i, 0], dtype=float)[:, :16]; vmax = seq.max()
dt = dt_maps(rdm, np.array([i]))[0]
fig = plt.figure(figsize=(12, 7.6))
mont = fig.add_axes([0.02, 0.5, 0.96, 0.46]); big = np.full((4 * 17, 8 * 33), np.nan)
for t in range(32):
    r, c = divmod(t, 8); big[r * 17:r * 17 + 16, c * 33:c * 33 + 32] = seq[t]
mont.imshow(big, cmap="viridis", vmin=0, vmax=vmax); mont.axis("off"); mont.set_title("(a) 32 frames, channel 0 (range rows 0-15 x Doppler bins 0-31)", loc="left")
for c in range(4):
    ax = fig.add_axes([0.02 + c * 0.17, 0.06, 0.15, 0.34]); ax.imshow(np.asarray(rdm[i, c, a.frame], dtype=float), cmap="viridis"); ax.axis("off"); ax.set_title(f"channel {c}, frame {a.frame}", fontsize=8)
fig.text(0.02, 0.42, "(b) the four receive channels at one frame", fontsize=10)
ax = fig.add_axes([0.74, 0.07, 0.24, 0.32]); ax.imshow(dt, cmap="magma", aspect="auto"); ax.set_xlabel("Doppler bin"); ax.set_ylabel("frame"); ax.set_title("(c) Doppler-time map", fontsize=10)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); fig.savefig(a.out, dpi=200, bbox_inches="tight")
