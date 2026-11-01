"""Attribution stability under frequency-domain perturbations (needs a trained-from-scratch CNN + SVM per run)."""
import argparse, os, time
import numpy as np, pandas as pd

from soli_attr import config as C
from soli_attr.data import load_cache
from soli_attr.features import dt_maps
from soli_attr.splits import fold_split
from soli_attr.stability import run_stability

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache"); ap.add_argument("--out", default="results")
ap.add_argument("--repeats", type=int, default=C.REPEATS); ap.add_argument("--folds", type=int, nargs="*", default=list(C.FOLDS))
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--device", default="cpu")
ap.add_argument("--max-cohort", type=int, default=150)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rdm, meta = load_cache(a.cache); y = meta["gesture"].astype(int)
dt = dt_maps(rdm, np.arange(len(y)))
rows, t0 = [], time.time()
for fold in a.folds:
    for rep in range(a.repeats):
        tr, va, te, users = fold_split(meta, fold, rep)
        rows += run_stability(dt, y, tr, va, te, dict(fold=fold, repeat=rep), 100 * fold + rep, a.device, a.epochs, a.max_cohort)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "stability_runs.csv"), index=False)
        print(f"[{time.time()-t0:6.0f}s] fold {fold} rep {rep} done", flush=True)
