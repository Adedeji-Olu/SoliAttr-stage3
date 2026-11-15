"""Stage 4 experiments (see src/soli_attr/stage4.py). Quick parts first, so partial results exist if a session dies.
Outputs (in --out): stage4_curve.csv, stage4_svmbudget.csv, stage4_stability_runs.csv, stage4_faithfulness.csv,
stage4_perclass.csv, stage4_grid.csv."""
import argparse, os, time
import numpy as np, pandas as pd

from soli_attr import config as C
from soli_attr.data import load_cache
from soli_attr.features import dt_maps
from soli_attr.splits import fold_split
from soli_attr.stage4 import run_stability2, learning_curve, svm_budget

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache"); ap.add_argument("--out", default="results")
ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--folds", type=int, nargs="*", default=list(C.FOLDS))
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--device", default="cpu")
ap.add_argument("--max-cohort", type=int, default=150)
ap.add_argument("--parts", nargs="*", default=["curve", "svmbudget", "stability2"])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rdm, meta = load_cache(a.cache); y = meta["gesture"].astype(int)
dt = dt_maps(rdm, np.arange(len(y)))
save = lambda rows, name: pd.DataFrame(rows).to_csv(os.path.join(a.out, name), index=False)
t0 = time.time()
tag = lambda msg: print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)

for part in a.parts:
    acc = {"stab": [], "faith": [], "pc": [], "grid": [], "curve": [], "svm": []}
    for fold in a.folds:
        for rep in range(a.repeats):
            tr, va, te, users = fold_split(meta, fold, rep); seed = 100 * fold + rep; base = dict(fold=fold, repeat=rep)
            if part == "curve":
                acc["curve"] += [dict(base, **r) for r in learning_curve(dt, y, tr, va, te, seed, a.device, a.epochs)]
                save(acc["curve"], "stage4_curve.csv")
            elif part == "svmbudget":
                acc["svm"].append(dict(base, **svm_budget(dt, y, tr, va, te, seed))); save(acc["svm"], "stage4_svmbudget.csv")
            elif part == "stability2":
                r, f, p, g = run_stability2(dt, y, tr, va, te, base, seed, a.device, a.epochs, a.max_cohort)
                acc["stab"] += r; acc["faith"] += f; acc["pc"] += p; acc["grid"] += g
                for rows, name in ((acc["stab"], "stage4_stability_runs.csv"), (acc["faith"], "stage4_faithfulness.csv"),
                                   (acc["pc"], "stage4_perclass.csv"), (acc["grid"], "stage4_grid.csv")):
                    save(rows, name)
            tag(f"{part}: fold {fold} rep {rep} done")
