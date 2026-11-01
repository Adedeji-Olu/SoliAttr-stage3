"""Baselines B1 (Gaussian Bayes, RBF-SVM), B2 (CNN-RNN), B3 (majority) and the project CNN,
under the user-grouped 4-fold protocol with repeats. One row per (method, fold, repeat)."""
import argparse, json, os, time
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

from soli_attr import config as C
from soli_attr.data import load_cache
from soli_attr.features import dt_maps
from soli_attr.models import tune, build, GRIDS
from soli_attr.splits import fold_split, random_split

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache"); ap.add_argument("--out", default="results")
ap.add_argument("--repeats", type=int, default=C.REPEATS); ap.add_argument("--folds", type=int, nargs="*", default=list(C.FOLDS))
ap.add_argument("--methods", nargs="*", default=["majority", "bayes", "svm", "cnn", "cnnrnn"])
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--device", default="cpu")
ap.add_argument("--random-split-reference", action="store_true", help="also run the (leaky) random split for the SVM")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rdm, meta = load_cache(a.cache)
y = meta["gesture"].astype(int)
allidx = np.arange(len(y))
dt = dt_maps(rdm, allidx)
rows, cms = [], {}
t0 = time.time()
for fold in a.folds:
    for rep in range(a.repeats):
        tr, va, te, users = fold_split(meta, fold, rep)
        seed = 100 * fold + rep
        for name in a.methods:
            if name == "cnnrnn":
                get = lambda i: np.asarray(rdm[np.sort(i)], dtype=np.float16).transpose(0, 2, 1, 3, 4)
                Xtr, Xva, Xte = get(tr), get(va), get(te)       # [n,T,4,32,32]; tr/va/te are already sorted
            else:
                Xtr, Xva, Xte = dt[tr], dt[va], dt[te]
            m, hp, vacc = tune(name, Xtr, y[tr], Xva, y[va], seed, a.device, a.epochs)
            pred = m.predict(Xte)
            rows.append(dict(method=name, fold=fold, repeat=rep, acc=float((pred == y[te]).mean()),
                             macro_f1=float(f1_score(y[te], pred, average="macro")), val_acc=vacc, hp=json.dumps(hp),
                             n_train=len(tr), n_val=len(va), n_test=len(te), test_users="+".join(users["test"]),
                             val_users="+".join(users["val"]), seed=seed))
            cms[name] = cms.get(name, 0) + confusion_matrix(y[te], pred, labels=range(C.N_CLASSES)) * (rep == 0)
            print(f"[{time.time()-t0:6.0f}s] fold {fold} rep {rep} {name:9s} acc={rows[-1]['acc']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "baselines_runs.csv"), index=False)

if a.random_split_reference:
    for rep in range(a.repeats):
        tr, va, te = random_split(meta, rep)
        m, hp, vacc = tune("svm", dt[tr], y[tr], dt[va], y[va], rep)
        rows.append(dict(method="svm_randomsplit_LEAKY", fold=-1, repeat=rep, acc=float((m.predict(dt[te]) == y[te]).mean()),
                         macro_f1=np.nan, val_acc=vacc, hp=json.dumps(hp), n_train=len(tr), n_val=len(va), n_test=len(te),
                         test_users="mixed", val_users="mixed", seed=rep))
pd.DataFrame(rows).to_csv(os.path.join(a.out, "baselines_runs.csv"), index=False)
np.savez(os.path.join(a.out, "confusion.npz"), **cms)
