"""Spearman correlation between map similarity and accuracy drop (RQ3).

Usage: python scripts/rq3_correlation.py --results results
Both quantities depend on perturbation severity, so a strong correlation does
not show that map change warns of accuracy loss earlier.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    r = pd.read_csv(os.path.join(a.results, "stability_runs.csv"))
    p = r[~r.kind.isin(["floor", "ref", "sanity", "skipped"])].copy()
    clean = np.where(p.model == "cnn", p.acc_clean_cnn, p.acc_clean_svm)
    p["drop"] = (clean - p.test_acc_perturbed) * 100
    rows = []
    for (m, me), g in p.groupby(["model", "method"]):
        rho, pv = spearmanr(g["drop"], g.spearman_all)
        h = g.groupby(["fold", "kind", "calib", "target_ssim"])[["drop", "spearman_all"]].mean().reset_index()
        rho2, pv2 = spearmanr(h["drop"], h.spearman_all)
        rows.append(dict(model=m, method=me, n_runs=len(g), rho=rho, p=pv,
                         n_fold_level=len(h), rho_fold_level=rho2, p_fold_level=pv2))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.results, "rq3_correlation.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
