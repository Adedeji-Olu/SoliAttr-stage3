"""Turn results/*.csv into Table 2 (baselines), the stability summary and figures."""
import argparse, os, json
import numpy as np, pandas as pd
from scipy import stats

from soli_attr import config as C

ap = argparse.ArgumentParser(); ap.add_argument("--res", default="results"); a = ap.parse_args()
NAMES = {"majority": "B3 Majority class", "bayes": "B1a PCA + Gaussian Bayes", "svm": "B1b RBF-SVM (pooled spectral)",
         "cnnrnn": "B2 CNN-RNN (Wang et al. style)", "cnn": "Project CNN (dt-map)", "svm_randomsplit_LEAKY": "RBF-SVM, random split (leaky)"}
ms = lambda x: f"{np.mean(x)*100:.1f} ± {np.std(x, ddof=1)*100:.1f}" if len(x) > 1 else f"{np.mean(x)*100:.1f}"

f = os.path.join(a.res, "baselines_runs.csv")
if os.path.exists(f):
    d = pd.read_csv(f); rows = []
    per_rep = d[d.fold >= 0].groupby(["method", "repeat"])[["acc", "macro_f1"]].mean().reset_index()
    for m in [k for k in NAMES if k in set(d.method)]:
        r = per_rep[per_rep.method == m] if m != "svm_randomsplit_LEAKY" else d[d.method == m]
        rows.append(dict(Method=NAMES[m], Accuracy=ms(r.acc), MacroF1=ms(r.macro_f1.dropna()) if r.macro_f1.notna().any() else "-", Runs=len(r)))
    tab = pd.DataFrame(rows); print(tab.to_string(index=False))
    with open(os.path.join(a.res, "table2.md"), "w") as fh: fh.write(tab.to_markdown(index=False) if hasattr(tab, "to_markdown") else tab.to_string(index=False))
    with open(os.path.join(a.res, "table2.tex"), "w") as fh:
        for _, r in tab.iterrows(): fh.write(f"{r.Method} & {r.Accuracy} & {r.MacroF1} \\\\\n")
    # per-fold means (mean over the 5 repeats of each fold): the spread across folds is the honest uncertainty
    ff = d[d.fold >= 0].groupby(["method", "fold"]).acc.mean().unstack("method") * 100
    ff.round(1).to_csv(os.path.join(a.res, "table2_per_fold.csv")); print(ff.round(1).to_string())
    # paired tests at fold level (n = 4 folds; runs inside a fold share test users, so they are not independent)
    tests = []
    for m1, m2 in [("cnn", "svm"), ("cnnrnn", "svm"), ("bayes", "svm"), ("cnnrnn", "cnn")]:
        if m1 in ff.columns and m2 in ff.columns:
            diff = ff[m1] - ff[m2]; t = stats.ttest_rel(ff[m1], ff[m2])
            tests.append(dict(pair=f"{m1} vs {m2}", mean_diff_pts=float(diff.mean()), folds_positive=int((diff > 0).sum()),
                              n_folds=len(diff), t_p_folds=float(t.pvalue)))
    pd.DataFrame(tests).to_csv(os.path.join(a.res, "paired_tests.csv"), index=False); print(pd.DataFrame(tests).round(4).to_string(index=False))
    cf = os.path.join(a.res, "confusion.npz")
    if os.path.exists(cf):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        z = np.load(cf); best = "cnn" if "cnn" in z.files else z.files[0]; cm = z[best].astype(float); cm /= np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(4.6, 4)); im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xlabel("Predicted gesture"); ax.set_ylabel("True gesture"); ax.set_title(f"{NAMES.get(best, best)}: row-normalised, repeat 0")
        ax.set_xticks(range(11)); ax.set_yticks(range(11)); fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(os.path.join(a.res, "confusion.png"), dpi=200)

f = os.path.join(a.res, "stability_runs.csv")
if os.path.exists(f):
    d = pd.read_csv(f); d = d[d.kind != "skipped"]
    out = {}
    for ref in ("floor", "ref", "sanity"):
        g = d[d.kind == ref].groupby(["model", "method"])[["spearman_all", "topk_all", "ssim_all"]].agg(["mean", "std"]).round(3)
        out[ref] = g; print(f"\n== {ref} ==\n", g.to_string())
    runs = d.groupby(["fold", "repeat"]).first()
    cohort = runs[["n_test", "n_clean_correct_all3"]].describe().loc[["mean", "min", "max"]].round(1)
    print("\n== cohort (clean-correct for CNN-A, CNN-B and SVM) ==\n", cohort.to_string())
    pert = d[~d.kind.isin(["floor", "ref", "sanity"])].copy()
    pert["acc_clean"] = np.where(pert.model == "cnn", pert.acc_clean_cnn, pert.acc_clean_svm)
    pert["acc_drop_pts"] = (pert.acc_clean - pert.test_acc_perturbed) * 100
    fl = d[d.kind == "floor"].groupby(["model", "method"])["spearman_all"].mean()
    pert["floor"] = [fl.get((m, me), np.nan) for m, me in zip(pert.model, pert.method)]
    pert["below_floor"] = pert.spearman_all < pert.floor
    pert["before_accuracy_falls"] = pert.below_floor & (pert.acc_drop_pts < 2.0)
    if "calib" not in pert: pert["calib"] = "ssim"
    summ = pert.groupby(["calib", "kind", "target_ssim", "model", "method"]).agg(
        retention=("retention", "mean"), acc_drop_pts=("acc_drop_pts", "mean"), spearman=("spearman_all", "mean"),
        spearman_sd=("spearman_all", "std"), topk=("topk_all", "mean"), ssim=("ssim_all", "mean"),
        spearman_retained=("spearman_ret", "mean"), floor=("floor", "mean"), frac_runs_before_acc_falls=("before_accuracy_falls", "mean"),
        reached=("reached", "mean"), p=("p", "mean"), runs=("fold", "size")).round(3).reset_index()
    summ.to_csv(os.path.join(a.res, "stability_summary.csv"), index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    kinds = sorted(summ.kind.unique()); calibs = [c for c in ("ssim", "acc") if c in set(summ.calib)]
    fig, axs = plt.subplots(len(calibs), len(kinds), figsize=(9, 2.8 * len(calibs)), sharey=True, squeeze=False)
    for r_, calib in enumerate(calibs):
        for c_, kind in enumerate(kinds):
            ax = axs[r_, c_]
            for (model, method), g in summ[(summ.kind == kind) & (summ.calib == calib)].groupby(["model", "method"]):
                g = g.sort_values("target_ssim"); ax.errorbar(g.target_ssim, g.spearman, yerr=g.spearman_sd, marker="o", ms=3, capsize=2, label=f"{model}/{method}")
                if (model, method) in fl.index: ax.axhline(fl[(model, method)], ls=":", lw=0.8, color="grey")
            ax.set_title(f"{kind} ({calib}-matched)", fontsize=8)
            ax.set_xlabel("target input SSIM" if calib == "ssim" else "target CNN accuracy drop (pts)", fontsize=7)
            if calib == "ssim": ax.invert_xaxis()
        axs[r_, 0].set_ylabel("Spearman(clean, perturbed map)", fontsize=7)
    axs[0, -1].legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(a.res, "stability.png"), dpi=200)
    print(summ.to_string(index=False))
