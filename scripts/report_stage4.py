"""Stage 4 tables and figures from results/stage4_*.csv (+ Stage 3 baselines and confusion matrices).
Writes results/stage4_tables/*.tex (booktabs) and results/figs/*.pdf (vector), plus results/stage4_numbers.json."""
import argparse, json, os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--res", default="results"); a = ap.parse_args()
R = a.res; T = os.path.join(R, "stage4_tables"); F = os.path.join(R, "figs"); os.makedirs(T, exist_ok=True); os.makedirs(F, exist_ok=True)
NAVY, ORANGE, TEAL, GREY, PURPLE = "#1E2761", "#DD8452", "#2A9D8F", "#8A8FA8", "#7B4F9D"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "serif"})
KINDS = ["band_mask", "smooth", "floor_shift", "snr_noise", "clutter"]
KNAME = {"band_mask": "Band mask", "smooth": "Smoothing", "floor_shift": "Floor shift", "snr_noise": "Noise (SNR)", "clutter": "Clutter"}
MAPS = [("cnn", "gradcam", "CNN Grad-CAM", NAVY), ("cnn", "occlusion", "CNN occlusion", ORANGE), ("cnn", "intgrad", "CNN Int. Gradients", PURPLE),
        ("svm", "occlusion_margin", "SVM occlusion (margin)", TEAL)]
NUM = {}

r = pd.read_csv(os.path.join(R, "stage4_stability_runs.csv"))
floor = r[r.kind == "floor"].groupby(["model", "method"])
fl = floor[["spearman_all", "topk_all", "ssim_all"]].mean(); fl_sd = floor.spearman_all.std()
P = r[~r.kind.isin(["floor", "ref", "sanity", "skipped"])].copy()
P["clean"] = np.where(P.model == "cnn", P.acc_clean_cnn, P.acc_clean_svm)
P["dacc"] = (P.clean - P.test_acc_perturbed) * 100                       # the explained model's own accuracy drop (points)
tex = lambda name, s: open(os.path.join(T, name), "w").write(s)
f3 = lambda x: f"{x:.3f}"
f1 = lambda x: f"{x:.1f}"

# ---- Table A: hypothesis H for every map (floor, conditions below floor, H met)
rows = []
for model, method, lab, _ in MAPS + [("svm", "occlusion_vote", "SVM occlusion (votes)", None)]:
    d = P[(P.model == model) & (P.method == method)]; fv = fl.loc[(model, method), "spearman_all"]
    below = d.spearman_all < fv; H = below & (d.dacc < 2)
    rows.append((lab, f"{fv:.3f} $\\pm$ {fl_sd[(model, method)]:.3f}", f"{int(below.sum())} / {len(d)}", f"{int(H.sum())} / {len(d)}"))
    NUM[f"H_{model}_{method}"] = [int(H.sum()), len(d)]; NUM[f"below_{model}_{method}"] = int(below.sum())
tex("tab_h.tex", "\\begin{tabular}{@{}lccc@{}}\\toprule\nMap & seed floor $\\rho$ & conditions below floor & H met (drop $<2$ pts)\\\\\\midrule\n" +
    "\n".join(" & ".join(x) + " \\\\" for x in rows) + "\n\\bottomrule\\end{tabular}\n")

# ---- Table B: strength calibrated to a 10-point CNN validation drop, five perturbations
A10 = P[(P.calib == "acc") & (P.target_ssim == 10.0)]
rows = []
for k in KINDS:
    d = A10[A10.kind == k]; g = lambda m, me: d[(d.model == m) & (d.method == me)]
    rows.append((KNAME[k], f1(g("cnn", "gradcam").dacc.mean()), f3(g("cnn", "gradcam").spearman_all.mean()), f3(g("cnn", "occlusion").spearman_all.mean()),
                 f3(g("cnn", "intgrad").spearman_all.mean()), f1(g("svm", "occlusion_margin").dacc.mean()), f3(g("svm", "occlusion_margin").spearman_all.mean()),
                 f"{100 * g('svm', 'occlusion_margin').retention.mean():.0f}\\%"))
tex("tab_10pt.tex", "\\begin{tabular}{@{}lccccccc@{}}\\toprule\n & CNN drop & \\multicolumn{3}{c}{CNN $\\rho$} & SVM drop & SVM $\\rho$ & SVM kept\\\\\n"
    " & (pts) & Grad-CAM & occl. & IG & (pts) & (margin) & \\\\\\midrule\n" + "\n".join(" & ".join(x) + " \\\\" for x in rows) + "\n\\bottomrule\\end{tabular}\n")
for k in KINDS:
    d = A10[A10.kind == k]
    NUM[f"rho10_{k}"] = {me: float(d[(d.method == me)].spearman_all.mean()) for me in ("gradcam", "occlusion", "intgrad", "occlusion_margin")}

# ---- Table C: where do maps leave their floor? (SSIM-matched; own model's mean accuracy drop at the last level above / first below)
S = P[P.calib == "ssim"].groupby(["kind", "model", "method", "target_ssim"]).agg(rho=("spearman_all", "mean"), dacc=("dacc", "mean")).reset_index()
rows = []
for k in KINDS:
    row = [KNAME[k]]
    for model, method, _, _ in MAPS:
        d = S[(S.kind == k) & (S.model == model) & (S.method == method)].sort_values("target_ssim", ascending=False); fv = fl.loc[(model, method), "spearman_all"]
        below = d[d.rho < fv]; above = d[d.rho >= fv]
        row.append("not crossed" if below.empty else f"{(above.dacc.max() if len(above) else 0):.0f}--{below.dacc.min():.0f}")
        NUM[f"cross_{k}_{method}_{model}"] = row[-1]
    rows.append(row)
tex("tab_cross.tex", "\\begin{tabular}{@{}lcccc@{}}\\toprule\nPerturbation & CNN Grad-CAM & CNN occlusion & CNN IG & SVM (margin)\\\\\\midrule\n" +
    "\n".join(" & ".join(x) + " \\\\" for x in rows) + "\n\\bottomrule\\end{tabular}\n")

# ---- Table D: ablation of the measurement protocol (H cases of 600 per map)
rows = []
variants = [("Full protocol (Spearman, all cohort)", "spearman_all", "spearman_all"), ("Similarity: top-10 overlap", "topk_all", "topk_all"),
            ("Similarity: SSIM", "ssim_all", "ssim_all"), ("Cohort: still-correct sequences only", "spearman_ret", "spearman_all")]
for name, col, fcol in variants:
    row = [name]
    for model, method in (("cnn", "gradcam"), ("cnn", "occlusion"), ("cnn", "intgrad"), ("svm", "occlusion_margin")):
        d = P[(P.model == model) & (P.method == method)]; fv = fl.loc[(model, method), fcol]
        row.append(f"{int(((d[col] < fv) & (d.dacc < 2)).sum())}")
    rows.append(row)
d = P[(P.model == "cnn") & (P.method == "gradcam")]
rows.append(["No floor (fixed threshold $\\rho<0.9$)"] + [f"{int(((P[(P.model == m) & (P.method == me)].spearman_all < 0.9) & (P[(P.model == m) & (P.method == me)].dacc < 2)).sum())}"
                                                       for m, me in (("cnn", "gradcam"), ("cnn", "occlusion"), ("cnn", "intgrad"), ("svm", "occlusion_margin"))])
tex("tab_ablate.tex", "\\begin{tabular}{@{}lcccc@{}}\\toprule\nProtocol variant & CNN Grad-CAM & CNN occl. & CNN IG & SVM margin\\\\\\midrule\n" +
    "\n".join(" & ".join(x) + " \\\\" for x in rows) + "\n\\bottomrule\\end{tabular}\n")
NUM["ablate"] = rows

# ---- RQ3: similarity vs accuracy drop, overall and within each condition (severity removed)
rows = []
Pc = P.copy(); Pc["cell"] = Pc.kind + "|" + Pc.calib + "|" + Pc.target_ssim.astype(str)
for model, method, lab, _ in MAPS + [("svm", "occlusion_vote", "SVM occlusion (votes)", None)]:
    d = Pc[(Pc.model == model) & (Pc.method == method)].copy()
    o = stats.spearmanr(d.dacc, d.spearman_all)
    d["dd"] = d.dacc - d.groupby("cell").dacc.transform("mean"); d["ds"] = d.spearman_all - d.groupby("cell").spearman_all.transform("mean")
    w = stats.pearsonr(d.dd, d.ds)
    rows.append((lab, f"{o.statistic:.2f}", f"{w.statistic:.2f}", "$<0.001$" if w.pvalue < 1e-3 else f"{w.pvalue:.3f}"))
    NUM[f"rq3_{model}_{method}"] = [float(o.statistic), float(w.statistic), float(w.pvalue)]
tex("tab_rq3.tex", "\\begin{tabular}{@{}lccc@{}}\\toprule\nMap & overall Spearman & within-condition $r$ & $p$\\\\\\midrule\n" +
    "\n".join(" & ".join(x) + " \\\\" for x in rows) + "\n\\bottomrule\\end{tabular}\n")

# ---- Faithfulness (deletion)
fa = pd.read_csv(os.path.join(R, "stage4_faithfulness.csv")); rows = []
for mp, lab in (("cnn_gc", "CNN Grad-CAM"), ("cnn_occ", "CNN occlusion"), ("cnn_ig", "CNN Int. Gradients"), ("svm_occ_margin", "SVM occlusion (margin)")):
    row = [lab]
    for k in (4, 8):
        d = fa[(fa["map"] == mp) & (fa.k == k)].groupby("fold")[["acc_top", "acc_random", "acc_bottom"]].mean()
        p = stats.ttest_rel(d.acc_top, d.acc_random).pvalue
        row += [f"{100 * d.acc_top.mean():.0f}", f"{100 * d.acc_random.mean():.0f}", f"{100 * d.acc_bottom.mean():.0f}", "$<0.01$" if p < 0.01 else f"{p:.3f}"]
        NUM[f"faith_{mp}_{k}"] = [float(d.acc_top.mean()), float(d.acc_random.mean()), float(d.acc_bottom.mean()), float(p)]
    rows.append(row)
tex("tab_faith.tex", "\\begin{tabular}{@{}lcccccccc@{}}\\toprule\n & \\multicolumn{4}{c}{delete 4 of 64 cells} & \\multicolumn{4}{c}{delete 8 of 64 cells}\\\\\n"
    "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}\nMap & top & random & bottom & $p$ & top & random & bottom & $p$\\\\\\midrule\n" + "\n".join(" & ".join(x) + " \\\\" for x in rows) + "\n\\bottomrule\\end{tabular}\n")

# ---- Sensitivity: grid size, training size, SVM budget
gd = pd.read_csv(os.path.join(R, "stage4_grid.csv")); rows = []
main_occ = {m: A10[(A10.model == m) & (A10.method == me)].groupby("kind").spearman_all.mean().mean() for m, me in (("cnn", "occlusion"), ("svm", "occlusion_margin"))}
for model, lab in (("cnn", "CNN occlusion"), ("svm", "SVM occlusion (margin)")):
    for g in (4, 8, 16):
        if g == 8:
            fv = fl.loc[(model, "occlusion" if model == "cnn" else "occlusion_margin"), "spearman_all"]; rho = main_occ[model]
        else:
            d = gd[(gd.model == model) & (gd.grid == g)]; fv = d[d.kind == "floor"].groupby("fold").spearman.mean().mean(); rho = d[d.kind != "floor"].groupby("kind").spearman.mean().mean()
        rows.append((lab if g == 4 else "", f"{g}$\\times${g}", f3(fv), f3(rho)))
        NUM[f"grid_{model}_{g}"] = [float(fv), float(rho)]
tex("tab_grid.tex", "\\begin{tabular}{@{}llcc@{}}\\toprule\nMap & grid & seed floor $\\rho$ & mean $\\rho$ at 10-pt strength\\\\\\midrule\n" + "\n".join(" & ".join(x) + " \\\\" for x in rows) + "\n\\bottomrule\\end{tabular}\n")

cu = pd.read_csv(os.path.join(R, "stage4_curve.csv")); cs = cu.groupby(["method", "frac"]).acc.agg(["mean", "std"]).reset_index()
rows = []
for m, lab in (("bayes", "PCA + Gaussian Bayes"), ("svm", "RBF-SVM"), ("cnn", "CNN")):
    rows.append([lab] + [f"{100 * cs[(cs.method == m) & (cs.frac == f)]['mean'].iloc[0]:.1f} $\\pm$ {100 * cs[(cs.method == m) & (cs.frac == f)]['std'].iloc[0]:.1f}" for f in (0.25, 0.5, 1.0)])
    for f in (0.25, 0.5, 1.0): NUM[f"curve_{m}_{f}"] = float(cs[(cs.method == m) & (cs.frac == f)]["mean"].iloc[0])
tex("tab_curve.tex", "\\begin{tabular}{@{}lccc@{}}\\toprule\nMethod & 25\\% of training data & 50\\% & 100\\%\\\\\\midrule\n" + "\n".join(" & ".join(x) + " \\\\" for x in rows) + "\n\\bottomrule\\end{tabular}\n")

sb = pd.read_csv(os.path.join(R, "stage4_svmbudget.csv")); b = pd.read_csv(os.path.join(R, "baselines_runs.csv"))
b4 = b[(b.method == "svm") & (b["repeat"] < 3)].groupby("fold").acc.mean(); b30 = sb.groupby("fold").acc.mean()
NUM["svm_budget"] = [float(b4.mean()), float(b30.mean()), float(stats.ttest_rel(b30, b4).pvalue), float(sb.hp.str.contains('"gamma": 0.002').mean())]
cnn3 = b[(b.method == "cnn") & (b["repeat"] < 3)].groupby("fold").acc.mean(); NUM["cnn_vs_svm30"] = [float(cnn3.mean()), float((cnn3 - b30).mean()), float(stats.ttest_rel(cnn3, b30).pvalue)]

# ---- Error analysis numbers
cm = np.load(os.path.join(R, "confusion.npz")); NUM["recall"] = {}
for m in ("svm", "cnn", "cnnrnn"):
    C = cm[m]; NUM["recall"][m] = (C.diagonal() / C.sum(1)).round(3).tolist()
C = cm["cnn"]; off = C.copy(); np.fill_diagonal(off, 0)
NUM["top_confusions"] = [[int(i), int(j), int(off[i, j]), float(off[i, j] / C[i].sum())] for i, j in
                         [np.unravel_index(f, off.shape) for f in np.argsort(-off.ravel())[:4]]]
pc = pd.read_csv(os.path.join(R, "stage4_perclass.csv")); p10 = pc[pc["drop"] == 10.0].groupby(["map", "gesture"]).spearman.mean().unstack("map")
rec = C.diagonal() / C.sum(1)
NUM["perclass_vs_recall"] = {m: [float(stats.spearmanr(p10[m], rec).statistic), float(stats.spearmanr(p10[m], rec).pvalue)] for m in p10.columns}

# ================= figures (vector PDF)
# Fig A: accuracy-matched similarity, five perturbations
fig, ax = plt.subplots(1, 5, figsize=(7.4, 2.15), sharey=True)
for a_, k in zip(ax, KINDS):
    for model, method, lab, col in MAPS:
        d = P[(P.calib == "acc") & (P.kind == k) & (P.model == model) & (P.method == method)].groupby("target_ssim").spearman_all.agg(["mean", "std"])
        a_.errorbar(d.index, d["mean"], d["std"], color=col, marker="o", ms=2.5, lw=1, capsize=1.5, label=lab)
        a_.axhline(fl.loc[(model, method), "spearman_all"], color=col, ls=":", lw=0.8)
    a_.set_title(KNAME[k]); a_.set_xlabel("target CNN drop (pts)"); a_.set_xticks([1, 2, 5, 10]); a_.set_ylim(0, 1.03)
ax[0].set_ylabel("Spearman $\\rho$")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=6, frameon=False)
fig.tight_layout(pad=0.4, rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(F, "stability5.pdf")); plt.close(fig)

# Fig B: deletion curves
fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.5), sharey=True)
for a_, mset, title in ((ax[0], [m for m in MAPS if m[0] == "cnn"], "CNN"), (ax[1], [m for m in MAPS if m[0] == "svm"], "SVM")):
    for model, method, lab, col in mset:
        key = {"gradcam": "cnn_gc", "occlusion": "cnn_occ", "intgrad": "cnn_ig", "occlusion_margin": "svm_occ_margin"}[method]
        d = fa[fa["map"] == key].groupby("k")[["acc_top", "acc_random", "acc_bottom"]].mean()
        a_.plot(d.index, 100 * d.acc_top, color=col, marker="o", ms=3, lw=1.2, label=lab + " (top-ranked)")
    k0 = {"CNN": "cnn_gc", "SVM": "svm_occ_margin"}[title]; d = fa[fa["map"] == k0].groupby("k")[["acc_random", "acc_bottom"]].mean()
    a_.plot(d.index, 100 * d.acc_random, color=GREY, ls="--", lw=1, label="random cells"); a_.plot(d.index, 100 * d.acc_bottom, color="k", ls=":", lw=1, label="bottom-ranked")
    a_.set_title(title); a_.set_xlabel("cells deleted (of 64)"); a_.set_xticks([4, 8, 16]); a_.legend(fontsize=5.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
ax[0].set_ylabel("accuracy on cohort (%)"); fig.tight_layout(pad=0.4); fig.savefig(os.path.join(F, "faithfulness.pdf")); plt.close(fig)

# Fig C: learning curves
fig, a_ = plt.subplots(figsize=(2.9, 2.0))
for m, lab, col in (("cnn", "CNN", NAVY), ("svm", "RBF-SVM", TEAL), ("bayes", "PCA + Bayes", ORANGE)):
    d = cs[cs.method == m]; a_.errorbar(100 * d.frac, 100 * d["mean"], 100 * d["std"], color=col, marker="o", ms=3, lw=1.2, capsize=2, label=lab)
a_.set_xlabel("training data used (%)"); a_.set_ylabel("test accuracy (%)"); a_.set_xticks([25, 50, 100]); a_.legend(fontsize=6, frameon=False)
fig.tight_layout(pad=0.4); fig.savefig(os.path.join(F, "curve.pdf")); plt.close(fig)

# Fig D: confusion (CNN) and per-class recall vs map stability
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1, 1.25]})
Cn = C / C.sum(1, keepdims=True); im = ax[0].imshow(Cn, cmap="Blues", vmin=0, vmax=1)
ax[0].set_xticks(range(11)); ax[0].set_yticks(range(11)); ax[0].set_xlabel("predicted gesture"); ax[0].set_ylabel("true gesture"); ax[0].set_title("CNN row-normalised confusion")
for i in range(11):
    for j in range(11):
        if Cn[i, j] >= 0.1: ax[0].text(j, i, f"{100 * Cn[i, j]:.0f}", ha="center", va="center", fontsize=5.5, color="white" if Cn[i, j] > 0.5 else "k")
x = np.arange(11); w = 0.27
ax[1].bar(x - w, 100 * rec, w, color=NAVY, label="CNN recall (%)");
for j, (m, lab, col) in enumerate((("cnn_gc", "Grad-CAM $\\rho$ x100", ORANGE), ("svm_occ_margin", "SVM margin $\\rho$ x100", TEAL))):
    ax[1].bar(x + w * j, 100 * p10[m].values, w, color=col, label=lab)
ax[1].set_xticks(x); ax[1].set_xlabel("gesture"); ax[1].set_ylim(0, 105); ax[1].legend(fontsize=5.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3); ax[1].set_title("Recall vs map stability at 10-pt strength")
fig.tight_layout(pad=0.4); fig.savefig(os.path.join(F, "errors.pdf")); plt.close(fig)

# Fig E: the SVM's own view (similarity against the SVM's own accuracy drop, and cohort retention)
Ps = P[(P.model == "svm") & (P.calib == "acc")]
fig, ax = plt.subplots(2, 5, figsize=(12.5, 5.2), sharey="row")
for j, k in enumerate(KINDS):
    d = Ps[Ps.kind == k]
    for m, c, lab in (("occlusion_margin", TEAL, "SVM margin map"), ("occlusion_vote", NAVY, "SVM vote map")):
        g = d[d.method == m].groupby("target_ssim")
        ax[0, j].errorbar(g.dacc.mean(), g.spearman_all.mean(), yerr=g.spearman_all.std(), color=c, marker="o", ms=3, lw=1.3, capsize=2, label=lab)
        ax[0, j].axhline(fl.loc[("svm", m), "spearman_all"], color=c, ls=":", lw=1)
    g = d[d.method == "occlusion_margin"].groupby("target_ssim")
    ax[1, j].errorbar(g.dacc.mean(), 100 * g.retention.mean(), yerr=100 * g.retention.std(), color=ORANGE, marker="o", ms=3, lw=1.3, capsize=2)
    ax[0, j].set_title(KNAME[k]); ax[1, j].set_xlabel("SVM accuracy drop (pts)"); ax[0, j].set_ylim(0, 1.05)
ax[0, 0].set_ylabel("Spearman $\\rho$ (clean vs perturbed map)"); ax[1, 0].set_ylabel("cohort still correct (%)"); ax[0, 0].legend(loc="lower left", fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(F, "svm_stability.pdf")); plt.close(fig)

json.dump(NUM, open(os.path.join(R, "stage4_numbers.json"), "w"), indent=1, default=str)
print("wrote", sorted(os.listdir(T)), sorted(os.listdir(F)))
