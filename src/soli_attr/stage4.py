"""Stage 4 experiments.

1. A second attribution family (Integrated Gradients) and a continuous SVM score (pairwise margins),
   so the SVM floor is no longer inflated by integer vote counts.
2. Deletion faithfulness: do the highest-ranked cells really carry the decision?
3. Two radar-motivated impairments (additive noise at a set SNR, static zero-Doppler clutter) next to the
   three Stage 3 perturbations.
4. Sensitivity: attribution grid size, training-set size, SVM tuning budget.
"""
import json
import numpy as np

from . import config as C
from . import explain as E
from . import perturb as P
from .models import tune, build, MarginView
from .stability import _row, TARGET_SSIMS, ACC_DROPS

KINDS5 = tuple(P.KINDS) + tuple(P.PHYS_KINDS)
MAPS = {"cnn_gc": ("cnn", "gradcam"), "cnn_occ": ("cnn", "occlusion"), "cnn_ig": ("cnn", "intgrad"),
        "svm_occ_vote": ("svm", "occlusion_vote"), "svm_occ_margin": ("svm", "occlusion_margin")}
SVM_BIG = [dict(C=c, gamma=g) for c in (1, 3, 10, 30, 100, 300) for g in (0.002, 0.005, 0.01, 0.02, 0.05)]


def maps5(cnn, svm, X, y):
    """Normalised maps (true class) on the 8x8 grid: Grad-CAM, occlusion and Integrated Gradients for the CNN;
    occlusion on the vote-count score and on the continuous margin score for the SVM."""
    mv = MarginView(svm)
    return {"cnn_gc": E.norm_map(E.gradcam(cnn, X, y)), "cnn_occ": E.norm_map(E.occlusion(cnn, X, y)),
            "cnn_ig": E.norm_map(E.integrated_gradients(cnn, X, y)),
            "svm_occ_vote": E.norm_map(E.occlusion(svm, X, y)), "svm_occ_margin": E.norm_map(E.occlusion(mv, X, y))}


def deletion_acc(model, X, y, maps, ks=(4, 8, 16), n_rand=5, seed=0, grid=C.GRID):
    """Accuracy after replacing the k top-ranked, k random or k bottom-ranked grid cells by the training mean.
    A faithful map makes the top-k deletion hurt far more than the random one."""
    n, t, d = X.shape; bt, bd = t // grid, d // grid; rng = np.random.default_rng(seed)
    order = np.argsort(-maps.reshape(n, -1), axis=1, kind="stable")

    def acc_after(cells):
        m = np.zeros((n, grid * grid), bool); np.put_along_axis(m, cells, True, axis=1)
        M = np.repeat(np.repeat(m.reshape(n, grid, grid), bt, 1), bd, 2)
        return float((model.predict(np.where(M, model.mu, X).astype(np.float32)) == y).mean())

    out = []
    for k in ks:
        rnd = np.mean([acc_after(np.argsort(rng.random((n, grid * grid)), axis=1)[:, :k]) for _ in range(n_rand)])
        out.append(dict(k=k, acc_top=acc_after(order[:, :k]), acc_random=float(rnd), acc_bottom=acc_after(order[:, -k:])))
    return out


def run_stability2(dt, y, tr, va, te, base, seed, device="cpu", epochs=30, max_cohort=150, kinds=KINDS5,
                   levels=TARGET_SSIMS, acc_drops=ACC_DROPS, grids=(4, 16)):
    """Returns (stability rows, faithfulness rows, per-class rows, grid-sensitivity rows)."""
    rows, faith, perclass, gridrows = [], [], [], []
    cnnA, _, _ = tune("cnn", dt[tr], y[tr], dt[va], y[va], seed, device, epochs)
    cnnB, _, _ = tune("cnn", dt[tr], y[tr], dt[va], y[va], seed + 1000, device, epochs)
    svm, svm_hp, _ = tune("svm", dt[tr], y[tr], dt[va], y[va], seed)
    ok = (cnnA.predict(dt[te]) == y[te]) & (cnnB.predict(dt[te]) == y[te]) & (svm.predict(dt[te]) == y[te])
    coh = np.sort(np.random.default_rng(seed).permutation(te[ok])[:max_cohort])
    base = dict(base, n_test=len(te), n_clean_correct_all3=int(ok.sum()),
                acc_clean_cnn=float((cnnA.predict(dt[te]) == y[te]).mean()), acc_clean_svm=float((svm.predict(dt[te]) == y[te]).mean()))
    if len(coh) < 5:
        return [dict(base, kind="skipped", note="cohort too small")], [], [], []
    X, yc, n = dt[coh], y[coh], len(coh)
    s = E.similarity

    mA, mB = maps5(cnnA, svm, X, yc), maps5(cnnB, svm, X, yc)
    sub = np.sort(np.random.default_rng(seed + 7).permutation(tr)[: int(0.8 * len(tr))])
    svmB = build("svm", svm.hp, seed + 1000).fit(dt[sub], y[sub])
    mB["svm_occ_vote"] = E.norm_map(E.occlusion(svmB, X, yc)); mB["svm_occ_margin"] = E.norm_map(E.occlusion(MarginView(svmB), X, yc))
    for k, (model, method) in MAPS.items():                                   # seed / refit floors
        rows.append(_row(base, "floor", 0, 0, 1, model, method, n, n, s(mA[k], mB[k]), None, {"ref": "seed_or_refit_floor"}))
    rows += [_row(base, "ref", 0, 0, 1, "cnn", "intgrad_vs_gradcam", n, n, s(mA["cnn_ig"], mA["cnn_gc"]), None, {"ref": "method_agreement"}),
             _row(base, "ref", 0, 0, 1, "cnn", "intgrad_vs_occlusion", n, n, s(mA["cnn_ig"], mA["cnn_occ"]), None, {"ref": "method_agreement"}),
             _row(base, "ref", 0, 0, 1, "svm", "margin_vs_vote", n, n, s(mA["svm_occ_margin"], mA["svm_occ_vote"]), None, {"ref": "method_agreement"}),
             _row(base, "ref", 0, 0, 1, "cnn_vs_svm", "occlusion_margin", n, n, s(mA["cnn_occ"], mA["svm_occ_margin"]), None, {"ref": "cnn_occ_vs_svm_margin"})]
    # sanity checks: random-weight CNN, label-permuted SVM
    rnd = cnnA.untrained_copy(); perm = build("svm", svm.hp, seed).fit(dt[tr], np.random.default_rng(seed).permutation(y[tr]))
    rmap = {"cnn_gc": E.norm_map(E.gradcam(rnd, X, yc)), "cnn_occ": E.norm_map(E.occlusion(rnd, X, yc)),
            "cnn_ig": E.norm_map(E.integrated_gradients(rnd, X, yc)),
            "svm_occ_vote": E.norm_map(E.occlusion(perm, X, yc)), "svm_occ_margin": E.norm_map(E.occlusion(MarginView(perm), X, yc))}
    for k, (model, method) in MAPS.items():
        rows.append(_row(base, "sanity", 0, 0, 1, model, method, n, n, s(mA[k], rmap[k]), None, {"ref": "random_or_permuted"}))
    # deletion faithfulness on the clean cohort
    for k, (model, method) in MAPS.items():
        for r in deletion_acc(cnnA if model == "cnn" else svm, X, yc, mA[k], seed=seed):
            faith.append(dict(base, map=k, model=model, method=method, **r))

    p10 = {}
    def one(kind, calib, lvl, p, ach):
        Xp, Xte_p = P.apply(X, kind, p), P.apply(dt[te], kind, p)
        acc = {"cnn": float((cnnA.predict(Xte_p) == y[te]).mean()), "svm": float((svm.predict(Xte_p) == y[te]).mean())}
        mp = maps5(cnnA, svm, Xp, yc)
        ret = {"cnn": cnnA.predict(Xp) == yc, "svm": svm.predict(Xp) == yc}
        reached = bool(abs(ach - lvl) < 0.03) if calib == "ssim" else bool(abs(ach - lvl) < max(0.5, 0.25 * lvl))
        for k, (model, method) in MAPS.items():
            sim = s(mA[k], mp[k]); r = ret[model]
            rows.append(_row(base, kind, lvl, p, ach, model, method, n, int(r.sum()), sim, {a: v[r] for a, v in sim.items()},
                             {"test_acc_perturbed": acc[model], "calib": calib, "reached": reached}))
            if calib == "acc" and lvl in (5.0, 10.0) and k in ("cnn_gc", "cnn_occ", "svm_occ_margin"):
                for c in np.unique(yc):
                    perclass.append(dict(base, kind=kind, drop=lvl, map=k, gesture=int(c), n=int((yc == c).sum()),
                                         spearman=float(np.mean(sim["spearman"][yc == c]))))

    for kind in kinds:
        for lvl in levels:
            p, ach = P.calibrate(kind, dt[va], lvl); one(kind, "ssim", lvl, p, ach)
        for drop in acc_drops:
            p, ach = P.calibrate_acc(kind, cnnA, dt[va], y[va], drop); one(kind, "acc", drop, p, ach)
            if drop == 10.0: p10[kind] = (p, ach)

    # grid-size sensitivity (occlusion only; Grad-CAM lives on the model's native 8x8 map)
    for g in grids:
        oc = lambda m, Z, v: E.norm_map(E.occlusion(m, Z, v, grid=g))
        a_c, b_c = oc(cnnA, X, yc), oc(cnnB, X, yc)
        a_s, b_s = oc(MarginView(svm), X, yc), oc(MarginView(svmB), X, yc)
        for model, a, b in (("cnn", a_c, b_c), ("svm", a_s, b_s)):
            gridrows.append(dict(base, grid=g, kind="floor", model=model, spearman=float(np.mean(s(a, b)["spearman"]))))
        for kind, (p, ach) in p10.items():
            Xp = P.apply(X, kind, p)
            for model, a, mod in (("cnn", a_c, cnnA), ("svm", a_s, MarginView(svm))):
                gridrows.append(dict(base, grid=g, kind=kind, model=model, spearman=float(np.mean(s(a, oc(mod, Xp, yc))["spearman"]))))
    return rows, faith, perclass, gridrows


def learning_curve(dt, y, tr, va, te, seed, device="cpu", epochs=30, fracs=(0.25, 0.5, 1.0), methods=("bayes", "svm", "cnn")):
    """Test accuracy against the share of training sequences used (same tuning budget at every size)."""
    rows = []
    for f in fracs:
        sub = np.sort(np.random.default_rng(seed + int(100 * f)).permutation(tr)[: max(int(f * len(tr)), 50)])
        for name in methods:
            m, hp, vacc = tune(name, dt[sub], y[sub], dt[va], y[va], seed, device, epochs)
            rows.append(dict(frac=f, n_train=len(sub), method=name, acc=float((m.predict(dt[te]) == y[te]).mean()), val_acc=vacc))
    return rows


def svm_budget(dt, y, tr, va, te, seed, grid=SVM_BIG):
    """RBF-SVM with a 30-configuration search instead of 4: does a larger budget change the SVM baseline?"""
    best = (-1, None, None)
    for hp in grid:
        m = build("svm", hp, seed).fit(dt[tr], y[tr]); v = float((m.predict(dt[va]) == y[va]).mean())
        if v > best[0]: best = (v, hp, m)
    return dict(budget=len(grid), acc=float((best[2].predict(dt[te]) == y[te]).mean()), val_acc=best[0], hp=json.dumps(best[1]))
