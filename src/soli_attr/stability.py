"""Attribution-stability experiment on a fixed clean-correct cohort."""
import numpy as np

from . import config as C
from . import explain as E
from . import perturb as P
from .models import tune, build

TARGET_SSIMS = (0.99, 0.97, 0.95, 0.85, 0.75, 0.65)   # severity matched by input SSIM
ACC_DROPS = (1.0, 2.0, 5.0, 10.0)                     # severity matched by CNN accuracy drop (points) on validation users


def _maps(cnn, svm, X, y, do_cnn=True):
    """Normalised maps for the true class. Grad-CAM and occlusion for the CNN, occlusion for the SVM."""
    out = {"svm_occ": E.norm_map(E.occlusion(svm, X, y))}
    if do_cnn:
        out["cnn_gc"] = E.norm_map(E.gradcam(cnn, X, y)); out["cnn_occ"] = E.norm_map(E.occlusion(cnn, X, y))
    return out


def _row(base, kind, level, p, achieved, model, method, n_cohort, n_ret, sim_all, sim_ret, extra=None):
    r = dict(base, kind=kind, target_ssim=level, p=p, achieved_ssim=achieved, model=model, method=method,
             n_cohort=n_cohort, n_retained=n_ret, retention=n_ret / max(n_cohort, 1))
    for m in ("spearman", "topk", "ssim"):
        r[f"{m}_all"] = float(np.mean(sim_all[m])) if len(sim_all[m]) else np.nan
        r[f"{m}_ret"] = float(np.mean(sim_ret[m])) if sim_ret is not None and len(sim_ret[m]) else np.nan
    r.update(extra or {}); return r


def run_stability(dt, y, tr, va, te, base, seed, device="cpu", epochs=30, max_cohort=150, levels=TARGET_SSIMS, kinds=P.KINDS, acc_drops=ACC_DROPS):
    """Returns a list of result rows (dicts). `base` carries fold/repeat labels."""
    rows = []
    cnnA, hpA, _ = tune("cnn", dt[tr], y[tr], dt[va], y[va], seed, device, epochs)
    cnnB, _, _ = tune("cnn", dt[tr], y[tr], dt[va], y[va], seed + 1000, device, epochs)
    svm, _, _ = tune("svm", dt[tr], y[tr], dt[va], y[va], seed)

    # ---- common clean-correct cohort (test users only)
    ok = (cnnA.predict(dt[te]) == y[te]) & (cnnB.predict(dt[te]) == y[te]) & (svm.predict(dt[te]) == y[te])
    coh = te[ok]
    rng = np.random.default_rng(seed); coh = np.sort(rng.permutation(coh)[:max_cohort])
    base = dict(base, n_test=len(te), n_clean_correct_all3=int(ok.sum()),
                acc_clean_cnn=float((cnnA.predict(dt[te]) == y[te]).mean()), acc_clean_svm=float((svm.predict(dt[te]) == y[te]).mean()))
    if len(coh) < 5:
        return [dict(base, kind="skipped", note="cohort too small")]
    X, yc = dt[coh], y[coh]

    mA, mB = _maps(cnnA, svm, X, yc), _maps(cnnB, svm, X, yc)
    # SVM is deterministic, so its own reference "floor" retrains it on a random 80% of the training pool
    sub = np.sort(np.random.default_rng(seed + 7).permutation(tr)[: int(0.8 * len(tr))])
    svmB = build("svm", svm.hp, seed + 1000).fit(dt[sub], y[sub])
    smB = E.norm_map(E.occlusion(svmB, X, yc))
    s = lambda a, b: E.similarity(a, b)
    n = len(coh)
    # ---- seed floor and cross-method / cross-model reference (clean vs clean)
    rows += [
        _row(base, "floor", 0, 0, 1, "cnn", "gradcam", n, n, s(mA["cnn_gc"], mB["cnn_gc"]), None, {"ref": "cnnA_vs_cnnB"}),
        _row(base, "floor", 0, 0, 1, "cnn", "occlusion", n, n, s(mA["cnn_occ"], mB["cnn_occ"]), None, {"ref": "cnnA_vs_cnnB"}),
        _row(base, "floor", 0, 0, 1, "svm", "occlusion", n, n, s(mA["svm_occ"], smB), None, {"ref": "svm_vs_svm_80pct_refit"}),
        _row(base, "ref", 0, 0, 1, "cnn_vs_svm", "occlusion", n, n, s(mA["cnn_occ"], mA["svm_occ"]), None, {"ref": "cnnA_occ_vs_svm_occ"}),
        _row(base, "ref", 0, 0, 1, "cnn", "gradcam_vs_occ", n, n, s(mA["cnn_gc"], mA["cnn_occ"]), None, {"ref": "method_agreement"}),
    ]
    # ---- sanity checks (Adebayo et al.): maps must depend on the learned model
    rnd = cnnA.untrained_copy(); rmaps = _maps(rnd, svm, X, yc)["cnn_gc"], E.norm_map(E.occlusion(rnd, X, yc))
    perm = build("svm", svm.hp, seed).fit(dt[tr], np.random.default_rng(seed).permutation(y[tr]))
    smap = E.norm_map(E.occlusion(perm, X, yc))
    rows += [_row(base, "sanity", 0, 0, 1, "cnn", "gradcam", n, n, s(mA["cnn_gc"], rmaps[0]), None, {"ref": "trained_vs_random_init"}),
             _row(base, "sanity", 0, 0, 1, "cnn", "occlusion", n, n, s(mA["cnn_occ"], rmaps[1]), None, {"ref": "trained_vs_random_init"}),
             _row(base, "sanity", 0, 0, 1, "svm", "occlusion", n, n, s(mA["svm_occ"], smap), None, {"ref": "trained_vs_label_permuted"})]

    # ---- perturbations. Two severity matchings, both calibrated on validation users only:
    #      "ssim" = target input SSIM; "acc" = target drop in CNN validation accuracy (points)
    def one(kind, calib, lvl, p, ach):
        Xte_p = P.apply(dt[te], kind, p)
        acc = {"cnn": float((cnnA.predict(Xte_p) == y[te]).mean()), "svm": float((svm.predict(Xte_p) == y[te]).mean())}
        Xp = P.apply(X, kind, p)
        mp = _maps(cnnA, svm, Xp, yc)
        ret = {"cnn": cnnA.predict(Xp) == yc, "svm": svm.predict(Xp) == yc}
        jobs = [("cnn", "gradcam", mA["cnn_gc"], mp["cnn_gc"]), ("cnn", "occlusion", mA["cnn_occ"], mp["cnn_occ"]),
                ("svm", "occlusion", mA["svm_occ"], mp["svm_occ"])]
        for model, method, a, b in jobs:
            r = ret[model]; sim = s(a, b)
            sim_r = {k: v[r] for k, v in sim.items()}
            reached = bool(abs(ach - lvl) < 0.03) if calib == "ssim" else bool(abs(ach - lvl) < max(0.5, 0.25 * lvl))
            rows.append(_row(base, kind, lvl, p, ach, model, method, n, int(r.sum()), sim, sim_r,
                             {"test_acc_perturbed": acc[model], "calib": calib, "reached": reached}))

    for kind in kinds:
        for lvl in levels:
            p, ach = P.calibrate(kind, dt[va], lvl)
            one(kind, "ssim", lvl, p, ach)
        for drop in acc_drops:
            p, ach = P.calibrate_acc(kind, cnnA, dt[va], y[va], drop)
            one(kind, "acc", drop, p, ach)
    return rows
