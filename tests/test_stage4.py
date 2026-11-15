import numpy as np, pytest
from soli_attr import config as C, perturb as P, explain as E
from soli_attr.models import build, MarginView
from soli_attr.stage4 import deletion_acc


def toy(n=330, seed=0):
    """Two Gaussian blobs in the dt-map per class: enough for an SVM to learn, fast to fit."""
    rng = np.random.default_rng(seed); y = np.arange(n) % C.N_CLASSES
    X = rng.normal(0.1, 0.05, (n, 32, 32)).astype(np.float32)
    for i, c in enumerate(y):
        X[i, (c * 3) % 24:(c * 3) % 24 + 8, 10:22] += 0.8
    return np.clip(X, 0, 1), y


def test_physical_perturbations_identity_shape_and_nonneg():
    X, _ = toy(20)
    for k in P.PHYS_KINDS:
        assert np.allclose(P.apply(X, k, 0.0), X)
        out = P.apply(X, k, 0.5); assert out.shape == X.shape and (out >= 0).all() and not np.allclose(out, X)
    a, b = P.apply(X, "snr_noise", 0.3), P.apply(X, "snr_noise", 0.3)
    assert np.allclose(a, b)                               # fixed noise realisation => reproducible


def test_noise_strength_matches_snr():
    X, _ = toy(50); p = 0.1
    noise = (P.apply(X, "snr_noise", p) - X)
    snr = 10 * np.log10((X ** 2).mean() / (noise ** 2).mean())
    assert abs(snr - (-20 * np.log10(p))) < 3.0            # clipping at 0 changes it a little


def test_margin_scores_agree_with_vote_prediction_most_of_the_time():
    X, y = toy(); m = build("svm", dict(C=10, gamma=0.01), 0).fit(X, y)
    s = m.margin_scores(X); assert s.shape == (len(X), C.N_CLASSES) and np.isfinite(s[:, :C.N_CLASSES]).all()
    assert (s.argmax(1) == m.predict(X)).mean() > 0.9
    assert (MarginView(m).scores(X) == s).all()
    # the temporary switch of decision_function_shape must be restored
    assert m.clf.steps[-1].decision_function_shape == "ovr"


def test_integrated_gradients_shape_and_finite():
    pytest.importorskip("torch")
    X, y = toy(66); m = build("cnn", dict(width=4, lr=1e-3), 0, "cpu", 1).fit(X, y)
    ig = E.integrated_gradients(m, X[:10], y[:10], steps=4)
    assert ig.shape == (10, C.GRID, C.GRID) and np.isfinite(ig).all()


def test_deletion_top_cells_hurts_more_than_random_for_a_faithful_map():
    X, y = toy(); m = build("svm", dict(C=10, gamma=0.01), 0).fit(X, y)
    maps = E.norm_map(E.occlusion(MarginView(m), X[:60], y[:60]))
    r = deletion_acc(m, X[:60], y[:60], maps, ks=(8,))[0]
    assert r["acc_top"] <= r["acc_random"] + 1e-9
