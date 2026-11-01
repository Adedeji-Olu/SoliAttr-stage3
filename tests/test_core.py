import numpy as np, pytest
from soli_attr import config as C, perturb as P, explain as E
from soli_attr.splits import fold_split, assert_no_leakage, random_split
from soli_attr.features import block_features, pool_cells, Norm, dt_maps
from soli_attr.data import resample_time, parse_name


def fake_meta():
    g, s = np.meshgrid(np.arange(12), np.arange(15), indexing="ij")
    return {"gesture": g.ravel(), "session": s.ravel(), "instance": np.zeros(g.size, int)}


@pytest.mark.parametrize("fold", range(4))
def test_user_grouped_splits_have_no_overlap(fold):
    m = fake_meta()
    for rep in range(5):
        tr, va, te, users = fold_split(m, fold, rep)
        assert_no_leakage(m, tr, va, te)
        assert "A" not in users["val"] and not (m["gesture"][np.r_[tr, va, te]] == C.BACKGROUND_GESTURE).any()
        assert len(te) > 0 and len(tr) > len(te) * 0.5


def test_leakage_check_catches_a_random_split():
    m = fake_meta(); tr, va, te = random_split(m, 0)
    with pytest.raises(AssertionError):
        assert_no_leakage(m, tr, va, te)


def test_every_session_belongs_to_exactly_one_fold():
    sess = [s for f in C.FOLDS.values() for u in f for s in C.USER_SESSIONS[u]]
    assert sorted(sess) == list(range(15))


def test_perturbations_identity_and_shape():
    X = np.random.default_rng(0).normal(size=(5, 32, 32)).astype(np.float32)
    for k in P.KINDS:
        assert np.allclose(P.apply(X, k, 0.0), X) and P.apply(X, k, 0.5).shape == X.shape
        assert not np.allclose(P.apply(X, k, 0.8), X)


def test_calibration_hits_target():
    X = np.abs(np.random.default_rng(1).normal(size=(30, 32, 32))).astype(np.float32)
    X = P.apply(X, "smooth", 0.05)
    for k in P.KINDS:
        p, ach = P.calibrate(k, X, 0.8)
        assert 0 <= p <= 1 and (abs(ach - 0.8) < 0.08 or p == 1.0)


def test_pool_and_features():
    X = np.arange(2 * 32 * 32, dtype=np.float32).reshape(2, 32, 32)
    assert pool_cells(X).shape == (2, 8, 8) and block_features(X).shape == (2, 128)
    assert np.isclose(pool_cells(X)[0, 0, 0], X[0, :4, :4].mean())


def test_norm_fits_on_given_data_only():
    a = np.random.default_rng(0).normal(5, 2, (100, 32, 32)); n = Norm().fit(a)
    assert abs(n.mu - 5) < 0.2 and abs(n.sd - 2) < 0.2


def test_dt_maps_axis_choice():
    rdm = np.zeros((1, 4, 2, 32, 32), np.float32); rdm[..., :, 5] = 1.0       # energy in column 5
    d1 = dt_maps(rdm, np.array([0]), doppler_axis=1); assert d1.shape == (1, 2, 32) and d1[0, 0, 5] == 1 and d1[0, 0, 4] == 0
    d0 = dt_maps(rdm, np.array([0]), doppler_axis=0); assert np.allclose(d0[0, 0], d0[0, 0, 0])   # energy sits on the range axis, so Doppler profile is flat


def test_map_normalisation_and_metrics():
    m = np.random.default_rng(0).normal(size=(4, 8, 8)); n = E.norm_map(m)
    assert np.allclose(n.sum((1, 2)), 1) and (n >= 0).all()
    assert np.allclose(E.norm_map(np.zeros((1, 8, 8))).sum(), 1)
    sim = E.similarity(n, n)
    assert np.allclose(sim["spearman"], 1) and np.allclose(sim["topk"], 1) and np.allclose(sim["ssim"], 1)
    other = E.similarity(n, n[::-1][::-1][:, ::-1, ::-1]); assert all(v.shape == (4,) for v in other.values())


def test_resample_time():
    x = np.arange(10, dtype=float).reshape(10, 1); assert resample_time(x, 19).shape == (19, 1)
    assert np.isclose(resample_time(x, 19)[-1, 0], 9) and parse_name("3_12_7.h5") == (3, 12, 7)


def test_occlusion_matches_a_known_model():
    class Toy:                                   # score = mean of the top-left cell, so only that cell matters
        mu = 0.0
        def scores(self, X):
            s = np.zeros((len(X), C.N_CLASSES), np.float32); s[:, 0] = X[:, :4, :4].mean((1, 2)); return s
    X = np.ones((3, 32, 32), np.float32); m = E.occlusion(Toy(), X, np.zeros(3, int))
    assert np.allclose(m[:, 0, 0], 1) and np.allclose(m.sum((1, 2)), 1)
