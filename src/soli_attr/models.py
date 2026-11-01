"""Baselines and the project CNN. Every model takes RAW inputs and owns its normaliser (fit on train only)."""
import copy
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C
from .features import Norm, block_features


def _full_scores(classes, s):
    out = np.full((s.shape[0], C.N_CLASSES), -1e9, dtype=np.float32)
    out[:, classes] = s
    return out


class SkModel:
    """B3 majority, B1a PCA+Gaussian Bayes, B1b RBF-SVM. Input kind: 'dt' [N,T,D]."""
    kind = "dt"

    def __init__(self, name, hp=None, seed=0):
        self.name, self.hp, self.seed = name, hp or {}, seed

    def fit(self, X, y, Xv=None, yv=None):
        self.norm = Norm().fit(X); self.mu = self.norm.mu
        if self.name == "majority":
            self.maj = int(np.bincount(y, minlength=C.N_CLASSES).argmax()); return self
        F = block_features(self.norm(X))
        if self.name == "bayes":
            self.clf = _Pipe(StandardScaler(), PCA(self.hp["k"], random_state=self.seed),
                             GaussianBayes(self.hp["reg"]))
        elif self.name == "svm":
            self.clf = _Pipe(StandardScaler(), SVC(C=self.hp["C"], gamma=self.hp["gamma"], kernel="rbf",
                                                   decision_function_shape="ovr", random_state=self.seed))
        else:
            raise ValueError(self.name)
        self.clf.fit(F, y); self.classes = self.clf.steps[-1].classes_
        return self

    def scores(self, X):
        if self.name == "majority":
            s = np.zeros((len(X), C.N_CLASSES), np.float32); s[:, self.maj] = 1; return s
        F = block_features(self.norm(X)); last = self.clf.steps[-1]
        Z = self.clf.transform(F)
        s = last.predict_log_proba(Z) if self.name == "bayes" else last.decision_function(Z)
        return _full_scores(self.classes, np.clip(s, -1e6, 1e6).astype(np.float32))

    def predict(self, X):
        return self.scores(X).argmax(1)


class GaussianBayes:
    """Bayes classifier with Gaussian class-conditionals, g_c(z) = log p(z|c) + log P(c) (Duda et al., ch. 2).
    Class covariance is shrunk towards the pooled covariance, S_c = (1-r) S_c + r S_pool (+ ridge), so it
    stays invertible when a class has fewer samples than dimensions. r=1 gives LDA, r=0 gives QDA."""
    def __init__(self, reg): self.reg = reg
    def fit(self, Z, y):
        self.classes_ = np.unique(y); d = Z.shape[1]
        pooled = np.cov(Z.T) + 1e-6 * np.eye(d); self.m, self.L, self.ld, self.lp = [], [], [], []
        for c in self.classes_:
            Zc = Z[y == c]; S = np.cov(Zc.T) if len(Zc) > 1 else pooled
            S = (1 - self.reg) * S + self.reg * pooled + 1e-6 * np.eye(d)
            self.m.append(Zc.mean(0)); self.L.append(np.linalg.cholesky(S))
            self.ld.append(2 * np.log(np.diag(self.L[-1])).sum()); self.lp.append(np.log(len(Zc) / len(y)))
        return self
    def predict_log_proba(self, Z):
        g = []
        for m, L, ld, lp in zip(self.m, self.L, self.ld, self.lp):
            w = np.linalg.solve(L, (Z - m).T); g.append(-0.5 * (w ** 2).sum(0) - 0.5 * ld + lp)
        g = np.stack(g, 1); return g - g.max(1, keepdims=True)


class _Pipe:
    def __init__(self, *steps): self.steps = list(steps)
    def fit(self, F, y):
        Z = F
        for s in self.steps[:-1]:
            Z = s.fit_transform(Z)
        self.steps[-1].fit(Z, y); return self
    def transform(self, F):
        Z = F
        for s in self.steps[:-1]:
            Z = s.transform(Z)
        return Z


# ---------------------------------------------------------------- torch models
def _torch():
    import torch, torch.nn as nn
    return torch, nn


def make_net(arch, hp):
    torch, nn = _torch()
    if arch == "cnn":
        w = hp["width"]

        class SpecCNN(nn.Module):
            """dt-map [B,32,32] -> logits. Last conv map is 8x8, matching the attribution grid."""
            def __init__(s):
                super().__init__()
                blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
                s.b1, s.b2, s.b3 = blk(1, w), blk(w, 2 * w), blk(2 * w, 4 * w)
                s.pool = nn.MaxPool2d(2); s.drop = nn.Dropout(hp.get("drop", 0.3)); s.fc = nn.Linear(4 * w, C.N_CLASSES)

            def forward(s, x, feat=False):
                x = s.pool(s.b1(x.unsqueeze(1))); x = s.pool(s.b2(x)); f = s.b3(x)      # f: [B,4w,8,8]
                out = s.fc(s.drop(f.mean((2, 3))))
                return (out, f) if feat else out
        return SpecCNN()
    if arch == "cnnrnn":
        h = hp["hidden"]

        class CNNRNN(nn.Module):
            """Wang et al. (2016)-style: per-frame CNN on the 4-channel RDM, then a recurrent layer."""
            def __init__(s):
                super().__init__()
                s.cnn = nn.Sequential(nn.Conv2d(C.N_CHANNELS, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(2), nn.Flatten())
                s.rnn = nn.GRU(128, h, batch_first=True); s.fc = nn.Linear(h, C.N_CLASSES)

            def forward(s, x):
                b, t = x.shape[:2]
                z = s.cnn(x.reshape(b * t, *x.shape[2:])).reshape(b, t, -1)
                return s.fc(s.rnn(z)[0][:, -1])
        return CNNRNN()
    raise ValueError(arch)


class TorchModel:
    def __init__(self, arch, hp, seed=0, device="cpu", epochs=30, patience=6, bs=64):
        self.arch, self.hp, self.seed, self.device = arch, hp, seed, device
        self.epochs, self.patience, self.bs = epochs, patience, bs
        self.kind = "dt" if arch == "cnn" else "rdm"
        self.name = arch

    def _t(self, X):
        torch, _ = _torch()
        return ((torch.from_numpy(np.asarray(X, dtype=np.float32)) - self.mu) / self.sd).to(self.device)

    def fit(self, X, y, Xv=None, yv=None):
        torch, nn = _torch()
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        nrm = Norm().fit(X); self.mu, self.sd = nrm.mu, nrm.sd
        self.net = make_net(self.arch, self.hp).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hp["lr"], weight_decay=self.hp.get("wd", 0.0))
        rng = np.random.default_rng(self.seed); best, bad, state = -1, 0, None
        yt = torch.from_numpy(y).long().to(self.device)
        for ep in range(self.epochs):
            self.net.train(); perm = rng.permutation(len(X))
            for s in range(0, len(perm), self.bs):
                ii = np.sort(perm[s:s + self.bs])
                if len(ii) < 2: continue
                loss = nn.functional.cross_entropy(self.net(self._t(X[ii])), yt[ii])
                opt.zero_grad(); loss.backward(); opt.step()
            if Xv is None: continue
            acc = float((self.predict(Xv) == yv).mean())
            if acc > best: best, bad, state = acc, 0, copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= self.patience: break
        if state is not None: self.net.load_state_dict(state)
        self.net.eval(); return self

    def scores(self, X):
        torch, _ = _torch(); self.net.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(X), 256):
                out.append(self.net(self._t(X[s:s + 256])).cpu().numpy())
        return np.concatenate(out, 0)

    def predict(self, X):
        return self.scores(X).argmax(1)

    def untrained_copy(self, seed=12345):
        """Same architecture and normaliser, freshly initialised weights (sanity check)."""
        torch, _ = _torch(); torch.manual_seed(seed)
        m = TorchModel(self.arch, self.hp, seed, self.device, self.epochs, self.patience, self.bs)
        m.mu, m.sd = self.mu, self.sd; m.net = make_net(self.arch, self.hp).to(self.device).eval(); return m


# ---------------------------------------------------------------- tuning (validation users only, equal budget)
GRIDS = {
    "bayes": [dict(k=k, reg=r) for k, r in [(16, 0.1), (24, 0.3), (32, 0.5), (48, 0.8)]],
    "svm": [dict(C=c, gamma=g) for c, g in [(1, 0.005), (10, 0.005), (10, 0.02), (100, 0.01)]],
    "cnn": [dict(width=w, lr=lr, wd=wd) for w, lr, wd in [(8, 1e-3, 1e-4), (16, 1e-3, 1e-4), (16, 3e-4, 1e-3), (8, 3e-3, 1e-4)]],
    "cnnrnn": [dict(hidden=h, lr=lr) for h, lr in [(32, 1e-3), (64, 1e-3), (64, 3e-4), (32, 3e-3)]],
}


def build(name, hp, seed, device="cpu", epochs=30):
    if name in ("majority", "bayes", "svm"):
        return SkModel(name, hp, seed)
    return TorchModel(name, hp, seed, device, epochs)


def tune(name, Xtr, ytr, Xva, yva, seed=0, device="cpu", epochs=30, budget=C.BUDGET):
    """Fit `budget` configurations on train, pick the best by validation accuracy. Test data is never touched."""
    grid = GRIDS.get(name, [{}])[:budget]
    best = (-1, None, None)
    for hp in grid:
        m = build(name, hp, seed, device, epochs).fit(Xtr, ytr, Xva, yva)
        acc = float((m.predict(Xva) == yva).mean())
        if acc > best[0]: best = (acc, hp, m)
    return best[2], best[1], best[0]
