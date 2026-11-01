"""User-grouped splits with explicit leakage checks."""
import numpy as np
from . import config as C


def sessions_of(users):
    return [s for u in users for s in C.USER_SESSIONS[u]]


def gesture_mask(meta):
    return meta["gesture"] < C.N_CLASSES          # drops the background class (gesture 11)


def fold_split(meta, fold, repeat, n_val_users=C.N_VAL_USERS):
    """Test users are fixed by the fold. Validation users are drawn per repeat from the
    training pool (never user A, who holds 53% of the data). Returns index arrays + user names."""
    test_users = list(C.FOLDS[fold])
    pool = [u for u in C.USER_SESSIONS if u not in test_users]
    candidates = [u for u in pool if u != "A"]
    rng = np.random.default_rng(1000 * fold + repeat)
    val_users = [str(u) for u in rng.choice(candidates, n_val_users, replace=False)]
    train_users = [u for u in pool if u not in val_users]
    ok = gesture_mask(meta)
    sess = meta["session"]
    idx = lambda users: np.where(ok & np.isin(sess, sessions_of(users)))[0]
    tr, va, te = idx(train_users), idx(val_users), idx(test_users)
    assert_no_leakage(meta, tr, va, te)
    return tr, va, te, {"train": train_users, "val": val_users, "test": test_users}


def random_split(meta, repeat, frac=(0.70, 0.15, 0.15)):
    """Stratified random split by sequence. NOT leakage-safe (same users on both sides);
    kept only as an inflated reference to show how much leakage would add."""
    from sklearn.model_selection import train_test_split
    ok = np.where(gesture_mask(meta))[0]
    y = meta["gesture"][ok]
    tr, rest = train_test_split(ok, train_size=frac[0], stratify=y, random_state=repeat)
    va, te = train_test_split(rest, train_size=frac[1] / (frac[1] + frac[2]), stratify=meta["gesture"][rest], random_state=repeat)
    return tr, va, te


def assert_no_leakage(meta, tr, va, te):
    sets = [set(meta["session"][i].tolist()) for i in (tr, va, te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2]), "session overlap between splits"
    user_of = {s: u for u, ss in C.USER_SESSIONS.items() for s in ss}
    users = [set(user_of[s] for s in S) for S in sets]
    assert not (users[0] & users[1]) and not (users[0] & users[2]) and not (users[1] & users[2]), "user overlap between splits"
    assert len(set(tr) & set(te)) == 0 and len(set(tr) & set(va)) == 0 and len(set(va) & set(te)) == 0
