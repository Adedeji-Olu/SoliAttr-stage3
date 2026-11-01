"""Reading Soli HDF5 sequences and building a compact cache.

File names are `[gesture]_[session]_[instance].h5`; each file holds datasets ch0..ch3
with shape [frames, 1024] and a per-frame `label` (deep-soli README).
"""
import io, json, os, re, zipfile
import numpy as np

from . import config as C

NAME_RE = re.compile(r"(\d+)_(\d+)_(\d+)\.h5$")


def parse_name(name):
    m = NAME_RE.search(os.path.basename(name))
    if not m:
        raise ValueError(f"unexpected file name: {name}")
    return tuple(int(g) for g in m.groups())  # gesture, session, instance


def resample_time(x, t_out):
    """Linearly resample axis 0 of x to t_out frames. x: [T, ...].

Why: sequences have 28-145 frames, but the models, the 8x8 explanation grid
and the perturbation bins all need one fixed shape. Padding would add fake
zeros, cropping would cut gestures, and a variable-length model would give the
CNN-RNN a different input from the other methods.
Cost: lossy. Gesture duration is discarded, and downsampling blends
neighbouring frames, so fast Doppler changes are slightly smoothed.
"""
    t_in = x.shape[0]
    if t_in == t_out:
        return x
    pos = np.linspace(0, t_in - 1, t_out)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, t_in - 1)
    w = (pos - lo).reshape(-1, *([1] * (x.ndim - 1)))
    return x[lo] * (1 - w) + x[hi] * w


def read_sequence(fobj):
    """Return (rdm [4, T0, 32, 32] float32, labels [T0])."""
    import h5py
    with h5py.File(fobj, "r") as f:
        chans = [f[f"ch{c}"][()] for c in range(C.N_CHANNELS)]
        labels = np.asarray(f["label"][()]).reshape(-1)
    arr = np.stack(chans, axis=0).astype(np.float32)          # [4, T0, 1024]
    arr = arr.reshape(C.N_CHANNELS, -1, C.RDM_SIZE, C.RDM_SIZE)
    return arr, labels


def _iter_sources(src):
    """Yield (name, file-like) from a zip archive or a directory of .h5 files."""
    if os.path.isdir(src):
        for n in sorted(os.listdir(src)):
            if n.endswith(".h5"):
                yield n, os.path.join(src, n)
    else:
        with zipfile.ZipFile(src) as z:
            for n in sorted(z.namelist()):
                if n.endswith(".h5"):
                    yield n, io.BytesIO(z.read(n))


def build_cache(src, out_dir, t_frames=C.T_FRAMES, dtype="float16"):
    """Read every sequence once, verify it, and write a cache plus a data report."""
    os.makedirs(out_dir, exist_ok=True)
    names = [n for n, _ in _iter_sources(src)]
    n = len(names)
    rdm = np.zeros((n, C.N_CHANNELS, t_frames, C.RDM_SIZE, C.RDM_SIZE), dtype=dtype)
    gesture = np.zeros(n, int); session = np.zeros(n, int); instance = np.zeros(n, int)
    frames = np.zeros(n, int)
    nonconst = mismatch = nan_files = 0
    vmin, vmax = np.inf, -np.inf
    for i, (name, fobj) in enumerate(_iter_sources(src)):
        g, s, k = parse_name(name)
        arr, labels = read_sequence(fobj)
        gesture[i], session[i], instance[i], frames[i] = g, s, k, arr.shape[1]
        if len(np.unique(labels)) != 1:
            nonconst += 1
        if not np.all(labels == g):
            mismatch += 1
        if not np.isfinite(arr).all():
            nan_files += 1
            arr = np.nan_to_num(arr)
        vmin, vmax = min(vmin, float(arr.min())), max(vmax, float(arr.max()))
        arr = resample_time(arr.transpose(1, 0, 2, 3), t_frames).transpose(1, 0, 2, 3)
        rdm[i] = arr.astype(dtype)
    np.save(os.path.join(out_dir, "rdm.npy"), rdm)
    np.savez(os.path.join(out_dir, "meta.npz"), gesture=gesture, session=session, instance=instance, frames=frames)
    report = {
        "n_files": int(n), "frames_min": int(frames.min()), "frames_median": float(np.median(frames)),
        "frames_max": int(frames.max()), "t_frames_after_resampling": t_frames,
        "files_with_nonconstant_label": int(nonconst), "files_where_label_differs_from_filename_gesture": int(mismatch),
        "files_with_nonfinite_values": int(nan_files), "value_min": vmin, "value_max": vmax,
        "cache_dtype": dtype,
        "files_per_session": {int(k): int(v) for k, v in zip(*np.unique(session, return_counts=True))},
        "files_per_gesture": {int(k): int(v) for k, v in zip(*np.unique(gesture, return_counts=True))},
    }
    with open(os.path.join(out_dir, "data_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def load_cache(cache_dir, mmap=True):
    rdm = np.load(os.path.join(cache_dir, "rdm.npy"), mmap_mode="r" if mmap else None)
    m = np.load(os.path.join(cache_dir, "meta.npz"))
    return rdm, {k: m[k] for k in m.files}
