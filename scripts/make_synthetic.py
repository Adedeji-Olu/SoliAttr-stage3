"""Write a small synthetic dataset in the Soli file format so the pipeline can be smoke-tested without the real data."""
import argparse, os, numpy as np, h5py

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--instances", type=int, default=2)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(0)
dop = np.arange(32)
for g in range(12):
    for s in range(15):
        for k in range(a.instances):
            T = int(rng.integers(16, 25)); t = np.linspace(0, 1, T)
            centre = 16 + 9 * np.sin(2 * np.pi * (0.4 + 0.12 * g) * t + 0.3 * g + 0.2 * s)
            frame = np.exp(-0.5 * ((dop[None, :] - centre[:, None]) / (2.0 + 0.2 * (g % 4))) ** 2)     # [T,32] Doppler profile
            rdm = frame[:, None, :] * np.exp(-0.5 * ((np.arange(32)[None, :, None] - 10 - g) / 4.0) ** 2)  # [T,32 range,32 dop]
            with h5py.File(os.path.join(a.out, f"{g}_{s}_{k}.h5"), "w") as f:
                for c in range(4):
                    x = rdm.reshape(T, 1024) * (0.8 + 0.1 * c) + rng.normal(0, 0.15 + 0.01 * s, (T, 1024))
                    f.create_dataset(f"ch{c}", data=x.astype("float32"), compression="gzip")
                f.create_dataset("label", data=np.full(T, g, dtype="int32"))
print("wrote", len(os.listdir(a.out)), "files")
