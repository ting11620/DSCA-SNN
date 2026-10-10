import argparse
from pathlib import Path
import numpy as np
import tables

def binary_image_readout(times, units, dt=1e-3):
    times = np.array(times, copy=True)
    units = np.array(units, copy=True)
    out = []
    for i in range(int(1/dt)):
        idxs = np.argwhere(times <= i*dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vec = np.zeros(700, dtype=np.float32)
        vec[700-vals] = 1
        times = np.delete(times, idxs)
        units = np.delete(units, idxs)
        out.append(vec)
    return np.asarray(out, dtype=np.float32)

def generate_dataset(h5_file, output_dir, dt):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tables.open_file(str(h5_file), mode="r") as f:
        units, times, labels = f.root.spikes.units, f.root.spikes.times, f.root.labels
        for i in range(len(times)):
            x = binary_image_readout(times[i], units[i], dt)
            np.save(output_dir/f"ID:{i}_{int(labels[i])}.npy", x)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-h5", required=True)
    p.add_argument("--test-h5", required=True)
    p.add_argument("--output-root", default="./data/SHD")
    p.add_argument("--dt", type=float, default=1e-3)
    a = p.parse_args()
    tag = f"{a.dt*1000:g}ms"
    generate_dataset(a.train_h5, Path(a.output_root)/f"train_{tag}", a.dt)
    generate_dataset(a.test_h5, Path(a.output_root)/f"test_{tag}", a.dt)

if __name__ == "__main__":
    main()
