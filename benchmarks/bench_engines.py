"""Benchmark harness for PtyLab reconstruction engines.

Reports wall time *and* peak GPU memory. Both are acceptance criteria: this
group runs on cards as small as 32 GB, so an optimization that trades memory
for speed is a regression, not an improvement.

Datasets are synthesized in-process at a requested size, so the harness runs
anywhere without the (gitignored) files in ``example_data/``.

Usage::

    python benchmarks/bench_engines.py                 # default sweep
    python benchmarks/bench_engines.py --quick         # one config per regime
    python benchmarks/bench_engines.py --cpu           # CPU baseline too
    python benchmarks/bench_engines.py --markdown      # BASELINE.md table

Regimes (see BASELINE.md for the measurements behind these):
  A  field <= ~12 MB   launch/dispatch bound   -- graphs & fusion pay
  B  field >= ~50 MB   HBM bandwidth + cuFFT   -- little headroom
  C  OPR               orthogonalization       -- two thirds linear algebra

The "+ orth" config runs the mode-orthogonalization constraint every iteration.
Compare it against the otherwise identical config without the suffix to price
the constraint; the gap is sensitive to whether cuSOLVER is usable, since a
CuPy install that cannot load libcusolver sends orthogonalizeModes to the host
and back on every call.
"""

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

# Progress bars cost host time in the loop being measured (mPIE draws a bar per
# scan position); disable before PtyLab imports tqdm.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # never open a window; showReconstruction is disabled anyway

from PtyLab import Engines  # noqa: E402
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData  # noqa: E402
from PtyLab.Monitor.Monitor import DummyMonitor  # noqa: E402
from PtyLab.Params.Params import Params  # noqa: E402
from PtyLab.Reconstruction.Reconstruction import Reconstruction  # noqa: E402

try:
    import cupy as cp

    HAS_GPU = cp.cuda.is_available()
except Exception:  # pragma: no cover
    cp = None
    HAS_GPU = False


# name, engine, propagator, (nlambda,nosm,npsm,nslice), Nd, n_frames, regime
CONFIGS = [
    ("simu-like  128, 100fr",       "mPIE", "Fraunhofer", (1, 1, 1, 1), 128, 100, "A"),
    ("USAF-like  364, 102fr",       "mPIE", "Fraunhofer", (1, 1, 1, 1), 364, 102, "A"),
    ("mixed npsm=4  364",           "mPIE", "Fraunhofer", (1, 1, 4, 1), 364, 102, "A"),
    ("Brain-like poly=7  182",      "ePIE", "polychromeASP", (7, 1, 1, 1), 182, 100, "A"),
    ("multislice nslice=4  364",    "e3PIE", "Fraunhofer", (1, 1, 1, 4), 364, 102, "A"),
    ("mixed npsm=4  364 + orth",    "mPIE", "Fraunhofer", (1, 1, 4, 1), 364, 102, "A"),
    ("heavy 7x2x4  364",            "mPIE", "Fraunhofer", (7, 2, 4, 1), 364, 40, "B"),
    ("OPR 364, 202fr, 4 modes",     "OPR", "Fraunhofer", (1, 1, 4, 1), 364, 202, "C"),
]

QUICK = {"simu-like  128, 100fr", "heavy 7x2x4  364", "OPR 364, 202fr, 4 modes"}

# Configs that run the mode-orthogonalization constraint every iteration. Every
# other config leaves orthogonalizationSwitch off, which left orthogonalizeModes
# -- the most expensive constraint, and one that reaches into cuSOLVER -- with no
# benchmark coverage at all. Paired with "mixed npsm=4  364", which is otherwise
# identical, so the difference between the two rows is the constraint's cost.
ORTHOGONALIZED = {"mixed npsm=4  364 + orth"}


def synth_dataset(path, nd, n_frames, seed=7):
    """Write a deterministic CPM dataset of the requested size."""
    import h5py

    rng = np.random.default_rng(seed)
    grid = int(np.ceil(np.sqrt(n_frames)))
    step = 3e-6
    coords = (np.arange(grid) - (grid - 1) / 2) * step
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    encoder = np.stack([yy.ravel(), xx.ravel()], axis=1)[:n_frames]

    ptychogram = rng.random((n_frames, nd, nd)).astype(np.float32)

    with h5py.File(path, "w") as hf:
        hf.create_dataset("ptychogram", data=ptychogram, dtype="f")
        hf.create_dataset("encoder", data=encoder, dtype="f")
        hf.create_dataset("dxd", data=np.array(75e-6))
        hf.create_dataset("zo", data=np.array(0.05))
        hf.create_dataset("wavelength", data=np.array(632.8e-9))
        hf.create_dataset("entrancePupilDiameter", data=np.array(400e-6))
    return path


def build(path, config, gpu):
    _name, engine_name, propagator, modes, _nd, _nfr, _regime = config
    nlambda, nosm, npsm, nslice = modes

    data = ExperimentalData(str(path), operationMode="CPM")
    params = Params()
    params.gpuSwitch = gpu
    params.propagatorType = propagator
    params.positionOrder = "sequential"
    if _name in ORTHOGONALIZED:
        params.orthogonalizationSwitch = True
        params.orthogonalizationFrequency = 1

    reconstruction = Reconstruction(data, params)
    reconstruction.nlambda = nlambda
    reconstruction.nosm = nosm
    reconstruction.npsm = npsm
    reconstruction.nslice = nslice
    if nlambda > 1:
        base = float(np.atleast_1d(reconstruction.wavelength)[0])
        reconstruction.spectralDensity = base * np.linspace(0.98, 1.02, nlambda)
    if nslice > 1:
        reconstruction.dz = 1e-4
        reconstruction.refrIndex = 1.0

    np.random.seed(0)
    reconstruction.initializeObjectProbe()

    monitor = DummyMonitor()
    engine = getattr(Engines, engine_name)(reconstruction, data, params, monitor)
    if engine_name == "OPR":
        params.OPR_modes = np.arange(npsm)
        params.OPR_subspace = min(4, _nfr)
        engine.OPR_modes = params.OPR_modes
        engine.n_subspace = params.OPR_subspace
    return data, reconstruction, params, engine


def field_mb(config):
    _n, _e, _p, (nl, nos, nps, nsl), nd, _f, _r = config
    return nl * nos * nps * nsl * nd * nd * 8 / 2**20


def reset_memory():
    if HAS_GPU:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_memory_pool().set_limit(size=0)  # no cap; just reset accounting
    gc.collect()


def peak_gpu_gb():
    if not HAS_GPU:
        return float("nan")
    return cp.get_default_memory_pool().total_bytes() / 2**30


def sync():
    if HAS_GPU:
        cp.cuda.runtime.deviceSynchronize()


def run_one(path, config, gpu, iters, warmup_iters=1):
    """Time ``iters`` iterations after a warm-up, returning (s/iter, peak GB)."""
    _name, _engine_name, _prop, _modes, _nd, n_frames, _regime = config

    data, reconstruction, params, engine = build(path, config, gpu)

    # Warm up: JIT, cuFFT plans, lru_cache'd transfer functions, first-touch
    # allocations. Without this the first iteration dominates the measurement.
    engine.numIterations = warmup_iters
    engine.reconstruct()
    sync()

    if HAS_GPU:
        cp.get_default_memory_pool().free_all_blocks()
    before = peak_gpu_gb()

    engine.numIterations = iters
    sync()
    t0 = time.perf_counter()
    engine.reconstruct()
    sync()
    elapsed = time.perf_counter() - t0

    peak = max(peak_gpu_gb(), before)
    del data, reconstruction, params, engine
    reset_memory()
    return elapsed / iters, peak, n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one config per regime")
    ap.add_argument("--cpu", action="store_true", help="also benchmark the CPU path")
    ap.add_argument("--iters", type=int, default=3, help="timed iterations")
    ap.add_argument("--markdown", action="store_true", help="emit a markdown table")
    ap.add_argument("--only", type=str, default=None, help="substring filter")
    args = ap.parse_args()

    logging.disable(logging.WARNING)

    configs = [c for c in CONFIGS if not args.quick or c[0] in QUICK]
    if args.only:
        configs = [c for c in configs if args.only.lower() in c[0].lower()]

    backends = [("gpu", True)] if HAS_GPU else []
    if args.cpu or not HAS_GPU:
        backends.append(("cpu", False))
    if not backends:
        print("no backend available")
        return

    if HAS_GPU:
        dev = cp.cuda.Device()
        _free, total = dev.mem_info
        name = cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode()
        print(f"# GPU: {name}, {total / 2**30:.1f} GB total, CuPy {cp.__version__}")
    print(f"# numpy {np.__version__}, {args.iters} timed iterations after warm-up\n")

    header = ["config", "reg", "field MB", "backend", "ms/iter", "us/pos", "peak GB"]
    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("---" for _ in header) + "|")
    else:
        print(f"{'config':<28s} {'reg':>3s} {'fieldMB':>8s} {'backend':>7s} "
              f"{'ms/iter':>9s} {'us/pos':>8s} {'peakGB':>8s}")
        print("-" * 80)

    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="ptylab_bench_"))
    for config in configs:
        name, engine_name, _p, _m, nd, n_frames, regime = config
        path = synth_dataset(tmpdir / f"{nd}_{n_frames}.hdf5", nd, n_frames)
        for backend_name, gpu in backends:
            if engine_name == "OPR" and not gpu:
                continue  # OPR has no CPU path (OPR.py calls cp.* directly)
            try:
                s_per_iter, peak, nfr = run_one(path, config, gpu, args.iters)
                row = [name, regime, f"{field_mb(config):.1f}", backend_name,
                       f"{s_per_iter * 1e3:.1f}",
                       f"{s_per_iter / nfr * 1e6:.0f}",
                       f"{peak:.2f}"]
            except Exception as exc:  # keep going; report the failure in-band
                row = [name, regime, f"{field_mb(config):.1f}", backend_name,
                       f"FAILED: {type(exc).__name__}", "-", "-"]
            if args.markdown:
                print("| " + " | ".join(row) + " |")
            else:
                print(f"{row[0]:<28s} {row[1]:>3s} {row[2]:>8s} {row[3]:>7s} "
                      f"{row[4]:>9s} {row[5]:>8s} {row[6]:>8s}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
