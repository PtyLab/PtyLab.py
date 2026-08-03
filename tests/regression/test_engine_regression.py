"""Golden-output regression tests for the reconstruction engines.

These pin the *numerical output* of each engine so that performance work
(kernel fusion, CUDA graphs, alternative linear algebra) cannot silently change
what the library computes.

Coverage includes a mixed-state configuration, not just single-mode: that
exercises the 6D ``(nlambda, nosm, npsm, nslice, Ny, Nx)`` broadcasting which is
the easiest thing to get wrong when rewriting the update rules, and exactly what
a single-mode-only suite would let through. Polychromatic and multislice
configurations follow once the engines that need them are testable.

To re-record the goldens after an *intended* numerical change::

    PTYLAB_REGEN_GOLDENS=1 uv run pytest tests/regression -q

Review the resulting diff in ``tests/regression/data/`` before committing it.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import DummyMonitor
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.utils.gpuUtils import asNumpyArray

try:
    import cupy

    HAS_GPU = cupy.cuda.is_available()
except Exception:
    HAS_GPU = False

GOLDEN_DIR = Path(__file__).parent / "data"
REGEN = os.environ.get("PTYLAB_REGEN_GOLDENS", "") not in ("", "0")
SEED = 20240607

# name -> engine, propagator, (nlambda, nosm, npsm, nslice), iterations
#
# Engines are added here as they become testable. Still absent: zPIE, aPIE and
# mPIE_tv, which do not run on this branch at all -- see the PRs that repair them.
CONFIGS = {
    "epie_fraunhofer": ("ePIE", "Fraunhofer", (1, 1, 1, 1), 3),
    "epie_asp": ("ePIE", "ASP", (1, 1, 1, 1), 3),
    "epie_fresnel": ("ePIE", "Fresnel", (1, 1, 1, 1), 3),
    "epie_mixed_state": ("ePIE", "Fraunhofer", (1, 2, 3, 1), 3),
    "epie_polychrome": ("ePIE", "polychromeASP", (3, 1, 1, 1), 2),
    "mpie_single": ("mPIE", "Fraunhofer", (1, 1, 1, 1), 3),
    "mpie_mixed_state": ("mPIE", "Fraunhofer", (1, 2, 3, 1), 3),
    "qnewton_single": ("qNewton", "Fraunhofer", (1, 1, 1, 1), 3),
    "e3pie_multislice": ("e3PIE", "Fraunhofer", (1, 1, 1, 3), 2),
}


def build(dataset, config, gpu):
    """Construct a fully-determined reconstruction for ``config``."""
    _engine_name, propagator, (nlambda, nosm, npsm, nslice), _iters = config

    data = ExperimentalData(str(dataset), operationMode="CPM")
    params = Params()
    params.gpuSwitch = gpu
    params.propagatorType = propagator
    # 'random' shuffles positions via the global numpy RNG; pin the order so the
    # golden does not depend on RNG call sequence elsewhere in the engine.
    params.positionOrder = "sequential"

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

    # initialProbeOrObject() adds 0.001 * np.random.rand(...) noise to break mode
    # degeneracy, so the initial guess itself needs the global seed pinned.
    np.random.seed(SEED)
    reconstruction.initializeObjectProbe()

    return data, reconstruction, params, DummyMonitor()


def run(dataset, config, gpu):
    """Run ``config`` to completion and return the arrays worth pinning."""
    from PtyLab import Engines

    engine_name, _propagator, _modes, iters = config
    data, reconstruction, params, monitor = build(dataset, config, gpu)

    engine = getattr(Engines, engine_name)(reconstruction, data, params, monitor)
    engine.numIterations = iters

    # mPIE fires its momentum update on np.random.rand(1) > 0.95; seed again so
    # the decision sequence is fixed regardless of how much RNG setup consumed.
    np.random.seed(SEED)
    engine.reconstruct()

    return {
        "object": asNumpyArray(reconstruction.object),
        "probe": asNumpyArray(reconstruction.probe),
        "error": np.asarray(asNumpyArray(reconstruction.error), dtype=np.float64),
    }


def compare(result, golden_path, rtol, atol, label):
    if REGEN or not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(golden_path, **result)
        pytest.skip(f"recorded golden {golden_path.name}; re-run to verify")

    golden = np.load(golden_path)
    assert sorted(golden.files) == sorted(result), (
        f"{label}: golden holds {sorted(golden.files)} but got {sorted(result)}; "
        f"re-record with PTYLAB_REGEN_GOLDENS=1"
    )
    for key in sorted(result):
        actual, expected = result[key], golden[key]
        assert actual.shape == expected.shape, (
            f"{label}: {key} shape {actual.shape} != golden {expected.shape}"
        )
        np.testing.assert_allclose(
            actual, expected, rtol=rtol, atol=atol,
            err_msg=f"{label}: {key} drifted from golden",
        )


def relative_error(actual, expected):
    """Frobenius-norm relative error.

    Elementwise relative error is the wrong metric here: these arrays contain
    near-zero elements where a 1e-7 absolute wobble reads as a huge relative
    one. The norm ratio measures what actually matters -- whether the
    reconstruction as a whole moved.
    """
    denom = np.linalg.norm(np.asarray(expected).ravel())
    return float(np.linalg.norm((np.asarray(actual) - expected).ravel()) / max(denom, 1e-30))


@pytest.mark.parametrize("name", list(CONFIGS))
def test_engine_cpu_golden(regression_dataset, name):
    """CPU output must match the recorded golden."""
    result = run(regression_dataset, CONFIGS[name], gpu=False)
    compare(result, GOLDEN_DIR / f"{name}.npz", rtol=1e-5, atol=1e-7,
            label=f"{name} [cpu]")


@pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU available")
@pytest.mark.parametrize("name", list(CONFIGS))
def test_engine_gpu_agrees_with_cpu(regression_dataset, name):
    """GPU and CPU backends must agree to within float32 accumulation noise.

    cuFFT and CuPy reductions accumulate in a different order than NumPy, so
    exact equality is not expected. Measured divergence across these configs is
    1e-8 to 2.2e-4; the 1e-3 bound leaves roughly 5x headroom while still being
    tight enough to catch a genuinely wrong kernel.
    """
    cpu = run(regression_dataset, CONFIGS[name], gpu=False)
    gpu = run(regression_dataset, CONFIGS[name], gpu=True)

    for key in ("object", "probe", "error"):
        err = relative_error(gpu[key], cpu[key])
        assert err < 1e-3, (
            f"{name}: GPU {key} diverges from CPU by {err:.2e} "
            f"(relative Frobenius norm, tolerance 1e-3)"
        )
