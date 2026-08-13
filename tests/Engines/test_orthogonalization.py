"""Device-independence tests for the mode orthogonalization constraint.

``purityProbe``/``purityObject`` are derived from the eigenvalues returned by
``orthogonalizeModes``, which are a cupy array on the GPU path but a numpy array
whenever the GPU eigendecomposition falls back to the CPU. Anything downstream
that assumes one of the two (e.g. calling ``.get()``) breaks on the other, so
these tests run the constraint on both devices and pin the host-side type.
"""

import numpy as np
import pytest

from PtyLab.Engines.mPIE import mPIE
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import DummyMonitor
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.utils.gpuUtils import asNumpyArray
from PtyLab.utils.utils import orthogonalizeModes

try:
    import cupy

    HAS_GPU = cupy.cuda.is_available()
except Exception:
    HAS_GPU = False

SEED = 20240607


def run_orthogonalized(dataset, nosm, npsm, gpu, iterations=2):
    """Run mPIE with the orthogonalization constraint on at every iteration."""
    data = ExperimentalData(str(dataset), operationMode="CPM")
    params = Params()
    params.gpuSwitch = gpu
    params.propagatorType = "Fraunhofer"
    params.positionOrder = "sequential"
    params.orthogonalizationSwitch = True
    params.orthogonalizationFrequency = 1

    reconstruction = Reconstruction(data, params)
    reconstruction.nosm = nosm
    reconstruction.npsm = npsm

    np.random.seed(SEED)
    reconstruction.initializeObjectProbe()

    engine = mPIE(reconstruction, data, params, DummyMonitor())
    engine.numIterations = iterations
    np.random.seed(SEED)
    engine.reconstruct()

    return reconstruction


@pytest.mark.parametrize(
    "gpu", [False, pytest.param(True, marks=pytest.mark.skipif(not HAS_GPU, reason="no GPU available"))]
)
def test_probe_purity_is_a_host_float(generate_simu_hdf5, gpu):
    reconstruction = run_orthogonalized(generate_simu_hdf5, nosm=1, npsm=3, gpu=gpu)

    assert isinstance(reconstruction.purityProbe, float)
    assert 0.0 < reconstruction.purityProbe <= 1.0
    assert len(reconstruction.purityProbeHist) == 2
    assert all(isinstance(purity, float) for purity in reconstruction.purityProbeHist)


@pytest.mark.parametrize(
    "gpu", [False, pytest.param(True, marks=pytest.mark.skipif(not HAS_GPU, reason="no GPU available"))]
)
def test_object_purity_is_a_host_float(generate_simu_hdf5, gpu):
    reconstruction = run_orthogonalized(generate_simu_hdf5, nosm=2, npsm=1, gpu=gpu)

    assert isinstance(reconstruction.purityObject, float)
    assert 0.0 < reconstruction.purityObject <= 1.0


def test_orthogonalized_modes_are_orthogonal():
    """The modes that come out have to actually be mutually orthogonal."""
    rng = np.random.default_rng(3)
    p = (rng.normal(size=(4, 32, 32)) + 1j * rng.normal(size=(4, 32, 32))).astype(
        np.complex64
    )

    modes, normalizedEigenvalues, _ = orthogonalizeModes(p, method="snapShots")

    gram = modes.reshape(4, -1).conj() @ modes.reshape(4, -1).T
    offdiagonal = np.abs(gram - np.diag(np.diag(gram))).max()
    assert offdiagonal / np.abs(np.diag(gram)).max() < 1e-6
    # descending, as the callers assume when they read off the dominant mode
    assert np.all(np.diff(normalizedEigenvalues) <= 0)
    assert np.isclose(np.sum(normalizedEigenvalues), 1.0)


@pytest.mark.skipif(not HAS_GPU, reason="no GPU available")
def test_cusolver_failure_falls_back_to_the_host():
    """A GPU decomposition that raises must not lose the reconstruction.

    cuSOLVER fails for reasons unrelated to the data -- a CUDA install that
    cannot load libcusolver, or a memory pool that has left it no workspace --
    so the fallback has to produce the same modes, back on the device.
    """
    rng = np.random.default_rng(3)
    p = (rng.normal(size=(4, 32, 32)) + 1j * rng.normal(size=(4, 32, 32))).astype(
        np.complex64
    )

    on_device, _, _ = orthogonalizeModes(cupy.asarray(p), method="snapShots")

    original_svd = cupy.linalg.svd
    cupy.linalg.svd = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("simulated cuSOLVER failure")
    )
    try:
        fallback, _, _ = orthogonalizeModes(cupy.asarray(p), method="snapShots")
    finally:
        cupy.linalg.svd = original_svd

    assert isinstance(fallback, cupy.ndarray), "must come back on the device"
    scale = float(np.abs(asNumpyArray(on_device)).max())
    assert np.allclose(
        asNumpyArray(on_device), asNumpyArray(fallback), rtol=1e-4, atol=1e-4 * scale
    )
