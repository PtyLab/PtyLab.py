"""Regression tests for the OPR engine's orthogonalization.

OPR calls ``cp.*`` directly, so unlike the other engines it has no CPU path and
these tests only run where a GPU is present. The algorithm-level properties of
the replacement factorization are covered on the CPU in
``tests/Engines/test_opr_linalg.py``.

Two things are pinned here:

* the **legacy** path (``OPR_tsvd_type="numpy"``, ``OPR_fast_orthogonalization``
  off) against a recorded golden, so the documented escape hatch back to
  pre-0.3.0 output is a tested claim rather than an assertion; and
* the **new defaults** against that same legacy run, bounded by the algorithm's
  own measured sensitivity to its inputs.
"""

import numpy as np
import pytest

from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import DummyMonitor
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.utils.gpuUtils import asNumpyArray

from test_engine_regression import GOLDEN_DIR, HAS_GPU, SEED, compare, relative_error

ITERATIONS = 3
N_MODES = 2


def _run(dataset, tsvd_type, fast_orthogonalization):
    from PtyLab import Engines

    data = ExperimentalData(str(dataset), operationMode="CPM")
    params = Params()
    params.gpuSwitch = True
    params.propagatorType = "Fraunhofer"
    params.positionOrder = "sequential"
    params.OPR_tsvd_type = tsvd_type
    params.OPR_fast_orthogonalization = fast_orthogonalization
    params.OPR_modes = np.arange(N_MODES)
    params.OPR_subspace = 2

    reconstruction = Reconstruction(data, params)
    reconstruction.npsm = N_MODES

    np.random.seed(SEED)
    reconstruction.initializeObjectProbe()

    engine = Engines.OPR(reconstruction, data, params, DummyMonitor())
    engine.numIterations = ITERATIONS
    engine.OPR_modes = params.OPR_modes
    engine.n_subspace = params.OPR_subspace

    np.random.seed(SEED)
    engine.reconstruct()

    return {
        "object": asNumpyArray(reconstruction.object),
        "probe": asNumpyArray(reconstruction.probe),
        "error": np.asarray(asNumpyArray(reconstruction.error), dtype=np.float64),
    }


@pytest.mark.skipif(not HAS_GPU, reason="OPR is GPU-only")
def test_opr_legacy_path_golden(regression_dataset):
    """OPR_tsvd_type='numpy' must still reproduce the pre-0.3.0 output."""
    result = _run(regression_dataset, "numpy", False)
    compare(result, GOLDEN_DIR / "opr_gpu.npz", rtol=1e-5, atol=1e-7,
            label="opr [legacy]")


@pytest.mark.skipif(not HAS_GPU, reason="OPR is GPU-only")
def test_opr_gram_matches_legacy_on_gauge_invariant_quantities(regression_dataset):
    """The new defaults must agree with the legacy path on everything physical.

    A naive elementwise comparison of the probe reports a relative error of
    2.0 -- which looks catastrophic and is not. Mode vectors are defined only up
    to a per-mode global phase, and the Gram route makes a different arbitrary
    choice than LAPACK does. A phase flip on one mode is exactly relative
    error 2.

    So this asserts the quantities that do not depend on that choice:

    * the **object** and the **error metric**, which are what the reconstruction
      actually delivers;
    * the **per-mode powers**, which carry the mixed-state physics;
    * ``|probe|``, the mode amplitudes.

    Measured on this fixture: object 1.4e-07, error metric 4.6e-08, mode powers
    4.1e-07 and 1.1e-06, ``|probe|`` 8.0e-07. After aligning the per-mode phase,
    the mode vectors themselves agree to 8.2e-07 and 1.7e-04 -- so the
    difference really is only the gauge.

    Over a longer run the object does drift, because the solver amplifies any
    perturbation: at 364 px / 202 frames / 30 iterations the Gram route moves
    the object by 1.6e-02, while perturbing the *initial object* of the
    unmodified legacy path by a relative 1e-6 moves it by 2.8e-02. The change
    stays inside the algorithm's own sensitivity to its inputs.
    """
    legacy = _run(regression_dataset, "numpy", False)
    gram = _run(regression_dataset, "gram", True)

    assert relative_error(gram["object"], legacy["object"]) < 1e-4
    assert relative_error(gram["error"], legacy["error"]) < 1e-4
    assert relative_error(np.abs(gram["probe"]), np.abs(legacy["probe"])) < 1e-4

    for mode in range(legacy["probe"].shape[2]):
        a = legacy["probe"][0, 0, mode, 0]
        b = gram["probe"][0, 0, mode, 0]
        power_a, power_b = np.linalg.norm(a), np.linalg.norm(b)
        assert abs(power_b - power_a) / power_a < 1e-4, (
            f"mode {mode} power changed: {power_a:.6g} -> {power_b:.6g}"
        )

        # align the arbitrary global phase before comparing the vectors
        overlap = np.vdot(a.ravel(), b.ravel())
        phase = overlap / abs(overlap) if abs(overlap) > 0 else 1.0
        residual = np.linalg.norm((b - phase * a).ravel()) / power_a
        assert residual < 1e-3, (
            f"mode {mode} differs by more than a global phase: {residual:.2e}"
        )
