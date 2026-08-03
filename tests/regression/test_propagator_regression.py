"""Golden-output regression tests for the propagators themselves.

The property tests in ``tests/Operators/test_operators_integration.py`` pin the
mathematical invariants that hold (round-trip identity for the unitary
propagators, linearity for all of them). These pin the actual numbers, which is
what catches a subtly wrong transfer function, a changed FFT convention, or a
dropped ``fftshift``.

Re-record with ``PTYLAB_REGEN_GOLDENS=1`` (see test_engine_regression).
"""

import numpy as np
import pytest

from PtyLab import Operators
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import Reconstruction

from test_engine_regression import GOLDEN_DIR, HAS_GPU, SEED, compare, relative_error

if HAS_GPU:
    import cupy as cp

# scaledpolychromeasp/polychromeasp/twosteppolychrome need nlambda > 1; the rest
# run single-wavelength.
PROPAGATORS = {
    "fraunhofer": 1,
    "fresnel": 1,
    "asp": 1,
    "scaledasp": 1,
    "identity": 1,
    "polychromeasp": 3,
    "scaledpolychromeasp": 3,
    "twosteppolychrome": 3,
}


def _setup(dataset, propagator, nlambda):
    data = ExperimentalData(str(dataset), operationMode="CPM")
    params = Params()
    params.gpuSwitch = False
    params.propagatorType = propagator
    params.fftshiftSwitch = False

    reconstruction = Reconstruction(data, params)
    reconstruction.nlambda = nlambda
    if nlambda > 1:
        base = float(np.atleast_1d(reconstruction.wavelength)[0])
        reconstruction.spectralDensity = base * np.linspace(0.98, 1.02, nlambda)

    np.random.seed(SEED)
    reconstruction.initializeObjectProbe()
    return data, reconstruction, params


@pytest.mark.parametrize("propagator", list(PROPAGATORS))
def test_propagator_output_golden(regression_dataset, propagator):
    nlambda = PROPAGATORS[propagator]
    _data, reconstruction, params = _setup(regression_dataset, propagator, nlambda)

    field = reconstruction.probe.copy()
    reconstruction.esw = field
    _, forward = Operators.Operators.object2detector(field, params, reconstruction)
    # BaseEngine.intensityProjection sets reconstruction.ESW before propagating
    # back; propagate_twoStepPolychrome_inv reads it, so mirror that here.
    reconstruction.ESW = forward
    _, back = Operators.Operators.detector2object(forward, params, reconstruction)

    result = {
        "forward": np.asarray(forward),
        "back": np.asarray(back),
        # zero-size arrays are awkward in npz; store the norms as a cheap
        # scalar summary that fails loudly on any global scaling change
        "error": np.array(
            [np.linalg.norm(forward.ravel()), np.linalg.norm(back.ravel())]
        ),
    }
    compare(result, GOLDEN_DIR / f"propagator_{propagator}.npz",
            rtol=1e-5, atol=1e-7, label=f"propagator {propagator}")


@pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU available")
@pytest.mark.parametrize("propagator", list(PROPAGATORS))
def test_propagator_gpu_matches_cpu(regression_dataset, propagator):
    """Propagating a GPU field must match the CPU result.

    ``params.gpuSwitch`` is deliberately left False while GPU arrays are passed
    in. Device placement has to follow the data, not the global switch --
    ``BaseEngine._checkGPU`` and the engines both move arrays independently of
    when a propagator's cached transfer function is first built, so a
    transfer function built from ``params.gpuSwitch`` lands on the wrong device.

    This covers all four call sites that read the switch, including the three
    polychrome propagators that need nlambda > 1 and so cannot be reached from
    the single-wavelength suite in tests/Operators.
    """
    nlambda = PROPAGATORS[propagator]
    _data, reconstruction, params = _setup(regression_dataset, propagator, nlambda)

    field_cpu = reconstruction.probe.copy()
    reconstruction.esw = field_cpu
    _, out_cpu = Operators.Operators.object2detector(field_cpu, params, reconstruction)

    field_gpu = cp.asarray(field_cpu)
    reconstruction.esw = field_gpu
    _, out_gpu = Operators.Operators.object2detector(field_gpu, params, reconstruction)

    err = relative_error(cp.asnumpy(out_gpu), np.asarray(out_cpu))
    assert err < 1e-3, (
        f"{propagator}: GPU output diverges from CPU by {err:.2e} "
        f"(relative Frobenius norm, tolerance 1e-3)"
    )
