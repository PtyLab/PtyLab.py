import pytest
import numpy as np
from numpy.testing import assert_allclose

from PtyLab import Operators, easyInitialize

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False


@pytest.mark.skip(reason="requires cupyx and GPU; performance benchmark only")
def test_caching_aspw():
    xp = cp if HAS_GPU else np
    E = xp.random.rand(10, 1, 3, 512, 512)
    z = 1e-3
    wl = 512e-9
    pixel_pitch = 10e-6
    L = pixel_pitch * E.shape[-1]

    E_prop = Operators.Operators.aspw_cached(E, z, wl, L)
    if HAS_GPU:
        E_prop = xp.asnumpy(E_prop)

    E_prop2 = Operators.Operators.aspw(E, z, wl, L)[0]
    if HAS_GPU:
        E_prop2 = E_prop2.get()

    assert_allclose(E_prop, E_prop2)


def test_object2detector():
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )
    params.gpuSwitch = False
    reconstruction._move_data_to_cpu()
    for operator_name in Operators.Operators.forward_lookup_dictionary:
        params.propagatorType = operator_name
        reconstruction.esw = reconstruction.probe
        Operators.Operators.object2detector(reconstruction.esw, params, reconstruction)


# Unitary propagators: forward followed by inverse must return the input.
UNITARY_PROPAGATORS = ["fraunhofer", "fresnel", "identity"]

# Band-limited propagators: forward followed by inverse is *not* the identity.
# Their transfer functions suppress out-of-band spatial frequencies, and that
# energy is genuinely gone -- measured ||P(x)-x||/||x|| is 3.4e-01 for asp and
# 5.4e-02 for scaledasp.
#
# What that leaves differs between the two, which is why neither is asserted
# here: asp is a clean projection (||P(P(x))-P(x)||/||P(x)|| = 3.1e-07, i.e. a
# hard 0/1 cutoff), while scaledasp only approximately so (1.6e-03) because its
# chirp and rescaling factors are not idempotent. Linearity is the property both
# genuinely share; the numbers themselves are pinned by
# tests/regression/test_propagator_regression.py.
BANDLIMITED_PROPAGATORS = ["asp", "scaledasp"]

# The polychrome variants need spectralDensity with nlambda > 1; they are
# covered end-to-end by tests/regression instead.
ALL_ROUND_TRIP_PROPAGATORS = UNITARY_PROPAGATORS + BANDLIMITED_PROPAGATORS


def _round_trip(field, params, reconstruction):
    reconstruction.esw = field
    _, forward = Operators.Operators.object2detector(field, params, reconstruction)
    _, back = Operators.Operators.detector2object(forward, params, reconstruction)
    return back


@pytest.mark.parametrize("propagator", UNITARY_PROPAGATORS)
def test_propagator_round_trip_is_identity(propagator):
    """Forward then inverse must return the input field.

    This is the property every performance change to the propagators has to
    preserve, and it is what the previous smoke tests (which called the
    operators but asserted nothing) failed to check.
    """
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )
    params.gpuSwitch = False
    params.propagatorType = propagator
    params.fftshiftSwitch = False
    reconstruction._move_data_to_cpu()

    field = reconstruction.probe.copy()
    back = _round_trip(field, params, reconstruction)

    scale = np.abs(field).max()
    assert_allclose(
        back, field, rtol=1e-4, atol=1e-5 * scale,
        err_msg=f"{propagator} does not round-trip",
    )


@pytest.mark.parametrize("propagator", ALL_ROUND_TRIP_PROPAGATORS)
def test_propagator_is_linear(propagator):
    """Every propagator must be a linear operator: P(a*x) == a*P(x).

    For the band-limited propagators this is the strongest property that
    actually holds -- their transfer functions carry amplitude masks
    (|Q| spans [0, 1]), so forward-then-inverse is neither unitary nor
    idempotent. The exact numerical output is pinned separately by
    tests/regression/test_propagator_regression.py.
    """
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )
    params.gpuSwitch = False
    params.propagatorType = propagator
    params.fftshiftSwitch = False
    reconstruction._move_data_to_cpu()

    field = reconstruction.probe.copy()
    alpha = 3.0

    reconstruction.esw = field
    _, base = Operators.Operators.object2detector(field, params, reconstruction)
    scaled_field = (alpha * field).astype(field.dtype)
    reconstruction.esw = scaled_field
    _, scaled = Operators.Operators.object2detector(
        scaled_field, params, reconstruction
    )

    tol = 1e-5 * np.abs(alpha * base).max()
    assert_allclose(
        scaled, alpha * base, rtol=1e-4, atol=tol,
        err_msg=f"{propagator} is not linear",
    )


@pytest.mark.parametrize("propagator", ALL_ROUND_TRIP_PROPAGATORS)
@pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU available")
def test_propagator_gpu_matches_cpu(propagator):
    """The GPU propagator path must agree with the CPU one."""
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )
    params.propagatorType = propagator
    params.fftshiftSwitch = False

    params.gpuSwitch = False
    reconstruction._move_data_to_cpu()
    field_cpu = reconstruction.probe.copy()
    reconstruction.esw = field_cpu
    _, out_cpu = Operators.Operators.object2detector(field_cpu, params, reconstruction)

    field_gpu = cp.asarray(field_cpu)
    reconstruction.esw = field_gpu
    _, out_gpu = Operators.Operators.object2detector(field_gpu, params, reconstruction)

    scale = np.abs(out_cpu).max()
    assert_allclose(
        cp.asnumpy(out_gpu), out_cpu, rtol=1e-4, atol=1e-5 * scale,
        err_msg=f"{propagator}: GPU output disagrees with CPU",
    )


def test_propagate_fresnel():
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )
    reconstruction.initializeObjectProbe()
    reconstruction.esw = 2
    params.gpuSwitch = False
    reconstruction._move_data_to_cpu()

    for operator in [
        Operators.Operators.propagate_fresnel,
        Operators.Operators.propagate_ASP,
        Operators.Operators.propagate_scaledASP,
        Operators.Operators.propagate_twoStepPolychrome,
        Operators.Operators.propagate_scaledPolychromeASP,
    ]:
        operator(reconstruction.probe, params, reconstruction)


@pytest.mark.skip(reason="placeholder - not implemented")
def test_aspw_cached():
    pass


def test_propagate_asp_fft_equivalence():
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        "example:simulation_cpm"
    )
    reconstruction.esw = None
    a = reconstruction.probe
    P1 = Operators.Operators.propagate_ASP(a, params, reconstruction, z=1e-3, fftflag=False)[1]
    P2 = Operators.Operators.propagate_ASP(a, params, reconstruction, z=1e-3, fftflag=True)[1]
    assert_allclose(P1, P2)
