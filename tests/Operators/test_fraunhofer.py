"""CPU validation of the discrete Fraunhofer operator, without reconstruction setup.

These checks establish transform conventions, not far-field validity for a
particular experiment. Physical sampling and analytical beams are separate checks.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from PtyLab.Operators import Operators as operators


@pytest.fixture(params=[31, 32], ids=["odd", "even"])
def size(request):
    return request.param


@pytest.fixture(params=[np.complex64, np.complex128], ids=["complex64", "complex128"])
def precision(request):
    # Allow float32 FFT/reduction roundoff while detecting scaling/ordering errors.
    dtype = request.param
    return dtype, 2e-6 if dtype == np.complex64 else 2e-13


@pytest.fixture(params=[(), (1, 2, 3, 1)], ids=["single-field", "multiple-modes"])
def leading_shape(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["centered", "unshifted"])
def params(request):
    return SimpleNamespace(fftshiftSwitch=request.param, propagatorType="Fraunhofer")


def numpy_reference(field, unshifted):
    """Independent NumPy expression: default FFT scaling, explicit unitary factor."""
    axes = (-2, -1)
    source = field if unshifted else np.fft.ifftshift(field, axes=axes)
    spectrum = np.fft.fft2(source, axes=axes)
    spectrum /= np.sqrt(field.shape[-2] * field.shape[-1])
    return spectrum if unshifted else np.fft.fftshift(spectrum, axes=axes)


def test_fraunhofer_transform_contract(size, precision, leading_shape, params):
    dtype, tolerance = precision
    rng = np.random.default_rng(42)
    shape = leading_shape + (size, size)
    field = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
    detector_test = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
    before = field.copy()
    detector_before = detector_test.copy()
    reconstruction = SimpleNamespace(esw=field)

    original, forward = operators.propagate_fraunhofer(field, params, reconstruction)
    inverse_original, recovered = operators.propagate_fraunhofer_inv(
        forward, params, reconstruction
    )
    adjoint = operators.propagate_fraunhofer_inv(
        detector_test, params, reconstruction
    )[1]

    reference = numpy_reference(field, params.fftshiftSwitch)
    assert forward.shape == field.shape
    assert original is field
    assert inverse_original is field
    assert np.linalg.norm(forward - reference) / np.linalg.norm(reference) < tolerance
    assert np.linalg.norm(recovered - field) / np.linalg.norm(field) < tolerance

    # Check each mode independently, not only total energy across all modes.
    energy = np.sum(np.abs(field) ** 2, axis=(-2, -1), dtype=np.float64)
    output_energy = np.sum(np.abs(forward) ** 2, axis=(-2, -1), dtype=np.float64)
    assert_allclose(output_energy, energy, rtol=tolerance, atol=0)
    adjoint_error = abs(np.vdot(forward, detector_test) - np.vdot(field, adjoint))
    assert adjoint_error / (np.linalg.norm(field) * np.linalg.norm(detector_test)) < tolerance
    assert_array_equal(field, before)
    assert_array_equal(detector_test, detector_before)

    # Exercise the engine-facing dispatcher, including its None-field fallback.
    reconstruction.ESW = forward
    for supplied in (field, None):
        original, dispatched = operators.object2detector(supplied, params, reconstruction)
        assert original is field
        assert_array_equal(dispatched, forward)
    for supplied in (forward, None):
        original, dispatched = operators.detector2object(supplied, params, reconstruction)
        assert original is field
        assert_array_equal(dispatched, recovered)


def test_fraunhofer_tilt_has_expected_frequency_peak(size, params):
    """An integer-bin plane wave fixes sign, axis order, centering and amplitude."""
    coordinates = np.arange(size) - size // 2
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    field = np.exp(2j * np.pi * (3 * x - 2 * y) / size)
    expected = np.zeros((size, size), dtype=complex)
    expected[size // 2 - 2, size // 2 + 3] = size
    if params.fftshiftSwitch:
        field = np.fft.ifftshift(field)
        expected = np.fft.ifftshift(expected)
    actual = operators.propagate_fraunhofer(
        field, params, SimpleNamespace(esw=field)
    )[1]
    assert_allclose(actual, expected, rtol=2e-13, atol=2e-13 * size)
