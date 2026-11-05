"""Regression checks for Fresnel phase sampling on the centered FFT grid.

Guard against endpoint-inclusive coordinates that alter the pixel spacing
and displace the phase origin on even grids.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from PtyLab.Operators._propagation_kernels import __make_quad_phase
from PtyLab.Operators.Operators import propagate_fresnel


@pytest.mark.parametrize("size", [31, 32], ids=["odd", "even"])
def test_quadratic_phase_uses_centered_pixel_coordinates(size):
    """The phase origin and pixel spacing must match the centered FFT grid."""
    wavelength, distance, pitch = 532e-9, 0.02, 4e-6
    coordinates = (np.arange(size) - size // 2) * pitch
    x, y = np.meshgrid(coordinates, coordinates)
    expected = np.exp(1j * np.pi * (x**2 + y**2) / (wavelength * distance))
    actual = __make_quad_phase(distance, wavelength, size, pitch, False)
    assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("size", [511, 512], ids=["odd", "even"])
def test_fresnel_gaussian_matches_finite_distance_intensity(size):
    """A centered waist must broaden analytically without acquiring a tilt.

    At z=2 cm the 80 um waist is well resolved (20 source pixels per waist),
    the source window exceeds 25 waists, and the detector window contains
    the propagated beam. An independent correctly sampled Fresnel integral
    verifies that truncation/discretization is below the chosen tolerance.
    This compares intensity only: the operator omits the detector phase.
    """
    wavelength, distance, pitch, waist = 532e-9, 0.02, 4e-6, 80e-6
    coordinates = (np.arange(size) - size // 2) * pitch
    x, y = np.meshgrid(coordinates, coordinates)
    field = np.exp(-(x**2 + y**2) / waist**2).astype(np.complex128)
    detector_coordinates = (
        np.fft.fftshift(np.fft.fftfreq(size, d=pitch)) * wavelength * distance
    )
    qx, qy = np.meshgrid(detector_coordinates, detector_coordinates)
    rayleigh_range = np.pi * waist**2 / wavelength
    width = waist * np.sqrt(1 + (distance / rayleigh_range)**2)
    expected = np.exp(-2 * (qx**2 + qy**2) / width**2)

    phase = np.exp(1j * np.pi * (x**2 + y**2) / (wavelength * distance))
    reference_field = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(field * phase), norm="ortho")
    )
    reference_intensity = np.abs(reference_field)**2
    reference_intensity /= reference_intensity.max()
    assert np.linalg.norm(reference_intensity - expected) / np.linalg.norm(expected) < 1e-10

    reconstruction = SimpleNamespace(esw=field, wavelength=wavelength, zo=distance, dxp=pitch)
    _, output = propagate_fresnel(
        field, SimpleNamespace(fftshiftSwitch=False), reconstruction
    )
    intensity = np.abs(output)**2
    intensity /= intensity.max()
    error = np.linalg.norm(intensity - expected) / np.linalg.norm(expected)
    cx = np.sum(qx * intensity) / intensity.sum()
    cy = np.sum(qy * intensity) / intensity.sum()
    assert error < 1e-10, (
        f"Relative intensity error={error:.6g}; "
        f"centroid=({cx * 1e6:.6g}, {cy * 1e6:.6g}) um"
    )
    assert abs(cx) / width < 1e-10
    assert abs(cy) / width < 1e-10
