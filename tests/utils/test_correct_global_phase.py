import numpy as np
from numpy.testing import assert_allclose

from PtyLab.utils.utils import correctGlobalPhase


def random_field(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(shape) + 0.5 + 1j * rng.standard_normal(shape)


def test_removes_the_global_phase_of_each_frame():
    reference = random_field((3, 32, 32))
    # One phase per frame, so each frame is aligned on its own.
    phases = np.array([0.4, -2.0, 3.0])[:, None, None]
    estimate = reference * np.exp(-1j * phases)
    assert_allclose(correctGlobalPhase(estimate, reference), reference, atol=1e-12)


def test_removes_a_ramp_only_when_asked():
    reference = random_field((32, 32))
    rows, cols = np.mgrid[:32, :32] - 16
    estimate = reference * np.exp(-1j * (0.05 * rows - 0.03 * cols + 2.0))
    assert_allclose(
        correctGlobalPhase(estimate, reference, ramp=True), reference, atol=1e-12
    )
    assert not np.allclose(
        correctGlobalPhase(estimate, reference), reference, atol=1e-3
    )


def test_weight_ignores_pixels_outside_the_scan():
    reference = random_field((32, 32))
    estimate = reference * np.exp(-0.7j)
    estimate[:, :8] = random_field((32, 8), seed=1)  # unscanned pixels: noise
    weight = np.ones((32, 32))
    weight[:, :8] = 0
    aligned = correctGlobalPhase(estimate, reference, weight=weight)
    assert_allclose(aligned[:, 8:], reference[:, 8:], atol=1e-12)


def test_keeps_the_dtype():
    reference = random_field((16, 16)).astype(np.complex64)
    assert (
        correctGlobalPhase(reference * np.complex64(1j), reference).dtype
        == np.complex64
    )
