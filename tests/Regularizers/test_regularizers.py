import pytest
import numpy as np
from numpy.testing import assert_allclose
from PtyLab.Regularizers import TV, divergence, grad_TV


@pytest.fixture
def complex_object():
    shape = (1, 1, 1, 1, 380, 380)
    obj = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    return obj - (0.5 + 0.5j)


def test_grad_tv_matches_divergence(complex_object):
    epsilon = 1e-2
    gradient = np.gradient(complex_object, axis=(4, 5))
    norm = (gradient[0] + gradient[1]) ** 2
    temp = [gradient[0] / np.sqrt(norm + epsilon), gradient[1] / np.sqrt(norm + epsilon)]
    TV_update = divergence(temp)
    TV_update_2 = grad_TV(complex_object, epsilon)
    assert_allclose(TV_update, TV_update_2)


def test_tv_matches_reference(complex_object):
    aleph = 1e-3

    # pad the last two axes with wrap-around so every pixel has 4 neighbours
    pad = [(0, 0)] * (complex_object.ndim - 2) + [(1, 1), (1, 1)]
    p = np.pad(complex_object, pad, mode="wrap")
    grad_x = p[..., 1:-1, 2:] - p[..., 1:-1, :-2]  # right - left
    grad_y = p[..., 2:, 1:-1] - p[..., :-2, 1:-1]  # below - above
    expected = np.sum(np.sqrt(abs(grad_x) ** 2 + abs(grad_y) ** 2 + aleph))

    assert_allclose(TV(complex_object, aleph), expected)
