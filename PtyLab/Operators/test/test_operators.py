from unittest import TestCase
from PtyLab.Operators.Operators import aspw, scaledASP, scaledASPinv, fresnelPropagator
import numpy as np
from PtyLab.utils.utils import circ
from numpy.testing import assert_almost_equal
import unittest


class TestOperators(TestCase):
    def setUp(self) -> None:
        self.dx = 5e-6
        N = 100
        x = np.arange(-N / 2, N / 2) * self.dx
        X, Y = np.meshgrid(x, x)
        self.E_in = circ(X, Y, N / 2 * self.dx)
        self.wavelength = 600e-9
        self.z = 1e-4
        self.L = self.dx * N

    def test_aspw(self):
        E_1, _ = aspw(self.E_in, 0, self.wavelength, self.L)
        E_2, _ = aspw(E_1, self.z, self.wavelength, self.L)
        E_3, _ = aspw(E_2, -self.z, self.wavelength, self.L)
        assert_almost_equal(self.E_in, E_1)
        assert_almost_equal(abs(E_1), abs(E_3))

    def test_scaledASP(self):
        E_1, _, _ = scaledASP(self.E_in, self.z, self.wavelength, self.dx, self.dx)
        E_2, _, _ = scaledASP(E_1, -self.z, self.wavelength, self.dx, self.dx)
        assert_almost_equal(abs(E_2), abs(self.E_in))

    @unittest.skip()
    def test_fresnelPropagator(self):
        E_out = fresnelPropagator(self.E_in, 0, self.wavelength, self.L)
        assert_almost_equal(self.E_in, E_out)


if __name__ == "__main__":
    unittest.main()
