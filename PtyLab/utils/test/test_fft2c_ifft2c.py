import unittest
from unittest import TestCase
import numpy as np
from PtyLab.utils.utils import fft2c, ifft2c
from numpy.testing import assert_almost_equal


class TestOperators(TestCase):
    def test_fft2c_ifft2c(self):
        """
        Test that fft2c and ifft2c are unitary.
        :return:
        """
        E_in = (
            np.random.rand(5, 100, 100) + 1j * np.random.rand(5, 100, 100) - 0.5 - 0.5j
        )
        # FFT(IFFT(x)) == x
        assert_almost_equal(ifft2c(fft2c(E_in)), E_in)
        # and vice versa
        assert_almost_equal(fft2c(ifft2c(E_in)), E_in)
        # assert_almost_equal(E_in, abs(E_out))


if __name__ == "__main__":
    unittest.main()
