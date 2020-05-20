import unittest
from unittest import TestCase
import numpy as np
from fracPy.utils.utils import fft2c, ifft2c
from numpy.testing import assert_almost_equal


class TestOperators(TestCase):

    def test_fft2c_ifft2c(self):
        E_in = np.random.rand(5,100,100)
        E_temp = fft2c(E_in)
        E_out = ifft2c(E_temp)
        # assert_almost_equal(self.E_in, E_out)
        assert_almost_equal(E_in, abs(E_out))


if __name__ == '__main__':
    unittest.main()