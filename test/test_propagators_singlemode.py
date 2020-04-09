from unittest import TestCase
from fracPy.physics import propagators_singlemode as pps
import numpy as np
from numpy.testing import assert_almost_equal


# For every method, create a class with a testcase
class TestFresnelPropagator(TestCase):
    def test_fresnelPropagator(self):
        # Implement anything here that enables us to test that it works as expected.

        # for instance: propagating 0 distance leads to no change
        E_in = np.zeros((32,32))
        E_out = pps.fresnelPropagator(E_in, wavelength=500e-6, distance=0,
                                      pixel_size=1e-6)
        assert_almost_equal(E_in, E_out)