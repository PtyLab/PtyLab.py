from unittest import TestCase
import logging
logging.basicConfig(level=logging.DEBUG)
from numpy.testing import assert_almost_equal

from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable

class TestOptimizable(TestCase):
    def setUp(self):
        # first, create a ExperimentalData dataset
        data = ExperimentalData('test:nodata')
        #data = ExperimentalData('example:simulation_fpm')

        data.wavelength = 1234
        optimizable = Optimizable(data)
        self.data = data
        self.optimizable = optimizable

    def test_copyAttributesFromExperiment(self):
        self.check_scalar_property()
        self.check_array_property()

    def check_scalar_property(self):

        # check that the wavelength is properly copied
        self.assertEqual(self.optimizable.wavelength, self.data.wavelength)
        # now if we change it, it should no longer be the same
        self.optimizable.wavelength = 4321
        self.assertNotEqual(self.optimizable.wavelength, self.data.wavelength)

    def check_array_property(self):
        """
        Arrays are passed by reference by default. Check that they are properly copied as well
        :return:
        """
        # TODO: This is really really confusing and we should fix it.

        offset = self.data.No / 2 - self.data.Np / 2
        try:
            assert_almost_equal(self.optimizable.positions + offset, self.data.positions)
        except AssertionError:
            print(self.optimizable.positions[:2], self.data.positions[:2])
            raise
        # make sure that they are not pointing to the same thing..
        self.optimizable.positions += 1
        assert_almost_equal(self.optimizable.positions - 1-offset, self.data.positions)

