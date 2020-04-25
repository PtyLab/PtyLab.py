from unittest import TestCase

from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.engines.BaseReconstructor import BaseReconstructor


class TestBaseReconstructor(TestCase):

    def setUp(self) -> None:
        # For almost all reconstructor properties we need both a data and an optimizable object.
        self.experimentalData = ExperimentalData('test:nodata')
        self.optimizable = Optimizable(self.experimentalData)
        self.BR = BaseReconstructor(self.optimizable, self.experimentalData)

    def test_change_optimizable(self):
        """
        Make sure the optimizable can be changed
        :return:
        """
        optimizable2 = Optimizable(self.experimentalData)
        self.BR.changeOptimizable(optimizable2)
        self.assertEqual(self.BR.optimizable, optimizable2)

    def test_getErrorMetrics(self):
        '''
        Dont know how to test it
        :return:
        '''
        pass