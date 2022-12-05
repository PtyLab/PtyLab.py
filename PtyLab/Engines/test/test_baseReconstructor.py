from unittest import TestCase

from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.FixedData.DefaultExperimentalData import ExperimentalData
from PtyLab.Engines.BaseEngine import BaseEngine


class TestBaseReconstructor(TestCase):
    def setUp(self) -> None:
        # For almost all reconstructor properties we need both a data and an reconstruction object.
        self.experimentalData = ExperimentalData("test:nodata")
        self.optimizable = Reconstruction(self.experimentalData)
        self.BR = BaseEngine(self.optimizable, self.experimentalData)

    def test_change_optimizable(self):
        """
        Make sure the reconstruction can be changed
        :return:
        """
        optimizable2 = Reconstruction(self.experimentalData)
        self.BR.changeOptimizable(optimizable2)
        self.assertEqual(self.BR.reconstruction, optimizable2)

    def test_setPositionOrder(self):
        """
        Make sure the position of positionIndices is set 'sequential' or 'random'
        :return:
        """
        pass

    def test_getErrorMetrics(self):
        """
        Dont know how to test it
        :return:
        """
        pass
