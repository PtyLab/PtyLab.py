from unittest import TestCase

from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.FixedData.DefaultExperimentalData import ExperimentalData
from PtyLab.Engines.ePIE import ePIE


class TestEPIE(TestCase):
    def setUp(self) -> None:
        # For almost all reconstructor properties we need both a data and an reconstruction object.
        self.experimentalData = ExperimentalData("test:nodata")
        # self.experimentalData = FixedData('example:simulation_fpm')
        self.optimizable = Reconstruction(self.experimentalData)
        self.ePIE = ePIE(self.optimizable, self.experimentalData)

    def test_init(self):
        pass
