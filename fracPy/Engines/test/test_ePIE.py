from unittest import TestCase

from fracPy.Optimizables.Optimizable import Optimizable
from fracPy.FixedData.DefaultExperimentalData import ExperimentalData
from fracPy.Engines.ePIE_reconstructor import ePIE

class TestEPIE(TestCase):

    def setUp(self) -> None:
        # For almost all reconstructor properties we need both a data and an optimizable object.
        self.experimentalData = ExperimentalData('test:nodata')
        #self.experimentalData = FixedData('example:simulation_fpm')
        self.optimizable = Optimizable(self.experimentalData)
        self.ePIE = ePIE(self.optimizable, self.experimentalData)

    def test_init(self):
        pass
