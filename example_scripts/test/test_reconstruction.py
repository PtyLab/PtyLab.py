import matplotlib
matplotlib.use('tkagg')
import unittest

from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines import ePIE
from fracPy.monitors.Monitor import Monitor


class test_singlemode_multimode_reconstruction(unittest.TestCase):

    def setUp(self) -> None:
        self.exampleData = ExperimentalData()
        self.exampleData.loadData('example:simulation_ptycho')
        self.exampleData.operationMode = 'CPM'
        self.optimizable = Optimizable(self.exampleData)


        # Set monitor properties
        self.monitor = Monitor()

        # now we want to run an optimizer. First create it.
        self.ePIE_engine = ePIE.ePIE(self.optimizable, self.exampleData, self.monitor)
        self.ePIE_engine.numIterations = 2



    def test_singleModeReconstruction(self):
        self.optimizable.npsm = 1  # Number of probe modes to reconstruct
        self.optimizable.prepare_reconstruction()
        self.ePIE_engine.doReconstruction()

    def test_multiModeReconstruction(self):
        self.optimizable.npsm = 4
        self.optimizable.prepare_reconstruction()
        self.ePIE_engine.doReconstruction()



if __name__ == '__main__':
    unittest.main()
