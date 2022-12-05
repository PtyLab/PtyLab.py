import unittest
from unittest import TestCase
from numpy.testing import assert_almost_equal
from PtyLab.FixedData.DefaultExperimentalData import ExperimentalData
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.Engines import ePIE, mPIE, qNewton
from PtyLab.Monitor.Monitor import Monitor as Monitor


class TestPropagator(TestCase):
    def test_FresnelPropagator(self):
        exampleData = ExperimentalData()
        exampleData.loadData("example:simulation_ptycho")
        exampleData.operationMode = "CPM"

        # now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
        # now create an object to hold everything we're eventually interested in
        optimizable = Reconstruction(exampleData)
        optimizable.npsm = 1  # Number of probe modes to reconstruct
        optimizable.nosm = 1  # Number of object modes to reconstruct
        optimizable.nlambda = 1  # Number of wavelength
        optimizable.initializeObjectProbe()

        # this will copy any attributes from experimental data that we might care to optimize
        # # Set monitor properties
        monitor = Monitor()

        # Compare mPIE to ePIE
        # ePIE_engine = ePIE.ePIE_GPU(reconstruction, experimentalData, monitor)
        ePIE_engine = ePIE.ePIE(optimizable, exampleData, monitor)
        ePIE_engine.propagatorType = "ASP"
        ePIE_engine.numIterations = 1
        ePIE_engine.reconstruct()

        A = optimizable.esw
        ePIE_engine.object2detector()
        ePIE_engine.detector2object()

        assert_almost_equal(A, optimizable.esw)


if __name__ == "__main__":
    unittest.main()
