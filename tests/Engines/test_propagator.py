import pytest
from numpy.testing import assert_almost_equal

from PtyLab.Engines.ePIE import ePIE
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import Reconstruction


@pytest.mark.skip(reason="missing example data file: simulation_ptycho")
def test_fresnel_propagator_round_trip():
    exampleData = ExperimentalData()
    exampleData.loadData("example:simulation_ptycho")
    exampleData.operationMode = "CPM"

    params = Params()
    params.propagatorType = "ASP"

    optimizable = Reconstruction(exampleData, params)
    optimizable.npsm = 1
    optimizable.nosm = 1
    optimizable.nlambda = 1
    optimizable.initializeObjectProbe()

    monitor = Monitor()
    ePIE_engine = ePIE(optimizable, exampleData, params, monitor)
    ePIE_engine.numIterations = 1
    ePIE_engine.reconstruct()

    A = optimizable.esw
    ePIE_engine.object2detector()
    ePIE_engine.detector2object()

    assert_almost_equal(A, optimizable.esw)
