import numpy as np
import pytest

from PtyLab.Engines.BaseEngine import BaseEngine
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.Monitor.Plots import ObjectProbeErrorPlot
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import Reconstruction


@pytest.mark.skip(reason="Visual test - requires manual inspection")
class TestMatplotlibMonitor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.monitor = ObjectProbeErrorPlot()

    def test_create_figure(self):
        pass

    def test_live_update(self):
        error_metrics = []
        for k in range(100):
            error_metrics.append(np.random.rand())
            self.monitor.updateObject(np.random.rand(100, 100), objectPlot="abs")
            self.monitor.updateError(error_metrics)
            self.monitor.drawNow()


@pytest.mark.skip(reason="Visual test - requires manual inspection")
class TestPlotFromBaseReconstructor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.experimentalData = ExperimentalData("example:simulationTiny")
        self.params = Params()
        self.monitor = Monitor()
        self.optimizable = Reconstruction(self.experimentalData, self.params)
        self.optimizable.initializeObjectProbe()
        self.BR = BaseEngine(
            self.optimizable, self.experimentalData, self.params, self.monitor
        )

    def test_show_reconstruction(self):
        self.BR.reconstruction.initializeObjectProbe()
        self.BR.monitor.figureUpdateFrequency = 20
        self.BR.showReconstruction(0)
        for i in range(1000):
            self.BR.showReconstruction(i)
