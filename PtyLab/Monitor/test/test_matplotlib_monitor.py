from unittest import TestCase
from PtyLab.Monitor.Plots import ObjectProbeErrorPlot
import time
import numpy as np
import unittest

from PtyLab.Engines import BaseEngine

from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.FixedData.DefaultExperimentalData import ExperimentalData
from PtyLab.Engines.BaseEngine import BaseEngine

# To run the tests in this file, set this to TRUE
VISUAL_TESTS = False


@unittest.skipUnless(
    VISUAL_TESTS,
    "Visual tests are disabled by default. To turn them on, set VISUAL_TESTS to true",
)
class TestMatplotlib_monitor(TestCase):
    def setUp(self):
        self.monitor = ObjectProbeErrorPlot()

    def test_createFigure(self):
        pass

    def test_live_update(self):
        error_metrics = []
        for k in range(100):
            error_metrics.append(np.random.rand())
            self.monitor.updateObject(np.random.rand(100, 100))
            self.monitor.updateError(error_metrics)
            self.monitor.drawNow()


@unittest.skipUnless(
    VISUAL_TESTS,
    "Visual tests are disabled by default. To turn them on, set VISUAL_TESTS to true",
)
class TestPlotFromBaseReconstructor(TestCase):
    def setUp(self):
        # For almost all reconstructor properties we need both a data and an reconstruction object.
        self.experimentalData = ExperimentalData("example:simulationTiny")
        self.optimizable = Reconstruction(self.experimentalData)
        self.optimizable.initializeObjectProbe()
        self.BR = BaseEngine(self.optimizable, self.experimentalData)

    def test_showReconstruction(self):
        self.BR.reconstruction.initializeObjectProbe()
        self.BR.figureUpdateFrequency = 20
        self.BR.showReconstruction(0)
        for i in range(1000):
            self.BR.showReconstruction(i)
