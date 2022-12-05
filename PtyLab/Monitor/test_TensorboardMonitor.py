from unittest import TestCase
from PtyLab.Monitor.TensorboardMonitor import TensorboardMonitor
import numpy as np


class Test(TestCase):
    def setUp(self):
        self.monitor = TensorboardMonitor(name="testing purposes only")

    def test_diffractionData(self):
        import imageio

        estimated = imageio.imread("imageio:camera.png")
        measured = np.fliplr(estimated)
        for i in range(10):
            self.monitor.i = i
            self.monitor.updateDiffractionDataMonitor(estimated, measured)

    def test_UpdateObjectProbe(self):
        object_estimate = np.random.rand(1, 1280, 1280)
        probe_estimate = np.random.rand(1, 64, 64) * 1j
        for i in range(10):
            self.monitor.updatePlot(object_estimate, probe_estimate)

    def test_updateError(self):
        errors = np.random.rand(100).cumsum()[::-1]
        for e in errors:
            self.monitor.i += 1
            self.monitor.updateObjectProbeErrorMonitor(e)
            # this often happens in various optimizers for some unclear reason
            self.monitor.updateObjectProbeErrorMonitor([e])

    def test_update_z(self):

        z = np.random.rand(100) / 100
        z[0] = 10
        z[30:50] = 0
        z[80:120] = 0
        z = z.cumsum()
        for zi in z:
            self.monitor.i += 1
            self.monitor.update_z(zi)
        # .cumsum()[::-1]

    def test_update_positions(self):
        positions = np.random.rand(100, 2).cumsum(axis=1)
        positions = np.cos(positions)
        scaling = 1.5
        other_positions = positions + np.random.rand(*positions.shape) * 1e-2
        self.monitor.update_positions(positions, scaling * other_positions, scaling)
