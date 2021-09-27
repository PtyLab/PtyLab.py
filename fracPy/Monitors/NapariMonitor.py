from fracPy.Monitors.Monitor import DummyMonitor
import napari
import numpy as np
from fracPy.utils.visualisation import complex2rgb

class NapariMonitor(DummyMonitor):
    def __init__(self):
        super(DummyMonitor, self).__init__()
        self.viewer = napari.Viewer()
        self.viewer.show()

    def add_or_update(self, name, item):
        if np.iscomplex(item):
            item = complex2rgb(item)

        try:
            self.viewer.layers[name].data = item
        except KeyError:
            self.viewer.add_image(name=name, data=item)

    def updatePlot(self, object_estimate, probe_estimate):
        self.add_or_update('object estimate', object_estimate)
        self.add_or_update('probe estimate', probe_estimate)

    def initializeMonitors(self, *args, **kwargs):
        self.viewer.show()


