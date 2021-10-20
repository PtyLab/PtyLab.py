import matplotlib.pyplot as plt
import napari

import numpy as np
from .Plots import ObjectProbeErrorPlot, DiffractionDataPlot
from fracPy.utils.visualisation import setColorMap, complex2rgb


class Monitor(object):
    """
    Monitor contains two submonitors: ObjectProbeErrorPlot (object,probe,error) and DiffractionDataPlot (diffraction
    intensity estimate and measurement)
    """
    def __init__(self):
        # settings for visualization
        self.figureUpdateFrequency = 1
        self.objectPlot = 'complex'
        self.verboseLevel = 'low'
        self.objectZoom = 1
        self.probeZoom = 1
        self.objectPlotContrast = 1
        self.probePlotContrast = 1
        self.reconstruction = None
        self.cmapDiffraction = setColorMap()
        self.defaultMonitor = None


    def initializeMonitors(self):
        """
        Create the figure and axes etc.
        :return:
        """
        # only initialize if it hasn't been done
        if self.defaultMonitor is None:
            self.defaultMonitor = ObjectProbeErrorPlot()

        if self.verboseLevel == 'high':
            self.diffractionDataMonitor = DiffractionDataPlot()


    def updateObjectProbeErrorMonitor(self, error, object_estimate, probe_estimate, zo=None,
                                      purity_probe=None, purity_object=None):
        """
        update the probe object plots
        :param object_estimate:
        :return:
        """
        self.defaultMonitor.updateError(error)#self.reconstruction.error)
        print(f'Object plot: {self.objectPlot}')
        self.defaultMonitor.updateObject(object_estimate, self.reconstruction, objectPlot=self.objectPlot,
                                         pixelSize=self.reconstruction.dxo, axisUnit='mm',
                                         amplitudeScalingFactor=self.objectPlotContrast)
        self.defaultMonitor.updateProbe(probe_estimate, self.reconstruction,
                                        pixelSize=self.reconstruction.dxp, axisUnit='mm',
                                        amplitudeScalingFactor=self.probePlotContrast)
        self.defaultMonitor.update_z(zo)
        self.defaultMonitor.drawNow()

    def updateDiffractionDataMonitor(self, Iestimated, Imeasured):
        """
        update the diffraction plots
        """
        from matplotlib.colors import LogNorm
        self.diffractionDataMonitor.update_view(Iestimated, Imeasured, cmap=self.cmapDiffraction)
        # self.diffractionDataMonitor.updateIestimated(Iestimated, cmap=self.cmapDiffraction)
        # self.diffractionDataMonitor.updateImeasured(Imeasured, cmap=self.cmapDiffraction)
        self.diffractionDataMonitor.drawNow()

class DummyMonitor(object):
    """ Monitor without any visualisation so it won't consume any time """
    objectZoom=1
    probeZoom=1
    # remains from mPIE
    figureUpdateFrequency=1000000
    verboseLevel='low'
    def updatePlot(self, object_estimate, probe_estimate):
        pass

    def getOverlap(self, ind1, ind2, probePixelsize):
        pass

    def initializeVisualisation(self):
        pass

    def initializeMonitors(self):
        pass

    def updateObjectProbeErrorMonitor(self, *args, **kwargs):
        pass

    def updateDiffractionDataMonitor(self, *args, **kwargs):
        pass




class NapariMonitor(DummyMonitor):

    def initializeVisualisation(self):
        self.viewer = napari.Viewer()
        self.viewer.show()

        self.viewer.add_image(name='object estimate', data = np.random.rand(100,100))
        self.viewer.add_image(name='probe estimate', data = np.random.rand(100,100))

        self.Iestimated = self.viewer.add_image(name='I estimated', data = np.random.rand(100,100))
        self.Imeasured = self.viewer.add_image(name='I measured', data = np.random.rand(100,100))
        self.rawdatamonitor = self.viewer.add_image(name='raw data (unset)', data=np.random.rand(100,100), visible=False)

    def initializeMonitors(self):
        self.initializeVisualisation()


    def add_ptychogram(self, experimentalData):
        self.rawdatamonitor.name = 'raw data'
        self.rawdatamonitor.data = experimentalData.ptychogram
        self.rawdatamonitor.visible = True


    def update_probe_image(self, new_probe):
        RGB_probe = complex2rgb(new_probe)
        self.viewer.layers['probe_estimate'].data = RGB_probe

    def update_object_image(self, object_estimate):
        RGB_object = complex2rgb(object_estimate)
        self.viewer.layers['object estimate'].data = RGB_object

    def updatePlot(self, object_estimate, probe_estimate):
        self.update_probe_image(probe_estimate)
        self.update_object_image(object_estimate)

    def updateDiffractionDataMonitor(self, Iestimated, Imeasured):
        self.Iestimated.data = Iestimated
        self.Imeasured.data = Imeasured






