import numpy as np

from .Plots import ObjectProbeErrorPlot, DiffractionDataPlot
from fracPy.utils.visualisation import setColorMap

class DummyMonitor(object):
    """ Monitor without any visualisation so it won't consume any time """

    def updatePlot(self, object_estimate, probe_estimate):
        pass

    def getOverlap(self, ind1, ind2, probePixelsize):
        pass

    def initializeVisualisation(self):
        pass

class Monitor(DummyMonitor):
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

    def _setObjectROI(self, positions_range: np.ndarray, positions_center: np.ndarray, No: int, Np: int):
        # object
        positions_range = np.subtract(positions_range[0], positions_range[1])
        range_ = ((positions_range + Np)/self.objectZoom).astype(int)
        center = positions_center
        start = np.clip(center-range_//2, 0, No).astype(int)
        end = np.clip(start + range_, 0, No).astype(int)
        sy, sx = [slice(s, e) for (s,e) in zip(start,end)]
        self.objectROI = [sy, sx]

    def _setProbeROI(self, entrancePupilDiameter,  dx_probe, Np):
        r = np.int(entrancePupilDiameter / dx_probe / self.probeZoom)
        center = Np//2
        start = np.clip(center - r//2, 0, Np).astype(int)
        end = np.clip(start + r, 0, Np).astype(int)
        ss = slice(start, end)
        self.probeROI = [ss,ss]

    def initializeMonitors(self):
        """
        Create the figure and axes etc.
        :return:
        """
        self.defaultMonitor = ObjectProbeErrorPlot()
        if self.verboseLevel == 'high':
            self.diffractionDataMonitor = DiffractionDataPlot()


    def updateObjectProbeErrorMonitor(self, object_estimate, probe_estimate):
        """
        update the probe object plots
        :param object_estimate:
        :return:
        """
        self.defaultMonitor.updateError(self.reconstruction.error)
        self.defaultMonitor.updateObject(object_estimate, self.reconstruction, objectPlot=self.objectPlot,
                                         pixelSize=self.reconstruction.dxo, axisUnit='mm',
                                         amplitudeScalingFactor=self.objectPlotContrast)
        self.defaultMonitor.updateProbe(probe_estimate, self.reconstruction,
                                        pixelSize=self.reconstruction.dxp, axisUnit='mm',
                                        amplitudeScalingFactor=self.probePlotContrast)
        self.defaultMonitor.drawNow()

    def updateDiffractionDataMonitor(self, Iestimated, Imeasured):
        """
        update the diffraction plots
        """
        self.diffractionDataMonitor.updateIestimated(Iestimated, cmap=self.cmapDiffraction)
        self.diffractionDataMonitor.updateImeasured(Imeasured, cmap=self.cmapDiffraction)
        self.diffractionDataMonitor.drawNow()

