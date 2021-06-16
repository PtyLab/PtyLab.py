from .Plots import ObjectProbeErrorPlot, DiffractionDataPlot
from fracPy.utils.visualisation import setColorMap

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
        self.optimizable = None
        self.cmapDiffraction = setColorMap()


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
        self.defaultMonitor.updateError(self.optimizable.error)
        self.defaultMonitor.updateObject(object_estimate, self.optimizable, objectPlot=self.objectPlot,
                                         pixelSize=self.optimizable.dxo, axisUnit='mm',
                                         amplitudeScalingFactor=self.objectPlotContrast)
        self.defaultMonitor.updateProbe(probe_estimate, self.optimizable,
                                        pixelSize=self.optimizable.dxp, axisUnit='mm',
                                        amplitudeScalingFactor=self.probePlotContrast)
        self.defaultMonitor.drawNow()

    def updateDiffractionDataMonitor(self, Iestimated, Imeasured):
        """
        update the diffraction plots
        """
        self.diffractionDataMonitor.updateIestimated(Iestimated, cmap=self.cmapDiffraction)
        self.diffractionDataMonitor.updateImeasured(Imeasured, cmap=self.cmapDiffraction)
        self.diffractionDataMonitor.drawNow()

class DummyMonitor(object):
    """ Monitor without any visualisation so it won't consume any time """

    def updatePlot(self, object_estimate, probe_estimate):
        pass

    def getOverlap(self, ind1, ind2, probePixelsize):
        pass

    def initializeVisualisation(self):
        pass