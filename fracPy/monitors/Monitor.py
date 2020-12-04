from .default_visualisation import DefaultMonitor, DiffractionDataMonitor


class Monitor(object):
    """

    """
    def __init__(self):
        # settings for visualization
        self.figureUpdateFrequency = 1
        self.objectPlot = 'complex'
        self.verboseLevel = 'high'
        self.objectPlotZoom = 1
        self.probePlotZoom = 1
        self.objectPlotContrast = 1
        self.probePlotContrast = 1
        self.optimizable = None

    def initializeVisualisation(self):
        """
        Create the figure and axes etc.
        :return:
        """
        self.defaultMonitor = DefaultMonitor()
        if self.verboseLevel == 'high':
            self.diffractionDataMonitor = DiffractionDataMonitor()


    def updateDefaultMonitor(self, object_estimate, probe_estimate):
        """
        update the probe object plots
        :param object_estimate:
        :return:
        """
        self.defaultMonitor.updateError(self.optimizable.error)
        self.defaultMonitor.updateObject(object_estimate, objectPlot=self.objectPlot,
                                         pixelSize=self.optimizable.data.dxo, axisUnit='mm',
                                         amplitudeScalingFactor=self.objectPlotContrast)
        self.defaultMonitor.updateProbe(probe_estimate, self.optimizable,
                                        pixelSize=self.optimizable.data.dxp, axisUnit='mm',
                                        amplitudeScalingFactor=self.probePlotContrast)
        self.defaultMonitor.drawNow()

    def updateDiffractionDataMonitor(self, Iestimated, Imeasured):
        """
        update the diffraction plots
        """
        self.diffractionDataMonitor.updateIestimated(Iestimated)
        self.diffractionDataMonitor.updateImeasured(Imeasured)
        self.diffractionDataMonitor.drawNow()

    def getOverlap(self, ind1,ind2,probePixelsize):
        """
        todo
        Get area overlap between position ind1 and ind2
        ( note: the area overlap is only calculated for main mode of probe)
        :return:
        """
        # calculate shifts (positions is row - column order, shifts are xy order)
        sy = abs(self.optimizable.positions(ind2, 1) - self.optimizable.positions(ind1, 1)) * probePixelsize
        sx = abs(self.optimizable.positions(ind2, 2) - self.optimizable.positions(ind1, 2)) * probePixelsize
        raise NotImplementedError()

class DummyMonitor(object):
    """ Monitor without any visualisation so it won't consume any time """

    def updatePlot(self, object_estimate, probe_estimate):
        pass

    def getOverlap(self, ind1, ind2, probePixelsize):
        pass

    def initializeVisualisation(self):
        pass