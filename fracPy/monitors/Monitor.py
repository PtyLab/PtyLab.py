from .default_visualisation import DefaultMonitor,DiffractionDataMonitor

class Monitor(object):
    """

    """
    def __init__(self):
        # settings for visualization
        self.figureUpdateFrequency = 1
        self.objectPlot = 'complex'
        self.verboseLevel = 'high'
        self.optimizable = None

    def initializeVisualisation(self):
        """
        Create the figure and axes etc.
        :return:
        """
        self.defaultMonitor = DefaultMonitor()
        if self.verboseLevel == 'high':
            self.diffractionDataMonitor = DiffractionDataMonitor()


    def updatePlot(self, object_estimate):
        """
        update initialized plots
        :param object_estimate:
        :return:
        """
        self.defaultMonitor.updateError(self.optimizable.error)
        self.defaultMonitor.updateObject(object_estimate, objectPlot=self.objectPlot, pixelSize=self.optimizable.data.dxo)
        self.defaultMonitor.updateProbe(self.optimizable, pixelSize=self.optimizable.data.dxp)
        self.defaultMonitor.drawNow()
        if self.verboseLevel == 'high':
            self.diffractionDataMonitor.updateIestimated(self.optimizable.Iestimated)
            self.diffractionDataMonitor.updateImeasured(self.optimizable.Imeasured)
            self.diffractionDataMonitor.drawNow()

    def getOverlap(self, ind1,ind2,probePixelsize):
        """
        Get area overlap between position ind1 and ind2
        ( note: the area overlap is only calculated for main mode of probe)
        :return:
        """
        # calculate shifts (positions is row - column order, shifts are xy order)
        sy = abs(self.optimizable.positions(ind2, 1) - self.optimizable.positions(ind1, 1)) * probePixelsize
        sx = abs(self.optimizable.positions(ind2, 2) - self.optimizable.positions(ind1, 2)) * probePixelsize
        raise NotImplementedError()
