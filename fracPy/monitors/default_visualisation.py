import matplotlib as mpl
# mpl.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from fracPy.utils.visualisation import modeTile, complex_plot, complex_to_rgb


class DefaultMonitor(object):

    def __init__(self, figNum=1):
        """ Create a monitor.

        In principle, to use this method all you have to do is initialize the monitor and then call

        updateObject, updateErrorMetric and drawnow to ensure that something is drawn immediately.

        For example usage, see test_matplot_monitor.py.

        """
        self.figNum = figNum
        self._createFigure()

    def _createFigure(self) -> None:
        """
        Create the figure.
        :return:
        """

        # add an axis for the object
        self.figure, axes= plt.subplots(1, 3, num=self.figNum, squeeze=False, clear=True)
        self.ax_object = axes[0][0]
        self.ax_probe = axes[0][1]
        self.ax_error_metric = axes[0][2]
        self.ax_object.set_title('Object estimate')
        self.ax_probe.set_title('Probe estimate')
        self.ax_error_metric.set_title('Error metric')
        self.firstrun = True

    def updateObject(self, object_estimate, objectPlot,**kwargs):
        OE = modeTile(object_estimate, normalize=True)
        if objectPlot == 'complex':
            OE = complex_to_rgb(OE)
        elif objectPlot == 'abs':
            OE = abs(OE)
        elif objectPlot == 'angle':
            OE = np.angle(OE)

        if self.firstrun:
            if objectPlot == 'complex':
                self.im_object = complex_plot(OE, ax=self.ax_object)
            else:
                self.im_object = self.ax_object.imshow(OE, cmap='gray')

        else:
            self.im_object.set_data(OE)

    def updateProbe(self, probe_estimate,probeROI = None):

        PE = complex_to_rgb(modeTile(probe_estimate,normalize=True))

        if self.firstrun:
            self.im_probe = complex_plot(PE, ax=self.ax_probe)
        else:
            self.im_probe.set_data(PE)

    def updateError(self, error_estimate: np.ndarray) -> None:
        """
        Update the error estimate plot.
        :param error_estimate:
        :return:
        """
        if self.firstrun:
            self.error_metric_plot = self.ax_error_metric.plot(error_estimate)[0]
        else:
            self.error_metric_plot.set_data(range(len(error_estimate)), error_estimate)
            self.ax_error_metric.set_ylim(0, np.max(error_estimate))
            self.ax_error_metric.set_xlim(0, len(error_estimate))


    def drawNow(self):
        """
        Forces the image to be drawn
        :return:
        """
        if self.firstrun:
            self.figure.show()
            self.firstrun = False
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
