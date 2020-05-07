import matplotlib as mpl
# mpl.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np


class DefaultMonitor(object):

    def __init__(self, figNum=1):
        """ Create a monitor.

        In principle, to use this method all you have to do is initialize the monitor and then call

        updateObject, updateErrorMetric and drawnow to ensure that something is drawn immediately.

        For example usage, see test_matplot_monitor.py.

        """
        self.fig_num = figNum
        self._createFigure()

    def _createFigure(self) -> None:
        """
        Create the figure.
        :return:
        """

        # add an axis for the object
        self.figure, axes= plt.subplots(1, 2, num=self.fig_num, squeeze=False, clear=True)
        self.ax_object = axes[0][0]
        self.ax_error_metric = axes[0][1]
        self.ax_object.set_title('Object estimate')
        self.ax_error_metric.set_title('Error metric')
        self.firstrun = True

    def updateObject(self, object_estimate):
        from fracPy.utils.utils import ifft2c


        #OE = abs(ifft2c(object_estimate))
        OE = abs(object_estimate)
        if OE.ndim == 3:
            # Put the object estimate components next to each other.
            OE = np.hstack(OE)

        if self.firstrun:
            self.im_object = self.ax_object.imshow(OE, cmap='gray')

        else:
            self.im_object.set_data(OE)

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
