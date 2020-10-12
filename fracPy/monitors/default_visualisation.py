import matplotlib as mpl
# mpl.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from fracPy.utils import gpuUtils
from fracPy.utils.visualisation import modeTile, complex_plot, complex_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable

class DefaultMonitor(object):

    def __init__(self, figNum=1):
        """ Create a monitor.

        In principle, to use this method all you have to do is initialize the monitor and then call

        updateObject, updateProbe, updateErrorMetric and drawnow to ensure that something is drawn immediately.

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
        self.figure, axes = plt.subplots(1, 3, num=self.figNum, squeeze=False, clear=True, figsize=(10, 3))
        self.ax_object = axes[0][0]
        self.ax_probe = axes[0][1]
        self.ax_error_metric = axes[0][2]
        self.ax_object.set_title('Object estimate')
        self.txt_purity = self.ax_probe.set_title('Probe estimate')
        # self.txt_purity = plt.text(0,1.2,'',transform = self.ax_probe.transAxes)
        self.ax_error_metric.set_title('Error metric')
        self.ax_error_metric.grid(True)
        self.ax_error_metric.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        self.ax_error_metric.set_xlabel('iterations')
        self.ax_error_metric.set_ylabel('error')
        self.figure.tight_layout()
        self.firstrun = True


    def updateObject(self, object_estimate, objectPlot, **kwargs):
        OE = modeTile(object_estimate, normalize=True)
        if objectPlot == 'complex':
            OE = complex_to_rgb(OE)
        elif objectPlot == 'abs':
            OE = abs(OE)
        elif objectPlot == 'angle':
            OE = np.angle(OE)

        if self.firstrun:
            if objectPlot == 'complex':
                self.im_object = complex_plot(OE, ax=self.ax_object, **kwargs)
            else:
                self.im_object = self.ax_object.imshow(OE, cmap='gray', interpolation=None)
                divider = make_axes_locatable(self.ax_object)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                self.objectCbar = plt.colorbar(self.im_object, ax=self.ax_object, cax=cax)
        else:
            self.im_object.set_data(OE)
        self.im_object.autoscale()

    def updateProbe(self, probe_estimate, optimizable, **kwargs):

        PE = complex_to_rgb(modeTile(probe_estimate, normalize=True))

        if self.firstrun:
            self.im_probe = complex_plot(PE, ax=self.ax_probe, **kwargs)
        else:
            self.im_probe.set_data(PE)
            if optimizable.npsm > 1:
                self.txt_purity.set_text('Probe estimate\nPurity: %i' %(100*optimizable.purity)+'%')
        # self.im_probe.autoscale()
        
    def updateError(self, error_estimate: np.ndarray) -> None:
        """
        Update the error estimate plot.
        :param error_estimate:
        :return:
        """
        if self.firstrun:
            self.error_metric_plot = self.ax_error_metric.plot(error_estimate, 'o-',
                                                               mfc='none')[0]
        else:
            self.error_metric_plot.set_data(range(len(error_estimate)), error_estimate)
            self.ax_error_metric.set_ylim(np.min(error_estimate), np.max(error_estimate))
            self.ax_error_metric.set_xlim(0, len(error_estimate))
        self.ax_error_metric.set_aspect(1/self.ax_error_metric.get_data_ratio())


    def drawNow(self):
        """
        Forces the image to be drawn
        :return:
        """
        if self.firstrun:
            self.figure.show()
            self.firstrun = False

        # Reopen the figure if the window is closed
        if not plt.fignum_exists(self.figNum):
            self.figure.show()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()



class DiffractionDataMonitor(object):
    def __init__(self, figNum=2):
        """ Create a monitor.

        In principle, to use this method all you have to do is initialize the monitor and then call

        updateImeasured, updateIestimated and drawnow to ensure that something is drawn immediately.

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
        self.figure, axes= plt.subplots(1, 2, num=self.figNum, squeeze=False, clear=True, figsize=(8, 3))
        self.ax_Iestimated = axes[0][0]
        self.ax_Imeasured = axes[0][1]
        self.ax_Iestimated.set_title('Estimated intensity')
        self.ax_Imeasured.set_title('Measured intensity')
        self.figure.tight_layout()
        self.firstrun = True


    def updateIestimated(self, Iestimate, cmap='gray',**kwargs):
        # move it to CPU if it's on the GPU
        if hasattr(Iestimate, 'device'):
            Iestimate = Iestimate.get()

        if self.firstrun:
            self.im_Iestimated = self.ax_Iestimated.imshow(np.log10(np.squeeze(Iestimate+1)),
                                                           cmap=cmap, interpolation=None)
            divider = make_axes_locatable(self.ax_Iestimated)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            self.IestimatedCbar = plt.colorbar(self.im_Iestimated, ax=self.ax_Iestimated, cax=cax)

        else:
            self.im_Iestimated.set_data(np.log10(np.squeeze(Iestimate+1)))
        self.im_Iestimated.autoscale()

    def updateImeasured(self, Imeasured, cmap='gray', **kwargs):
        Imeasured = gpuUtils.asNumpyArray(Imeasured)
        if self.firstrun:
            self.im_Imeasured = self.ax_Imeasured.imshow(np.log10(np.squeeze(Imeasured + 1)),
                                                         cmap=cmap, interpolation=None)
            divider = make_axes_locatable(self.ax_Imeasured)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            self.ImeasuredCbar = plt.colorbar(self.im_Imeasured, ax=self.ax_Imeasured, cax=cax)
        else:
            self.im_Imeasured.set_data(np.log10(np.squeeze(Imeasured + 1)))
        self.im_Imeasured.autoscale()

    def drawNow(self):
        """
        Forces the image to be drawn
        :return:
        """
        if self.firstrun:
            self.figure.show()
            self.firstrun = False

        # Reopen the figure if the window is closed
        if not plt.fignum_exists(self.figNum):
            self.figure.show()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
