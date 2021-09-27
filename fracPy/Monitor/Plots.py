import matplotlib as mpl
# mpl.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np

from fracPy import Reconstruction
from fracPy.utils import gpuUtils
from fracPy.utils.visualisation import modeTile, complexPlot, complex2rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ObjectProbeErrorPlot(object):

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
        self.figure, axes = plt.subplots(1, 3, num=self.figNum, squeeze=False, clear=True, figsize=(10, 3),
                                         gridspec_kw={'width_ratios': [1,1/3,1],
                                                      })


        self.ax_object: plt.Axes = axes[0][0]

        self.ax_probe = axes[0][1]
        self.ax_error_metric = axes[0][2]
        # self.ax_object.set_title
        self.txt_purityProbe = self.ax_probe.set_title('Probe estimate')
        self.txt_purityObject = self.ax_object.set_title('Object estimate')
        self.ax_error_metric.set_title('Error metric')
        self.ax_error_metric.grid(True)
        self.ax_error_metric.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        self.ax_error_metric.set_xlabel('iterations')
        self.ax_error_metric.set_ylabel('error')
        self.ax_error_metric.set_xscale('log')
        self.ax_error_metric.set_yscale('log')
        self.ax_error_metric.axis('image')
        self.figure.tight_layout()
        self.firstrun = True

        # normally this is where you'd insert the probe insert, but we don't as we don't know the relative sizes just yet.

        #self.ax_probe_inset.set_visible(False)

    @property
    def probe_inset(self):
        return self.ax_probe_inset.get_visible()

    @probe_inset.setter
    def probe_inset(self, new_value):
        self.ax_probe_inset.set_visible(new_value)


    def extract_roi_object(self, object_estimate, reconstruction: Reconstruction, zoom=1):
        rx, ry = ((np.max(reconstruction.positions, axis=0) - np.min(reconstruction.positions, axis=0) \
                   + reconstruction.Np) / zoom).astype(int)
        xc, yc = ((np.max(reconstruction.positions, axis=0) + np.min(reconstruction.positions, axis=0) \
                   + reconstruction.Np) / 2).astype(int)

        sx, sy = [slice(max(0, yc - ry // 2),
                                        min(reconstruction.No, yc + ry // 2)),
                                  slice(max(0, xc - rx // 2),
                                        min(reconstruction.No, xc + rx // 2))]

        return object_estimate[...,sy, sx]

    def extract_roi_probe(self, probe_estimate, reconstruction: Reconstruction, zoom=1):
        N = probe_estimate.shape[-1]
        start = np.max([0, N//2-N//2/zoom]).astype(int)
        end = np.min([N, N//2+N//2/zoom]).astype(int)


        # #r = np.int(self.experimentalData.entrancePupilDiameter / self.reconstruction.dxp / self.monitor.probeZoom)
        # #sy, sx = [slice(max(0, reconstruction.Np // 2 - r),
        #                                min(self.reconstruction.Np, self.reconstruction.Np // 2 + r)),
        #                          slice(max(0, self.reconstruction.Np // 2 - r),
        #                                min(self.reconstruction.Np, self.reconstruction.Np // 2 + r))]
        # N = probe_estimate.shape[-1]
        # dN = N //2 - N/2/zoom
        # sy = slice(N//2-dN)
        return probe_estimate[...,start:end, start:end]

    def updateObject(self, object_estimate, optimizable:Reconstruction, objectPlot, amplitudeScalingFactor=1, zoom=1, **kwargs):
        # change the size if the zoom is given
        self.object_npix = object_estimate.shape[-1]
        object_estimate = self.extract_roi_probe(object_estimate, optimizable, zoom)
        OE = modeTile(object_estimate, normalize=True)

        if objectPlot == 'complex':
            OE = complex2rgb(OE, amplitudeScalingFactor=amplitudeScalingFactor)

        elif objectPlot == 'abs':
            OE = abs(OE)
        elif objectPlot == 'angle':
            OE = np.angle(OE)

        if self.firstrun:
            if objectPlot == 'complex':
                self.im_object = complexPlot(OE, ax=self.ax_object, **kwargs)
                self.add_encoder_positions(optimizable.positions0/1e6)
            else:
                self.im_object = self.ax_object.imshow(OE, cmap='gray', interpolation=None)
                divider = make_axes_locatable(self.ax_object)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                self.objectCbar = plt.colorbar(self.im_object, ax=self.ax_object, cax=cax)
        else:
            self.im_object.set_data(OE)
            if optimizable.nosm >1:
                self.txt_purityObject.set_text('Object estimate\nPurity: %i' % (100 * optimizable.purityObject) + '%')



        self.im_object.autoscale()

    def updateProbe(self, probe_estimate, optimizable, amplitudeScalingFactor=1, zoom=1, zoom_object=1, **kwargs):
        self.probe_npix = probe_estimate.shape[-1]
        probe_estimate = self.extract_roi_probe(probe_estimate, optimizable, zoom)
        PE = complex2rgb(modeTile(probe_estimate, normalize=True), amplitudeScalingFactor=amplitudeScalingFactor)

        # PE1 = complex2rgb(probe_estimate)


        if self.firstrun:
            self.im_probe = complexPlot(PE, ax=self.ax_probe, **kwargs)
            # only show one plot
            try:
                relative_size = self.probe_npix / self.object_npix * zoom / zoom_object

                print('Relative size is', relative_size)
            except AttributeError:
                print('Self.object_npix is not available yet. Please first run updateObject before running updateProbe')
            position_orig = self.ax_probe.get_position()

            height = position_orig.height * relative_size
            position_new = [position_orig.x0, position_orig.y0, height, height]
            self.ax_probe.set_position(position_new)
            self.ax_probe.set_title(f'Probe [{zoom/zoom_object} x larger than object')
            # self.ax_probe_inset = self.ax_object.inset_axes([0.0, 0.0, relative_size, relative_size], transform=self.ax_object.transAxes)
            # self.ax_probe_inset.set_title('Probe inset')
            # print(probe_estimate.shape)
            # get only the largest coefficient in case of a larger probe
            # self.im_probe_inset = complexPlot(PE1, self.ax_probe_inset, **kwargs)
            self.ax_probe: plt.Axes = self.ax_probe

        else:
            self.im_probe.set_data(PE)
            # self.im_probe_inset.set_data(PE1)
            if optimizable.npsm > 1:
                self.txt_purityProbe.set_text('Probe estimate\nPurity: %i' %(100*optimizable.purityProbe)+'%')
        self.im_probe.autoscale()
        
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
            if len(error_estimate) > 1:
                self.error_metric_plot.set_data(np.arange(len(error_estimate)) + 1, error_estimate)
                self.ax_error_metric.set_xlim(1, len(error_estimate))
                self.ax_error_metric.set_ylim(np.min(error_estimate), np.max(error_estimate))
                data_aspect = (np.log(np.max(error_estimate)/np.min(error_estimate))/np.log(len(error_estimate)))
                self.ax_error_metric.set_aspect(1/data_aspect)


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

    def add_encoder_positions(self, positions):
        meanpos = positions.mean(-1)

        self.ax_object.scatter(positions[:,0], positions[:,1])


class DiffractionDataPlot(object):
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
        Iestimate = gpuUtils.asNumpyArray(Iestimate)

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
