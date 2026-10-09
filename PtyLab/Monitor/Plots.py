import warnings

import matplotlib as mpl
import numpy as np
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PtyLab.utils import gpuUtils
from PtyLab.utils.visualisation import complex2rgb, complexPlot, modeTile


def is_inline():
    """Default IPython (jupyter notebook) backend"""
    return True if "inline" in mpl.get_backend().lower() else False


class _LiveFigure:
    """Drawing logic shared by the monitor figures.

    In a script the figure lives in its own GUI window that is redrawn in place.
    In Jupyter (inline backend) the figure is shown once through a display handle,
    and every later draw replaces that output, so the rest of the cell output
    (prints, progress bar, other figures) stays untouched.
    """

    def _attach_figure(self, figure) -> None:
        self.figure = figure
        self.canvas = figure.canvas
        self.display_handle = None
        self.firstrun = True
        if is_inline():
            # the display handle shows this figure; detach it from pyplot so the
            # inline backend does not draw it a second time at the end of the cell
            plt.close(figure)

    def drawNowScript(self):
        """Redraw the figure window, reopening it if it was closed."""
        if self.firstrun or not plt.fignum_exists(self.figNum):
            self.figure.show()
        self.firstrun = False
        self.canvas.draw_idle()
        self.canvas.flush_events()

    def drawNowIpython(self):
        """Show the figure on the first call, then update that same output."""
        if self.display_handle is None:
            self.display_handle = display(self.figure, display_id=True)
        else:
            self.display_handle.update(self.figure)
        self.firstrun = False

    def drawNow(self):
        if is_inline():
            self.drawNowIpython()
        else:
            self.drawNowScript()


class ObjectProbeErrorPlot(_LiveFigure):
    def __init__(self, figNum=1):
        """Create a monitor.

        In principle, to use this method all you have to do is initialize the monitor and then call

        updateObject, updateProbe, updateErrorMetric and drawnow to ensure that something is drawn immediately.

        For example usage, see test_matplot_monitor.py.

        """
        self.figNum = figNum
        self._createFigure()

    def update_z(self, *args, **kwargs):
        """Update the sample-detector distance. Does nothing at the moment."""
        pass

    def _createFigure(self) -> None:
        """
        Create the figure.
        :return:
        """

        # interactive mode only matters for GUI windows; leave notebook sessions alone
        if not is_inline():
            plt.ion()
        self.figure, axes = plt.subplot_mosaic(
            """Ape""",
            num=self.figNum,
            clear=True,  # a new monitor must not draw on top of an old figure
            figsize=(10, 3),
            empty_sentinel=" ",
            constrained_layout=False,
        )

        self.ax_object = axes["A"]
        self.ax_probe = axes["p"]
        # self.ax_probe_ff = axes["P"]
        self.ax_error_metric = axes["e"]
        # self.ax_probe_ff.set_title("FF probe")
        self.ax_probe.set_title("Probe")
        # self.ax_object = axes[0][0]
        # self.ax_probe = axes[0][1]
        # self.ax_error_metric = axes[0][2]
        # self.ax_object.set_title
        self.txt_purityProbe = self.ax_probe.set_title("Probe estimate")
        self.txt_purityObject = self.ax_object.set_title("Object estimate")
        self.ax_error_metric.set_title("Error metric")
        self.ax_error_metric.grid(True)
        self.ax_error_metric.grid(
            animated=True, which="minor", color="#999999", linestyle="-", alpha=0.2
        )
        self.ax_error_metric.set_xlabel("iterations")
        self.ax_error_metric.set_ylabel("error")
        self.ax_error_metric.set_xscale("log")
        self.ax_error_metric.set_yscale("log")
        self.ax_error_metric.axis("image")
        self.figure.tight_layout()
        self._attach_figure(self.figure)

    def updateObject(
        self,
        object_estimate,
        optimizable,
        objectPlot,
        amplitudeScalingFactor=1,
        **kwargs,
    ):
        OE = modeTile(object_estimate, normalize=True)
        if objectPlot == "complex":
            OE = complex2rgb(OE, amplitudeScalingFactor=amplitudeScalingFactor)

        elif objectPlot == "abs":
            # original
            # OE = OE / abs(OE).max()
            # better
            AOE = abs(OE)
            OE = OE / (AOE.mean() + np.std(AOE))
            # OE = OE / abs(OE.max())
            OE = abs(OE)
        elif objectPlot == "angle":
            OE = np.angle(OE)

        if self.firstrun:
            if objectPlot == "complex":
                self.im_object = complexPlot(OE, ax=self.ax_object, **kwargs)
            else:
                self.im_object = self.ax_object.imshow(OE, interpolation=None)
                divider = make_axes_locatable(self.ax_object)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                self.objectCbar = self.figure.colorbar(
                    self.im_object, ax=self.ax_object, cax=cax
                )
        else:
            self.im_object.set_data(OE)
            if optimizable.nosm > 1:
                self.txt_purityObject.set_text(
                    "Object estimate\nPurity: %i" % (100 * optimizable.purityObject)
                    + "%"
                )

        self.im_object.autoscale()

    def updateProbe(
        self,
        probe_estimate,
        optimizable,
        amplitudeScalingFactor=1,
        label="Probe estimate",
        **kwargs,
    ):

        # from PtyLab.Operators.Operators import fft2c
        #
        # probe_estimate_ff = fft2c(probe_estimate)
        # PE_ff = complex2rgb(modeTile(probe_estimate_ff, normalize=True))

        PE = complex2rgb(
            modeTile(probe_estimate, normalize=True),
            amplitudeScalingFactor=amplitudeScalingFactor,
        )

        if self.firstrun:
            self.im_probe = complexPlot(PE, ax=self.ax_probe, **kwargs)
            self.txt_purityProbe = self.ax_probe.set_title(label)
            # self.im_probe_ff = complexPlot(PE_ff, self.ax_probe_ff, **kwargs)
        else:
            self.im_probe.set_data(PE)
            # self.im_probe_ff.set_data(PE_ff)
            if (
                optimizable.npsm > 1
                and optimizable.purityProbe == optimizable.purityProbe
            ):
                self.txt_purityProbe.set_text(
                    "%s\nPurity: %.2f" % (label, 100 * optimizable.purityProbe) + "%"
                )
        self.im_probe.autoscale()

    def updateError(self, error_estimate: np.ndarray) -> None:
        """
        Update the error estimate plot.
        :param error_estimate:
        :return:
        """

        if self.firstrun:
            self.error_metric_plot = self.ax_error_metric.plot(
                error_estimate, "o-", mfc="none"
            )[0]
        else:
            if len(error_estimate) > 1 and error_estimate[-1] == error_estimate[-1]:
                self.error_metric_plot.set_data(
                    np.arange(len(error_estimate)) + 1, error_estimate
                )
                self.ax_error_metric.set_xlim(1, len(error_estimate))
                self.ax_error_metric.set_ylim(
                    np.min(error_estimate), np.max(error_estimate)
                )
                data_aspect = np.log(
                    np.max(error_estimate) / np.min(error_estimate)
                ) / np.log(len(error_estimate))
                self.ax_error_metric.set_aspect(1 / data_aspect)
                self.ax_error_metric.set_title(
                    f"Error metric (it {len(error_estimate)})"
                )

class DiffractionDataPlot(_LiveFigure):
    def __init__(self, figNum=2):
        """Create a monitor.

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
        # interactive mode only matters for GUI windows; leave notebook sessions alone
        if not is_inline():
            plt.ion()
        self.figure, axes = plt.subplots(
            1, 2, num=self.figNum, squeeze=False, clear=True, figsize=(8, 3)
        )
        self.ax_Iestimated = axes[0][0]
        self.ax_Imeasured = axes[0][1]
        self.ax_Iestimated.set_title("Estimated intensity")
        self.ax_Imeasured.set_title("Measured intensity")
        self.figure.tight_layout()
        self._attach_figure(self.figure)

    def updateIestimated(self, Iestimate, cmap="gray", **kwargs):
        # move it to CPU if it's on the GPU
        Iestimate = gpuUtils.asNumpyArray(Iestimate)

        if self.firstrun:

            self.im_Iestimated: AxesImage = self.ax_Iestimated.imshow(
                np.log10(np.squeeze(Iestimate + 1)), cmap=cmap, interpolation=None
            )

            divider = make_axes_locatable(self.ax_Iestimated)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            self.IestimatedCbar = self.figure.colorbar(
                self.im_Iestimated, ax=self.ax_Iestimated, cax=cax
            )
            # scale it according to I measured

        else:
            self.im_Iestimated.set_data(np.log10(np.squeeze(Iestimate + 1)))
        # self.im_Iestimated.autoscale()
        # self.im_Iestimated.set_

    def updateImeasured(self, Imeasured, cmap="gray", **kwargs):
        Imeasured = gpuUtils.asNumpyArray(Imeasured)
        if self.firstrun:
            self.im_Imeasured: AxesImage = self.ax_Imeasured.imshow(
                np.log10(np.squeeze(Imeasured + 1)), cmap=cmap, interpolation=None
            )

            divider = make_axes_locatable(self.ax_Imeasured)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            self.ImeasuredCbar = self.figure.colorbar(
                self.im_Imeasured, ax=self.ax_Imeasured, cax=cax
            )

        else:
            self.im_Imeasured.set_data(np.log10(np.squeeze(Imeasured + 1)))
        self.im_Imeasured.autoscale()

    def update_view(self, Iestimated, Imeasured, cmap):
        """Update the I measured and I estimated and make sure that the colormaps have the same limits"""
        self.updateImeasured(Imeasured, cmap=cmap)
        self.updateIestimated(Iestimated, cmap=cmap)
        self._equalize_contrast()

    def _equalize_contrast(self):
        """Adopt the contrast limits from the measured data and apply them to the predicted"""
        clims = self.im_Imeasured.get_clim()
        self.im_Iestimated.set_clim(*clims)
