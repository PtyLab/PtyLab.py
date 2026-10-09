import math

import matplotlib as mpl
import numpy as np
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PtyLab.utils import gpuUtils
from PtyLab.utils.visualisation import complex2rgb, complexPlot, modeTile


def is_inline():
    """Whether matplotlib draws inline, as in a Jupyter notebook.

    Returns:
        bool: True for an inline backend, False for a GUI or file backend.
    """
    return "inline" in mpl.get_backend().lower()


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
        """Draw the figure, in a notebook output or in its window."""
        if is_inline():
            self.drawNowIpython()
        else:
            self.drawNowScript()


class ObjectProbeErrorPlot(_LiveFigure):
    """Figure with the object estimate, the probe estimate and the error metric.

    Call `updateObject`, `updateProbe` and `updateError` with the current state,
    then `drawNow` to show it. The first round of updates creates the images; later
    ones only replace their data.

    Args:
        figNum (int, optional): pyplot figure number. Defaults to 1.
    """

    def __init__(self, figNum=1):
        self.figNum = figNum
        self._createFigure()

    def _createFigure(self) -> None:
        """Create the figure with an object, a probe and an error-metric panel."""
        # interactive mode only matters for GUI windows; leave notebook sessions alone
        if not is_inline():
            plt.ion()
        # a new monitor must not draw on top of an old figure with the same number;
        # the size is set separately because pyplot ignores figsize for a reused one
        self.figure = plt.figure(num=self.figNum, clear=True)
        self.figure.set_size_inches(10, 3)
        axes = self.figure.subplot_mosaic("""Ape""", empty_sentinel=" ")

        self.ax_object = axes["A"]
        self.ax_probe = axes["p"]
        self.ax_error_metric = axes["e"]
        self.txt_purityObject = self.ax_object.set_title("Object estimate")
        self.txt_purityProbe = self.ax_probe.set_title("Probe estimate")

        ax = self.ax_error_metric
        ax.set_title("Error metric")
        ax.set_xlabel("iterations")
        ax.set_ylabel("error")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # plain tick labels (2, 3, 20) instead of 2x10^0, which overlap on a short
        # axis; LogFormatter still thins out the minor labels over many decades
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(ticker.LogFormatter())
            axis.set_minor_formatter(ticker.LogFormatter(labelOnlyBase=False))
        ax.grid(True, which="major", alpha=0.4)
        ax.grid(True, which="minor", alpha=0.15)
        self.figure.tight_layout()
        self._attach_figure(self.figure)

    def updateObject(
        self,
        object_estimate,
        objectPlot,
        amplitudeScalingFactor=1,
        purity=None,
        **kwargs,
    ):
        """Show the object estimate.

        Args:
            object_estimate (np.ndarray): Object, `(Ny, Nx)` or a stack of modes,
                which are tiled side by side.
            objectPlot (str): `"complex"` (phase as hue, amplitude as brightness),
                `"abs"` or `"angle"`.
            amplitudeScalingFactor (float, optional): Brightness scaling of the
                complex plot. Defaults to 1.
            purity (float, optional): Object purity, shown in the title; `None` for
                a single object mode. Defaults to None.
            **kwargs: Passed to `complexPlot` on the first call, e.g. `pixelSize`
                and `axisUnit`.
        """
        OE = modeTile(object_estimate, normalize=True)
        if objectPlot == "complex":
            OE = complex2rgb(OE, amplitudeScalingFactor=amplitudeScalingFactor)
        elif objectPlot == "abs":
            # normalise to mean + std rather than the maximum, so a few hot pixels
            # do not darken the whole image
            AOE = abs(OE)
            OE = abs(OE / (AOE.mean() + np.std(AOE)))
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
            if purity is not None:
                self.txt_purityObject.set_text(
                    f"Object estimate\nPurity: {int(100 * purity)}%"
                )

        self.im_object.autoscale()

    def updateProbe(
        self,
        probe_estimate,
        amplitudeScalingFactor=1,
        label="Probe estimate",
        purity=None,
        **kwargs,
    ):
        """Show the probe (or, for FPM, pupil) estimate as a complex plot.

        Args:
            probe_estimate (np.ndarray): Probe, `(Ny, Nx)` or a stack of modes,
                which are tiled side by side.
            amplitudeScalingFactor (float, optional): Brightness scaling of the
                complex plot. Defaults to 1.
            label (str, optional): Panel title. Defaults to `"Probe estimate"`.
            purity (float, optional): Probe purity, shown in the title; `None` for
                a single probe mode. Defaults to None.
            **kwargs: Passed to `complexPlot` on the first call, e.g. `pixelSize`
                and `axisUnit`.
        """
        PE = complex2rgb(
            modeTile(probe_estimate, normalize=True),
            amplitudeScalingFactor=amplitudeScalingFactor,
        )

        if self.firstrun:
            self.im_probe = complexPlot(PE, ax=self.ax_probe, **kwargs)
            self.txt_purityProbe = self.ax_probe.set_title(label)
        else:
            self.im_probe.set_data(PE)
            # purity is NaN until it is first computed
            if purity is not None and not math.isnan(float(purity)):
                self.txt_purityProbe.set_text(f"{label}\nPurity: {100 * purity:.2f}%")
        self.im_probe.autoscale()

    def updateError(self, error_estimate) -> None:
        """Plot the error metric of every iteration so far on log-log axes.

        Args:
            error_estimate (array-like): One error value per completed iteration.
        """
        error = np.asarray(gpuUtils.asNumpyArray(error_estimate), dtype=float)
        # iterations count from 1, as 0 cannot be shown on a log axis
        iterations = np.arange(1, len(error) + 1)

        if self.firstrun:
            (self.error_metric_plot,) = self.ax_error_metric.plot(
                iterations, error, "o-", mfc="none"
            )
        else:
            self.error_metric_plot.set_data(iterations, error)

        if len(error) > 0:
            ax = self.ax_error_metric
            # rescale to the data; the log axes ignore non-positive and NaN values
            ax.relim()
            ax.autoscale_view()
            ax.set_title(f"Error metric (it {len(error)})")


class DiffractionDataPlot(_LiveFigure):
    r"""Figure with the estimated and measured diffraction intensity, side by side.

    Both are shown as $\log_{10}(I + 1)$, with the colour limits of the measured
    intensity applied to the estimate so the two can be compared directly.

    Args:
        figNum (int, optional): pyplot figure number. Defaults to 2.
    """

    def __init__(self, figNum=2):
        self.figNum = figNum
        self._createFigure()

    def _createFigure(self) -> None:
        """Create the figure with an estimated and a measured intensity panel."""
        # interactive mode only matters for GUI windows; leave notebook sessions alone
        if not is_inline():
            plt.ion()
        # see ObjectProbeErrorPlot._createFigure for why size is set separately
        self.figure = plt.figure(num=self.figNum, clear=True)
        self.figure.set_size_inches(8, 3)
        axes = self.figure.subplots(1, 2, squeeze=False)
        self.ax_Iestimated = axes[0][0]
        self.ax_Imeasured = axes[0][1]
        self.ax_Iestimated.set_title("Estimated intensity")
        self.ax_Imeasured.set_title("Measured intensity")
        self.figure.tight_layout()
        self._attach_figure(self.figure)

    def updateIestimated(self, Iestimate, cmap="gray", **kwargs):
        """Show the estimated intensity; its colour limits are set by `update_view`.

        Args:
            Iestimate (np.ndarray): Estimated detector intensity, CPU or GPU.
            cmap (str or Colormap, optional): Colour map. Defaults to `"gray"`.
        """
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
        else:
            self.im_Iestimated.set_data(np.log10(np.squeeze(Iestimate + 1)))

    def updateImeasured(self, Imeasured, cmap="gray", **kwargs):
        """Show the measured intensity, with colour limits fitted to it.

        Args:
            Imeasured (np.ndarray): Measured detector intensity, CPU or GPU.
            cmap (str or Colormap, optional): Colour map. Defaults to `"gray"`.
        """
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
        """Show both intensities with the colour limits of the measured one.

        Args:
            Iestimated (np.ndarray): Estimated detector intensity.
            Imeasured (np.ndarray): Measured detector intensity.
            cmap (str or Colormap): Colour map of both panels.
        """
        self.updateImeasured(Imeasured, cmap=cmap)
        self.updateIestimated(Iestimated, cmap=cmap)
        self.im_Iestimated.set_clim(*self.im_Imeasured.get_clim())


class ParameterHistoryPlot(_LiveFigure):
    """Traces of the quantities an engine changes besides object and probe.

    Each update records one value per line, grouped into panels (e.g. the panel
    "purity [%]" with lines "object" and "probe"). A panel is only drawn once one of
    its lines has changed from its first value, so quantities the engine never
    touches (a fixed zo, a single-mode purity) stay hidden. The scan-position
    panel appears once position correction has moved a position.

    The figure is created on the first draw that has something to show, so no
    window opens for a reconstruction that changes nothing.
    """

    def __init__(self, figNum=3):
        self.figNum = figNum
        self.figure = None
        # panel title -> line label -> ([iterations], [values])
        self.history = {}
        # (measured, corrected) positions in m, only kept once they differ
        self.positions = None

    def record(self, iteration, values):
        """Append one value per line.

        Args:
            iteration (int): Iteration the values belong to.
            values (dict[str, dict[str, float]]): Panel title -> line label -> value.
        """
        for panel, lines in values.items():
            for label, value in lines.items():
                its, vals = self.history.setdefault(panel, {}).setdefault(
                    label, ([], [])
                )
                its.append(iteration)
                vals.append(value)

    def record_positions(self, measured, corrected):
        """Keep the scan positions if position correction has moved any of them."""
        measured = gpuUtils.asNumpyArray(measured)
        corrected = gpuUtils.asNumpyArray(corrected)
        if not np.array_equal(measured, corrected):
            # corrected is edited in place by the engine, so keep a copy
            self.positions = (measured, corrected.copy())

    def active_panels(self):
        """Titles of the panels whose values have changed, in recording order."""
        return [
            panel
            for panel, lines in self.history.items()
            if any(
                not np.allclose(vals, vals[0], rtol=1e-6, atol=0)
                for _, vals in lines.values()
            )
        ]

    def draw(self):
        """Redraw every active panel; does nothing while no panel is active."""
        panels = self.active_panels()
        n = len(panels) + (self.positions is not None)
        if n == 0:
            return

        if self.figure is None:
            # interactive mode only matters for GUI windows
            if not is_inline():
                plt.ion()
            self._attach_figure(plt.figure(num=self.figNum, clear=True))
        else:
            # the number of panels can grow, so rebuild the axes on every draw;
            # a handful of line plots is cheap next to the object/probe images
            self.figure.clf()
        self.figure.set_size_inches(3.2 * n, 3, forward=True)

        axes = np.atleast_1d(self.figure.subplots(1, n))
        for ax, panel in zip(axes, panels):
            for label, (its, vals) in self.history[panel].items():
                ax.plot(its, vals, ".-", label=label)
            ax.set_title(panel)
            ax.set_xlabel("iterations")
            # show absolute values (49.99 mm), not an offset plus tiny ticks
            ax.ticklabel_format(axis="y", useOffset=False)
            ax.grid(True, alpha=0.3)
            if len(self.history[panel]) > 1:
                ax.legend(fontsize="small")

        if self.positions is not None:
            measured, corrected = self.positions
            ax = axes[-1]
            # encoder positions are (y, x) in m; plot x horizontally, in mm
            ax.plot(
                measured[:, 1] * 1e3,
                measured[:, 0] * 1e3,
                "o",
                mfc="none",
                label="measured",
            )
            ax.plot(
                corrected[:, 1] * 1e3, corrected[:, 0] * 1e3, ".", label="corrected"
            )
            ax.set_title("scan positions [mm]")
            ax.set_aspect("equal", adjustable="datalim")
            ax.legend(fontsize="small")

        self.figure.tight_layout()
        self.drawNow()
