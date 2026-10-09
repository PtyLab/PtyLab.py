import warnings
from pathlib import Path

import numpy as np

from PtyLab.utils.visualisation import complex2rgb, setColorMap

from .frame import MonitorFrame
from .Plots import (
    DiffractionDataPlot,
    ObjectProbeErrorPlot,
    ParameterHistoryPlot,
    is_inline,
)


def tracked_values(frame: MonitorFrame) -> dict:
    """Scalars besides object and probe that an engine may change, in display units.

    Quantities the frame does not carry (`None`) or that are undefined for this
    reconstruction (purity of a single mode) are left out.

    Args:
        frame (MonitorFrame): Current reconstruction state.

    Returns:
        dict[str, dict[str, float]]: Panel title -> line label -> value.
    """
    values = {}
    if frame.zo is not None:
        values["zo [mm]"] = {"zo": 1e3 * float(frame.zo)}
    if frame.theta is not None:
        values["theta [deg]"] = {"theta": float(frame.theta)}
    purity = {}
    if frame.nosm > 1:
        purity["object"] = 100 * float(frame.purity_object)
    if frame.npsm > 1:
        purity["probe"] = 100 * float(frame.purity_probe)
    if purity:
        values["purity [%]"] = purity
    return values


class AbstractMonitor:
    """Base class of the reconstruction monitors; on its own it shows nothing.

    The engines call `update` with one `MonitorFrame` per update. A custom monitor
    can either override `update` and read the frame directly, or override the
    per-quantity hooks below (`updateObjectProbeErrorMonitor`, `update_encoder`,
    ...), which the default `update` calls. Every hook is a no-op here, so a
    subclass only implements the ones it uses.
    """

    def update(self, frame: MonitorFrame):
        """Show one snapshot of the reconstruction.

        The default forwards the frame to the per-quantity hooks, so monitors
        written against those hooks keep working. Fields the engine left empty
        (`None`) are skipped.

        Args:
            frame (MonitorFrame): Current reconstruction state.
        """
        self.updateObjectProbeErrorMonitor(
            error=frame.error,
            object_estimate=frame.object_view(),
            probe_estimate=frame.probe_view(),
            zo=frame.zo,
            purity_probe=frame.purity_probe,
            purity_object=frame.purity_object,
            encoder_positions=frame.positions,
        )
        if frame.engine_name is not None:
            self.writeEngineName(frame.engine_name)
        if frame.encoder_original is not None:
            self.update_encoder(
                corrected_positions=frame.encoder_corrected,
                original_positions=frame.encoder_original,
            )
        if frame.beam_width is not None:
            self.updateBeamWidth(*frame.beam_width)
        if frame.I_estimated is not None:
            self.updateDiffractionDataMonitor(
                Iestimated=frame.I_estimated, Imeasured=frame.I_measured
            )
        if frame.area_overlap is not None:
            self.update_overlap(frame.area_overlap, frame.linear_overlap)

    def initializeMonitors(self):
        """Create the figures or other outputs; the engine calls it before the first update."""

    def saveFigures(self, fileName="recent", dpi=150):
        """Save the monitor figures as PNG files. This monitor has none.

        Returns:
            list[Path]: Empty, as nothing is saved.
        """
        return []

    def describe_parameters(self, params):
        """Record the reconstruction parameters, e.g. as text in a log.

        Args:
            params (Params): Parameters of the reconstruction.
        """

    def update_focusing_metric(self, TV_value, AOI_image, metric_name, allmerits=None):
        """Show the result of one TV autofocus step.

        Args:
            TV_value (float): Focus metric at the current distance.
            AOI_image (np.ndarray): Propagated field in the area of interest.
            metric_name (str): Name of the focus metric, e.g. `"TV"`.
            allmerits (np.ndarray, optional): Focus metric at every distance that
                was tried. Defaults to None.
        """

    def writeEngineName(self, name):
        """Record which engine produced the current update.

        Args:
            name (str): Engine name.
        """

    def update_encoder(self, corrected_positions, original_positions, *args, **kwargs):
        """Show the scan positions before and after position correction.

        Args:
            corrected_positions (np.ndarray): Corrected encoder positions in m.
            original_positions (np.ndarray): Measured encoder positions in m.
        """

    def update_overlap(self, overlap_area, linear_overlap):
        """Show the estimated probe overlap between neighbouring scan positions.

        Args:
            overlap_area (float): Area overlap, between 0 and 1.
            linear_overlap (float): Linear overlap, between 0 and 1.
        """

    def updateObjectProbeErrorMonitor(
        self,
        error,
        object_estimate: np.ndarray,
        probe_estimate: np.ndarray,
        zo=None,
        purity_probe=None,
        purity_object=None,
        encoder_positions=None,
    ):
        """Show the object and probe estimates and the error metric.

        Args:
            error (array-like): Error metric of every iteration so far.
            object_estimate (np.ndarray): Object inside the display region.
            probe_estimate (np.ndarray): Probe inside the display region.
            zo (float, optional): Sample-detector distance in m. Defaults to None.
            purity_probe (float, optional): Probe purity. Defaults to None.
            purity_object (float, optional): Object purity. Defaults to None.
            encoder_positions (np.ndarray, optional): Scan positions in object
                pixels. Defaults to None.
        """

    def updateBeamWidth(self, beamwidth_y, beamwidth_x):
        """Show the probe width.

        Args:
            beamwidth_y (float): FWHM along y in m.
            beamwidth_x (float): FWHM along x in m.
        """

    def updateDiffractionDataMonitor(self, Iestimated, Imeasured):
        """Show the estimated and measured intensity at the last scan position.

        Args:
            Iestimated (np.ndarray): Estimated detector intensity.
            Imeasured (np.ndarray): Measured detector intensity.
        """


class Monitor(AbstractMonitor):
    """Live matplotlib monitor, in GUI windows or in Jupyter cell outputs.

    Shows up to three figures:

    - object, probe and error metric (`ObjectProbeErrorPlot`), always
    - estimated vs measured intensity (`DiffractionDataPlot`), for
      `verboseLevel = "high"`
    - zo, theta, purity and scan positions (`ParameterHistoryPlot`), once the
      engine changes one of them and while `showParameterHistory` is True

    Attributes:
        figureUpdateFrequency (int): Update the figures every this many iterations.
        verboseLevel (str): `"low"`, or `"high"` to add the diffraction figure.
        objectPlot (str): `"complex"`, `"abs"` or `"angle"`.
        objectZoom (float or None): Shrinks the shown object region around the
            scanned area; `None` or `"full"` shows the whole object.
        probeZoom (float or None): Same for the probe, around the pupil size.
        objectPlotContrast (float): Brightness scaling of the complex object plot.
        probePlotContrast (float): Brightness scaling of the complex probe plot.
        showParameterHistory (bool): Whether to show the third figure.
        screenshot_directory (str or None): If set, the first figure is saved
            there after every update as `frame_<iteration>.png`.
    """

    def __init__(self):
        # settings for visualization
        self._figureUpdateFrequency = 1
        self.objectPlot = "complex"
        self._verboseLevel = "low"
        self.objectZoom = 1
        self.probeZoom = 1
        self.objectPlotContrast = 1
        self.probePlotContrast = 1
        self.cmapDiffraction = setColorMap()
        self.defaultMonitor = None
        self.screenshot_directory = None
        self.diffractionDataMonitor = None
        # zo, theta, purity and scan positions in a third window, shown only
        # for the quantities the engine actually changes
        self.showParameterHistory = True
        self.parameterHistoryMonitor = None

    @property
    def figureUpdateFrequency(self):
        return self._figureUpdateFrequency

    @figureUpdateFrequency.setter
    def figureUpdateFrequency(self, value):
        self._figureUpdateFrequency = value
        if is_inline() and self.figureUpdateFrequency < 5:
            warnings.simplefilter("always", UserWarning)
            warnings.warn(
                "For faster update of the reconstruction plot, set `monitor.figureUpdateFrequency = 5` or higher."
            )

    @property
    def verboseLevel(self):
        return self._verboseLevel

    @verboseLevel.setter
    def verboseLevel(self, value):
        self._verboseLevel = value
        if is_inline() and self._verboseLevel == "high":
            warnings.simplefilter("always", UserWarning)
            warnings.warn(
                "For diffraction data plot, preferably use an interactive matplotlib backend or"
                ' set `monitor.verboseLevel = "low"`. '
            )

    def initializeMonitors(self):
        """Create the figures; a figure that already exists is reused."""
        if self.defaultMonitor is None:
            self.defaultMonitor = ObjectProbeErrorPlot()

        if self.verboseLevel == "high":
            self.diffractionDataMonitor = DiffractionDataPlot()

        if self.showParameterHistory and self.parameterHistoryMonitor is None:
            # creates no figure yet; that happens once a quantity changes
            self.parameterHistoryMonitor = ParameterHistoryPlot()

    def update(self, frame: MonitorFrame):
        """Draw the object, probe and error panels, plus the diffraction data when shown.

        Args:
            frame (MonitorFrame): Current reconstruction state.
        """
        plot = self.defaultMonitor
        plot.updateError(frame.error)
        plot.updateObject(
            frame.object_view(),
            objectPlot=self.objectPlot,
            pixelSize=frame.object_pixel_size,
            axisUnit="mm",
            amplitudeScalingFactor=self.objectPlotContrast,
            # purity is only defined for mixed states
            purity=frame.purity_object if frame.nosm > 1 else None,
        )
        plot.updateProbe(
            frame.probe_view(),
            pixelSize=frame.probe_pixel_size,
            axisUnit=frame.probe_axis_unit,
            label=frame.probe_label,
            amplitudeScalingFactor=self.probePlotContrast,
            purity=frame.purity_probe if frame.npsm > 1 else None,
        )
        plot.drawNow()

        if self.screenshot_directory is not None:
            plot.figure.savefig(
                Path(self.screenshot_directory) / f"frame_{frame.iteration}.png"
            )

        # the diffraction figure only exists for verboseLevel "high", and the
        # engine only fills the intensities then
        if self.diffractionDataMonitor is not None and frame.I_estimated is not None:
            self.diffractionDataMonitor.update_view(
                frame.I_estimated, frame.I_measured, cmap=self.cmapDiffraction
            )
            self.diffractionDataMonitor.drawNow()

        if self.parameterHistoryMonitor is not None:
            history = self.parameterHistoryMonitor
            history.record(frame.iteration, tracked_values(frame))
            if frame.encoder_corrected is not None:
                history.record_positions(
                    frame.encoder_original, frame.encoder_corrected
                )
            history.draw()

    def saveFigures(self, fileName="recent", dpi=150):
        """Save the current monitor figures as PNG files.

        Call it after `engine.reconstruct()` to keep the final state, or let
        `reconstruction.saveResults(fileName, snapshot=monitor)` call it next to
        the HDF5 file. Each figure that exists is written as
        `<fileName>_<figure>.png`:

        - `reconstruction`: object, probe and error metric
        - `diffraction`: estimated vs measured intensity (`verboseLevel = "high"`)
        - `history`: zo, theta, purity and scan positions, if any of them changed

        Args:
            fileName (str or Path, optional): Path prefix of the PNG files; missing
                parent directories are created. Defaults to `"recent"`.
            dpi (int, optional): Resolution of the PNG files. Defaults to 150.

        Returns:
            list[Path]: The files written.
        """
        figures = {
            "reconstruction": self.defaultMonitor,
            "diffraction": self.diffractionDataMonitor,
            "history": self.parameterHistoryMonitor,
        }
        prefix = Path(fileName)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        saved = []
        for name, plot in figures.items():
            # a plot object can exist without a figure, e.g. the history window
            # before any quantity changed
            if plot is None or getattr(plot, "figure", None) is None:
                continue
            path = prefix.with_name(f"{prefix.name}_{name}.png")
            plot.figure.savefig(path, dpi=dpi, bbox_inches="tight")
            saved.append(path)
        return saved


class DummyMonitor(AbstractMonitor):
    """Monitor that shows nothing, for runs where plotting would only cost time.

    Inherits every no-op hook from `AbstractMonitor`, so any hook an engine calls
    (e.g. `update_focusing_metric` when `params.TV_autofocus` is on) exists here too.
    """

    objectZoom = 1
    probeZoom = 1
    # the engines only call the monitor every `figureUpdateFrequency` iterations
    figureUpdateFrequency = 1000000
    verboseLevel = "low"

    def update(self, frame):
        # the default would crop and copy object and probe to the CPU for nothing
        pass


class NapariMonitor(DummyMonitor):
    """Shows the object, probe and diffraction data as layers in a napari viewer.

    Needs napari, which is not a PtyLab dependency.
    """

    def initializeVisualisation(self):
        """Open the viewer and create its image layers."""
        # currently a hacky way for this class, these napari implementations must
        # later be moved an optional sub-package.
        try:
            import napari

            self.viewer = napari.Viewer()
            self.viewer.show()
        except ImportError:
            msg = "Install napari to access this `NapariMonitor` implementation"
            raise ImportError(msg)

        # object and probe are shown as complex2rgb images, so these layers must be
        # RGB from the start: napari fixes a layer's `rgb` flag when it is created
        self.viewer.add_image(
            name="object estimate", data=np.zeros((100, 100, 3)), rgb=True
        )
        self.viewer.add_image(
            name="probe estimate", data=np.zeros((100, 100, 3)), rgb=True
        )

        self.Iestimated = self.viewer.add_image(
            name="I estimated", data=np.random.rand(100, 100)
        )
        self.Imeasured = self.viewer.add_image(
            name="I measured", data=np.random.rand(100, 100)
        )
        self.rawdatamonitor = self.viewer.add_image(
            name="raw data (unset)", data=np.random.rand(100, 100), visible=False
        )

    def initializeMonitors(self):
        self.initializeVisualisation()

    def add_ptychogram(self, experimentalData):
        self.rawdatamonitor.name = "raw data"
        self.rawdatamonitor.data = experimentalData.ptychogram
        self.rawdatamonitor.visible = True

    def update_probe_image(self, new_probe):
        RGB_probe = complex2rgb(new_probe)
        self.viewer.layers["probe estimate"].data = RGB_probe

    def update_object_image(self, object_estimate):
        RGB_object = complex2rgb(object_estimate)
        self.viewer.layers["object estimate"].data = RGB_object

    def updatePlot(
        self, object_estimate, probe_estimate, zo=None, encoder_positions=None
    ):
        self.update_probe_image(probe_estimate)
        self.update_object_image(object_estimate)

    def updateObjectProbeErrorMonitor(
        self, error, object_estimate, probe_estimate, *args, **kwargs
    ):
        # this is the hook `BaseEngine.showReconstruction` calls; without it the
        # viewer never receives the estimates
        self.updatePlot(object_estimate, probe_estimate)

    def updateDiffractionDataMonitor(self, Iestimated, Imeasured):
        self.Iestimated.data = Iestimated
        self.Imeasured.data = Imeasured
