import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from PtyLab.utils.visualisation import complex2rgb, setColorMap

from .Plots import DiffractionDataPlot, ObjectProbeErrorPlot, is_inline


class AbstractMonitor(object):
    """
    This monitor implements all the basic features that you have to override for a custom monitor.

    Alternatively, you can instantiate this class to create a monitor that does not do anything and therefore,
    will not take time to run.

    """

    def initializeMonitors(self):
        """
        This code is run after __init__, use it to build a.k.a. GUI elements.

        :return: Nothing
        """

        pass

    def update_focusing_metric(self, TV_value, AOI_image, metric_name, allmerits=None):
        """
        Show the total variation of the object estimate inside the area of interest.
        :param TV_value: Value of TV
        :return:
        """
        pass

    def writeEngineName(self, name):
        """
        Save the engine name for this particular iteration.

        :param name:  Name of the engine that is used for this iteration.

        :return:
        """
        pass

    def visualize_probe_engine(self, estimate):
        """
        For nonlinear imaging only. Return the electric field of the probe.
        """
        pass

    def updatePlot(
        self,
        object_estimate: np.ndarray,
        probe_estimate: np.ndarray,
        zo=None,
        encoder_positions=None,
    ):
        """
        Update the visualisation of both probe and and object estimate.

        Please note, that only the part that is defined by probeZoom and objectZoom is uploaded.

        #TODO: Change that, upload the entire image and make the monitor decide which parts to show.

        :param object_estimate: Array with the object estimate.
        :param probe_estimate:  Array with the probe estimate.
        :return:
        """
        pass

    def getOverlap(self, *args, **kwargs):
        """
        Get the overlap between subsequent probes.

        Todo(dbs660): Move this.

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: remove this from here, it should go in some utils
        pass

    def update_positions(self, *args, **kwargs):
        """Update the position information."""
        pass

    def update_encoder(self, corrected_positions, original_positions, *args, **kwargs):
        """Update the image of the encoder positions."""
        pass

    def update_overlap(self, overlap_area, linear_overlap):
        pass

    def updateObjectProbeErrorMonitor(
        self,
        error: float,
        object_estimate: np.ndarray,
        probe_estimate: np.ndarray,
        zo=None,
        purity_probe=None,
        purity_object=None,
        encoder_positions=None,
    ):
        """
        Update the Object and Probe error monitor, and any associated metrics.

        :param error: Value of the loss for the current iteration.
        :param object_estimate: Current object estimate
        :param probe_estimate: Current probe estimate
        :param zo: Sample-detector distance
        :param purity_probe: purity of the probe
        :param purity_object: purity of the object
        :return:
        """

        pass

    def updateBeamWidth(self, beamwidth_x, beamwidth_y):
        """
        Update the beam width in X and Y

        Parameters
        ----------
        beamwidth_x: beamwidth x in meter
        beamwidth_y: beamwidth y in meter

        Returns
        -------

        """
        pass

    def updateDiffractionDataMonitor(self, Iestimated, Imeasured):
        """
        Update the diffraction data estimate at the current iteration.

        :param Iestimated: Estimated intensity at the current position
        :param Imeasured:  Measured intensity at the current position
        :return:
        """

        pass


class Monitor(AbstractMonitor):
    """
    Monitor contains two submonitors: ObjectProbeErrorPlot (object,probe,error) and DiffractionDataPlot (diffraction
    intensity estimate and measurement)
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
        self.reconstruction = None
        self.cmapDiffraction = setColorMap()
        self.defaultMonitor = None
        self.screenshot_directory = None
        self.diffractionDataMonitor = None

    @property
    def figureUpdateFrequency(self):
        return self._figureUpdateFrequency

    @figureUpdateFrequency.setter
    def figureUpdateFrequency(self, value):
        self._figureUpdateFrequency = value
        if is_inline() and self.figureUpdateFrequency < 5:
            warnings.simplefilter("always", UserWarning)
            warnings.warn(
                "For faster reconstruction with inline backend, set `monitor.figureUpdateFrequency = 5` or higher."
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
                'set `monitor.verboseLevel = "low"`. '
            )

    def initializeMonitors(self):
        """
        Create the figure and axes etc.
        :return:
        """
        # only initialize if it hasn't been done
        if self.defaultMonitor is None:
            self.defaultMonitor = ObjectProbeErrorPlot()

        if self.verboseLevel == "high":
            self.diffractionDataMonitor = DiffractionDataPlot()

    def updateObjectProbeErrorMonitor(
        self,
        error,
        object_estimate,
        probe_estimate,
        zo=None,
        purity_probe=None,
        purity_object=None,
        encoder_positions=None,
    ):
        """
        update the probe object plots
        :param object_estimate:
        :return:
        """
        self.defaultMonitor.updateError(error)  # self.reconstruction.error)
        # print(f"Object plot: {self.objectPlot}")
        self.defaultMonitor.updateObject(
            object_estimate,
            self.reconstruction,
            objectPlot=self.objectPlot,
            pixelSize=self.reconstruction.dxo,
            axisUnit="mm",
            amplitudeScalingFactor=self.objectPlotContrast,
        )
        self.defaultMonitor.updateProbe(
            probe_estimate,
            self.reconstruction,
            pixelSize=self.reconstruction.dxp,
            axisUnit="mm",
            amplitudeScalingFactor=self.probePlotContrast,
        )
        self.defaultMonitor.update_z(zo)
        self.defaultMonitor.drawNow()

        if self.screenshot_directory is not None:
            self.defaultMonitor.figure.savefig(
                Path(self.screenshot_directory) / f"{len(error)}.png"
            )

    def describe_parameters(self, *args, **kwargs):
        pass

    def updateDiffractionDataMonitor(self, Iestimated, Imeasured):
        """
        update the diffraction plots
        """

        self.diffractionDataMonitor.update_view(
            Iestimated, Imeasured, cmap=self.cmapDiffraction
        )
        # self.diffractionDataMonitor.updateIestimated(Iestimated, cmap=self.cmapDiffraction)
        # self.diffractionDataMonitor.updateImeasured(Imeasured, cmap=self.cmapDiffraction)
        self.diffractionDataMonitor.drawNow()


class DummyMonitor(object):
    """Monitor without any visualisation so it won't consume any time"""

    objectZoom = 1
    probeZoom = 1
    # remains from mPIE
    figureUpdateFrequency = 1000000
    verboseLevel = "low"

    def updatePlot(self, object_estimate, probe_estimate):
        pass

    def getOverlap(self, ind1, ind2, probePixelsize):
        pass

    def initializeVisualisation(self):
        pass

    def initializeMonitors(self):
        pass

    def updateObjectProbeErrorMonitor(self, *args, **kwargs):
        pass

    def updateDiffractionDataMonitor(self, *args, **kwargs):
        pass

    def writeEngineName(self, *args, **kwargs):
        pass


class NapariMonitor(DummyMonitor):
    try:
        import napari
    except ImportError:
        print("Install napari for this implementation")

    def initializeVisualisation(self):
        self.viewer = napari.Viewer()
        self.viewer.show()

        self.viewer.add_image(name="object estimate", data=np.random.rand(100, 100))
        self.viewer.add_image(name="probe estimate", data=np.random.rand(100, 100))

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
        self.viewer.layers["probe_estimate"].data = RGB_probe

    def update_object_image(self, object_estimate):
        RGB_object = complex2rgb(object_estimate)
        self.viewer.layers["object estimate"].data = RGB_object

    def updatePlot(self, object_estimate, probe_estimate):
        self.update_probe_image(probe_estimate)
        self.update_object_image(object_estimate)

    def updateDiffractionDataMonitor(self, Iestimated, Imeasured):
        self.Iestimated.data = Iestimated
        self.Imeasured.data = Imeasured

    def update_positions(self, *args, **kwargs):
        """
        Update the positions. USeful for position correction.
        :param args:
        :param kwargs:
        :return:
        """
        pass
