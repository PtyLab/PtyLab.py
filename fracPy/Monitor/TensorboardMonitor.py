import time
from pathlib import Path

from fracPy.Monitor.Monitor import DummyMonitor
from fracPy.utils.gpuUtils import asNumpyArray
from fracPy.utils.visualisation import complex2rgb, complex2rgb_vectorized
from tensorflow import summary as tfs


class TensorboardMonitor(DummyMonitor):
    # objectZoom = 1
    # probeZoom = 1
    # figureUpdateFrequency = 100000
    # verboseLevel = "low"

    # maximum number of probe state mixtures that we use
    max_npsm = 10
    # maximum number of object state mixtures
    max_nosm = 10

    # the probe is shown as an inset in the object. This specifies how much to downsample it
    probe_downsampling = 3

    def __init__(self, logdir="./logs_tensorboard", name=None):
        if name is None:
            starttime = time.strftime("%H%M")
            name = f"Start {starttime}"
            print("Name", name)
        path = Path(logdir) / name
        self.writer: tfs.SummaryWriter = tfs.create_file_writer(
            logdir=str(path), name=name
        )
        self.i = 0

    def _update_object_estimate(self, object_estimate, probe_estimate_rgb):
        # first, convert it to images
        object_estimate_rgb = complex2rgb_vectorized(object_estimate)
        # add the probe to the top right
        probe_estimate_rgb_ss = probe_estimate_rgb[
            ..., :: self.probe_downsampling, :: self.probe_downsampling, :
        ]
        Ny, Nx, _ = probe_estimate_rgb_ss.shape[-3:]

        # ensure it's 4 d as that's what is needed by tensorflow
        if object_estimate_rgb.ndim == 3:
            object_estimate_rgb = object_estimate_rgb[None]

        # last channel is for color
        object_estimate_rgb[..., :Ny, :Nx, :] = probe_estimate_rgb_ss[0]

        with self.writer.as_default():
            tfs.image(
                "object estimate",
                object_estimate_rgb,
                self.i,
                max_outputs=self.max_npsm,
            )

    def _update_probe_estimate(self, probe_estimate):
        # first, convert it to images
        probe_estimate_rgb = complex2rgb_vectorized(probe_estimate)
        # ensure it's 4 d as that's what is needed by tensorflow
        if probe_estimate_rgb.ndim == 3:
            probe_estimate_rgb = probe_estimate_rgb[None]

        with self.writer.as_default():
            tfs.image(
                "probe estimate", probe_estimate_rgb, self.i, max_outputs=self.max_npsm,
            )
        return probe_estimate_rgb

    def updatePlot(
        self, object_estimate, probe_estimate,
    ):
        self.i += 1

        probe_estimate_rgb = self._update_probe_estimate(probe_estimate)
        self._update_object_estimate(object_estimate, probe_estimate_rgb)

    def updateObjectProbeErrorMonitor(
        self,
        error,
        object_estimate,
        probe_estimate,
        zo,
        purity_object=None,
        purity_probe=None,
        *args,
        **kwargs,
    ):
        # The input can be either an empty array, an array with length 1 or a list of all the errors so far.
        # In the last case, we only want the last value.
        # or just a number. This part should account of all of them.

        self._update_error_estimate(error)
        self.updatePlot(object_estimate, probe_estimate)
        self.update_z(zo)
        self.update_purities(asNumpyArray(purity_probe), asNumpyArray(purity_object))

    def updateDiffractionDataMonitor(self, Iestimated, Imeasured):
        import numpy as np

        Itotal = np.hstack([Iestimated, Imeasured])
        Itotal = Itotal[None, ..., None]
        Itotal = Itotal / Itotal.max() * 255
        Itotal = Itotal.astype(np.uint8)
        with self.writer.as_default():
            print(Itotal.shape)

            tfs.image(
                "Iest Ipred",
                Itotal,
                step=self.i,
                description="Estimated (left) and predicted (right) image data",
            )

    def update_z(self, z):
        with self.writer.as_default():
            tfs.scalar("zo (mm)", 1e3*z, step=self.i, description="Z of the reconstruction")

    def _update_error_estimate(self, error):

        if error == []:
            return  # initialization, not required for tensorboard
        import numpy as np

        try:
            error = np.array(error).ravel()[-1]
        except:
            error = float(error)
        with self.writer.as_default():

            tfs.scalar("error_metric", error, self.i, "Error metric for the last image")

    def _update_probe_purity(self, probe_purity):
        if probe_purity is None:
            return
        with self.writer.as_default():
            tfs.scalar(
                "probe purity",
                probe_purity,
                self.i,
                "Probe purity. See HTML output for the individual traces",
            )

    def _update_object_purity(self, object_purity):
        if object_purity is None:
            return
        with self.writer.as_default():
            tfs.scalar(
                "object purity",
                object_purity,
                self.i,
                "Object purity. See HTML output for the individual traces",
            )
    def update_purities(self, probe_purity, object_purity):
        self._update_object_purity(object_purity)
        self._update_probe_purity(probe_purity)