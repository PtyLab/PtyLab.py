import numpy as np
import time
from pathlib import Path
from matplotlib.ticker import EngFormatter
from fracPy.Monitor.Monitor import AbstractMonitor
from fracPy import Params
from fracPy.utils.gpuUtils import asNumpyArray
from fracPy.utils.utils import fft2c
from fracPy.utils.visualisation import complex2rgb, complex2rgb_vectorized
from tensorflow import summary as tfs
from scipy import ndimage


def center_angle(object_estimate):
    # first, align the angle of the object based on the zeroth order mode
    object_estimate_0 = object_estimate.copy()

    while object_estimate.ndim > 2:
        object_estimate_0 = object_estimate_0[0]

    F_obj = asNumpyArray(abs(fft2c(object_estimate_0) ** 2))

    N = object_estimate.shape[-1]
    cmass = N // 2 - np.array(ndimage.center_of_mass(F_obj))
    # if N%2 == 1:
    #     cmass += 0.5

    phase_mask = np.fft.fftshift(
        ndimage.fourier_shift(np.ones_like(object_estimate_0), cmass)
    )
    object_estimate = asNumpyArray(object_estimate) * phase_mask.conj()

    return object_estimate, cmass


class TensorboardMonitor(AbstractMonitor):

    # maximum number of probe state mixtures that we want to show
    max_npsm = 10
    # maximum number of object state mixtures we want to show
    max_nosm = 10
    # remove any phase slant from the object
    center_angle_object = True
    # Turn on to show the FFT of the object. Usually not useful.
    show_farfield_object = False

    # the probe is shown as an inset in the object. This specifies how much to downsample it
    probe_downsampling = 2

    def __init__(self, logdir="./logs_tensorboard", name=None):
        super(AbstractMonitor).__init__()
        if name is None:
            starttime = time.strftime("%H%M")
            name = f"Start {starttime}"
            print("Name", name)
        path = Path(logdir) / name
        self.writer: tfs.SummaryWriter = tfs.create_file_writer(
            logdir=str(path), name=name
        )
        self.i = 0

    # These are the codes that have to be implemented

    def updatePlot(self, object_estimate, probe_estimate, highres=True):
        self.i += 1
        if not highres:
            probe_estimate = probe_estimate[..., ::4, ::4]
            object_estimate = object_estimate[..., ::4, ::4]
        probe_estimate_rgb = self._update_probe_estimate(
            probe_estimate, highres=highres
        )
        self._update_object_estimate(
            object_estimate, probe_estimate_rgb, highres=highres
        )

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

        self.updatePlot(object_estimate, probe_estimate, highres=self.i % 5 == 0)

        self.update_z(zo)
        self.update_purities(asNumpyArray(purity_probe), asNumpyArray(purity_object))

    def updateDiffractionDataMonitor(self, Iestimated, Imeasured):


        Itotal = np.hstack([Iestimated, Imeasured])
        Itotal = Itotal[None, ..., None]
        Itotal = Itotal / Itotal.max() * 255
        Itotal = Itotal.astype(np.uint8)
        self.__safe_upload_image(
            "Estimated and measured intensity example", Itotal, self.i, 1
        )

    def writeEngineName(self, name, *args, **kwargs):
        with self.writer.as_default():
            tfs.text(
                "reconstructor type",
                name,
                step=self.i,
                description="reconstruction name",
            )

    def update_TV(self, TV_value, AOI_image):
        if TV_value is not None:
            self.__safe_upload_scalar('TV object', TV_value, self.i, 'Total Variation of the object')
        if AOI_image is not None:
            # print(AOI_image)

            self.__smart_upload_image_couldbecomplex('TV autofocus AOI', AOI_image, self.i, 1, 'AOI used by TV')



    def update_positions(
        self, positions: np.ndarray, original_positions: np.ndarray, scaling: float
    ) -> None:
        """
        Update the stage position images.
        :param positions:
        :param original_positions:
        :param scaling:
        :return:
        """

        positions = positions * 1e-3
        original_positions = original_positions * 1e-3

        positions = positions - positions.mean(axis=0, keepdims=True)
        original_positions = original_positions - original_positions.mean(
            axis=0, keepdims=True
        )
        import matplotlib
        import io
        from tensorflow import image

        matplotlib.use("Agg")  # no images output
        import matplotlib.pyplot as plt

        # make a fov that makes sense
        scale_0 = 1.5
        if scaling > scale_0:
            scale_0 = scaling

        position_range = np.min(original_positions.flatten()), np.max(
            original_positions.flatten()
        )
        diff = np.diff(position_range)
        mean = np.mean(position_range)
        position_range = mean - scale_0 * diff / 2, mean + scale_0 * diff / 2
        fig, axes = plt.subplot_mosaic(
            """
        ON
        .S""",
            constrained_layout=True,
            figsize=(10, 5),
        )

        axes["O"].set_title("Original positions")
        axes["N"].set_title("New and old")
        axes["S"].set_title("New (scaled)\n old")

        # plot the original one everywhere

        for name, ax in axes.items():
            ax: plt.Axes = ax
            ax.scatter(
                original_positions[:, 0],
                original_positions[:, 1],
                color="C0",
                marker=".",
                label="original",
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect(1)
            ax.xaxis.set_major_formatter(EngFormatter(unit="pix"))
            ax.yaxis.set_major_formatter(EngFormatter(unit="pix"))
            ax.set_xlim(ax.set_ylim(position_range))

        # plot the new one in the middle image
        axes["N"].scatter(
            positions[:, 0], positions[:, 1], color="C1", marker="x", label="new"
        )
        # scaled version on the right (should only show displacement, not magnification)
        axes["S"].scatter(
            positions[:, 0] * scaling,
            positions[:, 1] * scaling,
            color="C1",
            marker="x",
            label="new",
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        plt.close(fig)
        buf.seek(0)
        with self.writer.as_default():
            img = image.decode_png(buf.getvalue(), channels=4)
            img = np.expand_dims(img, 0)
            tfs.image("position correction", img, self.i)


    # internal use for tensorboardMonitor
    def _update_object_estimate(
        self, object_estimate, probe_estimate_rgb, highres=True
    ):
        """
        Update the object estimate. This ensures that within the web interface the object and the probe estimate are available.


        :param object_estimate:
        :param probe_estimate_rgb:
        :param highres:
        :return:
        """
        if self.center_angle_object:
            object_estimate, shift1 = center_angle(object_estimate)
            object_estimate, shift2 = center_angle(object_estimate)
            print("Angle shifts: ", shift1, shift2)

        # convert the object estimate to colour
        object_estimate_rgb = complex2rgb_vectorized(object_estimate)

        # ensure it's 4 d as that's what is needed by tensorflow
        if object_estimate_rgb.ndim == 3:
            object_estimate_rgb = object_estimate_rgb[None]

        # patch in an image of the first probe in the top right
        object_estimate_rgb = self.__pad_probe_in_object_estimate(
            probe_estimate_rgb, object_estimate_rgb
        )

        # Add an object, but if it's the low-resolution version add it to the (low_res) version.
        tag = "object estimate"
        if not highres:
            tag += "(low res)"
        if self.show_farfield_object:
            I_ff = asNumpyArray(
                abs(
                    fft2c(
                        object_estimate
                        - object_estimate.mean(axis=(-2, -1), keepdims=True)
                    )
                )
                ** 2
            ) ** (0.2)
            I_ff = I_ff / I_ff.max() * 255
            I_ff = np.clip(I_ff, 0, 255).astype(np.uint8)
            self.__safe_upload_image("I ff estimate", I_ff, self.i, self.max_nosm)

        self.__safe_upload_image(tag, object_estimate_rgb, self.i, self.max_npsm)

        if highres:
            I_object = abs(object_estimate ** 2)

            std_obj = I_object.std()
            mean_obj = I_object.mean()
            min_int = 0  # mean_obj - 2 * std_obj
            # max_int = np.min((mean_obj + 2 * std_obj, I_object.max()))
            N = I_object.shape[-1]
            roi = slice(N // 2 - N // 5, N // 2 + N // 5)
            # max_int = I_object[...,roi,roi].max()
            from scipy import ndimage

            max_int = ndimage.gaussian_filter(I_object, 3)[..., roi, roi].max()

            if min_int == max_int:
                max_int += 1

            I_object = (I_object - min_int) / (
                max_int - min_int
            )  # +1 to ensure that this always works
            logI = np.log(255 * I_object.astype(float) + 1)
            logI -= logI.min()
            logI /= logI.max() / 255
            I_object = np.clip(I_object * 255, 0, 255).astype(np.uint8)
            self.__safe_upload_image(
                "I object estimate", I_object, self.i, self.max_nosm
            )

            self.__safe_upload_image(
                "I object log estimate", logI.astype(np.uint8), self.i, self.max_nosm
            )

    def _update_probe_estimate(self, probe_estimate, highres=True):
        # first, convert it to images
        # while probe_estimate.ndim <= 3:
        #     probe_estimate = probe_estimate[None]
        probe_estimate_rgb = complex2rgb_vectorized(probe_estimate)
        # ensure it's 4 d as that's what is needed by tensorflow
        tag = "probe estimate"
        if not highres:
            tag += "(low res)"
        self.__safe_upload_image(tag, probe_estimate_rgb, self.i, self.max_npsm)

        if highres:
            ff_probe = fft2c(probe_estimate)
            ff_probe = complex2rgb_vectorized(ff_probe)
            self.__safe_upload_image("FF " + tag, ff_probe, self.i, self.max_npsm)
        return probe_estimate_rgb

    def __smart_upload_image_couldbecomplex(self, name, data, step, max_outputs=3, description=None):
        """
        Safely upload an image that could be complex. If it is, cast it to colour before uploading.

        """
        data = asNumpyArray(data)
        if np.iscomplexobj(data):
            print('Got complex datatype')
            data = complex2rgb_vectorized(data)
        else:
            print('Got real datatype')
            # auto scale
            data = data / data.max() * 255
            data = data.astype(np.uint8)
            # convert to black-white


        self.__safe_upload_image(name, data, step, max_outputs, description)


    def __safe_upload_image(self, name, data, step, max_outputs=3, description=None):
        data = asNumpyArray(data)
        if data.shape[-1] not in [1, 3]:
            data = data[..., None]
        while data.ndim < 4:
            data = data[None]
        with self.writer.as_default():
            tfs.image(
                name, data, step, max_outputs=max_outputs, description=description
            )

    def __safe_upload_scalar(self, name, data, step, description=None):
        if data == []:
            return  # initialization, not required for tensorboard, ignore it
        data = asNumpyArray(data)
        try:
            # only take the last one in case of a list
            data = np.array(data).ravel()[-1]
        except:
            data = float(data)

        with self.writer.as_default():
            tfs.scalar(name, data, step, description)



    def update_z(self, z):
        self.__safe_upload_scalar(
            "zo (mm)", 1e3 * z, step=self.i, description="Propagation distance"
        )

    def _update_error_estimate(self, error):
        self.__safe_upload_scalar(
            "error metric", error, self.i, "Error metric (single image)"
        )

    def _update_probe_purity(self, probe_purity):
        if probe_purity is None:
            return
        self.__safe_upload_scalar("probe purity", probe_purity, self.i, "probe purity")

    def _update_object_purity(self, object_purity):

        if object_purity is None:
            return
        self.__safe_upload_scalar(
            "object purity", object_purity, self.i, "object purity"
        )

    def update_purities(self, probe_purity, object_purity):
        self._update_object_purity(object_purity)
        self._update_probe_purity(probe_purity)

    def describe_parameters(self, params: Params):
        text = "\n".join(["%s: %s" % (k, d) for (k, d) in params.__dict__.items()])
        with self.writer.as_default():
            tfs.text(
                "summary parameters", text, step=self.i, description="initial settings"
            )

    def __pad_probe_in_object_estimate(self, probe_estimate_rgb, object_estimate_rgb):
        probe_estimate_rgb_ss = probe_estimate_rgb[
            ..., :: self.probe_downsampling, :: self.probe_downsampling, :
        ]

        self.__safe_upload_scalar('probe downsampling in inset', self.probe_downsampling, self.i, 'probe downsampling in the inset images')
        Ny, Nx, _ = probe_estimate_rgb_ss.shape[-3:]
        if object_estimate_rgb.shape[-2] < Nx:
            raise RuntimeError(
                "The downsampled probe size would be larger than the downsampled object size."
                ""
                "Try setting monitor.probe_downsampling higher or monitor.object_downsampling lower."
            )
        # add a red marker around the edge
        probe_estimate_rgb_ss[..., -1, -1] = 255
        probe_estimate_rgb_ss[..., -1, :, -1] = 255

        # last channel is for color
        # print(Ny, Nx, object_estimate_rgb.shape, probe_estimate_rgb_ss.shape)
        object_estimate_rgb[..., :Ny, :Nx, :] = probe_estimate_rgb_ss[0]

        return object_estimate_rgb
