import pathlib

from PtyLab.utils.visualisation import modeTile
import io

import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
from matplotlib.ticker import EngFormatter
from PtyLab.Monitor.Monitor import AbstractMonitor
from PtyLab import Params
from PtyLab.utils.gpuUtils import asNumpyArray
from PtyLab.utils.utils import fft2c
from PtyLab.utils.visualisation import complex2rgb, complex2rgb_vectorized
from tensorflow import summary as tfs
from scipy import ndimage

import matplotlib

from tensorflow import image
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
    # show probe intensity. Makes sense if it is known
    show_probe_intensity = False

    # maximum number of probe state mixtures that we want to show
    max_npsm = 10
    # maximum number of object state mixtures we want to show
    max_nosm = 10
    # remove any phase slant from the object
    center_angle_object = False
    # Turn on to show the FFT of the object. Usually not useful.
    show_farfield_object = False

    xminmax = 0, -1
    yminmax = 0, -1

    # the probe is shown as an inset in the object. This specifies how much to downsample it
    probe_downsampling = 2

    # downsample all images by the same amount to make it run a bit faster
    downsample_everything = 1

    def __init__(self, logdir="./logs_tensorboard", name=None):
        super(AbstractMonitor).__init__()
        # if true, all phases are centered in such a way that the average phase in the center of any RGB plot is zero.
        self.center_phases = True
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

    def update_aux(self, engine):
        self.plot_error_per_position(engine)


    def plot_error_per_position(self, engine):
        probabilities = asNumpyArray(engine.reconstruction.errorAtPos)
        probabilities = probabilities / probabilities.sum()
        indices, updates = np.unique(engine.positionIndices, return_counts=True)
        N = len(engine.positionIndices)

        cmap: matplotlib.colors.Colormap = matplotlib.colormaps['gray'].copy()
        cmap.set_over('r')
        cmap.set_under('g')

        fig, axes = plt.subplot_mosaic("EPDHh\nEPDHh", num=11, clear=True, layout="constrained",
                                       height_ratios=[1, 1],
                                       width_ratios=[1, 1, 1, 0.5, 0.5],
                                       figsize=(12, 3))
        ax = axes['E']
        ax2 = axes['H']
        ax3 = axes['P']
        ax3.sharex(ax)
        ax3.sharey(ax)
        ax_density = axes['D']

        for key in 'EPD':
            axes[key].set_aspect(1)
            # show the error per position
        ax.set_title('Rel. error')
        cm = ax.scatter(engine.reconstruction.positions[:, 1], engine.reconstruction.positions[:, 0], c=probabilities * N,
                        vmin=0.5, vmax=2,
                        cmap=cmap)
        plt.colorbar(cm, orientation='horizontal', ticks=np.array([0.5, 1.0, 1.5, 2.0]), extend='both')
        # show the new positions for the next run
        cm = ax3.scatter(engine.reconstruction.positions[indices, 1], engine.reconstruction.positions[indices, 0],
                         c=updates + 0.5, cmap='Accent',
                         vmin=0, vmax=8)
        plt.colorbar(cm, orientation='horizontal', extend='both')
        ax3.set_title('Processed pos')

        # histogram of error distribution
        ax2.hist(probabilities * N, bins=np.linspace(0, 5, 100))
        ax2.set_title('Relative error')
        ax2.set_xlabel('Relative error')
        ax2.set_ylabel('#')

        axes['h'].hist(engine.counts_per_position / engine.counts_per_position.mean(),
                       bins=len(engine.counts_per_position) // 2)

        axes['h'].set_title('Visiting frequency')
        axes['h'].set_xlabel('Visit freq')
        for label in 'Hh':
            from matplotlib.ticker import MaxNLocator, FormatStrFormatter
            axes[label].yaxis.set_major_locator(MaxNLocator(5, integer=True))
            axes[label].yaxis.set_major_formatter(FormatStrFormatter('%.3d'))
            axes[label].set_xlim(0, 3)
        # density of positions processed
        ax_density.set_title('# visiting freq.')
        cmap = matplotlib.colormaps['viridis'].copy()
        cmap.set_under('gray')
        cmap.set_over('r')
        cm = ax_density.scatter(engine.reconstruction.positions[:, 1],
                                engine.reconstruction.positions[:, 0],
                                cmap=cmap,
                                c=engine.counts_per_position / engine.counts_per_position.mean(),
                                vmin=0.5, vmax=2)
        plt.colorbar(cm, orientation='horizontal', extend='both')
        lims = ax_density.get_ylim()
        for a in 'EPD':
            # flip the orientation to make it match with the images we reconstruct
            axes[a].set_ylim(lims[::-1])


        # P = pathlib.Path(f'probability_{self.params.positionOrder}')
        # P.mkdir(exist_ok=True)
        # plt.savefig(P / f'{i if i is not None else "prob"}.png', dpi=300)
        # plt.savefig('density.png')
        # plt.show()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        with self.writer.as_default():
            img = image.decode_png(buf.getvalue(), channels=4)
            img = np.expand_dims(img, 0)
            tfs.image('position sampling', img, self.i)

    def updatePlot(self, object_estimate, probe_estimate, zo=None, encoder_positions=None, highres=True, TV_weight=None,
                   TV_frequency=None):
        self.i += 1
        self.update_TV_reg_weight(TV_weight, TV_frequency)
        if not highres:
            Np = probe_estimate.shape[-1]
            No = object_estimate.shape[-1]
            xmax, ymax = np.clip(encoder_positions.max(axis=0) + 2 * Np // 3, 0, No)
            xmin, ymin = np.clip(encoder_positions.min(axis=0) + Np // 3, 0, No)

            self.yminmax = ymin, ymax
            self.xminmax = xmin, xmax

            probe_estimate = probe_estimate[..., ::self.probe_downsampling, ::self.probe_downsampling]
            object_estimate = object_estimate[..., ymin:ymax, xmin:xmax]
        if self.show_probe_intensity:
            I_probe = abs(probe_estimate)
            I_probe = I_probe / I_probe.max() * 255
            self.__safe_upload_image('probe/Intensity', np.hstack(np.clip(I_probe, 0, 255).astype(np.uint8)),
                                     step=self.i)

        probe_estimate_rgb = self._update_probe_estimate(
            probe_estimate, highres=highres
        )
        self._update_object_estimate(
            object_estimate, probe_estimate_rgb, highres=highres, zo=zo
        )


    def visualize_probe_engine(self, engine):
        RGB_image = complex2rgb_vectorized(engine.get_fundamental(), center_phase=self.center_phases)
        self.__safe_upload_image("original probe", np.squeeze(RGB_image), self.i)
        pass

    def updateObjectProbeErrorMonitor(self, error, object_estimate, probe_estimate, zo=None, purity_probe=None,
                                      purity_object=None, encoder_positions=None, normalized_probe_powers=None,
                                      TV_weight=None, TV_frequency=None):
        object_estimate = asNumpyArray(object_estimate)
        probe_estimate = asNumpyArray(probe_estimate)
        # The input can be either an empty array, an array with length 1 or a list of all the errors so far.
        # In the last case, we only want the last value.
        # or just a number. This part should account of all of them.
        if encoder_positions is None:
            raise ValueError("Please submit encoder positions or the code won't work.")
        self._update_error_estimate(error)

        self.updatePlot(object_estimate, probe_estimate, zo=zo, encoder_positions=encoder_positions, TV_weight=TV_weight, TV_frequency=TV_frequency)

        self.update_z(zo)
        self.update_purities(asNumpyArray(purity_probe), asNumpyArray(purity_object))

        self.update_normalized_probe_powers(normalized_probe_powers, purity_probe)

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

    def update_focusing_metric(self, TV_value, AOI_image, metric_name, allmerits=None):
        if TV_value is not None:
            self.__safe_upload_scalar(
                f"Autofocus {metric_name}",
                TV_value,
                self.i,
                "Total Variation of the object",
            )
        if AOI_image is not None:
            # print(AOI_image)

            self.__smart_upload_image_couldbecomplex(
                f"Autofocus {metric_name} AOI",
                AOI_image,
                self.i,
                1,
                "AOI used by autofocus",
                center_phase=True,
            )
        if allmerits is not None:
            import matplotlib.pyplot as plt
            allmerits, new_z = allmerits
            fig, ax = plt.subplot_mosaic('A')
            ax = ax['A']
            ax.plot(allmerits[0], allmerits[1], '-ro')
            ax.vlines(new_z, *ax.get_ylim())
            ax.set_title(f'{metric_name}')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=70)
            plt.close(fig)
            buf.seek(0)
            with self.writer.as_default():
                img = image.decode_png(buf.getvalue(), channels=4)
                img = np.expand_dims(img, 0)
                tfs.image('Autofocus dz score', img, self.i)

    def updateBeamWidth(self, beamwidth_y, beamwidth_x):
        self.__safe_upload_scalar('beamwidth/x_um', beamwidth_x*1e6, step=self.i)
        self.__safe_upload_scalar('beamwidth/y_um', beamwidth_y*1e6, step=self.i)

    def update_overlap(self, overlap_area, linear_overlap):
        self.__safe_upload_scalar('overlap/area', overlap_area, step=self.i)
        self.__safe_upload_scalar('overlap/linear', linear_overlap, step=self.i)


    def update_encoder(
        self,
        corrected_positions: np.ndarray,
        original_positions: np.ndarray,
        scaling: float = 1.0,
            beamwidth=None
    ) -> None:
        """
        Update the stage position images.
        :param corrected_positions:
        :param original_positions:
        :param scaling:
        :return:
        """
        # convert the positions to mm
        corrected_positions = corrected_positions
        original_positions = original_positions
        # set them to mean 0
        corrected_positions = corrected_positions - corrected_positions.mean(
            axis=0, keepdims=True
        )
        original_positions = original_positions - original_positions.mean(
            axis=0, keepdims=True
        )



        matplotlib.use("Agg")  # no images output
        import matplotlib.pyplot as plt

        # make a fov that makes sense
        scale_0 = 1.1
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
        ONS""",
            constrained_layout=True,
            figsize=(15, 8),
        )

        meandiff = np.mean(
            abs(1e6 * corrected_positions - 1e6 * original_positions)
        )

        self.__safe_upload_scalar(
            "mean position displacement in micron",
            meandiff,
            self.i,
            "mean absolute displacement",
        )

        axes["O"].set_title("Original positions")
        axes["N"].set_title("Updated positions")
        axes["S"].set_title(f"Diff. Mean: {meandiff} $\mu$m")

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
            ax.xaxis.set_major_formatter(EngFormatter(unit="m"))
            ax.yaxis.set_major_formatter(EngFormatter(unit="m"))
            ax.set_xlim(ax.set_ylim(position_range))

        # plot the new one in the middle image
        axes["N"].scatter(
            corrected_positions[:, 0],
            corrected_positions[:, 1],
            color="C1",
            marker="x",
            label="new",
        )
        # scaled version on the right (should only show displacement, not magnification)
        diff = corrected_positions - original_positions
        axes["S"].quiver(
            original_positions[:, 0],
            original_positions[:, 1],
            diff[:, 0],
            diff[:, 1],
            angles="xy",
            units="xy",
            scale=1,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=70)
        plt.close(fig)
        buf.seek(0)
        with self.writer.as_default():
            img = image.decode_png(buf.getvalue(), channels=4)
            img = np.expand_dims(img, 0)
            tfs.image("position correction", img, self.i)

    # internal use for tensorboardMonitor
    def _update_object_estimate(
        self,
        object_estimate,
        probe_estimate_rgb,
        highres=True,
        zo=None,
    ):
        """
        Update the object estimate. This ensures that within the web interface the object and the probe estimate are available.


        :param object_estimate:
        :param probe_estimate_rgb:
        :param highres:
        :param z: Object-detector distance. Not required.
        :return:
        """
        if self.center_angle_object:
            object_estimate, shift1 = center_angle(object_estimate)
            object_estimate, shift2 = center_angle(object_estimate)
            print("Angle shifts: ", shift1, shift2)

        # convert the object estimate to colour
        object_estimate_rgb = complex2rgb_vectorized(object_estimate, center_phase=self.center_phases
                                                     )

        # ensure it's 4 d as that's what is needed by tensorflow
        if object_estimate_rgb.ndim == 3:
            object_estimate_rgb = object_estimate_rgb[None]

        # patch in an image of the first probe in the top right
        object_estimate_rgb = self.__pad_probe_in_object_estimate(
            probe_estimate_rgb, object_estimate_rgb
        )

        # Add an object, but if it's the low-resolution version add it to the (low_res) version.
        tag = "object/estimate"
        if not highres:
            tag += " (low res)"
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
            self.__safe_upload_image("object/I ff", I_ff, self.i, self.max_nosm)

        self.__safe_upload_image(tag, object_estimate_rgb, self.i, self.max_npsm)

        if highres:
            sy = slice(*self.yminmax)
            sx = slice(*self.xminmax)
            I_object = abs(object_estimate[...,sy,sx]**2)


            min_int = 0  # mean_obj - 2 * std_obj
            from scipy import ndimage

            max_int = ndimage.gaussian_filter(I_object, 3).max()

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
                "object/I", I_object, self.i, self.max_nosm
            )

            self.__safe_upload_image(
                "object/I log", logI.astype(np.uint8), self.i, self.max_nosm
            )
            self.save_intensities = False
            if self.save_intensities:
                p = Path("./intensities")
                p.mkdir(exist_ok=True)

                import matplotlib.pyplot as plt

                plt.clf()
                im = plt.imshow(I_object)
                plt.colorbar(im)
                if zo is not None:
                    plt.title(f"z = {zo*1e3:.3f} mm")
                plt.savefig(f"intensities/{self.i}.png")
                plt.savefig(f"intensities/AAA.png")

    def _update_probe_estimate(self, probe_estimate, highres=True):
        # first, convert it to images
        # while probe_estimate.ndim <= 3:
        #     probe_estimate = probe_estimate[None]
        # probe_estimate_rgb = complex2rgb_vectorized(probe_estimate, center_phase=self.center_phases)

        probe_estimate_rgb  = complex2rgb(modeTile(probe_estimate), center_phase=self.center_phases)

        #probe_estimate_rgb = complex2rgb(np.hstack(probe_estimate), center_phase=self.center_phases)
        # ensure it's 4 d as that's what is needed by tensorflow
        tag = "probe/estimate"
        if not highres:
            tag += "(low res)"
        self.__safe_upload_image(tag, probe_estimate_rgb, self.i, self.max_npsm)
        N = probe_estimate.shape[-1]
        probe_estimate_rgb = probe_estimate_rgb[None, :N, :N, :]

        if highres:
            # this is an expensive operation, and we usually only need one, so take the first one instead
            # of all of them
            #ff_probe = fft2c(probe_estimate[:1])
            ff_probe = complex2rgb(fft2c(probe_estimate[0]), center_phase=self.center_phases)
            self.__safe_upload_image(tag + 'FF', ff_probe, self.i, self.max_npsm)

        # make a probe COM estimate
        from scipy import ndimage
        P = probe_estimate
        while P.ndim > 2:
            P = P[0]
        cy, cx = ndimage.center_of_mass(abs(P**2))
        N = probe_estimate.shape[-1]
        self.__safe_upload_scalar('probe/cy', cy-N//2, self.i)
        self.__safe_upload_scalar('probe/cx', cx-N//2, self.i)
        return probe_estimate_rgb



    def __smart_upload_image_couldbecomplex(
        self, name, data, step, max_outputs=3, description=None, center_phase=False,
    ):
        """
        Safely upload an image that could be complex. If it is, cast it to colour before uploading.

        """
        data = asNumpyArray(data)


        if np.iscomplexobj(data):
            if center_phase:
                phexp = data.sum((-2,-1), keepdims=True)
                phexp = phexp.conj() / (abs(phexp) + 1e-9)
            else:
                phexp = 1
            print("Got complex datatype")
            data = complex2rgb_vectorized(data*phexp)
        else:
            print("Got real datatype")
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
                name,
                data[
                    ..., :: self.downsample_everything, :: self.downsample_everything, :
                ],
                step,
                max_outputs=max_outputs,
                description=description,
            )

    def __safe_upload_scalar(self, name, data, step, description=None):
        if isinstance(data, list):
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

    def update_TV_reg_weight(self, new_weight, new_frequency):
        if new_weight is not None:
            self.__safe_upload_scalar('regularization/TV (1e6)', new_weight*1e6, step=self.i, description='TV reg weight')
        if new_frequency is not None:
            self.__safe_upload_scalar('regularization/TV freq', new_frequency, step=self.i)

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

        self.__safe_upload_scalar(
            "probe downsampling in inset",
            self.probe_downsampling,
            self.i,
            "probe downsampling in the inset images",
        )
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

    def update_normalized_probe_powers(self, normalized_probe_powers, purity_probe):
        """
        Make a plot of the probe powers (bar diagram).
        Parameters
        ----------
        normalized_probe_powers

        Returns
        -------

        """
        if normalized_probe_powers is None:
            return
        fig, ax = plt.subplot_mosaic('P')
        normalized_probe_powers = asNumpyArray(normalized_probe_powers)
        ax['P'].bar(np.arange(len(normalized_probe_powers)), normalized_probe_powers)
        ax['P'].set_xticks(np.arange(len(normalized_probe_powers)))
        ax['P'].set_ylim(0,1)
        # save and upload
        ax['P'].set_title(f'Purity {purity_probe:.2f}')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=70)
        plt.close(fig)
        buf.seek(0)
        with self.writer.as_default():
            img = image.decode_png(buf.getvalue(), channels=4)
            img = np.expand_dims(img, 0)
            tfs.image('probe/relative_powers', img, self.i)


