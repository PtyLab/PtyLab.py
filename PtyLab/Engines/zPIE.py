import numpy as np
import tqdm
from matplotlib import pyplot as plt

try:
    import cupy as cp
except ImportError:
    # print("Cupy not available, will not be able to run GPU based computation")
    # Still define the name, we'll take care of it later but in this way it's still possible
    # to see that gPIE exists for example.
    cp = None

import logging
import sys

from PtyLab.Engines.BaseEngine import BaseEngine
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.Operators.Operators import aspw
from PtyLab.Params.Params import Params

# PtyLab imports
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.utils.gpuUtils import asNumpyArray, getArrayModule


class zPIE(BaseEngine):
    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger("zPIE")
        self.logger.info("Sucesfully created zPIE zPIE_engine")
        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        self.initializeReconstructionParams()
        self.name = "zPIE"

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.numIterations = 50
        self.DoF = self.reconstruction.DoF
        self.zPIEgradientStepSize = 100  # gradient step size for axial position correction (typical range [1, 100])
        self.zPIEfriction = 0.7
        self.focusObject = True
        self.zMomentun = 0

    def show_defocus(self, viewer=None, scanrange_times_dof=1000, N_points=10):
        z = np.linspace(-1, 1, N_points) * scanrange_times_dof * self.reconstruction.DoF

        from PtyLab.Operators.Operators import aspw

        reconstruction = self.reconstruction
        defocii = np.abs(
            np.array(
                [
                    aspw(
                        reconstruction.object,
                        dz,
                        reconstruction.wavelength,
                        reconstruction.Lo,
                    )[0]
                    for dz in z
                ]
            )
            ** 2
        )

        if viewer is None:
            # currently a hacky way for this, these napari implementations must
            # later be moved to an optional sub-package.
            try:
                import napari

                viewer = napari.Viewer()
            except ImportError:
                msg = "Install napari to access this `NapariMonitor` implementation"
                raise ImportError(msg)

        viewer.add_image(defocii)

    def reconstruct(self, experimentalData=None, reconstruction=None):
        self.changeExperimentalData(experimentalData)
        self.changeOptimizable(reconstruction)
        self._prepareReconstruction()

        ###################################### actual reconstruction zPIE_engine #######################################

        xp = getArrayModule(self.reconstruction.object)
        if not hasattr(self.reconstruction, "zHistory"):
            self.reconstruction.zHistory = []

        # preallocate grids
        if self.params.propagatorType == "ASP":
            n = self.reconstruction.Np * 1
        else:
            n = 2 * self.reconstruction.Np

        if not self.focusObject:
            n = self.reconstruction.Np

        X, Y = xp.meshgrid(xp.arange(-n // 2, n // 2), xp.arange(-n // 2, n // 2))
        w = xp.exp(-((xp.sqrt(X**2 + Y**2) / self.reconstruction.Np) ** 4))

        self.pbar = tqdm.trange(
            self.numIterations, desc="zPIE", file=sys.stdout, leave=True
        )  # in order to change description to the tqdm progress bar
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()
            imProps = []

            # get positions
            if loop == 1:
                zNew = self.reconstruction.zo.copy()
            else:
                d = 10

                dz = np.linspace(-1, 1, 11) * d * self.DoF
                self.dz = dz

                merit = []
                # todo, mixed states implementation, check if more need to be put on GPU to speed up
                for k in np.arange(len(dz)):
                    imProp = None
                    if self.focusObject:
                        roi = slice(
                            self.reconstruction.No // 2 - n // 2,
                            self.reconstruction.No // 2 + n // 2,
                        )
                        imProp, _ = aspw(
                            u=xp.squeeze(self.reconstruction.object[..., roi, roi]),
                            z=dz[k],
                            wavelength=self.reconstruction.wavelength,
                            L=self.reconstruction.dxo * n,
                            bandlimit=False,
                        )
                    else:
                        if self.reconstruction.nlambda == 1:
                            imProp, _ = aspw(
                                u=xp.squeeze(self.reconstruction.probe[..., :, :]),
                                z=dz[k],
                                wavelength=self.reconstruction.wavelength,
                                L=self.reconstruction.Lp,
                            )
                        else:
                            nlambda = self.reconstruction.nlambda // 2
                            imProp, _ = aspw(
                                xp.squeeze(
                                    self.reconstruction.probe[nlambda, ..., :, :]
                                ),
                                dz[k],
                                self.reconstruction.spectralDensity[nlambda],
                                self.reconstruction.Lp,
                            )
                    imProps.append(imProp.get())
                    # TV approach
                    aleph = 1e-2
                    gradx = xp.roll(imProp, -1, axis=-1) - xp.roll(imProp, 1, axis=-1)
                    grady = xp.roll(imProp, -1, axis=-2) - xp.roll(imProp, 1, axis=-2)
                    merit.append(
                        xp.sum(xp.sqrt(abs(gradx) ** 2 + abs(grady) ** 2 + aleph))
                    )
                    # take a tiny break, we may overask the GPU
                    # yield 0, 0

                merit = xp.array(merit)
                if not hasattr(self.reconstruction, "TV_history"):
                    self.reconstruction.TV_history = []

                self.reconstruction.TV_history.append(
                    float(merit[len(merit) // 2].get())
                )
                if xp is not np:
                    merit = merit.get()
                feedback = np.sum(dz * merit) / np.sum(
                    merit
                )  # at optimal z, feedback term becomes 0

                print("Step size: ", feedback)
                self.zMomentun = (
                    self.zPIEfriction * self.zMomentun
                    + self.zPIEgradientStepSize * feedback
                )
                zNew = self.reconstruction.zo + self.zMomentun

                # asdlkcmasldk

            self.reconstruction.zHistory.append(self.reconstruction.zo)

            # print updated z
            self.pbar.set_description(
                "zPIE: update z = %.3f mm (dz = %.1f um)"
                % (self.reconstruction.zo * 1e3, self.zMomentun * 1e6)
            )

            # reset coordinates
            self.reconstruction.zo = zNew

            # re-sample is automatically done by using @property
            if self.params.propagatorType != "ASP":
                self.reconstruction.dxp = (
                    self.reconstruction.wavelength
                    * self.reconstruction.zo
                    / self.reconstruction.Ld
                )
                # reset propagatorType
                # self.reconstruction.quadraticPhase = xp.array(np.exp(1.j * np.pi / (self.reconstruction.wavelength * self.reconstruction.zo)
                #                                                      * (self.reconstruction.Xp ** 2 + self.reconstruction.Yp ** 2)))
            ##################################################################################################################

            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # print('Starting normal reconstruction loop')
                ### patch1 ###
                # get object patch
                row, col = self.reconstruction.positions[positionIndex]
                sy = slice(row, row + self.reconstruction.Np)
                sx = slice(col, col + self.reconstruction.Np)
                # note that object patch has size of probe array
                objectPatch = self.reconstruction.object[..., sy, sx].copy()

                # make exit surface wave
                self.reconstruction.esw = objectPatch * self.reconstruction.probe

                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.reconstruction.eswUpdate - self.reconstruction.esw

                # object update
                self.reconstruction.object[..., sy, sx] = self.objectPatchUpdate(
                    objectPatch, DELTA
                )

                # probe update
                self.reconstruction.probe = self.probeUpdate(objectPatch, DELTA)
            # yield positionLoop, positionIndex

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)
            # display it
            # self.showReconstruction(loop)

            self.merit = merit
            self.zNew = zNew
            self.reconstruction.merit = merit
            self.reconstruction.dz = dz

            self.reconstruction.make_alignment_plot(True)
            # show reconstruction
            if False:
                if loop == 0:
                    figure, axes = plt.subplots(
                        1, 3, num=666, squeeze=True, clear=True, figsize=(5, 5)
                    )
                    ax = axes[0]
                    ax_score = axes[1]
                    ax.set_title("Estimated distance (object-camera)")
                    ax.set_xlabel("iteration")
                    ax.set_ylabel("estimated z (mm)")
                    ax.set_xscale("symlog")

                    ax_score.set_title("TV score")
                    ax_score.set_xlabel("Distance [um]")
                    ax_score.set_ylabel("TV")
                    (score_line,) = ax_score.plot(dz * 1e6, merit)
                    (line,) = ax.plot(0, zNew, "o-")
                    plt.tight_layout()
                    plt.show(block=False)

                elif np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
                    idx = np.linspace(
                        0,
                        np.log10(len(self.reconstruction.zHistory) - 1),
                        np.minimum(len(self.reconstruction.zHistory), 100),
                    )
                    idx = np.rint(10**idx).astype("int")

                    line.set_xdata(idx)
                    line.set_ydata(np.array(self.reconstruction.zHistory)[idx] * 1e3)

                    score_line.set_ydata(merit)
                    ax_score.set_ylim(merit.min() - 1, merit.max() + 1)
                    ax.set_xlim(0, np.max(idx))
                    ax.set_ylim(
                        np.min(self.reconstruction.zHistory) * 1e3,
                        np.max(self.reconstruction.zHistory) * 1e3,
                    )

                    figure.canvas.draw()
                    figure.canvas.flush_events()
            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info("switch to cpu")
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)

        frac = self.reconstruction.probe.conj() / xp.max(
            xp.sum(xp.abs(self.reconstruction.probe) ** 2, axis=(0, 1, 2, 3))
        )
        return objectPatch + self.betaObject * xp.sum(
            frac * DELTA, axis=(0, 2, 3), keepdims=True
        )

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = objectPatch.conj() / xp.max(
            xp.sum(xp.abs(objectPatch) ** 2, axis=(0, 1, 2, 3))
        )
        r = self.reconstruction.probe + self.betaProbe * xp.sum(
            frac * DELTA, axis=(0, 1, 3), keepdims=True
        )
        return r

    # def display_focus_bokeh(self):
    #     # first, make a document
    #     from bokeh.plotting import figure, output_file, save
    #     from bokeh.io import hplot
    #     from pathlib import Path
    #
    #     folder = Path('plots/zPIE.html')
    #     output_file(folder)
    #     s1, s2 = self.reconstruction.make_alignment_plot(False)
    #     s3 = figure(width=250, height=250, title='TV per focus')
    #
    #
    #     s3.xaxix.axis_label = 'Distance [um]'
    #     s3.yaxix.axis_label = 'TV'
    #     s3.line(self.dz*1e6, self.merit, )
    #     pass
    #
    #
