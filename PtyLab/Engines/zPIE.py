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
    r"""
    Ptychographic iterative engine with axial-distance refinement.

    zPIE extends the standard PIE reconstruction by jointly refining the
    object/probe estimate and the axial propagation distance $z$.[^loetgering2020]

    During reconstruction, a set of candidate axial offsets is generated
    around the current propagation distance:

    $$
    \Delta z_k \in [-d\,\mathrm{DoF},\, d\,\mathrm{DoF}]
    $$

    For each candidate offset, either the reconstructed object or probe is
    propagated to the corresponding defocus plane using angular-spectrum
    propagation. The quantity used for axial optimization is selected through
    `focusObject`.

    A total-variation-based focus metric is evaluated from the propagated
    complex field:

    $$
    M_k = \sum \sqrt{|\nabla_x U_k|^2 + |\nabla_y U_k|^2 + \epsilon}
    $$

    where $U_k$ is the propagated field at axial offset $\Delta z_k$ and
    $\epsilon$ is a small numerical stabilization term.

    The axial feedback is calculated from the merit values as

    $$
    f_z = \frac{\sum_k \Delta z_k M_k}{\sum_k M_k}
    $$

    and accumulated using a momentum-like update:

    $$
    m_z^{(n)} = \gamma m_z^{(n-1)} + \eta f_z
    $$

    followed by

    $$
    z^{(n+1)} = z^{(n)} + m_z^{(n)}
    $$

    where $\gamma$ is `zPIEfriction` and $\eta$ is
    `zPIEgradientStepSize`.

    The updated axial distance is stored in `reconstruction.zo`. For
    propagation schemes whose sampling depends explicitly on distance, the
    corresponding propagation coordinates are updated after each axial
    correction.

    The object and probe are subsequently updated using the standard PIE
    object and probe update rules.

    Parameters specific to zPIE include:

    - `DoF`: depth-of-field scale used to define the axial search range.
    - `zPIEgradientStepSize`: strength of the axial feedback update.
    - `zPIEfriction`: momentum retention factor for axial refinement.
    - `focusObject`: if `True`, evaluate the focus metric on the object;
      otherwise evaluate it on the probe.
    - `zMomentun`: accumulated axial momentum.

    [^loetgering2020]: L. Loetgering, M. Du, K. S. E. Eikema, and S. Witte,
        "zPIE: an autofocusing algorithm for ptychography,"
        Optics Letters 45, 2030-2033 (2020).
        https://doi.org/10.1364/OL.389492
    """
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
        Initialize zPIE-specific reconstruction parameters.

        The default parameters are:

        - `betaProbe = 0.25`:
        Probe update step size used during the PIE reconstruction.

        - `betaObject = 0.25`:
        Object update step size used during the PIE reconstruction.

        - `numIterations = 50`:
        Number of reconstruction iterations.

        - `DoF = reconstruction.DoF`:
        Depth-of-field scale used to define the axial search interval.

        - `zPIEgradientStepSize = 100`:
        Scaling factor applied to the axial feedback term when updating the
        sample-to-detector distance.

        - `zPIEfriction = 0.7`:
        Momentum retention factor used for axial-distance refinement.

        - `focusObject = True`:
        If `True`, evaluate the axial focus metric on the reconstructed object.
        If `False`, evaluate it on the reconstructed probe.

        - `zMomentun = 0`:
        Initial accumulated axial momentum used for iterative z refinement.

        These values can be modified after creating the zPIE engine and before
        calling `reconstruct()`.
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
        r"""
        Visualize a defocus stack of the reconstructed object.

        A set of axial offsets is generated symmetrically around the current
        reconstruction plane:

        $$
        \Delta z_k \in [-s\,\mathrm{DoF},\; s\,\mathrm{DoF}]
        $$

        where `s` is given by `scanrange_times_dof` and `DoF` is taken from
        `reconstruction.DoF`.

        For each axial offset, the current reconstructed object is propagated
        using angular-spectrum propagation (`aspw`). The propagated intensity
        is calculated as the squared magnitude of the complex field and collected
        into a defocus stack.

        The resulting stack is displayed in a napari viewer. If no viewer is
        provided, a new napari viewer is created automatically.

        Args:
            viewer (napari.Viewer, optional):
                Existing napari viewer to which the defocus stack is added.
                If `None`, a new viewer is created.

            scanrange_times_dof (float, optional):
                Half-width of the axial scan expressed in units of the
                reconstruction depth of field. Default is `1000`.

            N_points (int, optional):
                Number of equally spaced axial positions in the defocus stack.
                Default is `10`.

        Raises:
            ImportError:
                If `viewer` is `None` and napari is not installed.

        Notes:
            This function is intended as a diagnostic visualization tool and does
            not modify the reconstructed object or the current axial distance.
        """
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
        r"""
        Run the zPIE reconstruction with iterative axial-distance refinement.

        The reconstruction jointly updates the object, probe, and
        sample-to-detector propagation distance. Before the iterative loop starts,
        the experimental data and reconstruction state are updated if new objects
        are supplied, followed by the standard reconstruction preparation.

        During each zPIE iteration, a set of candidate axial offsets is generated
        around the current propagation distance:

        $$
        \Delta z_k \in [-10\,\mathrm{DoF},\;10\,\mathrm{DoF}]
        $$

        using 11 equally spaced candidate positions.

        Depending on `focusObject`, either the reconstructed object or probe is
        propagated to each candidate defocus plane using angular-spectrum
        propagation. A total-variation-based merit value is calculated for each
        propagated field.

        The axial feedback is then obtained from the weighted candidate offsets:

        $$
        f_z = \frac{\sum_k \Delta z_k M_k}{\sum_k M_k}
        $$

        where $M_k$ is the focus merit evaluated at candidate offset
        $\Delta z_k$.

        The feedback is accumulated through the axial momentum update

        $$
        m_z^{(n)} = \gamma m_z^{(n-1)} + \eta f_z
        $$

        where $\gamma$ is `zPIEfriction` and $\eta$ is
        `zPIEgradientStepSize`. The propagation distance is then updated as

        $$
        z^{(n+1)} = z^{(n)} + m_z^{(n)}
        $$

        After the axial update, a standard PIE reconstruction loop is performed
        over all scan positions. The exit surface wave is calculated from the
        current object patch and probe, followed by the intensity constraint and
        object/probe updates.

        The current axial distance is stored during reconstruction in
        `reconstruction.zHistory`. The total-variation merit values and candidate
        axial offsets are also stored in `reconstruction.merit` and
        `reconstruction.dz` for diagnostic use.

        For propagation schemes whose object-plane sampling depends explicitly on
        the propagation distance, the sampling `reconstruction.dxp` is updated
        after each axial correction.

        At the end of each iteration, reconstruction error metrics and enabled
        constraints are evaluated, followed by an update of the reconstruction
        monitor.

        Args:
            experimentalData (ExperimentalData, optional):
                Experimental dataset to use. If provided, it replaces the dataset
                currently attached to the engine.

            reconstruction (Reconstruction, optional):
                Reconstruction state to optimize. If provided, it replaces the
                reconstruction currently attached to the engine.

        See Also:
            `show_defocus`
                Visualize the reconstructed object over a range of axial defocus
                positions.

            `objectPatchUpdate`
                Update the reconstructed object patch during the PIE loop.

            `probeUpdate`
                Update the reconstructed probe during the PIE loop.
        """
        
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
        Update the object patch using the standard ePIE object-update rule.

        This implementation is identical to `ePIE.objectPatchUpdate()`.

        See Also:
            `ePIE.objectPatchUpdate`
                Standard ePIE object update.
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
        Update the probe using the standard ePIE probe-update rule.

        This implementation is identical to `ePIE.probeUpdate()`.

        See Also:
            `ePIE.probeUpdate`
                Standard ePIE probe update.
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
