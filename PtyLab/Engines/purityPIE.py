import numpy as np
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

import tqdm

from PtyLab.Engines.BaseEngine import BaseEngine
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.Params.Params import Params

# PtyLab imports
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.utils.gpuUtils import asNumpyArray, getArrayModule
from PtyLab.utils.utils import fft2c, ifft2c


class purityPIE(BaseEngine):
    '''
    purityPIE extends mPIE-style reconstruction with purity-based axial calibration.
    '''
    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        """
        Initialize the mPIE reconstruction engine.

        Shared reconstruction state is initialized through `BaseEngine`, followed
        by the mPIE-specific reconstruction parameters and momentum buffers.

        Momentum acceleration is enabled through
        `params.momentumAcceleration`, allowing shared BaseEngine operations such
        as modal orthogonalization to keep the corresponding momentum and buffer
        arrays consistent with the reconstructed object and probe.

        Args:
            reconstruction (Reconstruction):
                Reconstruction state containing the current object, probe, and
                geometry.
            experimentalData (ExperimentalData):
                Experimental diffraction data and acquisition parameters.
            params (Params):
                Shared reconstruction parameters and constraint settings.
            monitor (Monitor):
                Monitor used for reconstruction visualization and progress
                reporting.
    """
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger("purityPIE")
        self.logger.info("Successfully created purityPIE engine")
        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)

        self.initializeReconstructionParams()
        self.params.momentumAcceleration = True
        self.name = "purityPIE"

    @property
    def keepPatches(self):
        """
        Whether to store the reconstructed object patch for every scan position.

        Enable with `engine.keepPatches = True` and disable with `engine.keepPatches = False`.

        This option is intended for debugging or detailed analysis and may require a large amount of additional memory.
        """
        return hasattr(self, "patches")

    @keepPatches.setter
    def keepPatches(self, keep_them):

        if keep_them:
            self.logger.info("Keeping patches!")
            self.patches = np.zeros(
                (
                    self.experimentalData.ptychogram.shape[0],
                    *self.reconstruction.shape_O,
                ),
                np.complex64,
            )
        else:
            self.logger.info("Not keeping patches")
            if hasattr(self, "patches"):
                del self.patches

    def initializeReconstructionParams(self):
        """
        Initialize mPIE-specific reconstruction parameters and momentum state.

        The default mPIE parameters are:

        - `betaObject = 0.25`: object update step size.
        - `betaProbe = 0.25`: probe update step size.
        - `alphaObject = 0.1`: object-update regularization parameter.
        - `alphaProbe = 0.1`: probe-update regularization parameter.
        - `feedbackM = 0.3`: momentum feedback strength.
        - `frictionM = 0.7`: momentum memory coefficient.
        - `numIterations = 50`: number of reconstruction iterations.

        Object and probe momentum arrays are initialized together with corresponding
        buffers that store the reconstruction state used by the momentum updates.
        """
        # self.eswUpdate = self.reconstruction.esw.copy()
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.alphaProbe = 0.1  # probe regularization
        self.alphaObject = 0.1  # object regularization
        self.feedbackM = 0.3  # feedback
        self.frictionM = 0.7  # friction
        self.numIterations = 50

        # initialize momentum
        self.reconstruction.initializeObjectMomentum()
        self.reconstruction.initializeProbeMomentum()
        # set object and probe buffers
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()

        self.reconstruction.probeWindow = np.abs(self.reconstruction.probe)

    def reconstruct(
        self,
        experimentalData=None,
        reconstruction=None,
    ):
        """
        Run purity-based axial calibration.

        Each candidate propagation distance is reconstructed from the same
        initial state using `numIterations` mPIE-style iterations. The distance
        yielding the highest probe purity is selected, and the corresponding
        reconstruction state is retained.

        If `params.purityZScanPlot` is enabled, the resulting purity-versus-z
        curve is displayed after calibration.

        Args:
            experimentalData (ExperimentalData, optional):
                Experimental dataset to use. If provided, it replaces the
                currently attached dataset.
            reconstruction (Reconstruction, optional):
                Reconstruction state to calibrate. If provided, it replaces the
                currently attached reconstruction.

        Returns:
            tuple:
                `(best_z, best_purity)`, where `best_z` is the selected axial
                propagation distance and `best_purity` is the corresponding
                probe purity.
        """

        self.changeExperimentalData(experimentalData)
        self.changeOptimizable(reconstruction)

        self._prepareReconstruction()

        # Reset momentum buffers after preparation
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()

        # Purity-based axial calibration
        best_z, best_purity = self.purityZScan()

        # Optional scan visualization
        if self.params.purityZScanPlot:
            self.plotPurityZScan()

        # Return reconstruction data to CPU
        if self.params.gpuFlag:
            self.logger.info("switch to cpu")
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

        return best_z, best_purity

    def purityZScan(self):
        """
        Scan a fixed axial range and select the distance that maximizes
        reconstructed probe purity.

        Every candidate z is evaluated from the same initial reconstruction
        state using `numIterations` mPIE iterations.
        """

        if self.reconstruction.npsm < 2:
            raise ValueError(
                "Purity-based z scanning requires at least two probe modes "
                "(reconstruction.npsm >= 2)."
            )

        # Initial z guess defines the center of the scan
        z0 = self.reconstruction.zo
        self.purityZInitialGuess = z0

        z_values = np.linspace(
            z0 - self.params.purityZScanRange,
            z0 + self.params.purityZScanRange,
            self.params.purityZScanPoints,
        )

        # Save the common starting state
        object0 = self.reconstruction.object.copy()
        probe0 = self.reconstruction.probe.copy()

        objectMomentum0 = self.reconstruction.objectMomentum.copy()
        probeMomentum0 = self.reconstruction.probeMomentum.copy()

        objectBuffer0 = self.reconstruction.objectBuffer.copy()
        probeBuffer0 = self.reconstruction.probeBuffer.copy()

        purity_values = []

        best_purity = -np.inf
        best_z = z0
        best_state = None

        # Do not allow orthogonalization during the internal mPIE iterations.
        # Purity is evaluated once after each candidate reconstruction.
        #orthogonalization_switch = self.params.orthogonalizationSwitch
        #self.params.orthogonalizationSwitch = False

        tqdm.tqdm.write("")
        tqdm.tqdm.write("Starting purity-based z scan")
        tqdm.tqdm.write(
            f"Initial z = {z0 * 1e3:.6f} mm | "
            f"range = ±{self.params.purityZScanRange * 1e6:.1f} µm | "
            f"points = {len(z_values)} | "
            f"iterations/z = {self.numIterations}"
        )
        tqdm.tqdm.write("")

        #try:
        for z_index, z in enumerate(z_values):

            # Restore exactly the same starting state for every candidate z
            self.reconstruction.object = object0.copy()
            self.reconstruction.probe = probe0.copy()

            self.reconstruction.objectMomentum = objectMomentum0.copy()
            self.reconstruction.probeMomentum = probeMomentum0.copy()

            self.reconstruction.objectBuffer = objectBuffer0.copy()
            self.reconstruction.probeBuffer = probeBuffer0.copy()

            # Set candidate propagation distance
            self.reconstruction.zo = z

            dz_um = (z - z0) * 1e6

            tqdm.tqdm.write(
                f"[{z_index + 1}/{len(z_values)}] "
                f"Reconstructing z = {z * 1e3:.6f} mm "
                f"(Δz = {dz_um:+.2f} µm)"
            )

            # Run mPIE reconstruction at this z
            for loop in range(self.numIterations):
                self._run_single_iteration(loop)
                #self.showReconstruction(loop)

            # Orthogonalize once and evaluate purity
            self.orthogonalization()
            

            purity = float(
                np.asarray(
                    asNumpyArray(self.reconstruction.purityProbe)
                ).squeeze()
            )

            purity_values.append(purity)
            # Force monitor to display the final state and final purity
            self.showReconstruction(
                self.numIterations - 1,
                force=True,
            )
            tqdm.tqdm.write(
                f"    Probe purity = {purity:.6f}"
            )

            if purity > best_purity:
                best_purity = purity
                best_z = z

                best_state = {
                    "object": self.reconstruction.object.copy(),
                    "probe": self.reconstruction.probe.copy(),
                    "objectMomentum": self.reconstruction.objectMomentum.copy(),
                    "probeMomentum": self.reconstruction.probeMomentum.copy(),
                    "objectBuffer": self.reconstruction.objectBuffer.copy(),
                    "probeBuffer": self.reconstruction.probeBuffer.copy(),
                }

                tqdm.tqdm.write(
                    f"    NEW BEST: z = {best_z * 1e3:.6f} mm, "
                    f"purity = {best_purity:.6f}"
                )

            tqdm.tqdm.write("")

        #finally:
        #        self.params.orthogonalizationSwitch = orthogonalization_switch

        if best_state is None:
            raise RuntimeError(
                "Purity z scan did not produce a valid purity value."
            )

        # Restore best-z reconstruction state
        self.reconstruction.object = best_state["object"]
        self.reconstruction.probe = best_state["probe"]

        self.reconstruction.objectMomentum = best_state["objectMomentum"]
        self.reconstruction.probeMomentum = best_state["probeMomentum"]

        self.reconstruction.objectBuffer = best_state["objectBuffer"]
        self.reconstruction.probeBuffer = best_state["probeBuffer"]

        self.reconstruction.zo = best_z

        # Store scan results
        self.purityZValues = z_values
        self.purityZMetrics = np.asarray(purity_values)
        self.purityBestZ = best_z
        self.purityBestValue = best_purity

        # Final summary
        tqdm.tqdm.write("=" * 60)
        tqdm.tqdm.write("Purity-based z scan finished")
        tqdm.tqdm.write(
            f"Best z      = {best_z * 1e3:.6f} mm"
        )
        tqdm.tqdm.write(
            f"Best Δz     = {(best_z - z0) * 1e6:+.2f} µm"
        )
        tqdm.tqdm.write(
            f"Best purity = {best_purity:.6f}"
        )
        tqdm.tqdm.write("=" * 60)

        return best_z, best_purity

    def purityZScan_X(self):
        """
        Scan a fixed axial range and select the distance that maximizes
        reconstructed probe purity.

        Every candidate z is evaluated from the same initial reconstruction
        state using `numIterations` mPIE iterations.
        """

        if self.reconstruction.npsm < 2:
            raise ValueError(
                "Purity-based z scanning requires at least two probe modes "
                "(reconstruction.npsm >= 2)."
            )

        # Initial z guess defines the center of the scan
        z0 = self.reconstruction.zo
        self.purityZInitialGuess = z0

        z_values = np.linspace(
            z0 - self.params.purityZScanRange,
            z0 + self.params.purityZScanRange,
            self.params.purityZScanPoints,
        )

        # Save the common starting state
        object0 = self.reconstruction.object.copy()
        probe0 = self.reconstruction.probe.copy()

        objectMomentum0 = self.reconstruction.objectMomentum.copy()
        probeMomentum0 = self.reconstruction.probeMomentum.copy()

        objectBuffer0 = self.reconstruction.objectBuffer.copy()
        probeBuffer0 = self.reconstruction.probeBuffer.copy()

        purity_values = []

        best_purity = -np.inf
        best_z = z0
        best_state = None

        # Do not allow orthogonalization during the internal mPIE iterations.
        # Purity is evaluated once after each candidate reconstruction.
        orthogonalization_switch = self.params.orthogonalizationSwitch
        self.params.orthogonalizationSwitch = False

        tqdm.tqdm.write("")
        tqdm.tqdm.write("Starting purity-based z scan")
        tqdm.tqdm.write(
            f"Initial z = {z0 * 1e3:.6f} mm | "
            f"range = ±{self.params.purityZScanRange * 1e6:.1f} µm | "
            f"points = {len(z_values)} | "
            f"iterations/z = {self.numIterations}"
        )
        tqdm.tqdm.write("")

        try:
            for z_index, z in enumerate(z_values):

                self._resetCandidateState(z)
                #np.random.seed(0)
                

                dz_um = (z - z0) * 1e6

                tqdm.tqdm.write(
                    f"[{z_index + 1}/{len(z_values)}] "
                    f"Reconstructing z = {z * 1e3:.6f} mm "
                    f"(Δz = {dz_um:+.2f} µm)"
                )

                # Run mPIE reconstruction at this z
                for loop in range(self.numIterations):
                    self._run_single_iteration(loop)
                    self.showReconstruction(loop)

                # Orthogonalize once and evaluate purity
                self.orthogonalization()
                

                purity = float(
                    np.asarray(
                        asNumpyArray(self.reconstruction.purityProbe)
                    ).squeeze()
                )
                self.showReconstruction(0)

                purity_values.append(purity)

                tqdm.tqdm.write(
                    f"    Probe purity = {purity:.6f}"
                )

                if purity > best_purity:
                    best_purity = purity
                    best_z = z

                    best_state = {
                        "object": self.reconstruction.object.copy(),
                        "probe": self.reconstruction.probe.copy(),
                        "objectMomentum": self.reconstruction.objectMomentum.copy(),
                        "probeMomentum": self.reconstruction.probeMomentum.copy(),
                        "objectBuffer": self.reconstruction.objectBuffer.copy(),
                        "probeBuffer": self.reconstruction.probeBuffer.copy(),
                    }

                    tqdm.tqdm.write(
                        f"    NEW BEST: z = {best_z * 1e3:.6f} mm, "
                        f"purity = {best_purity:.6f}"
                    )

                tqdm.tqdm.write("")

        finally:
                self.params.orthogonalizationSwitch = orthogonalization_switch

        if best_state is None:
            raise RuntimeError(
                "Purity z scan did not produce a valid purity value."
            )

        # Restore best-z reconstruction state
        self.reconstruction.object = best_state["object"]
        self.reconstruction.probe = best_state["probe"]

        self.reconstruction.objectMomentum = best_state["objectMomentum"]
        self.reconstruction.probeMomentum = best_state["probeMomentum"]

        self.reconstruction.objectBuffer = best_state["objectBuffer"]
        self.reconstruction.probeBuffer = best_state["probeBuffer"]

        self.reconstruction.zo = best_z

        # Store scan results
        self.purityZValues = z_values
        self.purityZMetrics = np.asarray(purity_values)
        self.purityBestZ = best_z
        self.purityBestValue = best_purity

        # Final summary
        tqdm.tqdm.write("=" * 60)
        tqdm.tqdm.write("Purity-based z scan finished")
        tqdm.tqdm.write(
            f"Best z      = {best_z * 1e3:.6f} mm"
        )
        tqdm.tqdm.write(
            f"Best Δz     = {(best_z - z0) * 1e6:+.2f} µm"
        )
        tqdm.tqdm.write(
            f"Best purity = {best_purity:.6f}"
        )
        tqdm.tqdm.write("=" * 60)

        return best_z, best_purity

    def _resetCandidateState(self, z):
        """
        Reset the reconstruction state for one candidate z.

        This implementation is intended for analytic initialization:
        initialProbe = "circ"
        initialObject = "ones"

        The object/probe are regenerated after setting the candidate z so that
        their physical sampling is consistent with the candidate geometry.
        Momentum and buffers are also reset.
        """
        if self.reconstruction.initialProbe != "circ":
            raise NotImplementedError(
                "Candidate-z reinitialization is currently implemented "
                "only for initialProbe='circ'."
            )

        if self.reconstruction.initialObject != "ones":
            raise NotImplementedError(
                "Candidate-z reinitialization is currently implemented "
                "only for initialObject='ones'."
            )

        # Set candidate geometry first
        self.reconstruction.zo = z

        # Rebuild object/probe on the candidate-z geometry
        self.reconstruction.initializeObjectProbe()

        self._initialProbePowerCorrection()

        # Reset momentum
        self.reconstruction.initializeObjectMomentum()
        self.reconstruction.initializeProbeMomentum()

        # Reset buffers
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()

        self._prepareReconstruction()

    def plotPurityZScan(self, show=True):
        """
        Plot probe purity as a function of axial distance.

        The axial coordinate is shown relative to the initial z guess used
        for the purity scan.

        Args:
            show (bool, optional):
                Display the figure immediately. Default is `True`.

        Returns:
            tuple:
                Matplotlib `(figure, axis)` objects.
        """

        if not hasattr(self, "purityZValues"):
            raise RuntimeError(
                "No purity z-scan results are available. "
                "Run purityZScan() first."
            )

        if not hasattr(self, "purityZInitialGuess"):
            raise RuntimeError(
                "Initial z guess is not available."
            )

        z_relative_um = (
            np.asarray(self.purityZValues)
            - self.purityZInitialGuess
        ) * 1e6

        purity = np.asarray(self.purityZMetrics)

        best_index = np.argmax(purity)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            z_relative_um,
            purity,
            marker="o",
            label="Probe purity",
        )

        # Initial z guess
        ax.axvline(
            0,
            linestyle="--",
            alpha=0.5,
            label="Initial z guess",
        )

        # Best z
        ax.plot(
            z_relative_um[best_index],
            purity[best_index],
            marker="*",
            markersize=12,
            linestyle="None",
            label=(
                f"Best Δz = {z_relative_um[best_index]:.1f} µm"
            ),
        )

        ax.set_xlabel("Axial offset from initial z (µm)")
        ax.set_ylabel("Probe purity")
        ax.set_title("Purity-based axial calibration")

        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax
    
    def _run_single_iteration(self, loop):
        """
        Run one complete mPIE reconstruction iteration.
        """

        # set position order
        self.setPositionOrder()

        self.pbar_pos = tqdm.tqdm(
            self.positionIndices,
            leave=False,
            desc="ptychogram",
            file=sys.stdout,
        )

        for positionLoop, positionIndex in enumerate(self.pbar_pos):

            row, col = self.reconstruction.positions[positionIndex]
            sy = slice(row, row + self.reconstruction.Np)
            sx = slice(col, col + self.reconstruction.Np)

            objectPatch = self.reconstruction.object[..., sy, sx].copy()

            # exit surface wave
            self.reconstruction.esw = (
                objectPatch * self.reconstruction.probe
            )

            # intensity constraint
            self.intensityProjection(positionIndex)

            DELTA = (
                self.reconstruction.eswUpdate
                - self.reconstruction.esw
            )

            # object update
            if (
                self.params.objectTVregSwitch
                and loop % self.params.objectTVfreq == 0
            ):
                object_patch = self.objectPatchUpdate_TV(
                    objectPatch,
                    DELTA,
                )
            else:
                object_patch = self.objectPatchUpdate(
                    objectPatch,
                    DELTA,
                )

            self.reconstruction.object[..., sy, sx] = object_patch

            if self.keepPatches:
                self.patches[positionIndex, ..., sy, sx] = (
                    asNumpyArray(abs(object_patch) ** 2)
                )

            # probe update
            weight = 1
            if self.params.weigh_probe_updates_by_intensity:
                weight = self.experimentalData.relative_intensity(
                    positionIndex
                )

            self.reconstruction.probe = self.probeUpdate(
                objectPatch,
                DELTA,
                weight,
            )

            # position correction
            if self.params.positionCorrectionSwitch:
                self.positionCorrection(
                    objectPatch,
                    positionIndex,
                    sy,
                    sx,
                )

            # momentum updates
            if np.random.rand(1) > 0.95:
                self.objectMomentumUpdate()
                self.probeMomentumUpdate()

        self.getErrorMetrics()
        self.applyConstraints(loop)

    def objectMomentumUpdate(self):
        r"""
        Apply the mPIE momentum update to the reconstructed object.

        The change in the object since the previous momentum update is estimated
        from the stored object buffer:

        $$
        G_O^{(n)} = O_{\mathrm{buf}}^{(n)} - O^{(n)}
        $$

        The object momentum is updated according to

        $$
        M_O^{(n)} = G_O^{(n)} + \eta M_O^{(n-1)}
        $$

        where $\eta$ is `frictionM`.

        The accumulated momentum is then fed back into the object estimate:

        $$
        O^{(n+1)} = O^{(n)} - \gamma M_O^{(n)}
        $$

        where $\gamma$ is `feedbackM`.

        After the momentum correction, `objectBuffer` is updated with the current
        object estimate for the next momentum step.

        Notes:
            This update is triggered stochastically from `reconstruct()` rather
            than after every scan-position update.
        """
        gradient = self.reconstruction.objectBuffer - self.reconstruction.object
        self.reconstruction.objectMomentum = (
            gradient + self.frictionM * self.reconstruction.objectMomentum
        )
        self.reconstruction.object = (
            self.reconstruction.object
            - self.feedbackM * self.reconstruction.objectMomentum
        )
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()

    def probeMomentumUpdate(self):
        r"""
        Apply the mPIE momentum update to the reconstructed probe. Similar to `objectMomentumUpdate()`.

        See Also:
            `objectMomentumUpdate`
                Equivalent momentum update applied to the reconstructed object.
        """
        gradient = self.reconstruction.probeBuffer - self.reconstruction.probe
        self.reconstruction.probeMomentum = (
            gradient + self.frictionM * self.reconstruction.probeMomentum
        )
        self.reconstruction.probe = (
            self.reconstruction.probe
            - self.feedbackM * self.reconstruction.probeMomentum
        )
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        r"""
        Update the object patch using the regularized mPIE/rPIE object-update rule.

        The probe intensity is first evaluated and its maximum value is used as a
        global normalization scale:

        $$
        P_{\max} = \max_{x,y}\sum |P(x,y)|^2
        $$

        For conventional ptychography, the probe weighting is

        $$
        W_P =\frac{P^*}{\alpha_O P_{\max} + (1-\alpha_O)|P|^2}
        $$

        and the object patch is updated according to

        $$
        O'_j =O_j + \beta_O \sum W_P\Delta\Psi_j
        $$

        where $\Delta\Psi_j$ is the exit-wave correction, $\beta_O$ is
        `betaObject`, and $\alpha_O$ is `alphaObject`.

        The parameter `alphaObject` controls the balance between global
        normalization by the maximum probe intensity and local normalization by
        the spatially varying probe intensity.

        For Fourier ptychography (`operationMode == "FPM"`), an additional
        probe-amplitude weighting is applied:

        $$
        W_P^{\mathrm{FPM}} =\frac{|P|}{P_{\max}}\frac{P^*}{\alpha_O P_{\max} + (1-\alpha_O)|P|^2}
        $$

        In the multidimensional PtyLab representation, the object correction is
        summed over the probe-mode axis before being added to the current object
        patch.

        Args:
            objectPatch (ndarray):
                Current object patch at the active scan position.
            DELTA (ndarray):
                Exit-wave correction
                `reconstruction.eswUpdate - reconstruction.esw`.

        Returns:
            ndarray:
                Updated object patch.
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        absP2 = xp.abs(self.reconstruction.probe) ** 2
        Pmax = xp.max(xp.sum(absP2, axis=(0, 1, 2, 3)), axis=(-1, -2))
        if self.experimentalData.operationMode == "FPM":
            frac = (
                abs(self.reconstruction.probe)
                / Pmax
                * self.reconstruction.probe.conj()
                / (self.alphaObject * Pmax + (1 - self.alphaObject) * absP2)
            )
        else:
            frac = self.reconstruction.probe.conj() / (
                self.alphaObject * Pmax + (1 - self.alphaObject) * absP2
            )

        return objectPatch + self.betaObject * xp.sum(
            frac * DELTA, axis=2, keepdims=True
        )

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray, weight: float):
        r"""
        Update the probe using the regularized mPIE/rPIE probe-update rule.

        The object intensity is first evaluated and its maximum value is used as
        a global normalization scale:

        $$
        O_{\max} = \max_{x,y}\sum |O_j(x,y)|^2
        $$

        The object weighting is then calculated as

        $$
        W_O = \frac{O_j^*}{\alpha_P O_{\max} + (1-\alpha_P)|O_j|^2}
        $$

        and the probe is updated according to

        $$
        P' = P + w\beta_P\sum W_O\Delta\Psi_j
        $$

        where $\Delta\Psi_j$ is the exit-wave correction, $\beta_P$ is
        `betaProbe`, $\alpha_P$ is `alphaProbe`, and $w$ is the optional
        intensity-dependent update weight.

        The parameter `alphaProbe` controls the balance between global
        normalization by the maximum object intensity and local normalization by
        the spatially varying object intensity.

        By default, $w=1$. If `params.weigh_probe_updates_by_intensity` is
        enabled in `reconstruct()`, $w$ is set to the relative intensity of the
        current diffraction frame.

        In the current multidimensional PtyLab representation, the probe
        correction is summed over axis `1` before being added to the current
        probe estimate.

        Args:
            objectPatch (ndarray):
                Current object patch at the active scan position.
            DELTA (ndarray):
                Exit-wave correction
                `reconstruction.eswUpdate - reconstruction.esw`.
            weight (float):
                Multiplicative weight applied to the probe update. Typically `1`,
                or the relative intensity of the current diffraction frame when
                intensity-weighted probe updates are enabled.

        Returns:
            ndarray:
                Updated probe.
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        absO2 = xp.abs(objectPatch) ** 2
        Omax = xp.max(xp.sum(absO2, axis=(0, 1, 2, 3)), axis=(-1, -2))
        frac = objectPatch.conj() / (
            self.alphaProbe * Omax + (1 - self.alphaProbe) * absO2
        )
        r = self.reconstruction.probe + weight * self.betaProbe * xp.sum(
            frac * DELTA, axis=1, keepdims=True
        )
        return r
