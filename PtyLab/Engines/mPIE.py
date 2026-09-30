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


class mPIE(BaseEngine):
    r"""
    Momentum-accelerated ptychographic iterative engine (mPIE).

    mPIE extends the standard ePIE reconstruction by combining
    regularized PIE (rPIE) object/probe updates with momentum
    acceleration.[^maiden2017]

    As in ePIE, the exit-wave correction at scan position $j$ is

    $$
    \Delta\Psi_j = \Psi'_j - \Psi_j
    $$

    For conventional ptychography, the object update is regularized as

    $$
    O'_j = O_j + \beta_O \frac{P^*}{\alpha_O P_{\max} + (1-\alpha_O)|P|^2}\Delta\Psi_j
    $$

    where $P_{\max}=\max |P|^2$, $\beta_O$ controls the object update
    step size, and $\alpha_O$ controls the spatial regularization.

    The probe is updated analogously:

    $$
    P' = P + \beta_P \frac{O_j^*}{\alpha_P O_{\max} + (1-\alpha_P)|O_j|^2}\Delta\Psi_j
    $$

    where $O_{\max}=\max |O_j|^2$, and $\alpha_P$ and $\beta_P$ control
    probe regularization and update strength.

    Compared with ePIE, these denominators retain stronger updates in
    moderately illuminated regions while suppressing unstable updates where
    the corresponding probe or object intensity is small.

    Momentum acceleration is periodically applied to both object and probe.
    In the PtyLab implementation, the momentum state is updated as

    $$
    M^{(n)} = G^{(n)} + \eta M^{(n-1)}
    $$

    followed by

    $$
    X^{(n+1)} = X^{(n)} - \gamma M^{(n)}
    $$

    where $X$ denotes the object or probe, $\eta$ is `frictionM`, and
    $\gamma$ is `feedbackM`. The current implementation applies these
    momentum updates stochastically during the scan-position loop.

    The default mPIE parameters are `betaObject = 0.25`,
    `betaProbe = 0.25`, `alphaObject = 0.1`, `alphaProbe = 0.1`,
    `feedbackM = 0.3`, and `frictionM = 0.7`.

    [^maiden2017]: A. M. Maiden, D. Johnson, and P. Li,
            "Further improvements to the ptychographical iterative engine,"
            Optica 4, 736-745 (2017).
            https://doi.org/10.1364/OPTICA.4.000736

    Attributes:
        keepPatches (bool):
            If enabled, store the reconstructed object patch associated with each
            scan position for debugging or detailed analysis. This option can
            require substantial additional memory.
    See Also:
        `ePIE`
            Baseline ePIE reconstruction without rPIE regularization or
            momentum acceleration.
    """
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
        self.logger = logging.getLogger("mPIE")
        self.logger.info("Successfully created mPIE engine")
        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        # initialize mPIE Params
        self.initializeReconstructionParams()
        self.params.momentumAcceleration = True
        self.name = "mPIE"

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

    def reconstruct(self, experimentalData=None, reconstruction=None, vis_after_each_iteration=None):
        r"""
        Run the mPIE reconstruction to completion.

        The reconstruction follows the standard ptychographic position loop.
        At each scan position $j$, the current object patch and probe form the
        exit surface wave

        $$
        \Psi_j = O_j P
        $$

        which is propagated to the detector plane and constrained by the measured
        diffraction intensity through `intensityProjection()`. The corresponding
        exit-wave correction is

        $$
        \Delta\Psi_j = \Psi'_j - \Psi_j
        $$

        The object and probe are then updated using the mPIE/rPIE update rules
        implemented by `objectPatchUpdate()` and `probeUpdate()`.

        If `params.objectTVregSwitch` is enabled, the TV-regularized object update
        `objectPatchUpdate_TV()` is used every `params.objectTVfreq` iterations.
        The native mPIE object update is retained and an additional TV
        regularization term is added with strength controlled by
        `params.objectTVregStepSize`.

        If `params.weigh_probe_updates_by_intensity` is enabled, the probe update
        is scaled by the relative intensity of the current diffraction frame.

        If `params.positionCorrectionSwitch` is enabled, scan-position correction
        is applied after the object and probe updates.

        Momentum acceleration is applied stochastically during the scan-position
        loop. In the current implementation, each position update has
        approximately a 5% probability of triggering `objectMomentumUpdate()` and
        `probeMomentumUpdate()`.

        If `keepPatches` is enabled, the reconstructed patch associated with each
        scan position is additionally stored in `self.patches` for diagnostics or
        further analysis.

        After all scan positions in an iteration have been processed,
        `getErrorMetrics()` evaluates the reconstruction error,
        `applyConstraints()` applies the enabled reconstruction constraints, and
        `showReconstruction()` updates the reconstruction monitor.

        Args:
            experimentalData (ExperimentalData, optional):
                Experimental dataset to use for the reconstruction. If provided,
                it replaces the currently attached experimental data.
            reconstruction (Reconstruction, optional):
                Reconstruction state to optimize. If provided, it replaces the
                currently attached reconstruction object.
            vis_after_each_iteration (callable, optional):
                Callback executed after each reconstruction iteration as
                `vis_after_each_iteration(loop, reconstruction)`.

        Notes:
            The object and probe momentum buffers are reset after
            `_prepareReconstruction()` so that they remain synchronized with any
            initialization changes applied before reconstruction starts.
        """
        self.changeExperimentalData(experimentalData)
        self.changeOptimizable(reconstruction)

        self._prepareReconstruction()
        # set object and probe buffers, in case object and probe are changed in the _prepareReconstruction() step
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()
        # actual reconstruction MPIE_engine
        self.pbar = tqdm.trange(
            self.numIterations, desc="mPIE", file=sys.stdout, leave=True
        )
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()
            self.pbar_pos = tqdm.tqdm(
                self.positionIndices, leave=False, desc="ptychogram", file=sys.stdout
            )
            for positionLoop, positionIndex in enumerate(self.pbar_pos):
                # get object patch, stored as self.probe
                # self.reconstruction.make_probe(positionIndex)

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
                # self.viewer.layers['update'].data[positionIndex] = abs(DELTA ** 2).get()
                # import pyqtgraph as pg
                # pg.QtGui.QGuiApplication.processEvents()

                # object update
                if (
                    self.params.objectTVregSwitch
                    and loop % self.params.objectTVfreq == 0
                ):
                    object_patch = self.objectPatchUpdate_TV(objectPatch, DELTA)
                else:
                    object_patch = self.objectPatchUpdate(objectPatch, DELTA)

                self.reconstruction.object[..., sy, sx] = object_patch
                if self.keepPatches:
                    self.patches[positionIndex, ..., sy, sx] = asNumpyArray(
                        abs(object_patch) ** 2
                    )

                # probe update
                weight = 1
                if self.params.weigh_probe_updates_by_intensity:
                    weight = self.experimentalData.relative_intensity(positionIndex)
                    # print(f'for position {positionIndex}, using weight {weight}')

                self.reconstruction.probe = self.probeUpdate(objectPatch, DELTA, weight)
                # self.reconstruction.push_probe_update(self.reconstruction.probe, positionIndex, self.experimentalData.ptychogram.shape[0])

                if self.params.positionCorrectionSwitch:
                    self.positionCorrection(
                        objectPatch, positionIndex, sy, sx
                    )
                    # self.pbar_pos.write(f'Corr: {shifter[0]*1e6:.2f} um x {shifter[1]*1e6:.2f} um')

                # momentum updates
                if np.random.rand(1) > 0.95:
                    self.objectMomentumUpdate()
                    self.probeMomentumUpdate()
                # yield positionLoop, positionIndex

            # get error metric
            self.getErrorMetrics()
            # yield 1,1

            # apply Constraints
            self.applyConstraints(loop)
            # yield 1, 1

            # show reconstruction
            self.showReconstruction(loop)

            if callable(vis_after_each_iteration):
                vis_after_each_iteration(loop, self.reconstruction)

        if self.params.gpuFlag:
            self.logger.info("switch to cpu")
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

            # todo clearMemory implementation

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


class pcPIE(mPIE):
    """
    Backward-compatible wrapper for :class:`mPIE`.

    Position correction is now provided by `mPIE` through
    `params.positionCorrectionSwitch`. This class is retained only for
    compatibility with existing code using `Engines.pcPIE`.
    """

    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        super().__init__(
            reconstruction,
            experimentalData,
            params,
            monitor,
        )

        self.name = "pcPIE"
        self.logger = logging.getLogger("pcPIE")

    @property
    def betaM(self):
        """Deprecated alias for `feedbackM`."""
        return self.feedbackM

    @betaM.setter
    def betaM(self, value):
        self.feedbackM = value

    @property
    def stepM(self):
        """Deprecated alias for `frictionM`."""
        return self.frictionM

    @stepM.setter
    def stepM(self, value):
        self.frictionM = value