import numpy as np
from matplotlib import pyplot as plt

# PtyLab imports
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
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.utils.gpuUtils import getArrayModule
from PtyLab.utils.utils import fft2c, ifft2c


class mqNewton(BaseEngine):
    r"""
    Experimental momentum-accelerated quasi-Newton-inspired ptychographic
    reconstruction engine.
    `mqNewton` extends `qNewton` by combining the same locally regularized,
    curvature-like object and probe updates with adaptive momentum
    acceleration.

    The current implementation supports three momentum schemes:
    `"momentum"`, `"ADAM"`, and `"NADAM"`. Object and probe momentum updates
    are applied after each scan-position update. These methods should  be interpreted as adaptive
    momentum variants unless further theoretical validation establishes a more
    specific correspondence.

    The local object and probe updates are inherited conceptually from the
    `qNewton` formulation, while the momentum contribution is applied through
    `objectMomentumUpdate()` and `probeMomentumUpdate()`.

    Notes:
        This engine is retained as an experimental algorithm implementation.
        The qNewton normalization, momentum formulation, and theoretical basis
        should be validated against the original method or reference before
        further development.
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
        self.logger = logging.getLogger("mqNewton")
        self.logger.info("Sucesfully created momentum accelerated qNewton mqNewton")

        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        self.initializeReconstructionParams()
        # initialize momentum
        self.reconstruction.initializeObjectMomentum()
        self.reconstruction.initializeProbeMomentum()
        # set object and probe buffers
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()
        self.params.momentumAcceleration = True

    def initializeReconstructionParams(self):
        """
        Initialize parameters specific to the experimental mqNewton reconstruction.

        This method sets the qNewton-style object and probe update strengths and
        regularization parameters, together with the parameters used by the
        momentum-acceleration scheme.

        `beta1` and `beta2` control the first- and second-moment estimates used by
        the adaptive momentum methods. `betaObject_m` and `betaProbe_m` determine
        the strength of the resulting momentum corrections.

        The momentum strategy is selected through `momentum_method`, with the
        current implementation supporting `"momentum"`, `"ADAM"`, and `"NADAM"`.
        """
        self.betaProbe = 1
        self.betaObject = 1
        self.regObject = 1
        self.regProbe = 1
        self.beta1 = 0.5
        self.beta2 = 0.5
        self.betaProbe_m = 0.25
        self.betaObject_m = 0.25
        self.feedbackM = 0.3  # feedback
        self.frictionM = 0.7  # friction
        self.momentum_method = "ADAM"  # which optimizer to use for momentum updates
        self.numIterations = 50

    def initializeAdaptiveMomentum(self):
        """
        Initialize the selected momentum-acceleration scheme.

        The update function is selected from `momentum`, `ADAM`, or `NADAM`
        according to `momentum_method` and stored in `momentum_engine`.

        For the adaptive `"ADAM"` and `"NADAM"` variants, additional second-moment
        buffers are initialized for both the object and probe momentum terms.
        These buffers are stored in `objectMomentum_v` and `probeMomentum_v`.

        Raises:
            AttributeError:
                If `momentum_method` does not correspond to an implemented
                momentum function.
        """
        self.momentum_engine = getattr(mqNewton, self.momentum_method)
        print("Momentum Engines implemented: momentum, ADAM, NADAM")
        print("Momentum mqNewton used: {}".format(self.momentum_method))
        if self.momentum_method in ["ADAM", "NADAM"]:
            # 2nd order momentum terms
            self.reconstruction.objectMomentum_v = (
                self.reconstruction.objectMomentum.copy()
            )
            self.reconstruction.probeMomentum_v = (
                self.reconstruction.probeMomentum.copy()
            )

    def reconstruct(self, experimentalData: ExperimentalData = None):
        """
        Run the momentum-accelerated qNewton reconstruction.

        The reconstruction follows the standard PIE workflow using the
        qNewton-specific object and probe update rules. After each scan-position
        update, additional object and probe momentum corrections are applied
        through `objectMomentumUpdate()` and `probeMomentumUpdate()`.

        The momentum update scheme is selected by `momentum_method` and may use
        standard momentum, ADAM, or NADAM. If
        `params.positionCorrectionSwitch` is enabled, position correction is also
        applied during the scan loop.

        Args:
            experimentalData (ExperimentalData, optional):
                Experimental dataset to use for the reconstruction. If provided,
                it replaces the dataset currently attached to the engine.
        """
        if experimentalData is not None:
            self.experimentalData = experimentalData
            self.reconstruction.data = experimentalData
        self._prepareReconstruction()
        self.initializeAdaptiveMomentum()

        self.pbar = tqdm.trange(
            self.numIterations, desc="mqNewton", file=sys.stdout, leave=True
        )
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()

            for positionLoop, positionIndex in enumerate(self.positionIndices):
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

                # momentum updates
                self.objectMomentumUpdate(loop)
                self.probeMomentumUpdate(loop)

                if self.params.positionCorrectionSwitch:
                    self.positionCorrection(objectPatch, positionIndex, sy, sx)

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info("switch to cpu")
            self._move_data_to_cpu()
            self.params.gpuFlag = 0
            # todo clearMemory implementation

    def ADAM(self, grad, mt, vt, itr):
        r"""
        Compute an ADAM-inspired adaptive momentum update.

        ADAM (Adaptive Moment Estimation) combines momentum with adaptive step
        scaling. It tracks both the running average of the gradient direction and
        a running estimate of the gradient magnitude, allowing the update direction
        and effective step size to adapt during optimization.

        The first-moment estimate is updated as

        $$
        m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
        $$

        while the second-moment estimate uses the global squared L2 norm of the
        gradient,

        $$
        v_t = \beta_2 v_{t-1} + (1-\beta_2)\|g_t\|_2^2
        $$

        Bias-corrected estimates are then formed as

        $$
        \hat{m}_t = \frac{m_t}{1-\beta_1^t}
        $$

        $$
        \hat{v}_t = \frac{v_t}{1-\beta_2^t}
        $$

        and the normalized momentum update is

        $$
        u_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
        $$

        with $\epsilon = 10^{-8}$.


        Args:
            grad (array-like):
                Current object or probe gradient.

            mt (array-like):
                Previous first-moment estimate.

            vt (array-like):
                Previous second-moment estimate.

            itr (int):
                Current iteration index used for bias correction.

        Returns:
            tuple:
                `(update, mt, vt)`, containing the normalized momentum update and
                the updated first- and second-moment estimates.
        """
        xp = getArrayModule(grad)
        beta1_scale = 1 - self.beta1**itr
        beta2_scale = 1 - self.beta2**itr
        mt = self.beta1 * mt + (1 - self.beta1) * grad
        vt = (
            self.beta2 * vt
            + (1 - self.beta2) * xp.linalg.norm(grad.flatten().squeeze(), 2) ** 2
        )
        m_hat = mt / beta1_scale
        v_hat = vt / beta2_scale
        return m_hat / (v_hat**0.5 + 1e-8), mt, vt

    def NADAM(self, grad, mt, vt, itr):
        r"""
        Compute a NADAM-inspired adaptive momentum update.

        NADAM combines ADAM-style adaptive moment estimation with a
        Nesterov-type momentum correction. Compared with ADAM, the update includes
        an additional contribution from the current gradient, giving the momentum
        step a more anticipatory character.

        The first- and second-moment estimates are updated as in `ADAM()`, using the current gradient `g_t`.
        The NADAM-style update is then computed as

        $$
        u_t =\frac{\beta_1 \hat{m}_t+\frac{1-\beta_1}{1-\beta_1^t}g_t}{\sqrt{\hat{v}_t}+\epsilon}
        $$

        with $\epsilon = 10^{-8}$.

        Args:
            grad (array-like):
                Current object or probe gradient.

            mt (array-like):
                Previous first-moment estimate.

            vt (array-like):
                Previous second-moment estimate.

            itr (int):
                Current iteration index used for bias correction.

        Returns:
            tuple:
                `(update, mt, vt)`, containing the adaptive momentum update and
                the updated first- and second-moment estimates.
        """
        xp = getArrayModule(grad)

        beta1_scale = 1 - self.beta1**itr
        beta2_scale = 1 - self.beta2**itr

        norm_sq = xp.linalg.norm(grad.flatten(), 2) ** 2
        mt = self.beta1 * mt + (1 - self.beta1) * grad
        vt = self.beta2 * vt + (1 - self.beta2) * norm_sq
        m_hat = mt / beta1_scale
        v_hat = vt / beta2_scale
        update = (self.beta1 * m_hat + grad * (1 - self.beta1) / beta1_scale) / (
            v_hat**0.5 + 1e-8
        )
        return update, mt, vt

    def momentum(self, grad, mt, vt, itr):
        r"""
        Compute a standard momentum update.

        The current gradient is combined with the previous momentum according to

        $$
        m_t = g_t + \gamma m_{t-1}
        $$

        where `frictionM` defines the momentum retention factor $\gamma$.

        The resulting momentum is used directly as the update direction. The
        second-moment variable `vt` is not modified and is returned unchanged so
        that this method shares the same calling interface as `ADAM()` and
        `NADAM()`.

        Args:
            grad (array-like):
                Current object or probe gradient.

            mt (array-like):
                Previous momentum estimate.

            vt (array-like):
                Second-moment buffer. It is not used by standard momentum and is
                returned unchanged.

            itr (int):
                Iteration index. It is not used by standard momentum but is
                accepted for a common interface with the adaptive momentum methods.

        Returns:
            tuple:
                `(update, mt, vt)`, where `update` and `mt` are the updated momentum
                term and `vt` is returned unchanged.
        """
        mt = grad + self.frictionM * mt
        return mt, mt, vt

    def objectMomentumUpdate(self, loop):
        r"""
        Apply the selected momentum update to the reconstructed object.

        The momentum gradient is defined from the difference between the
        buffered object state and the current reconstruction,

        $$
        g_t = O_{\mathrm{buffer}} - O_t
        $$

        The selected momentum method (`momentum`, `ADAM`, or `NADAM`) is then
        used to compute the update direction and refresh the corresponding
        momentum buffers.

        The reconstructed object is updated according to

        $$
        O_{t+1} = O_t - \beta_{O,m} u_t
        $$

        where `betaObject_m` controls the strength of the momentum correction
        and $u_t$ is the update returned by the selected momentum method.

        After the update, the object buffer is replaced by the new object state
        for use in the next momentum step.

        Args:
            loop (int):
                Current reconstruction iteration index.
        """
        gradient = self.reconstruction.objectBuffer - self.reconstruction.object
        (
            update,
            self.reconstruction.objectMomentum,
            self.reconstruction.objectMomentum_v,
        ) = self.momentum_engine(
            self,
            gradient,
            self.reconstruction.objectMomentum,
            self.reconstruction.objectMomentum_v,
            loop + 1,
        )

        self.reconstruction.object -= self.betaObject_m * update
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()

    def probeMomentumUpdate(self, loop):
        r"""
        Apply the selected momentum update to the reconstructed probe.

        The momentum gradient is defined from the difference between the
        buffered probe state and the current reconstruction,

        $$
        g_t = P_{\mathrm{buffer}} - P_t
        $$

        The selected momentum method (`momentum`, `ADAM`, or `NADAM`) is used
        to compute the probe update direction and refresh the corresponding
        momentum buffers.

        The probe is updated according to

        $$
        P_{t+1} = P_t - \beta_{P,m} u_t
        $$

        where `betaProbe_m` controls the strength of the momentum correction.

        After the update, the probe buffer is replaced by the new probe state
        for use in the next momentum step.

        Args:
            loop (int):
                Current reconstruction iteration index.
        """
        gradient = self.reconstruction.probeBuffer - self.reconstruction.probe
        (
            update,
            self.reconstruction.probeMomentum,
            self.reconstruction.probeMomentum_v,
        ) = self.momentum_engine(
            self,
            gradient,
            self.reconstruction.probeMomentum,
            self.reconstruction.probeMomentum_v,
            loop + 1,
        )

        self.reconstruction.probe -= self.betaProbe_m * update
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        r"""
        Update the object patch using the qNewton-inspired correction.

        The update is weighted by the current probe according to

        $$
        \frac{|P|}{P_{\max}}\frac{P^*}{|P|^2+\lambda_O}
        $$

        where `regObject` provides the regularization term $\lambda_O$.

        This local qNewton-style update is applied before the additional momentum
        correction performed by `objectMomentumUpdate()`.

        Args:
            objectPatch (np.ndarray):
                Current object patch at the scan position.

            DELTA (np.ndarray):
                Exit-wave correction after the intensity projection.

        Returns:
            np.ndarray:
                Updated object patch.
        """
        xp = getArrayModule(objectPatch)
        Pmax = xp.max(xp.sum(xp.abs(self.reconstruction.probe), axis=(0, 1, 2, 3)))
        frac = (
            xp.abs(self.reconstruction.probe)
            / Pmax
            * self.reconstruction.probe.conj()
            / (xp.abs(self.reconstruction.probe) ** 2 + self.regObject)
        )
        return objectPatch + self.betaObject * xp.sum(
            frac * DELTA, axis=(0, 2, 3), keepdims=True
        )

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        r"""
        Update the probe using the qNewton-inspired correction.

        The update is weighted by the current object patch according to

        $$
        \frac{|O|}{O_{\max}}\frac{O^*}{|O|^2+\lambda_P}
        $$

        where `regProbe` provides the regularization term $\lambda_P$.

        This local qNewton-style update is applied before the additional momentum
        correction performed by `probeMomentumUpdate()`.

        Args:
            objectPatch (np.ndarray):
                Current object patch at the scan position.

            DELTA (np.ndarray):
                Exit-wave correction after the intensity projection.

        Returns:
            np.ndarray:
                Updated probe.
        """
        xp = getArrayModule(objectPatch)
        Omax = xp.max(xp.sum(xp.abs(self.reconstruction.object), axis=(0, 1, 2, 3)))
        frac = (
            xp.abs(objectPatch)
            / Omax
            * objectPatch.conj()
            / (xp.abs(objectPatch) ** 2 + self.regProbe)
        )
        r = self.reconstruction.probe + self.betaProbe * xp.sum(
            frac * DELTA, axis=(0, 1, 3), keepdims=True
        )
        return r
