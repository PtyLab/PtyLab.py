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
        Set parameters that are specific to the qNewton settings.
        :return:
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
        """
        NADAM optimizer uses adaptive momentum updates (ADAM) with Nesterov
        momentum acceleration
        :return:
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
        """
        standard momentum update
        :return:
        """
        mt = grad + self.frictionM * mt
        return mt, mt, vt

    def objectMomentumUpdate(self, loop):
        """
        momentum update object, save updated objectMomentum and objectBuffer.
        :return:
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
        """
        momentum update probe, save updated probeMomentum and probeBuffer.
        :return:
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
