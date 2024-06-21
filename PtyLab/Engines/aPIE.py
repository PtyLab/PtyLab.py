from typing import Any

import numpy as np
import tqdm
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d

from PtyLab.utils.visualisation import hsvplot

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


class aPIE(BaseEngine):
    """
    aPIE: angle correction PIE: ePIE combined with Luus-Jaakola algorithm (the latter for angle correction) + momentum
    """

    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to aPIE reconstruction
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger("aPIE")
        self.logger.info("Sucesfully created aPIE aPIE_engine")
        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.aPIEfriction = 0.7
        self.feedback = 0.5
        self.numIterations = 50

        if not hasattr(self.reconstruction, "thetaMomentum"):
            self.reconstruction.thetaMomentum = 0
        if not hasattr(self.reconstruction, "thetaHistory"):
            self.reconstruction.thetaHistory = np.array([])

        self.thetaSearchRadiusMin = 0.01
        self.thetaSearchRadiusMax = 0.1
        self.ptychogramUntransformed = self.experimentalData.ptychogram.copy()
        self.experimentalData.W = np.ones_like(self.reconstruction.Xd)

        if self.reconstruction.theta == None:
            raise ValueError("theta value is not given")

    def doReconstruction(self):
        self._prepareReconstruction()

        xp = getArrayModule(self.reconstruction.object)

        # linear search
        thetaSearchRadiusList = np.linspace(
            self.thetaSearchRadiusMax, self.thetaSearchRadiusMin, self.numIterations
        )

        self.pbar = tqdm.trange(
            self.numIterations, desc="aPIE", file=sys.stdout, leave=True
        )
        for loop in self.pbar:
            # save theta search history
            self.reconstruction.thetaHistory = np.append(
                self.reconstruction.thetaHistory,
                asNumpyArray(self.reconstruction.theta),
            )

            # select two angles (todo check if three angles behave better)
            theta = (
                np.array(
                    [
                        self.reconstruction.theta,
                        self.reconstruction.theta
                        + thetaSearchRadiusList[loop] * (-1 + 2 * np.random.rand()),
                    ]
                )
                + self.reconstruction.thetaMomentum
            )

            # save object and probe
            probeTemp = self.reconstruction.probe.copy()
            objectTemp = self.reconstruction.object.copy()

            # probe and object buffer (todo maybe there's more elegant way )
            probeBuffer = xp.zeros_like(
                probeTemp
            )  # shape=(np.array([probeTemp, probeTemp])).shape)
            probeBuffer = [probeBuffer, probeBuffer]
            objectBuffer = xp.zeros_like(
                objectTemp
            )  # , shape=(np.array([objectTemp, objectTemp])).shape)  # for polychromatic case this will need to be multimode
            objectBuffer = [objectBuffer, objectBuffer]
            # initialize error
            errorTemp = np.zeros((2, 1))

            for k in range(2):
                self.reconstruction.probe = probeTemp
                self.reconstruction.object = objectTemp
                # reset ptychogram (transform into estimate coordinates)
                Xq = T_inv(
                    self.reconstruction.Xd,
                    self.reconstruction.Yd,
                    self.reconstruction.zo,
                    theta[k],
                )  # todo check if 1D is enough to save time
                for l in range(self.experimentalData.numFrames):
                    temp = self.ptychogramUntransformed[l]
                    f = interp2d(
                        self.reconstruction.xd,
                        self.reconstruction.xd,
                        temp,
                        kind="linear",
                        fill_value=0,
                    )
                    temp2 = abs(f(Xq[0], self.reconstruction.xd))
                    temp2 = np.nan_to_num(temp2)
                    temp2[temp2 < 0] = 0
                    self.experimentalData.ptychogram[l] = xp.array(temp2)

                # renormalization(for energy conservation) # todo not layer by layer?
                self.experimentalData.ptychogram = (
                    self.experimentalData.ptychogram
                    / np.linalg.norm(self.experimentalData.ptychogram)
                    * np.linalg.norm(self.ptychogramUntransformed)
                )

                self.experimentalData.W = np.ones_like(self.reconstruction.Xd)
                fw = interp2d(
                    self.reconstruction.xd,
                    self.reconstruction.xd,
                    self.experimentalData.W,
                    kind="linear",
                    fill_value=0,
                )
                self.experimentalData.W = abs(fw(Xq[0], self.reconstruction.xd))
                self.experimentalData.W = np.nan_to_num(self.experimentalData.W)
                self.experimentalData.W[self.experimentalData.W == 0] = 1e-3
                self.experimentalData.W = xp.array(self.experimentalData.W)

                # todo check if it is right
                if self.params.fftshiftSwitch:
                    self.experimentalData.ptychogram = xp.fft.ifftshift(
                        self.experimentalData.ptychogram, axes=(-1, -2)
                    )
                    self.experimentalData.W = xp.fft.ifftshift(
                        self.experimentalData.W, axes=(-1, -2)
                    )

                # set position order
                self.setPositionOrder()

                for positionLoop, positionIndex in enumerate(self.positionIndices):
                    ### patch1 ###
                    # get object patch1
                    row1, col1 = self.reconstruction.positions[positionIndex]
                    sy = slice(row1, row1 + self.reconstruction.Np)
                    sx = slice(col1, col1 + self.reconstruction.Np)
                    # note that object patch has size of probe array
                    objectPatch = self.reconstruction.object[..., sy, sx].copy()

                    # make exit surface wave
                    self.reconstruction.esw = objectPatch * self.reconstruction.probe

                    # propagate to camera, intensityProjection, propagate back to object
                    self.intensityProjection(positionIndex)

                    # difference term1
                    DELTA = self.reconstruction.eswUpdate - self.reconstruction.esw

                    # object update
                    self.reconstruction.object[..., sy, sx] = self.objectPatchUpdate(
                        objectPatch, DELTA
                    )

                    # probe update
                    self.reconstruction.probe = self.probeUpdate(objectPatch, DELTA)

                # get error metric
                self.getErrorMetrics()
                # remove error from error history
                errorTemp[k] = self.reconstruction.error[-1]
                self.reconstruction.error = np.delete(self.reconstruction.error, -1)

                # apply Constraints
                self.applyConstraints(loop)
                # update buffer
                probeBuffer[k] = self.reconstruction.probe
                objectBuffer[k] = self.reconstruction.object

            if errorTemp[1] < errorTemp[0]:
                dtheta = theta[1] - theta[0]
                self.reconstruction.theta = theta[1]
                self.reconstruction.probe = probeBuffer[1]
                self.reconstruction.object = objectBuffer[1]
                self.reconstruction.error = np.append(
                    self.reconstruction.error, errorTemp[1]
                )
            else:
                dtheta = 0
                self.reconstruction.theta = theta[0]
                self.reconstruction.probe = probeBuffer[0]
                self.reconstruction.object = objectBuffer[0]
                self.reconstruction.error = np.append(
                    self.reconstruction.error, errorTemp[0]
                )

            self.reconstruction.thetaMomentum = (
                self.feedback * dtheta
                + self.aPIEfriction * self.reconstruction.thetaMomentum
            )
            # print updated theta
            self.pbar.set_description(
                "aPIE: update a=%.3f deg (search radius=%.3f deg, thetaMomentum=%.3f deg)"
                % (
                    self.reconstruction.theta,
                    thetaSearchRadiusList[loop],
                    self.reconstruction.thetaMomentum,
                )
            )

            # show reconstruction
            if loop == 0:
                figure, ax = plt.subplots(
                    1, 1, num=777, squeeze=True, clear=True, figsize=(5, 5)
                )
                ax.set_title("Estimated angle")
                ax.set_xlabel("iteration")
                ax.set_ylabel("estimated theta [deg]")
                ax.set_xscale("symlog")
                line = plt.plot(0, self.reconstruction.theta, "o-")[0]
                plt.tight_layout()
                plt.show(block=False)

            elif np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
                idx = np.linspace(
                    0,
                    np.log10(len(self.reconstruction.thetaHistory) - 1),
                    np.minimum(len(self.reconstruction.thetaHistory), 100),
                )
                idx = np.rint(10**idx).astype("int")

                line.set_xdata(idx)
                line.set_ydata(np.array(self.reconstruction.thetaHistory)[idx])
                ax.set_xlim(0, np.max(idx))
                ax.set_ylim(
                    min(self.reconstruction.thetaHistory),
                    max(self.reconstruction.thetaHistory),
                )

                figure.canvas.draw()
                figure.canvas.flush_events()

            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info("switch to cpu")
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

        # self.thetaSearchRadiusMax = thetaSearchRadiusList[loop]

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


def T(x, y, z, theta):
    """
    Coordinate transformation
    """
    r0 = np.sqrt(x**2 + y**2 + z**2)
    yd = y
    xd = x * np.cos(toDegree(theta)) - np.sin(toDegree(theta)) * (r0 - z)
    return xd, yd


def T_inv(xd, yd, z, theta):
    """
    inverse coordinate transformation
    """
    if theta != 45:
        rootTerm = np.sqrt(
            (z * np.cos(toDegree(theta))) ** 2
            + xd**2
            + yd**2 * np.cos(toDegree(2 * theta))
            - 2 * xd * z * np.sin(toDegree(theta))
        )
        x = (
            xd * np.cos(toDegree(theta))
            - z * np.sin(toDegree(theta)) * np.cos(toDegree(theta))
            + np.sin(toDegree(theta)) * rootTerm
        ) / np.cos(toDegree(2 * theta))
    else:
        x = (xd**2 - (yd**2) / 2 - xd * np.sqrt(2) * z) / (xd * np.sqrt(2) - z)
    return x


def toDegree(theta: Any) -> float:
    return np.pi * theta / 180
