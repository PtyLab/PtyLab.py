import numpy as np
from matplotlib import pyplot as plt
import tqdm
from typing import Any
from scipy.interpolate import interp2d
from fracPy.utils.visualisation import hsvplot

try:
    import cupy as cp
except ImportError:
    print('Cupy not available, will not be able to run GPU based computation')
    # Still define the name, we'll take care of it later but in this way it's still possible
    # to see that gPIE exists for example.
    cp = None

# fracPy imports
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines.BaseReconstructor import BaseReconstructor
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Params.Params import Params
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.monitors.Monitor import Monitor
from fracPy.operators.operators import aspw
import logging
import sys

class aPIE(BaseReconstructor):
    """
    aPIE: angle correction PIE: ePIE combined with Luus-Jaakola algorithm (the latter for angle correction) + momentum
    """

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, params: Params, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to aPIE reconstruction
        super().__init__(optimizable, experimentalData, params, monitor)
        self.logger = logging.getLogger('aPIE')
        self.logger.info('Sucesfully created aPIE aPIE_engine')
        self.logger.info('Wavelength attribute: %s', self.experimentalData.wavelength)
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

        if not hasattr(self.optimizable, 'thetaMomentum'):
            self.optimizable.thetaMomentum = 0
        if not hasattr(self.optimizable, 'thetaHistory'):
            self.optimizable.thetaHistory = np.array([])

        self.thetaSearchRadiusMin = 0.01
        self.thetaSearchRadiusMax = 0.1
        self.experimentalData.W = np.ones_like(self.experimentalData.Xd)

        if self.optimizable.theta == None:
            raise ValueError('theta value is not given')

    def doReconstruction(self):
        self._prepareReconstruction()

        xp = getArrayModule(self.optimizable.object)

        # linear search
        thetaSearchRadiusList = np.linspace(self.thetaSearchRadiusMax, self.thetaSearchRadiusMin,
                                            self.numIterations)

        self.pbar = tqdm.trange(self.numIterations, desc='aPIE', file=sys.stdout, leave=True)
        for loop in self.pbar:
            # save theta search history
            self.optimizable.thetaHistory = np.append(self.optimizable.thetaHistory, asNumpyArray(self.optimizable.theta))

            # select two angles (todo check if three angles behave better)
            theta = np.array([self.optimizable.theta, self.optimizable.theta + thetaSearchRadiusList[loop] *
                              (-1 + 2 * np.random.rand())]) + self.optimizable.thetaMomentum

            # save object and probe
            probeTemp = self.optimizable.probe.copy()
            objectTemp = self.optimizable.object.copy()

            # probe and object buffer (todo maybe there's more elegant way )
            probeBuffer = xp.zeros_like(probeTemp) # shape=(np.array([probeTemp, probeTemp])).shape)
            probeBuffer = [probeBuffer, probeBuffer]
            objectBuffer = xp.zeros_like(objectTemp) #, shape=(np.array([objectTemp, objectTemp])).shape)  # for polychromatic case this will need to be multimode
            objectBuffer = [objectBuffer,objectBuffer]
            # initialize error
            errorTemp = np.zeros((2, 1))

            for k in range(2):
                self.optimizable.probe = probeTemp
                self.optimizable.object = objectTemp
                # reset ptychogram (transform into estimate coordinates)
                Xq = T_inv(self.experimentalData.Xd, self.experimentalData.Yd, self.experimentalData.zo, theta[k]) # todo check if 1D is enough to save time
                for l in range(self.experimentalData.numFrames):
                    temp = asNumpyArray(self.experimentalData.ptychogramUntransformed[l])
                    f = interp2d(self.experimentalData.xd, self.experimentalData.xd, temp, kind='linear', fill_value=0)
                    temp2 = abs(f(Xq[0], self.experimentalData.xd))
                    temp2 = np.nan_to_num(temp2)
                    temp2[temp2 < 0] = 0
                    self.experimentalData.ptychogram[l] = xp.array(temp2)

                # renormalization(for energy conservation) # todo not layer by layer?
                self.experimentalData.ptychogram = self.experimentalData.ptychogram / np.linalg.norm(
                    self.experimentalData.ptychogram) * np.linalg.norm(self.experimentalData.ptychogramUntransformed)

                self.experimentalData.W = np.ones_like(self.experimentalData.Xd)
                fw = interp2d(self.experimentalData.xd, self.experimentalData.xd, self.experimentalData.W, kind='linear', fill_value=0)
                self.experimentalData.W = abs(fw(Xq[0], self.experimentalData.xd))
                self.experimentalData.W = np.nan_to_num(self.experimentalData.W)
                self.experimentalData.W[self.experimentalData.W == 0] = 1e-3
                self.experimentalData.W = xp.array(self.experimentalData.W)


                # todo check if it is right
                if self.params.fftshiftSwitch:
                    self.experimentalData.ptychogram = xp.fft.ifftshift(self.experimentalData.ptychogram, axes=(-1, -2))
                    self.experimentalData.W = xp.fft.ifftshift(self.experimentalData.W, axes=(-1, -2))

                # set position order
                self.setPositionOrder()

                for positionLoop, positionIndex in enumerate(self.positionIndices):
                    ### patch1 ###
                    # get object patch1
                    row1, col1 = self.experimentalData.positions[positionIndex]
                    sy = slice(row1, row1 + self.experimentalData.Np)
                    sx = slice(col1, col1 + self.experimentalData.Np)
                    # note that object patch has size of probe array
                    objectPatch = self.optimizable.object[..., sy, sx].copy()

                    # make exit surface wave
                    self.optimizable.esw = objectPatch * self.optimizable.probe

                    # propagate to camera, intensityProjection, propagate back to object
                    self.intensityProjection(positionIndex)

                    # difference term1
                    DELTA = self.optimizable.eswUpdate - self.optimizable.esw

                    # object update
                    self.optimizable.object[..., sy, sx] = self.objectPatchUpdate(objectPatch, DELTA)

                    # probe update
                    self.optimizable.probe = self.probeUpdate(objectPatch, DELTA)


                # get error metric
                self.getErrorMetrics()
                # remove error from error history
                errorTemp[k] = self.optimizable.error[-1]
                self.optimizable.error = np.delete(self.optimizable.error, -1)

                # apply Constraints
                self.applyConstraints(loop)
                # update buffer
                probeBuffer[k] = self.optimizable.probe
                objectBuffer[k] = self.optimizable.object

            if errorTemp[1] < errorTemp[0]:
                dtheta = theta[1] - theta[0]
                self.optimizable.theta = theta[1]
                self.optimizable.probe = probeBuffer[1]
                self.optimizable.object = objectBuffer[1]
                self.optimizable.error = np.append(self.optimizable.error, errorTemp[1])
            else:
                dtheta = 0
                self.optimizable.theta = theta[0]
                self.optimizable.probe = probeBuffer[0]
                self.optimizable.object = objectBuffer[0]
                self.optimizable.error = np.append(self.optimizable.error, errorTemp[0])

            self.optimizable.thetaMomentum = self.feedback * dtheta + self.aPIEfriction * self.optimizable.thetaMomentum
            # print updated theta
            self.pbar.set_description('aPIE: update a=%.3f deg (search radius=%.3f deg, thetaMomentum=%.3f deg)'
                                      % (self.optimizable.theta, thetaSearchRadiusList[loop], self.optimizable.thetaMomentum))

            # show reconstruction
            if loop == 0:
                figure, ax = plt.subplots(1, 1, num=777, squeeze=True, clear=True, figsize=(5, 5))
                ax.set_title('Estimated angle')
                ax.set_xlabel('iteration')
                ax.set_ylabel('estimated theta [deg]')
                ax.set_xscale('symlog')
                line = plt.plot(0, self.optimizable.theta, 'o-')[0]
                plt.tight_layout()
                plt.show(block=False)

            elif np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
                idx = np.linspace(0, np.log10(len(self.optimizable.thetaHistory) - 1),
                                  np.minimum(len(self.optimizable.thetaHistory), 100))
                idx = np.rint(10 ** idx).astype('int')

                line.set_xdata(idx)
                line.set_ydata(np.array(self.optimizable.thetaHistory)[idx])
                ax.set_xlim(0, np.max(idx))
                ax.set_ylim(min(self.optimizable.thetaHistory), max(self.optimizable.thetaHistory))

                figure.canvas.draw()
                figure.canvas.flush_events()

            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info('switch to cpu')
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

        frac = self.optimizable.probe.conj() / xp.max(xp.sum(xp.abs(self.optimizable.probe) ** 2, axis=(0, 1, 2, 3)))
        return objectPatch + self.betaObject * xp.sum(frac * DELTA, axis=(0, 2, 3), keepdims=True)

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = objectPatch.conj() / xp.max(xp.sum(xp.abs(objectPatch) ** 2, axis=(0, 1, 2, 3)))
        r = self.optimizable.probe + self.betaProbe * xp.sum(frac * DELTA, axis=(0, 1, 3), keepdims=True)
        return r


def T(x, y, z, theta):
    """
    Coordinate transformation
    """
    r0 = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    yd = y
    xd = x * np.cos(toDegree(theta)) - np.sin(toDegree(theta)) * (r0 - z)
    return xd, yd


def T_inv(xd, yd, z, theta):
    """
    inverse coordinate transformation
    """
    if theta != 45:
        rootTerm = np.sqrt((z * np.cos(toDegree(theta))) ** 2 + xd ** 2 + yd ** 2 * np.cos(toDegree(2 * theta)) -
                           2 * xd * z * np.sin(toDegree(theta)))
        x = (xd * np.cos(toDegree(theta)) - z * np.sin(toDegree(theta)) * np.cos(toDegree(theta)) +
             np.sin(toDegree(theta)) * rootTerm) / np.cos(toDegree(2 * theta))
    else:
        x = (xd ** 2 - (yd ** 2) / 2 - xd * np.sqrt(2) * z) / (xd * np.sqrt(2) - z)
    return x


def toDegree(theta: Any) -> float:
    return np.pi * theta / 180
