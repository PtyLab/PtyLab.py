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
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.monitors.Monitor import Monitor
from fracPy.operators.operators import aspw
import logging


class aPIE(BaseReconstructor):
    """
    aPIE: angle correction PIE: ePIE combined with Luus-Jaakola algorithm (the latter for angle correction) + momentum
    """

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to aPIE reconstruction
        super().__init__(optimizable, experimentalData, monitor)
        self.logger = logging.getLogger('aPIE')
        self.logger.info('Sucesfully created aPIE aPIE_engine')
        self.logger.info('Wavelength attribute: #s', self.optimizable.wavelength)
        self.initializeReconstructionParams()
        # # initialize momentum
        # self.optimizable.initializeObjectMomentum()
        # self.optimizable.initializeProbeMomentum()
        # # set object and probe buffers
        # self.optimizable.objectBuffer = self.optimizable.object.copy()
        # self.optimizable.probeBuffer = self.optimizable.probe.copy()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.aPIEfriction = 0.7
        self.thetaMomentum = 0
        self.feedback = 0.5

        if not hasattr(self, 'thetaHistory'):
            self.thetaHistory = np.array([])

        self.ptychogramUntransformed = self.experimentalData.ptychogram.copy()
        self.thetaSearchRadiusMin = 0
        self.thetaSearchRadiusMax = 0.1
        self.W = np.ones_like(self.experimentalData.Xd)

        # self.experimentalData.Xq = self.experimentalData.Xd.copy()
        # self.experimentalData.Yq =


    def _prepare_doReconstruction(self):
        """
        This function is called just before the reconstructions start.

        Can be used to (for instance) transfer data to the GPU at the last moment.
        :return:
        """
        pass

    def doReconstruction(self):
        self._initializeParams()
        self._prepare_doReconstruction()
        if not hasattr(self, 'theta'):
            raise ValueError('theta value is not given')
        xp = getArrayModule(self.optimizable.object)

        # linear search
        thetaSearchRadius = np.flipud(np.linspace(self.thetaSearchRadiusMin, self.thetaSearchRadiusMax, self.numIterations))

        self.pbar = tqdm.trange(self.numIterations, desc='angle updated: theta = ',
                           leave=True)  # in order to change description to the tqdm progress bar
        for loop in self.pbar:
            # save theta search history
            self.thetaHistory = np.append(self.thetaHistory, asNumpyArray(self.theta))

            # select two angles (todo check if three angles behave better)
            theta = np.squeeze(np.array([self.theta, self.theta + thetaSearchRadius[loop] * (-1 + 2 * np.random.rand(1,1))] ) + self.thetaMomentum)

            # save object and probe
            probeTemp = self.optimizable.probe.copy()
            objectTemp = self.optimizable.object.copy()

            # probe and object buffer
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
                Xq = T_inv(self.experimentalData.Xd, self.experimentalData.Yd, self.experimentalData.zo, theta[k])
                for l in range(self.experimentalData.numFrames):
                    temp = self.ptychogramUntransformed[l].get()
                    f = interp2d(self.experimentalData.xd, self.experimentalData.xd, temp, kind='linear', fill_value=0)
                    temp2 = abs(f(Xq[0], self.experimentalData.xd))
                    temp2 = np.nan_to_num(temp2)
                    temp2[temp2 < 0] = 0
                    self.experimentalData.ptychogram[l] = cp.array(temp2)

                # renormalization(for energy conservation)
                self.experimentalData.ptychogram = self.experimentalData.ptychogram / np.linalg.norm(
                    self.experimentalData.ptychogram) * np.linalg.norm(self.ptychogramUntransformed)

                 # todo check how interp2d with edge values
                fw = interp2d(self.experimentalData.xd, self.experimentalData.xd, self.W, kind='linear', fill_value=0)
                self.W = abs(fw(Xq[0], self.experimentalData.xd))
                self.W = np.nan_to_num(self.W)
                self.W[self.W == 0] = 1e-3

                # todo check if it is necessary
                # if self.gpuSwitch:
                #     self.experimentalData.ptychogram = xp.array(self.experimentalData.ptychogram)
                #     self.W = xp.array(self.W)
                #
                # if self.fftshiftSwitch:
                #     self.ptychogram = xp.ifftshift(ifftshift(self.ptychogram, 1), 2)
                #     self.params.W = xp.ifftshift(ifftshift(self.params.W, 1), 2)

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

                # update buffer
                probeBuffer[k] = self.optimizable.probe
                objectBuffer[k] = self.optimizable.object
                # get error metric
                self.getErrorMetrics()
                # remove error from error history
                errorTemp[k] = self.optimizable.error[-1]
                self.optimizable.error = np.delete(self.optimizable.error, -1)

                # apply Constraints
                self.applyConstraints(loop)

            if errorTemp[1] < (1 - 1e-4) * errorTemp[0]:
                dtheta = theta[1] - theta[0]
                self.theta = theta[1]
                self.optimizable.probe = probeBuffer[1]
                self.optimizable.object = objectBuffer[1]
                self.optimizable.error = np.append(self.optimizable.error, errorTemp[1])
            else:
                dtheta = 0
                self.theta = theta[0]
                self.optimizable.probe = probeBuffer[0]
                self.optimizable.object = objectBuffer[0]
                self.optimizable.error = np.append(self.optimizable.error, errorTemp[0])

            thetaMomentum = self.feedback * dtheta + self.aPIEfriction * self.thetaMomentum

            # show reconstruction
            if loop == 0:
                figure, ax = plt.subplots(1, 1, num=666, squeeze=True, clear=True, figsize=(5, 5))
                ax.set_title('Estimated angle')
                ax.set_xlabel('iteration')
                ax.set_ylabel('estimated theta [deg]')
                ax.set_xscale('symlog')
                line = plt.plot(0, self.theta, 'o-')[0]
                plt.tight_layout()
                plt.show(block=False)

            elif np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
                idx = np.linspace(0, np.log10(len(self.thetaHistory) - 1), np.minimum(len(self.thetaHistory), 100))
                idx = np.rint(10 ** idx).astype('int')

                line.set_xdata(idx)
                line.set_ydata(np.array(self.thetaHistory)[idx])
                ax.set_xlim(0, np.max(idx))
                ax.set_ylim(min(self.thetaHistory), max(self.thetaHistory))

                figure.canvas.draw()
                figure.canvas.flush_events()
            self.showReconstruction(loop)

        self.thetaSearchRadiusMax = thetaSearchRadius[loop]
        self.thetaMomentun = thetaMomentum

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


class aPIE_GPU(aPIE):
    """
    GPU-based implementation of aPIE
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cp is None:
            raise ImportError('Could not import cupy')
        self.logger = logging.getLogger('aPIE_GPU')
        self.logger.info('Hello from aPIE_GPU')

    def _prepare_doReconstruction(self):
        self.logger.info('Ready to start transferring stuff to the GPU')
        self._move_data_to_gpu()

    def _move_data_to_gpu(self):
        """
        Move the data to the GPU
        :return:
        """
        # optimizable parameters
        self.optimizable.probe = cp.array(self.optimizable.probe, cp.complex64)
        self.optimizable.object = cp.array(self.optimizable.object, cp.complex64)
        # self.optimizable.probeBuffer = cp.array(self.optimizable.probeBuffer, cp.complex64)
        # self.optimizable.objectBuffer = cp.array(self.optimizable.objectBuffer, cp.complex64)
        # self.optimizable.probeMomentum = cp.array(self.optimizable.probeMomentum, cp.complex64)
        # self.optimizable.objectMomentum = cp.array(self.optimizable.objectMomentum, cp.complex64)

        # non-optimizable parameters
        self.experimentalData.ptychogram = cp.array(self.experimentalData.ptychogram, cp.float32)
        # self.experimentalData.probe = cp.array(self.experimentalData.probe, cp.complex64)
        # self.optimizable.Imeasured = cp.array(self.optimizable.Imeasured)

        # ePIE parameters
        self.logger.info('Detector error shape: #s', self.detectorError.shape)
        self.detectorError = cp.array(self.detectorError)

        # proapgators to GPU
        if self.propagator == 'Fresnel':
            self.optimizable.quadraticPhase = cp.array(self.optimizable.quadraticPhase)
        elif self.propagator == 'ASP' or self.propagator == 'polychromeASP':
            self.optimizable.transferFunction = cp.array(self.optimizable.transferFunction)
        elif self.propagator == 'scaledASP' or self.propagator == 'scaledPolychromeASP':
            self.optimizable.Q1 = cp.array(self.optimizable.Q1)
            self.optimizable.Q2 = cp.array(self.optimizable.Q2)

        # other parameters
        if self.backgroundModeSwitch:
            self.background = cp.array(self.background)
        if self.absorbingProbeBoundary:
            self.probeWindow = cp.array(self.probeWindow)


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
