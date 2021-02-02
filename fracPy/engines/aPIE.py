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
        # self.betaProbe = 0.25
        # self.betaObject = 0.25
        # self.DoF = self.experimentalData.DoF.copy()
        # self.zPIEgradientStepSize = 100  #gradient step size for axial position correction (typical range [1, 100])
        self.aPIEfriction = 0.7
        self.thetaMomentun = 0
        # set aPIE flag
        self.aPIEflag = True

        if not hasattr(self, 'thetaHisory'):
            self.thetaHistory = []

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
        xp = getArrayModule(self.optimizable.object)


        # preallocate grids
        if self.propagator == 'ASP':
            raise NotImplementedError()
        else:
            n = 2*self.experimentalData.Np

        d = self.thetaSearchRadius*np.fliplr(np.linspace(0, 1, self.numIterations))

        pbar = tqdm.trange(self.numIterations, desc='angle updated: theta = ', leave=True)  # in order to change description to the tqdm progress bar
        for loop in pbar:
            # save theta search history
            self.thetaHistory = [self.thetaHistory, asNumpyArray(self.theta)]
            self.zHistory = [self.zHistory, asNumpyArray((self.experimentalData.zo))]

            # select two angles
            theta = [self.theta, self.theta + d[loop] * (-1 + 2 * np.random.rand(1))] + self.thetaMomentum

            # save object and probe
            probeTemp = self.optimizable.probe.copy()
            objectTemp = self.optimizable.object.copy()

            # probe and object buffer
            probeBuffer = np.zeros_like(self.optimizable.probe, shape = ())
            objectBuffer = np.zeros_like(self.No, self.No, 2, 'like',
                                 self.optimizable.probe) # for polychromatic case this will need to be multimode

            # initialize error
            errorTemp = np.zeros(2, 1)

            for k in range(2):
                self.optimizable.probe = probeTemp
                self.optimizable.object = objectTemp
                # reset ptychogram(transform into estimate coordinates)
                Xq = T_inv(self.experimentalData.Xq, self.experimentalData.Yq, self.experimentalData.zo, theta)
                for l in range(self.numFrames):
                    temp = self.ptychogramUntransformed[l]
                    temp2 = abs(interp2d(self.experimentalData.Xd, self.experimentalData.Yd, temp, kind = 'linear')
                                (Xq, self.experimentalData.Yd))
                    temp2 = np.nan_to_num(temp2,0)
                    temp2[temp2 < 0] = 0
                    self.experimentalData.ptychogram[l] = temp2

                # renormalization(for energy conservation)
                self.ptychogram = self.ptychogram / np.linalg.norm(self.experimentalData.ptychogram) * np.linalg.norm(self.ptychogramUntransformed)

                self.W = np.ones_like(self.experimentalData.Np)
                self.W = abs(interp2d(self.experimentalData.Xd, self.experimentalData.Yd, self.W, kind = 'linear')
                             (Xq, self.experimentalDataYd))
                self.W = np.nan_to_num(0)
                self.W[self.W == 0] = 1e-3

                if self.gpuSwitch:
                    self.experimentalData.ptychogram = xp.array(self.experimentalData.ptychogram)
                    self.W = xp.array(self.W)

                if self.fftshiftSwitch:
                    self.ptychogram = ifftshift(ifftshift(self.ptychogram, 1), 2)
                    self.params.W = ifftshift(ifftshift(self.params.W, 1), 2)

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
                    self.optimizable.esw = objectPatch * self.optimizable.beam

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
                errorTemp[k] = self.error[-1]
                self.error[-1] = []

                # apply Constraints
                self.applyConstraints(loop)

            if errorTemp[2] < (1 - 1e-4) * errorTemp[1]:
                dtheta = theta[2] - theta[1]
                self.theta = theta[2]
                self.optimizable.probe = probeBuffer[2]
                self.optimizable.object = objectBuffer[2]
                self.error.append(errorTemp[2])
            else:
                dtheta = 0
                self.theta = theta[1]
                self.optimizable.probe = probeBuffer[1]
                self.optimizable.object = objectBuffer[1]
                self.error.append(errorTemp[1])

            thetaMomentum = 0.5 * dtheta + self.aPIEfriction * self.thetaMomentum

            # show reconstruction
            if loop == 0:
                figure, ax = plt.subplots(1, 1, num=666, squeeze=True, clear=True, figsize=(5, 5))
                ax.set_title('Estimated angle')
                ax.set_xlabel('iteration')
                ax.set_ylabel('estimated theta [deg]')
                ax.set_xscale('symlog')
                line = plt.plot(0, theta, 'o-')[0]
                plt.tight_layout()
                plt.show(block=False)

            elif np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
                idx = np.linspace(0, np.log10(len(self.thetaHistory)-1), np.minimum(len(self.thetaHistory), 100))
                idx = np.rint(10**idx).astype('int')

                line.set_xdata(idx)
                line.set_ydata(np.array(self.thetaHistory)[idx])
                ax.set_xlim(0, np.max(idx))
                ax.set_ylim(np.min(self.thetaHistory), np.max(self.thetaHistory))

                figure.canvas.draw()
                figure.canvas.flush_events()
            self.showReconstruction(loop)

        self.thetaSearchRadius = d[loop]
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
        r = self.optimizable.probe + self.betaObject * xp.sum(frac * DELTA, axis=(0, 1, 3), keepdims=True)
        return r


class aPIE_GPU(zPIE):
    """
    GPU-based implementation of zPIE
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
        self.optimizable.beam = cp.array(self.optimizable.beam, cp.complex64)
        self.optimizable.object = cp.array(self.optimizable.object, cp.complex64)
        self.optimizable.probeBuffer = cp.array(self.optimizable.probeBuffer, cp.complex64)
        self.optimizable.objectBuffer = cp.array(self.optimizable.objectBuffer, cp.complex64)
        self.optimizable.probeMomentum = cp.array(self.optimizable.probeMomentum, cp.complex64)
        self.optimizable.objectMomentum = cp.array(self.optimizable.objectMomentum, cp.complex64)

        # non-optimizable parameters
        self.experimentalData.ptychogram = cp.array(self.experimentalData.ptychogram, cp.float32)
        # self.experimentalData.probe = cp.array(self.experimentalData.probe, cp.complex64)
        #self.optimizable.Imeasured = cp.array(self.optimizable.Imeasured)

        # ePIE parameters
        self.logger.info('Detector error shape: #s', self.detectorError.shape)
        self.detectorError = cp.array(self.detectorError)

        # proapgators to GPU
        if self.propagator == 'Fresnel':
            self.optimizable.quadraticPhase = cp.array(self.optimizable.quadraticPhase)
        elif self.propagator == 'ASP' or self.propagator == 'polychromeASP':
            self.optimizable.transferFunction = cp.array(self.optimizable.transferFunction)
        elif self.propagator =='scaledASP' or self.propagator == 'scaledPolychromeASP':
            self.optimizable.Q1 = cp.array(self.optimizable.Q1)
            self.optimizable.Q2 = cp.array(self.optimizable.Q2)

def T(x, y, z, theta):
    """
    Coordinate transformation
    """
    r0 = np.sqrt(x**2+y**2+z**2)
    yd = y
    xd = x*np.cos(toDegree(theta))-np.sin(toDegree(theta))*(r0-z)
    return xd, yd

def T_inv(xd, yd, z, theta):
    """
    inverse coordinate transformation
    """
    if theta !=45:
        rootTerm = np.sqrt((z*np.cos(toDegree(theta)))**2 + xd**2+yd**2*np.cos(toDegree(2*theta))-
                           2*xd*z*np.sin(toDegree(theta)))
        x = (xd*np.cos(toDegree(theta)) - z*np.sin(toDegree(theta))*np.cos(toDegree(theta)) +
             np.sin(toDegree(theta))*rootTerm)/np.cos(toDegree(2*theta))
    else:
        x = (xd**2-(yd**2)/2-xd*np.sqrt(2)*z)/(xd*np.sqrt(2)-z)
    return x


def toDegree(theta: Any)-> float:
    return np.pi*theta/180