import numpy as np
from matplotlib import pyplot as plt

try:
    import cupy as cp
except ImportError:
    print('Cupy not available, will not be able to run GPU based computation')
    # Still define the name, we'll take care of it later but in this way it's still possible
    # to see that gPIE exists for example.
    cp = None

# fracPy imports
from fracPy.Optimizables.Optimizable import Optimizable
from fracPy.Engines.BaseReconstructor import BaseReconstructor
from fracPy.FixedData.DefaultExperimentalData import ExperimentalData
from fracPy.Params.ReconstructionParameters import Reconstruction_parameters
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.Monitors.Monitor import Monitor
from fracPy.utils.utils import fft2c, ifft2c
import logging
import tqdm
import sys


class multiPIE(BaseReconstructor):

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, params: Reconstruction_parameters, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(optimizable, experimentalData, params, monitor)
        self.logger = logging.getLogger('multiPIE')
        self.logger.info('Sucesfully created multiPIE multiPIE_engine')
        self.logger.info('Wavelength attribute: %s', self.optimizable.wavelength)
        # initialize multiPIE params
        self.initializeReconstructionParams()
        self.params.momentumAcceleration = True

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the multiPIE settings.
        :return:
        """
        # self.eswUpdate = self.optimizable.esw.copy()
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.alphaProbe = 0.1  # probe regularization
        self.alphaObject = 0.1  # object regularization
        self.betaM = 0.3  # feedback
        self.stepM = 0.7  # friction
        # self.optimizable.probeWindow = np.abs(self.optimizable.probe)

        # initialize momentum
        self.optimizable.initializeObjectMomentum()
        self.optimizable.initializeProbeMomentum()
        # set object and probe buffers
        self.optimizable.objectBuffer = self.optimizable.object.copy()
        self.optimizable.probeBuffer = self.optimizable.probe.copy()

    def doReconstruction(self):
        self._prepareReconstruction()

        self.pbar = tqdm.trange(self.numIterations, desc='multiPIE', file=sys.stdout, leave=True)

        for loop in self.pbar:
            # set position order
            self.setPositionOrder()

            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                row, col = self.optimizable.positions[positionIndex]
                sy = slice(row, row + self.optimizable.Np)
                sx = slice(col, col + self.optimizable.Np)
                # note that object patch has size of probe array
                objectPatch = self.optimizable.object[..., sy, sx].copy()

                # make exit surface wave
                self.optimizable.esw = objectPatch * self.optimizable.probe

                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.optimizable.eswUpdate - self.optimizable.esw

                # object update
                self.optimizable.object[..., sy, sx] = self.objectPatchUpdate(objectPatch, DELTA)

                # probe update
                self.optimizable.probe = self.probeUpdate(objectPatch, DELTA)

                # momentum updates todo: make this every T iteration?
                # Todo @lars explain this
                if np.random.rand(1) > 0.95:
                    self.objectMomentumUpdate()
                    self.probeMomentumUpdate()

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

            # todo clearMemory implementation

        if self.params.gpuFlag:
            self.logger.info('switch to cpu')
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

    def objectMomentumUpdate(self):
        """
        momentum update object, save updated objectMomentum and objectBuffer.
        :return:
        """
        gradient = self.optimizable.objectBuffer - self.optimizable.object
        self.optimizable.objectMomentum = gradient + self.stepM * self.optimizable.objectMomentum
        self.optimizable.object = self.optimizable.object - self.betaM * self.optimizable.objectMomentum
        self.optimizable.objectBuffer = self.optimizable.object.copy()

    def probeMomentumUpdate(self):
        """
        momentum update probe, save updated probeMomentum and probeBuffer.
        :return:
        """
        gradient = self.optimizable.probeBuffer - self.optimizable.probe
        self.optimizable.probeMomentum = gradient + self.stepM * self.optimizable.probeMomentum
        self.optimizable.probe = self.optimizable.probe - self.betaM * self.optimizable.probeMomentum
        self.optimizable.probeBuffer = self.optimizable.probe.copy()

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        # xp = getArrayModule(objectPatch)
        # absP2 = xp.abs(self.optimizable.probe[0]) ** 2
        # Pmax = xp.max(xp.sum(absP2, axis=(0, 1, 2)), axis=(-1, -2))
        # if self.experimentalData.operationMode == 'FPM':
        #     frac = abs(self.optimizable.probe) / Pmax * \
        #            self.optimizable.probe[0].conj() / (self.alphaObject * Pmax + (1 - self.alphaObject) * absP2)
        # else:
        #     frac = self.optimizable.probe[0].conj() / (self.alphaObject * Pmax + (1 - self.alphaObject) * absP2)
        # return objectPatch + self.betaObject * frac * DELTA
        xp = getArrayModule(objectPatch)
        absP2 = xp.abs(self.optimizable.probe)**2
        Pmax = xp.max(xp.sum(absP2, axis=(0, 1, 2, 3)), axis=(-1, -2))
        if self.experimentalData.operationMode =='FPM':
            frac = abs(self.optimizable.probe)/Pmax*\
                   self.optimizable.probe.conj()/(self.alphaObject*Pmax+(1-self.alphaObject)*absP2)
        else:
            frac = self.optimizable.probe.conj()/(self.alphaObject*Pmax+(1-self.alphaObject)*absP2)
        return objectPatch + self.betaObject * xp.sum(frac * DELTA, axis=2, keepdims=True)

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        absO2 = xp.abs(objectPatch) ** 2
        Omax = xp.max(xp.sum(absO2, axis=(0, 1, 2, 3)), axis=(-1, -2))
        frac = objectPatch.conj() / (self.alphaProbe * Omax + (1 - self.alphaProbe) * absO2)
        r = self.optimizable.probe + self.betaProbe * xp.sum(frac * DELTA, axis=(0, 1), keepdims=True)
        return r


