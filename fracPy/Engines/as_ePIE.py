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
from fracPy.Reconstruction.Reconstruction import Reconstruction
from fracPy.Engines.BaseEngine import BaseEngine
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Params.Params import Params
from fracPy.utils.gpuUtils import getArrayModule
from fracPy.Monitor.Monitor import Monitor
from fracPy.utils.utils import fft2c, ifft2c
import logging
import tqdm
import sys


class as_ePIE(BaseEngine):
    # automatic step size ePIE

    def __init__(self, reconstruction: Reconstruction, experimentalData: ExperimentalData, params: Params,
                 monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger('as_ePIE')
        self.logger.info('Sucesfully created as_ePIE_engine')
        self.logger.info('Wavelength attribute: %s', self.reconstruction.wavelength)
        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.numIterations = 50

    def reconstruct(self):
        self._prepareReconstruction()

        self.stepProbeHistory = np.zeros(self.numIterations)
        self.stepObjectHistory = np.zeros(self.numIterations)

        # actual reconstruction ePIE_engine
        self.pbar = tqdm.trange(self.numIterations, desc='as_ePIE', file=sys.stdout, leave=True)
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()

            stepObjectAverage = 0
            stepProbeAverage = 0

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
                self.reconstruction.object[..., sy, sx],stepSizeObject = self.objectPatchUpdate(objectPatch, DELTA)
                stepObjectAverage+=stepSizeObject

                # probe update
                self.reconstruction.probe, stepSizeProbe = self.probeUpdate(objectPatch, DELTA)
                stepProbeAverage += stepSizeProbe

            # record stepsize history
            self.stepObjectHistory[loop] = stepObjectAverage / self.experimentalData.numFrames
            self.stepProbeHistory[loop] = stepProbeAverage / self.experimentalData.numFrames

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info('switch to cpu')
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        objectGrad = self.reconstruction.probe.conj()*DELTA

        # compute step size object
        stepSizeObject = xp.sum(np.real(DELTA*(objectGrad*self.reconstruction.probe).conj())) /\
                         (xp.sum(abs(objectGrad*self.reconstruction.probe)**2) + 1)
        return objectPatch + stepSizeObject * objectGrad, stepSizeObject

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        probeGrad = objectPatch.conj()*DELTA
        stepSizeProbe = xp.sum(np.real(DELTA*(probeGrad * objectPatch).conj()), axis=(-1, -2), keepdims=True)/\
                        (xp.sum(abs(probeGrad*objectPatch)**2) + 1)
        return self.reconstruction.probe + stepSizeProbe*probeGrad, stepSizeProbe


