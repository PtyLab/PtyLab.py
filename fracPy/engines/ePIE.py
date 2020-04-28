import numpy as np
from matplotlib import pyplot as plt
# fracPy imports
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines.BaseReconstructor import BaseReconstructor
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.utils.gpuUtils import getArrayModule
from fracPy.utils.utils import fft2c, ifft2c
import logging


class ePIE(BaseReconstructor):

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(optimizable, experimentalData)
        self.logger = logging.getLogger('ePIE')
        self.logger.info('Sucesfully created ePIE ePIE_engine')

        self.logger.info('Wavelength attribute: %s', self.optimizable.wavelength)

        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        # self.eswUpdate = self.optimizable.esw.copy()
        self.betaProbe = 0.25
        self.betaObject = 0.25

    def doReconstruction(self):
        # actual reconstruction ePIE_engine
        for loop in range(self.numIterations):
            # set position order
            self.positionIndices = self.setPositionOrder()
            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                row, col = self.optimizable.positions[positionIndex]
                sy = slice(row, row + self.experimentalData.Np)
                sx = slice(col, col + self.experimentalData.Np)
                # note that object patch has size of probe array
                objectPatch = self.optimizable.object[:, sy, sx].copy()
                
                # make exit surface wave
                self.optimizable.esw = objectPatch * self.optimizable.probe
                # TODO implementing esw for mix state, where the probe has one more dimension than the object patch
                # plt.figure(1)
                # plt.imshow(abs(self.optimizable.esw[0,:,:]))
                # plt.pause(0.1)
                
                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.optimizable.eswUpdate - self.optimizable.esw

                
                # object update
                self.optimizable.object[:, sy, sx] = self.objectPatchUpdate(objectPatch, DELTA)

                # probe update
                self.optimizable.probe = self.probeUpdate( objectPatch, DELTA)

            # get error metric
            # self.getErrorMetrics()

            # self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

        
    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        
        frac = self.optimizable.probe.conj() / xp.max(xp.sum(abs(self.optimizable.probe) ** 2, 0))
        # this is two statements in matlab but it should only be one in python
        return objectPatch + self.betaObject * frac * DELTA
       
    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = objectPatch.conj() / xp.max(xp.sum(abs(objectPatch) ** 2, 0))
        # this is two statements in matlab but it should only be one in python
        # TODO figure out unit tests and padding dimensions
        r = self.optimizable.probe + self.betaProbe * frac * DELTA
        if self.absorbingProbeBoundary:
            aleph = 1e-3
            r = (1 - aleph) * r + aleph * r * self.probeWindow
        return r

