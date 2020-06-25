import numpy as np
from matplotlib import pyplot as plt
# fracPy imports
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines.BaseReconstructor import BaseReconstructor
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.monitors.Monitor import Monitor
from fracPy.utils.gpuUtils import getArrayModule
from fracPy.utils.utils import fft2c, ifft2c
import logging


class qNewton(BaseReconstructor):

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(optimizable, experimentalData, monitor)
        self.logger = logging.getLogger('qNewton')
        self.logger.info('Sucesfully created qNewton qNewton_engine')

        self.logger.info('Wavelength attribute: %s', self.optimizable.wavelength)

        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.betaProbe = 1
        self.betaObject = 1
        self.regObject = 1
        self.regProbe = 1
        self.positionOrder = 'NA'
        
    def doReconstruction(self):
        # actual reconstruction ePIE_engine
        for loop in range(self.numIterations):
            # set position order
            self.setPositionOrder()
            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                row, col = self.optimizable.positions[positionIndex]
                sy = slice(row, row + self.experimentalData.Np)
                sx = slice(col, col + self.experimentalData.Np)
                # note that object patch has size of probe array
                objectPatch = self.optimizable.object[..., sy, sx].copy()
                
                # make exit surface wave
                self.optimizable.esw = objectPatch * self.optimizable.probe
                # TODO implementing esw for mix state, where the probe has one more dimension than the object patch
                
                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.optimizable.eswUpdate - self.optimizable.esw

                
                # object update
                self.optimizable.object[..., sy, sx] = self.objectPatchUpdate(objectPatch, DELTA)

                # probe update
                self.optimizable.probe = self.probeUpdate( objectPatch, DELTA)

            # get error metric
            self.getErrorMetrics()

            # add the aperture constraint onto the probe
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)


        
    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Temporary barebones update
        """
        xp = getArrayModule(objectPatch)
        
        Pmax = xp.max(xp.sum(xp.abs(self.optimizable.probe), axis=(0,1,2,3)))
        frac = xp.abs(self.optimizable.probe) * self.optimizable.probe.conj() /  (Pmax * (xp.abs(self.optimizable.probe)**2 + self.regObject)) 
        return objectPatch + self.betaObject * xp.sum(frac * DELTA, axis=(0,2,3), keepdims=True)

       
    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Temporary barebones update

        """
        xp = getArrayModule(objectPatch)

        # Omax = xp.max(xp.sum(xp.abs(self.optimizable.object), axis=(0,1,2,3)))
        Omax = xp.max(xp.sum(xp.abs(objectPatch), axis=(0,1,2,3)))
        frac = xp.abs(objectPatch) * objectPatch.conj() /  (Omax * (xp.abs(objectPatch)**2 + self.regProbe)) 
        r = self.optimizable.probe + self.betaObject * xp.sum(frac * DELTA, axis = (0,1,3), keepdims=True)
        return r

