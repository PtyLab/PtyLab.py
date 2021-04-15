import numpy as np
from matplotlib import pyplot as plt
# fracPy imports
try:
    import cupy as cp
except ImportError:
    print('Cupy not available, will not be able to run GPU based computation')
    # Still define the name, we'll take care of it later but in this way it's still possible
    # to see that gPIE exists for example.
    cp = None
from fracPy.Optimizables.Optimizable import Optimizable
from fracPy.Engines.BaseReconstructor import BaseReconstructor
from fracPy.FixedData.DefaultExperimentalData import ExperimentalData
from fracPy.Params.ReconstructionParameters import Reconstruction_parameters
from fracPy.Monitors.Monitor import Monitor
from fracPy.utils.gpuUtils import getArrayModule
from fracPy.utils.utils import fft2c, ifft2c
import logging
import tqdm
import sys

class qNewton(BaseReconstructor):

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, params: Reconstruction_parameters, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(optimizable, experimentalData, params, monitor)
        self.logger = logging.getLogger('qNewton')
        self.logger.info('Sucesfully created qNewton qNewton_engine')

        self.logger.info('Wavelength attribute: %s', self.optimizable.wavelength)
        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the qNewton settings.
        :return:
        """
        self.betaProbe = 1
        self.betaObject = 1
        self.regObject = 1
        self.regProbe = 1

    def doReconstruction(self):
        self._prepareReconstruction()

        self.pbar = tqdm.trange(self.numIterations, desc='qNewton', file=sys.stdout, leave=True)
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

            #todo clearMemory implementation
        
    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Temporary barebones update
        """
        xp = getArrayModule(objectPatch)
        Pmax = xp.max(xp.sum(xp.abs(self.optimizable.probe), axis=(0, 1, 2, 3)))
        frac = xp.abs(self.optimizable.probe)/Pmax * self.optimizable.probe.conj() / (xp.abs(self.optimizable.probe)**2 + self.regObject)
        return objectPatch + self.betaObject * xp.sum(frac * DELTA, axis=(0,2,3), keepdims=True)

       
    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Temporary barebones update

        """
        xp = getArrayModule(objectPatch)
        Omax = xp.max(xp.sum(xp.abs(self.optimizable.object), axis=(0, 1, 2, 3)))
        frac = xp.abs(objectPatch)/Omax * objectPatch.conj() / (xp.abs(objectPatch)**2 + self.regProbe)
        r = self.optimizable.probe + self.betaProbe * xp.sum(frac * DELTA, axis=(0,1,3), keepdims=True)
        return r
