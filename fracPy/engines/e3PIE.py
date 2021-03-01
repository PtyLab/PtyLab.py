import numpy as np
from matplotlib import pyplot as plt
import tqdm

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


class e3PIE(BaseReconstructor):

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, params: Params, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to e3PIE reconstruction
        super().__init__(optimizable, experimentalData, params, monitor)
        self.logger = logging.getLogger('e3PIE')
        self.logger.info('Sucesfully created e3PIE e3PIE_engine')

        self.logger.info('Wavelength attribute: %s', self.optimizable.wavelength)

        self.initializeReconstructionParams()


    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the e3PIE settings.
        :return:
        """
        self.params.betaProbe = 0.25
        self.params.betaObject = 0.25
        # preallocate transfer function
        self.optimizable.H = aspw(np.squeeze(self.optimizable.probe[0, 0, 0, 0, ...]), self.optimizable.dz,
                                  self.optimizable.wavelength/self.optimizable.refrIndex,self.optimizable.Lp)[1]
        # shift transfer function to avoid fftshifts for FFTS
        self.optimizable.H = np.fft.ifftshift(self.optimizableH)


    def doReconstruction(self):
        self._prepareReconstruction()

        # initialize esw
        self.optimizable.esw = self.optimizable.probe.copy()
        # get module
        xp = getArrayModule(self.optimizable.object)
        # actual reconstruction e3PIE_engine
        for loop in tqdm.tqdm(range(self.params.numIterations)):
            # set position order
            if loop == self.params.numIterations - 1:
                noreason = True
            self.setPositionOrder()
            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                row, col = self.optimizable.positions[positionIndex]
                sy = slice(row, row + self.optimizable.Np)
                sx = slice(col, col + self.optimizable.Np)
                # note that object patch has size of probe array
                objectPatch = self.optimizable.object[..., sy, sx].copy()
                objectPatch2 = self.optimizable.object[..., :,:].copy()
                # form first slice esw (exit surface wave)
                self.optimizable.esw[:,:,:,0,...] = objectPatch[:,:,:,0,...] * self.optimizable.probe[:,:,:,0,...]

                # propagate through object
                for sliceLoop in range(1,self.optimizable.nslice):
                    self.optimizable.probe[:,:,:,sliceLoop,...] = xp.fft.ifft2(xp.fft.fft2(self.optimizable.esw[:,:,:,sliceLoop-1,...])* self.H)
                    self.optimizable.esw[:,:,:,sliceLoop,...] = self.optimizable.probe[:,:,:,sliceLoop,...] \
                                                   * objectPatch[:,:,:,sliceLoop,...]

                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = (self.optimizable.eswUpdate - self.optimizable.esw)[:, :, :, -1, ...]

                # update object slice
                for loopTemp in range(self.optimizable.nslice-1):
                    sliceLoop = self.optimizable.nslice-1-loopTemp
                    # compute and update current object slice
                    self.optimizable.object[..., sliceLoop, sy, sx] = \
                        self.objectPatchUpdate(objectPatch[:,:,:,sliceLoop,...], DELTA,
                                               self.optimizable.probe[:,:,:,sliceLoop,...])
                    # eswTemp update (here probe incident on last slice)
                    beth = 1 # todo, why need beth, not betaProbe, changable?
                    self.optimizable.probe[:,:,:,sliceLoop,...] = \
                        self.probeUpdate(objectPatch[:,:,:, sliceLoop,...], DELTA,
                                         self.optimizable.probe[:,:,:,sliceLoop,...], beth)

                    # back-propagate and calculate gradient term
                    DELTA = xp.fft.ifft2(xp.fft.fft2(self.optimizable.probe[:,:,:,sliceLoop,...]) * self.optimizableH.conj()) \
                            - self.optimizable.esw[:,:,:,sliceLoop-1,...]

                # update last object slice
                self.optimizable.object[...,0, sy, sx] = self.objectPatchUpdate(objectPatch[:, :, :, 0, ...], DELTA,
                                                      self.optimizable.probe[:, :, :, 0, ...])
                # update probe
                self.optimizable.probe[:,:,:,0,...] = self.probeUpdate(objectPatch[:, :, :, 0, ...], DELTA,
                                                      self.optimizable.probe[:, :, :, 0, ...], self.params.betaProbe)


            # set porduct of all object slices
            self.optimizable.objectProd = np.prod(self.optimizable.object, 3)

            # get error metric
            self.getErrorMetrics()

            # apply Constraints todo uncomment orthogonalization? check object smootheness regularization
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info('switch to cpu')
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

    def objectPatchUpdate(self, objectPatch:np.ndarray, DELTA:np.ndarray, localProbe:np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = localProbe.conj() / xp.max(xp.sum(xp.abs(localProbe) ** 2, axis=(0, 1, 2)))
        return objectPatch + self.params.betaObject * xp.sum(frac * DELTA, axis=(0, 2), keepdims=True)

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray, localProbe: np.ndarray, beth):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = objectPatch.conj() / xp.max(xp.sum(xp.abs(objectPatch) ** 2, axis=(0, 1, 2)))
        r = localProbe + self.params.betaProbe * xp.sum(frac * DELTA, axis=(0, 1), keepdims=True)
        return r
