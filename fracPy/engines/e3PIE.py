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
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.monitors.Monitor import Monitor
from fracPy.operators.operators import aspw
import logging


class e3PIE(BaseReconstructor):

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to e3PIE reconstruction
        super().__init__(optimizable, experimentalData, monitor)
        self.logger = logging.getLogger('e3PIE')
        self.logger.info('Sucesfully created e3PIE e3PIE_engine')

        self.logger.info('Wavelength attribute: %s', self.optimizable.wavelength)

        self.initializeReconstructionParams()


    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.betaProbe = 0.25
        self.betaObject = 0.25

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

        # preallocate transfer function
        self.H = aspw(self.optimizable.probe[..., :, :], self.experimentalData.zo, self.experimentalData.wavelength,
                      self.experimentalData.Lp)[1]
        # shift transfer function to avoid fftshifts for FFTS
        self.H = np.fft2.ifftshift(H)

        # initial temporal slices TODO: did not use probeTemp, check if it is okay
        eswTemp = np.ones(self.nlambda, self.nosm, self.npsm, self.nslice, np.int(self.data.Np), np.int(self.data.Np)) #todo check Np, check why eswTemp is not only 1 slice

        # actual reconstruction e3PIE_engine
        for loop in tqdm.tqdm(range(self.numIterations)):
            # set position order
            self.setPositionOrder()
            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                row, col = self.optimizable.positions[positionIndex]
                sy = slice(row, row + self.experimentalData.Np)
                sx = slice(col, col + self.experimentalData.Np)
                # note that object patch has size of probe array
                objectPatch = self.optimizable.object[..., sy, sx].copy()

                # form first slice esw (exit surface wave)
                eswTemp[:,:,:,0,...] = objectPatch[:,:,:,0,...] * self.optimizable.probe[:,:,:,0,...]
                # propagate through object
                for sliceLoop in range(1,self.optimizable.nslice):
                    self.optimizable.probe[:,:,:,sliceLoop,...] = np.fft2.ifft(eswTemp[:,:,:,sliceLoop-1,...]* self.H)
                    eswTemp[:,:,:,sliceLoop,...] = self.optimizable.probe[:,:,:,sliceLoop,...] \
                                                   * objectPatch[:,:,:,sliceLoop,...]

                # set esw behind the last slice for FFT
                self.optimizable.esw = eswTemp[:,:,:,self.optimizable.nslice-1,...]

                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.optimizable.eswUpdate - self.optimizable.esw

                # update object slice
                for loopTemp in range(self.optimizable.nslice-1):
                    sliceLoop = self.optimizable.nslice-loopTemp+1
                    # compute current ojbect slice todo: check if objectUpdate is necessary
                    objectUpdate = self.objectPatchUpdate(self.optimizable.object[..., sliceLoop, sy, sx], DELTA,
                                                    self.optimizable.probe[:,:,:,sliceLoop,...])
                    # eswTemp update (here probe incident on last slice)
                    beth = 1 # todo, why need beth, not betaProbe, changable?
                    self.optimizable.probe[:,:,:,sliceLoop,...] = self.probeUpdate(
                        self.optimizable.object[..., sliceLoop, sy, sx], DELTA,
                        self.optimizable.probe[:,:,:,sliceLoop,...], beth)

                    # update object
                    self.optimizable.object[..., sliceLoop, sy, sx] = objectUpdate

                    # back-propagate and calculate gradient term
                    DELTA = np.fft2.ifft(np.fft2.fft(self.optimizable.probe[:,:,:,sliceLoop,...]) * H.conj()) \
                            - eswTemp[:,:,:,sliceLoop-1,...]

                # update last object slice
                objectUpdate = self.objectPatchUpdate(self.optimizable.object[..., 0, sy, sx], DELTA,
                                                      self.optimizable.probe[:, :, :, 0, ...])
                # update probe
                self.optimizable.probe[:,:,:,0,...] = self.probeUpdate(self.optimizable.object[..., 0, sy, sx], DELTA,
                                                      self.optimizable.probe[:, :, :, 0, ...], self.betaProbe)
                # update last object
                self.optimizable.object[...,0, sy, sx] = objectUpdate


            # set porduct of all object slices
            self.optimizable.objectProd = np.prod(self.optimizable.object,4)

            # get error metric
            self.getErrorMetrics()

            # apply Constraints todo uncomment orthogonalization? check object smootheness regularization
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

    def objectPatchUpdate(self, objectPatch:np.ndarray, DELTA:np.ndarray, localProbe:np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)

        frac = localProbe.conj() / xp.max(
            xp.sum(xp.abs(localProbe) ** 2, axis=(0, 1, 2, 3)))
        return objectPatch + self.betaObject * xp.sum(frac * DELTA, axis=(0, 2, 3), keepdims=True)

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray, localProbe: np.ndarray, beth):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = objectPatch.conj() / xp.max(xp.sum(xp.abs(objectPatch) ** 2, axis=(0, 1, 2, 3)))
        r = localProbe + beth * xp.sum(frac * DELTA, axis=(0, 1, 3), keepdims=True)
        # if self.absorbingProbeBoundary:
        #     aleph = 1e-3
        #     r = (1 - aleph) * r + aleph * r * self.probeWindow
        return r

class ePIE_GPU(e3PIE):
    """
    GPU-based implementation of e3PIE
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cp is None:
            raise ImportError('Could not import cupy')
        self.logger = logging.getLogger('e3PIE_GPU')
        self.logger.info('Hello from e3PIE_GPU')

    def _prepare_doReconstruction(self):
        self.logger.info('Ready to start transfering stuff to the GPU')
        self._move_data_to_gpu()

    def _move_data_to_gpu(self):
        """
        Move the data to the GPU
        :return:
        """
        # optimizable parameters
        self.optimizable.probe = cp.array(self.optimizable.probe, cp.complex64)
        self.optimizable.object = cp.array(self.optimizable.object, cp.complex64)

        # non-optimizable parameters
        self.experimentalData.ptychogram = cp.array(self.experimentalData.ptychogram, cp.float32)
        self.experimentalData.probe = cp.array(self.experimentalData.probe, cp.complex64)

        # ePIE parameters
        self.logger.info('Detector error shape: %s', self.detectorError.shape)
        self.detectorError = cp.array(self.detectorError)

        # proapgators to GPU
        if self.propagator == 'Fresnel':
            self.optimizable.quadraticPhase = cp.array(self.optimizable.quadraticPhase)
        elif self.propagator == 'ASP':
            self.optimizable.transferFunction = cp.array(self.optimizable.transferFunction)
        elif self.propagator == 'scaledASP':
            self.optimizable.Q1 = cp.array(self.optimizable.Q1)
            self.optimizable.Q2 = cp.array(self.optimizable.Q2)
