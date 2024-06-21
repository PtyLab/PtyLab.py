import numpy as np
import tqdm
from matplotlib import pyplot as plt

try:
    import cupy as cp
except ImportError:
    # print('Cupy not available, will not be able to run GPU based computation')
    # Still define the name, we'll take care of it later but in this way it's still possible
    # to see that gPIE exists for example.
    cp = None

import logging
import sys

from fracPy.Engines.BaseEngine import BaseEngine
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Monitor.Monitor import Monitor
from fracPy.Operators.Operators import aspw
from fracPy.Params.Params import Params

# fracPy imports
from fracPy.Reconstruction.Reconstruction import Reconstruction
from fracPy.utils.gpuUtils import asNumpyArray, getArrayModule


class e3PIE(BaseEngine):

    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to e3PIE reconstruction
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger("e3PIE")
        self.logger.info("Sucesfully created e3PIE e3PIE_engine")

        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)

        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the e3PIE settings.
        :return:
        """
        self.params.betaProbe = 0.25
        self.params.betaObject = 0.25
        self.numIterations = 50

        if False:
            # preallocate transfer function
            self.reconstruction.H = aspw(
                np.squeeze(self.reconstruction.probe[0, 0, 0, 0, ...]),
                self.reconstruction.dz,
                self.reconstruction.wavelength / self.reconstruction.refrIndex,
                self.reconstruction.Lp,
            )[1]
            # shift transfer function to avoid fftshifts for FFTS
            # self.reconstruction.H = np.fft.ifftshift(self.optimizableH)
            self.reconstruction.H = np.fft.ifftshift(self.reconstruction.H)

        if True:
            import cupy as xp

            # preallocate transfer function
            self.reconstruction.H = aspw(
                xp.squeeze(self.reconstruction.probe[0, 0, 0, 0, ...]),
                self.reconstruction.dz,
                self.reconstruction.wavelength / self.reconstruction.refrIndex,
                self.reconstruction.Lp,
            )[1]
            # shift transfer function to avoid fftshifts for FFTS
            # self.reconstruction.H = np.fft.ifftshift(self.optimizableH)
            self.reconstruction.H = xp.fft.ifftshift(self.reconstruction.H)

    def reconstruct(self):
        self._prepareReconstruction()

        # initialize esw
        self.reconstruction.esw = self.reconstruction.probe.copy()
        # get module
        xp = getArrayModule(self.reconstruction.object)

        self.pbar = tqdm.trange(
            self.numIterations, desc="e3PIE", file=sys.stdout, leave=True
        )

        # self.pbar = (1, 2)

        for loop in self.pbar:
            if loop == self.numIterations - 1:
                noreason = True
            self.setPositionOrder()
            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                row, col = self.reconstruction.positions[positionIndex]
                sy = slice(row, row + self.reconstruction.Np)
                sx = slice(col, col + self.reconstruction.Np)
                # note that object patch has size of probe array
                objectPatch = self.reconstruction.object[..., sy, sx].copy()
                # objectPatch2 = self.reconstruction.object[..., :, :].copy()

                # form first slice esw (exit surface wave)
                self.reconstruction.esw[:, :, :, 0, ...] = (
                    objectPatch[:, :, :, 0, ...]
                    * self.reconstruction.probe[:, :, :, 0, ...]
                )

                # propagate through object
                for sliceLoop in range(1, self.reconstruction.nslice):
                    self.reconstruction.probe[:, :, :, sliceLoop, ...] = xp.fft.ifft2(
                        xp.fft.fft2(
                            self.reconstruction.esw[:, :, :, sliceLoop - 1, ...]
                        )
                        * self.reconstruction.H
                    )
                    self.reconstruction.esw[:, :, :, sliceLoop, ...] = (
                        self.reconstruction.probe[:, :, :, sliceLoop, ...]
                        * objectPatch[:, :, :, sliceLoop, ...]
                    )

                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = (self.reconstruction.eswUpdate - self.reconstruction.esw)[
                    :, :, :, -1, ...
                ]
                # update object slice
                for loopTemp in range(self.reconstruction.nslice - 1):

                    sliceLoop = self.reconstruction.nslice - 1 - loopTemp
                    # compute and update current object slice
                    self.reconstruction.object[..., sliceLoop, sy, sx] = (
                        self.objectPatchUpdate(
                            objectPatch[:, :, :, sliceLoop, ...],
                            DELTA,
                            self.reconstruction.probe[:, :, :, sliceLoop, ...],
                        )
                    )
                    # eswTemp update (here probe incident on last slice)
                    beth = 1  # todo, why need beth, not betaProbe, changable?
                    self.reconstruction.probe[:, :, :, sliceLoop, ...] = (
                        self.probeUpdate(
                            objectPatch[:, :, :, sliceLoop, ...],
                            DELTA,
                            self.reconstruction.probe[:, :, :, sliceLoop, ...],
                            beth,
                        )
                    )

                    # back-propagate and calculate gradient term
                    DELTA = (
                        xp.fft.ifft2(
                            xp.fft.fft2(
                                self.reconstruction.probe[:, :, :, sliceLoop, ...]
                            )
                            * self.reconstruction.H.conj()
                        )
                        - self.reconstruction.esw[:, :, :, sliceLoop - 1, ...]
                    )

                # update last object slice
                self.reconstruction.object[..., 0, sy, sx] = self.objectPatchUpdate(
                    objectPatch[:, :, :, 0, ...],
                    DELTA,
                    self.reconstruction.probe[:, :, :, 0, ...],
                )
                # update probe
                self.reconstruction.probe[:, :, :, 0, ...] = self.probeUpdate(
                    objectPatch[:, :, :, 0, ...],
                    DELTA,
                    self.reconstruction.probe[:, :, :, 0, ...],
                    self.betaProbe,
                )

            # set porduct of all object slices
            self.reconstruction.objectProd = np.prod(self.reconstruction.object, 3)

            # get error metric
            self.getErrorMetrics()

            # apply Constraints todo uncomment orthogonalization? check object smootheness regularization
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info("switch to cpu")
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

    def objectPatchUpdate(
        self, objectPatch: np.ndarray, DELTA: np.ndarray, localProbe: np.ndarray
    ):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = localProbe.conj() / xp.max(
            xp.sum(xp.abs(localProbe) ** 2, axis=(0, 1, 2))
        )
        return objectPatch + self.betaObject * xp.sum(
            frac * DELTA, axis=(0, 2), keepdims=True
        )

    def probeUpdate(
        self, objectPatch: np.ndarray, DELTA: np.ndarray, localProbe: np.ndarray, beth
    ):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = objectPatch.conj() / xp.max(
            xp.sum(xp.abs(objectPatch) ** 2, axis=(0, 1, 2))
        )
        r = localProbe + beth * xp.sum(frac * DELTA, axis=(0, 1), keepdims=True)
        return r
