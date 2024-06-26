import numpy as np
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

import tqdm

from PtyLab.Engines.BaseEngine import BaseEngine
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.Params.Params import Params

# fracPy imports
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.utils.fsvd import rsvd
from PtyLab.utils.gpuUtils import asNumpyArray, getArrayModule, isGpuArray


class OPR(BaseEngine):

    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger("ePIE")
        self.logger.info("Sucesfully created ePIE ePIE_engine")
        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE/OPR engine
        """
        self.alpha = self.params.OPR_alpha
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.numIterations = 50
        self.OPR_modes = self.params.OPR_modes
        self.n_subspace = self.params.OPR_subspace

    def reconstruct(self):
        self._prepareReconstruction()

        # OPR parameters
        Nmodes = self.OPR_modes.shape[0]
        Np = self.reconstruction.Np
        Nframes = self.experimentalData.numFrames
        mode_slice = self.OPR_modes
        n_subspace = self.n_subspace

        self.reconstruction.probe_stack = cp.zeros(
            (1, 1, Nmodes, 1, Np, Np, Nframes), dtype=cp.complex64
        )

        for i, mode in enumerate(self.OPR_modes):
            # fill the probe-stack with the inital guess of the probes
            self.reconstruction.probe_stack[0, 0, i, 0, :, :, :] = cp.repeat(
                self.reconstruction.probe[0, 0, mode, 0, :, :, cp.newaxis],
                Nframes,
                axis=2,
            )

        # actual reconstruction ePIE_engine
        self.pbar = tqdm.trange(
            self.numIterations, desc="OPR", file=sys.stdout, leave=True
        )
        for loop in self.pbar:
            self.it = loop
            # set position order
            self.setPositionOrder()
            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                row, col = self.reconstruction.positions[positionIndex]
                sy = slice(row, row + self.reconstruction.Np)
                sx = slice(col, col + self.reconstruction.Np)
                # note that object patch has size of probe array
                objectPatch = self.reconstruction.object[..., sy, sx].copy()

                # Get dim reduced probe
                self.reconstruction.probe[:, :, mode_slice, :, :, :] = (
                    self.reconstruction.probe_stack[..., positionIndex]
                )

                # make exit surface wave
                self.reconstruction.esw = objectPatch * self.reconstruction.probe

                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.reconstruction.eswUpdate - self.reconstruction.esw

                if loop % self.params.OPR_tv_freq == 0 and self.params.OPR_tv:
                    self.reconstruction.object[..., sy, sx] = self.objectPatchUpdate_TV(
                        objectPatch, DELTA
                    )
                else:
                    # object update
                    self.reconstruction.object[..., sy, sx] = self.objectPatchUpdate(
                        objectPatch, DELTA
                    )

                # probe update
                self.reconstruction.probe = self.probeUpdate(
                    objectPatch, DELTA, weight=1
                )

                # save first, dominant probe mode
                self.reconstruction.probe_stack[..., positionIndex] = cp.copy(
                    self.reconstruction.probe[:, :, mode_slice, :, :, :]
                )

            # get error metric
            self.getErrorMetrics()

            if self.params.OPR_orthogonalize_modes:
                self.orthogonalizeIncoherentModes()

            self.reconstruction.probe_stack = self.orthogonalizeProbeStack(
                self.reconstruction.probe_stack, n_subspace
            )

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info("switch to cpu")
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

    def orthogonalizeIncoherentModes(self):
        """
        Function which cycles through the probe stack and orthogonalizes
        all incoherent modes of all postions
        """
        nFrames = self.experimentalData.numFrames
        n = self.reconstruction.Np
        nModes = self.reconstruction.probe_stack.shape[2]
        for pos in range(nFrames):
            probe = self.reconstruction.probe_stack[0, 0, :, 0, :, :, pos]
            probe = probe.reshape(nModes, n * n)

            U, s, Vh = self.svd(probe)

            modes = (s[:, None] * Vh).reshape(nModes, n, n)
            self.reconstruction.probe_stack[0, 0, :, 0, :, :, pos] = modes

    def average(self, arr):
        """
        Calculates the average from neighboring values of a numpy array
        :param arr: 1-dimensional input array, which is used to
        calculate the average
        :return: 1-dimensionl array with the same shape as the input array
        """
        arr_start = arr[:-1]
        arr_end = arr[1:]
        arr_end = cp.append(arr_end, 0)
        arr_start = cp.append(0, arr_start)
        divider = cp.ones_like(arr) * 3
        divider[0] = 2
        divider[-1] = 2
        return (arr + arr_end + arr_start) / divider

    def svd(self, P):
        if isGpuArray(P):
            try:
                return cp.linalg.svd(P, full_matrices=False)
            except:
                print(
                    "Something is wrong with SVD on cuda. Probably an installation error"
                )
                raise
        A, v, At = np.linalg.svd(asNumpyArray(P), full_matrices=False)
        if isGpuArray(P):
            A = cp.array(A)
            v = cp.array(v)
            At = cp.array(At)
        return A, v, At

    def rsvd(self, P, n_dim):
        return rsvd(P, n_dim)
        # A, v, At = self.svd(P)
        # v[n_dim:] = 0
        # return A, v, At

    def orthogonalizeProbeStack(self, probe_stack, n_dim):
        """
        Takes the probe stack maps it by a truncated singular value decomposition in to
        a lower dimensional (n_dim) space.
        :param probe_stack: Probes of all positions
        :param n_dim: Dimension of the lower dimensional sub space
        :return: reduced probe stack
        """
        n = self.reconstruction.Np
        nFrames = self.experimentalData.numFrames

        for i, mode in enumerate(self.OPR_modes):

            if self.params.OPR_tsvd_type == "randomized":
                U, s, Vh = self.rsvd(
                    probe_stack[:, :, i, :, :, :].reshape(n * n, nFrames), n_dim
                )
            elif self.params.OPR_tsvd_type == "numpy":
                U, s, Vh = cp.linalg.svd(
                    probe_stack[:, :, i, :, :, :].reshape(n * n, nFrames),
                    full_matrices=False,
                )
                s[n_dim:] = 0

            if self.params.OPR_neighbor_constraint:
                # Calculate the average of neigboring singular values
                content = cp.dot(cp.diag(s), Vh)
                for j in range(n_dim):
                    content[j] = self.average(content[j])

                probe_stack[:, :, i, :, :, :] = self.alpha * probe_stack[
                    :, :, i, :, :, :
                ] + (1 - self.alpha) * cp.dot(U, content).reshape(n, n, nFrames)
            else:
                update = (U @ (s[:, None] * Vh)).reshape(n, n, nFrames)
                probe_stack[:, :, i, :, :, :] *= self.alpha
                probe_stack[:, :, i, :, :, :] += (1 - self.alpha) * update

        return probe_stack

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        ePIE object update function
        :param objectPatch: Slice of the object array
        :param DELTA:
        :return: updated object patch
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)

        frac = self.reconstruction.probe.conj() / xp.max(
            xp.sum(xp.abs(self.reconstruction.probe) ** 2, axis=(0, 1, 2, 3))
        )
        return objectPatch + self.betaObject * xp.sum(
            frac * DELTA, axis=(0, 2, 3), keepdims=True
        )

    def probeUpdate(
        self, objectPatch: np.ndarray, DELTA: np.ndarray, weight: float, gimmel=0.1
    ):
        """
        Update the probe
        :param objectPatch: Slice of the object array
        :param DELTA:
        :return: updated probe
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = objectPatch.conj() / (
            xp.max(xp.sum(xp.abs(objectPatch) ** 2, axis=(0, 1, 2, 3))) + gimmel
        )
        frac = frac * weight
        r = self.reconstruction.probe + self.betaProbe * xp.sum(
            frac * DELTA, axis=(0, 1, 3), keepdims=True
        )
        return r
