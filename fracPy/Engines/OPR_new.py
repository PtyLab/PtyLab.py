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


class OPR_new(BaseEngine):

    def __init__(self, reconstruction: Reconstruction, experimentalData: ExperimentalData, params: Params, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger('ePIE')
        self.logger.info('Sucesfully created ePIE ePIE_engine')
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

        # OPR parameters
        # self.OPR_modes = np.array([0, 1, 2, 3])
        self.OPR_modes = np.array([0, 1, 2, 3])
        Nmodes = self.OPR_modes.shape[0]
        Np = self.reconstruction.Np
        Nframes = self.experimentalData.numFrames
        mode_slice = slice(np.min(self.OPR_modes), np.max(self.OPR_modes) + 1) 
        n_subspace = 4 
        import copy
        
        self.reconstruction.probe_stack = cp.zeros((1, 1, Nmodes, 1, Np, Np, Nframes), dtype=cp.complex64)
        for i in self.OPR_modes:
            test = cp.repeat(self.reconstruction.probe[0, 0, i, 0, :, :, cp.newaxis], Nframes, axis=2)
            self.reconstruction.probe_stack[0, 0, i, 0, :, :, :] = copy.deepcopy(test) 
        print('shapes')
        print(mode_slice) 
        print(self.reconstruction.probe_stack.shape)
        print(self.reconstruction.probe.shape)

        # actual reconstruction ePIE_engine
        self.pbar = tqdm.trange(self.numIterations, desc='OPR_new', file=sys.stdout, leave=True)
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
                self.reconstruction.probe[:, :, mode_slice, :, :, :] = self.reconstruction.probe_stack[..., positionIndex]

                # make exit surface wave
                self.reconstruction.esw = objectPatch * self.reconstruction.probe 
                
                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.reconstruction.eswUpdate - self.reconstruction.esw

                # object update
                self.reconstruction.object[..., sy, sx] = self.objectPatchUpdate(objectPatch, DELTA)
                
                # probe update
                self.reconstruction.probe = self.probeUpdate(objectPatch, DELTA)

                # save first, dominant probe mode
                self.reconstruction.probe_stack[..., positionIndex] = cp.copy(self.reconstruction.probe[:, :, mode_slice, :, :, :]) 

            # get error metric
            self.getErrorMetrics()
            
            # Orthogonalization does not seem to be necessary
            # self.orthogonalizeIncoherentModes()

            OPR_constraint = True
            if OPR_constraint:
               self.reconstruction.probe_stack = self.orthogonalizeProbeStack(self.reconstruction.probe_stack, n_subspace)

            # apply Constraints
            self.applyConstraints(loop)

            ''' 
            OPR_constraint = True
            if OPR_constraint:
               self.reconstruction.probe_stack = self.orthogonalizeProbeStack(self.reconstruction.probe_stack, 3)
            '''

            # show reconstruction
            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info('switch to cpu')
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

    def orthogonalizeIncoherentModes(self):
        # This function orthogonalizes all modes
        nFrames = self.experimentalData.numFrames 
        n = self.reconstruction.Np
        print(n)
        nModes = self.reconstruction.probe.shape[2]
        print(nModes)
        print(self.reconstruction.probe_stack.shape)
        for pos in range(nFrames):
            probe = self.reconstruction.probe_stack[0, 0, :, 0, :, :, pos]
            probe = probe.reshape(nModes, n*n)
            U, s, Vh = cp.linalg.svd(probe, full_matrices=False)
            modes = cp.dot(cp.diag(s), Vh).reshape(nModes, n, n)
            self.reconstruction.probe_stack[0, 0, :, 0, :, :, pos] = modes

    def orthogonalizeProbeStack(self, probe_stack, n_dim):
        plot_cycle = 10
        plot = False 
        n = self.reconstruction.Np
        nFrames = self.experimentalData.numFrames

        for i in self.OPR_modes:
            # temp = cp.copy(probe_stack[:, :, i, :, :, :].reshape(n * n, nFrames)) 
            U, s, Vh = cp.linalg.svd(probe_stack[:, :, i, :, :, :].reshape(n * n, nFrames), full_matrices=False)
            s[n_dim:] = 0
            probe_stack[:, :, i, :, :, :] = cp.dot(U, cp.dot(cp.diag(s), Vh)).reshape(n, n, nFrames) 
        
        if self.it == 0 and plot:
            plt.ion()
            self.fig = plt.figure('content')
            self.fig_2 = plt.figure('second modes')
            self.ax = self.fig.add_subplot(111)
            self.ax_2 = self.fig_2.add_subplot(121)
            self.ax_3 = self.fig_2.add_subplot(122)
      
        if self.it % plot_cycle == 0 and plot:
            self.reconstruction.modes = U.reshape(n, n, nFrames)
            content = cp.dot(cp.diag(s), Vh)
            self.ax.imshow(np.log10(np.abs(content.get())[0:3, :]), aspect='auto')
            self.ax_2.imshow(np.log10(np.abs(self.reconstruction.modes[:, :, 1].get())))
            self.ax_3.imshow(np.log10(np.abs(self.reconstruction.modes[:, :, 2].get())))
            self.fig.canvas.draw() 

        return probe_stack 

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)

        frac = self.reconstruction.probe.conj() / xp.max(xp.sum(xp.abs(self.reconstruction.probe) ** 2, axis=(0, 1, 2, 3)))
        return objectPatch + self.betaObject * xp.sum(frac * DELTA, axis=(0,2,3), keepdims=True)

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = objectPatch.conj() / xp.max(xp.sum(xp.abs(objectPatch) ** 2, axis=(0,1,2,3)))
        r = self.reconstruction.probe + self.betaProbe * xp.sum(frac * DELTA, axis=(0, 1, 3), keepdims=True)
        return r
    
    def probeUpdate_new(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        # frac = objectPatch.conj() / xp.max(xp.sum(xp.abs(objectPatch) ** 2, axis=(0,1,2,3)))
        self.reconstruction.probe += self.betaProbe * xp.sum(objectPatch.conj() / xp.max(xp.sum(xp.abs(objectPatch) ** 2, axis=(0,1,2,3))) * DELTA, axis=(0, 1, 3), keepdims=True)


