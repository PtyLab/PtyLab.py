import napari
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


class ePIE(BaseEngine):

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

        import napari
        from napari.qt.threading import  thread_worker


        # viewer = napari.Viewer()
        # viewer.add_image(abs(self.reconstruction.object), name='object')
        # viewer.add_image(abs(self.reconstruction.probe), name='probe')

        # def update_estimate(*inputs):
        #     i, probe, objectPatch, objectsubset = inputs
        #     viewer.layers['object'].data = abs(objectsubset)
        #     viewer.layers['probe'].data = abs(probe)
        #
        # keep_waiting = True
        # def return_function(*inputs):
        #     keep_waiting = False

        # rfun = thread_worker(self._reconstruct, connect={'yield': update_estimate,
        #                                                  'return': return_function}, start_thread=True)
        # while keep_waiting:
        #     import pyqtgraph as pg
        #     pg.QtGui.QApplication.processEvents()
        #
        #     #print('Waiting')
        #     import time
        #     time.sleep(.1)
#        return [r for r in self._reconstruct()][-1]
        recs = self._reconstruct()
        while True:
            x = next(recs)
            print('iterating')








    def _reconstruct(self):
        # self._prepareReconstruction()
        print('Starting reconstruction')
        # actual reconstruction ePIE_engine
        self.pbar = tqdm.trange(self.numIterations, desc='ePIE', file=sys.stdout, leave=True)
        i = -1
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()
            for positionLoop, positionIndex in enumerate(self.positionIndices):
                i += 1
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
                self.reconstruction.object[..., sy, sx] = self.objectPatchUpdate(objectPatch, DELTA)

                # probe update
                self.reconstruction.probe = self.probeUpdate(objectPatch, DELTA)
                # yield a few items so we can plot them
                yield i, self.reconstruction.probe, objectPatch, self.reconstruction.object[..., sy, sx]

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            # self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info('switch to cpu')
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

        return None


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


