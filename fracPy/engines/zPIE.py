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
from fracPy.utils.gpuUtils import getArrayModule
from fracPy.monitors.Monitor import Monitor
from fracPy.operators.operators import aspw
import logging


class zPIE(BaseReconstructor):

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(optimizable, experimentalData, monitor)
        self.logger = logging.getLogger('ePIE')
        self.logger.info('Sucesfully created zPIE zPIE_engine')

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


    def _prepare_doReconstruction(self):
        """
        This function is called just before the reconstructions start.

        Can be used to (for instance) transfer data to the GPU at the last moment.
        :return:
        """
        pass

    def doReconstruction(self):
        self._prepare_doReconstruction()
        # actual reconstruction zPIE_engine
        if not hasattr(self, 'zHisory'):
            self.zHistory = []
        zMomentun = 0

        # preallocate grids
        if self.propagator == 'ASP':
            n = self.experimentalData.Np
        else:
            n = 2*self.experimentalData.Np

        X,Y = np.meshgrid(np.arange(-n//2,n//2),np.arange(-n//2,n//2))
        w = np.exp(-(np.sqrt(X**2+Y**2)/self.experimentalData.Np)**4)

        for loop in tqdm.tqdm(range(self.numIterations)):
            # set position order
            self.setPositionOrder()

            # get positions
            if loop==1:
                zNew = self.experimentalData.zo
            else:
                d = 10;
                dz = np.linspace(-d,d,11)/10*self.optimizable.DoF
                merit = []
                # todo, mixed states implementation
                for k in np.arange(len(dz)):
                    imProp = aspw(np.squeeze(w*self.optimizable.object[...,(self.experimentalData.No//2-n//2):
                    (self.experimentalData.No//2+n//2),(self.experimentalData.No//2-n//2):(self.experimentalData.No//2+n//2)]),
                                  dz[k],self.experimentalData.wavelength,n*self.experimentalData.dxo)

                    # TV approach
                    aleph = 1e-2
                    gradx = imProp-circshift(imProp,[0 1])
                    grady = imProp-circshift(imProp,[1 0])
                    merit.append(np.sum(np.sqrt(abs(gradx)**2+abs(grady)**2+aleph)))

                dz = np.sum(dz*merit)/np.sum(merit)
                c = 10000
                eta = 0.7
                zMomentun = eta*zMomentun+c*dz
                zNew = self.experimentalData.zo+zMomentun

            self.zHistory.append(self.experimentalData.zo) # todo check if it is on GPU, matlab uses gather
            print('position updated: z ='+ str(self.experimentalData.zo*1e3)+'mm (dz = '+
                  str(round(zMomentun*1e7)/10)+'um)')

            # reset coordinates
            self.experimentalData.zo = zNew

            # resample
            if self.propagator!='aspw':
                self.experimentalData.dxo = self.experimentalData.wavelength*self.experimentalData.zo/self.experimentalData.Ld
                self.experimentalData.positions = round(self.experimentalData.encoder/self.experimentalData.dxo)
                self.experimentalData.positions = self.experimentalData.positions+self.experimentalData.No//2\
                                                  -round(self.experimentalData.Np//2)

                # object coordinates
                self.experimentalData.Lo = self.experimentalData.No*self.experimentalData.dxo
                self.experimentalData.xo = np.arange(-self.experimentalData.No//2,self.experimentalData.No//2)\
                                           *self.experimentalData.dxo
                self.experimentalData.Xo, self.experimentalData.Yo = np.meshgrid(self.experimentalData.xo,self.experimentalData.xo)

                # probe coordinates
                self.experimentalData.dxp = self.experimentalData.dxo
                self.experimentalData.Np = self.experimentalData.Nd
                self.experimentalData.Lp = self.experimentalData.Np*self.experimentalData.dxp
                self.experimentalData.xp = np.arange(-self.experimentalData.Np // 2, self.experimentalData.Np // 2) \
                                           * self.experimentalData.dxp
                self.experimentalData.Xp, self.experimentalData.Yp = np.meshgrid(self.experimentalData.xp,self.experimentalData.xp)

                # reset propagator
                self.optimizable.quadraticPhase = np.exp(1.j * np.pi/(self.experimentalData.wavelength * self.experimentalData.zo)
                                                         * (self.experimentalData.Xp**2 + self.experimentalData.Yp**2))

            for positionLoop, positionIndex in enumerate(self.positionIndices):
                ### patch1 ###
                # get object patch
                row, col = self.optimizable.positions[positionIndex]
                sy = slice(row, row + self.experimentalData.Np)
                sx = slice(col, col + self.experimentalData.Np)
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

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)

        frac = self.optimizable.probe.conj() / xp.max(xp.sum(xp.abs(self.optimizable.probe) ** 2, axis=(0, 1, 2, 3)))
        return objectPatch + self.betaObject * xp.sum(frac * DELTA, axis=(0, 2, 3), keepdims=True)

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = objectPatch.conj() / xp.max(xp.sum(xp.abs(objectPatch) ** 2, axis=(0, 1, 2, 3)))
        r = self.optimizable.probe + self.betaObject * xp.sum(frac * DELTA, axis=(0, 1, 3), keepdims=True)
        if self.absorbingProbeBoundary:
            aleph = 1e-3
            r = (1 - aleph) * r + aleph * r * self.probeWindow
        return r


class ePIE_GPU(ePIE):
    """
    GPU-based implementation of ePIE
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cp is None:
            raise ImportError('Could not import cupy')
        self.logger = logging.getLogger('ePIE_GPU')
        self.logger.info('Hello from ePIE_GPU')

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
        # self.optimizable.Imeasured = cp.array(self.optimizable.Imeasured)

        # ePIE parameters
        self.logger.info('Detector error shape: %s', self.detectorError.shape)
        self.detectorError = cp.array(self.detectorError)


