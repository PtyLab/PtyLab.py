import numpy as np
from matplotlib import pyplot as plt
import tqdm
from fracPy.utils.visualisation import hsvplot

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


class zPIE(BaseReconstructor):

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(optimizable, experimentalData, monitor)
        self.logger = logging.getLogger('zPIE')
        self.logger.info('Sucesfully created zPIE zPIE_engine')
        self.logger.info('Wavelength attribute: %s', self.optimizable.wavelength)
        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.DoF = self.experimentalData.DoF.copy()
        self.zPIEgradientStepSize = 100  #gradient step size for axial position correction (typical range [1, 100])
        self.zPIEfriction = 0.7

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
        xp = getArrayModule(self.optimizable.object)

        # actual reconstruction zPIE_engine
        if not hasattr(self, 'zHistory'):
            self.zHistory = []

        zMomentun = 0
        # preallocate grids
        if self.propagator == 'ASP':
            n = self.experimentalData.Np.copy()
        else:
            n = 2*self.experimentalData.Np

        X,Y = xp.meshgrid(xp.arange(-n//2, n//2), xp.arange(-n//2, n//2))
        w = xp.exp(-(xp.sqrt(X**2+Y**2)/self.experimentalData.Np)**4)

        self.pbar = tqdm.trange(self.numIterations, desc='zPIE', leave=True)  # in order to change description to the tqdm progress bar
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()

            # get positions
            if loop == 1:
                zNew = self.experimentalData.zo.copy()
            else:
                d = 10
                dz = np.linspace(-d * self.DoF, d * self.DoF, 11)/10
                merit = []
                # todo, mixed states implementation, check if more need to be put on GPU to speed up
                for k in np.arange(len(dz)):
                    imProp, _ = aspw(
                        w * xp.squeeze(self.optimizable.object[..., (self.experimentalData.No // 2 - n // 2):
                                                                    (self.experimentalData.No // 2 + n // 2),
                                       (self.experimentalData.No // 2 - n // 2):(
                                                   self.experimentalData.No // 2 + n // 2)]),
                        dz[k], self.experimentalData.wavelength, n * self.experimentalData.dxo)

                    # TV approach
                    aleph = 1e-2
                    gradx = imProp-xp.roll(imProp, 1, axis=1)
                    grady = imProp-xp.roll(imProp, 1, axis=0)
                    merit.append(asNumpyArray(xp.sum(xp.sqrt(abs(gradx)**2+abs(grady)**2+aleph))))

                feedback = np.sum(dz*merit)/np.sum(merit)    # at optimal z, feedback term becomes 0
                zMomentun = self.zPIEfriction*zMomentun+self.zPIEgradientStepSize*feedback
                zNew = self.experimentalData.zo+zMomentun

            self.zHistory.append(self.experimentalData.zo)

            # print updated z
            self.pbar.set_description('zPIE: update z = %.3f mm (dz = %.1f um)' % (self.experimentalData.zo*1e3, zMomentun*1e6))

            # reset coordinates
            self.experimentalData.zo = zNew

            # re-sample is automatically done by using @property
            if self.propagator != 'ASP':
                self.experimentalData.dxp = self.experimentalData.wavelength*self.experimentalData.zo\
                                            /self.experimentalData.Ld
                # reset propagator
                self.optimizable.quadraticPhase = xp.array(np.exp(1.j * np.pi/(self.experimentalData.wavelength * self.experimentalData.zo)
                                                                  * (self.experimentalData.Xp**2 + self.experimentalData.Yp**2)))

            for positionLoop, positionIndex in enumerate(self.positionIndices):
                ### patch1 ###
                # get object patch
                row, col = self.experimentalData.positions[positionIndex]
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
            if loop == 0:
                figure, ax = plt.subplots(1, 1, num=666, squeeze=True, clear=True, figsize=(5, 5))
                ax.set_title('Estimated distance (object-camera)')
                ax.set_xlabel('iteration')
                ax.set_ylabel('estimated z (mm)')
                ax.set_xscale('symlog')
                line = plt.plot(0, zNew, 'o-')[0]
                plt.tight_layout()
                plt.show(block=False)

            elif np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
                idx = np.linspace(0, np.log10(len(self.zHistory)-1), np.minimum(len(self.zHistory), 100))
                idx = np.rint(10**idx).astype('int')

                line.set_xdata(idx)
                line.set_ydata(np.array(self.zHistory)[idx]*1e3)
                ax.set_xlim(0, np.max(idx))
                ax.set_ylim(np.min(self.zHistory)*1e3, np.max(self.zHistory)*1e3)

                figure.canvas.draw()
                figure.canvas.flush_events()
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
        r = self.optimizable.probe + self.betaProbe * xp.sum(frac * DELTA, axis=(0, 1, 3), keepdims=True)
        return r

class zPIE_GPU(zPIE):
    """
    GPU-based implementation of zPIE
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cp is None:
            raise ImportError('Could not import cupy')
        self.logger = logging.getLogger('zPIE_GPU')
        self.logger.info('Hello from zPIE_GPU')

    def _prepare_doReconstruction(self):
        self.logger.info('Ready to start transferring stuff to the GPU')
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
        # self.experimentalData.probe = cp.array(self.experimentalData.probe, cp.complex64)
        # self.optimizable.Imeasured = cp.array(self.optimizable.Imeasured)

        # zPIE parameters
        self.logger.info('Detector error shape: %s', self.detectorError.shape)
        self.detectorError = cp.array(self.detectorError)

        # proapgators to GPU
        if self.propagator == 'Fresnel':
            self.optimizable.quadraticPhase = cp.array(self.optimizable.quadraticPhase)
        elif self.propagator == 'ASP' or self.propagator == 'polychromeASP':
            self.optimizable.transferFunction = cp.array(self.optimizable.transferFunction)
        elif self.propagator == 'scaledASP' or self.propagator == 'scaledPolychromeASP':
            self.optimizable.Q1 = cp.array(self.optimizable.Q1)
            self.optimizable.Q2 = cp.array(self.optimizable.Q2)

        # other parameters
        if self.backgroundModeSwitch:
            self.background = cp.array(self.background)
        if self.absorbingProbeBoundary:
            self.probeWindow = cp.array(self.probeWindow)

