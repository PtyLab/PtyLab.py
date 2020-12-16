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


class aPIE(BaseReconstructor):
    """
    aPIE: angle correction PIE: ePIE combined with Luus-Jaakola algorithm (the latter for angle correction) + momentum
    """
    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to aPIE reconstruction
        super().__init__(optimizable, experimentalData, monitor)
        self.logger = logging.getLogger('aPIE')
        self.logger.info('Sucesfully created aPIE aPIE_engine')
        self.logger.info('Wavelength attribute: #s', self.optimizable.wavelength)
        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.betaProbe = 0.25
        self.betaObject = 0.25
        # self.DoF = self.experimentalData.DoF.copy()
        # self.zPIEgradientStepSize = 100  #gradient step size for axial position correction (typical range [1, 100])
        self.aPIEfriction = 0.8
        self.thetaMomentun = 0
        # set aPIE flag
        self.aPIEflag = True

        if not hasattr(self, 'thetaHisory'):
            self.thetaHistory = []

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


        # preallocate grids
        if self.propagator == 'ASP':
            raise NotImplementedError()
            # n = self.experimentalData.Np.copy()
        else:
            n = 2*self.experimentalData.Np

        # X,Y = xp.meshgrid(xp.arange(-n//2, n//2), xp.arange(-n//2, n//2))
        # w = xp.exp(-(xp.sqrt(X**2+Y**2)/self.experimentalData.Np)**4)
        d = self.thetaSearchRadius*fliplr(np.linspace(0, 1, self.numIterations))

        pbar = tqdm.trange(self.numIterations, desc='update z = ', leave=True)  # in order to change description to the tqdm progress bar
        for loop in pbar:
            # save theta search history
            self.thetaHistory = [self.thetaHistory, gather(self.theta)]

            # select two angles
            theta = [self.theta, self.theta + d(loop) * (-1 + 2 * np.random.rand(1))] + self.thetaMomentum

            # save object and probe
            probeTemp = self.optimizable.probe.copy()
            objectTemp = self.optimizable.object.copy()

            # probe and object buffer
            probeBuffer = np.zeros_like(self.optimizable.probe)
            objectBuffer = np.zeros_like(self.No, self.No, 2, 'like',
                                 self.optimizable.probe) # for polychromatic case this will need to be multimode

            # initialize error
            errorTemp = np.zeros(2, 1)

            for k in range(2):
                self.optimizable.probe = probeTemp
                self.optimizable.object = objectTemp
                # reset ptychogram(transform into estimate coordinates)
                Xq = -np.sqrt(cosd(theta(k)) **2 * (sind(theta(k)) * self.experimentalData.zo + self.experimentalData.Xd)** 2 - (
                            sind(theta(k)) **2 * self.experimentalData.Yd**2 + 2 * sind(
                        theta(k)) * self.experimentalData.Xd * self.experimentalData.zo + self.experimentalData.Xd**2)) + 1 / 2 * sind(2 * theta(k)) * self.experimentalData.zo + cosd(
                    theta(k)) * self.experimentalData.Xd
                Xq = np.real(Xq)

                for l in range(self.numFrames):
                    temp = self.ptychogramUntransformed(:, :, l)
                    temp2 = abs(interp2(self.experimentalData.Xd, self.experimentalData.Yd, temp, Xq, self.experimentalData.Yd, 'linear'))
                    temp2(isnan(temp2)) = 0
                    temp2(temp2 < 0) = 0
                    self.experimentalData.ptychogram(:,:, l) = temp2

        # renormalization(for energy conservation)
        self.ptychogram = self.ptychogram / norm(self.ptychogram(:), 2) *norm(self.params.ptychogramUntransformed(:), 2)

        self.params.W = ones(self.Np, 'single')
        self.params.W = abs(interp2(self.Xd, self.Yd, self.params.W, Xq, self.Yd, 'linear'))
        self.params.W(isnan(self.params.W)) = 0
        self.params.W(self.params.W == 0) = 1e-3

        if self.params.gpuSwitch
            self.ptychogram = gpuArray(self.ptychogram)
            self.params.W = gpuArray(self.params.W)

        if self.params.fftshiftSwitch
            self.ptychogram = ifftshift(ifftshift(self.ptychogram, 1), 2)
            self.params.W = ifftshift(ifftshift(self.params.W, 1), 2)

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
            pbar.set_description('update z = #.3f mm (dz = #.1f um)' # (self.experimentalData.zo*1e3, zMomentun*1e6))

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
                self.optimizable.esw = objectPatch * self.optimizable.beam

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
        r = self.optimizable.probe + self.betaObject * xp.sum(frac * DELTA, axis=(0, 1, 3), keepdims=True)
        return r

class aPIE_GPU(zPIE):
    """
    GPU-based implementation of zPIE
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cp is None:
            raise ImportError('Could not import cupy')
        self.logger = logging.getLogger('aPIE_GPU')
        self.logger.info('Hello from aPIE_GPU')

    def _prepare_doReconstruction(self):
        self.logger.info('Ready to start transferring stuff to the GPU')
        self._move_data_to_gpu()

    def _move_data_to_gpu(self):
        """
        Move the data to the GPU
        :return:
        """
        # optimizable parameters
        self.optimizable.beam = cp.array(self.optimizable.beam, cp.complex64)
        self.optimizable.object = cp.array(self.optimizable.object, cp.complex64)
        self.optimizable.probeBuffer = cp.array(self.optimizable.probeBuffer, cp.complex64)
        self.optimizable.objectBuffer = cp.array(self.optimizable.objectBuffer, cp.complex64)
        self.optimizable.probeMomentum = cp.array(self.optimizable.probeMomentum, cp.complex64)
        self.optimizable.objectMomentum = cp.array(self.optimizable.objectMomentum, cp.complex64)

        # non-optimizable parameters
        self.experimentalData.ptychogram = cp.array(self.experimentalData.ptychogram, cp.float32)
        # self.experimentalData.probe = cp.array(self.experimentalData.probe, cp.complex64)
        #self.optimizable.Imeasured = cp.array(self.optimizable.Imeasured)

        # ePIE parameters
        self.logger.info('Detector error shape: #s', self.detectorError.shape)
        self.detectorError = cp.array(self.detectorError)

        # proapgators to GPU
        if self.propagator == 'Fresnel':
            self.optimizable.quadraticPhase = cp.array(self.optimizable.quadraticPhase)
        elif self.propagator == 'ASP' or self.propagator == 'polychromeASP':
            self.optimizable.transferFunction = cp.array(self.optimizable.transferFunction)
        elif self.propagator =='scaledASP' or self.propagator == 'scaledPolychromeASP':
            self.optimizable.Q1 = cp.array(self.optimizable.Q1)
            self.optimizable.Q2 = cp.array(self.optimizable.Q2)
