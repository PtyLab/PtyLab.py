# momentum-based multi PIE
import numpy as np
import tqdm
from matplotlib import pyplot as plt

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
from fracPy.utils.utils import fft2c, ifft2c
import logging


class multiPIE(BaseReconstructor):

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(optimizable, experimentalData, monitor)
        self.logger = logging.getLogger('multiPIE')
        self.logger.info('Sucesfully created multiPIE multiPIE_engine')

        self.logger.info('Wavelength attribute: %s', self.optimizable.wavelength)

        self.initializeReconstructionParams()

        # todo check if this is the same as mPIEoperation
        self.optimizable.initializeObjectMomentum()
        self.optimizable.initializeProbeMomentum()
        self.optimizable.objectBuffer = self.optimizable.object.copy()
        self.optimizable.probeBuffer = self.optimizable.probe.copy()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the mPIE settings.
        :return:
        """
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.beta = 0.5   # feedback
        self.eta = 0.9  # friction
        self.t = 0

        # modulus enforced probe
        if self.modulusEnforcedProbeSwitch:
            self.modulusEnforcedProbe()

        # initialize spectral weights
        self.spectralDensityWeights = np.sum(abs(self.optimizable.probe)**2, axis=(-1, -2))

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
        # actual reconstruction multiPIE_engine
        for loop in tqdm.tqdm(range(self.numIterations)):
            # set position order
            self.setPositionOrder()
            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # time increment
                self.t = self.t+1
                # get object patch
                row, col = self.optimizable.positions[positionIndex]
                sy = slice(row, row + self.experimentalData.Np)
                sx = slice(col, col + self.experimentalData.Np)
                # note that object patch has size of probe array
                objectPatch = self.optimizable.object[..., sy, sx].copy()

                # make exit surface wave
                self.optimizable.esw = objectPatch * self.optimizable.probe
                # normalize exit wave
                self.optimizable.esw = self.optimizable.esw * xp.sqrt(self.spectralDensity) \
                                       / xp.sqrt(sum(self.spectralDensity, axis=0))

                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.optimizable.eswUpdate - self.optimizable.esw

                # object update
                self.optimizable.object[..., sy, sx] = self.objectPatchUpdate(objectPatch, DELTA)

                # probe update
                self.optimizable.probe = self.probeUpdate(objectPatch, DELTA)

                # update spectral density
                self.spectralDensityWeights= xp.sum(abs(self.optimizable.probe)**2, axis=(-1, -2))

                # momentum updates
                if len(self.optimizable.error) > 2*max(self.objectUpdateStart, self.probeUpdateStart) \
                        and np.random.rand(1) > 0.99:
                    self.objectMomentumUpdate()
                    self.probeMomentumUpdate()
                elif len(self.optimizable.error) < 2*max(self.objectUpdateStart, self.probeUpdateStart):
                    self.optimizable.objectBuffer = self.optimizable.object.copy()
                    self.optimizable.probeButter = self.optimizable.probe.copy()

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # modulus enforced probe todo check if the switch is necessary
            if self.modulusEnforcedProbeSwitch:
                self.modulusEnforcedProbe()

            # show reconstruction
            self.showReconstruction(loop)



    def momentumUpdate(self, array, buffer, momentum):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        gradient = buffer - array
        momentum = gradient + self.betaM * momentum
        array = array - self.stepM * momentum
        buffer = array.copy()
        return array, buffer, momentum

    def objectMomentumUpdate(self):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        gradient = self.optimizable.objectBuffer - self.optimizable.object
        self.optimizable.objectMomentum = gradient + self.eta * self.optimizable.objectMomentum
        self.optimizable.object = self.optimizable.object - self.beta * self.optimizable.objectMomentum
        self.optimizable.objectBuffer = self.optimizable.object.copy()

    def probeMomentumUpdate(self):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        gradient = self.optimizable.probeBuffer - self.optimizable.probe
        self.optimizable.probeMomentum = gradient + self.eta * self.optimizable.probeMomentum
        self.optimizable.probe = self.optimizable.probe - self.beta * self.optimizable.probeMomentum
        self.optimizable.probeBuffer = self.optimizable.probe.copy()

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        ePIE-type updat
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)

        frac = self.optimizable.probe.conj() / xp.max(xp.sum(xp.abs(self.optimizable.probe) ** 2, axis=(1, 2, 3)), axis=(-1,-2))
        return objectPatch + self.betaObject * frac * DELTA


def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        ePIE-type update
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)

        frac = objectPatch.conj() / xp.max(xp.sum(xp.abs(objectPatch) ** 2, axis=(1, 2, 3)), axis=(-1,-2))
        r = self.optimizable.probe + self.betaProbe * frac * DELTA
        return r


class multiPIE_GPU(multiPIE):
    """
    GPU-based implementation of multiPIE
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cp is None:
            raise ImportError('Could not import cupy')
        self.logger = logging.getLogger('multiPIE_GPU')
        self.logger.info('Hello from multiPIE_GPU')

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
        self.optimizable.probeBuffer = cp.array(self.optimizable.probeBuffer, cp.complex64)
        self.optimizable.objectBuffer = cp.array(self.optimizable.objectBuffer, cp.complex64)
        self.optimizable.probeMomentum = cp.array(self.optimizable.probeMomentum, cp.complex64)
        self.optimizable.objectMomentum = cp.array(self.optimizable.objectMomentum, cp.complex64)

        # non-optimizable parameters
        self.experimentalData.ptychogram = cp.array(self.experimentalData.ptychogram, cp.float32)
        self.experimentalData.probe = cp.array(self.experimentalData.probe, cp.complex64)
        # self.optimizable.Imeasured = cp.array(self.optimizable.Imeasured)

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



