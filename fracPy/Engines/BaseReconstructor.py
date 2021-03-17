from fracPy.Monitors.default_visualisation import DefaultMonitor,DiffractionDataMonitor
import numpy as np
from scipy.signal import get_window
import logging
import warnings
import h5py
# fracPy imports
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.utils.initializationFunctions import initialProbeOrObject
from fracPy.FixedData.DefaultExperimentalData import ExperimentalData
from fracPy.Optimizables.Optimizable import Optimizable
from fracPy.Params.ReconstructionParameters import Reconstruction_parameters
from fracPy.utils.utils import ifft2c, fft2c, orthogonalizeModes, circ, p2bin
from fracPy.operators.operators import aspw, scaledASP
from fracPy.Monitors.Monitor import Monitor
from fracPy.utils.visualisation import hsvplot
from matplotlib import pyplot as plt
import cupy as cp
from skimage.transform import rescale


class BaseReconstructor(object):
    """
    Common properties that are common for all reconstruction Engines are defined here.

    Unless you are testing the code, there's hardly any need to create this object. For your own implementation,
    inherit from this object

    """
    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, params: Reconstruction_parameters, monitor: Monitor):
        # These statements don't copy any data, they just keep a reference to the object
        self.optimizable = optimizable
        self.experimentalData = experimentalData
        self.params = params
        self.monitor = monitor
        self.monitor.optimizable = optimizable

        # datalogger
        self.logger = logging.getLogger('BaseReconstructor')


    def _prepareReconstruction(self):
        """
        Initialize everything that depends on user changeable attributes.
        :return:
        """
        # check miscellaneous quantities specific for certain Engines
        self._checkMISC()
        self._checkFFT()
        self._initializeQuadraticPhase()
        self._initialProbePowerCorrection()
        self._probeWindow()
        self._initializeErrors()
        self._setObjectProbeROI()
        self._showInitialGuesses()
        self._initializePCParameters()
        self._checkGPU()  # checkGPU needs to be the last

    def _initializePCParameters(self):
        if self.params.positionCorrectionSwitch:
            # additional pcPIE parameters as they appear in Matlab
            self.daleth = 0.5  # feedback
            self.beth = 0.9  # friction
            self.adaptStep = 1  # adaptive step size
            self.D = np.zeros((self.experimentalData.numFrames, 2))  # position search direction
            # predefine shifts
            self.rowShifts = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
            self.colShifts = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
            self.startAtIteration = 20
            self.meanEncoder00 = np.mean(self.experimentalData.encoder[:, 0]).copy()
            self.meanEncoder01 = np.mean(self.experimentalData.encoder[:, 1]).copy()

    def _initializeErrors(self):
        """
        initialize all kinds of errors:
        detectorError is a matrix calculated at each iteration (numFrames,Nd,Nd);
        errorAtPos sums over detectorError at each iteration, (numFrames,1);
        optimizable.error sums over errorAtPos, one number at each iteration;
        """
        # initialize detector error matrices
        if self.params.saveMemory:
            self.optimizable.detectorError = 0
        else:
            if not hasattr(self.optimizable, 'detectorError'):
                self.optimizable.detectorError = np.zeros((self.experimentalData.numFrames,
                                               self.optimizable.Nd, self.optimizable.Nd))
        # initialize energy at each scan position
        if not hasattr(self.optimizable, 'errorAtPos'):
            self.optimizable.errorAtPos = np.zeros((self.experimentalData.numFrames, 1), dtype=np.float32)
        # initialize final error
        if not hasattr(self.optimizable, 'error'):
            self.optimizable.error = []

    def _initialProbePowerCorrection(self):
        if self.params.probePowerCorrectionSwitch:
            self.optimizable.probe = self.optimizable.probe/np.sqrt(
                np.sum(self.optimizable.probe*self.optimizable.probe.conj()))*self.experimentalData.maxProbePower

    def _probeWindow(self):
        # absorbing probe boundary: filter probe with super-gaussian window function
        if not self.params.saveMemory or self.params.absorbingProbeBoundary:
            self.probeWindow = np.exp(-((self.optimizable.Xp**2+self.optimizable.Yp**2)/
                                        (2*(3/4*self.optimizable.Np*self.optimizable.dxp/2.355)**2))**10)

        if self.params.probeBoundary:
            self.probeWindow = circ(self.optimizable.Xp, self.optimizable.Yp,
                                    self.experimentalData.entrancePupilDiameter + self.experimentalData.entrancePupilDiameter*0.2)

    def _setObjectProbeROI(self):
        """
        Set object/probe ROI for monitoring
        """
        if not hasattr(self.optimizable, 'objectROI'):
            rx,ry = ((np.max(self.optimizable.positions, axis=0)-np.min(self.optimizable.positions, axis=0)\
                    +self.optimizable.Np)/self.monitor.objectPlotZoom).astype(int)
            xc,yc = ((np.max(self.optimizable.positions, axis=0)+np.min(self.optimizable.positions, axis=0)\
                    +self.optimizable.Np)/2).astype(int)

            self.optimizable.objectROI = [slice(max(0, yc-ry//2),
                                                min(self.optimizable.No, yc + ry//2)),
                                          slice(max(0, xc - rx // 2),
                                                min(self.optimizable.No, xc + rx//2))]

        if not hasattr(self.optimizable, 'probeROI'):
            r = np.int(self.experimentalData.entrancePupilDiameter/self.optimizable.dxp/self.monitor.probePlotZoom)
            self.optimizable.probeROI = [slice(max(0, self.optimizable.Np//2-r),
                                               min(self.optimizable.Np, self.optimizable.Np//2+r)),
                                         slice(max(0, self.optimizable.Np // 2 - r),
                                               min(self.optimizable.Np, self.optimizable.Np//2+r))]

    def _showInitialGuesses(self):
        self.monitor.initializeVisualisation()
        object_estimate = np.squeeze(self.optimizable.object
                                     [..., self.optimizable.objectROI[0], self.optimizable.objectROI[1]])
        probe_estimate = np.squeeze(self.optimizable.probe
                                    [..., self.optimizable.probeROI[0], self.optimizable.probeROI[1]])
        self.monitor.updateDefaultMonitor(object_estimate=object_estimate, probe_estimate=probe_estimate)

    def _initializeQuadraticPhase(self):
        """
        # initialize quadraticPhase term or transferFunctions used in propagators
        """
        if self.params.propagator == 'Fresnel':
            self.optimizable.quadraticPhase = np.exp(1.j*np.pi/(self.optimizable.wavelength*self.optimizable.zo)
                                                     *(self.optimizable.Xp**2+self.optimizable.Yp**2))
        elif self.params.propagator == 'ASP':
            if self.params.fftshiftSwitch:
                raise ValueError('ASP propagator works only with fftshiftSwitch = False!')
            if self.optimizable.nlambda>1:
                raise ValueError('For multi-wavelength, polychromeASP needs to be used instead of ASP')

            dummy = np.ones((1, self.optimizable.nosm, self.optimizable.npsm,
                                      1, self.optimizable.Np, self.optimizable.Np), dtype='complex64')
            self.optimizable.transferFunction = np.array(
                [[[[aspw(dummy[nlambda, nosm, npsm, nslice, :, :],
                        self.optimizable.zo, self.optimizable.wavelength,
                        self.optimizable.Lp)[1]
                    for nslice in range(1)]
                   for npsm in range(self.optimizable.npsm)]
                  for nosm in range(self.optimizable.nosm)]
                 for nlambda in range(self.optimizable.nlambda)])

        elif self.params.propagator == 'polychromeASP':
            dummy = np.ones((self.optimizable.nlambda, self.optimizable.nosm, self.optimizable.npsm,
                                      1, self.optimizable.Np, self.optimizable.Np), dtype='complex64')
            self.optimizable.transferFunction = np.array(
                [[[[aspw(dummy[nlambda, nosm, npsm, nslice, :, :],
                         self.optimizable.zo, self.optimizable.spectralDensity[nlambda],
                         self.optimizable.Lp)[1]
                    for nslice in range(1)]
                   for npsm in range(self.optimizable.npsm)]
                  for nosm in range(self.optimizable.nosm)]
                 for nlambda in range(self.optimizable.nlambda)])
            if self.params.fftshiftSwitch:
                raise ValueError('ASP propagator works only with fftshiftSwitch = False!')

        elif self.params.propagator =='scaledASP':
            if self.params.fftshiftSwitch:
                raise ValueError('scaledASP propagator works only with fftshiftSwitch = False!')
            if self.optimizable.nlambda > 1:
                raise ValueError('For multi-wavelength, scaledPolychromeASP needs to be used instead of scaledASP')
            dummy = np.ones((1, self.optimizable.nosm, self.optimizable.npsm,
                                      1, self.optimizable.Np, self.optimizable.Np), dtype='complex64')
            self.optimizable.Q1 = np.ones_like(dummy)
            self.optimizable.Q2 = np.ones_like(dummy)
            for nosm in range(self.optimizable.nosm):
                for npsm in range(self.optimizable.npsm):
                    _, self.optimizable.Q1[0,nosm,npsm,0,...], self.optimizable.Q2[0,nosm,npsm,0,...] = scaledASP(
                        dummy[0, nosm, npsm, 0, :, :], self.optimizable.zo, self.optimizable.wavelength,
                        self.optimizable.dxo, self.optimizable.dxd)

        # todo check if Q1 Q2 are bandlimited

        elif self.params.propagator == 'scaledPolychromeASP':
            if self.params.fftshiftSwitch:
                raise ValueError('scaledPolychromeASP propagator works only with fftshiftSwitch = False!')
            dummy = np.ones((self.optimizable.nlambda, self.optimizable.nosm, self.optimizable.npsm,
                                          1, self.optimizable.Np, self.optimizable.Np), dtype='complex64')
            self.optimizable.Q1 = np.ones_like(dummy)
            self.optimizable.Q2 = np.ones_like(dummy)
            for nlmabda in range(self.optimizable.nlambda):
                for nosm in range(self.optimizable.nosm):
                    for npsm in range(self.optimizable.npsm):
                        _, self.optimizable.Q1[nlmabda, nosm, npsm, 0, ...], self.optimizable.Q2[
                            nlmabda, nosm, npsm, 0, ...] = scaledASP(
                            dummy[nlmabda, nosm, npsm, 0, :, :], self.optimizable.zo,
                            self.optimizable.spectralDensity[nlmabda], self.optimizable.dxo,
                            self.optimizable.dxd)

        elif self.params.propagator == 'twoStepPolychrome':
            if self.params.fftshiftSwitch:
                raise ValueError('twoStepPolychrome propagator works only with fftshiftSwitch = False!')
            dummy = np.ones((self.optimizable.nlambda, self.optimizable.nosm, self.optimizable.npsm,
                             1, self.optimizable.Np, self.optimizable.Np), dtype='complex64')
            # self.optimizable.quadraticPhase = np.ones_like(dummy)
            self.optimizable.transferFunction = np.array(
                [[[[aspw(dummy[nlambda, nosm, npsm, nslice, :, :],
                         self.optimizable.zo *
                            (1-self.optimizable.spectralDensity[0]/self.optimizable.spectralDensity[nlambda]),
                         self.optimizable.spectralDensity[nlambda],
                         self.optimizable.Lp)[1]
                    for nslice in range(1)]
                   for npsm in range(self.optimizable.npsm)]
                  for nosm in range(self.optimizable.nosm)]
                 for nlambda in range(self.optimizable.nlambda)])
            self.optimizable.quadraticPhase = np.exp(
                1.j * np.pi / (self.optimizable.spectralDensity[0] * self.optimizable.zo)
                * (self.optimizable.Xp ** 2 + self.optimizable.Yp ** 2))

    def _checkMISC(self):
        """
        checks miscellaneous quantities specific certain Engines
        """
        if self.params.backgroundModeSwitch:
            self.optimizable.background = 1e-1*np.ones((self.optimizable.Np, self.optimizable.Np))

        # preallocate intensity scaling vector
        if self.params.intensityConstraint == 'fluctuation':
            self.intensityScaling = np.ones(self.experimentalData.numFrames)

        if self.params.intensityConstraint == 'interferometric':
            self.optimizable.reference = np.ones(self.experimentalData.ptychogram[0].shape)

        # todo check if there is data on gpu that shouldnt be there

        # check if both probePoprobePowerCorrectionSwitch and modulusEnforcedProbeSwitch are on.
        # Since this can cause a contradiction, it raises an error
        if self.params.probePowerCorrectionSwitch and self.params.modulusEnforcedProbeSwitch:
            raise ValueError('probePowerCorrectionSwitch and modulusEnforcedProbeSwitch '
                             'can not simultaneously be switched on!')

        if not self.params.fftshiftSwitch:
            warnings.warn('fftshiftSwitch set to false, this may lead to reduced performance')

        if self.params.propagator == 'ASP' and self.params.fftshiftSwitch:
                raise ValueError('ASP propagator works only with fftshiftSwitch = False')
        if self.params.propagator == 'scaledASP' and self.params.fftshiftSwitch:
                raise ValueError('scaledASP propagator works only with fftshiftSwitch = False')

    def _checkFFT(self):
        """
        shift arrays to accelerate fft
        """
        if self.params.fftshiftSwitch:
            if self.params.fftshiftFlag == 0:
                print('check fftshift...')
                print('fftshift data for fast far-field update')
                # shift detector quantities
                self.experimentalData.ptychogram = np.fft.ifftshift(self.experimentalData.ptychogram, axes=(-1, -2))
                if hasattr(self.optimizable, 'ptychogramDownsampled'):
                    self.optimizable.ptychogramDownsampled = np.fft.ifftshift(
                        self.optimizable.ptychogramDownsampled, axes=(-1, -2))
                if self.experimentalData.W != None:
                    self.experimentalData.W = np.fft.ifftshift(self.experimentalData.W, axes=(-1, -2))
                if self.experimentalData.emptyBeam != None:
                    self.experimentalData.emptyBeam = np.fft.ifftshift(
                        self.experimentalData.emptyBeam, axes=(-1, -2))
                if self.experimentalData.PSD != None:
                    self.experimentalData.PSD = np.fft.ifftshift(
                        self.experimentalData.PSD, axes=(-1, -2))
                self.params.fftshiftFlag = 1
        else:
            if self.params.fftshiftFlag == 1:
                print('check fftshift...')
                print('ifftshift data')
                self.experimentalData.ptychogram = np.fft.fftshift(self.experimentalData.ptychogram, axes=(-1, -2))
                if hasattr(self.optimizable, 'ptychogramDownsampled'):
                    self.optimizable.ptychogramDownsampled = np.fft.fftshift(
                        self.optimizable.ptychogramDownsampled, axes=(-1, -2))
                if self.experimentalData.W != None:
                    self.experimentalData.W = np.fft.fftshift(self.experimentalData.W, axes=(-1, -2))
                if self.experimentalData.emptyBeam != None:
                    self.experimentalData.emptyBeam = np.fft.fftshift(
                        self.experimentalData.emptyBeam, axes=(-1, -2))
                self.params.fftshiftFlag = 0

    def _move_data_to_gpu(self):
        """
        Move the data to the GPU, called when the gpuSwitch is on.
        :return:
        """
        # optimizable parameters
        self.optimizable.probe = cp.array(self.optimizable.probe, cp.complex64)
        self.optimizable.object = cp.array(self.optimizable.object, cp.complex64)

        if self.params.momentumAcceleration:
            self.optimizable.probeBuffer = cp.array(self.optimizable.probeBuffer, cp.complex64)
            self.optimizable.objectBuffer = cp.array(self.optimizable.objectBuffer, cp.complex64)
            self.optimizable.probeMomentum = cp.array(self.optimizable.probeMomentum, cp.complex64)
            self.optimizable.objectMomentum = cp.array(self.optimizable.objectMomentum, cp.complex64)


        # non-optimizable parameters
        self.experimentalData.ptychogram = cp.array(self.experimentalData.ptychogram, cp.float32)
        self.optimizable.detectorError = cp.array(self.optimizable.detectorError)

        # propagators
        if self.params.propagator == 'Fresnel':
            self.optimizable.quadraticPhase = cp.array(self.optimizable.quadraticPhase)
        elif self.params.propagator == 'ASP' or self.params.propagator == 'polychromeASP':
            self.optimizable.transferFunction = cp.array(self.optimizable.transferFunction)
        elif self.params.propagator == 'scaledASP' or self.params.propagator == 'scaledPolychromeASP':
            self.optimizable.Q1 = cp.array(self.optimizable.Q1)
            self.optimizable.Q2 = cp.array(self.optimizable.Q2)
        elif self.params.propagator =='twoStepPolychrome':
            self.optimizable.quadraticPhase = cp.array(self.optimizable.quadraticPhase)
            self.optimizable.transferFunction = cp.array(self.optimizable.transferFunction)

        # other parameters
        if self.params.backgroundModeSwitch:
            self.background = cp.array(self.optimizable.background)
        if self.params.absorbingProbeBoundary or self.params.probeBoundary:
            self.probeWindow = cp.array(self.probeWindow)
        if self.params.modulusEnforcedProbeSwitch:
            self.experimentalData.emptyBeam = cp.array(self.experimentalData.emptyBeam)
        if self.params.intensityConstraint == 'interferometric':
            self.optimizable.reference = cp.array(self.optimizable.reference)


    def _move_data_to_cpu(self):
        """
        Move the data to the CPU, called when the gpuSwitch is off.
        :return:
        """
        # optimizable parameters
        self.optimizable.probe = self.optimizable.probe.get()
        self.optimizable.object = self.optimizable.object.get()

        if self.params.momentumAcceleration:
            self.optimizable.probeBuffer = self.optimizable.probeBuffer.get()
            self.optimizable.objectBuffer = self.optimizable.objectBuffer.get()
            self.optimizable.probeMomentum = self.optimizable.probeMomentum.get()
            self.optimizable.objectMomentum = self.optimizable.objectMomentum.get()

        # non-optimizable parameters
        self.experimentalData.ptychogram = self.experimentalData.ptychogram.get()
        self.optimizable.detectorError = self.optimizable.detectorError.get()

        # propagators
        if self.params.propagator == 'Fresnel':
            self.optimizable.quadraticPhase = self.optimizable.quadraticPhase.get()
        elif self.params.propagator == 'ASP' or self.params.propagator == 'polychromeASP':
            self.optimizable.transferFunction = self.optimizable.transferFunction.get()
        elif self.params.propagator == 'scaledASP' or self.params.propagator == 'scaledPolychromeASP':
            self.optimizable.Q1 = self.optimizable.Q1.get()
            self.optimizable.Q2 = self.optimizable.Q2.get()
        elif self.params.propagator =='twoStepPolychrome':
            self.optimizable.quadraticPhase = self.optimizable.quadraticPhase.get()
            self.optimizable.transferFunction = self.optimizable.transferFunction.get()

        # other parameters
        if self.params.backgroundModeSwitch:
            self.optimizable.background = self.optimizable.background.get()
        if self.params.absorbingProbeBoundary or self.params.probeBoundary:
            self.probeWindow = self.probeWindow.get()
        if self.params.modulusEnforcedProbeSwitch:
            self.experimentalData.emptyBeam = self.experimentalData.emptyBeam.get()
        if self.params.intensityConstraint == 'interferometric':
            self.optimizable.reference = self.optimizable.reference.get()

    def _checkGPU(self):
        if not hasattr(self.params, 'gpuFlag'):
            self.params.gpuFlag = 0

        if self.params.gpuSwitch:
            if cp is None:
                raise ImportError('Could not import cupy, turn gpuSwitch to false, perform CPU reconstruction')
            if not self.params.gpuFlag:
                self.logger.info('switch to gpu')

                # clear gpu to prevent memory issues todo

                # load data to gpu
                self._move_data_to_gpu()
                self.params.gpuFlag = 1
        else:
            if self.params.gpuFlag:
                self.logger.info('switch to cpu')
                self._move_data_to_cpu()
                self.params.gpuFlag = 0

    def setPositionOrder(self):
        if self.params.positionOrder == 'sequential':
            self.positionIndices = np.arange(self.experimentalData.numFrames)

        elif self.params.positionOrder == 'random':
            if len(self.optimizable.error) == 0:
                self.positionIndices = np.arange(self.experimentalData.numFrames)
            else:
                if len(self.optimizable.error) < 2:
                    self.positionIndices = np.arange(self.experimentalData.numFrames)
                else:
                    self.positionIndices = np.arange(self.experimentalData.numFrames)
                    np.random.shuffle(self.positionIndices)
        
        # order by illumiantion angles. Use smallest angles first
        # (i.e. start with brightfield data first, then add the low SNR
        # darkfield)
        # todo check this with Antonios
        elif self.params.positionOrder == 'NA':
            rows = self.optimizable.positions[:, 0] - np.mean(self.optimizable.positions[:, 0])
            cols = self.optimizable.positions[:, 1] - np.mean(self.optimizable.positions[:, 1])
            dist = np.sqrt(rows**2 + cols**2)
            self.positionIndices = np.argsort(dist)
        else:
            raise ValueError('position order not properly set')

    def changeExperimentalData(self, experimentalData:ExperimentalData):
        self.experimentalData = experimentalData

    def changeOptimizable(self, optimizable: Optimizable):
        self.optimizable = optimizable

    def convert2single(self):
        """
        Convert the datasets to single precision. Matches: convert2single.m
        :return:
        """
        self.dtype_complex = np.complex64
        self.dtype_real = np.float32
        self._match_dtypes_complex()
        self._match_dtypes_real()

    def _match_dtypes_complex(self):
        raise NotImplementedError()

    def _match_dtypes_real(self):
        raise NotImplementedError()

    def object2detector(self):
        """
        Implements object2detector.m
        :return:
        """
        if self.params.propagator == 'Fraunhofer':
            self.fft2s()
        elif self.params.propagator == 'Fresnel':
            self.optimizable.esw = self.optimizable.esw * self.optimizable.quadraticPhase
            self.fft2s()
        elif self.params.propagator == 'ASP' or self.params.propagator == 'polychromeASP':
            self.optimizable.ESW = ifft2c(fft2c(self.optimizable.esw) * self.optimizable.transferFunction)
        elif self.params.propagator == 'scaledASP' or self.params.propagator == 'scaledPolychromeASP':
            self.optimizable.ESW = ifft2c(fft2c(self.optimizable.esw * self.optimizable.Q1) * self.optimizable.Q2)
        elif self.params.propagator == 'twoStepPolychrome':
            self.optimizable.esw = ifft2c(fft2c(self.optimizable.esw) * self.optimizable.transferFunction) * \
                self.optimizable.quadraticPhase
            self.fft2s()
        else:
            raise Exception('Propagator is not properly set, choose from Fraunhofer, Fresnel, ASP and scaledASP')


    def detector2object(self):
        """
        Propagate the ESW to the object plane (in-place).

        Matches: detector2object.m
        :return:
        """
        if self.params.propagator == 'Fraunhofer':
            self.ifft2s()
        elif self.params.propagator == 'Fresnel':
            self.ifft2s()
            self.optimizable.esw = self.optimizable.esw * self.optimizable.quadraticPhase.conj()
            self.optimizable.eswUpdate = self.optimizable.eswUpdate * self.optimizable.quadraticPhase.conj()
        elif self.params.propagator == 'ASP' or self.params.propagator == 'polychromeASP':
            self.optimizable.eswUpdate = ifft2c(fft2c(self.optimizable.ESW) * self.optimizable.transferFunction.conj())
        elif self.params.propagator == 'scaledASP' or self.params.propagator == 'scaledPolychromeASP':
            self.optimizable.eswUpdate = ifft2c(fft2c(self.optimizable.ESW) * self.optimizable.Q2.conj()) \
                                         * self.optimizable.Q1.conj()
        elif self.params.propagator == 'twoStepPolychrome':
            self.ifft2s()
            self.optimizable.esw = ifft2c(fft2c(self.optimizable.esw *
                                                      self.optimizable.quadraticPhase.conj()) *
                                                self.optimizable.transferFunction.conj())
            self.optimizable.eswUpdate = ifft2c(fft2c(self.optimizable.eswUpdate *
                                               self.optimizable.quadraticPhase.conj()) *
                                                self.optimizable.transferFunction.conj())
        else:
            raise Exception('Propagator is not properly set, choose from Fraunhofer, Fresnel, ASP and scaledASP')

    def exportOjb(self, extension='.mat'):
        """
        Export the object.

        If extension == '.mat', export to matlab file.
        If extension == '.png', export to a png file (with amplitude-phase)

        Matches: exportObj (except for the PNG)

        :return:
        """
        raise NotImplementedError()

    def fft2s(self):
        """
        Computes the fourier transform of the exit surface wave.
        :return:
        """
        # find out if this should be performed on the GPU
        xp = getArrayModule(self.optimizable.esw)

        if self.params.fftshiftSwitch:
            self.optimizable.ESW = xp.fft.fft2(self.optimizable.esw, norm='ortho')
        else:
            self.optimizable.ESW = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(self.optimizable.esw),norm='ortho'))

    def getBeamWidth(self):
        """
        Calculate probe beam width (Full width half maximum)
        :return:
        """
        P = np.sum(abs(asNumpyArray(self.optimizable.probe[..., -1, :, :])) ** 2, axis=(0, 1, 2))
        P = P/np.sum(P, axis=(-1, -2))
        xMean = np.sum(self.optimizable.Xp * P, axis=(-1, -2))
        yMean = np.sum(self.optimizable.Yp * P, axis=(-1, -2))
        xVariance = np.sum((self.optimizable.Xp - xMean) ** 2 * P, axis=(-1, -2))
        yVariance = np.sum((self.optimizable.Yp - yMean) ** 2 * P, axis=(-1, -2))

        c = 2 * np.sqrt(2 * np.log(2)) # constant for converting variance to FWHM (see e.g. https://en.wikipedia.org/wiki/Full_width_at_half_maximum)
        self.optimizable.beamWidthX = c * np.sqrt(xVariance)
        self.optimizable.beamWidthY = c * np.sqrt(yVariance)

    def getOverlap(self, ind1, ind2):
        """
        Calculate linear and area overlap between two scan positions indexed ind1 and ind2
        """
        sy = abs(self.optimizable.positions[ind2, 0] - self.optimizable.positions[ind1, 0]) * self.optimizable.dxp
        sx = abs(self.optimizable.positions[ind2, 1] - self.optimizable.positions[ind1, 1]) * self.optimizable.dxp

        # task 1: get linear overlap
        self.getBeamWidth()
        self.optimizable.linearOverlap = 1 - np.sqrt(sx**2+sy**2)/\
                                         np.minimum(self.optimizable.beamWidthX, self.optimizable.beamWidthY)
        self.optimizable.linearOverlap = np.maximum(self.optimizable.linearOverlap, 0)

        # task 2: get area overlap
        # spatial frequency pixel size
        df = 1/(self.optimizable.Np*self.optimizable.dxp)
        # spatial frequency meshgrid
        fx = np.arange(-self.optimizable.Np//2, self.optimizable.Np//2) * df
        Fx, Fy = np.meshgrid(fx, fx)
        # absolute value of probe and 2D fft
        P = abs(asNumpyArray(self.optimizable.probe[:, 0, 0, -1,...]))
        Q = fft2c(P)
        # calculate overlap between positions
        self.optimizable.areaOverlap = np.mean(abs(np.sum(Q**2*np.exp(-1.j*2*np.pi*(Fx*sx+Fy*sy)), axis=(-1, -2)))/\
                                       np.sum(abs(Q)**2, axis=(-1, -2)), axis=0)


    def getErrorMetrics(self):
        """
        matches getErrorMetrics.m
        :return:
        """
        if not self.params.saveMemory:
            # Calculate mean error for all positions (make separate function for all of that)
            if self.params.FourierMaskSwitch:
                self.optimizable.errorAtPos = np.sum(np.abs(self.optimizable.detectorError) * 
                                                     self.experimentalData.W, axis=(-1, -2))
            else:
                self.optimizable.errorAtPos = np.sum(np.abs(self.optimizable.detectorError), axis=(-1, -2))
        self.optimizable.errorAtPos = asNumpyArray(self.optimizable.errorAtPos)/asNumpyArray(self.experimentalData.energyAtPos + 1)
        eAverage = np.sum(self.optimizable.errorAtPos)

        # append to error vector (for plotting error as function of iteration)
        self.optimizable.error = np.append(self.optimizable.error, eAverage)


    def getRMSD(self, positionIndex):
        """
        Root mean square deviation between ptychogram and intensity estimate
        :param positionIndex:
        :return:
        """
        # find out wether or not to use the GPU
        xp = getArrayModule(self.optimizable.Iestimated)
        self.currentDetectorError = abs(self.optimizable.Imeasured - self.optimizable.Iestimated)

        # if it's on the GPU, transfer it back
        # if hasattr(self.currentDetectorError, 'device'):
        #     self.currentDetectorError = self.currentDetectorError.get()
        if self.params.saveMemory:
            if self.params.FourierMaskSwitch and not self.params.CPSCswitch:
                self.optimizable.errorAtPos[positionIndex] = xp.sum(self.currentDetectorError*self.experimentalData.W)
            elif self.params.FourierMaskSwitch and self.params.CPSCswitch:
                raise NotImplementedError
            else:
                self.optimizable.errorAtPos[positionIndex] = xp.sum(self.currentDetectorError)
        else:
            self.optimizable.detectorError[positionIndex] = self.currentDetectorError

    def ifft2s(self):
        """ Inverse FFT"""
        # find out if this should be performed on the GPU
        xp = getArrayModule(self.optimizable.esw)

        if self.params.fftshiftSwitch:
            self.optimizable.eswUpdate = xp.fft.ifft2(self.optimizable.ESW, norm='ortho')
        else:
            self.optimizable.eswUpdate = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(self.optimizable.ESW),norm='ortho'))

    def intensityProjection(self, positionIndex):
        """ Compute the projected intensity.
            Barebones, need to implement other methods
        """
        # figure out whether or not to use the GPU
        xp = getArrayModule(self.optimizable.esw)

        # zero division mitigator
        gimmel = 1e-10

        # propagate to detector
        self.object2detector()

        # get estimated intensity (2D array, in the case of multislice, only take the last slice)
        if self.params.intensityConstraint == 'interferometric':
            self.optimizable.Iestimated = xp.sum(xp.abs(self.optimizable.ESW+self.optimizable.reference) ** 2, axis=(0, 1, 2))[-1]
        else:
            self.optimizable.Iestimated = xp.sum(xp.abs(self.optimizable.ESW) ** 2, axis=(0, 1, 2))[-1]
        if self.params.backgroundModeSwitch:
            self.optimizable.Iestimated += self.optimizable.background

        # get measured intensity todo implement CPSC, kPIE
        if self.params.CPSCswitch:
            self.decompressionProjection(self.positionIndices)
        else:
            self.optimizable.Imeasured = xp.array(self.experimentalData.ptychogram[positionIndex])

        # adaptive denoising
        if self.params.adaptiveDenoisingSwitch:
            self.adaptiveDenoising()

        self.getRMSD(positionIndex)

        # intensity projection constraints
        if self.params.intensityConstraint == 'fluctuation':
            # scaling
            if self.params.FourierMaskSwitch:
                aleph = xp.sum(self.optimizable.Imeasured*self.optimizable.Iestimated*self.experimentalData.W) / \
                        xp.sum(self.optimizable.Imeasured*self.optimizable.Imeasured*self.experimentalData.W)
            else:
                aleph = xp.sum(self.optimizable.Imeasured * self.optimizable.Iestimated) / \
                        xp.sum(self.optimizable.Imeasured * self.optimizable.Imeasured)
            self.params.intensityScaling[positionIndex] = aleph
            # scaled projection
            frac = (1+aleph)/2*self.optimizable.Imeasured/(self.optimizable.Iestimated+gimmel)

        elif self.params.intensityConstraint == 'exponential':
            x = self.currentDetectorError/(self.optimizable.Iestimated+gimmel)
            W = xp.exp(-0.05 * x)
            frac = xp.sqrt(self.optimizable.Imeasured / (self.optimizable.Iestimated + gimmel))
            frac = W * frac + (1-W)

        elif self.params.intensityConstraint == 'poission':
            frac = self.optimizable.Imeasured / (self.optimizable.Iestimated + gimmel)

        elif self.params.intensityConstraint == 'standard' or self.params.intensityConstraint == 'interferometric':
            frac = xp.sqrt(self.optimizable.Imeasured / (self.optimizable.Iestimated + gimmel))


        # elif self.params.intensityConstraint == 'interferometric':
        #     frac = xp.sqrt(self.optimizable.Imeasured / (self.optimizable.Iestimated + self.optimizable.background + gimmel))
        else:
            raise ValueError('intensity constraint not properly specified!')

        # apply mask
        if self.params.FourierMaskSwitch and self.params.CPSCswitch and len(self.optimizable.error) > 5:
            frac = self.experimentalData.W * frac + (1-self.experimentalData.W)

        # update ESW
        if self.params.intensityConstraint == 'interferometric':
            temp = (self.optimizable.ESW+self.optimizable.reference)*frac-self.optimizable.ESW
            self.optimizable.ESW = (self.optimizable.ESW+self.optimizable.reference)*frac-self.optimizable.reference
            self.optimizable.reference = temp
        else:
            self.optimizable.ESW = self.optimizable.ESW * frac

        # update background (see PhD thsis by Peng Li)
        if self.params.backgroundModeSwitch:
            if self.params.FourierMaskSwitch:
                self.optimizable.background = self.optimizable.background*(1+1/self.experimentalData.numFrames*(xp.sqrt(frac)-1))**2*self.experimentalData.W
            else:
                self.optimizable.background = self.optimizable.background*(1+1/self.experimentalData.numFrames*(xp.sqrt(frac)-1))**2

        # back propagate to object plane
        self.detector2object()

    # def decompressionProjection(self, positionIndex):
        # overwrite the measured intensity (just to have same dimentions as Iestimated,
        # further below the actual decompression projection takes place)
        # self.optimizable.Imeasured = self.optimizable.Iestimated.copy()
        #
        # # determine downsampled fraction
        # frac = np.ones_like(self.optimizable.ptychogramDownsampled[0])
        # self.optimizableImeasuredDownsampled = self.optimizable.ptychogramDownsampled[positionIndex]
        # I = self.optimizable.Iestimated.copy()

    # def setCPSC(self):
    #     """
    #
    #     """
        # # define temporary image
        # im = rescale(self.experimentalData.ptychogram[0], self.params.upsamplingFactor)
        #
        # # get upsampling index
        # _,self.optimizable.upsampledIndex, self.optimizable.downsampledIndex = p2bin(im, self.params.upsamplingFactor)
        # self.optimizable.ptychograpmDownsampled = self.experimentalData.ptychogram
        #
        # # update coordinates (only need to update the dxd, the rest updates automatically)
        # self.optimizable.dxd = self.optimizable.dxd/self.params.upsamplingFactor
        #
        # # upsample probe
        # probeTemp = self.optimizable.probe.copy()


    def showReconstruction(self, loop):
        """
        Show the reconstruction process.
        :param loop: the iteration number
        :return:
        """
        if np.mod(loop, self.monitor.figureUpdateFrequency) == 0:

            if self.experimentalData.operationMode == 'FPM':
                object_estimate = np.squeeze(asNumpyArray(
                    fft2c(self.optimizable.object)[..., self.optimizable.objectROI[0], self.optimizable.objectROI[1]]))
                probe_estimate = np.squeeze(asNumpyArray(
                    self.optimizable.probe[..., self.optimizable.probeROI[0], self.optimizable.probeROI[1]]))
            else:
                object_estimate = np.squeeze(asNumpyArray(
                    self.optimizable.object[..., self.optimizable.objectROI[0], self.optimizable.objectROI[1]]))
                probe_estimate = np.squeeze(asNumpyArray(
                    self.optimizable.probe[..., self.optimizable.probeROI[0], self.optimizable.probeROI[1]]))

            self.monitor.updateDefaultMonitor(object_estimate=object_estimate, probe_estimate=probe_estimate)

            if self.monitor.verboseLevel =='high':
                if self.params.fftshiftSwitch:
                    Iestimated = np.fft.fftshift(asNumpyArray(self.optimizable.Iestimated))
                    Imeasured = np.fft.fftshift(asNumpyArray(self.optimizable.Imeasured))
                else:
                    Iestimated = asNumpyArray(self.optimizable.Iestimated)
                    Imeasured = asNumpyArray(self.optimizable.Imeasured)

                self.monitor.updateDiffractionDataMonitor(Iestimated=Iestimated, Imeasured=Imeasured)

                self.getOverlap(0, 1)

                self.pbar.write('')
                self.pbar.write('iteration: %i' % loop)
                self.pbar.write('error: %.1f' % self.optimizable.error[loop])
                self.pbar.write('estimated linear overlap: %.1f %%' % (100*self.optimizable.linearOverlap))
                self.pbar.write('estimated area overlap: %.1f %%' % (100*self.optimizable.areaOverlap))
                # self.pbar.write('coherence structure:')

            if self.params.positionCorrectionSwitch:
                # show reconstruction
                if (len(self.optimizable.error) > self.startAtIteration): #& (np.mod(loop,
                                                                                   #self.monitor.figureUpdateFrequency) == 0):
                    figure, ax = plt.subplots(1, 1, num=102, squeeze=True, clear=True, figsize=(5, 5))
                    ax.set_title('Estimated scan grid positions')
                    ax.set_xlabel('(um)')
                    ax.set_ylabel('(um)')
                    # ax.set_xscale('symlog')
                    plt.plot(self.optimizable.positions0[:, 1] * self.optimizable.dxo * 1e6,
                             self.optimizable.positions0[:, 0] * self.optimizable.dxo * 1e6, 'bo')
                    plt.plot(self.optimizable.positions[:, 1] * self.optimizable.dxo * 1e6,
                             self.optimizable.positions[:, 0] * self.optimizable.dxo * 1e6, 'yo')
                    # plt.xlabel('(um))')
                    # plt.ylabel('(um))')
                    # plt.show()
                    plt.tight_layout()
                    plt.show(block=False)

                    figure2, ax2 = plt.subplots(1, 1, num=103, squeeze=True, clear=True, figsize=(5, 5))
                    ax2.set_title('Displacement')
                    ax2.set_xlabel('(um)')
                    ax2.set_ylabel('(um)')
                    plt.plot(self.D[:, 1] * self.optimizable.dxo * 1e6,
                             self.D[:, 0] * self.optimizable.dxo * 1e6, 'o')
                    # ax.set_xscale('symlog')
                    plt.tight_layout()
                    plt.show(block=False)

                    # elif np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
                    figure.canvas.draw()
                    figure.canvas.flush_events()
                    figure2.canvas.draw()
                    figure2.canvas.flush_events()
                    # self.showReconstruction(loop)
            # print('iteration:%i' %len(self.optimizable.error))
            # print('runtime:')
            # print('error:')
        # TODO: print info

    def positionCorrection(self, objectPatch, positionIndex, sy, sx):
        """
        Modified from pcPIE. Position correction is done by using positionCorrection and positionCorrectionUpdate
        :param objectPatch:
        :param positionIndex:
        :param sy:
        :param sx:
        :return:
        """
        xp = getArrayModule(objectPatch)
        if len(self.optimizable.error) > self.startAtIteration:
            # position gradients
            # shiftedImages = xp.zeros((self.rowShifts.shape + objectPatch.shape))
            cc = xp.zeros((len(self.rowShifts), 1))
            for shifts in range(len(self.rowShifts)):
                tempShift = xp.roll(objectPatch, self.rowShifts[shifts], axis=-2)
                # shiftedImages[shifts, ...] = xp.roll(tempShift, self.colShifts[shifts], axis=-1)
                shiftedImages = xp.roll(tempShift, self.colShifts[shifts], axis=-1)
                cc[shifts] = xp.squeeze(xp.sum(shiftedImages.conj() * self.optimizable.object[..., sy, sx],
                                               axis=(-2, -1)))
            # truncated cross - correlation
            # cc = xp.squeeze(xp.sum(shiftedImages.conj() * self.optimizable.object[..., sy, sx], axis=(-2, -1)))
            cc = abs(cc)
            betaGrad = 1000
            normFactor = xp.sum(objectPatch.conj() * objectPatch, axis=(-2, -1)).real
            grad_x = betaGrad * xp.sum((cc.T - xp.mean(cc)) / normFactor * xp.array(self.colShifts))
            grad_y = betaGrad * xp.sum((cc.T - xp.mean(cc)) / normFactor * xp.array(self.rowShifts))
            r = 3
            if abs(grad_x) > r:
                grad_x = r * grad_x / abs(grad_x)
            if abs(grad_y) > r:
                grad_y = r * grad_y / abs(grad_y)
            self.D[positionIndex, :] = self.daleth * asNumpyArray([grad_y, grad_x]) + self.beth * \
                                       self.D[positionIndex, :]

    def positionCorrectionUpdate(self):
        if len(self.optimizable.error) > self.startAtIteration:
            # update positions
            self.experimentalData.encoder = (self.optimizable.positions - self.adaptStep * self.D -
                                             self.optimizable.No // 2 + self.optimizable.Np // 2) * \
                                            self.optimizable.dxo
            # fix center of mass of positions
            self.experimentalData.encoder[:, 0] = self.experimentalData.encoder[:, 0] - \
                                                  np.mean(self.experimentalData.encoder[:, 0]) + self.meanEncoder00
            self.experimentalData.encoder[:, 1] = self.experimentalData.encoder[:, 1] - \
                                                  np.mean(self.experimentalData.encoder[:, 1]) + self.meanEncoder01

            # self.optimizable.positions[:,0] = self.optimizable.positions[:,0] - \
            #         np.round(np.mean(self.optimizable.positions[:,0]) -
            #                   np.mean(self.optimizable.positions0[:,0]) )
            # self.optimizable.positions[:, 1] = self.optimizable.positions[:, 1] - \
            #                                         np.around(np.mean(self.optimizable.positions[:, 1]) -
            #                                                   np.mean(self.optimizable.positions0[:, 1]))



    def applyConstraints(self, loop):
        """
        Apply constraints.
        :param loop: loop number
        :return:
        """
        # enforce empty beam constraint
        if self.params.modulusEnforcedProbeSwitch:
            self.modulusEnforcedProbe()

        if self.params.orthogonalizationSwitch:
            if np.mod(loop, self.params.orthogonalizationFrequency) == 0:
                self.orthogonalization()

        # probe normalization to measured PSD todo: check for multiwave and multi object states
        if self.params.probePowerCorrectionSwitch:
            self.optimizable.probe = self.optimizable.probe / np.sqrt(
                np.sum(self.optimizable.probe * self.optimizable.probe.conj())) * self.experimentalData.maxProbePower

        if self.params.comStabilizationSwitch:
            self.comStabilization()

        if self.params.PSDestimationSwitch:
            raise NotImplementedError()

        if self.params.probeBoundary:
            self.optimizable.probe *= self.probeWindow

        if self.params.absorbingProbeBoundary:
            if self.experimentalData.operationMode =='FPM':
                self.absorbingProbeBoundaryAleph = 1

            self.optimizable.probe = (1 - self.params.absorbingProbeBoundaryAleph)*self.optimizable.probe+\
                                     self.params.absorbingProbeBoundaryAleph*self.optimizable.probe*self.probeWindow

        # Todo: objectSmoothenessSwitch,probeSmoothenessSwitch,
        if self.params.probeSmoothenessSwitch:
            raise NotImplementedError()

        if self.params.objectSmoothenessSwitch:
            raise NotImplementedError()


        if self.params.absObjectSwitch:
            self.optimizable.object = (1-self.params.absObjectBeta)*self.optimizable.object+\
                                      self.params.absObjectBeta*abs(self.optimizable.object)

        if self.params.absProbeSwitch:
            self.optimizable.probe = (1-self.params.absProbeBeta)*self.optimizable.probe+\
                                      self.params.absProbeBeta*abs(self.optimizable.probe)

        # this is intended to slowly push non-measured object region to abs value lower than
        # the max abs inside object ROI allowing for good contrast when monitoring object
        if self.params.objectContrastSwitch:
            self.optimizable.object = 0.995*self.optimizable.object+0.005*\
                                      np.mean(abs(self.optimizable.object[..., self.optimizable.objectROI[0],
                                                                          self.optimizable.objectROI[1]]))
        if self.params.couplingSwitch and self.optimizable.nlambda > 1:
            self.optimizable.probe[0] = (1 - self.params.couplingAleph) * self.optimizable.probe[0] + \
                                        self.params.couplingAleph * self.optimizable.probe[1]
            for lambdaLoop in np.arange(1, self.optimizable.nlambda - 1):
                self.optimizable.probe[lambdaLoop] = (1 - self.params.couplingAleph) * self.optimizable.probe[lambdaLoop] + \
                                                     self.params.couplingAleph * (self.optimizable.probe[lambdaLoop + 1] +
                                                                           self.optimizable.probe[
                                                                               lambdaLoop - 1]) / 2

            self.optimizable.probe[-1] = (1 - self.params.couplingAleph) * self.optimizable.probe[-1] + \
                                         self.params.couplingAleph * self.optimizable.probe[-2]
        if self.params.binaryWFSSwitch:
            self.optimizable.probe = (1 - self.params.binaryWFSAleph) * self.optimizable.probe + \
                                     self.params.binaryWFSAleph * abs(self.optimizable.probe)

        if self.params.positionCorrectionSwitch:
            self.positionCorrectionUpdate()

    def orthogonalization(self):
        """
        Perform orthogonalization
        :return:
        """
        xp = getArrayModule(self.optimizable.probe)
        if self.optimizable.npsm > 1:
            # orthogonalize the probe for each wavelength and each slice
            for id_l in range(self.optimizable.nlambda):
                for id_s in range(self.optimizable.nslice):
                    self.optimizable.probe[id_l, 0, :, id_s, :, :], self.normalizedEigenvaluesProbe, self.MSPVprobe = \
                        orthogonalizeModes(self.optimizable.probe[id_l, 0, :, id_s, :, :])
                    self.optimizable.purityProbe = np.sqrt(np.sum(self.normalizedEigenvaluesProbe ** 2))

                    # orthogonolize momentum operator
                    if self.params.momentumAcceleration:
                        # orthogonalize probe Buffer
                        p = self.optimizable.probeBuffer[id_l, 0, :, id_s, :, :].reshape(
                            (self.optimizable.npsm, self.optimizable.Np ** 2))
                        self.optimizable.probeBuffer[id_l, 0, :, id_s, :, :] = (xp.array(self.MSPVprobe) @ p).reshape(
                            (self.optimizable.npsm, self.optimizable.Np, self.optimizable.Np))
                        # orthogonalize probe momentum
                        p = self.optimizable.probeMomentum[id_l, 0, :, id_s, :, :].reshape(
                            (self.optimizable.npsm, self.optimizable.Np ** 2))
                        self.optimizable.probeMomentum[id_l, 0, :, id_s, :, :] = (xp.array(self.MSPVprobe) @ p).reshape(
                            (self.optimizable.npsm, self.optimizable.Np, self.optimizable.Np))

                        # if self.comStabilizationSwitch:
                        #     self.comStabilization()



        elif self.optimizable.nosm > 1:
            # orthogonalize the object for each wavelength and each slice
            for id_l in range(self.optimizable.nlambda):
                for id_s in range(self.optimizable.nslice):
                    self.optimizable.object[id_l, :, 0, id_s, :, :], self.normalizedEigenvaluesObject, self.MSPVobject = \
                        orthogonalizeModes(self.optimizable.object[id_l, :, 0, id_s, :, :], method='snapShots')
                    self.optimizable.purityObject = np.sqrt(np.sum(self.normalizedEigenvaluesObject ** 2))

                    # orthogonolize momentum operator
                    if self.params.momentumAcceleration:
                        # orthogonalize object Buffer
                        p = self.optimizable.objectBuffer[id_l, :, 0, id_s, :, :].reshape(
                            (self.optimizable.nosm, self.optimizable.No ** 2))
                        self.optimizable.objectBuffer[id_l, :, 0, id_s, :, :] = (xp.array(self.MSPVobject) @ p).reshape(
                            (self.optimizable.nosm, self.optimizable.No, self.optimizable.No))
                        # orthogonalize object momentum
                        p = self.optimizable.objectMomentum[id_l, :, 0, id_s, :, :].reshape(
                            (self.optimizable.nosm, self.optimizable.No ** 2))
                        self.optimizable.objectMomentum[id_l, :, 0, id_s, :, :] = (xp.array(self.MSPVobject) @ p).reshape(
                            (self.optimizable.nosm, self.optimizable.No, self.optimizable.No))

        else:
            pass

    def comStabilization(self):
        """
        Perform center of mass stabilization (center the probe)
        :return:
        """
        xp = getArrayModule(self.optimizable.probe)
        # calculate center of mass of the probe
        P2 = xp.sum(abs(self.optimizable.probe[:,:,:,-1,...])**2, axis=(0,1,2))
        demon = xp.sum(P2)*self.optimizable.dxp
        xc = xp.int(xp.around(xp.sum(xp.array(self.optimizable.Xp, xp.float32) * P2) / demon))
        yc = xp.int(xp.around(xp.sum(xp.array(self.optimizable.Yp, xp.float32) * P2) / demon))
        # shift only if necessary
        if xc**2+yc**2>1:
            # shift probe
            for k in xp.arange(self.optimizable.npsm): # todo check for multislice
                self.optimizable.probe[:,:,k,-1,...] = \
                    xp.roll(self.optimizable.probe[:,:,k,-1,...], (-yc, -xc), axis=(-2, -1))
                # for mPIE
                if self.params.momentumAcceleration:
                    self.optimizable.probeMomentum[:,:,k,-1,...] = \
                        xp.roll(self.optimizable.probeMomentum[:,:,k,-1,...], (-yc, -xc), axis=(-2, -1))
                    self.optimizable.probeBuffer[:,:,k,-1,...] = \
                        xp.roll(self.optimizable.probeBuffer[:,:,k,-1,...], (-yc, -xc), axis=(-2, -1))


            # shift object
            for k in xp.arange(self.optimizable.nosm): # todo check for multislice
                self.optimizable.object[:,k,:,-1,...] = \
                    xp.roll(self.optimizable.object[:,k,:,-1,...], (-yc, -xc), axis=(-2, -1))
                # for mPIE
                if self.params.momentumAcceleration:
                    self.optimizable.objectMomentum[:, k, :, -1, ...] = \
                        xp.roll(self.optimizable.objectMomentum[:, k, :, -1, ...], (-yc, -xc), axis=(-2, -1))
                    self.optimizable.objectBuffer[:, k, :, -1, ...] = \
                        xp.roll(self.optimizable.objectBuffer[:, k, :, -1, ...], (-yc, -xc), axis=(-2, -1))

            # if self.optimizable.nlambda > 1:
            #     for k in xp.arange(self.optimizable.nlambda): # todo check for multislice
            #         P2 = xp.sum(abs(self.optimizable.probe[k, :, :, -1, ...]) ** 2, axis=(1, 2))
            #         demon = xp.sum(P2) * self.optimizable.dxp
            #         xc = xp.int(xp.around(xp.sum(xp.array(self.optimizable.Xp, xp.float32) * P2) / demon))
            #         yc = xp.int(xp.around(xp.sum(xp.array(self.optimizable.Yp, xp.float32) * P2) / demon))
            #         self.optimizable.probe[k,:,:,-1,...] = \
            #             xp.roll(self.optimizable.probe[k,:,:,-1,...], (-yc, -xc), axis=(-2, -1))
            #         self.optimizable.object[k, :, :, -1, ...] = \
            #             xp.roll(self.optimizable.object[k, :, :, -1, ...], (-yc, -xc), axis=(-2, -1))
                # todo implement for mPIE

    def modulusEnforcedProbe(self):
        # propagate probe to detector
        self.optimizable.esw = self.optimizable.probe
        self.object2detector()

        if self.params.FourierMaskSwitch:
            self.optimizable.ESW = self.optimizable.ESW*xp.sqrt(
                self.experimentalData.emptyBeam/1e-10+xp.sum(xp.abs(self.optimizable.ESW)**2, axis=(0, 1, 2, 3)))*self.experimentalData.W\
                                   +self.optimizable.ESW*(1-self.experimentalData.W)
        else:
            self.optimizable.ESW = self.optimizable.ESW * np.sqrt(
                self.experimentalData.emptyBeam / (1e-10 + xp.sum(abs(self.optimizable.ESW) ** 2, axis=(0, 1, 2, 3))))

        self.detector2object()
        self.optimizable.probe = self.optimizable.esw

    def adaptiveDenoising(self):
        """
        Use the difference of mean intensities between the low-resolution
        object estimate and the low-resolution raw data to estimate the
        noise floor to be clipped.
        :return:
        """
        # figure out wether or not to use the GPU
        xp = getArrayModule(self.optimizable.esw)

        Ameasured = self.optimizable.Imeasured**0.5
        Aestimated = xp.abs(self.optimizable.Iestimated)**0.5

        noise = xp.abs(xp.mean(Ameasured - Aestimated))

        Ameasured = Ameasured - noise
        Ameasured[Ameasured<0]=0
        self.optimizable.Imeasured = Ameasured**2
