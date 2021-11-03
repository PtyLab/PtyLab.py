from fracPy.Monitor.Plots import ObjectProbeErrorPlot,DiffractionDataPlot
import numpy as np
from scipy.signal import get_window
import logging
import warnings
import h5py
# fracPy imports
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray, transfer_fields_to_gpu, transfer_fields_to_cpu
from fracPy.utils.initializationFunctions import initialProbeOrObject
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Reconstruction.Reconstruction import Reconstruction
from fracPy.Params.Params import Params
from fracPy.utils.utils import ifft2c, fft2c, orthogonalizeModes, circ, posit
from fracPy.Operators.Operators import aspw, scaledASP
from fracPy.Monitor.Monitor import Monitor
from fracPy.utils.visualisation import hsvplot
from matplotlib import pyplot as plt

try:
    import cupy as cp

except ImportError:
    print("Cupy not installed")

try:
    from skimage.transform import rescale
except ImportError:
    print("Skimage not installed")


class BaseEngine(object):
    """
    Common properties that are common for all reconstruction Engines are defined here.

    Unless you are testing the code, there's hardly any need to create this object. For your own implementation,
    inherit from this object

    """

    def __init__(self, reconstruction: Reconstruction, experimentalData: ExperimentalData, params: Params, monitor: Monitor):
        # These statements don't copy any data, they just keep a reference to the object
        self.reconstruction: Reconstruction = reconstruction
        self.experimentalData = experimentalData
        self.params = params
        self.monitor: Monitor = monitor
        self.monitor.reconstruction = reconstruction

        # datalogger
        self.logger = logging.getLogger('BaseEngine')

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

    def _setCPSC(self):
        """
        set constrained-pixel-sum constraint:
        -save measured diffraction patterns into ptychograpmDownsampled
        -pad the probe (useful when having a pre-calibrated probe)
        -update the coordinates
        """

        # save the measured ptychogram into ptychograpmDownsampled
        self.experimentalData.ptychogramDownsampled = self.experimentalData.ptychogram

        # pad the probe
        padNum_before = (self.params.CPSCupsamplingFactor - 1) * self.reconstruction.Np // 2
        padNum_after = (self.params.CPSCupsamplingFactor - 1) * self.reconstruction.Np - padNum_before
        self.reconstruction.probe = np.pad(self.reconstruction.probe,
                                        ((0, 0), (0, 0), (0, 0), (0, 0), (padNum_before, padNum_after),
                                         (padNum_before, padNum_after)))

        # pad the momentums, buffers
        if hasattr(self.reconstruction, 'probeBuffer'):
            self.reconstruction.probeBuffer = self.reconstruction.probe.copy()
        if hasattr(self.reconstruction, 'probeMomentum'):
            self.reconstruction.probeMomentum = np.pad(self.reconstruction.probeMomentum,
                                                    ((0, 0), (0, 0), (0, 0), (0, 0), (padNum_before, padNum_after),
                                                     (padNum_before, padNum_after)))

        # update coordinates (only need to update the Nd and dxd, the rest updates automatically)
        self.reconstruction.Nd = self.experimentalData.ptychogramDownsampled.shape[-1] * self.params.CPSCupsamplingFactor
        self.reconstruction.dxd = self.reconstruction.dxd / self.params.CPSCupsamplingFactor

        self.logger.info('CPSCswitch is on, coordinates(dxd,dxp,dxo) have been updated')

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
        reconstruction.error sums over errorAtPos, one number at each iteration;
        """
        # initialize detector error matrices
        if self.params.saveMemory:
            self.reconstruction.detectorError = 0
        else:
            if not hasattr(self.reconstruction, 'detectorError'):
                self.reconstruction.detectorError = np.zeros((self.experimentalData.numFrames,
                                                           self.reconstruction.Nd, self.reconstruction.Nd))
        # initialize energy at each scan position
        if not hasattr(self.reconstruction, 'errorAtPos'):
            self.reconstruction.errorAtPos = np.zeros((self.experimentalData.numFrames, 1), dtype=np.float32)
        # initialize final error
        if not hasattr(self.reconstruction, 'error'):
            self.reconstruction.error = []

    def _initialProbePowerCorrection(self):
        if self.params.probePowerCorrectionSwitch:
            self.reconstruction.probe = self.reconstruction.probe / np.sqrt(
                np.sum(self.reconstruction.probe * self.reconstruction.probe.conj())) * self.experimentalData.maxProbePower

    def _probeWindow(self):
        # absorbing probe boundary: filter probe with super-gaussian window function
        if not self.params.saveMemory or self.params.absorbingProbeBoundary:
            self.probeWindow = np.exp(-((self.reconstruction.Xp ** 2 + self.reconstruction.Yp ** 2) /
                                        (2 * (3 / 4 * self.reconstruction.Np * self.reconstruction.dxp / 2.355) ** 2)) ** 10)

        if self.params.probeBoundary:
            self.probeWindow = circ(self.reconstruction.Xp, self.reconstruction.Yp,
                                    self.experimentalData.entrancePupilDiameter + self.experimentalData.entrancePupilDiameter * 0.2)

    def _setObjectProbeROI(self, update=False):
        """
        Set object/probe ROI for monitoring
        """
        if not hasattr(self.monitor, 'objectROI') or update:
            self.monitor._setObjectROI(self.reconstruction.position_range, self.reconstruction.position_center,
                                       self.reconstruction.No, self.reconstruction.Np)


        if not hasattr(self.monitor, 'probeROI') or update:
            self.monitor._setProbeROI(self.experimentalData.entrancePupilDiameter, self.reconstruction.dxp,
                                      self.reconstruction.Np)
            #
            # self.monitor.probeROI = [slice(max(0, self.reconstruction.Np // 2 - r),
            #                                    min(self.reconstruction.Np, self.reconstruction.Np // 2 + r)),
            #                              slice(max(0, self.reconstruction.Np // 2 - r),
            #                                    min(self.reconstruction.Np, self.reconstruction.Np // 2 + r)) ]

    def _showInitialGuesses(self):
        self.monitor.initializeMonitors()

        objectEstimate = np.squeeze(self.reconstruction.object
                                     [..., self.monitor.objectROI[-2], self.monitor.objectROI[-1]])
        probeEstimate = np.squeeze(self.reconstruction.probe
                                    [..., self.monitor.probeROI[-2], self.monitor.probeROI[-1]])
        self.monitor.updateObjectProbeErrorMonitor(object_estimate=objectEstimate, probe_estimate=probeEstimate)

    def _initializeQuadraticPhase(self):
        """
        # initialize quadraticPhase term or transferFunctions used in propagators
        """
        if self.params.propagatorType == 'Fresnel':
            self.reconstruction.quadraticPhase = np.exp(1.j * np.pi / (self.reconstruction.wavelength * self.reconstruction.zo)
                                                     * (self.reconstruction.Xp ** 2 + self.reconstruction.Yp ** 2))
        elif self.params.propagatorType == 'ASP':
            if self.params.fftshiftSwitch:
                raise ValueError('ASP propagatorType works only with fftshiftSwitch = False!')
            if self.reconstruction.nlambda > 1:
                raise ValueError('For multi-wavelength, polychromeASP needs to be used instead of ASP')

            dummy = np.ones((1, self.reconstruction.nosm, self.reconstruction.npsm,
                             1, self.reconstruction.Np, self.reconstruction.Np), dtype='complex64')
            self.reconstruction.transferFunction = np.array(
                [[[[aspw(dummy[nlambda, nosm, npsm, nslice, :, :],
                         self.reconstruction.zo, self.reconstruction.wavelength,
                         self.reconstruction.Lp)[1]
                    for nslice in range(1)]
                   for npsm in range(self.reconstruction.npsm)]
                  for nosm in range(self.reconstruction.nosm)]
                 for nlambda in range(self.reconstruction.nlambda)])

        elif self.params.propagatorType == 'polychromeASP':
            dummy = np.ones((self.reconstruction.nlambda, self.reconstruction.nosm, self.reconstruction.npsm,
                             1, self.reconstruction.Np, self.reconstruction.Np), dtype='complex64')
            self.reconstruction.transferFunction = np.array(
                [[[[aspw(dummy[nlambda, nosm, npsm, nslice, :, :],
                         self.reconstruction.zo, self.reconstruction.spectralDensity[nlambda],
                         self.reconstruction.Lp)[1]
                    for nslice in range(1)]
                   for npsm in range(self.reconstruction.npsm)]
                  for nosm in range(self.reconstruction.nosm)]
                 for nlambda in range(self.reconstruction.nlambda)])
            if self.params.fftshiftSwitch:
                raise ValueError('ASP propagatorType works only with fftshiftSwitch = False!')

        elif self.params.propagatorType == 'scaledASP':
            if self.params.fftshiftSwitch:
                raise ValueError('scaledASP propagatorType works only with fftshiftSwitch = False!')
            if self.reconstruction.nlambda > 1:
                raise ValueError('For multi-wavelength, scaledPolychromeASP needs to be used instead of scaledASP')
            dummy = np.ones((1, self.reconstruction.nosm, self.reconstruction.npsm,
                             1, self.reconstruction.Np, self.reconstruction.Np), dtype='complex64')
            self.reconstruction.Q1 = np.ones_like(dummy)
            self.reconstruction.Q2 = np.ones_like(dummy)
            for nosm in range(self.reconstruction.nosm):
                for npsm in range(self.reconstruction.npsm):
                    _, self.reconstruction.Q1[0, nosm, npsm, 0, ...], self.reconstruction.Q2[
                        0, nosm, npsm, 0, ...] = scaledASP(
                        dummy[0, nosm, npsm, 0, :, :], self.reconstruction.zo, self.reconstruction.wavelength,
                        self.reconstruction.dxo, self.reconstruction.dxd)

        # todo check if Q1 Q2 are bandlimited

        elif self.params.propagatorType == 'scaledPolychromeASP':
            if self.params.fftshiftSwitch:
                raise ValueError('scaledPolychromeASP propagatorType works only with fftshiftSwitch = False!')
            dummy = np.ones((self.reconstruction.nlambda, self.reconstruction.nosm, self.reconstruction.npsm,
                             1, self.reconstruction.Np, self.reconstruction.Np), dtype='complex64')
            self.reconstruction.Q1 = np.ones_like(dummy)
            self.reconstruction.Q2 = np.ones_like(dummy)
            for nlmabda in range(self.reconstruction.nlambda):
                for nosm in range(self.reconstruction.nosm):
                    for npsm in range(self.reconstruction.npsm):
                        _, self.reconstruction.Q1[nlmabda, nosm, npsm, 0, ...], self.reconstruction.Q2[
                            nlmabda, nosm, npsm, 0, ...] = scaledASP(
                            dummy[nlmabda, nosm, npsm, 0, :, :], self.reconstruction.zo,
                            self.reconstruction.spectralDensity[nlmabda], self.reconstruction.dxo,
                            self.reconstruction.dxd)

        elif self.params.propagatorType == 'twoStepPolychrome':
            if self.params.fftshiftSwitch:
                raise ValueError('twoStepPolychrome propagatorType works only with fftshiftSwitch = False!')
            dummy = np.ones((self.reconstruction.nlambda, self.reconstruction.nosm, self.reconstruction.npsm,
                             1, self.reconstruction.Np, self.reconstruction.Np), dtype='complex64')
            # self.reconstruction.quadraticPhase = np.ones_like(dummy)
            self.reconstruction.transferFunction = np.array(
                [[[[aspw(dummy[nlambda, nosm, npsm, nslice, :, :],
                         self.reconstruction.zo *
                         (1 - self.reconstruction.spectralDensity[0] / self.reconstruction.spectralDensity[nlambda]),
                         self.reconstruction.spectralDensity[nlambda],
                         self.reconstruction.Lp)[1]
                    for nslice in range(1)]
                   for npsm in range(self.reconstruction.npsm)]
                  for nosm in range(self.reconstruction.nosm)]
                 for nlambda in range(self.reconstruction.nlambda)])
            self.reconstruction.quadraticPhase = np.exp(
                1.j * np.pi / (self.reconstruction.spectralDensity[0] * self.reconstruction.zo)
                * (self.reconstruction.Xp ** 2 + self.reconstruction.Yp ** 2))

    def _checkMISC(self):
        """
        checks miscellaneous quantities specific certain Engines
        """
        if self.params.backgroundModeSwitch:
            self.reconstruction.background = 1e-1 * np.ones((self.reconstruction.Np, self.reconstruction.Np))

        # preallocate intensity scaling vector
        if self.params.intensityConstraint == 'fluctuation':
            self.intensityScaling = np.ones(self.experimentalData.numFrames)

        if self.params.intensityConstraint == 'interferometric':
            self.reconstruction.reference = np.ones(self.reconstruction.probe[0, 0, 0, 0, ...].shape)

        # check if both probePoprobePowerCorrectionSwitch and modulusEnforcedProbeSwitch are on.
        # Since this can cause a contradiction, it raises an error
        if self.params.probePowerCorrectionSwitch and self.params.modulusEnforcedProbeSwitch:
            raise ValueError('probePowerCorrectionSwitch and modulusEnforcedProbeSwitch '
                             'can not simultaneously be switched on!')

        if not self.params.fftshiftSwitch:
            warnings.warn('fftshiftSwitch set to false, this may lead to reduced performance')

        if self.params.propagatorType == 'ASP' and self.params.fftshiftSwitch:
            raise ValueError('ASP propagatorType works only with fftshiftSwitch = False')
        if self.params.propagatorType == 'scaledASP' and self.params.fftshiftSwitch:
            raise ValueError('scaledASP propagatorType works only with fftshiftSwitch = False')

        if self.params.CPSCswitch:
            if not hasattr(self.experimentalData, 'ptychogramDownsampled'):
                if self.params.CPSCupsamplingFactor == None:
                    raise ValueError('CPSCswitch is on, CPSCupsamplingFactor need to be set')
                else:
                    self._setCPSC()

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
                if hasattr(self.experimentalData, 'ptychogramDownsampled'):
                    self.experimentalData.ptychogramDownsampled = np.fft.ifftshift(
                        self.experimentalData.ptychogramDownsampled, axes=(-1, -2))
                if self.experimentalData.W is not None:
                    self.experimentalData.W = np.fft.ifftshift(self.experimentalData.W, axes=(-1, -2))
                if self.experimentalData.emptyBeam is not None:
                    self.experimentalData.emptyBeam = np.fft.ifftshift(
                        self.experimentalData.emptyBeam, axes=(-1, -2))
                if self.experimentalData.PSD is not None:
                    self.experimentalData.PSD = np.fft.ifftshift(
                        self.experimentalData.PSD, axes=(-1, -2))
                self.params.fftshiftFlag = 1
        else:
            if self.params.fftshiftFlag == 1:
                print('check fftshift...')
                print('ifftshift data')
                self.experimentalData.ptychogram = np.fft.fftshift(self.experimentalData.ptychogram, axes=(-1, -2))
                if hasattr(self.experimentalData, 'ptychogramDownsampled'):
                    self.experimentalData.ptychogramDownsampled = np.fft.fftshift(
                        self.experimentalData.ptychogramDownsampled, axes=(-1, -2))
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

        self.reconstruction._move_data_to_gpu()
        self.experimentalData._move_data_to_gpu()

        transfer_fields_to_gpu(self, ['probeWindow',], self.logger)# '.probeWindow = cp.array(self.probeWindow)

        # reconstruction parameters
        # self.reconstruction.probe = cp.array(self.reconstruction.probe, cp.complex64)
        # self.reconstruction.object = cp.array(self.reconstruction.object, cp.complex64)
        # self.reconstruction.detectorError = cp.array(self.reconstruction.detectorError, cp.float32)

        # if self.params.momentumAcceleration:
        #     self.reconstruction.probeBuffer = cp.array(self.reconstruction.probeBuffer, cp.complex64)
        #     self.reconstruction.objectBuffer = cp.array(self.reconstruction.objectBuffer, cp.complex64)
        #     self.reconstruction.probeMomentum = cp.array(self.reconstruction.probeMomentum, cp.complex64)
        #     self.reconstruction.objectMomentum = cp.array(self.reconstruction.objectMomentum, cp.complex64)

        # for doing the coordinate transform and especially the otherwise slow interpolation of aPIE on the gpu
        if hasattr(self.params, 'aPIEflag'):
            if self.params.aPIEflag == True:
                fields_to_transfer = [
                    'ptychogramUntransformed',
                'Uq',
                'Vq',
                'theta',
                'wavelength',
                'Xd',
                'Yd'
                'dxd',
                'zo',
                ]
                self.theta = self.reconstruction.theta
                self.wavelength = self.reconstruction.wavelength

                transfer_fields_to_gpu(self, fields_to_transfer, self.logger)
                # self.ptychogramUntransformed = cp.array(self.ptychogramUntransformed)
                # self.Uq = cp.array(self.Uq)
                # self.Vq = cp.array(self.Vq)
                # self.theta = cp.array(self.reconstruction.theta)
                # self.wavelength = cp.array(self.reconstruction.wavelength)
                # self.Xd = cp.array(self.Xd)
                # self.Yd = cp.array(self.Yd)
                # self.dxd = cp.array(self.dxd)
                # self.zo = cp.array(self.zo)
                # self.experimentalData.W = cp.array(self.experimentalData.W)

        # non-reconstruction parameters
        # if hasattr(self.experimentalData, 'ptychogramDownsampled'):
        #     self.experimentalData.ptychogramDownsampled = cp.array(self.experimentalData.ptychogramDownsampled,
        #                                                            cp.float32)
        # else:
        #     self.experimentalData.ptychogram = cp.array(self.experimentalData.ptychogram, cp.float32)

        # propagators
        # if self.params.propagatorType == 'Fresnel':
        #     self.reconstruction.quadraticPhase = cp.array(self.reconstruction.quadraticPhase)
        # elif self.params.propagatorType == 'ASP' or self.params.propagatorType == 'polychromeASP':
        #     self.reconstruction.transferFunction = cp.array(self.reconstruction.transferFunction)
        # elif self.params.propagatorType == 'scaledASP' or self.params.propagatorType == 'scaledPolychromeASP':
        #     self.reconstruction.Q1 = cp.array(self.reconstruction.Q1)
        #     self.reconstruction.Q2 = cp.array(self.reconstruction.Q2)
        # elif self.params.propagatorType == 'twoStepPolychrome':
        #     self.reconstruction.quadraticPhase = cp.array(self.reconstruction.quadraticPhase)
        #     self.reconstruction.transferFunction = cp.array(self.reconstruction.transferFunction)

        # other parameters
        # if self.params.backgroundModeSwitch:
        #     self.reconstruction.background = cp.array(self.reconstruction.background)
        # if self.params.absorbingProbeBoundary or self.params.probeBoundary:

        # if self.params.modulusEnforcedProbeSwitch:
        #     self.experimentalData.emptyBeam = cp.array(self.experimentalData.emptyBeam)
        # if self.params.intensityConstraint == 'interferometric':
        #     self.reconstruction.reference = cp.array(self.reconstruction.reference)


    def _move_data_to_cpu(self):
        """
        Move the data to the CPU, called when the gpuSwitch is off.
        :return:
        """
        # reconstruction parameters
        from fracPy.utils.gpuUtils import asNumpyArray

        self.reconstruction._move_data_to_cpu()
        self.experimentalData._move_data_to_cpu()
        transfer_fields_to_cpu(self, ['probeWindow', ], self.logger)

        # self.reconstruction.move_to_CPU()
        # self.params.move_to_CPU()

        # self.reconstruction.probe = asNumpyArray(self.reconstruction.probe)
        # self.reconstruction.object = asNumpyArray(self.reconstruction.object)

        # if self.params.momentumAcceleration:
            # reconstruction_fields_to_transfer = ['probeBuffer', 'objectBuffer', 'probeMomentum',' objectMomentum']
            # for field in reconstruction_fields_to_transfer:
            #
            #     setattr(self.reconstruction, field,)
            # self.reconstruction.probeBuffer = self.reconstruction.probeBuffer.get()
            # self.reconstruction.objectBuffer = self.reconstruction.objectBuffer.get()
            # self.reconstruction.probeMomentum = self.reconstruction.probeMomentum.get()
            # self.reconstruction.objectMomentum = self.reconstruction.objectMomentum.get()

            # for doing the coordinate transform and especially the otherwise slow interpolation of aPIE on the gpu
        # if hasattr(self.params, 'aPIEflag'):
        #     if self.params.aPIEflag:
        #         self.theta = self.theta.get()
        #
        fields_to_transfer = ['theta', 'probeWindow']
        # self.probeWindow = self.probeWindow.get()

        # non-reconstruction parameters
        # if hasattr(self.experimentalData, 'ptychogramDownsampled'):
        #     self.experimentalData.ptychogramDownsampled = self.experimentalData.ptychogramDownsampled.get()
        # else:
        #     self.experimentalData.ptychogram = self.experimentalData.ptychogram.get()
        # self.reconstruction.detectorError = self.reconstruction.detectorError.get()

        # propagators
        # if self.params.propagatorType == 'Fresnel':
            # self.reconstruction.quadraticPhase = self.reconstruction.quadraticPhase.get()
        # elif self.params.propagatorType == 'ASP' or self.params.propagatorType == 'polychromeASP':
        #     self.reconstruction.transferFunction = self.reconstruction.transferFunction.get()
        # elif self.params.propagatorType == 'scaledASP' or self.params.propagatorType == 'scaledPolychromeASP':
        #     self.reconstruction.Q1 = self.reconstruction.Q1.get()
        #     self.reconstruction.Q2 = self.reconstruction.Q2.get()
        # elif self.params.propagatorType == 'twoStepPolychrome':
        #     self.reconstruction.quadraticPhase = self.reconstruction.quadraticPhase.get()
        #     self.reconstruction.transferFunction = self.reconstruction.transferFunction.get()

        # other parameters
        # if self.params.backgroundModeSwitch:
        #     # self.reconstruction.background = self.reconstruction.background.get()
        # if self.params.absorbingProbeBoundary or self.params.probeBoundary:

        # if self.params.modulusEnforcedProbeSwitch:
        #     self.experimentalData.emptyBeam = self.experimentalData.emptyBeam.get()
        # # if self.params.intensityConstraint == 'interferometric':
        #     self.reconstruction.reference = self.reconstruction.reference.get()

    def _checkGPU(self):
        if not hasattr(self.params, 'gpuFlag'):
            self.params.gpuFlag = 0

        if self.params.gpuSwitch:
            if cp is None:
                raise ImportError('Could not import cupy, therefore no GPU reconstruction is possible. To reconstruct, set the params.gpuSwitch to False.')
            if not self.params.gpuFlag:
                self.logger.info('switch to gpu')

                # load data to gpu
                self._move_data_to_gpu()
                self.params.gpuFlag = 1
            # always do this as it gets away with hard to debug errors
            self._move_data_to_gpu()
        else:
            self._move_data_to_cpu()
            if self.params.gpuFlag:
                self.logger.info('switch to cpu')
                self._move_data_to_cpu()
                self.params.gpuFlag = 0

    def setPositionOrder(self):
        if self.params.positionOrder == 'sequential':
            self.positionIndices = np.arange(self.experimentalData.numFrames)

        elif self.params.positionOrder == 'random':
            if len(self.reconstruction.error) == 0:
                self.positionIndices = np.arange(self.experimentalData.numFrames)
            else:
                if len(self.reconstruction.error) < 2:
                    self.positionIndices = np.arange(self.experimentalData.numFrames)
                else:
                    self.positionIndices = np.arange(self.experimentalData.numFrames)
                    np.random.shuffle(self.positionIndices)

        # order by illumiantion angles. Use smallest angles first
        # (i.e. start with brightfield data first, then add the low SNR
        # darkfield)
        # todo check this with Antonios
        elif self.params.positionOrder == 'NA':
            rows = self.reconstruction.positions[:, 0] - np.mean(self.reconstruction.positions[:, 0])
            cols = self.reconstruction.positions[:, 1] - np.mean(self.reconstruction.positions[:, 1])
            dist = np.sqrt(rows ** 2 + cols ** 2)
            self.positionIndices = np.argsort(dist)
        else:
            raise ValueError('position order not properly set')

    def changeExperimentalData(self, experimentalData: ExperimentalData):
        self.experimentalData = experimentalData

    def changeOptimizable(self, optimizable: Reconstruction):
        self.reconstruction = optimizable

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
        if self.params.propagatorType == 'Fraunhofer':
            self.fft2s()
        elif self.params.propagatorType == 'Fresnel':
            self.reconstruction.esw = self.reconstruction.esw * self.reconstruction.quadraticPhase
            self.fft2s()
        elif self.params.propagatorType == 'ASP' or self.params.propagatorType == 'polychromeASP':
            self.reconstruction.ESW = ifft2c(fft2c(self.reconstruction.esw) * self.reconstruction.transferFunction)
        elif self.params.propagatorType == 'scaledASP' or self.params.propagatorType == 'scaledPolychromeASP':
            self.reconstruction.ESW = ifft2c(fft2c(self.reconstruction.esw * self.reconstruction.Q1) * self.reconstruction.Q2)
        elif self.params.propagatorType == 'twoStepPolychrome':
            self.reconstruction.esw = ifft2c(fft2c(self.reconstruction.esw) * self.reconstruction.transferFunction) * \
                                   self.reconstruction.quadraticPhase
            self.fft2s()
        else:
            raise Exception('Propagator is not properly set, choose from Fraunhofer, Fresnel, ASP and scaledASP')

    def detector2object(self):
        """
        Propagate the ESW to the object plane (in-place).

        Matches: detector2object.m
        :return:
        """
        if self.params.propagatorType == 'Fraunhofer':
            self.ifft2s()
        elif self.params.propagatorType == 'Fresnel':
            self.ifft2s()
            self.reconstruction.esw = self.reconstruction.esw * self.reconstruction.quadraticPhase.conj()
            self.reconstruction.eswUpdate = self.reconstruction.eswUpdate * self.reconstruction.quadraticPhase.conj()
        elif self.params.propagatorType == 'ASP' or self.params.propagatorType == 'polychromeASP':
            self.reconstruction.eswUpdate = ifft2c(fft2c(self.reconstruction.ESW) * self.reconstruction.transferFunction.conj())
        elif self.params.propagatorType == 'scaledASP' or self.params.propagatorType == 'scaledPolychromeASP':
            self.reconstruction.eswUpdate = ifft2c(fft2c(self.reconstruction.ESW) * self.reconstruction.Q2.conj()) \
                                         * self.reconstruction.Q1.conj()
        elif self.params.propagatorType == 'twoStepPolychrome':
            self.ifft2s()
            self.reconstruction.esw = ifft2c(fft2c(self.reconstruction.esw *
                                                self.reconstruction.quadraticPhase.conj()) *
                                          self.reconstruction.transferFunction.conj())
            self.reconstruction.eswUpdate = ifft2c(fft2c(self.reconstruction.eswUpdate *
                                                      self.reconstruction.quadraticPhase.conj()) *
                                                self.reconstruction.transferFunction.conj())
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
        xp = getArrayModule(self.reconstruction.esw)

        if self.params.fftshiftSwitch:
            self.reconstruction.ESW = xp.fft.fft2(self.reconstruction.esw, norm='ortho')
        else:
            self.reconstruction.ESW = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(self.reconstruction.esw), norm='ortho'))

    def getBeamWidth(self):
        """
        Calculate probe beam width (Full width half maximum)
        :return:
        """
        P = np.sum(abs(asNumpyArray(self.reconstruction.probe[..., -1, :, :])) ** 2, axis=(0, 1, 2))
        P = P / np.sum(P, axis=(-1, -2))
        xMean = np.sum(self.reconstruction.Xp * P, axis=(-1, -2))
        yMean = np.sum(self.reconstruction.Yp * P, axis=(-1, -2))
        xVariance = np.sum((self.reconstruction.Xp - xMean) ** 2 * P, axis=(-1, -2))
        yVariance = np.sum((self.reconstruction.Yp - yMean) ** 2 * P, axis=(-1, -2))

        c = 2 * np.sqrt(2 * np.log(
            2))  # constant for converting variance to FWHM (see e.g. https://en.wikipedia.org/wiki/Full_width_at_half_maximum)
        self.reconstruction.beamWidthX = c * np.sqrt(xVariance)
        self.reconstruction.beamWidthY = c * np.sqrt(yVariance)

    def getOverlap(self, ind1, ind2):
        """
        Calculate linear and area overlap between two scan positions indexed ind1 and ind2
        """
        sy = abs(self.reconstruction.positions[ind2, 0] - self.reconstruction.positions[ind1, 0]) * self.reconstruction.dxp
        sx = abs(self.reconstruction.positions[ind2, 1] - self.reconstruction.positions[ind1, 1]) * self.reconstruction.dxp

        # task 1: get linear overlap
        self.getBeamWidth()
        self.reconstruction.linearOverlap = 1 - np.sqrt(sx ** 2 + sy ** 2) / \
                                         np.minimum(self.reconstruction.beamWidthX, self.reconstruction.beamWidthY)
        self.reconstruction.linearOverlap = np.maximum(self.reconstruction.linearOverlap, 0)

        # task 2: get area overlap
        # spatial frequency pixel size
        df = 1 / (self.reconstruction.Np * self.reconstruction.dxp)
        # spatial frequency meshgrid
        fx = np.arange(-self.reconstruction.Np // 2, self.reconstruction.Np // 2) * df
        Fx, Fy = np.meshgrid(fx, fx)
        # absolute value of probe and 2D fft
        P = abs(asNumpyArray(self.reconstruction.probe[:, 0, 0, -1, ...]))
        Q = fft2c(P)
        # calculate overlap between positions
        self.reconstruction.areaOverlap = np.mean(
            abs(np.sum(Q ** 2 * np.exp(-1.j * 2 * np.pi * (Fx * sx + Fy * sy)), axis=(-1, -2))) / \
            np.sum(abs(Q) ** 2, axis=(-1, -2)), axis=0)

    def getErrorMetrics(self):
        """
        matches getErrorMetrics.m
        :return:
        """
        if not self.params.saveMemory:
            # Calculate mean error for all positions (make separate function for all of that)
            if self.params.FourierMaskSwitch:
                self.reconstruction.errorAtPos = np.sum(np.abs(self.reconstruction.detectorError) *
                                                     self.experimentalData.W, axis=(-1, -2))
            else:
                self.reconstruction.errorAtPos = np.sum(np.abs(self.reconstruction.detectorError), axis=(-1, -2))
        self.reconstruction.errorAtPos = asNumpyArray(self.reconstruction.errorAtPos)/asNumpyArray(self.experimentalData.energyAtPos + 1e-20)
        eAverage = np.sum(self.reconstruction.errorAtPos)

        # append to error vector (for plotting error as function of iteration)
        self.reconstruction.error = np.append(self.reconstruction.error, eAverage)

    def getRMSD(self, positionIndex):
        """
        Root mean square deviation between ptychogram and intensity estimate
        :param positionIndex:
        :return:
        """
        # find out wether or not to use the GPU
        xp = getArrayModule(self.reconstruction.Iestimated)
        self.currentDetectorError = abs(self.reconstruction.Imeasured - self.reconstruction.Iestimated)

        # todo saveMemory implementation
        if self.params.saveMemory:
            if self.params.FourierMaskSwitch and not self.params.CPSCswitch:
                self.reconstruction.errorAtPos[positionIndex] = xp.sum(self.currentDetectorError * self.experimentalData.W)
            elif self.params.FourierMaskSwitch and self.params.CPSCswitch:
                raise NotImplementedError
            else:
                self.reconstruction.errorAtPos[positionIndex] = xp.sum(self.currentDetectorError)
        else:
            self.reconstruction.detectorError[positionIndex] = self.currentDetectorError

    def ifft2s(self):
        """ Inverse FFT"""
        # find out if this should be performed on the GPU
        xp = getArrayModule(self.reconstruction.esw)

        if self.params.fftshiftSwitch:
            self.reconstruction.eswUpdate = xp.fft.ifft2(self.reconstruction.ESW, norm='ortho')
        else:
            self.reconstruction.eswUpdate = xp.fft.fftshift(
                xp.fft.ifft2(xp.fft.ifftshift(self.reconstruction.ESW), norm='ortho'))

    def intensityProjection(self, positionIndex):
        """ Compute the projected intensity.
            Barebones, need to implement other methods
        """
        # figure out whether or not to use the GPU
        xp = getArrayModule(self.reconstruction.esw)

        # zero division mitigator
        gimmel = 1e-10

        # propagate to detector
        self.object2detector()

        # get estimated intensity (2D array, in the case of multislice, only take the last slice)
        if self.params.intensityConstraint == 'interferometric':
            self.reconstruction.Iestimated = \
            xp.sum(xp.abs(self.reconstruction.ESW + self.reconstruction.reference) ** 2, axis=(0, 1, 2))[-1]
        else:
            self.reconstruction.Iestimated = xp.sum(xp.abs(self.reconstruction.ESW) ** 2, axis=(0, 1, 2))[-1]
        if self.params.backgroundModeSwitch:
            self.reconstruction.Iestimated += self.reconstruction.background

        # get measured intensity todo implement kPIE
        if self.params.CPSCswitch:
            self.decompressionProjection(positionIndex)
        else:
            self.reconstruction.Imeasured = self.experimentalData.ptychogram[positionIndex]

        self.getRMSD(positionIndex)

        # adaptive denoising
        if self.params.adaptiveDenoisingSwitch:
            self.adaptiveDenoising()

        # intensity projection constraints
        if self.params.intensityConstraint == 'fluctuation':
            # scaling
            if self.params.FourierMaskSwitch:
                aleph = xp.sum(self.reconstruction.Imeasured * self.reconstruction.Iestimated * self.experimentalData.W) / \
                        xp.sum(self.reconstruction.Imeasured * self.reconstruction.Imeasured * self.experimentalData.W)
            else:
                aleph = xp.sum(self.reconstruction.Imeasured * self.reconstruction.Iestimated) / \
                        xp.sum(self.reconstruction.Imeasured * self.reconstruction.Imeasured)
            self.params.intensityScaling[positionIndex] = aleph
            # scaled projection
            frac = (1 + aleph) / 2 * self.reconstruction.Imeasured / (self.reconstruction.Iestimated + gimmel)

        elif self.params.intensityConstraint == 'exponential':
            x = self.currentDetectorError / (self.reconstruction.Iestimated + gimmel)
            W = xp.exp(-0.05 * x)
            frac = xp.sqrt(self.reconstruction.Imeasured / (self.reconstruction.Iestimated + gimmel))
            frac = W * frac + (1 - W)

        elif self.params.intensityConstraint == 'poission':
            frac = self.reconstruction.Imeasured / (self.reconstruction.Iestimated + gimmel)

        elif self.params.intensityConstraint == 'standard' or self.params.intensityConstraint == 'interferometric':
            frac = xp.sqrt(self.reconstruction.Imeasured / (self.reconstruction.Iestimated + gimmel))

        else:
            raise ValueError('intensity constraint not properly specified!')

        # apply mask
        if self.params.FourierMaskSwitch and self.params.CPSCswitch and len(self.reconstruction.error) > 5:
            frac = self.experimentalData.W * frac + (1 - self.experimentalData.W)

        # update ESW
        if self.params.intensityConstraint == 'interferometric':
            temp = (self.reconstruction.ESW + self.reconstruction.reference) * frac - self.reconstruction.ESW
            self.reconstruction.ESW = (
                                               self.reconstruction.ESW + self.reconstruction.reference) * frac - self.reconstruction.reference
            self.reconstruction.reference = temp
        else:
            self.reconstruction.ESW = self.reconstruction.ESW * frac

        # update background (see PhD thsis by Peng Li)
        if self.params.backgroundModeSwitch:
            if self.params.FourierMaskSwitch:
                self.reconstruction.background = self.reconstruction.background * (1 + 1 / self.experimentalData.numFrames * (
                            xp.sqrt(frac) - 1)) ** 2 * self.experimentalData.W
            else:
                self.reconstruction.background = self.reconstruction.background * (
                            1 + 1 / self.experimentalData.numFrames * (xp.sqrt(frac) - 1)) ** 2

        # back propagate to object plane
        self.detector2object()

    def decompressionProjection(self, positionIndex):
        """
        calculate the upsampled Imeasured from downsampled Imeasured that is actually measured.
        :param positionIndex: index for scan positions
        :return:
        """
        # overwrite the measured intensity (just to have same dimensions as Iestimated)
        xp = getArrayModule(self.reconstruction.Iestimated)

        # determine downsampled fraction (Sl)
        frac = self.experimentalData.ptychogramDownsampled[positionIndex] / \
               (xp.sum(self.reconstruction.Iestimated.reshape(self.reconstruction.Nd // self.params.CPSCupsamplingFactor,
                                                           self.params.CPSCupsamplingFactor,
                                                           self.reconstruction.Nd // self.params.CPSCupsamplingFactor,
                                                           self.params.CPSCupsamplingFactor), axis=(1, 3)) + np.finfo(
                   np.float32).eps)
        if self.params.FourierMaskSwitch and len(self.reconstruction.error) > 5:
            frac = self.experimentalData.W * frac + (1 - self.experimentalData.W)
        # overwrite up-sampled measured intensity
        self.reconstruction.Imeasured = self.reconstruction.Iestimated * xp.repeat(
            xp.repeat(frac, self.params.CPSCupsamplingFactor, axis=-1), self.params.CPSCupsamplingFactor, axis=-2)

    def showReconstruction(self, loop):
        """
        Show the reconstruction process.
        :param loop: the iteration number
        :return:
        """
        if np.mod(loop, self.monitor.figureUpdateFrequency) == 0:

            if self.experimentalData.operationMode == 'FPM':
                object_estimate = np.squeeze(asNumpyArray(
                    fft2c(self.reconstruction.object)[..., self.monitor.objectROI[0], self.monitor.objectROI[1]]))
                probe_estimate = np.squeeze(asNumpyArray(
                    self.reconstruction.probe[..., self.monitor.probeROI[0], self.monitor.probeROI[1]]))
            else:
                object_estimate = np.squeeze(asNumpyArray(
                    self.reconstruction.object[..., self.monitor.objectROI[0], self.monitor.objectROI[1]]))
                probe_estimate = np.squeeze(asNumpyArray(
                    self.reconstruction.probe[0, ..., self.monitor.probeROI[0], self.monitor.probeROI[1]]))

            self.monitor.updateObjectProbeErrorMonitor(object_estimate=object_estimate, probe_estimate=probe_estimate)

            if self.monitor.verboseLevel == 'high':
                if self.params.fftshiftSwitch:
                    Iestimated = np.fft.fftshift(asNumpyArray(self.reconstruction.Iestimated))
                    Imeasured = np.fft.fftshift(asNumpyArray(self.reconstruction.Imeasured))
                else:
                    Iestimated = asNumpyArray(self.reconstruction.Iestimated)
                    Imeasured = asNumpyArray(self.reconstruction.Imeasured)

                self.monitor.updateDiffractionDataMonitor(Iestimated=Iestimated, Imeasured=Imeasured)

                self.getOverlap(0, 1)

                self.pbar.write('')
                self.pbar.write('iteration: %i' % loop)
                self.pbar.write('error: %.1f' % self.reconstruction.error[-1])
                self.pbar.write('estimated linear overlap: %.1f %%' % (100 * self.reconstruction.linearOverlap))
                self.pbar.write('estimated area overlap: %.1f %%' % (100 * self.reconstruction.areaOverlap))
                # self.pbar.write('coherence structure:')

            if self.params.positionCorrectionSwitch:
                # show reconstruction
                if (len(self.reconstruction.error) > self.startAtIteration):  # & (np.mod(loop,
                    # self.monitor.figureUpdateFrequency) == 0):
                    figure, ax = plt.subplots(1, 1, num=102, squeeze=True, clear=True, figsize=(5, 5))
                    ax.set_title('Estimated scan grid positions')
                    ax.set_xlabel('(um)')
                    ax.set_ylabel('(um)')
                    # ax.set_xscale('symlog')
                    line1, = plt.plot((self.reconstruction.positions0[:, 1] - self.reconstruction.No // 2 + self.reconstruction.Np // 2)* self.reconstruction.dxo * 1e6,
                                      (self.reconstruction.positions0[:, 0]- self.reconstruction.No // 2 + self.reconstruction.Np // 2) * self.reconstruction.dxo * 1e6, 'bo', label='before correction')
                    line2, = plt.plot((self.reconstruction.positions[:, 1]- self.reconstruction.No // 2 + self.reconstruction.Np // 2) * self.reconstruction.dxo * 1e6,
                                      (self.reconstruction.positions[:, 0]- self.reconstruction.No // 2 + self.reconstruction.Np // 2) * self.reconstruction.dxo * 1e6, 'yo', label='after correction')
                    # plt.xlabel('(um))')
                    # plt.ylabel('(um))')
                    # plt.show()
                    plt.legend(handles=[line1, line2])
                    plt.tight_layout()
                    plt.show(block=False)

                    figure2, ax2 = plt.subplots(1, 1, num=103, squeeze=True, clear=True, figsize=(5, 5))
                    ax2.set_title('Displacement')
                    ax2.set_xlabel('(um)')
                    ax2.set_ylabel('(um)')
                    plt.plot(self.D[:, 1] * self.reconstruction.dxo * 1e6,
                             self.D[:, 0] * self.reconstruction.dxo * 1e6, 'o')
                    # ax.set_xscale('symlog')
                    plt.tight_layout()
                    plt.show(block=False)

                    # elif np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
                    figure.canvas.draw()
                    figure.canvas.flush_events()
                    figure2.canvas.draw()
                    figure2.canvas.flush_events()
                    # self.showReconstruction(loop)
            # print('iteration:%i' %len(self.reconstruction.error))
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
        if len(self.reconstruction.error) > self.startAtIteration:
            # position gradients
            # shiftedImages = xp.zeros((self.rowShifts.shape + objectPatch.shape))
            cc = xp.zeros((len(self.rowShifts), 1))
            
            # use the real-space object (FFT for FPM)
            if self.experimentalData.operationMode =='FPM':
                O = fft2c(self.reconstruction.object)
                Opatch = fft2c(objectPatch)
            elif self.experimentalData.operationMode =='CPM':
                O = self.reconstruction.object
                Opatch = objectPatch
                
                
            for shifts in range(len(self.rowShifts)):
                tempShift = xp.roll(Opatch, self.rowShifts[shifts], axis=-2)
                # shiftedImages[shifts, ...] = xp.roll(tempShift, self.colShifts[shifts], axis=-1)
                shiftedImages = xp.roll(tempShift, self.colShifts[shifts], axis=-1)
                cc[shifts] = xp.squeeze(xp.sum(shiftedImages.conj() * O[..., sy, sx],
                                               axis=(-2, -1)))
            # truncated cross - correlation
            # cc = xp.squeeze(xp.sum(shiftedImages.conj() * self.reconstruction.object[..., sy, sx], axis=(-2, -1)))
            cc = abs(cc)
            betaGrad = 1000
            normFactor = xp.sum(Opatch.conj() * Opatch, axis=(-2, -1)).real
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
        if len(self.reconstruction.error) > self.startAtIteration:
            
            # update positions
            if self.experimentalData.operationMode == 'FPM':
                conv = -(1 / self.reconstruction.wavelength) * self.reconstruction.dxo * self.reconstruction.Np
                z = self.reconstruction.zled
                k = self.reconstruction.positions - self.adaptStep * self.D - \
                                                 self.reconstruction.No // 2 + self.reconstruction.Np // 2
                self.experimentalData.encoder = np.sign(conv) *  k * z / (np.sqrt(conv**2-k[:,0]**2-k[:,1]**2))[...,None]
            else:
                self.experimentalData.encoder = (self.reconstruction.positions - self.adaptStep * self.D -
                                                 self.reconstruction.No // 2 + self.reconstruction.Np // 2) * \
                                                self.reconstruction.dxo
                                                
            # fix center of mass of positions
            self.experimentalData.encoder[:, 0] = self.experimentalData.encoder[:, 0] - \
                                                  np.mean(self.experimentalData.encoder[:, 0]) + self.meanEncoder00
            self.experimentalData.encoder[:, 1] = self.experimentalData.encoder[:, 1] - \
                                                  np.mean(self.experimentalData.encoder[:, 1]) + self.meanEncoder01

            # self.reconstruction.positions[:,0] = self.reconstruction.positions[:,0] - \
            #         np.round(np.mean(self.reconstruction.positions[:,0]) -
            #                   np.mean(self.reconstruction.positions0[:,0]) )
            # self.reconstruction.positions[:, 1] = self.reconstruction.positions[:, 1] - \
            #                                         np.around(np.mean(self.reconstruction.positions[:, 1]) -
            #                                                   np.mean(self.reconstruction.positions0[:, 1]))

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
            self.reconstruction.probe = self.reconstruction.probe / np.sqrt(
                np.sum(self.reconstruction.probe * self.reconstruction.probe.conj())) * self.experimentalData.maxProbePower

        if self.params.comStabilizationSwitch:
            self.comStabilization()

        if self.params.PSDestimationSwitch:
            raise NotImplementedError()

        if self.params.probeBoundary:
            self.reconstruction.probe *= self.probeWindow

        if self.params.absorbingProbeBoundary:
            if self.experimentalData.operationMode == 'FPM':
                self.absorbingProbeBoundaryAleph = 1

            self.reconstruction.probe = (1 - self.params.absorbingProbeBoundaryAleph) * self.reconstruction.probe + \
                                     self.params.absorbingProbeBoundaryAleph * self.reconstruction.probe * self.probeWindow

        # Todo: objectSmoothenessSwitch,probeSmoothenessSwitch,
        if self.params.probeSmoothenessSwitch:
            raise NotImplementedError()

        if self.params.objectSmoothenessSwitch:
            raise NotImplementedError()

        if self.params.absObjectSwitch:
            self.reconstruction.object = (1 - self.params.absObjectBeta) * self.reconstruction.object + \
                                      self.params.absObjectBeta * abs(self.reconstruction.object)

        if self.params.absProbeSwitch:
            self.reconstruction.probe = (1 - self.params.absProbeBeta) * self.reconstruction.probe + \
                                     self.params.absProbeBeta * abs(self.reconstruction.probe)

        # this is intended to slowly push non-measured object region to abs value lower than
        # the max abs inside object ROI allowing for good contrast when monitoring object
        if self.params.objectContrastSwitch:
            self.reconstruction.object = 0.995 * self.reconstruction.object + 0.005 * \
                                      np.mean(abs(self.reconstruction.object[..., self.monitor.objectROI[0],
                                                                          self.monitor.objectROI[1]]))
        if self.params.couplingSwitch and self.reconstruction.nlambda > 1:
            self.reconstruction.probe[0] = (1 - self.params.couplingAleph) * self.reconstruction.probe[0] + \
                                        self.params.couplingAleph * self.reconstruction.probe[1]
            for lambdaLoop in np.arange(1, self.reconstruction.nlambda - 1):
                self.reconstruction.probe[lambdaLoop] = (1 - self.params.couplingAleph) * self.reconstruction.probe[
                    lambdaLoop] + \
                                                     self.params.couplingAleph * (
                                                                 self.reconstruction.probe[lambdaLoop + 1] +
                                                                 self.reconstruction.probe[
                                                                     lambdaLoop - 1]) / 2

            self.reconstruction.probe[-1] = (1 - self.params.couplingAleph) * self.reconstruction.probe[-1] + \
                                         self.params.couplingAleph * self.reconstruction.probe[-2]
        if self.params.binaryProbeSwitch:
            probePeakAmplitude = np.max(abs(self.reconstruction.probe))
            probeThresholded = self.reconstruction.probe.copy()
            probeThresholded[(abs(probeThresholded) < self.params.binaryProbeThreshold * probePeakAmplitude)] = 0

            self.reconstruction.probe = (1 - self.params.binaryProbeAleph) * self.reconstruction.probe + \
                                     self.params.binaryProbeAleph * probeThresholded

        if self.params.positionCorrectionSwitch:
            self.positionCorrectionUpdate()

    def orthogonalization(self):
        """
        Perform orthogonalization
        :return:
        """
        xp = getArrayModule(self.reconstruction.probe)
        if self.reconstruction.npsm > 1:
            # orthogonalize the probe for each wavelength and each slice
            for id_l in range(self.reconstruction.nlambda):
                for id_s in range(self.reconstruction.nslice):
                    self.reconstruction.probe[id_l, 0, :, id_s, :, :], self.normalizedEigenvaluesProbe, self.MSPVprobe = \
                        orthogonalizeModes(self.reconstruction.probe[id_l, 0, :, id_s, :, :], method='snapShots')
                    self.reconstruction.purityProbe = np.sqrt(np.sum(self.normalizedEigenvaluesProbe ** 2))

                    # orthogonolize momentum operator
                    if self.params.momentumAcceleration:
                        # orthogonalize probe Buffer
                        p = self.reconstruction.probeBuffer[id_l, 0, :, id_s, :, :].reshape(
                            (self.reconstruction.npsm, self.reconstruction.Np ** 2))
                        self.reconstruction.probeBuffer[id_l, 0, :, id_s, :, :] = (xp.array(self.MSPVprobe) @ p).reshape(
                            (self.reconstruction.npsm, self.reconstruction.Np, self.reconstruction.Np))
                        # orthogonalize probe momentum
                        p = self.reconstruction.probeMomentum[id_l, 0, :, id_s, :, :].reshape(
                            (self.reconstruction.npsm, self.reconstruction.Np ** 2))
                        self.reconstruction.probeMomentum[id_l, 0, :, id_s, :, :] = (xp.array(self.MSPVprobe) @ p).reshape(
                            (self.reconstruction.npsm, self.reconstruction.Np, self.reconstruction.Np))

                        # if self.comStabilizationSwitch:
                        #     self.comStabilization()



        elif self.reconstruction.nosm > 1:
            # orthogonalize the object for each wavelength and each slice
            for id_l in range(self.reconstruction.nlambda):
                for id_s in range(self.reconstruction.nslice):
                    self.reconstruction.object[id_l, :, 0, id_s, :, :], self.normalizedEigenvaluesObject, self.MSPVobject = \
                        orthogonalizeModes(self.reconstruction.object[id_l, :, 0, id_s, :, :], method='snapShots')
                    self.reconstruction.purityObject = np.sqrt(np.sum(self.normalizedEigenvaluesObject ** 2))

                    # orthogonolize momentum operator
                    if self.params.momentumAcceleration:
                        # orthogonalize object Buffer
                        p = self.reconstruction.objectBuffer[id_l, :, 0, id_s, :, :].reshape(
                            (self.reconstruction.nosm, self.reconstruction.No ** 2))
                        self.reconstruction.objectBuffer[id_l, :, 0, id_s, :, :] = (xp.array(self.MSPVobject) @ p).reshape(
                            (self.reconstruction.nosm, self.reconstruction.No, self.reconstruction.No))
                        # orthogonalize object momentum
                        p = self.reconstruction.objectMomentum[id_l, :, 0, id_s, :, :].reshape(
                            (self.reconstruction.nosm, self.reconstruction.No ** 2))
                        self.reconstruction.objectMomentum[id_l, :, 0, id_s, :, :] = (
                                    xp.array(self.MSPVobject) @ p).reshape(
                            (self.reconstruction.nosm, self.reconstruction.No, self.reconstruction.No))

        else:
            pass

    def comStabilization(self):
        """
        Perform center of mass stabilization (center the probe)
        :return:
        """
        xp = getArrayModule(self.reconstruction.probe)
        # calculate center of mass of the probe (for multislice cases, the probe for the last slice is used)
        P2 = xp.sum(abs(self.reconstruction.probe[:, :, :, -1, ...]) ** 2, axis=(0, 1, 2))
        demon = xp.sum(P2) * self.reconstruction.dxp
        xc = xp.int(xp.around(xp.sum(xp.array(self.reconstruction.Xp, xp.float32) * P2) / demon))
        yc = xp.int(xp.around(xp.sum(xp.array(self.reconstruction.Yp, xp.float32) * P2) / demon))
        # shift only if necessary
        if xc ** 2 + yc ** 2 > 1:
            # shift probe
            for k in xp.arange(self.reconstruction.npsm):
                self.reconstruction.probe[:, :, k, -1, ...] = \
                    xp.roll(self.reconstruction.probe[:, :, k, -1, ...], (-yc, -xc), axis=(-2, -1))
                # for mPIE
                if self.params.momentumAcceleration:
                    self.reconstruction.probeMomentum[:, :, k, -1, ...] = \
                        xp.roll(self.reconstruction.probeMomentum[:, :, k, -1, ...], (-yc, -xc), axis=(-2, -1))
                    self.reconstruction.probeBuffer[:, :, k, -1, ...] = \
                        xp.roll(self.reconstruction.probeBuffer[:, :, k, -1, ...], (-yc, -xc), axis=(-2, -1))

            # shift object
            for k in xp.arange(self.reconstruction.nosm):
                self.reconstruction.object[:, k, :, -1, ...] = \
                    xp.roll(self.reconstruction.object[:, k, :, -1, ...], (-yc, -xc), axis=(-2, -1))
                # for mPIE
                if self.params.momentumAcceleration:
                    self.reconstruction.objectMomentum[:, k, :, -1, ...] = \
                        xp.roll(self.reconstruction.objectMomentum[:, k, :, -1, ...], (-yc, -xc), axis=(-2, -1))
                    self.reconstruction.objectBuffer[:, k, :, -1, ...] = \
                        xp.roll(self.reconstruction.objectBuffer[:, k, :, -1, ...], (-yc, -xc), axis=(-2, -1))

    def modulusEnforcedProbe(self):
        # propagate probe to detector
        xp = getArrayModule(self.reconstruction.esw)
        self.reconstruction.esw = self.reconstruction.probe
        self.object2detector()

        if self.params.FourierMaskSwitch:
            self.reconstruction.ESW = self.reconstruction.ESW * xp.sqrt(
                self.experimentalData.emptyBeam / 1e-10 + xp.sum(xp.abs(self.reconstruction.ESW) ** 2,
                                                                 axis=(0, 1, 2, 3))) * self.experimentalData.W \
                                   + self.reconstruction.ESW * (1 - self.experimentalData.W)
        else:
            self.reconstruction.ESW = self.reconstruction.ESW * np.sqrt(
                self.experimentalData.emptyBeam / (1e-10 + xp.sum(abs(self.reconstruction.ESW) ** 2, axis=(0, 1, 2, 3))))

        self.detector2object()
        self.reconstruction.probe = self.reconstruction.esw

    def adaptiveDenoising(self):
        """
        Use the difference of mean intensities between the low-resolution
        object estimate and the low-resolution raw data to estimate the
        noise floor to be clipped.
        :return:
        """
        # figure out wether or not to use the GPU
        xp = getArrayModule(self.reconstruction.esw)

        Ameasured = self.reconstruction.Imeasured ** 0.5
        Aestimated = xp.abs(self.reconstruction.Iestimated) ** 0.5

        noise = xp.abs(xp.mean(Ameasured - Aestimated))

        Ameasured = Ameasured - noise
        Ameasured[Ameasured < 0] = 0
        self.reconstruction.Imeasured = Ameasured ** 2

    def reconstruct(self):
        raise NotImplementedError('Implement the reconstruction method in the subclasses yourself.')