from fracPy.monitors.default_visualisation import DefaultMonitor,DiffractionDataMonitor
import numpy as np
from scipy.signal import get_window
import logging
import warnings
# fracPy imports
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.utils.initializationFunctions import initialProbeOrObject
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.utils.utils import ifft2c, fft2c, orthogonalizeModes
from fracPy.operators.operators import aspw, scaledASP
from fracPy.monitors.Monitor import Monitor
from fracPy.utils.visualisation import hsvplot


class BaseReconstructor(object):
    """
    Common properties that are common for all reconstruction engines are defined here.

    Unless you are testing the code, there's hardly any need to create this object. For your own implementation,
    inherit from this object

    """
    def __init__(self, optimizable: Optimizable, experimentalData:ExperimentalData, monitor:Monitor):
        # These statements don't copy any data, they just keep a reference to the object
        self.optimizable = optimizable
        self.experimentalData = experimentalData
        self.monitor = monitor
        self.monitor.optimizable = optimizable

        # datalogger
        self.logger = logging.getLogger('BaseReconstructor')

        # Default settings
        # settings that involve how things are computed
        self.fftshiftSwitch = False
        self.fftshiftFlag = False
        self.FourierMaskSwitch = False
        self.CPSCswitch = False
        self.fontSize = 17
        self.intensityConstraint = 'standard'  # standard or sigmoid
        self.propagator = 'Fraunhofer'  # 'Fresnel' 'ASP'


        ## Specific reconstruction settings that are the same for all engines
        # This only makes sense on a GPU, not there yet
        self.saveMemory = False
        self.probeUpdateStart = 1
        self.objectUpdateStart = 1
        self.positionOrder = 'random'  # 'random' or 'sequential'

        ## Swtiches used in applyConstraints method:
        self.orthogonalizationSwitch = True
        self.orthogonalizationFrequency = 10  # probe orthogonalization frequency
        # object regularization
        self.objectSmoothenessSwitch = False
        self.objectSmoothenessWidth = 2  # # pixels over which object is assumed fairly smooth
        self.objectSmoothnessAleph = 1e-2  # relaxation constant that determines strength of regularization
        self.absObjectSwitch = False  # force the object to be abs-only
        self.absObjectBeta = 1e-2  # relaxation parameter for abs-only constraint
        self.objectContrastSwitch = False  # pushes object to zero outside ROI
        # probe regularization
        self.probeSmoothenessSwitch = False # enforce probe smootheness
        self.probeSmoothnessAleph = 5e-2  # relaxation parameter for probe smootheness
        self.probeSmoothenessWidth = 3  # loose object support diameter
        self.absorbingProbeBoundary = False  # controls if probe has period boundary conditions (zero)
        self.absorbingProbeBoundaryAleph = 5e-2
        self.probePowerCorrectionSwitch = True  # probe normalization to measured PSD
        self.modulusEnforcedProbeSwitch = False  # enforce empty beam
        self.comStabilizationSwitch = False  # center of mass stabilization for probe
        # other
        self.couplingSwitch = False  # couple adjacent wavelengths
        self.couplingAleph = 50e-2  # couple adjacent wavelengths (relaxation parameter)
        self.binaryWFSSwitch = False  # enforce WFS to be positive
        self.binaryWFSAleph =  10e-2  # relaxation parameter for binary constraint
        self.backgroundModeSwitch = False  # background estimate
        self.comStabilizationSwitch = True # center of mass stabilization for probe
        self.PSDestimationSwitch = False
        self.objectContrastSwitch = False # pushes object to zero outside ROI

    def _initializeParams(self):
        """
        Initialize everything that depends on user changeable attributes.
        :return:
        """
        # check miscellaneous quantities specific for certain engines
        self._checkMISC()
        self._checkFFT()
        self._initializeQuadraticPhase()
        self._initializeProbeEnergy()
        self._initializeProbePower()
        self._probeWindow()
        self._initializeErrors()
        self._setObjectProbeROI()
        self._showInitialGuesses()



    def _initializeErrors(self):
        """
        initialize all kinds of errors:
        detectorError is a matrix calculated at each iteration (numFrames,Nd,Nd);
        errorAtPos sums over detectorError at each iteration, (numFrames,1);
        optimizable.error sums over errorAtPos, one number at each iteration;
        """
        # initialize detector error matrices
        if self.saveMemory:
            self.detectorError = 0
        else:
            self.detectorError = np.zeros((self.experimentalData.numFrames,
                                           self.experimentalData.Nd, self.experimentalData.Nd))
        # initialize energy at each scan position
        if not hasattr(self, 'errorAtPos'):
            self.errorAtPos = np.zeros((self.experimentalData.numFrames, 1), dtype=np.float32)
        # initialize final error
        if not hasattr(self.optimizable, 'error'):
            self.optimizable.error = []

        # todo what is ptychogramDownsampled

    def _initializeProbeEnergy(self):
        if len(self.experimentalData.ptychogram) != 0:
            self.energyAtPos = np.sum(abs(self.experimentalData.ptychogram), (-1, -2))
        else:
            self.energyAtPos = np.sum(abs(self.experimentalData.ptychogramDownsampled), (-1, -2))

    def _initializeProbePower(self):
        # probe power correction
        if len(self.experimentalData.ptychogram) != 0:
            self.probePowerCorrection = np.sqrt(np.max(np.sum(self.experimentalData.ptychogram, (-1, -2))))
        else:
            self.probePowerCorrection = np.sqrt(
                np.max(np.sum(self.experimentalData.ptychogramDownsampled, (-1, -2))))
        if self.probePowerCorrectionSwitch:
            self.optimizable.probe = self.optimizable.probe/np.sqrt(
                np.sum(self.optimizable.probe*self.optimizable.probe.conj()))*self.probePowerCorrection

    def _probeWindow(self):
        # absorbing probe boundary: filter probe with super-gaussian window function
        if not self.saveMemory or self.absorbingProbeBoundary:
            self.probeWindow = np.exp(-((self.experimentalData.Xp**2+self.experimentalData.Yp**2)/
                                        (2*(3/4*self.experimentalData.Np*self.experimentalData.dxp/2.355)**2))**10)

    def _setObjectProbeROI(self):
        """
        Set object/probe ROI for monitoring
        """
        if not hasattr(self.optimizable, 'objectROI'):
            rx,ry = ((np.max(self.experimentalData.positions, axis=0)-np.min(self.experimentalData.positions, axis=0)\
                    +self.experimentalData.Np)/self.monitor.objectPlotZoom).astype(int)
            xc,yc = ((np.max(self.experimentalData.positions, axis=0)+np.min(self.experimentalData.positions, axis=0)\
                    +self.experimentalData.Np)/2).astype(int)

            self.optimizable.objectROI = [slice(max(0, yc-ry//2),
                                                min(self.experimentalData.No, yc + ry//2)),
                                          slice(max(0, xc - rx // 2),
                                                min(self.experimentalData.No, xc + rx//2))]

        if not hasattr(self.optimizable, 'probeROI'):
            r = np.int(self.experimentalData.entrancePupilDiameter/self.experimentalData.dxp/self.monitor.probePlotZoom)
            self.optimizable.probeROI = [slice(max(0, self.experimentalData.Np//2-r),
                                               min(self.experimentalData.Np, self.experimentalData.Np//2+r)),
                                         slice(max(0, self.experimentalData.Np // 2 - r),
                                               min(self.experimentalData.Np, self.experimentalData.Np//2+r))]

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
        if self.propagator == 'Fresnel':
            self.optimizable.quadraticPhase = np.exp(1.j*np.pi/(self.experimentalData.wavelength*self.experimentalData.zo)
                                                     *(self.experimentalData.Xp**2+self.experimentalData.Yp**2))
        elif self.propagator == 'ASP':
            if self.fftshiftSwitch:
                raise ValueError('ASP propagator works only with fftshiftSwitch = False!')
            if self.optimizable.nlambda>1:
                raise ValueError('For multi-wavelength, polychromeASP needs to be used instead of ASP')

            dummy = np.ones((1, self.optimizable.nosm, self.optimizable.npsm,
                                      1, self.experimentalData.Np, self.experimentalData.Np), dtype='complex64')
            self.optimizable.transferFunction = np.array(
                [[[[aspw(dummy[nlambda, nosm, npsm, nslice, :, :],
                        self.experimentalData.zo, self.experimentalData.wavelength,
                        self.experimentalData.Lp)[1]
                    for nslice in range(1)]
                   for npsm in range(self.optimizable.npsm)]
                  for nosm in range(self.optimizable.nosm)]
                 for nlambda in range(self.optimizable.nlambda)])

        elif self.propagator == 'polychromeASP':
            dummy = np.ones((self.optimizable.nlambda, self.optimizable.nosm, self.optimizable.npsm,
                                      1, self.experimentalData.Np, self.experimentalData.Np), dtype='complex64')
            self.optimizable.transferFunction = np.array(
                [[[[aspw(dummy[nlambda, nosm, npsm, nslice, :, :],
                         self.experimentalData.zo, self.experimentalData.spectralDensity[nlambda],
                         self.experimentalData.Lp)[1]
                    for nslice in range(1)]
                   for npsm in range(self.optimizable.npsm)]
                  for nosm in range(self.optimizable.nosm)]
                 for nlambda in range(self.optimizable.nlambda)])
            if self.fftshiftSwitch:
                raise ValueError('ASP propagator works only with fftshiftSwitch = False!')

        elif self.propagator =='scaledASP':
            if self.fftshiftSwitch:
                raise ValueError('scaledASP propagator works only with fftshiftSwitch = False!')
            if self.optimizable.nlambda > 1:
                raise ValueError('For multi-wavelength, scaledPolychromeASP needs to be used instead of scaledASP')
            dummy = np.ones((1, self.optimizable.nosm, self.optimizable.npsm,
                                      1, self.experimentalData.Np, self.experimentalData.Np), dtype='complex64')
            self.optimizable.Q1 = np.ones_like(dummy)
            self.optimizable.Q2 = np.ones_like(dummy)
            for nosm in range(self.optimizable.nosm):
                for npsm in range(self.optimizable.npsm):
                    _, self.optimizable.Q1[0,nosm,npsm,0,...], self.optimizable.Q2[0,nosm,npsm,0,...] = scaledASP(
                        dummy[0, nosm, npsm, 0, :, :], self.experimentalData.zo, self.experimentalData.wavelength,
                        self.experimentalData.dxo, self.experimentalData.dxd)

        # todo check if Q1 Q2 are bandlimited

        elif self.propagator == 'scaledPolychromeASP':
            if self.fftshiftSwitch:
                raise ValueError('scaledPolychromeASP propagator works only with fftshiftSwitch = False!')
            dummy = np.ones((self.optimizable.nlambda, self.optimizable.nosm, self.optimizable.npsm,
                                          1, self.experimentalData.Np, self.experimentalData.Np), dtype='complex64')
            self.optimizable.Q1 = np.ones_like(dummy)
            self.optimizable.Q2 = np.ones_like(dummy)
            for nlmabda in range(self.optimizable.nlambda):
                for nosm in range(self.optimizable.nosm):
                    for npsm in range(self.optimizable.npsm):
                        _, self.optimizable.Q1[nlmabda, nosm, npsm, 0, ...], self.optimizable.Q2[
                            nlmabda, nosm, npsm, 0, ...] = scaledASP(
                            dummy[nlmabda, nosm, npsm, 0, :, :], self.experimentalData.zo,
                            self.experimentalData.spectralDensity[nlmabda], self.experimentalData.dxo,
                            self.experimentalData.dxd)

    def _checkMISC(self):
        """
        checks miscellaneous quantities specific certain engines
        """
        # todo check what does rgn('shuffle') do in matlab
        if self.backgroundModeSwitch:
            self.background = 1e-1*np.ones((self.experimentalData.Np, self.experimentalData.Np))

        # preallocate intensity scaling vector
        if self.intensityConstraint == 'fluctuation':
            self.intensityScaling = np.ones(self.experimentalData.numFrames)

        # todo check if there is data on gpu that shouldnt be there

        # check if both probePoprobePowerCorrectionSwitch and modulusEnforcedProbeSwitch are on.
        # Since this can cause a contradiction, it raises an error
        if self.probePowerCorrectionSwitch and self.modulusEnforcedProbeSwitch:
            raise ValueError('probePowerCorrectionSwitch and modulusEnforcedProbeSwitch '
                             'can not simultaneously be switched on!')

        if not self.fftshiftSwitch:
            warnings.warn('fftshiftSwitch set to false, this may lead to reduced performance')

        if self.propagator == 'ASP' and self.fftshiftSwitch:
                raise ValueError('ASP propagator works only with fftshiftSwitch = False')
        if self.propagator == 'scaledASP' and self.fftshiftSwitch:
                raise ValueError('scaledASP propagator works only with fftshiftSwitch = False')

    def _checkFFT(self):
        """
        shift arrays to accelerate fft
        """
        if self.fftshiftSwitch:
            if self.fftshiftFlag == 0:
                print('check fftshift...')
                print('fftshift data for fast far-field update')
                # shift detector quantities
                self.experimentalData.ptychogram = np.fft.ifftshift(self.experimentalData.ptychogram, axes=(-1, -2))
                if hasattr(self.experimentalData, 'ptychogramDownsampled'):
                    self.experimentalData.ptychogramDownsampled = np.fft.ifftshift(
                        self.experimentalData.ptychogramDownsampled, axes=(-1, -2))
                if hasattr(self, 'W'):
                    self.W = np.fft.ifftshift(self.W, axes=(-1, -2))
                if hasattr(self.experimentalData, 'empyBeam'):
                    self.experimentalData.empyBeam = np.fft.ifftshift(
                        self.experimentalData.empyBeam, axes=(-1, -2))
                if hasattr(self.experimentalData, 'PSD'):
                    self.experimentalData.PSD = np.fft.ifftshift(
                        self.experimentalData.PSD, axes=(-1, -2))
                self.fftshiftFlag = 1
        else:
            if self.fftshiftFlag == 1:
                print('check fftshift...')
                print('ifftshift data')
                self.experimentalData.ptychogram = np.fft.fftshift(self.experimentalData.ptychogram, axes=(-1, -2))
                if hasattr(self.experimentalData, 'ptychogramDownsampled'):
                    self.experimentalData.ptychogramDownsampled = np.fft.fftshift(
                        self.experimentalData.ptychogramDownsampled, axes=(-1, -2))
                if hasattr(self, 'W'):
                    self.W = np.fft.fftshift(self.W, axes=(-1, -2))
                if hasattr(self.experimentalData, 'empyBeam'):
                    self.experimentalData.empyBeam = np.fft.fftshift(
                        self.experimentalData.empyBeam, axes=(-1, -2))
                self.fftshiftFlag = 0

    def setPositionOrder(self):
        if self.positionOrder == 'sequential':
            self.positionIndices = np.arange(self.experimentalData.numFrames)

        elif self.positionOrder == 'random':
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
        elif self.positionOrder == 'NA':
            rows = self.experimentalData.positions[:, 0] - np.mean(self.experimentalData.positions[:, 0])
            cols = self.experimentalData.positions[:, 1] - np.mean(self.experimentalData.positions[:, 1])
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
        if self.propagator == 'Fraunhofer':
            self.fft2s()
        elif self.propagator == 'Fresnel':
            self.optimizable.esw = self.optimizable.esw * self.optimizable.quadraticPhase
            self.fft2s()
        elif self.propagator == 'ASP' or self.propagator == 'polychromeASP':
            self.optimizable.ESW = ifft2c(fft2c(self.optimizable.esw) * self.optimizable.transferFunction)
        elif self.propagator == 'scaledASP' or self.propagator == 'scaledPolychromeASP':
            self.optimizable.ESW = ifft2c(fft2c(self.optimizable.esw * self.optimizable.Q1) * self.optimizable.Q2)
        else:
            raise Exception('Propagator is not properly set, choose from Fraunhofer, Fresnel, ASP and scaledASP')


    def detector2object(self):
        """
        Propagate the ESW to the object plane (in-place).

        Matches: detector2object.m
        :return:
        """
        if self.propagator == 'Fraunhofer':
            self.ifft2s()
        elif self.propagator == 'Fresnel':
            self.ifft2s()
            self.optimizable.esw = self.optimizable.esw * self.optimizable.quadraticPhase.conj()
            self.optimizable.eswUpdate = self.optimizable.eswUpdate * self.optimizable.quadraticPhase.conj()
        elif self.propagator == 'ASP' or self.propagator == 'polychromeASP':
            self.optimizable.eswUpdate = ifft2c(fft2c(self.optimizable.ESW) * self.optimizable.transferFunction.conj())
        elif self.propagator == 'scaledASP' or self.propagator == 'scaledPolychromeASP':
            self.optimizable.eswUpdate = ifft2c(fft2c(self.optimizable.ESW) * self.optimizable.Q2.conj()) \
                                         * self.optimizable.Q1.conj()
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

        if self.fftshiftSwitch:
            self.optimizable.ESW = xp.fft.fft2(self.optimizable.esw, norm='ortho')
        else:
            self.optimizable.ESW = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(self.optimizable.esw),norm='ortho'))

    def getBeamWidth(self):
        """
        Matches getBeamWith.m
        :return:
        """
        raise NotImplementedError()

    def getErrorMetrics(self):
        """
        matches getErrorMetrics.m
        :return:
        """
        if not self.saveMemory:
            # Calculate mean error for all positions (make separate function for all of that)
            if self.FourierMaskSwitch:
                self.errorAtPos = np.sum(np.abs(self.detectorError) * self.W, axis=(-1, -2))
            else:
                self.errorAtPos = np.sum(np.abs(self.detectorError), axis=(-1, -2))
        self.errorAtPos = asNumpyArray(self.errorAtPos)/asNumpyArray(self.energyAtPos + 1)
        eAverage = np.sum(self.errorAtPos)

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
        if self.saveMemory:
            if self.FourierMaskSwitch and not self.CPSCswitch:
                self.errorAtPos[positionIndex] = xp.sum(self.currentDetectorError*self.W)
            elif self.FourierMaskSwitch and self.CPSCswitch:
                raise NotImplementedError
            else:
                self.errorAtPos[positionIndex] = xp.sum(self.currentDetectorError)
        else:
            self.detectorError[positionIndex] = self.currentDetectorError

    def ifft2s(self):
        """ Inverse FFT"""
        # find out if this should be performed on the GPU
        xp = getArrayModule(self.optimizable.esw)

        if self.fftshiftSwitch:
            self.optimizable.eswUpdate = xp.fft.ifft2(self.optimizable.ESW, norm='ortho')
        else:
            self.optimizable.eswUpdate = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(self.optimizable.ESW),norm='ortho'))

    def intensityProjection(self, positionIndex):
        """ Compute the projected intensity.
            Barebones, need to implement other methods
        """
        # figure out wether or not to use the GPU
        xp = getArrayModule(self.optimizable.esw)

        # zero division mitigator
        gimmel = 1e-10

        # propafate to detector
        self.object2detector()

        # get estimated intensity (2D array, in the case of multislice, only take the last slice)
        if self.backgroundModeSwitch:
            self.optimizable.Iestimated = xp.sum(xp.abs(self.optimizable.ESW) ** 2, axis=(0, 1, 2))[-1]+self.background
        else:
            self.optimizable.Iestimated = xp.sum(xp.abs(self.optimizable.ESW) ** 2, axis=(0, 1, 2))[-1]
        # get measured intensity todo implement CPSC, kPIE
        if self.CPSCswitch:
            raise NotImplementedError
        else:
            self.optimizable.Imeasured = xp.array(self.experimentalData.ptychogram[positionIndex])

        self.getRMSD(positionIndex)

        # intensity projection constraints
        if self.intensityConstraint == 'fluctuation':
            # scaling
            if self.FourierMaskSwitch:
                aleph = xp.sum(self.optimizable.Imeasured*self.optimizable.Iestimated*self.W) / \
                        xp.sum(self.optimizable.Imeasured*self.optimizable.Imeasured*self.W)
            else:
                aleph = xp.sum(self.optimizable.Imeasured * self.optimizable.Iestimated) / \
                        xp.sum(self.optimizable.Imeasured * self.optimizable.Imeasured)
            self.intensityScaling[positionIndex] = aleph
            # scaled projection
            frac = (1+aleph)/2*self.optimizable.Imeasured/(self.optimizable.Iestimated+gimmel)

        elif self.intensityConstraint == 'exponential':
            x = self.currentDetectorError/(self.optimizable.Iestimated+gimmel)
            W = xp.exp(-0.05 * x)
            frac = xp.sqrt( self.optimizable.Imeasured / (self.optimizable.Iestimated + gimmel) )
            frac = W * frac + (1-W)

        elif self.intensityConstraint == 'poission':
            frac = self.optimizable.Imeasured / (self.optimizable.Iestimated + gimmel)

        elif self.intensityConstraint == 'standard':
            frac = xp.sqrt(self.optimizable.Imeasured / (self.optimizable.Iestimated + gimmel))
        else:
            raise ValueError('intensity constraint not properly specified!')

        # apply mask
        if self.FourierMaskSwitch and self.CPSCswitch and len(self.optimizable.error) > 5:
            frac = self.W * frac + (1-self.W)

        # update ESW
        self.optimizable.ESW = self.optimizable.ESW * frac

        # update background (see PhD thsis by Peng Li)
        if self.backgroundModeSwitch:
            if self.FourierMaskSwitch:
                self.background = self.background*(1+1/self.experimentalData.numFrames*(xp.sqrt(frac)-1))**2*self.W
            else:
                self.background = self.background*(1+1/self.experimentalData.numFrames*(xp.sqrt(frac)-1))**2

        # back propagate to object plane
        self.detector2object()

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
                if self.fftshiftSwitch:
                    Iestimated = np.fft.fftshift(asNumpyArray(self.optimizable.Iestimated))
                    Imeasured = np.fft.fftshift(asNumpyArray(self.optimizable.Imeasured))
                else:
                    Iestimated = asNumpyArray(self.optimizable.Iestimated)
                    Imeasured = asNumpyArray(self.optimizable.Imeasured)

                self.monitor.updateDiffractionDataMonitor(Iestimated=Iestimated, Imeasured=Imeasured)

            # print('iteration:%i' %len(self.optimizable.error))
            # print('runtime:')
            # print('error:')
        # TODO: print info

    def applyConstraints(self, loop):
        """
        Apply constraints.
        :param loop: loop number
        :return:
        """
        # enforce empty beam constraint
        if self.modulusEnforcedProbeSwitch:
            self.modulusEnforcedProbe()

        if self.orthogonalizationSwitch:
            if np.mod(loop, self.orthogonalizationFrequency) == 0:
                self.orthogonalization()

        # probe normalization to measured PSD todo: check for multiwave and multi object states
        if self.probePowerCorrectionSwitch:
            self.optimizable.probe = self.optimizable.probe / np.sqrt(
                np.sum(self.optimizable.probe * self.optimizable.probe.conj())) * self.probePowerCorrection

        if self.comStabilizationSwitch:
            self.comStabilization()

        if self.PSDestimationSwitch:
            raise NotImplementedError()

        if self.absorbingProbeBoundary:
            if self.experimentalData.operationMode =='FPM':
                self.absorbingProbeBoundaryAleph = 100e-2

            self.optimizable.probe = (1 - self.absorbingProbeBoundaryAleph)*self.optimizable.probe+\
                                     self.absorbingProbeBoundaryAleph*self.optimizable.probe*self.probeWindow

        # Todo: objectSmoothenessSwitch,probeSmoothenessSwitch,
        if self.probeSmoothenessSwitch:
            raise NotImplementedError()

        if self.objectSmoothenessSwitch:
            raise NotImplementedError()


        if self.absObjectSwitch:
            self.optimizable.object = (1-self.absObjectBeta)*self.optimizable.object+\
                                      self.absObjectBeta*abs(self.optimizable.object)

        # this is intended to slowly push non-measured object region to abs value lower than
        # the max abs inside object ROI allowing for good contrast when monitoring object
        if self.objectContrastSwitch:
            self.optimizable.object = 0.995*self.optimizable.object+0.005*\
                                      np.mean(abs(self.optimizable.object[..., self.optimizable.objectROI[0],
                                                                          self.optimizable.objectROI[1]]))
        if self.couplingSwitch and self.optimizable.nlambda > 1:
            self.optimizable.probe[0] = (1 - self.couplingAleph) * self.optimizable.probe[0] + \
                                        self.couplingAleph * self.optimizable.probe[1]
            for lambdaLoop in np.arange(1, self.optimizable.nlambda - 1):
                self.optimizable.probe[lambdaLoop] = (1 - self.couplingAleph) * self.optimizable.probe[lambdaLoop] + \
                                                     self.couplingAleph * (self.optimizable.probe[lambdaLoop + 1] +
                                                                           self.optimizable.probe[
                                                                               lambdaLoop - 1]) / 2

            self.optimizable.probe[-1] = (1 - self.couplingAleph) * self.optimizable.probe[-1] + \
                                         self.couplingAleph * self.optimizable.probe[-2]
        if self.binaryWFSSwitch:
            self.optimizable.probe = (1 - self.binaryWFSAleph) * self.optimizable.probe + \
                                     self.binaryWFSAleph * abs(self.optimizable.probe)



    def orthogonalization(self):
        """
        Perform orthogonalization
        :return:
        """
        if self.optimizable.npsm > 1:
            # orthogonalize the probe
            for id_l in range(self.optimizable.nlambda):
                for id_s in range(self.optimizable.nslice):
                    self.optimizable.probe[id_l, 0, :, id_s, :, :], self.normalizedEigenvaluesProbe, self.MSPVprobe = \
                        orthogonalizeModes(self.optimizable.probe[id_l, 0, :, id_s, :, :])
                    self.optimizable.purity = np.sqrt(np.sum(self.normalizedEigenvaluesProbe ** 2))

            # orthogonolize momentum operator
            try:
                self.optimizable.probeMomentum[id_l, 0, :, id_s, :, :], none, none = \
                    orthogonalizeModes(self.optimizable.probeMomentum[id_l, 0, :, id_s, :, :])
                # replace the orthogonolized buffer
                self.optimizable.probeBuffer = self.optimizable.probe.copy()
            except:
                pass

        elif self.optimizable.nosm > 1:
            # orthogonalize the object
            for id_l in range(self.optimizable.nlambda):
                for id_s in range(self.optimizable.nslice):
                    self.optimizable.object[id_l, :, 0, id_s, :, :], self.normalizedEigenvaluesObject, self.MSPVobject = \
                        orthogonalizeModes(self.optimizable.object[id_l, :, 0, id_s, :, :])

            # orthogonolize momentum operator
            try:
                self.optimizable.objectMomentum[id_l, :, 0, id_s, :, :], none, none = \
                    orthogonalizeModes(self.optimizable.objectMomentum[id_l, :, 0, id_s, :, :])
                # replace the orthogonolized buffer as well
                self.optimizable.objectBuffer = self.optimizable.object.copy()
            except:
                pass
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
        demon = xp.sum(P2)*self.experimentalData.dxp
        xc = xp.int(xp.around(xp.sum(xp.array(self.experimentalData.Xp, xp.float32) * P2) / demon))
        yc = xp.int(xp.around(xp.sum(xp.array(self.experimentalData.Yp, xp.float32) * P2) / demon))
        # shift only if necessary
        if xc**2+yc**2>1:
            # shift probe
            for k in xp.arange(self.optimizable.npsm): # todo check for multislice
                self.optimizable.probe[:,:,k,-1,...] = \
                    xp.roll(self.optimizable.probe[:,:,k,-1,...], (-yc, -xc), axis=(-2, -1))

                #todo implement for mPIE

            # shift object
            for k in xp.arange(self.optimizable.nosm): # todo check for multislice
                self.optimizable.object[:,:,k,-1,...] = \
                    xp.roll(self.optimizable.object[:,:,k,-1,...], (-yc, -xc), axis=(-2, -1))
                # todo implement for mPIE


    def modulusEnforcedProbe(self):
        # propagate probe to detector
        self.optimizable.esw = self.optimizable.probe
        self.object2detector()

        if self.FourierMaskSwitch:
            self.optimizable.ESW = self.optimizable.ESW*xp.sqrt(
                self.emptyBeam/1e-10+xp.sum(xp.abs(self.optimizable.ESW)**2, axis=(0, 1, 2, 3)))*self.W\
                                   +self.optimizable.ESW*(1-self.W)
        else:
            self.optimizable.ESW = self.optimizable.ESW * np.sqrt(
                self.emptyBeam / (1e-10 + xp.sum(abs(self.optimizable.ESW) ** 2, axis=(0, 1, 2, 3))))

        self.detector2object()
        self.optimizable.probe = self.optimizable.esw