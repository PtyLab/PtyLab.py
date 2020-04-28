from fracPy.monitors.default_visualisation import DefaultMonitor
import numpy as np
import logging

# fracPy imports
from fracPy.utils.initializationFunctions import initialProbeOrObject
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.utils.utils import ifft2c, fft2c

class BaseReconstructor(object):
    """
    Common properties for any reconstruction ePIE_engine are defined here.

    Unless you are testing the code, there's hardly any need to create this object. For your own implementation,
    inherit from this object

    """
    def __init__(self, optimizable: Optimizable, experimentalData:ExperimentalData):
        # These statements don't copy any data, they just keep a reference to the object
        self.optimizable = optimizable
        self.experimentalData = experimentalData

        # datalogger
        self.logger = logging.getLogger('BaseReconstructor')

        # Default settings
        # settings that involve how things are computed
        self.objectPlot = 'complex'
        self.fftshiftSwitch = False
        self.figureUpdateFrequency = 1
        self.FourierMaskSwitch = False
        self.fontSize = 17
        self.intensityConstraint = 'standard'  # standard or sigmoid
        self.propagator = 'fraunhofer'
        
        # Settings involving the intitial estimates
        # self.initialObject = 'ones'
        # self.initialProbe = 'circ'

        # Specific reconstruction settings that are the same for all engines
        self.absorbingProbeBoundary = False
        self.npsm = 1  # number of probe state mixtures
        self.nosm = 1  # number of object state mixtures

        # Things that should be overridden in every reconstructor
        self.numIterations = 1  # number of iterations

        self.objectUpdateStart = 1
        self.positionOrder = 'random'  # 'random' or 'sequential'

        self.probeSmoothnessSwitch = False
        self.absObjectSwitch = False
        self.comStabilizationSwitch = False
        self.objectContrastSwitch = False

    def setPositionOrder(self):
        # Tomas: simple implemenattion to get the basic reconstruction going
        positionNumber = self.experimentalData.positions.shape[0]
        if self.positionOrder == 'random':
            indices = np.arange(positionNumber)
            np.random.shuffle(indices)
        if self.positionOrder == 'sequential':
            indices = np.arange(positionNumber)
        return indices


    def changeExperimentalData(self, experimentalData:ExperimentalData):
        self.experimentalData = experimentalData

    def changeOptimizable(self, optimizable: Optimizable):

        self.optimizable = optimizable


    def startReconstruction(self):
        raise NotImplementedError()


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

    def detector2object(self):
        """
        Propagate the ESW to the object plane (in-place).

        Matches: detector2object.m
        :return:
        """
        if self.propagator == 'fraunhofer':
            self.ifft2s()
        else:
            raise NotImplementedError()

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
        fft2s.m
        :return:
        """
        if self.fftshiftSwitch:
            self.optimizable.ESW = np.fft.fft2(self.optimizable.esw, norm='ortho') #/ self.experimentalData.Np
        else:
            self.optimizable.ESW = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.optimizable.esw),norm='ortho')) #/ self.experimentalData.Np

    def getBeamWidth(self):
        """
        Matches getBeamWith.m
        :return:
        """
        raise NotImplementedError()


    def getErrorMetrics(self, testing_mode=False):
        """
        matches getErrorMetrics.m
        :return:
        """
        if not testing_mode:
            raise NotImplementedError()
        else:
            return np.random.rand(100)

    def getRMSD(self, positionIndex):
        """
        matches getRMSD.m
        :param positionIndex:
        :return:
        """
        raise NotImplementedError()

    def ifft2s(self):
        """ Inverse FFT"""
        if self.fftshiftSwitch:
            self.optimizable.eswUpdate = np.fft.ifft2(self.optimizable.ESW, norm='ortho')# * self.experimentalData.Np
        else:
            self.optimizable.eswUpdate = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.optimizable.ESW),norm='ortho')) #* self.experimentalData.Np


    def intensityProjection(self, positionIndex):
        """ Compute the projected intensity.
            Barebones, need to implement other methods
        """        
        self.object2detector()

        gimmel = 1e-10
        # these are amplitudes rather than intensities
        Iestimated = np.abs(self.optimizable.ESW)**2
        Imeasured = self.experimentalData.ptychogram[positionIndex,:,:]
        
        # TOOD: implement other update methods
        frac = np.sqrt(Imeasured / (Iestimated + gimmel))
        
        self.optimizable.ESW = self.optimizable.ESW * frac
        self.detector2object()
        # raise NotImplementedError()

    def object2detector(self):
        """
        Implements object2detector.m
        :return:
        """
        if self.propagator == 'fraunhofer':
            self.fft2s()
        else:
            raise NotImplementedError()

    def orthogonalize(self):
        """
        Implement orhtogonalize.m
        :return:
        """
        raise NotImplementedError()

    def prepare_reconstruction(self):
        pass

    def doReconstruction(self):
        """
        Reconstruct the object based on all the parameters that have been set beforehand.

        This method is overridden in every reconstruction ePIE_engine, therefore it is already finished.
        :return:
        """
        self.prepare_reconstruction()
        raise NotImplementedError()


    def initializeObject(self):
        """
        Initialize the object.
        :return:
        """
        self.optimizable.initialize_object()


    def showReconstruction(self, loop):
        """
        Show the reconstruction process.
        :param loop: the iteration number
        :return:
        """
        object_estimate = abs(ifft2c(self.optimizable.object))


        if loop == 0:
            self.initializeVisualisation()
        elif np.mod(loop, self.figureUpdateFrequency) == 0:
            # fpm mode visualization
            errorMetric = self.getErrorMetrics(testing_mode=True)
            self.monitor.updateError(errorMetric)
            self.monitor.updateObject(object_estimate)
            self.monitor.drawNow()

    def initializeVisualisation(self):
        """
        Create the figure and axes etc.
        :return:
        """

        self.monitor = DefaultMonitor()

    def applyConstraints(self, loop):
        """
        Apply constraints.
        :param loop: loop number
        :return:
        """

        # modulus enforced probe
        if self.modulusEnforcesProbeSwitch:
            raise NotImplementedError()
            # # propagate probe to detector
            # self.params.esw = self.probe
            # self.object2detector()
            #
        if np.mod(loop, self.orthogonalizationFrequency) == 0:
            self.orthogonalize()


        if self.objectSmoothenessSwitch:
            raise NotImplementedError()

        # not specific to ePIE, -> baseReconstructor
        if self.probeSmoothenessSwitch:
            raise NotImplementedError()

        # not specific to ePIE
        if self.absObjectSwitch:
            raise NotImplementedError()


        if self.comStabilizationSwitch:
            raise NotImplementedError()


        if self.objectContrastSwitch():
            raise NotImplementedError()



