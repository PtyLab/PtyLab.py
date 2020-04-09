from fracPy import ptyLab
import numpy as np
import logging

# fracPy imports
from fracPy.utils.initialization_functions import initial_probe_or_object
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable

class BaseReconstructor(object):
    """
    Common properties for any reconstruction engine are defined here.

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
        self.fftshiftSwitch = True
        self.figureUpdateFrequency = 1
        self.FourierMaskSwitch = False
        self.fontSize = 17
        self.intensityConstraint = 'standard'  # standard or sigmoid

        # Settings involving the intitial estimates

        self.initialObject = 'ones'
        self.initialProbe = 'circ'

        # Specific reconstruction settings that are the same for all engines
        self.absorbingProbeBoundary = False
        self.npsm = 1  # number of probe state mixtures
        self.nosm = 1  # number of object state mixtures

        # Things that should be overridden in every reconstructor
        self.numIterations = 1  # number of iterations

        self.objectUpdateStart = 1
        self.positionOrder = 'random'
        # TODO This list is not finished yet.


    def change_experimentalData(self, experimentalData:ExperimentalData):
        self.experimentalData = experimentalData

    def change_optimizable(self, optimizable: Optimizable):
        self.optimizable = optimizable


    def start_reconstruction(self):
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
        raise NotImplemented()

    def _match_dtypes_real(self):
        raise NotImplemented()

    def detector2object(self):
        """
        Propagate the ESW to the object plane (in-place).

        Matches: detector2object.m
        :return:
        """
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

    def ffts(self):
        """
        fft2s.m
        :return:
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def getRMSD(self, positionIndex):
        """
        matches getRMSD.m
        :param positionIndex:
        :return:
        """
        raise NotImplementedError()

    def ifft2s(self):
        """ Inverse FFT"""
        raise NotImplementedError()


    def intensityProjection(self):
        """ Compute the projected intensity.

        Should be implemented in the particular reconstruction object.

        """
        raise NotImplementedError()

    def object2detector(self):
        """
        Implements object2detector.m
        :return:
        """
        raise NotImplementedError()

    def orthogonalize(self):
        """
        Implement orhtogonalize.m
        :return:
        """
        raise NotImplementedError()

    def reconstruct(self):
        """
        Reconstruct the object based on all the parameters that have been set beforehand.

        This method is overridden in every reconstruction engine, therefore it is already finished.
        :return:
        """
        self.prepare_reconstruction()
        raise NotImplementedError()


    def initialize_object(self):
        """
        Initialize the object.
        :return:
        """
        self.params.initialObject = initial_probe_or_object((self.params.nosm, self.No, self.No),
                                                            self.params.initialObject)

    def initialize_probe(self):
        self.params.initialProbe = initial_probe_or_object((self.params.npsm, self.Np, self.Np))

