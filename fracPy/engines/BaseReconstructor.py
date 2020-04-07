from fracPy import ptyLab
from fracPy.utils.initialization_functions import initial_probe_or_object

class Params(object):
    """ Contains settings that are in initialParams.

    Will be loaded into baseReconstruction"""
    def __init__(self):

        self.deleteFrames = []
        self.absorbingProbeBoundary = False
        self.fftshiftSwitch = True
        self.figureUpdateFrequency = 1
        self.FourierMaskSwitch = False
        self.fontSize = 17
        self.initialObject = 'ones'
        self.initialProbe = 'circ'
        self.intensityConstraint = 'standard' # standard or sigmoid
        self.npsm = 1 # number of probe state mixtures
        self.nosm = 1 # number of object state mixtures
        self.numIterations = 1 # number of iterations
        self.objectPlot = 'complex'
        self.objectUpdateStart = 1
        self.positionOrder = 'random'


class BaseReconstructor(ptyLab.DataLoader):
    """
    Common properties for any reconstruction engine are defined here.

    Example properties:
        - propagation from one plane to another
        - converting data to single precision
        - general settings regarding reconstruction, like number of iterations

    """
    def __init__(self, datafolder=None):
        super().__init__(datafolder=datafolder)

        self.initialize_params()

    def initialize_params(self):
        """
        Initialize attributes that are the same for all reconstructions and that are in the
        params object.
        :return:
        """
        self.params = Params()
        # override stuff here

        self.initialize_object()
        self.initialize_probe()

    def start_reconstruction(self):
        raise NotImplementedError()

    def saveMemory(self):
        """
        Deletes fields that are not required. Python-implementation of checkMemory.m

        Should be implemented in the actual reconstruction script
        :return:
        """
        raise NotImplementedError()

    def convertToSingle(self):
        """
        Convert the datasets to single precision. Matches: convert2single.m
        :return:
        """
        raise NotImplementedError()

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

