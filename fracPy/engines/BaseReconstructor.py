from fracPy import ptyLab

class BaseReconstructor(ptyLab.Ptylab):
    """
    Common properties for a reconstruction engine can be defined here, for instance a loading and saving method.
    """
    def __init__(self):
        raise NotImplementedError()

    def start_reconstruction(self):
        raise NotImplementedError()

    def saveMemory(self):
        """
        Deletes fields that are not required. Python-implementation of checkMemory.m
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