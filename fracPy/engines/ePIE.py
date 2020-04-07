from fracPy.engines.BaseReconstructor import BaseReconstructor
import numpy as np
from fracPy import reconstruction_object as ro
from fracPy.utils.gpu_utils import get_array_module
from fracPy.Params import Params
import logging

class ePIE(BaseReconstructor):
    def __init__(self, dataFolder=None):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(dataFolder)
        self.logger = logging.getLogger('ePIE')
        self.logger.debug('Hello from ePIE')
        self.logger.info('Wavelength attribute: %s', self.wavelength)
        self.initialize_reconstruction_params()

    def initialize_reconstruction_params(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        # these are always initiali

        # initialize things like esw and final object
        self.esw = self.initialize_object()
        self.esw = self.params.esw
        self.eswUpdate = self.obj.params.eswUpdate
        self.probe = self.params.probe



    def do_reconstruction(self):
        # actual reconstruction engine
        for loop in range(self.numIterations):
            self.params.positionIndices = self.setPositionOrder()
            for positionLoop, positionIndex in enumerate(self.obj.numFrames):
                # get object patch
                row, col = self.positions[positionIndex]
                sy = slice(row, col+self.obj.Np)
                sx = slice(col, col+self.obj.Np)
                objectPatch = self.object[sy, sx]
                # note that object patch has size of probe array
                # make exit surface wave
                self.params.esw = objectPatch * self.params.probe

                # intensityProjection
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.params.eswUpdate - self.params.esw

                # object update
                objectPatch = self.objectPatchUpdate(self, objectPatch, DELTA)

                # probe update
                self.probe = self.probeUpdate(self, objectPatch, DELTA)

                # set updated object patch
                self.object[sy, sx] = objectPatch
            # get error metric
            self.getErrorMetrics()

            self.apply_constraints(loop)

            # show reconstruction
            if np.mod(loop, self.params.figureUpdateFrequency) == 0:
                self.showReconstruction()


    def objectPatchUpdate(self, objectPatch, DELTA):
        # find out which array module to use, numpy or cupy (or other...)
        xp = get_array_module(objectPatch)

        frac = objectPatch.conj() / xp.max(xp.sum(abs(objectPatch)**2, 0))
        # this is two statements in matlab but it should only be one in python
        # TODO figure out unit tests and padding dimensions
        r = self.probe + self.params.betaProbe * frac* DELTA
        if self.params.absorbingProbeBoundary:
            aleph = 1e-3
            r = (1-aleph) * r + aleph * r * self.params.probeWindow
        return r

    def apply_constraints(self, loop):

        # modulus enforced probe
        if self.params.modulusEnforcesProbeSwitch:
            raise NotImplementedError()
            # # propagate probe to detector
            # self.params.esw = self.probe
            # self.object2detector()
            #
        if np.mod(loop, self.params.orthogonalizationFrequency) == 0:
            self.orthogonalize()

        # not specific to ePIE, should go to baseReconstructor
        if self.params.objectSmoothenessSwitch:
            raise NotImplementedError()

        # not specific to ePIE, -> baseReconstructor
        if self.params.probeSmoothenessSwitch:
            raise NotImplementedError()

        # not specific to ePIE
        if self.params.absObjectSwitch:
            raise NotImplementedError()

        # not specific to ePIE
        if self.params.comStabilizationSwitch:
            raise NotImplementedError()

        # not specific to ePIE
        if self.params.objectContrastSwitch():
            raise NotImplementedError()


if __name__ == '__main__':
    from fracPy import ptyLab
    obj = ePIE('.')