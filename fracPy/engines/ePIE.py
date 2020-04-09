import numpy as np

# fracPy imports
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines.BaseReconstructor import BaseReconstructor
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.utils.gpu_utils import get_array_module
import logging


class ePIE(BaseReconstructor):

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(optimizable, experimentalData)
        self.logger = logging.getLogger('ePIE')
        self.logger.info('Sucesfully created ePIE engine')

        self.logger.info('Wavelength attribute: %s', self.optimizable.wavelength)

        # self.initialize_reconstruction_params()

    def initialize_reconstruction_params(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.eswUpdate = self.optimizable.esw.copy()
        self.betaProbe = 0.25
        self.betaObject = 0.25

    def do_reconstruction(self):
        # actual reconstruction engine
        for loop in range(self.numIterations):
            self.positionIndices = self.setPositionOrder()
            for positionLoop, positionIndex in enumerate(self.experimentalData.numFrames):
                # get object patch
                row, col = self.optimizable.positions[positionIndex]
                sy = slice(row, col + self.experimentalData.Np)
                sx = slice(col, col + self.experimentalData.Np)
                objectPatch = self.optimizable.object[sy, sx]
                # note that object patch has size of probe array
                # make exit surface wave
                self.optimizable.esw = objectPatch * self.optimizable.probe

                # intensityProjection
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.eswUpdate - self.optimizable.esw

                # object update
                objectPatch = self.objectPatchUpdate(self, objectPatch, DELTA)

                # probe update
                self.optimizable.probe = self.probeUpdate(self, objectPatch, DELTA)

                # set updated object patch
                self.object[sy, sx] = objectPatch
            # get error metric
            self.getErrorMetrics()

            self.apply_constraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = get_array_module(objectPatch)
        frac = objectPatch.conj() / xp.max(xp.sum(abs(objectPatch) ** 2, 0))
        # this is two statements in matlab but it should only be one in python
        # TODO figure out unit tests and padding dimensions
        r = self.optimizable.probe + self.betaProbe * frac * DELTA
        if self.absorbingProbeBoundary:
            aleph = 1e-3
            r = (1 - aleph) * r + aleph * r * self.probeWindow
        return r
