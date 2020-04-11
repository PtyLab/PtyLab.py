import numpy as np
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from copy import copy
import logging

class Optimizable(object):
    """
    This object will contain all the things that can be modified by a reconstruction ePIE_engine.

    In itself, it's little more than a data holder. It is initialized with an ExperimentalData object.

    From this object, it will copy anything in the
    listOfOptimizableProperties to itself so the parameters can be changed.
    """

    listOfOptimizableProperties = [
        'wavelength',
        'positions'
    ]
    def __init__(self, data:ExperimentalData):
        self.logger = logging.getLogger('Optimizable')
        self.copyAttributesFromExperiment(data)

    def copyAttributesFromExperiment(self, data:ExperimentalData):
        """
        Copy all the attributes from the experiment that are in listOfOptimizableProperties
        :param data:
                Experimental data to copy from
        :return:
        """
        self.logger.debug('Copying attributes from Experimental Data')
        for key in self.listOfOptimizableProperties:
            self.logger.debug('Copying attribute %s', key)
            setattr(self, key, copy(getattr(data, key)))

    def saveResults(self):
        raise NotImplementedError
