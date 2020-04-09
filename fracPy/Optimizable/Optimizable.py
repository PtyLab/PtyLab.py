import numpy as np
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from copy import copy
import logging

class Optimizable(object):
    """
    This object will contain all the things that can be modified by a reconstruction engine.

    In itself, it's little more than a data holder. It is initialized with an ExperimentalData object.

    From this object, it will copy anything in the
    list_of_optimizable_properties to itself so the parameters can be changed.
    """

    list_of_optimizable_properties = [
        'wavelength',
        'positions'
    ]
    def __init__(self, data:ExperimentalData):
        self.logger = logging.getLogger('Optimizable')
        self.copyAttributesFromExperiment(data)

    def copyAttributesFromExperiment(self, data:ExperimentalData):
        """
        Copy all the attributes from the experiment that are in list_of_optimizable_properties
        :param data:
                Experimental data to copy from
        :return:
        """
        self.logger.debug('Copying attributes from Experimental Data')
        for key in self.list_of_optimizable_properties:
            self.logger.debug('Copying attribute %s', key)
            setattr(self, key, copy(getattr(data, key)))
