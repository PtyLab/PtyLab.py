import numpy as np
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from copy import copy
import logging
# logging.basicConfig(level=logging.DEBUG)

from fracPy.utils.initializationFunctions import initialProbeOrObject


class Optimizable(object):
    """
    This object will contain all the things that can be modified by a reconstruction ePIE_engine.

    In itself, it's little more than a data holder. It is initialized with an ExperimentalData object.

    From this object, it will copy anything in the
    listOfOptimizableProperties to itself so the parameters can be changed.
    """

    listOfOptimizableProperties = [
        'wavelength',
        'positions',
    ]
    def __init__(self, data:ExperimentalData):
        self.logger = logging.getLogger('Optimizable')
        self.copyAttributesFromExperiment(data)
        self.initialize_other_settings()
        self.data = data

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
            setattr(self, key, copy(np.array(getattr(data, key))))
            # try:
            #     self.logger.debug('Set %s to %f', key, getattr(data,key))
            # except TypeError:
            #     pass

    def initialize_other_settings(self):
        """
        Initialize the attributes that have to do with a reconstruction but which are not given by data.

        This method just sets the settings. It sets the what kind of initial guess should be used for initialObject
        and initialProbe but it does not compute them yet. That will be done by calling prepare_reconstruction()

        :return:
        """
        self.npsm = 1
        self.nosm = 1
        self.initialObject = 'ones'
        self.initialProbe = 'ones'

    def prepare_reconstruction(self):
        self.object = np.zeros((self.npsm, self.data.No, self.data.No), np.complex64)
        self.initializeObject()
        self.initializeProbe()


    def saveResults(self):
        raise NotImplementedError

    def initializeObject(self):
        self.initialObject = initialProbeOrObject((self.nosm, self.data.No, self.data.No),
                                                     self.initialObject)

    def initializeProbe(self):
        self.initialProbe = initialProbeOrObject((self.npsm, self.data.Np, self.data.Np),
                                                        self.initialProbe)

