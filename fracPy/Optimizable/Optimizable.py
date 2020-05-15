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
        'probe'
    ]
    def __init__(self, data:ExperimentalData):
        self.logger = logging.getLogger('Optimizable')
        self.copyAttributesFromExperiment(data)
        self.data = data
        self.initialize_other_settings()
        # self.prepare_reconstruction()


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
        # create a 6D object where which allows to have:
        # 1. polychromatic = nlambda
        # 2. mixed state object - nosm
        # 3. mixed state probe - npsm
        # 4. multislice object (thick) - nslice
        self.npsm = 1
        self.nosm = 1
        self.nlambda = 1
        self.nslice = 1
        

        if self.data.operationMode == 'FPM':
            self.initialObject = 'upsampled'
            self.initialProbe = 'circ'
        elif self.data.operationMode == 'CPM':
            self.initialProbe = 'circ'
            self.initialObject = 'ones'
        else:
            self.initialProbe = 'circ'
            self.initialObject = 'ones'

    def prepare_reconstruction(self):
        
        # initialize object and probe
        self.initializeObject()
        self.initializeProbe()

        
        # set object and probe objects
        self.object = self.initialObject.copy()
        self.probe = self.initialProbe.copy()

        # initialize error
        self.error = np.zeros(0, dtype=np.float32)

    def saveResults(self):
        raise NotImplementedError

    def initializeObject(self):
        self.logger.info('Initial object set to %s', self.initialObject)
        self.shape_O = (self.nlambda, self.nosm, self.npsm, self.nslice, np.int(self.data.No), np.int(self.data.No))
        self.initialObject = initialProbeOrObject(self.shape_O, self.initialObject, self.data).astype(np.complex64)

    def initializeProbe(self):
        self.logger.info('Initial probe set to %s', self.initialProbe)
        self.shape_P = (self.nlambda, self.nosm, self.npsm, self.nslice, np.int(self.data.Np), np.int(self.data.Np))
        self.initialProbe = initialProbeOrObject(self.shape_P, self.initialProbe, self.data).astype(np.complex64)

