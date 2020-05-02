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
        self.prepare_reconstruction()


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

        if self.data.operationMode == 'FPM':
            self.initialObject = 'fpm'
            self.initialProbe = 'fpm_circ'
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
        
        # TODO: do you need both positions and positions0 to be re-centered here?
        # Dirk says: This is really confusing, we should definitely avoid this.
        # center positions within the object grid

        self.positions = self.positions + self.data.No/2 - self.data.Np/2
        
        # Positions should be integers otherwise we won't be able to slice. Define here?
        self.positions = self.positions.astype(int) 

    def saveResults(self):
        raise NotImplementedError

    def initializeObject(self):
        self.logger.info('Initial object set to %s', self.initialObject)
        self.initialObject = initialProbeOrObject((self.nosm, np.int(self.data.No), np.int(self.data.No)),
                                                      self.initialObject, self.data).astype(np.complex64)

    def initializeProbe(self):
        self.initialProbe = initialProbeOrObject((self.npsm, np.int(self.data.Np), np.int(self.data.Np)),
                                                        self.initialProbe, self.data).astype(np.complex64)

