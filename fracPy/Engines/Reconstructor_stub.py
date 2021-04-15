# Do not import this file!

import logging
# This is a template for anyone who wants to implement their own reconstructor.

from fracPy.FixedData.DefaultExperimentalData import ExperimentalData
from fracPy.Optimizables.Optimizable import Optimizable
from fracPy.Engines import BaseReconstructor

class MyOwnReconstructor(BaseReconstructor.BaseReconstructor):
    def __index__(self, optimizable: Optimizable, experimentalData: ExperimentalData):
        super().__init__(optimizable, experimentalData)
        self.logger = logging.getLogger(self.__name__)

        # Add your own super-specific items here


    def doReconstruction(self):
        """ This is where you implement your own ePIE_engine."""
        pass

