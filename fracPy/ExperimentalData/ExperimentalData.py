import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt
# from pathlib import Path
import logging
# import tables
from fracPy.io import readHdf5
# from fracPy.io import readExample
from fracPy.utils.visualisation import show3Dslider
from fracPy.utils.visualisation import setColorMap
from fracPy.utils.gpuUtils import getArrayModule


class ExperimentalData:
    """
    This is a container class for all the data associated with the ptychography reconstruction.
    It only holds attributes that are the same for every type of reconstruction.
    """

    def __init__(self, filename=None):
        self.logger = logging.getLogger('ExperimentalData')
        self.logger.debug('Initializing ExperimentalData object')

        self.filename = filename
        if filename is not None:
            self.loadData(filename)


    def loadData(self, filename=None, python_order=True):
        """
        Load data specified in filename.
        :type filename: str or Path
            Filename of dataset. There are three additional options:
                - example:simulation_cpm will load an example cmp dataset.
                - example:simulation_fpm will load an example fpm dataset.
                - test:nodata will load an essentially empty object
        :param python_order: bool
                Weather to change the input order of the files to match python convention.
                 Only in very special cases should this be false.
        :return:
        """
        import os
        if not os.path.exists(filename) and str(filename).startswith('example:'):
            # Todo @dbs660 Fix this so it works again
            self.filename = filename
            from fracPy.io.readExample import examplePath
            self.filename = examplePath(filename)  # readExample(filename, python_order=True)
        else:
            self.filename = filename

        # 1. check if the dataset contains what we need before loading
        readHdf5.checkDataFields(self.filename)
        # 2. load dictionary. Only the values specified by 'required_fields'
        # in readHdf.py file were loaded
        measurement_dict = readHdf5.loadInputData(self.filename)
        # 3. 'required_fields' will be the attributes that must be set
        attributes_to_set = measurement_dict.keys()
        # 4. set object attributes as the essential data fields
        # self.logger.setLevel(logging.DEBUG)
        for a in attributes_to_set:

            # make sure that property is not an  attribtue
            attribute = str(a)
            if not isinstance(getattr(type(self), attribute, None), property):
                setattr(self, attribute, measurement_dict[a])
            self.logger.debug('Setting %s', a)

        self._setData()


    def _setData(self):
        # Set the detector coordinates (detector pixelsize dxd must be given from the hdf5 file.)
        if self.Nd == None:
            self.Nd = self.ptychogram.shape[-1]

        # Detector coordinates 1D
        self.xd = np.linspace(-self.Nd/2, self.Nd/2-1, np.int(self.Nd))*self.dxd
        # Detector coordinates 2D
        self.Xd, self.Yd = np.meshgrid(self.xd, self.xd)
        # Detector size in SI units
        self.Ld = self.Nd * self.dxd

        # number of Frames
        self.numFrames = self.ptychogram.shape[0]
        # probe energy at each position
        self.energyAtPos = np.sum(abs(self.ptychogram), (-1, -2))
        # maximum probe power
        self.maxProbePower = np.sqrt(np.max(np.sum(self.ptychogram, (-1, -2))))


    def showPtychogram(self):
        """
        show ptychogram.
        """
        xp = getArrayModule(self.ptychogram)
        show3Dslider(xp.log10(xp.swapaxes(self.ptychogram, 1,2)+1))
        print('Maximum count in ptychogram is %d'%(np.max(self.ptychogram)))  #todo: make this the title