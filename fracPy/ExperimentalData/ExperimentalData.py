from fracPy.utils.gpuUtils import transfer_fields_to_cpu, transfer_fields_to_gpu
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
        self.logger = logging.getLogger('FixedData')
        self.logger.debug('Initializing FixedData object')

        self.filename = filename
        if filename is not None:
            self.loadData(filename)

        # which fields have to be transferred if GPU operation is required?
        # not all of them are always used, but the class will determine by itself which ones are
        # required
        self.fields_to_transfer = [
            'emptyBeam',
            'ptychogram',
            'ptychogramDownsampled',
            'W', # for aPIE
        ]






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


    def setOrientation(self, orientation):
        """
        Sets the correct orientation. This function follows the ptypy convention.
        """
        if not isinstance(orientation, int):
            raise TypeError("Orientation value is not valid.")
        if orientation == 1:
            # Invert column
            self.ptychogram = np.fliplr(self.ptychogram)
        elif orientation == 2:
            # Invert rows
            self.ptychogram = np.flipud(self.ptychogram)
        elif orientation == 3:
            # invert columns and rows
            self.ptychogram = np.fliplr(self.ptychogram)
            self.ptychogram = np.flipud(self.ptychogram)
        elif orientation == 4:
            # Transpose 
            self.ptychogram = np.transpose(self.ptychogram, (0, 2, 1)) 
        elif orientation == 5:
            self.ptychogram = np.transpose(self.ptychogram, (0, 2, 1)) 
            self.ptychogram = np.fliplr(self.ptychogram)
        elif orientation == 6:
            self.ptychogram = np.transpose(self.ptychogram, (0, 2, 1)) 
            self.ptychogram = np.flipud(self.ptychogram)
        elif orientation == 7:
            self.ptychogram = np.transpose(self.ptychogram, (0, 2, 1)) 
            self.ptychogram = np.fliplr(self.ptychogram)
            self.ptychogram = np.flipud(self.ptychogram)
        
    def _setData(self):
        # Set the detector coordinates (detector pixelsize dxd must be given from the hdf5 file.)
        if self.Nd == None:
            self.Nd = self.ptychogram.shape[-1]
        if isinstance(self.spectralDensity, type(None)):
            self.spectralDensity = np.atleast_1d(self.wavelength)
        if not hasattr(self, 'operationMode'):
            self.operationMode = 'CPM'

        # Detector coordinates 1D
        self.xd = np.linspace(-self.Nd/2, self.Nd/2, np.int(self.Nd))*self.dxd
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

    def _move_data_to_cpu(self):
        """ Move all required data to the CPU """
        transfer_fields_to_cpu(self, self.fields_to_transfer, self.logger)

    def _move_data_to_gpu(self):
        """ Move all required fata to the GPU"""
        transfer_fields_to_gpu(self, self.fields_to_transfer, self.logger)