from fracPy.utils.gpuUtils import transfer_fields_to_cpu, transfer_fields_to_gpu
import numpy as np
try:
    import pyqtgraph as pg
except ImportError:
    print('Cannot use pyqtgraph')
import matplotlib.pyplot as plt
# from pathlib import Path
import logging
# import tables
from fracPy.io import readHdf5
# from fracPy.io import readExample
from fracPy.utils.visualisation import show3Dslider
from fracPy.utils.visualisation import setColorMap
from fracPy.utils.gpuUtils import getArrayModule, transfer_fields_to_gpu, transfer_fields_to_cpu


class ExperimentalData:
    """
    This is a container class for all the data associated with the ptychography reconstruction.
    It only holds attributes that are the same for every type of reconstruction.
    """

    def __init__(self, filename=None, operationMode='CPM'):
        self.logger = logging.getLogger('ExperimentalData')
        self.logger.debug('Initializing ExperimentalData object')

        self.operationMode = operationMode        # operationMode: 'CPM' or 'FPM', default is CPM is not given
        self._setFields()
        if filename is not None:
            self.loadData(filename)

        # which fields have to be transferred if GPU operation is required?
        # not all of them are always used, but the class will determine by itself which ones are
        # required
        self.fields_to_transfer = [
            'emptyBeam',
            'ptychogram',
            'ptychogramDownsampled',
            'W',  # for aPIE
        ]

    def _setFields(self):
        """
        Set the required and optional fields for ptyLab to work.
        ALL VALUES MUST BE IN METERS.
        """
        # These are the fields required for ptyLab to work (depending on the operationMode)
        if self.operationMode == 'CPM':
            self.requiredFields = [
                'ptychogram',  # 3D image stack
                'wavelength',  # illumination lambda
                'encoder',  # diffracted field positions
                'dxd',  # pixel size
                'zo'  # sample to detector distance
            ]
            self.optionalFields = [
                'entrancePupilDiameter',  # used in CPM as the probe diameter
                'spectralDensity',  # CPM parameters: different wavelengths required for polychromatic ptychography
                'theta',  # CPM parameters: reflection tilt angle, required for
                'emptyBeam' # image of the probe
            ]

        elif self.operationMode == 'FPM':
            self.requiredFields = [
                'ptychogram',  # 3D image stack
                'wavelength',  # illumination lambda
                'encoder',  # diffracted field positions
                'dxd',  # detector pixel size
                'zo',  # sample to detector distance
                'zled',  # LED to sample distance
                'entrancePupilDiameter',
                # entrance pupil diameter, defined in lens-based microscopes as the aperture diameter, reqquired for FPM
            ]
            self.optionalFields = [
                'dxp',  # probe pixel size
                'magnification',  # magnification, used for FPM computations of dxp
            ]
        else:
            raise ValueError('operationMode is not properly set, choose "CPM" or "FPM"')



    def loadData(self, filename=None):
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
        readHdf5.checkDataFields(self.filename, self.requiredFields)
        # 2. load dictionary. Only the values specified by 'requiredFields'
        # in readHdf.py file were loaded
        measurementDict = readHdf5.loadInputData(self.filename, self.requiredFields, self.optionalFields)
        # 3. 'requiredFields' will be the attributes that must be set
        attributesToSet = measurementDict.keys()
        # 4. set object attributes as the essential data fields
        # self.logger.setLevel(logging.DEBUG)
        for a in attributesToSet:
            # make sure that property is not an attribtue
            attribute = str(a)
            if not isinstance(getattr(type(self), attribute, None), property):
                setattr(self, attribute, measurementDict[a])
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

        else:
            raise ValueError(f'Orientation {orientation} is not implemented')


    def _setData(self):

        # Set the detector coordinates
        self.Nd = self.ptychogram.shape[-1]
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
        print(f'Min max ptychogram: {np.min(self.ptychogram)}, {self.ptychogram.max()}')
        log_ptychogram = xp.log10(
            xp.swapaxes(
                np.clip(self.ptychogram.astype(np.float),0, None),
                1,2)+1)
        print(f'Min max ptychogram: {np.min(log_ptychogram)}, {log_ptychogram.max()}')
        show3Dslider(log_ptychogram)


    def _move_data_to_cpu(self):
        """ Move all required data to the CPU """
        transfer_fields_to_cpu(self, self.fields_to_transfer, self.logger)

    def _move_data_to_gpu(self):
        """ Move all required fata to the GPU"""
        transfer_fields_to_gpu(self, self.fields_to_transfer, self.logger)