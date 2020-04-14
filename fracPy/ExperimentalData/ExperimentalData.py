import numpy as np
from pathlib import Path
import logging
import tables
from fracPy.io import readHdf5
from fracPy.io import readExample

class ExperimentalData:
    """
    This is a container class for all the data associated with the ptychography reconstruction.

    It only holds attributes that are the same for every type of reconstruction.

    Things that belong to a particular type of reconstructor are stored in the .params class of that particular reconstruction.

    """

    def __init__(self, filename=None):
        self.logger = logging.getLogger('ExperimentalData')
        self.logger.debug('Initializing ExperimentalData object')

        self.filename = filename
        self.initializeAttributes()
        if filename is not None:
            self.loadData(filename)


    def initializeAttributes(self):
        """
        Initialize all the attributes to PtyLab.
        """
        # required properties

        # operation
        self.operationMode = None  # 'FPM' or 'CPM': defines operation mode(FP / CP: Fourier / conventional ptychography)

        # physical properties
        self.wavelength = None  # (operational) wavelength, scalar quantity
        self.spectralDensity = None  # spectral density S = S(wavelength), vectorial quantity
        #  note: spectral density is required for polychromatic operation.
        # In this case, wavelength is still scalar and determines the lateral
        # pixel size of the meshgridgrid that all other wavelengths are
        # interpolated onto.

        # (entrance) pupil / probe sampling
        self.dxp = None  # pixel size (entrance pupil plane)
        self.Np = None  # number of pixel (entrance pupil plane)
        self.xp = None  # 1D coordinates (entrance pupil plane)
        self.Xp = None  # 2D meshgrid in x-direction (entrance pupil plane)
        self.Yp = None  # 2D meshgrid in y-direction (entrance pupil plane)
        self.Lp = None  # field of view (entrance pupil plane)
        self.zp = None  # distance to next plane of interest

        # object sampling

        # object sampling
        self.dxo = None  # pixel size (object plane)
        self.No = None  # number of pixel (object plane)
        self.xo = None  # 1D coordinates (object plane)
        self.Xo = None  # 2D meshgrid in x-direction (object plane)
        self.Yo = None  # 2D meshgrid in y-direction (object plane)
        self.Lo = None  # field of view (object plane)
        self.zo = None  # distance to next plane of interest

        # detector sampling
        self.dxd = None  # pixel size (detector plane)
        self.Nd = None  # number of pixel (detector plane)
        self.xd = None  # 1D coordinates (detector plane)
        self.Xd = None  # 2D meshgrid in x-direction (detector plane)
        self.Yd = None  # 2D meshgrid in y-direction (detector plane)
        self.Ld = None  # field of view (detector plane)

        # measured intensities
        self.ptychogram = None  # intensities [Nd, Nd, numPos]
        self.numFrames = None  # number of measurements (positions (CPM) / LED tilts (FPM))
        self.background = None  # background
        self.binningFactor = None  # binning factor that was applied to raw data

        # measured positions
        self.positions0 = None  # initial positions in pixel units (real space for CPM, Fourier space for FPM)
        self.positions = None  # estimated positions in pixel units (real space for CPM, Fourier space for FPM)
        # note: Positions are given in row-column order and refer to the
        # pixel in the upper left corner of the respective data matrix;
        # -1st example: suppose the 2nd row of positions0 is [3, 4] and the
        # operation mode is 'CPM'. That implies that the second intensity
        # in the spectrogram updates an object patch that has
        # its left uppper corner pixel at the pixel coordinates [3, 4]
        # -2nd example: suppose the 2nd row of positions0 is [3, 4] and the
        # operation mode is 'FPM'. That implies that the second intensity
        # in the spectrogram is updates a patch which has pixel coordinates
        # [3,4] in the high-resolution Fourier transform

        self.ptychogram = None

        # Python-only
        # checkGPU

        # python-only
        self._on_gpu = False # Sets wether things are on the GPU or not


    def transfer_to_gpu_if_applicable(self):
        """ Implements checkGPU"""
        pass


    ### Functions that still have to be implemented
    def checkDataset(self):
        raise NotImplementedError()

    def _loadDummyData(self):
        """
        For testing purposes, we often don't need a full dataset. This function will populate the
        attributes with dummy settings.
        So far it performs the following tasks:
            * it sets the wavelength to 1234
            * if sets the positions to np.random.rand(100,2)
        :return:
        """
        self.wavelength = 1234
        self.positions = np.random.rand(100,2)


    def initialParams(self):
        """ Initialize the params object and attach it to the object

        Note that this is a little bit different from the matlab implementation
        """
        # self.params = Params()
        raise NotImplementedError()


    def setPositionOrder(self):
        raise NotImplementedError()

    def showDiffractionData(self):
        raise NotImplementedError()

    def showPtychogram(self):
        raise NotImplementedError()

    ## Special properties
    # so far none

    def loadData(self, filename=None, python_order=True):
        """
        Load data specified in filename.
        
        :type filename: str or Path
            Filename of dataset. There are three additional options:
                - example:simulationTiny will load a tiny simulation.
                - example:fpm_dataset will load an example fpm dataset.
                - test:nodata will load an essentially empty object
        :param python_order: bool
                Wether to change the input order of the files to match python convention.
                 Only in very special cases should this be false.
        :return:
        """

        if filename is not None:
            self.filename = filename

        if filename == 'test:nodata':
            # Just create the attributes but don't load data.
            # This is mainly useful for testing the object structure
            self._loadDummyData()
            return

        if str(filename).startswith('example:'):
            # only take the key
            filename = str(filename).split('example:')[-1]
            # All the examples have the normal ordering of variables, so this should be true
            if not python_order:
                self.logger.error('Requested to load an example with python_order = False. ' +\
                                  'All the examples are supposed to be loaded with python_order=True, so ignoring it.')
            from fracPy.io.readExample import examplePath
            self.filename = examplePath(filename)#readExample(filename, python_order=True)


        if self.filename is not None:
            # 1. check if the dataset contains what we need before loading
            readHdf5.checkDataFields(self.filename)
            # 2. load dictionary. Only the values specified by 'required_fields' 
            # in readHdf.py file were loaded 
            measurement_dict = readHdf5.loadInputData(self.filename, python_order)
            # 3. 'required_fields' will be the attributes that must be set
            attributes_to_set = measurement_dict.keys()
            # 4. set object attributes as the essential data fields
            for a in attributes_to_set:
                setattr(self, a, measurement_dict[a])
        
        self._checkData()

    def _checkData(self):
        """
        Check that at least all the data we need has been initialized.
        :return: None
        :raise: ValueError when one of the required fields are missing.
        """
        if self.ptychogram is None:
            raise ValueError('ptychogram is not loaded correctly.')
        # TODO: Check all the necessary requirements

if __name__ == '__main__':
    e = ExperimentalData('hoi')
