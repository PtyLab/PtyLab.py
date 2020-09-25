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
        self.M = 1 # magnification, will change if a lens is used (for FPM)               
        #  note: spectral density is required for polychromatic operation.
        # In this case, wavelength is still scalar and determines the lateral
        # pixel size of the meshgridgrid that all other wavelengths are
        # interpolated onto.
        
        # measured intensities
        self.ptychogram = None  # intensities [Nd, Nd, numPos]


        self.background = None  # background
        self.binningFactor = None  # binning factor that was applied to raw data

        # measured positions
        # self.positions0 = None  # initial positions in pixel units (real space for CPM, Fourier space for FPM)
        # self.positions = None  # estimated positions in pixel units (real space for CPM, Fourier space for FPM)
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

        self.probe = None

        self.ptychogram = None

        # Things implemented as property :
        # self.numFrames = None  # number of measurements (positions (CPM) / LED tilts (FPM))

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
        self.probe = np.zeros((1, 32,32), np.complex64)
        self.aperture = np.zeros((1, 32,32), np.complex64)
        self.object = np.zeros((1,33,33), np.complex64)
        self.No = 32
        self.Nd = 55
        self.Np = 32
        self.zo = 1
        self.dxd = 1
        self.entrancePupilDiameter = 10
        #self.Np = 33


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
            measurement_dict = readHdf5.loadInputData(self.filename)
            # 3. 'required_fields' will be the attributes that must be set
            attributes_to_set = measurement_dict.keys()
            # print('Attributes_to_set:', list(attributes_to_set))
            # 4. set object attributes as the essential data fields
            # self.logger.setLevel(logging.DEBUG)
            for a in attributes_to_set:
                setattr(self, str(a), measurement_dict[a])
                self.logger.debug('Setting %s', a)

            # 5. Set other attributes based on this
            # they are set automatically with the functions defined by the
            # @property operators
            if str(filename).startswith('simulation'):
                # Positions and Np, No should be integers otherwise we won't be able to slice. Define here?
                self.positions = self.positions.astype(int)
                self.Np = self.Np.astype(int)
                self.Nd = self.Nd.astype(int)
                self.No = self.No.astype(int)
                self.encoder = (self.positions + self.No / 2 - self.Np / 2) * self.dxo



        self._checkData()


    
    # Detector property list
    @property
    def xd(self):
        """ Detector coordinates 1D """
        try:
            return np.linspace(-self.Nd/2,self.Nd/2-1, np.int(self.Nd))*self.dxd
        except AttributeError as e:
            raise AttributeError(e, 'pixel number "Nd" and/or pixel size "dxd" not defined yet')

    @property
    def Xd(self):
        """ Detector coordinates 2D """
        Xd, Yd = np.meshgrid(self.xd, self.xd)
        return Xd
            
    @property
    def Yd(self):
        """ Detector coordinates 2D """
        Xd, Yd = np.meshgrid(self.xd, self.xd)
        return Yd
            
    @property       
    def Ld(self):
        """ Detector size in SI units. """
        try:
            return self.Nd * self.dxd
        except AttributeError as e:
            raise AttributeError(e, 'pixel number "Nd" and/or pixel size "dxd" not defined yet')
    
    
    
    # Probe property list
    @property
    def dxp(self):
        """ Probe sampling. Requires the probe to be set."""
        try:
            return self.wavelength * self.zo / self.Ld # 1./(self.Ld/self.M)
        except AttributeError as e:
            raise AttributeError(e, 'Detector size "Ld" and/or magnification "M" not defined yet')
         
    # if thre is no probe known let the user just provide Np
    # @property
    # def Np(self):
    #     """ Number of pixels of the probe. Requires the probe to be set."""
    #     try:
    #         return self.probe.shape[-1]
    #     except AttributeError as e:
    #         raise AttributeError(e, 'probe is not defined yet')

    @property
    def Lp(self):
        """ Field of view (entrance pupil plane) """
        return self.Np * self.dxp
   
    @property
    def xp(self):
        """ Detector coordinates 1D """
        try:
            return np.linspace(-self.Np/2,self.Np/2-1, np.int(self.Np))*self.dxp
        except AttributeError as e:
            raise AttributeError(e, 'probe pixel number "Np" and/or probe sampling "dxp" not defined yet')
            
    @property
    def Xp(self):
        """ Detector coordinates 2D """
        Xp, Yp = np.meshgrid(self.xp, self.xp)
        return Xp
            
    @property
    def Yp(self):
        """ Detector coordinates 2D """
        Xp, Yp = np.meshgrid(self.xp, self.xp)
        return Yp

    @property
    def numFrames(self):
        return self.ptychogram.shape[0]
    # @property
    # def entrancePupilDiameter(self):
    #     """ pupil diameter in pixels (FPM property, not CPM) """    
    
    
    # Object property list
    @property
    def dxo(self):
        """ Probe sampling. Requires the probe to be set."""
        try:
            # also obj.lambda * obj.zo / obj.Ld ?
            # return 1./(self.Ld/self.M)
            return self.wavelength * self.zo / self.Ld
        except AttributeError as e:
            raise AttributeError(e, 'Detector size "Ld" and/or magnification "M" not defined yet')
        
    @property
    def Lo(self):
        """ Field of view (entrance pupil plane) """
        return self.No * self.dxo
   
    @property
    def xo(self):
        """ Detector coordinates 1D """
        try:
            return np.linspace(-self.No/2,self.No/2-1, np.int(self.No))*self.dxo
        except AttributeError as e:
            raise AttributeError(e, 'object pixel number "No" and/or pixel size "dxo" not defined yet')
            
    @property
    def Xo(self):
        """ Detector coordinates 2D """
        Xo, Yo = np.meshgrid(self.xo, self.xo)
        return Xo
            
    @property
    def Yo(self):
        """ Detector coordinates 2D """
        Xo, Yo = np.meshgrid(self.xo, self.xo)
        return Yo

    @property
    def positions(self):
        """scan positions in pixel"""
        positions = np.round(self.encoder/self.dxo)  # encoder is in m, positions0 and positions are in pixels
        positions = positions + self.No//2 - self.Np//2
        return positions.astype(int)

    # system property list
    @property
    def NAd(self):
        """ Detection NA"""
        NAd = self.Ld/(2*self.zo)
        return NAd

    @property
    def DoF(self):
        """expected Depth of field"""
        # DoF = self.wavelength[0]/self.NAd**2
        DoF = self.wavelength / self.NAd ** 2 #TODO: implement for multiwave
        return DoF


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
    e = ExperimentalData('example:simulation_fpm')
