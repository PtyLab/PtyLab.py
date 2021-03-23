import numpy as np
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from copy import copy
import logging
import h5py
# logging.basicConfig(level=logging.DEBUG)

from fracPy.utils.initializationFunctions import initialProbeOrObject


class Optimizable(object):
    """
    This object will contain all the things that can be modified by a reconstruction engine.

    In itself, it's little more than a data holder. It is initialized with an ExperimentalData object.

    Some parameters which are "immutable" within the ExperimentalData can be modified
    (e.g. zo modification by zPIE during the reconstruction routine). All of them
    are defined in the listOfOptimizableProperties
    """
    listOfOptimizableProperties = [
            'wavelength',
            'zo',
            'spectralDensity',
            'dxd',
            'dxp',
            'No',
            'entrancePupilDiameter'
        ]
    
    def __init__(self, data:ExperimentalData):
        self.logger = logging.getLogger('Optimizable')
        self.data = data
        self.copyAttributesFromExperiment(data)
        self.computeOptionalParameters(data)
        self.initialize_settings()

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
       
    def computeOptionalParameters(self, data:ExperimentalData):
        """
        There is a list of optional parameters within the readHdf5 class
        which can be loaded by the user or are set to None.
        If they are set to None, some of them will be computed in this
        function since they might be crucial for FPM but not CPM etc.
        :param data:
                Experimental data to copy from
        :return:
        """
        self.logger.debug('Computing optional attributed from Experimental Data')
        
        # Probe pixel size (depending on the propagator, if none given, assum Fraunhofer/Fresnel)
        if self.dxp == None:
            # CPM dxp
            if self.data.operationMode == 'CPM':
                self.dxp = self.wavelength * self.zo / self.Ld
                
            # FPM dxp (different from CPM due to lens-based systems)
            elif self.data.operationMode == 'FPM':
                if self.data.magnification != None:
                    self.dxp = self.dxd / self.data.magnification
                else:
                    self.logger.error('Neither dxp or magnification was provided. Add one of the parameters to the .hdf5 file')

        # Upsampled object plane dimensions
        if self.No == None:
            self.No = 2**11
            # self.No = self.Np+np.max(self.positions0[:,0])-np.min(self.positions0[:,0])

            
    def initialize_settings(self):
        """
        Initialize the attributes that have to do with a reconstruction 
        or experimentalData fields which will become "optimizable"

        This method just sets the settings. It sets the what kind of initial guess should be used for initialObject
        and initialProbe but it does not compute them yet. That will be done by calling prepare_reconstruction()

        :return:
        """
        # create a 6D object where which allows to have:
        # 1. polychromatic = nlambda
        # 2. mixed state object - nosm
        # 3. mixed state probe - npsm
        # 4. multislice object (thick) - nslice
        self.nlambda = 1
        self.nosm = 1
        self.npsm = 1
        self.nslice = 1

        # beam and object purity (# default initial value for plots.)
        self.purityProbe = 1
        self.purityObject = 1

        # decide whether the positions will be recomputed each time they are called or whether they will be fixed
        # without the switch, positions are computed from the encoder values
        # with the switch calling optimizable.positions will return positions0
        # positions0 and positions are pixel number, encoder is in meter,
        # positions0 stores the original scan grid, positions is defined as property, automatically updated with dxo
        self.fixedPositions = False
        self.positions0 = self.positions.copy()
        

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
        self.object = self.initialGuessObject.copy()
        self.probe = self.initialGuessProbe.copy()


    def initializeObject(self):
        self.logger.info('Initial object set to %s', self.initialObject)
        self.shape_O = (self.nlambda, self.nosm, 1, self.nslice, np.int(self.No), np.int(self.No))
        self.initialGuessObject = initialProbeOrObject(self.shape_O, self.initialObject, self).astype(np.complex64)

    def initializeProbe(self):
        self.logger.info('Initial probe set to %s', self.initialProbe)
        self.shape_P = (self.nlambda, 1, self.npsm, self.nslice, np.int(self.Np), np.int(self.Np))
        self.initialGuessProbe = initialProbeOrObject(self.shape_P, self.initialProbe, self).astype(np.complex64)

    def initializeObjectMomentum(self):
        self.objectMomentum = np.zeros_like(self.initialGuessObject)
        
    def initializeProbeMomentum(self):
        self.probeMomentum = np.zeros_like(self.initialGuessProbe)



    def saveResults(self, fileName='recent', type='all'):
        if type == 'all':
            hf = h5py.File(fileName + '_Reconstruction.hdf5', 'w')
            hf.create_dataset('probe', data=self.probe, dtype='complex64')
            hf.create_dataset('object', data=self.object, dtype='complex64')
            hf.create_dataset('error', data=self.error, dtype='f')
            hf.create_dataset('zo', data=self.zo, dtype='f')
            hf.create_dataset('wavelength', data=self.wavelength, dtype='f')
            hf.create_dataset('dxp', data=self.dxp, dtype='f')
            if hasattr(self, 'theta'):
                hf.create_dataset('theta', data=self.theta, dtype='f')
        elif type == 'probe':
            hf = h5py.File(fileName + '_probe.hdf5', 'w')
            hf.create_dataset('probe', data=self.probe, dtype='complex64')
        elif type == 'object':
            hf = h5py.File(fileName + '_object.hdf5', 'w')
            hf.create_dataset('object', data=self.object, dtype='complex64')

        hf.close()
        print('The reconstruction results (%s) have been saved' % type)

    # detector coordinates
    @property
    def Nd(self):
        return self.data.ptychogram.shape[1]


    @property
    def xd(self):
        """ Detector coordinates 1D """
        return np.linspace(-self.Nd / 2, self.Nd / 2 - 1, np.int(self.Nd)) * self.dxd

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
        return self.Nd * self.dxd



    # probe coordinates
    @property
    def Np(self):
        """Probe pixel numbers"""
        Np = self.Nd
        return Np

    @property
    def Lp(self):
        """ probe size in SI units """
        Lp = self.Np * self.dxp
        return Lp

    @property
    def xp(self):
        """ Probe coordinates 1D """
        try:
            return np.linspace(-self.Np / 2, self.Np / 2 - 1, np.int(self.Np)) * self.dxp
        except AttributeError as e:
            raise AttributeError(e, 'probe pixel number "Np" and/or probe sampling "dxp" not defined yet')

    @property
    def Xp(self):
        """ Probe coordinates 2D """
        Xp, Yp = np.meshgrid(self.xp, self.xp)
        return Xp

    @property
    def Yp(self):
        """ Probe coordinates 2D """
        Xp, Yp = np.meshgrid(self.xp, self.xp)
        return Yp



    # Object coordinates
    @property
    def dxo(self):
        """ object pixel size, always equal to probe pixel size."""
        dxo = self.dxp
        return dxo

    @property
    def Lo(self):
        """ Field of view (entrance pupil plane) """
        return self.No * self.dxo

    @property
    def xo(self):
        """ object coordinates 1D """
        try:
            return np.linspace(-self.No / 2, self.No / 2 - 1, np.int(self.No)) * self.dxo
        except AttributeError as e:
            raise AttributeError(e, 'object pixel number "No" and/or pixel size "dxo" not defined yet')

    @property
    def Xo(self):
        """ Object coordinates 2D """
        Xo, Yo = np.meshgrid(self.xo, self.xo)
        return Xo

    @property
    def Yo(self):
        """ Object coordinates 2D """
        Xo, Yo = np.meshgrid(self.xo, self.xo)
        return Yo

    # scan positions in pixel
    @property
    def positions(self):
        """estimated positions in pixel numbers(real space for CPM, Fourier space for FPM)
        note: Positions are given in row-column order and refer to the
        pixel in the upper left corner of the respective data matrix;
        -1st example: suppose the 2nd row of positions0 is [3, 4] and the
        operation mode is 'CPM'. That implies that the second intensity
        in the spectrogram updates an object patch that has
        its left uppper corner pixel at the pixel coordinates [3, 4]
        -2nd example: suppose the 2nd row of positions0 is [3, 4] and the
        operation mode is 'FPM'. That implies that the second intensity
        in the spectrogram is updates a patch which has pixel coordinates
        [3,4] in the high-resolution Fourier transform
        """
        if self.fixedPositions:
            return self.positions0
        else:
            if self.data.operationMode == 'FPM':
                conv = -(1 / self.wavelength) * self.dxo * self.Np
                positions = np.round(
                    conv * self.data.encoder / np.sqrt(self.data.encoder[:, 0] ** 2 + self.data.encoder[:, 1] ** 2 + self.zo ** 2)[
                        ..., None])
            else:
                positions = np.round(self.data.encoder / self.dxo)  # encoder is in m, positions0 and positions are in pixels
            positions = positions + self.No // 2 - self.Np // 2
            return positions.astype(int)

    # system property list
    @property
    def NAd(self):
        """ Detection NA"""
        NAd = self.Ld / (2 * self.zo)
        return NAd

    @property
    def DoF(self):
        """expected Depth of field"""
        DoF = self.wavelength / self.NAd ** 2
        return DoF