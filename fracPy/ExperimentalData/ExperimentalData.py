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
    Things that belong to a particular type of reconstructor are stored in the .params class of that particular reconstruction.
    """

    def __init__(self, filename=None):
        self.logger = logging.getLogger('ExperimentalData')
        self.logger.debug('Initializing ExperimentalData object')

        self._initializeAttributes()
        self.filename = filename
        if filename is not None:
            self.loadData(filename)

    def _initializeAttributes(self):
        """
        Initialize all the attributes to PtyLab.
        """
        # operation
        self.operationMode = None  # 'FPM' or 'CPM': defines operation mode(FP / CP: Fourier / conventional ptychography)

        # physical properties
        self.entrancePupilDiameter = None
        self.spectralDensity = None  # spectral density is required for polychromatic operation. self.wavelength = min(self.spectralDensity)
        self.binningFactor = None  # binning factor that was applied to raw data
        self.padFactor = None
        self.magnificaiton = None
        self.dxp = None
        self.No = None
        self.positions0 = None
        # positions0 and positions are pixel number, encoder is in meter,
        # positions0 stores the original scan grid, positions is defined as property, automatically updated with dxo

        # python-only
        self._on_gpu = False # Sets wether things are on the GPU or not


    def loadData(self, filename=None, python_order=True):
        """
        Load data specified in filename.
        
        :type filename: str or Path
            Filename of dataset. There are three additional options:
                - example:simulationTiny will load a tiny simulation.
                - example:fpm_dataset will load an example fpm dataset.
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
            setattr(self, str(a), measurement_dict[a])
            self.logger.debug('Setting %s', a)

        self._setGrid()
        # self._checkData()

    def _setGrid(self):
        """
        Set the probe pixel size (by default far-field ptychography), and the object pixel number,
        which determines all calculation grid for the probe and object.
        """
        if self.dxp is None:
            self.dxp = self.wavelength * self.zo / self.Ld
        if self.No is None:
            self.No = 2**10
        if self.positions0 is None:
            self.positions0 = self.positions.copy()
        if self.spectralDensity is None:
            self.spectralDensity = [self.wavelength]


    # def _checkData(self):
    #     """
    #     Check that at least all the data we need has been initialized.
    #     :return: None
    #     :raise: ValueError when one of the required fields are missing.
    #     """
    #     if self.ptychogram is None:
    #         raise ValueError('ptychogram is not loaded correctly.')


    def showPtychogram(self):
        """
        show ptychogram.
        """
        xp = getArrayModule(self.ptychogram)
        show3Dslider(xp.log10(xp.swapaxes(self.ptychogram, 1,2)+1))
        print('Maximum count in ptychogram is %d'%(np.max(self.ptychogram)))


    # Set attributes using @property operators: they are set automatically with the functions defined by the
    # @property operators

    # Detector property list
    @property
    def xd(self):
        """ Detector coordinates 1D """
        try:
            return np.linspace(-self.Nd/2, self.Nd/2-1, np.int(self.Nd))*self.dxd
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
            return np.linspace(-self.Np/2,self.Np/2-1, np.int(self.Np))*self.dxp
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

    @property
    def numFrames(self):
        return self.ptychogram.shape[0]

    # Object property list
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
            return np.linspace(-self.No/2,self.No/2-1, np.int(self.No))*self.dxo
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
        DoF = self.wavelength / self.NAd ** 2
        return DoF


if __name__ == '__main__':
    app = pg.mkQApp()
    e = ExperimentalData('example:simulation_ptycho')
    e.showPtychogram()
    app.exec_()

