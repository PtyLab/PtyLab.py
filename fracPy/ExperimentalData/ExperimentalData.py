import numpy as np
from pathlib import Path

class ExperimentalData(object):
    def __init__(self, filename=None):
        # instance attributes are copied from ptyLab matlab implementation
        # @Tomas: You know best what the Matlab implementation will be like
        # so feel free to change the name of filename.
        self.initialize_attributes()
        self.load_data(filename)




    def initialize_attributes(self):
        """
        Initialize all the attributes to PtyLab.
        """

        self.dataFolder = Path(self.dataFolder)
        if not self.dataFolder.exists():
            self.logger.info('Datafolder %s does not exist yet. Creating it.',
                             self.dataFolder)
            self.dataFolder.mkdir()

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
        self.to_GPU = False


    def load_data(self, filename=None):
        """
        @Tomas: Please implement your hdf5 loader here
        :return:
        """
        if filename is not None:
            self.filename = filename

        if self.filename is not None:
            # check that all the data is present.
            from fracPy import io
            io.read_hdf5.check_data_fields(self.filename)
            raise NotImplementedError('Loading files is not implemented yet')

        self._checkdata()

    def _checkdata(self):
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
