from fracPy.initialParams import parser
import pickle
from pathlib import Path


class Params:
    """A fudge empty class to have the same form of obj.params as in MATLAB"""

    def __init__(self):
        return


class Ptylab:

    def __init__(self, datafolder):

        self.dataFolder = Path(datafolder)
        if not self.dataFolder.exists():
            self.dataFolder.mkdir()



        # required properties
        

        # operation
        self.operationMode= None # 'FPM' or 'CPM': defines operation mode(FP / CP: Fourier / conventional ptychography)

        # physical properties
        self.wavelength= None # (operational) wavelength, scalar quantity
        self.spectralDensity = None # spectral density S = S(wavelength), vectorial quantity
        #  note: spectral density is required for polychromatic operation.
        # In this case, wavelength is still scalar and determines the lateral
        # pixel size of the meshgridgrid that all other wavelengths are
        # interpolated onto.

        # (entrance) pupil / probe sampling
        self.dxp= None # pixel size (entrance pupil plane)
        self.Np= None # number of pixel (entrance pupil plane)
        self.xp= None # 1D coordinates (entrance pupil plane)
        self.Xp= None # 2D meshgrid in x-direction (entrance pupil plane)
        self.Yp= None # 2D meshgrid in y-direction (entrance pupil plane)
        self.Lp= None # field of view (entrance pupil plane)
        self.zp= None # distance to next plane of interest

         # object sampling



        self.ptychogram = None

        # constructor
        self.params = Params()
        self.params.__dict__ = parser

    def save(self, name='obj'):
        with open(self.dataFolder.joinpath('%s.pkl' % name), 'wb') as openfile:
            pickle.dump(self, openfile)
