from fracPy.initialParams import parser
import pickle
from pathlib import Path
import logging
import tables

logging.basicConfig(level=logging.DEBUG)

class DataLoader:
    """
    This is a container class for all the data associated with the ptychography reconstruction.

    It only holds attributes that are the same for every type of reconstruction.

    Things that belong to a particular type of reconstructor are stored in the .params class of that particular reconstruction.

    """

    def __init__(self, datafolder):
        self.logger = logging.getLogger('PtyLab')
        self.logger.debug('Initializing PtyLab object')
        self.dataFolder = datafolder
        self.initialize_attributes()

        #self.prepare_reconstruction()



    def prepare_visualisation(self):
        """ Create figure and axes for visual feedback.

        """
        return NotImplementedError()





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

        # constructor
        #self.params = Params()
        #self.params.__dict__ = parser

        # things that are implemented as a property:
        # checkGPU

        # python-only
        self._on_gpu = False # Sets wether things are on the GPU or not

    def save(self, name='obj'):
        with open(self.dataFolder.joinpath('%s.pkl' % name), 'wb') as openfile:
            pickle.dump(self, openfile)

    def load_from_hdf5(self):
        try:
            hdf5_file = tables.open_file(next(self.dataFolder.glob('*.hdf5')), mode='r')
            # image_array_crops is name given for the images stored within hdf5
            try:
                images = hdf5_file.root.image_array_crops[:,:,:]
            except Exception as e:
                print(e)
                hdf5_file.close()   
            
            return images
        except Exception as e:
            print(e)
            return None

    def load(self, name='obj'):
        raise NotImplementedError()

    def transfer_to_gpu_if_applicable(self):
        """ Implements checkGPU"""
        pass



    ### Functions that still have to be implemented
    def checkDataset(self):
        raise NotImplementedError()

    def checkFFT(self):
        raise NotImplementedError()

    def getOverlap(self):
        """
        :return:
        """
        raise NotImplementedError()

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

    def prepare_reconstruction(self):
        """
        Prepare the reconstruction. So far, followin matlab, it checks the following things:

        - minimize memory footprint
        -
        :return:
        """
        # delete parts of the memory that are not required.
        self.checkMemory()
        # do something with the modes
        self.checkModes
        # prepare FFTs
        self.checkFFT()
        # prepare vis
        self.prepare_visualisation()
        # This should not be necessary
        # obj.checkMISC
        # transfer to GPU if req'd
        self.checkGPU()

    # Things we'd like to change the name of
    def checkGPU(self, *args, **kwargs):
        return self.transfer_to_gpu_if_applicable(*args, **kwargs)





if __name__ == '__main__':
    obj = DataLoader('.')
