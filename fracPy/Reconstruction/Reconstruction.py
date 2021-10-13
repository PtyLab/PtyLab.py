import numpy as np
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from copy import copy
import logging
import h5py
# logging.basicConfig(level=logging.DEBUG)

from fracPy.utils.initializationFunctions import initialProbeOrObject
from fracPy.utils.gpuUtils import transfer_fields_to_cpu, transfer_fields_to_gpu
from fracPy import Params
from fracPy.Operators.Operators import aspw, scaledASP, scaledASPinv


class Reconstruction(object):
    """
    This object will contain all the things that can be modified by a reconstruction.

    In itself, it's little more than a data holder. It is initialized with an ExperimentalData object.

    Some parameters which are "immutable" within the ExperimentalData can be modified
    (e.g. zo modification by zPIE during the reconstruction routine). All of them
    are defined in the listOfReconstructionProperties
    """
    listOfReconstructionPropertiesCPM = [
            'wavelength',
            'zo',
            'dxd',
            'theta',
            'spectralDensity',
            'entrancePupilDiameter'
        ]
    listOfReconstructionPropertiesFPM = [
            'wavelength',
            'zo',
            'dxd',
            'zled',
            'dxp',
            'entrancePupilDiameter'
        ]

    def __init__(self, data:ExperimentalData, params: Params):
        self.wavelength = None
        self.zo = None
        self.dxd = None
        self.theta = None

        self.logger = logging.getLogger('Reconstruction')
        self.data = data
        self.params = params
        self.copyAttributesFromExperiment(data)
        self.computeParameters()
        self.initializeSettings()

        # list of the fields that have to be transfered back and forth from the GPU
        self.possible_GPU_fields = ['probe',
                       'object',
                       'probeBuffer',
                       'objectBuffer',
                       'probeMomentum',
                       'objectMomentum',
                       'detectorError',
                       'background',
                                    'reference',
                                    ]

    def copyAttributesFromExperiment(self, data:ExperimentalData):
        """
        Copy all the attributes from the experiment that are in listOfReconstructionProperties (CPM or FPM)
        """
        self.logger.debug('Copying attributes from Experimental Data')
        if self.data.operationMode == 'CPM':
            listOfReconstructionProperties = self.listOfReconstructionPropertiesCPM
        elif self.data.operationMode == 'FPM':
            listOfReconstructionProperties = self.listOfReconstructionPropertiesFPM
        for key in listOfReconstructionProperties:
            self.logger.debug('Copying attribute %s', key)
            # setattr(self, key, copy(np.array(getattr(data, key))))
            setattr(self, key, copy(getattr(data, key)))


    def computeParameters(self):
        """
        compute parameters that can be altered by the user later.
        """

        if self.data.operationMode == 'CPM':
            # CPM dxp (depending on the propagatorType, if none given, assum Fraunhofer/Fresnel)
            self.dxp = self.wavelength * self.zo / self.Ld
            # if entrancePupilDiameter is not provided in the hdf5 file, set it to be one third of the probe FoV.
            if isinstance(self.entrancePupilDiameter, type(None)):
                self.entrancePupilDiameter = (self.Lp/3).copy()
            # if spectralDensity is not provided in the hdf5 file, set it to be a 1d array of the wavelength
            if isinstance(self.spectralDensity, type(None)):
                # this is a confusing name, it should be the wavelengths, not the intensity of the different
                # wavelengths
                self.spectralDensity = np.atleast_1d(self.wavelength)

        elif self.data.operationMode == 'FPM':
            # FPM dxp (different from CPM due to lens-based systems)
            if isinstance(self.dxp, type(None)):
                self.dxp = self.dxd / self.data.magnification

        # set object pixel numbers
        self.No = self.Np*2**2 # unimportant but leave it here as it's required for self.positions
        # we need space for the probe as well, on both sides that would be half the probe
        range_pixels = np.max(self.positions, axis=0) - np.min(self.positions, axis=0)
        print(range_pixels)
        range_pixels = np.max(range_pixels) + self.Np*2
        if range_pixels %2 == 1:
            range_pixels += 1
        self.No = np.max([self.Np, range_pixels])

    def make_alignment_plot(self, saveit=False):
        import time
        t0 = time.time()
        p_new = self.positions.T
        p_old = self.positions0.T

        from bokeh.plotting import figure, output_file, save
        from bokeh.layouts import row

        from pathlib import Path
        if saveit:
            output = Path('plots/alignment.html')
            output.parent.mkdir(exist_ok=True)
            # set output to static HTML file

            output_file(filename=output, title="Static HTML file", mode='inline')



        # create a new plot with a specific size
        p = figure(sizing_mode="stretch_width", max_width=500, height=500,
                   title=f'alignment (updated {time.strftime("%Y%h%d, %H:%M:%S")})',
                   )
        p.match_aspect = True
        square = p.square(p_old[0], p_old[1], fill_color='yellow', size=5, legend_label='original')
        # add a circle renderer for the new points
        circle = p.circle(p_new[0], p_new[1], fill_color="red", size=5, legend_label='new')

        p.xaxis.axis_label= 'Position x [um]'
        p.yaxis.axis_label = 'Position y [um]'

        p2 = None
        p3 = None

        figsize=  500 # px

        if hasattr(self, 'zHistory'): # display the plot of the defocus
            p2 = figure(sizing_mode="stretch_width", max_width=figsize, height=figsize, title='focus history',
                       )
            p2.circle(np.arange(len(self.zHistory)), np.array(self.zHistory)*1e3)
            p2.xaxis.axis_label = 'Iteration #'
            p2.yaxis.axis_label = 'Position [mm]'
            # p = vplot(p, p2)

        if hasattr(self, 'merit'): # display the merit as well for defocii
            p3 = figure(sizing_mode="stretch_width", max_width=figsize, height=figsize, title='merit TV',
                       )
            p3.circle(self.dz*1e3, np.array(self.merit),legend_label='original')
            p3.square(-self.dz * 1e3, np.array(self.merit), legend_label='mirrored')
            p3.xaxis.axis_label = 'Defocus [mm]'
            p3.yaxis.axis_label = 'Score [a.u.]'
            # p = vplot(p, p3)
        # only add the plots that are available
        p_list = filter(lambda x: x is not None, [p, p2, p3])
        p = row(*p_list)




        if saveit:
            save(p, )
        t1 = time.time()
        print(f'Alignment display took {t1-t0} secs')
        return p


    def initializeSettings(self):
        """
        Initialize the attributes that have to do with a reconstruction 
        or experimentalData fields which will become "reconstruction"

        This method just sets the settings. It sets the what kind of initial guess should be used for initialObject
        and initialProbe but it does not compute them yet. That will be done by calling initializeObjectProbe()

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


    def initializeObjectProbe(self):
        
        # initialize object and probe
        self.initializeObject()
        self.initializeProbe()
        
        # set object and probe objects
        self.object = self.initialGuessObject.copy()
        self.probe = self.initialGuessProbe.copy()


    def initializeObject(self, type_of_init=None):
        if type_of_init is not None:
            self.initialObject = type_of_init
        self.logger.info('Initial object set to %s', self.initialObject)
        self.shape_O = (self.nlambda, self.nosm, 1, self.nslice, np.int(self.No), np.int(self.No))
        self.initialGuessObject = initialProbeOrObject(self.shape_O, self.initialObject, self, self.logger).astype(np.complex64)

        self.initialGuessObject *= 1e-2

    def initializeProbe(self):
        self.logger.info('Initial probe set to %s', self.initialProbe)
        self.shape_P = (self.nlambda, 1, self.npsm, self.nslice, np.int(self.Np), np.int(self.Np))
        self.initialGuessProbe = initialProbeOrObject(self.shape_P, self.initialProbe, self).astype(np.complex64)

    # initialize momentum, called in specific engines with momentum accelaration
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
                if self.theta!=None:
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
        return np.linspace(-self.Nd / 2, self.Nd / 2, np.int(self.Nd)) * self.dxd

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
            return np.linspace(-self.Np / 2, self.Np / 2, int(self.Np)) * self.dxp
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
            return np.linspace(-self.No / 2, self.No / 2, np.int(self.No)) * self.dxo
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
        if self.data.operationMode == 'FPM':
            conv = -(1 / self.wavelength) * self.dxo * self.Np
            positions = np.round(
                conv * self.data.encoder / np.sqrt(self.data.encoder[:, 0] ** 2 + self.data.encoder[:, 1] ** 2 + self.zled ** 2)[
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


    def _move_data_to_cpu(self):
        """
        Move all the required fields to the CPU
        :return:
        """
        transfer_fields_to_cpu(self, self.possible_GPU_fields, self.logger)

    def _move_data_to_gpu(self):
        transfer_fields_to_gpu(self, self.possible_GPU_fields, self.logger)


    def describe_reconstruction(self):
        info = f"""
        Experimental data:
        - Number of ptychograms: {self.data.ptychogram.shape}
        - Number of pixels ptychogram: {self.data.Nd}
        - Ptychogram size: {self.data.Ld*1e3} mm
        - Pixel pitch: {self.data.dxd*1e6} um
        - Scan size: {1e3*(self.data.encoder.max(axis=0) - self.data.encoder.min(axis=0))} mm 
        
        Reconstruction:
        - number of pixels: {self.No}
        - Pixel pitch: {self.dxo*1e6} um
        - Field of view: {self.Lo*1e3} mm
        - Scan size in pixels: {self.positions.max(axis=0)- self.positions.min(axis=0)}
        - Propagation distance: {self.zo*1e3} mm
        - Probe FoV: {self.Lp*1e3} mm
        
        Derived parameters:
        - NA detector: {self.NAd}
        - DOF: {self.DoF*1e6} um
        
        """
        self.logger.info(info)
        return info

    @property
    def quadraticPhase(self):
        """ These functions are cached internally in Python and therefore no longer required. """
        raise NotImplementedError('Quadratic phase is no longer cached. ')

    @property
    def transferFunction(self):
        raise NotImplementedError('Quad phase is not longer cached')

    @property
    def Q1(self):
        raise NotImplementedError('Q1 is no longer available')

    @property
    def Q2(self):
        raise NotImplementedError('Q2 is no longer available')