import time
import numpy as np
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from copy import copy
import logging
import h5py

# logging.basicConfig(level=logging.DEBUG)
from PtyLab.Regularizers import metric_at, TV

from PtyLab.utils.initializationFunctions import initialProbeOrObject
from PtyLab.utils.gpuUtils import (
    transfer_fields_to_cpu,
    transfer_fields_to_gpu,
    getArrayModule,
)
from PtyLab import Params
from PtyLab.utils.gpuUtils import asNumpyArray


def calculate_pixel_positions(encoder_corrected, dxo, No, Np, asint):
    """
    Convert real-space scan positions to object-array pixel indices.

    The returned positions correspond to the upper-left corner of the
    object patch illuminated by the probe.

    Args:
        encoder_corrected (np.ndarray):
            Corrected scan positions in meters, typically with shape
            ``(numFrames, 2)``.

        dxo (float):
            Object-plane pixel size in meters.

        No (int):
            Number of pixels along one dimension of the object array.

        Np (int):
            Number of pixels along one dimension of the probe array.

        asint (bool):
            If True, return positions as integer pixel indices.

    Returns:
        np.ndarray:
            Pixel coordinates of the upper-left corner of each object patch.
    """
    positions = np.round(
        encoder_corrected / dxo
    )  # encoder is in m, positions0 and positions are in pixels
    positions = positions + No // 2 - Np // 2
    if asint:
        positions = positions.astype(int)
    return positions


class Reconstruction(object):
    """
    Store and manage the mutable state of a PtyLab reconstruction.

    The reconstruction state is initialized from an ``ExperimentalData`` object
    and a ``Params`` instance. Experimental quantities that may change during
    reconstruction are copied from ``ExperimentalData``, while derived sampling,
    coordinate grids, scan positions, object/probe settings, and reconstruction
    state are maintained by this class.

    Args:
        data (ExperimentalData):
            Experimental data and acquisition geometry used to initialize the
            reconstruction.

        params (Params):
            Reconstruction parameters and algorithm settings.

    Attributes:
        wavelength (float or np.ndarray):
            Illumination wavelength in meters.

        dxd (float):
            Detector pixel size in meters.

        zo (float):
            Propagation distance used by the reconstruction. For CPM, this is the
            sample-to-detector distance. Updating ``zo`` also updates ``dxp``.

        dxp (float):
            Probe-plane pixel size in meters. For CPM, it is derived from the
            wavelength, propagation distance, and detector field of view. For FPM,
            it is derived from the detector pixel size and microscope magnification.

        theta (float or None):
            Angular geometry parameter used for CPM, when provided by the
            experimental dataset.

        spectralDensity (np.ndarray or None):
            Spectral weight used for polychromatic CPM reconstruction.

        entrancePupilDiameter (float or None):
            Effective probe or pupil diameter associated with the reconstruction.

        zled (float):
            LED-to-sample distance in meters for FPM.

        NA (float or None):
            Numerical aperture used for FPM.

        encoder_corrected (np.ndarray):
            Measurement coordinates currently used by the reconstruction,
            including any applied position corrections.

        positions0 (np.ndarray):
            Reconstruction positions at initialization, in pixel coordinates.

        nlambda (int):
            Number of reconstructed wavelength modes.

        nosm (int):
            Number of object modes.

        npsm (int):
            Number of probe modes.

        nslice (int):
            Number of object slices.

        No (int):
            Number of pixels along one dimension of the reconstructed object.

        initialObject (str):
            Object initialization method.

        initialProbe (str):
            Probe initialization method.

        object (np.ndarray):
            Current complex-valued reconstructed object.

        probe (np.ndarray):
            Current complex-valued reconstructed probe.

        error (np.ndarray):
            Reconstruction error history. Available after reconstruction has
            produced an error metric.

        purityProbe (float):
            Current probe-purity value used by reconstruction diagnostics.

        purityObject (float):
            Current object-purity value used by reconstruction diagnostics.

        Nd (int):
            Number of detector pixels along one dimension.

        xd (np.ndarray):
            One-dimensional detector-plane coordinates in meters.

        Xd (np.ndarray):
            Two-dimensional detector-plane x-coordinate grid in meters.

        Yd (np.ndarray):
            Two-dimensional detector-plane y-coordinate grid in meters.

        Ld (float):
            Physical width of the detector grid in meters.

        Np (int):
            Number of pixels along one dimension of the probe grid.

        xp (np.ndarray):
            One-dimensional probe-plane coordinates in meters.

        Xp (np.ndarray):
            Two-dimensional probe-plane x-coordinate grid in meters.

        Yp (np.ndarray):
            Two-dimensional probe-plane y-coordinate grid in meters.

        Lp (float):
            Physical field of view of the probe grid in meters.

        dxo (float):
            Object-grid pixel size in meters. In the current implementation this
            is equal to ``dxp``. For the real-space FPM object sampling, use
            ``dxo_fpm``.

        xo (np.ndarray):
            One-dimensional object-grid coordinates in meters.

        Xo (np.ndarray):
            Two-dimensional object-grid x-coordinate array in meters.

        Yo (np.ndarray):
            Two-dimensional object-grid y-coordinate array in meters.

        Lo (float):
            Physical field of view associated with the object grid.

        dxo_fpm (float):
            Real-space object pixel size for FPM in meters.

        Lo_fpm (float):
            Real-space field of view of the FPM object in meters.

        dfp (float):
            Spatial-frequency sampling of the FPM probe grid in inverse meters.

        positions (np.ndarray):
            Reconstruction positions in pixel coordinates. Positions are given in
            row-column order and refer to the upper-left corner of each
            reconstructed patch. For CPM they refer to real-space object patches;
            for FPM they refer to patches in the high-resolution Fourier-space
            representation.

        NAd (float):
            Effective detection numerical aperture.

        DoF (float):
            Estimated depth of field in meters.

        TV (float):
            Total-variation metric of the current reconstructed object.

    Notes:
        ``Reconstruction`` contains quantities that may evolve during an
        iterative reconstruction.

        Several geometric and sampling quantities are exposed as properties and
        are derived from the experimental data and current reconstruction state.
    """

    _Nd = None

    # Note: zo, the sample-detector distance, is always read.
    listOfReconstructionPropertiesCPM = [
        "wavelength",
        # 'zo',
        "dxd",
        "theta",
        "spectralDensity",
        "entrancePupilDiameter",
    ]
    listOfReconstructionPropertiesFPM = [
        "wavelength",
        # 'zo',
        "dxd",
        "zled",
        "NA",
    ]

    def __init__(self, data: ExperimentalData, params: Params):
        """
        Initialize the reconstruction state from experimental data and parameters.

        Args:
            data (ExperimentalData):
                Experimental data and acquisition geometry.

            params (Params):
                Reconstruction parameters and algorithm settings.
        """
        self.zMomentum = 0
        self.wavelength = None
        self._zo = None
        self.dxd = None
        self.theta = None

        # positions including possible misalignment correction
        self.encoder_corrected = None

        self.logger = logging.getLogger("Reconstruction")
        self.data = data
        self.params = params
        self.copyAttributesFromExperiment(data)
        self.computeParameters()
        self.initializeSettings()

        # list of the fields that have to be transfered back and forth from the GPU
        self.possible_GPU_fields = [
            "probe",
            "object",
            "probeBuffer",
            "objectBuffer",
            "probeMomentum",
            "objectMomentum",
            "detectorError",
            "background",
            "reference",
            "intensity_mask",
            # multislice (e3PIE) transfer function, built on the host in
            # e3PIE.initializeReconstructionParams and used inside the position loop
            "H",
        ]

    # @property
    # def probe(self):
    #     # convenience function. Updates the temporary probe. Nothing in probe is updated
    #     # return self._probe
    #     return self.probe_storage.get_temporary()#_probe_storage.get(None)
    #
    # @probe.setter
    # def probe(self, new_probe):
    #     # ignore this for now
    #     # self._probe = new_probe
    #     # self.probe_storage.set_temporary(new_probe)

    def copyAttributesFromExperiment(self, data: ExperimentalData):
        """
        Copy reconstruction-relevant attributes from the experimental data.

        The attributes copied depend on the selected operation mode. The propagation distance
        and corrected measurement positions are handled separately.

        Args:
            data (ExperimentalData):
                Experimental data object from which reconstruction parameters and
                measurement positions are copied.

        Notes:
            For CPM, ``zo`` is assigned after the other geometry parameters because
            setting ``zo`` also updates the probe-plane sampling ``dxp``.

            ``encoder_corrected`` is initialized from ``data.encoder`` only if it
            has not already been set, preserving any existing position corrections.
        """
        self.logger.debug("Copying attributes from Experimental Data")
        if self.data.operationMode == "CPM":
            listOfReconstructionProperties = self.listOfReconstructionPropertiesCPM
        elif self.data.operationMode == "FPM":
            listOfReconstructionProperties = self.listOfReconstructionPropertiesFPM
        for key in listOfReconstructionProperties:
            self.logger.info("Copying attribute %s", key)
            # setattr(self, key, copy(np.array(getattr(data, key))))
            setattr(self, key, copy(getattr(data, key)))

        # set the distance, this has to be last
        # In FPM the sample to detector distance is irrelevant
        # LED-to-sample distance is the more important factor that affects
        # wave propagation and illumination angle
        if self.data.operationMode == "CPM":
            self.zo = getattr(data, "zo")

        # set the original positions
        if self.encoder_corrected is None:
            self.encoder_corrected = data.encoder.copy()

    def reset_positioncorrection(self):
        """
        Reset corrected measurement positions to the original encoder positions.

        The current ``encoder_corrected`` values are replaced by a copy of
        ``ExperimentalData.encoder``, removing any position corrections applied
        during reconstruction.
        """
        self.encoder_corrected = self.data.encoder.copy()

    @property
    def zo(self):
        """
        Propagation distance used by the reconstruction.

        For CPM, this represents the sample-to-detector distance. Updating
        ``zo`` also updates the probe-plane pixel size ``dxp``.
        """
        return self._zo

    @zo.setter
    def zo(self, new_value):
        self._zo = new_value
        if self.data.operationMode == "CPM":
            self.logger.debug(f"Changing sample-detector distance to {new_value}")
            self.dxp = self.wavelength * self._zo / self.Ld
        elif self.data.operationMode == "FPM":
             self.logger.debug(f"Changing illumination-to-sample distance to {new_value}")
             self.zled = self._zo
             
    def computeParameters(self):
        """
        Compute reconstruction geometry and mode-dependent default parameters.

        For CPM, missing probe and spectral parameters are initialized from the current reconstruction geometry. 
        For FPM, the sample-plane sampling and pupil geometry are derived from the microscope magnification and numerical
        aperture.

        The object-array size is determined from the range of reconstruction
        positions with additional space for the probe.

        Notes:
            This method may update both the ``Reconstruction`` instance and its
            associated ``ExperimentalData`` object.

            If ``No`` has not been defined, a temporary value is assigned first so
            that pixel positions can be evaluated before the final object size is
            determined.
        """

        if self.data.operationMode == "CPM":
            # CPM dxp (depending on the propagatorType, if none given, assum Fraunhofer/Fresnel)
            # self.dxp = self.wavelength * self._zo / self.Ld
            # if entrancePupilDiameter is not provided in the hdf5 file, set it to be one third of the probe FoV.
            if self.data.entrancePupilDiameter is None:
                self.data.entrancePupilDiameter = self.Lp / 3
            # if spectralDensity is not provided in the hdf5 file, set it to be a 1d array of the wavelength
            if isinstance(self.spectralDensity, type(None)):
                # this is a confusing name, it should be the wavelengths, not the intensity of the different
                # wavelengths
                self.spectralDensity = np.atleast_1d(self.wavelength)

        elif self.data.operationMode == "FPM":
            # FPM dxp (different from CPM due to lens-based systems)
            self.dxp = self.dxd / self.data.magnification
            # the propagation distance that is meaningful in this context is the
            # illumination to sample distance for LED array based microscopes
            self.zo = self.zled
            # if NA is not provided in the hdf5 file, set Fourier pupil entrance diameter it to be half of the Fourier space FoV.
            # then estimate the NA from the pupil diameter in the Fourier plane
            if isinstance(self.NA, type(None)):
                self.data.entrancePupilDiameter = self.Lp / 2
                self.NA = (
                    self.data.entrancePupilDiameter
                    * self.wavelength
                    / (2 * self.dxp**2 * self.Np)
                )
            else:
                # compute the pupil radius in the Fourier plane
                self.data.entrancePupilDiameter = (
                    2 * self.dxp**2 * self.Np * self.NA / self.wavelength
                )

        # set object pixel numbers
        if not hasattr(self, 'No'):
            self.No = (
                self.Np * 2**2
            )  # unimportant but leave it here as it's required for self.positions
            # we need space for the probe as well, on both sides that would be half the probe
            range_pixels = np.max(self.positions, axis=0) - np.min(self.positions, axis=0)
            # print(range_pixels)
            range_pixels = np.max(range_pixels) + self.Np * 2
            if range_pixels % 2 == 1:
                range_pixels += 1
            self.No = np.max([self.Np, range_pixels])

    def make_alignment_plot(self, saveit=False):
        """
        Create diagnostic plots for position alignment and autofocus history.

        The main plot compares the initial reconstruction positions with the
        current corrected positions. Additional plots are included when autofocus
        or total-variation history is available.

        Args:
            saveit (bool, optional):
                If True, save the diagnostic plots to
                ``plots/alignment.html``. Defaults to False.

        Returns:
            bokeh.layouts.LayoutDOM:
                Bokeh layout containing the available diagnostic plots.

        Notes:
            Position coordinates are derived from ``positions`` and ``positions0``.
            These quantities are expressed in reconstruction pixels.
        """
        t0 = time.time()
        p_new = self.positions.T
        p_old = self.positions0.T

        from bokeh.plotting import figure, output_file, save
        from bokeh.layouts import row

        from pathlib import Path

        if saveit:
            output = Path("plots/alignment.html")
            output.parent.mkdir(exist_ok=True)
            # set output to static HTML file

            output_file(filename=output, title="Static HTML file", mode="inline")

        # create a new plot with a specific size
        p = figure(
            sizing_mode="stretch_width",
            max_width=500,
            height=500,
            title=f'alignment (updated {time.strftime("%Y%h%d, %H:%M:%S")})',
        )
        p.match_aspect = True
        p.square(
            p_old[0], p_old[1], fill_color="yellow", size=5, legend_label="original"
        )
        # add a circle renderer for the new points
        p.circle(
            p_new[0], p_new[1], fill_color="red", size=5, legend_label="new"
        )

        p.xaxis.axis_label = "Position x [um]"
        p.yaxis.axis_label = "Position y [um]"

        p2 = None
        p3 = None
        p4 = None

        figsize = 500  # px

        if hasattr(self, "zHistory"):  # display the plot of the defocus
            p2 = figure(
                sizing_mode="stretch_width",
                max_width=figsize,
                height=figsize,
                title="focus history",
            )
            p2.circle(np.arange(len(self.zHistory)), np.array(self.zHistory) * 1e3)
            p2.xaxis.axis_label = "Iteration #"
            p2.yaxis.axis_label = "Position [mm]"
            # p = vplot(p, p2)

        if hasattr(self, "merit"):  # display the merit as well for defocii
            p3 = figure(
                sizing_mode="stretch_width",
                max_width=figsize,
                height=figsize,
                title="merit TV",
            )
            p3.circle(self.dz * 1e3, np.array(self.merit), legend_label="original")
            p3.square(
                -self.dz * 1e3,
                np.array(self.merit),
                legend_label="mirrored",
                color="red",
            )
            p3.xaxis.axis_label = "Defocus [mm]"
            p3.yaxis.axis_label = "Score [a.u.]"
            # p = vplot(p, p3)
        if hasattr(self, "TV_history"):
            if len(self.TV_history) >= 1:
                p4 = figure(
                    sizing_mode="stretch_width",
                    max_width=figsize,
                    height=figsize,
                    title="TV history",
                )
                p4.square(np.arange(len(self.TV_history)), self.TV_history)
                p4.xaxis.axis_label = "Iteration"
                p4.yaxis.axis_label = "TV score"
        # only add the plots that are available
        p_list = filter(lambda x: x is not None, [p, p2, p4, p3])
        p = row(*p_list)

        if saveit:
            save(
                p,
            )
        t1 = time.time()
        print(f"Alignment display took {t1-t0} secs")
        return p

    def initializeSettings(self):
        """
        Initialize the default reconstruction model and initialization settings.

        This method sets the number of wavelength, object, probe, and slice modes,
        initializes purity-related state ``purityProbe`` and ``purityObject``, stores the initial reconstruction
        positions, and selects the default object and probe initialization methods.

        The object and probe arrays are not created by this method. They are
        initialized later by ``initializeObjectProbe()``.

        Notes:
            The default reconstruction model uses one wavelength, one object mode,
            one probe mode, and one object slice.

            CPM initializes the object with ``"ones"`` and the probe with
            ``"circ"``, while FPM uses ``"upsampled"`` for the object and
            ``"circ"`` for the probe.
        """
        # Configure the reconstruction model dimensions.
        # These support multiple wavelengths, mixed object/probe states,
        # and multislice reconstruction.
        self.nlambda = 1
        self.nosm = 1
        self.npsm = 1
        self.nslice = 1

        # beam and object purity (# default initial value for plots.)
        self.purityProbe = 1.0
        self.purityObject = 1.0
        self.purityProbeHist = []

        self.positions0 = self.positions.copy()

        if self.data.operationMode == "FPM":
            self.initialObject = "upsampled"
            self.initialProbe = "circ"
        elif self.data.operationMode == "CPM":
            self.initialProbe = "circ"
            self.initialObject = "ones"
        else:
            self.initialProbe = "circ"
            self.initialObject = "ones"

    def prepare_probe(self, i):
        """
        Replace the current probe with a selected TSVD probe estimate.

        This method is intended for OPRP implementations and must be overridden
        by a reconstruction class that provides the corresponding probe estimates.

        Args:
            i (int):
                Index of the TSVD probe estimate to use.

        Raises:
            NotImplementedError:
                Always raised by the base ``Reconstruction`` implementation.
        """
        raise NotImplementedError()

    def initializeObjectProbe(self, force=True):
        """
        Initialize the object and probe used for reconstruction.

        Initial object and probe estimates are generated using
        ``initializeObject()`` and ``initializeProbe()``, then copied to
        ``self.object`` and ``self.probe`` as the mutable reconstruction state.

        Args:
            force (bool, optional):
                Forwarded to the object and probe initialization methods.
                Defaults to True.
        """ 
        # initialize object and probe
        self.initializeObject(force=force)
        self.initializeProbe(force=force)

        # set object and probe objects
        self.object = self.initialGuessObject.copy()
        self.probe = self.initialGuessProbe.copy()

    def initializeObject(self, type_of_init=None, force=True):
        """
        Initialize the object estimate used for reconstruction.

        The object shape is determined from the configured wavelength, object-mode,
        slice, and spatial dimensions. The initial object is either generated using
        the selected initialization method or loaded from a previous reconstruction.

        Args:
            type_of_init (str, optional):
                Object initialization method. If provided, this overrides
                ``self.initialObject``. If None, the currently configured
                initialization method is used.

            force (bool, optional):
                Whether to force object initialization. Defaults to True.
                The current implementation does not support ``False``.

        Raises:
            NotImplementedError:
                If ``force`` is False.

        Notes:
            The initialized object has shape
            ``(nlambda, nosm, 1, nslice, No, No)`` and is stored as
            ``complex64`` when generated by ``initialProbeOrObject()``.
        """
        if not force:
            raise NotImplementedError()
        if type_of_init is not None:
            self.initialObject = type_of_init
        self.logger.info("Initial object set to %s", self.initialObject)
        self.shape_O = (
            self.nlambda,
            self.nosm,
            1,
            self.nslice,
            self.No,
            self.No,
        )
        if self.initialObject == 'recon':
            # Load the object from an existing reconstruction. Confusing filename, but it contains both object and probe.
            self.initialGuessObject = self.loadResults(self.initialProbe_filename, datatype='object')
        else:
            self.initialGuessObject = initialProbeOrObject(self.shape_O, self.initialObject, self, self.logger).astype(np.complex64)

        # self.initialGuessObject *= 1e-2

    @staticmethod
    def loadResults(fileName, datatype='probe'):
        '''
        Load an object or probe from a saved PtyLab reconstruction.

        Args:
            fileName (str or Path):
                Path to the reconstruction HDF5 file.

            datatype (str, optional):
                Name of the dataset to load, typically ``"probe"`` or
                ``"object"``. Defaults to ``"probe"``.

        Returns:
            np.ndarray:
                Copy of the requested reconstruction dataset.
        '''
        with h5py.File(fileName) as archive:
            data = np.copy(np.array(archive[datatype]))
        return data

    def initializeProbe(self, force=False):
        """
        Initialize the probe estimate used for reconstruction.

        The probe shape is determined from the configured wavelength, probe-mode,
        slice, and spatial dimensions. The initial probe is either generated using
        the selected initialization method or loaded from a previous reconstruction.

        Args:
            force (bool, optional):
                Whether to reset the existing initial probe before generating a new
                estimate. Defaults to False.

        Notes:
            The initialized probe has shape
            ``(nlambda, 1, npsm, nslice, Np, Np)`` and is stored as
            ``complex64`` when generated by ``initialProbeOrObject()``.

            If ``entrancePupilDiameter`` is not available, it is set to one third
            of the probe field of view before initialization.
        """
        if self.data.entrancePupilDiameter is None:
            # if it is not set, set it to something reasonable
            self.logger.warning(
                "entrancePupilDiameter not set. Setting to one third of the FoV of the probe."
            )
            self.data.entrancePupilDiameter = self.Lp / 3
        self.logger.info("Initial probe set to %s", self.initialProbe)
        self.shape_P = (
            self.nlambda,
            1,
            self.npsm,
            self.nslice,
            int(self.Np),
            int(self.Np),
        )

        if self.initialProbe == 'recon':
            self.initialGuessProbe = self.loadResults(self.initialProbe_filename, datatype='probe')
        else:
            if force:
                self.initialGuessProbe = None
            # if force:
            #     self.initialProbe = "circ"
            self.initialGuessProbe = initialProbeOrObject(
                self.shape_P, self.initialProbe, self
            ).astype(np.complex64)

    # initialize momentum, called in specific engines with momentum accelaration
    def initializeObjectMomentum(self):
        """Initialize the object momentum buffer with zeros."""
        self.objectMomentum = np.zeros_like(self.initialGuessObject)

    def initializeProbeMomentum(self):
        """Initialize the probe momentum buffer with zeros."""
        self.probeMomentum = np.zeros_like(self.initialGuessProbe)

    def load_object(self, filename):
        """
        Load an object from a previous reconstruction.

        The saved object is truncated to the dimensions required by the current
        reconstruction and assigned to ``self.object``.

        Args:
            filename (str or Path):
                Path to a PtyLab reconstruction HDF5 file containing an
                ``"object"`` dataset.

        Raises:
            RuntimeError:
                If the loaded object cannot be matched to ``self.shape_O``.

        Notes:
            ``shape_O`` must already be defined before calling this method.
        """
        with h5py.File(filename, "r") as archive:
            obj = np.array(archive["object"])
            obj = obj[
                : self.shape_O[0],
                : self.shape_O[1],
                : self.shape_O[2],
                : self.shape_O[3],
                : self.shape_O[4],
                : self.shape_O[5],
            ]
            if np.all(np.array(obj.shape) == np.array(self.shape_O)):
                self.object = obj
            else:
                raise RuntimeError(
                    f'Shape of saved object cannot be extended to shape of required object. File: {archive["object"].shape}. Need: {self.shape_O}'
                )

    def load_probe(self, filename, expand_npsm=False, center_phase=False):
        """
        Load a probe from a previous reconstruction.

        The saved probe is center-cropped to the current probe size and truncated
        to the wavelength, probe-mode, and slice dimensions required by the
        current reconstruction.

        Args:
            filename (str or Path):
                Path to a PtyLab reconstruction HDF5 file containing a
                ``"probe"`` dataset.

            expand_npsm (bool, optional):
                Reserved for probe-mode expansion. This argument is currently not
                used by the implementation. Defaults to False.

            center_phase (bool, optional):
                If True, center the probe propagation angle after loading.
                Defaults to False.

        Raises:
            RuntimeError:
                If the loaded probe cannot be matched to ``self.shape_P``.

        Notes:
            ``shape_P`` must already be defined before calling this method.

        """
        with h5py.File(filename, "r") as archive:
            probe = np.array(archive["probe"])
            N_probe_read = probe.shape[-1]
            # roughly extract the center
            ss = slice(np.clip(N_probe_read//2-self.Np//2, 0, None), np.clip(N_probe_read//2-self.Np//2+int(self.Np), 0, N_probe_read))
            probe = probe[
                : self.nlambda,
                :1,
                : self.npsm,
                : self.nslice,
                ss,ss
            ]
            if np.all(np.array(probe.shape) == np.array(self.shape_P)):
                self.probe = probe
            else:
                raise RuntimeError(
                    f'Shape of saved probe cannot be extended to shape of required probe. File: {archive["probe"].shape}. Need: {self.shape_P}'
                )
        if center_phase:
            self._center_probe_angle()

    def _center_probe_angle(self):
        """
        Remove the global propagation-angle offset from the probe.

        The offset is estimated from the first probe mode and corrected by
        applying a compensating phase factor.
        """
        from skimage.registration import phase_cross_correlation
        from scipy.ndimage import fourier_shift
        p0 = np.squeeze(self.probe)[0]
        shift = phase_cross_correlation(p0, 0 * p0 + 1, normalization=None, space='fourier')[0]
        phexp = np.fft.fftshift(fourier_shift(0 * p0 + 1j, -shift / 2))
        self.probe *= phexp

    def load(self, filename):
        """
        Load a previously saved reconstruction state.

        This method restores the reconstructed object and probe together with
        selected reconstruction metadata from a PtyLab HDF5 result file.

        Args:
            filename (str or Path):
                Path to a reconstruction HDF5 file produced by
                ``saveResults(type="all")``.

        Notes:
            The current implementation expects only CPM-style result fields and does
            not restore all saved reconstruction state, such as
            ``encoder_corrected``.

            Unlike ``load_object()`` and ``load_probe()``, this method does not
            adapt or validate the loaded object and probe shapes.
        """
        with h5py.File(filename, "r") as archive:

            self.probe = np.array(archive["probe"])
            self.object = np.array(archive["object"])
            self.error = np.array(archive["error"])
            self.wavelength = np.array(archive["wavelength"])
            self.dxp = np.array(archive["dxp"])
            self.purityProbe = np.array(archive["purityProbe"])
            self.purityObject = np.array(archive["purityObject"])
            self.zo = np.array(archive["zo"])
            if "theta" in archive.keys():
                self.theta = np.array(archive["theta"])

    def saveResults(self, fileName="recent", type="all", squeeze=False):
        """
        Save reconstruction results to an HDF5 file.

        Args:
            fileName (str or Path, optional):
                Output filename. Defaults to ``"recent"``.

            type (str, optional):
                Type of reconstruction data to save. Supported values are
                ``"all"``, ``"object"``, ``"probe"``, and ``"probe_stack"``.
                Defaults to ``"all"``.

            squeeze (bool, optional):
                If True, remove singleton dimensions when saving only the object
                or probe. This option does not affect ``type="all"``.
                Defaults to False.

        Raises:
            NotImplementedError:
                If an unsupported save type is requested.

        Notes:
            The datasets saved by ``type="all"`` depend on the operation mode.
            CPM and FPM reconstruction files currently contain different sets of
            reconstruction metadata.

        """

        allowed_save_types = ["all", "object", "probe", "probe_stack"]
        if type not in allowed_save_types:
            raise NotImplementedError(
                f"Only {allowed_save_types} are allowed keywords for type"
            )
        if not squeeze:
            squeezefun = lambda x: x
        else:
            squeezefun = np.squeeze
        if type == "all":
            if self.data.operationMode == "CPM":
                with h5py.File(fileName, "w") as hf:
                    hf.create_dataset("probe", data=self.probe, dtype="complex64")
                    hf.create_dataset("object", data=self.object, dtype="complex64")
                    hf.create_dataset("error", data=self.error, dtype="f")
                    hf.create_dataset("zo", data=self._zo, dtype="f")
                    hf.create_dataset("wavelength", data=self.wavelength, dtype="f")
                    hf.create_dataset("dxp", data=self.dxp, dtype="f")
                    hf.create_dataset("purityProbe", data=self.purityProbe, dtype="f")
                    hf.create_dataset("purityObject", data=self.purityObject, dtype="f")
                    hf.create_dataset('I object', data=abs(self.object), dtype='f')
                    hf.create_dataset('I probe', data=abs(self.probe), dtype='f')
                    hf.create_dataset('encoder_corrected', data=self.encoder_corrected)

                    if hasattr(self, "theta"):
                        if self.theta != None:
                            hf.create_dataset("theta", data=self.theta, dtype="f")

            if self.data.operationMode == "FPM":
                hf = h5py.File(fileName, "w")
                hf.create_dataset("probe", data=self.probe, dtype="complex64")
                hf.create_dataset("object", data=self.object, dtype="complex64")
                hf.create_dataset("error", data=self.error, dtype="f")
                hf.create_dataset("zled", data=self.zled, dtype="f")
                hf.create_dataset("wavelength", data=self.wavelength, dtype="f")
                hf.create_dataset("dxp", data=self.dxp, dtype="f")
        elif type == "probe":
            with h5py.File(fileName, "w") as hf:
                hf.create_dataset(
                    "probe", data=squeezefun(self.probe), dtype="complex64"
                )
        elif type == "object":
            with h5py.File(fileName, "w") as hf:
                hf.create_dataset(
                    "object", data=squeezefun(self.object), dtype="complex64"
                )
        elif type == "probe_stack":
            hf = h5py.File(fileName + '_probe_stack.hdf5', 'w')
            hf.create_dataset('probe_stack', data=self.probe_stack.get(), dtype='complex64')
        print("The reconstruction results (%s) have been saved" % type)

    # detector coordinates
    @property
    def Nd(self):
        return self.data.ptychogram.shape[1]

    @property
    def xd(self):
        """Detector coordinates 1D"""
        return np.linspace(-self.Nd / 2, self.Nd / 2, np.int(self.Nd)) * self.dxd

    @property
    def Xd(self):
        """Detector coordinates 2D"""
        Xd, Yd = np.meshgrid(self.xd, self.xd)
        return Xd

    @property
    def Yd(self):
        """Detector coordinates 2D"""
        Xd, Yd = np.meshgrid(self.xd, self.xd)
        return Yd

    @property
    def Ld(self):
        """Detector size in SI units."""
        return self.Nd * self.dxd

    # probe coordinates
    @property
    def Np(self):
        """Probe pixel numbers"""
        Np = self.Nd
        return Np

    @property
    def Lp(self):
        """probe size in SI units"""
        Lp = self.Np * self.dxp
        return Lp

    @property
    def xp(self):
        """Probe coordinates 1D"""
        try:
            return np.linspace(-self.Np / 2, self.Np / 2, int(self.Np)) * self.dxp
        except AttributeError as e:
            raise AttributeError(
                e, 'probe pixel number "Np" and/or probe sampling "dxp" not defined yet'
            )

    @property
    def Xp(self):
        """Probe coordinates 2D"""
        Xp, Yp = np.meshgrid(self.xp, self.xp)
        return Xp

    @property
    def Yp(self):
        """Probe coordinates 2D"""
        Xp, Yp = np.meshgrid(self.xp, self.xp)
        return Yp

    # Object coordinates
    @property
    def dxo(self):
        """object pixel size, always equal to probe pixel size."""
        dxo = self.dxp
        return dxo

    @property
    def Lo(self):
        """Field of view (entrance pupil plane)"""
        return self.No * self.dxo

    @property
    def dxo_fpm(self):
        """Real-space object pixel size for FPM.
        """
        return self.dxp * self.Np / self.No

    @property
    def Lo_fpm(self):
        """Real-space field of view of the FPM object, equal to that of the raw images."""
        return self.No * self.dxo_fpm

    @property
    def dfp(self):
        """Spatial-frequency pixel size of the probe grid, 1 / Lp."""
        return 1 / self.Lp

    @property
    def xo(self):
        """object coordinates 1D"""
        try:
            return np.linspace(-self.No / 2, self.No / 2, np.int(self.No)) * self.dxo
        except AttributeError as e:
            raise AttributeError(
                e, 'object pixel number "No" and/or pixel size "dxo" not defined yet'
            )

    @property
    def Xo(self):
        """Object coordinates 2D"""
        Xo, Yo = np.meshgrid(self.xo, self.xo)
        return Xo

    @property
    def Yo(self):
        """Object coordinates 2D"""
        Xo, Yo = np.meshgrid(self.xo, self.xo)
        return Yo

    # scan positions in pixel
    @property
    def positions(self):
        """
        Reconstruction positions in pixel coordinates.

        Positions are stored in row-column order and refer to the upper-left
        corner of the reconstructed patch associated with each measurement.

        For CPM, the positions identify patches in the real-space object array.
        For FPM, they identify patches in the high-resolution Fourier-space
        object representation.
        """
        if self.data.operationMode == "FPM":
            conv = -(1 / self.wavelength) * self.dxo * self.Np
            positions = np.round(
                conv
                * self.encoder_corrected
                / np.sqrt(
                    self.encoder_corrected[:, 0] ** 2
                    + self.encoder_corrected[:, 1] ** 2
                    + self.zled**2
                )[..., None]
            )

            try:
                positions = positions + self.No // 2 - self.Np // 2
            except:
                pass

            return positions.astype(int)
        else:
            return calculate_pixel_positions(
                self.encoder_corrected, self.dxo, self.No, self.Np, asint=True
            )

    # system property list
    @property
    def NAd(self):
        """Effective detection numerical aperture."""
        NAd = self.Ld / (2 * self.zo)
        return NAd

    @property
    def DoF(self):
        """Estimated depth of field in meters."""
        DoF = self.wavelength / self.NAd**2
        # self.Dof2 = 5.2 *self.dxp**2 /self.wavelength
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
        """
        Print a summary of the reconstruction parameters and derived quantities.
        The summary includes experimental-data dimensions and sampling, reconstruction-grid parameters, propagation geometry, and derived
        quantities such as detector numerical aperture and depth of field.

        The summary is also written to the reconstruction logger.

        Returns:
            str:
                Formatted reconstruction summary.
        """
        minmax_tv = ''
        try:
            minmax_tv = f'(min: {self.params.TV_autofocus_min_z*1e3}, max: {self.params.TV_autofocus_max_z*1e3}.)'
        except TypeError: # one of them is none
            pass
        info = f"""
        Experimental data:
        - Ptychogram shape: {self.data.ptychogram.shape}
        - Ptychogram size[px]: {self.data.Nd}
        - Ptychogram size: {self.data.Ld*1e3} mm
        - Pixel pitch: {self.data.dxd*1e6} um
        - Scan size: {1e3*(self.data.encoder.max(axis=0) - self.data.encoder.min(axis=0))} mm 
        
        Reconstruction:
        - number of pixels: {self.No}
        - Pixel pitch: {self.dxo*1e6} um
        - Field of view: {self.Lo*1e3} mm
        - Scan size in pixels: {self.positions.max(axis=0)- self.positions.min(axis=0)}
        - Propagation distance: {self.zo * 1e3} mm {minmax_tv}
        - Probe FoV: {self.Lp*1e3} mm
        
        Derived parameters:
        - NA detector: {self.NAd}
        - Depth of field: {self.DoF*1e6} um
        
        """
        self.logger.info(info)
        return info

    @property
    def quadraticPhase(self):
        """Deprecated property; quadratic phase is no longer cached."""
        raise NotImplementedError("Quadratic phase is no longer cached. ")

    @property
    def transferFunction(self):
        """Deprecated property; transfer function is no longer cached."""
        raise NotImplementedError("Transfer function is not longer cached")

    @property
    def Q1(self):
        """Deprecated property; Q1 is no longer available."""
        raise NotImplementedError("Q1 is no longer available")

    @property
    def Q2(self):
        """Deprecated property; Q2 is no longer available."""
        raise NotImplementedError("Q2 is no longer available")

    def TV_autofocus(self, params: Params, loop):

        """
        Perform one autofocus update by optimizing a propagated-field metric.

        The selected object or probe field is propagated over a range of axial
        positions around the current propagation distance. A focus metric is
        evaluated at each plane and used to compute a momentum-based update of
        ``zo``.

        Args:
            params (Params):
                Reconstruction parameters controlling autofocus range, metric,
                update frequency, momentum, and axial bounds.

            loop (int or None):
                Current reconstruction iteration. If provided, autofocus is only
                run according to ``TV_autofocus_run_every``.

        Returns:
            tuple:
                Normalized metric at the current plane, selected propagated fields,
                and autofocus score information. Returns ``(None, None, None)``
                when no autofocus update is required.

        Raises:
            NotImplementedError:
                If used with FPM or with an unsupported autofocus target.
        """
        start_time = time.time()

        if self.data.operationMode == "FPM":
            raise NotImplementedError(
                f"Not implemented/tested for FPM. Set params.TV_autofocus to False. Got {params.TV_autofocus}"
            )
        if not params.TV_autofocus:
            return None, None, None
        if loop is not None:
            if loop % params.TV_autofocus_run_every != 0:
                return None, None, None

        if params.l2reg:
            self.logger.warning(
                "Both TV_autofocus and L2reg are turned on. This usually leads to poor performance. Consider disabling l2reg if the probe collapses to focal points"
            )

        d = params.TV_autofocus_range_dof
        nplanes = params.TV_autofocus_nplanes
        dz = np.linspace(-1, 1, nplanes) * d * self.DoF

        if params.TV_autofocus_what == "object":
            field = self.object[self.nlambda // 2, 0, 0, self.nslice // 2, :, :]
        elif params.TV_autofocus_what == "probe":
            field = self.probe[self.nlambda // 2, 0, 0, self.nslice // 2, :, :]
        else:
            raise NotImplementedError(
                f"So far, only object and probe are valid options for params.T_autofocus_what. Got {params.TV_autofocus_what}"
            )

        ss = params.TV_autofocus_roi
        if isinstance(ss, list):
            # semi-smart way to set up an AOI.
            # if the coordinates are a list, expand the list for y and x
            ss = np.array(ss)
            if ss.ndim == 1:
                ss = np.repeat(ss[None], axis=0, repeats=2)

            N = field.shape[-1]
            sy, sx = [slice(int(s[0] * N), int(s[1] * N)) for s in ss]
            # make them the same size if they're not
            sy = slice(sy.start, sy.start + sx.stop - sx.start)
        else:
            sy, sx = ss, ss

        merit, OEs = metric_at(
            field,
            dz,
            self.dxo,  # same as dxp
            self.wavelength,
            (sy, sx),
            intensity_only=self.params.TV_autofocus_intensityonly,
            metric=self.params.TV_autofocus_metric,
            return_propagated=True,
        )
        # from here on we are looking at 11 data points, work on CPU
        # as it's much more convenient and faster
        feedback = np.sum(dz * merit) / np.sum(merit)

        scores = np.vstack([self.zo + dz, merit])

        self.zMomentum *= params.TV_autofocus_friction
        self.zMomentum += params.TV_autofocus_stepsize * feedback
        # now, clip it to the bounds
        delta_z = self.zo - np.clip(
            self.zo + self.zMomentum,
            self.params.TV_autofocus_min_z,
            self.params.TV_autofocus_max_z,
        )
        self.zo -= delta_z
        end_time = time.time()
        self.logger.info(
            f"TV autofocus took {end_time-start_time} seconds, and moved focus by {-delta_z*1e6} micron"
        )
        indices = [nplanes//2, np.argmax(merit)]
        OEs = OEs[indices]
        phexp = OEs.sum((-2,-1), keepdims=True).conj()
        phexp = phexp / abs(phexp)
        OEs *= phexp
        return merit[nplanes//2] / asNumpyArray(abs(self.object[..., sy, sx]).mean()), np.hstack(OEs), (scores, self.zo)

    def reset_TV_autofocus(self):
        """
        Reset the autofocus momentum.

        This clears the accumulated ``zMomentum`` used by TV autofocus and can be useful when the autofocus updates become excessively large.
        """
        self.zMomentum = 0

    @property
    def TV(self):
        """Total-variation metric of the current reconstructed object."""
        return TV(self.object, 1e-2)
