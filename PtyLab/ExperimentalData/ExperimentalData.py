import numpy as np
from PtyLab.utils.gpuUtils import (transfer_fields_to_cpu,
                                   transfer_fields_to_gpu)

try:
    import pyqtgraph as pg
except ImportError:
    print("Cannot use pyqtgraph")
# from pathlib import Path
import logging

import matplotlib.pyplot as plt
# import tables
from PtyLab.io import readHdf5
from PtyLab.utils.gpuUtils import (getArrayModule, transfer_fields_to_cpu,
                                   transfer_fields_to_gpu)
# from PtyLab.io import readExample
from PtyLab.utils.visualisation import setColorMap, show3Dslider


class ExperimentalData:
    """
    Store experimental data and geometry for a PtyLab reconstruction.

    The class defines the dataset fields required for conventional ptychography
    (CPM) and Fourier ptychography (FPM), loads them from an HDF5 dataset, and
    derives detector and dataset quantities used during reconstruction.

    Args:
        filename (str or Path, optional):
            Path to an experimental HDF5 dataset. If None, the object is initialized without
            loading a dataset.

        operationMode (str, optional):
            Ptychographic operation mode, either conventional ptychography
            (``"CPM"``) or Fourier ptychography (``"FPM"``).
            Defaults to ``"CPM"``.

    Attributes:
        operationMode (str):
            Selected ptychographic operation mode.

        ptychogram (np.ndarray):
            Stack of measured intensity images. Available after data have been loaded.

        wavelength (float):
            Illumination wavelength in meters.

        encoder (np.ndarray):
            Position or illumination-coordinate data associated with the measurements.

        dxd (float):
            Detector pixel size in meters.

        Nd (int):
            Number of detector pixels along one dimension.

        Ld (float):
            Physical detector width in meters.

        numFrames (int):
            Number of measured frames.

        zo (float):
            Sample-to-detector distance in meters. Available for CPM datasets.

        entrancePupilDiameter (float or None):
            Effective probe diameter used for probe initialization. Optional for CPM datasets, not used in FPM.

        spectralDensity (np.ndarray or None):
            Spectral information used for polychromatic reconstruction. Optional for CPM datasets, not used in FPM.

        theta (float or None):
            Sample tilt or incidence-angle parameter used for reflection-mode. Optional for CPM datasets, not used in FPM.

        emptyBeam (np.ndarray or None):
            Reference illumination or probe image. Optional for CPM datasets, not used in FPM.

        zled (float):
            LED-to-sample distance in meters. Available for FPM datasets.

        magnification (float):
            Microscope magnification. Available for FPM datasets.
        
        NA (float or None):
            Microscope numerical aperture optional for FPM. If not provided, it is estimated from the Fourier-space pupil diameter during reconstruction.
        
        energyAtPos (np.ndarray):
            Integrated intensity of each measurement frame.

        maxProbePower (float):
            Maximum integrated-amplitude scale derived from the ptychogram.

    Raises:
        ValueError:
            If ``operationMode`` is neither ``"CPM"`` nor ``"FPM"``.

    Notes:
        CPM and FPM require different acquisition parameters. For CPM, the
        required fields include ``zo`` (sample-to-detector distance), whereas
        FPM requires ``zled`` and ``magnification``.

        Geometric quantities are expected in SI units, with distances given in meters.
    """

    def __init__(self, filename=None, operationMode="CPM"):
        self.logger = logging.getLogger("ExperimentalData")
        self.logger.debug("Initializing ExperimentalData object")

        self.operationMode = (
            operationMode  # Select the data schema for CPM or FPM.
        )
        self._setFields()
        if filename is not None:
            self.loadData(filename)

        # Arrays that may need to be transferred between CPU and GPU.
        # Some fields are only present for specific reconstruction modes or engines.
        self.fields_to_transfer = [
            "emptyBeam",
            "ptychogram",
            "ptychogramDownsampled",
            "W",  # used in aPIE
        ]

    def _setFields(self):
        """
        Set the required and optional fields for ptyLab to work.
        ALL VALUES MUST BE IN METERS.
        """
        if self.operationMode == "CPM":
            self.requiredFields = [
                "ptychogram",  # measured intensity stack, shape: (numFrames, Nd, Nd)
                "wavelength",  # illumination wavelength [m]
                "encoder",  # lateral scan positions for each diffraction frame [m]
                "dxd",  # detector pixel size [m]
                "zo",  # sample-to-detector propagation distance [m]
            ]
            self.optionalFields = [
                "entrancePupilDiameter",  # effective probe diameter used for CPM initialization [m]
                "spectralDensity",  # CPM parameters: spectral weights for polychromatic reconstruction
                "theta",  # CPM parameters: sample tilt / incidence angle for reflection-mode ptychography [rad]
                "emptyBeam",  # image of the probe
            ]

        elif self.operationMode == "FPM":
            self.requiredFields = [
                "ptychogram",  # measured intensity stack, shape: (numFrames, Nd, Nd)
                "wavelength",  # illumination wavelength [m]
                "encoder",  # lateral scan positions for each diffraction frame [m]
                "dxd",  # detector pixel size [m]
                "zled",  # LED to sample distance [m]
                "magnification",  # magnification, used for FPM computations of dxp [m]
            ]
            self.optionalFields = [
                # entrance pupil diameter, defined in lens-based microscopes as the aperture diameter, reqquired for FPM
                # 'entrancePupilDiameter'
                "NA",  # numerical aperture of the microscope
            ]
        else:
            raise ValueError('operationMode is not properly set, choose "CPM" or "FPM"')

    def loadData(self, filename=None):
        """
        Load a ptychography dataset and initialize the corresponding experimental data.

        The dataset fields required for loading depend on the selected operation mode.
        Example-data aliases can be used instead of explicit file paths.

        Args:
            filename (str or Path):
                Path to the dataset to load. The following special aliases are supported:

                - ``"example:simulation_cpm"``: synthetic CPM example dataset.
                - ``"example:simulation_fpm"``: synthetic FPM example dataset.
                - ``"test:nodata"``: initialize a minimal stub dataset for testing.

        Notes:
            After loading, the dataset fields are added as attributes of the
            ``ExperimentalData`` instance.

            Derived quantities such as detector coordinates, detector size,
            number of frames, and probe-power estimates are initialized by
            ``_setData()``.

            The ptychogram orientation is applied after loading according to the
            orientation metadata stored in the dataset.
        """
        import os

        if str(filename) == "test:nodata":
            self.filename = filename
            # Set minimal stub data so the object is usable without a file
            self.ptychogram = np.zeros((1, 16, 16), dtype=np.float32)
            self.encoder = np.zeros((1, 2), dtype=np.float64)
            self.wavelength = 500e-9
            self.dxd = 6.5e-6
            self.zo = 0.1
            # optional fields
            self.entrancePupilDiameter = None
            self.spectralDensity = None
            self.theta = None
            self.emptyBeam = None
            self._setData()
            return

        if not os.path.exists(filename) and str(filename).startswith("example:"):
            self.filename = filename
            from PtyLab.io.readExample import examplePath

            self.filename = examplePath(
                filename
            )  # readExample(filename, python_order=True)
        else:
            self.filename = filename

        # Validate that all required dataset fields are present.
        readHdf5.checkDataFields(self.filename, self.requiredFields)
        # Load all required fields and any available optional fields.
        measurementDict = readHdf5.loadInputData(
            self.filename, self.requiredFields, self.optionalFields
        )
        # Expose the loaded dataset fields as ExperimentalData attributes.
        attributesToSet = measurementDict.keys()
        
        # self.logger.setLevel(logging.DEBUG)
        for a in attributesToSet:
            # make sure that property is not an attribtue
            attribute = str(a)
            if not isinstance(getattr(type(self), attribute, None), property):
                setattr(self, attribute, measurementDict[a])
            self.logger.debug("Setting %s", a)

        self._setData()
        # Apply the stored orientation last, since this operation modifies
        # the ptychogram array.
        self.setOrientation(readHdf5.getOrientation(self.filename))

    def reduce_positions(self, start, end):
        """
            Restrict the dataset to a contiguous subset of measurement positions.

        The ptychogram and the corresponding encoder positions are sliced along
        their first dimension using standard Python slicing semantics.

        Args:
            start (int):
                Index of the first measurement position to retain.

            end (int):
                Index at which to stop the selection. The measurement at this
                index is not included.

        Notes: 
            This method modifies ``self.ptychogram`` and ``self.encoder`` in place.
        """
        self.ptychogram = self.ptychogram[start: end]
        self.encoder = self.encoder[start: end]
        self._setData()

    def cropCenter(self, size):
        '''
        Crop each diffraction pattern to a centered square region.

        The ptychogram is cropped along its two detector dimensions while the
        number of measurement frames is preserved.

        Args:
            size (int):
                Number of detector pixels retained along each dimension of the
                cropped diffraction patterns.

        Raises:
            TypeError:
                If ``size`` is not an integer.

        Notes:
            This method modifies ``self.ptychogram`` in place.

            Derived detector quantities such as ``Nd`` and ``Ld`` are currently
            not recomputed by this method.
        '''
        if not isinstance(size, int):
            raise TypeError('Crop value is not valid. Int expected')

        x = self.ptychogram.shape[-1]
        startx = x // 2 - (size // 2)

        startx += 1

        self.ptychogram = self.ptychogram[..., startx: startx + size, startx: startx + size]
        self._setData()

    def binData(self, binning):
        '''
        Spatially bin each diffraction pattern by averaging neighboring pixels.

        Each ``binning × binning`` detector region is replaced by its mean value,
        reducing both detector dimensions by the specified binning factor.

        Args:
            binning (int):
                Integer binning factor applied along both detector dimensions.
                The detector dimensions must be divisible by this value.

        Notes:
            This method modifies ``self.ptychogram`` in place.

            The current implementation performs mean binning, so each output pixel
            contains the average intensity of the corresponding input pixel block.
        '''
        Ndp = self.ptychogram.shape[0]
        Ny = self.ptychogram.shape[1]
        Nx = self.ptychogram.shape[2]

        ptychogram_temp = np.copy(self.ptychogram)
        self.ptychogram = np.zeros((Ndp, Ny // binning, Nx // binning))

        # Bin each diffraction pattern independently.
        for i in range(Ndp):
            temp = ptychogram_temp[i]
            reshaped_temp = temp.reshape(Ny // binning, binning, Nx // binning, binning)
            temp_binning = reshaped_temp.mean(axis=(1, 3))
            self.ptychogram[i] = np.copy(temp_binning)

    def setOrientation(self, orientation, force_contiguous=True):
        """
        Apply the detector orientation specified by the ptypy convention.

        The orientation is applied to the last two dimensions of
        ``self.ptychogram`` using combinations of axis flips and transposition.

        Args:
            orientation (int or None):
                Orientation code following the ptypy convention:

                - ``0``: no transformation.
                - ``1``: flip detector columns.
                - ``2``: flip detector rows.
                - ``3``: flip detector rows and columns.
                - ``4``: transpose the detector dimensions.
                - ``5``: transpose, then flip columns.
                - ``6``: transpose, then flip rows.
                - ``7``: transpose, then flip rows and columns.

                If None, no transformation is applied.

            force_contiguous (bool, optional):
                If True, convert the transformed ptychogram to a contiguous
                NumPy array. Defaults to True.

        Raises:
            TypeError:
                If ``orientation`` is not an integer or None.

            ValueError:
                If ``orientation`` is not one of the supported values from
                0 to 7.

        Notes:
            This method modifies ``self.ptychogram`` in place.
        """
        if orientation is None:  # do not update.
            return
        if not isinstance(orientation, int):
            raise TypeError("Orientation value is not valid.")
        if orientation == 0:  # don't change anything
            return
        if orientation == 1:
            # Invert column
            self.ptychogram = np.flip(self.ptychogram, axis=-1)
        elif orientation == 2:
            # Invert rows
            self.ptychogram = np.flip(self.ptychogram, axis=-2)
        elif orientation == 3:
            # invert columns and rows
            self.ptychogram = np.flip(self.ptychogram, axis=-1)
            self.ptychogram = np.flip(self.ptychogram, axis=-2)
        elif orientation == 4:
            # Transpose
            self.ptychogram = np.transpose(self.ptychogram, (0, 2, 1))
        elif orientation == 5:
            self.ptychogram = np.transpose(self.ptychogram, (0, 2, 1))
            self.ptychogram = np.flip(self.ptychogram, axis=-1)
        elif orientation == 6:
            self.ptychogram = np.transpose(self.ptychogram, (0, 2, 1))
            self.ptychogram = np.flip(self.ptychogram, axis=-2)
        elif orientation == 7:
            self.ptychogram = np.transpose(self.ptychogram, (0, 2, 1))
            self.ptychogram = np.flip(self.ptychogram, axis=-1)
            self.ptychogram = np.flip(self.ptychogram, axis=-2)

        else:
            raise ValueError(f"Orientation {orientation} is not implemented")
        if force_contiguous:
            # this almost always makes sense. It makes it easier to read chunks
            self.ptychogram = np.ascontiguousarray(self.ptychogram)

    def _setData(self):
        """
        Update detector geometry and dataset-derived quantities.

        This method derives detector coordinates, detector size, frame count,
        per-frame integrated intensity, and the maximum probe-amplitude scale
        from the current ``ptychogram`` and detector pixel size.

        Notes:
            This method should be called whenever the ptychogram shape or detector
            sampling changes.
        """
        # Set the detector coordinates
        self.Nd = self.ptychogram.shape[-1]
        # Detector coordinates 1D
        self.xd = np.linspace(-self.Nd / 2, self.Nd / 2, int(self.Nd)) * self.dxd
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
        Display the measured ptychogram stack on a logarithmic intensity scale.

        The diffraction patterns are clipped to non-negative values, converted to
        ``log10(I + 1)`` for visualization, and displayed with an interactive
        slider over the measurement frames.

        Notes:
            This method is intended for data inspection only and does not modify
            ``self.ptychogram``.
        """
        xp = getArrayModule(self.ptychogram)
        print(f"Min max ptychogram: {np.min(self.ptychogram)}, {self.ptychogram.max()}")
        log_ptychogram = xp.log10(
            xp.swapaxes(np.clip(self.ptychogram.astype(np.float32), 0, None), 1, 2) + 1
        )
        print(f"Min max ptychogram: {np.min(log_ptychogram)}, {log_ptychogram.max()}")
        show3Dslider(log_ptychogram)

    def _move_data_to_cpu(self):
        """Move all required data to the CPU"""
        transfer_fields_to_cpu(self, self.fields_to_transfer, self.logger)

    def _move_data_to_gpu(self):
        """Move all required fata to the GPU"""
        transfer_fields_to_gpu(self, self.fields_to_transfer, self.logger)


    def relative_intensity(self, index):
        """
        Return the normalized mean intensity of a selected ptychogram frame.

        The mean detector intensity is computed for each frame and
        normalized.

        Args:
            index (int):
                Index of the measurement frame.

        Returns:
            float:
                Normalized mean intensity of the selected frame.

        Notes:
            The normalization is defined as

            ``I_rel = I_mean / (mean(I_mean) + 2 * std(I_mean))``.

            The normalized intensities are cached after the first call.
        """
        if not hasattr(self, '_relative_intensity'):
            self._relative_intensity = self.ptychogram.mean((-2,-1))
            self._relative_intensity /= (self._relative_intensity.mean() + 2*self._relative_intensity.std())
        return self._relative_intensity[index]
