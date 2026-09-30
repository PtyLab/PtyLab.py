import logging
import os

import numpy as np
from matplotlib import pyplot as plt

from PtyLab import Operators
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.Regularizers import grad_TV

# PtyLab imports
from PtyLab.utils.gpuUtils import (
    asNumpyArray,
    getArrayModule,
    isGpuArray,
    transfer_fields_to_cpu,
    transfer_fields_to_gpu,
)
from PtyLab.utils.utils import circ, fft2c, orthogonalizeModes

try:
    import cupy as cp
    from cupyx.scipy.ndimage import fourier_gaussian as fourier_gaussian_gpu
except ImportError:
    from scipy.ndimage import fourier_gaussian as fourier_gaussian_gpu

    cp = None
from scipy.ndimage import fourier_gaussian as fourier_gaussian_cpu


def smooth_amplitude(
    field: np.ndarray, width: float, aleph: float, amplitude_only: bool = True
):
    """
    Smooth a complex field using a Gaussian filter.

    By default, the Gaussian smoothing is applied only to the field
    amplitude while preserving the original phase. If ``amplitude_only``
    is False, the complex field itself is smoothed.

    Args:
        field (np.ndarray):
            Complex-valued field to smooth. NumPy and CuPy arrays are
            supported.

        width (float):
            Width of the Gaussian smoothing filter applied along the two
            spatial dimensions.

        aleph (float):
            Weight of the smoothed field in the returned result. A value of
            zero leaves the field unchanged, while a value of one returns
            the fully smoothed field.

        amplitude_only (bool, optional):
            If True, smooth only the amplitude and preserve the original
            phase. If False, smooth the full complex field.
            Defaults to True.

    Returns:
        np.ndarray:
            Smoothed field with the same shape as the input.

    """
    xp = getArrayModule(field)
    smooth_fun = isGpuArray(field) and fourier_gaussian_gpu or fourier_gaussian_cpu
    gimmel = 1e-5
    if amplitude_only:
        ph_field = field / (xp.abs(field) + gimmel)
        A_field = abs(field)
    else:
        ph_field = 1
        A_field = field
    F_field = xp.fft.fft2(A_field)
    for ax in [-2, -1]:
        F_field = smooth_fun(F_field, width, axis=ax)
    field_smooth = xp.fft.ifft2(F_field)

    if amplitude_only:
        field_smooth = abs(field_smooth) * ph_field
    return aleph * field_smooth + (1 - aleph) * field


class BaseEngine(object):
    """
    Base class providing shared functionality for PtyLab reconstruction engines.

    ``BaseEngine`` coordinates operations that are common to different
    reconstruction algorithms, including wave propagation, intensity
    projection, error evaluation, reconstruction constraints, position
    correction, monitoring, and CPU/GPU data transfer.

    The engine operates on shared ``Reconstruction``, ``ExperimentalData``,
    ``Params``, and ``Monitor`` objects. These objects are referenced directly
    rather than copied, so updates performed by the engine are reflected in
    the associated reconstruction state.

    Specific reconstruction engines inherit from this class and implement
    their algorithm-specific reconstruction and update steps. ``BaseEngine``
    is therefore generally not intended to be instantiated directly.

    Many methods in `BaseEngine` implement functionality controlled by user-facing
    `Params` switches. The `Params` API documentation describes when and why these
    options are used, while the corresponding `BaseEngine` methods document their
    implementation details.

    Args:
        reconstruction (Reconstruction):
            Mutable reconstruction state containing the current object, probe,
            geometry, and reconstruction results.

        experimentalData (ExperimentalData):
            Experimental diffraction data and acquisition geometry.

        params (Params):
            Reconstruction parameters controlling propagation, constraints,
            correction methods, GPU usage, and other algorithm settings.

        monitor (Monitor):
            Monitor used to visualize and report reconstruction progress.

    Attributes:

        betaObject (float):
            Object update weight used by shared object-update routines.
            Initialized to ``0.25``.

    """

    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        # Keep references to the shared PtyLab components; no data are copied.
        self.betaObject = 0.25
        self.reconstruction: Reconstruction = reconstruction
        self.experimentalData = experimentalData
        self.params = params
        self.monitor = monitor
        self.monitor.reconstruction = reconstruction # share reconstruction state with monitor

        # datalogger
        self.logger = logging.getLogger("BaseEngine")

    def _prepareReconstruction(self):
        """
        Prepare the reconstruction state before iterative updates.

        This method validates user-configurable settings and initializes shared
        engine state, including FFT conventions, probe constraints, error arrays,
        monitoring regions, position-correction parameters, and CPU/GPU data
        placement.

        GPU setup is performed last because the preceding initialization steps
        operate on host-side reconstruction data.
        """
        # check miscellaneous quantities specific for certain Engines
        self._checkMISC()
        self._checkFFT()
        # self._initializeQuadraticPhase()
        self._initialProbePowerCorrection()
        self._probeWindow()
        self._initializeErrors()
        self._setObjectProbeROI()
        self._showInitialGuesses()
        self._initializePCParameters()
        self._checkGPU()  # checkGPU needs to be the last

        # self.reconstruction.probe_storage.push(self.reconstruction.probe, 0, self.experimentalData.ptychogram.shape[0])

    def _setCPSC(self):
        """
        Configure the constrained-pixel-sum (CPSC) model.

        CPSC reconstructs the diffraction field on a finer computational detector
        grid while constraining the summed intensity within each group of subpixels
        to match the corresponding measured detector pixel. The finer grid preserves
        the detector field of view and therefore the real-space sampling, while
        increasing the reconstruction field of view.
        """

        # save the measured ptychogram into ptychograpmDownsampled
        self.experimentalData.ptychogramDownsampled = self.experimentalData.ptychogram

        # pad the probe
        padNum_before = (
            (self.params.CPSCupsamplingFactor - 1) * self.reconstruction.Np // 2
        )
        padNum_after = (
            self.params.CPSCupsamplingFactor - 1
        ) * self.reconstruction.Np - padNum_before
        self.reconstruction.probe = np.pad(
            self.reconstruction.probe,
            (
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (padNum_before, padNum_after),
                (padNum_before, padNum_after),
            ),
        )

        # pad the momentums, buffers
        if hasattr(self.reconstruction, "probeBuffer"):
            self.reconstruction.probeBuffer = self.reconstruction.probe.copy()
        if hasattr(self.reconstruction, "probeMomentum"):
            self.reconstruction.probeMomentum = np.pad(
                self.reconstruction.probeMomentum,
                (
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (padNum_before, padNum_after),
                    (padNum_before, padNum_after),
                ),
            )

        # update coordinates (only need to update the Nd and dxd, the rest updates automatically)
        self.reconstruction.Nd = (
            self.experimentalData.ptychogramDownsampled.shape[-1]
            * self.params.CPSCupsamplingFactor
        )
        self.reconstruction.dxd = (
            self.reconstruction.dxd / self.params.CPSCupsamplingFactor
        )

        self.logger.info("CPSCswitch is on, coordinates(dxd,dxp,dxo) have been updated")

    def update_data(self, experimentalData, reconstruction=None):
        """
        Update the data objects referenced by the engine.

        Args:
            experimentalData (ExperimentalData):
                Experimental dataset to use for subsequent reconstruction steps.

            reconstruction (Reconstruction, optional):
                Replacement reconstruction state. If None, the current
                reconstruction object is retained.
        """
        self.experimentalData = experimentalData
        if reconstruction is not None:
            self.reconstruction = reconstruction

    def _initializePCParameters(self):
        """
        Initialize parameters and state for pcPIE correction.

        When position correction is enabled, this method initializes the
        feedback and momentum factors, the per-position correction vectors,
        the candidate pixel shifts used for local correlation searches, and
        the iteration threshold for starting position updates.
        """
        if self.params.positionCorrectionSwitch:
            # additional pcPIE parameters as they appear in Matlab
            self.daleth = 0.5  # feedback
            self.beth = 0.9  # friction
            self.adaptStep = 1  # adaptive step size
            self.D = np.zeros(
                (self.experimentalData.numFrames, 2)
            )  # position search direction
            # predefine shifts
            rmax = 2
            dy, dx = np.mgrid[-rmax : rmax + 1, -rmax : rmax + 1]

            # self.rowShifts = dy.flatten()#np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
            self.rowShifts = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
            # self.colShifts = dx.flatten()#np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
            self.colShifts = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
            self.startAtIteration = 1
            self.meanEncoder00 = np.mean(self.experimentalData.encoder[:, 0]).copy()
            self.meanEncoder01 = np.mean(self.experimentalData.encoder[:, 1]).copy()

    def _initializeErrors(self):
        """
        Initialize reconstruction error storage.

        Depending on ``saveMemory``, detector-plane errors are either stored for
        every scan position or reduced immediately to per-position error values.
        The method also initializes the per-position error array and the global
        reconstruction error history.
        """
        # initialize detector error matrices
        if self.params.saveMemory:
            self.reconstruction.detectorError = 0
        else:
            if not hasattr(self.reconstruction, "detectorError"):
                self.reconstruction.detectorError = np.zeros(
                    (
                        self.experimentalData.numFrames,
                        self.reconstruction.Nd,
                        self.reconstruction.Nd,
                    )
                )
        # initialize energy at each scan position
        if not hasattr(self.reconstruction, "errorAtPos"):
            self.reconstruction.errorAtPos = np.zeros(
                (self.experimentalData.numFrames, 1), dtype=np.float32
            )
        # initialize final error
        if not hasattr(self.reconstruction, "error"):
            self.reconstruction.error = []

    def _initialProbePowerCorrection(self):
        r"""
        Scale the initial probe to the measured diffraction power.

        When `probePowerCorrectionSwitch` is enabled, rescale the complex probe
        $P$ using the measured amplitude scale $P_{\max}$:

        $$
        P_{\mathrm{new}} = \frac{P}{\sqrt{\sum |P|^2}} P_{\max}
        $$

        The sum in the normalization includes every element of the probe array.
        The amplitude scale is derived from the brightest measured diffraction
        pattern:

        $$
        P_{\max} = \sqrt{\max_j \sum_{x,y} I_j(x,y)}
        $$

        Here, $I_j(x,y)$ is the measured intensity at detector pixel $(x,y)$
        in frame $j$. The correction preserves the probe shape and phase while
        placing its total power on the scale of the experimental data, reducing
        large amplitude corrections at the beginning of reconstruction.
        
        """
        if self.params.probePowerCorrectionSwitch:
            self.reconstruction.probe = (
                self.reconstruction.probe
                / np.sqrt(
                    np.sum(self.reconstruction.probe * self.reconstruction.probe.conj())
                )
                * self.experimentalData.maxProbePower
            )

    def _probeWindow(self):
        r"""
        Create the spatial window used by probe-boundary constraints.

        For `absorbingProbeBoundary`, a smooth super-Gaussian window is defined as

        $$
        W(x,y) = \exp\left[-\left(\frac{x^2+y^2}{2\sigma^2}\right)^{10}\right]
        $$

        where

        $$
        \sigma = \frac{3}{4}\frac{N_p dx_p}{2.355}
        $$

        For `probeBoundary`, a circular support window is generated from the
        entrance pupil diameter.

        The generated window is stored in `probeWindow` and applied later by
        `applyConstraints()`.
        """
        # absorbing probe boundary: filter probe with super-gaussian window function
        if not self.params.saveMemory or self.params.absorbingProbeBoundary:
            self.probeWindow = np.exp(
                -(
                    (
                        (self.reconstruction.Xp**2 + self.reconstruction.Yp**2)
                        / (
                            2
                            * (
                                3
                                / 4
                                * self.reconstruction.Np
                                * self.reconstruction.dxp
                                / 2.355
                            )
                            ** 2
                        )
                    )
                    ** 10
                )
            )

        if self.params.probeBoundary:
            self.probeWindow = circ(
                self.reconstruction.Xp,
                self.reconstruction.Yp,
                self.experimentalData.entrancePupilDiameter
                + self.experimentalData.entrancePupilDiameter * 0.2,
            )

    def _setObjectProbeROI(self, update=False):
        """
        Set the object and probe regions of interest used for monitoring.

        The object ROI is derived from the scan-position extent and probe size,
        scaled by ``monitor.objectZoom``. The probe ROI is centered on the probe
        grid and derived from the entrance pupil diameter and
        ``monitor.probeZoom``.

        If the corresponding zoom value is ``"full"`` or ``None``, the complete
        object or probe is displayed.

        Args:
            update (bool, optional):
                If True, recompute existing ROIs. Otherwise, ROIs are only created
                when they are not already defined. Defaults to False.
        """
        if not hasattr(self.monitor, "objectROI") or update:
            if self.monitor.objectZoom == "full" or self.monitor.objectZoom is None:
                self.monitor.objectROI = [
                    slice(None, None, None),
                    slice(None, None, None),
                ]
            else:
                rx, ry = (
                    (
                        np.max(self.reconstruction.positions, axis=0)
                        - np.min(self.reconstruction.positions, axis=0)
                        + self.reconstruction.Np
                    )
                    / self.monitor.objectZoom
                ).astype(int)
                xc, yc = (
                    (
                        np.max(self.reconstruction.positions, axis=0)
                        + np.min(self.reconstruction.positions, axis=0)
                        + self.reconstruction.Np
                    )
                    / 2
                ).astype(int)

                # self.monitor.objectROI = [
                #     slice(
                #         max(0, yc - ry // 2), min(self.reconstruction.No, yc + ry // 2)
                #     ),
                #     slice(
                #         max(0, xc - rx // 2), min(self.reconstruction.No, xc + rx // 2)
                #     ),
                # ]
                self.monitor.objectROI = [
                    slice(
                        max(0, xc - rx // 2), min(self.reconstruction.No, xc + rx // 2)
                    ),
                    slice(
                        max(0, yc - ry // 2), min(self.reconstruction.No, yc + ry // 2)
                    ),
                ]

        if not hasattr(self.monitor, "probeROI") or update:
            if self.monitor.probeZoom == "full" or self.monitor.probeZoom is None:
                self.monitor.probeROI = [slice(None, None), slice(None, None)]
            else:
                r = int(
                    self.experimentalData.entrancePupilDiameter
                    / self.reconstruction.dxp
                    / self.monitor.probeZoom
                )
                self.monitor.probeROI = [
                    slice(
                        max(0, self.reconstruction.Np // 2 - r),
                        min(self.reconstruction.Np, self.reconstruction.Np // 2 + r),
                    ),
                    slice(
                        max(0, self.reconstruction.Np // 2 - r),
                        min(self.reconstruction.Np, self.reconstruction.Np // 2 + r),
                    ),
                ]

    def _showInitialGuesses(self):
        """
        Display the initial object and probe estimates in the reconstruction monitor.

        The object and probe are cropped to the monitoring regions defined by
        ``objectROI`` and ``probeROI`` before being passed to the monitor together
        with the current reconstruction error, propagation distance, mode purities,
        and scan positions.
        """
        self.monitor.initializeMonitors()
        objectEstimate = np.squeeze(
            self.reconstruction.object[
                ..., self.monitor.objectROI[0], self.monitor.objectROI[1]
            ]
        )
        probeEstimate = np.squeeze(
            self.reconstruction.probe[
                ..., self.monitor.probeROI[0], self.monitor.probeROI[1]
            ]
        )

        self.monitor.updateObjectProbeErrorMonitor(
            error=self.reconstruction.error,
            object_estimate=objectEstimate,
            probe_estimate=probeEstimate,
            zo=self.reconstruction.zo,
            purity_probe=self.reconstruction.purityProbe,
            purity_object=self.reconstruction.purityObject,
            encoder_positions=self.reconstruction.positions,
        )

        # self.monitor.updateObjectProbeErrorMonitor()

    def _checkMISC(self):
        """
        Initialize auxiliary reconstruction state and validate special settings.

        This method prepares additional variables required by selected intensity
        constraints or background reconstruction, checks incompatible parameter
        combinations, and initializes the constrained-pixel-sum configuration
        when enabled.

        Raises:
            ValueError:
                If incompatible reconstruction options are enabled or required
                parameters for a selected constraint are missing.
        """
        if self.params.backgroundModeSwitch:
            self.reconstruction.background = 1e-1 * np.ones(
                (self.reconstruction.Np, self.reconstruction.Np)
            )

        # preallocate intensity scaling vector
        if self.params.intensityConstraint == "fluctuation":
            self.intensityScaling = np.ones(self.experimentalData.numFrames)

        if self.params.intensityConstraint == "interferometric":
            self.reconstruction.reference = np.ones(
                self.reconstruction.probe[0, 0, 0, 0, ...].shape
            )

        # check if both probePoprobePowerCorrectionSwitch and modulusEnforcedProbeSwitch are on.
        # Since this can cause a contradiction, it raises an error
        if (
            self.params.probePowerCorrectionSwitch
            and self.params.modulusEnforcedProbeSwitch
        ):
            raise ValueError(
                "probePowerCorrectionSwitch and modulusEnforcedProbeSwitch "
                "can not simultaneously be switched on!"
            )

        if self.params.propagatorType == "ASP" and self.params.fftshiftSwitch:
            raise ValueError(
                "ASP propagatorType works only with fftshiftSwitch = False"
            )
        if self.params.propagatorType == "scaledASP" and self.params.fftshiftSwitch:
            raise ValueError(
                "scaledASP propagatorType works only with fftshiftSwitch = False"
            )

        if self.params.CPSCswitch:
            if not hasattr(self.experimentalData, "ptychogramDownsampled"):
                if self.params.CPSCupsamplingFactor == None:
                    raise ValueError(
                        "CPSCswitch is on, CPSCupsamplingFactor need to be set"
                    )
                else:
                    self._setCPSC()

    def _checkFFT(self):
        """
        Synchronize detector-domain arrays with the selected FFT convention.

        When ``fftshiftSwitch`` is enabled, detector-side quantities are shifted
        to the FFT-native ordering using ``ifftshift``. When the switch is
        disabled after a previous shift, the arrays are restored using
        ``fftshift``.

        ``fftshiftFlag`` tracks the current data ordering to avoid applying the
        shift repeatedly.
        """
        if self.params.fftshiftSwitch:
            if self.params.fftshiftFlag == 0:
                print("check fftshift...")
                print("fftshift data for fast far-field update")
                # shift detector quantities
                self.experimentalData.ptychogram = np.fft.ifftshift(
                    self.experimentalData.ptychogram, axes=(-1, -2)
                )
                if hasattr(self.experimentalData, "ptychogramDownsampled"):
                    self.experimentalData.ptychogramDownsampled = np.fft.ifftshift(
                        self.experimentalData.ptychogramDownsampled, axes=(-1, -2)
                    )
                if hasattr(self.experimentalData, "W"):
                    if self.experimentalData.W is not None:
                        self.experimentalData.W = np.fft.ifftshift(
                            self.experimentalData.W, axes=(-1, -2)
                        )
                if self.experimentalData.emptyBeam is not None:
                    self.experimentalData.emptyBeam = np.fft.ifftshift(
                        self.experimentalData.emptyBeam, axes=(-1, -2)
                    )
                if hasattr(self.experimentalData, "PSD"):
                    if self.experimentalData.PSD is not None:
                        self.experimentalData.PSD = np.fft.ifftshift(
                            self.experimentalData.PSD, axes=(-1, -2)
                        )
                self.params.fftshiftFlag = 1
        else:
            if self.params.fftshiftFlag == 1:
                print("check fftshift...")
                print("ifftshift data")
                self.experimentalData.ptychogram = np.fft.fftshift(
                    self.experimentalData.ptychogram, axes=(-1, -2)
                )
                if hasattr(self.experimentalData, "ptychogramDownsampled"):
                    self.experimentalData.ptychogramDownsampled = np.fft.fftshift(
                        self.experimentalData.ptychogramDownsampled, axes=(-1, -2)
                    )
                if self.experimentalData.W != None:
                    self.experimentalData.W = np.fft.fftshift(
                        self.experimentalData.W, axes=(-1, -2)
                    )
                if self.experimentalData.emptyBeam != None:
                    self.experimentalData.emptyBeam = np.fft.fftshift(
                        self.experimentalData.emptyBeam, axes=(-1, -2)
                    )
                if hasattr(self.experimentalData, "PSD"):
                    if self.experimentalData.PSD is not None:
                        self.experimentalData.PSD = np.fft.fftshift(
                            self.experimentalData.PSD, axes=(-1, -2)
                    )
                self.params.fftshiftFlag = 0

    def _move_data_to_gpu(self):
        """
        Move reconstruction data required by the engine to the GPU.

        Reconstruction and experimental-data fields are transferred by their
        respective container classes. Engine-specific fields, including the probe
        window and additional aPIE data when required, are transferred separately.
        """

        self.reconstruction._move_data_to_gpu()
        self.experimentalData._move_data_to_gpu()

        transfer_fields_to_gpu(
            self,
            [
                "probeWindow",
            ],
            self.logger,
        )  # '.probeWindow = cp.array(self.probeWindow)

        # reconstruction parameters
        # self.reconstruction.probe = cp.array(self.reconstruction.probe, cp.complex64)
        # self.reconstruction.object = cp.array(self.reconstruction.object, cp.complex64)
        # self.reconstruction.detectorError = cp.array(self.reconstruction.detectorError, cp.float32)

        # if self.params.momentumAcceleration:
        #     self.reconstruction.probeBuffer = cp.array(self.reconstruction.probeBuffer, cp.complex64)
        #     self.reconstruction.objectBuffer = cp.array(self.reconstruction.objectBuffer, cp.complex64)
        #     self.reconstruction.probeMomentum = cp.array(self.reconstruction.probeMomentum, cp.complex64)
        #     self.reconstruction.objectMomentum = cp.array(self.reconstruction.objectMomentum, cp.complex64)

        # for doing the coordinate transform and especially the otherwise slow interpolation of aPIE on the gpu
        if hasattr(self.params, "aPIEflag"):
            if self.params.aPIEflag == True:
                fields_to_transfer = [
                    "ptychogramUntransformed",
                    "Uq",
                    "Vq",
                    "theta",
                    "wavelength",
                    "Xd",
                    "Yd",
                    "dxd",
                    "zo",
                ]
                self.theta = self.reconstruction.theta
                self.wavelength = self.reconstruction.wavelength

                transfer_fields_to_gpu(self, fields_to_transfer, self.logger)
                # self.ptychogramUntransformed = cp.array(self.ptychogramUntransformed)
                # self.Uq = cp.array(self.Uq)
                # self.Vq = cp.array(self.Vq)
                # self.theta = cp.array(self.reconstruction.theta)
                # self.wavelength = cp.array(self.reconstruction.wavelength)
                # self.Xd = cp.array(self.Xd)
                # self.Yd = cp.array(self.Yd)
                # self.dxd = cp.array(self.dxd)
                # self.zo = cp.array(self.zo)
                # self.experimentalData.W = cp.array(self.experimentalData.W)

        # non-reconstruction parameters
        # if hasattr(self.experimentalData, 'ptychogramDownsampled'):
        #     self.experimentalData.ptychogramDownsampled = cp.array(self.experimentalData.ptychogramDownsampled,
        #                                                            cp.float32)
        # else:
        #     self.experimentalData.ptychogram = cp.array(self.experimentalData.ptychogram, cp.float32)

        # propagators
        # if self.params.propagatorType == 'Fresnel':
        #     self.reconstruction.quadraticPhase = cp.array(self.reconstruction.quadraticPhase)
        # elif self.params.propagatorType == 'ASP' or self.params.propagatorType == 'polychromeASP':
        #     self.reconstruction.transferFunction = cp.array(self.reconstruction.transferFunction)
        # elif self.params.propagatorType == 'scaledASP' or self.params.propagatorType == 'scaledPolychromeASP':
        #     self.reconstruction.Q1 = cp.array(self.reconstruction.Q1)
        #     self.reconstruction.Q2 = cp.array(self.reconstruction.Q2)
        # elif self.params.propagatorType == 'twoStepPolychrome':
        #     self.reconstruction.quadraticPhase = cp.array(self.reconstruction.quadraticPhase)
        #     self.reconstruction.transferFunction = cp.array(self.reconstruction.transferFunction)

        # other parameters
        # if self.params.backgroundModeSwitch:
        #     self.reconstruction.background = cp.array(self.reconstruction.background)
        # if self.params.absorbingProbeBoundary or self.params.probeBoundary:

        # if self.params.modulusEnforcedProbeSwitch:
        #     self.experimentalData.emptyBeam = cp.array(self.experimentalData.emptyBeam)
        # if self.params.intensityConstraint == 'interferometric':
        #     self.reconstruction.reference = cp.array(self.reconstruction.reference)

    def _move_data_to_cpu(self):
        """
        Move reconstruction data required by the engine to the CPU.

        Reconstruction and experimental-data fields are transferred by their
        respective container classes, together with engine-specific fields such
        as the probe window.
        """
        # reconstruction parameters

        self.reconstruction._move_data_to_cpu()
        self.experimentalData._move_data_to_cpu()
        transfer_fields_to_cpu(
            self,
            [
                "probeWindow",
            ],
            self.logger,
        )

        # self.reconstruction.move_to_CPU()
        # self.params.move_to_CPU()

        # self.reconstruction.probe = asNumpyArray(self.reconstruction.probe)
        # self.reconstruction.object = asNumpyArray(self.reconstruction.object)

        # if self.params.momentumAcceleration:
        # reconstruction_fields_to_transfer = ['probeBuffer', 'objectBuffer', 'probeMomentum',' objectMomentum']
        # for field in reconstruction_fields_to_transfer:
        #
        #     setattr(self.reconstruction, field,)
        # self.reconstruction.probeBuffer = self.reconstruction.probeBuffer.get()
        # self.reconstruction.objectBuffer = self.reconstruction.objectBuffer.get()
        # self.reconstruction.probeMomentum = self.reconstruction.probeMomentum.get()
        # self.reconstruction.objectMomentum = self.reconstruction.objectMomentum.get()

        # for doing the coordinate transform and especially the otherwise slow interpolation of aPIE on the gpu
        # if hasattr(self.params, 'aPIEflag'):
        #     if self.params.aPIEflag:
        #         self.theta = self.theta.get()
        #
        # fields_to_transfer = ["theta", "probeWindow"]
        # self.probeWindow = self.probeWindow.get()

        # non-reconstruction parameters
        # if hasattr(self.experimentalData, 'ptychogramDownsampled'):
        #     self.experimentalData.ptychogramDownsampled = self.experimentalData.ptychogramDownsampled.get()
        # else:
        #     self.experimentalData.ptychogram = self.experimentalData.ptychogram.get()
        # self.reconstruction.detectorError = self.reconstruction.detectorError.get()

        # propagators
        # if self.params.propagatorType == 'Fresnel':
        # self.reconstruction.quadraticPhase = self.reconstruction.quadraticPhase.get()
        # elif self.params.propagatorType == 'ASP' or self.params.propagatorType == 'polychromeASP':
        #     self.reconstruction.transferFunction = self.reconstruction.transferFunction.get()
        # elif self.params.propagatorType == 'scaledASP' or self.params.propagatorType == 'scaledPolychromeASP':
        #     self.reconstruction.Q1 = self.reconstruction.Q1.get()
        #     self.reconstruction.Q2 = self.reconstruction.Q2.get()
        # elif self.params.propagatorType == 'twoStepPolychrome':
        #     self.reconstruction.quadraticPhase = self.reconstruction.quadraticPhase.get()
        #     self.reconstruction.transferFunction = self.reconstruction.transferFunction.get()

        # other parameters
        # if self.params.backgroundModeSwitch:
        #     # self.reconstruction.background = self.reconstruction.background.get()
        # if self.params.absorbingProbeBoundary or self.params.probeBoundary:

        # if self.params.modulusEnforcedProbeSwitch:
        #     self.experimentalData.emptyBeam = self.experimentalData.emptyBeam.get()
        # # if self.params.intensityConstraint == 'interferometric':
        #     self.reconstruction.reference = self.reconstruction.reference.get()

    def _checkGPU(self):
        """
        Synchronize reconstruction data with the selected computation device.

        If GPU execution is enabled, required reconstruction, experimental-data,
        and engine fields are transferred to the GPU. If GPU execution is
        disabled, the corresponding data are transferred back to the CPU.

        ``gpuFlag`` tracks the current device state, while repeated transfers
        ensure that fields created after a previous device switch are also
        synchronized.

        Raises:
            ImportError:
                If GPU execution is requested but CuPy is not available.
        """
        if not hasattr(self.params, "gpuFlag"):
            self.params.gpuFlag = 0

        if self.params._gpuSwitch:
            if cp is None:
                raise ImportError(
                    "Could not import cupy, therefore no GPU reconstruction is possible. To reconstruct, set the params.gpuSwitch to False."
                )
            if not self.params.gpuFlag:
                self.logger.info("switch to gpu")

                # load data to gpu
                self._move_data_to_gpu()
                self.params.gpuFlag = 1
            # always do this as it gets away with hard to debug errors
            self._move_data_to_gpu()
        else:
            self._move_data_to_cpu()
            if self.params.gpuFlag:
                self.logger.info("switch to cpu")
                self._move_data_to_cpu()
                self.params.gpuFlag = 0

    def setPositionOrder(self):
        """
        Set the order in which scan positions are processed.

        The ordering is controlled by ``params.positionOrder``:

        - ``"sequential"`` processes frames in their original order.
        - ``"random"`` uses the original order during the first two iterations
        and randomly shuffles the positions afterwards.
        - ``"NA"`` sorts positions by their distance from the scan center,
        processing central positions first. This ordering is intended for FPM,
        where central illumination angles correspond to bright-field data.

        The resulting frame indices are stored in ``positionIndices``.

        Raises:
            ValueError:
                If ``positionOrder`` is not one of the supported options.
        """
        if self.params.positionOrder == "sequential":
            self.positionIndices = np.arange(self.experimentalData.numFrames)

        elif self.params.positionOrder == "random":
            if len(self.reconstruction.error) == 0:
                self.positionIndices = np.arange(self.experimentalData.numFrames)
            else:
                if len(self.reconstruction.error) < 2:
                    self.positionIndices = np.arange(self.experimentalData.numFrames)
                else:
                    self.positionIndices = np.arange(self.experimentalData.numFrames)
                    np.random.shuffle(self.positionIndices)

        # order by illumiantion angles. Use smallest angles first
        # (i.e. start with brightfield data first, then add the low SNR
        # darkfield)
        # todo check this with Antonios
        elif self.params.positionOrder == "NA":
            rows = self.reconstruction.positions[:, 0] - np.mean(
                self.reconstruction.positions[:, 0]
            )
            cols = self.reconstruction.positions[:, 1] - np.mean(
                self.reconstruction.positions[:, 1]
            )
            dist = np.sqrt(rows**2 + cols**2)
            self.positionIndices = np.argsort(dist)
        else:
            raise ValueError("position order not properly set")

    def changeExperimentalData(self, experimentalData: ExperimentalData):
        """
        Replace the experimental-data object referenced by the engine.

        Args:
            experimentalData (ExperimentalData):
                Experimental dataset to use in subsequent reconstruction steps.

        Raises:
            TypeError:
                If ``experimentalData`` is not an ``ExperimentalData`` instance.
        """
        if experimentalData is not None:
            if not isinstance(experimentalData, ExperimentalData):
                raise TypeError("Experimental data should be of class ExperimentalData")
            self.experimentalData = experimentalData

    def changeOptimizable(self, optimizable: Reconstruction):
        """
        Replace the reconstruction object referenced by the engine.

        Args:
            optimizable (Reconstruction):
                Reconstruction state to use in subsequent engine operations.

        Raises:
            TypeError:
                If ``optimizable`` is not a ``Reconstruction`` instance.
        """
        if optimizable is not None:
            if not isinstance(optimizable, Reconstruction):
                raise TypeError(
                    f"Argument should be an subclass of Reconstruction, but it is {type(optimizable)}"
                )
            self.reconstruction = optimizable

    def convert2single(self):
        """
        Configure single-precision data types for reconstruction arrays.

        This method sets the target complex and real data types to
        ``numpy.complex64`` and ``numpy.float32`` and delegates the actual
        conversion to dtype-matching helpers.

        """
        self.dtype_complex = np.complex64
        self.dtype_real = np.float32
        self._match_dtypes_complex()
        self._match_dtypes_real()

    def _match_dtypes_complex(self):
        """
        Convert complex-valued arrays to the configured complex dtype.

        Complex arrays stored by the engine, reconstruction, and experimental
        data containers are converted to ``self.dtype_complex``. Non-array
        attributes and non-complex arrays are left unchanged.
        """
        array_types = (np.ndarray,)
        if cp is not None:
            array_types += (cp.ndarray,)

        for container in (self, self.reconstruction, self.experimentalData):
            for name, value in vars(container).items():
                if not isinstance(value, array_types):
                    continue

                if value.dtype.kind == "c":
                    setattr(
                        container,
                        name,
                        value.astype(self.dtype_complex, copy=False),
                    )

    def _match_dtypes_real(self):
        """
        Convert floating-point arrays to the configured real dtype.

        Floating-point arrays stored by the engine, reconstruction, and
        experimental data containers are converted to ``self.dtype_real``.
        Integer, boolean, complex, and non-array attributes are left unchanged.
        """
        array_types = (np.ndarray,)
        if cp is not None:
            array_types += (cp.ndarray,)

        for container in (self, self.reconstruction, self.experimentalData):
            for name, value in vars(container).items():
                if not isinstance(value, array_types):
                    continue

                if value.dtype.kind == "f":
                    setattr(
                        container,
                        name,
                        value.astype(self.dtype_real, copy=False),
                    )
        

    def object2detector(self, esw=None):
        """
        Propagate the exit surface wave from the object plane to the detector plane.

        If ``esw`` is not provided, ``reconstruction.esw`` is used. The propagation
        is performed by the operator selected through the reconstruction
        parameters, and the propagated detector-plane field is stored in
        ``reconstruction.ESW``.

        Args:
            esw (ndarray, optional):
                Object-plane exit surface wave. If None, use
                ``reconstruction.esw``.
        """
        if esw is None:
            # todo: check this, it seems weird to store it in self.esw
            esw = self.reconstruction.esw
        self.esw, self.reconstruction.ESW = Operators.Operators.object2detector(
            esw, self.params, self.reconstruction
        )

    def detector2object(self, ESW=None):
        """
        Propagate the detector-plane field back to the object plane.

        If ``ESW`` is not provided, ``reconstruction.ESW`` is used. The
        back-propagated exit surface wave is stored in ``reconstruction.esw``,
        together with the corresponding update field in
        ``reconstruction.eswUpdate``.

        Args:
            ESW (ndarray, optional):
                Detector-plane wavefield. If None, use
                ``reconstruction.ESW``.
        """
        if ESW is None:
            ESW = self.reconstruction.ESW
        esw, eswUpdate = Operators.Operators.detector2object(
            ESW, self.params, self.reconstruction
        )
        # Dirk is not sure why this has to be changed at all but it sometimes is changed for some reason
        self.reconstruction.esw = esw
        # this is the new estimate which will be processed later
        self.reconstruction.eswUpdate = eswUpdate

    def fft2s(self):
        """
        Computes the fourier transform of the exit surface wave.
        :return:
        """
        self.reconstruction.ESW = FT2(
            self.reconstruction.esw, self.params.fftshiftSwitch
        )

    def ifft2s(self):
        """Inverse FFT"""
        # find out if this should be performed on the GPU
        self.reconstruction.eswUpdate = IFT(
            self.reconstruction.ESW, self.params.fftshiftSwitch
        )

    def getBeamWidth(self):
        r"""
        Estimate the probe beam width from the second moment of its intensity.

        The probe intensity is summed over the non-spatial dimensions and
        normalized as

        $$
        \tilde{P}(x,y) = \frac{P(x,y)}{\sum_{x,y} P(x,y)}
        $$

        The intensity-weighted centroid and variance are then calculated, for
        example along $x$ as

        $$
        \langle x \rangle = \sum_{x,y} x\tilde{P}(x,y)
        $$

        $$
        \sigma_x^2 = \sum_{x,y}(x-\langle x\rangle)^2\tilde{P}(x,y)
        $$

        and converted to a Gaussian-equivalent full width at half maximum:

        $$
        \mathrm{FWHM}_x = 2\sqrt{2\ln 2}\sigma_x
        $$

        The same calculation is applied along $y$. 

        Returns:
            tuple:
                ``(beamWidthY, beamWidthX)`` in meters.
        
        Notes:
            For non-Gaussian probe
            profiles, the returned values should be interpreted as second-moment
            Gaussian-equivalent beam widths rather than direct half-maximum widths.
        """
        xp = getArrayModule(self.reconstruction.probe)
        P = xp.sum(
            abs((self.reconstruction.probe[..., -1, :, :])) ** 2,
            axis=(0, 1, 2),
        )
        P = P / xp.sum(P, axis=(-1, -2))
        P = asNumpyArray(P)
        xMean = np.sum(self.reconstruction.Xp * P, axis=(-1, -2))
        yMean = np.sum(self.reconstruction.Yp * P, axis=(-1, -2))
        xVariance = np.sum((self.reconstruction.Xp - xMean) ** 2 * P, axis=(-1, -2))
        yVariance = np.sum((self.reconstruction.Yp - yMean) ** 2 * P, axis=(-1, -2))

        c = (
            2 * xp.sqrt(2 * xp.log(2))
        )  # constant for converting variance to FWHM (see e.g. https://en.wikipedia.org/wiki/Full_width_at_half_maximum)

        self.reconstruction.beamWidthX = asNumpyArray(c * np.sqrt(xVariance))
        self.reconstruction.beamWidthY = asNumpyArray(c * np.sqrt(yVariance))

        return self.reconstruction.beamWidthY, self.reconstruction.beamWidthX

    def getOverlap(self, ind1, ind2):
        r"""
        Estimate the probe overlap between two scan positions.

        The physical displacement between the two positions is calculated from
        their pixel-coordinate difference and the probe-plane pixel size:

        $$
        s_x = |x_2-x_1|dx_p,\qquad s_y = |y_2-y_1|dx_p
        $$

        The linear overlap is estimated from the radial scan displacement and
        the smaller of the reconstructed probe widths:

        $$
        O_{\mathrm{linear}} = \max\left(1-\frac{\sqrt{s_x^2+s_y^2}}{\min(w_x,w_y)},0\right)
        $$

        where $w_x$ and $w_y$ are the Gaussian-equivalent probe widths returned
        by `getBeamWidth()`.

        The area overlap is calculated from the normalized autocorrelation of the
        probe amplitude. Let

        $$
        P(x,y) = |\mathrm{probe}(x,y)|
        $$

        and

        $$
        Q(f_x,f_y) = \mathcal{F}\{P(x,y)\}.
        $$

        Using the Fourier correlation theorem, the normalized area overlap is

        $$
        O_{\mathrm{area}} = \frac{1}{N_\lambda}\sum_\lambda\frac{\left|\sum_{f_x,f_y}|Q_\lambda(f_x,f_y)|^2\exp[-i2\pi(f_xs_x+f_ys_y)]\right|}{\sum_{f_x,f_y}|Q_\lambda(f_x,f_y)|^2}.
        $$

        This definition is invariant to the absolute centering of the probe and
        yields an overlap of one for zero displacement.

        The current implementation uses the first probe mode and the last slice
        when evaluating the area overlap.

        Args:
            ind1 (int):
                Index of the first scan position.
            ind2 (int):
                Index of the second scan position.

        Returns:
            tuple:
                ``(linearOverlap, areaOverlap)``.

        Notes:
            The calculated values are also stored in
            ``reconstruction.linearOverlap`` and ``reconstruction.areaOverlap``.
        """
        sy = (
            abs(
                self.reconstruction.positions[ind2, 0]
                - self.reconstruction.positions[ind1, 0]
            )
            * self.reconstruction.dxp
        )
        sx = (
            abs(
                self.reconstruction.positions[ind2, 1]
                - self.reconstruction.positions[ind1, 1]
            )
            * self.reconstruction.dxp
        )

        # task 1: get linear overlap
        self.getBeamWidth()
        self.reconstruction.linearOverlap = 1 - np.sqrt(sx**2 + sy**2) / np.minimum(
            self.reconstruction.beamWidthX, self.reconstruction.beamWidthY
        )
        self.reconstruction.linearOverlap = np.maximum(
            self.reconstruction.linearOverlap, 0
        )

        # task 2: get area overlap
        # spatial frequency pixel size
        df = 1 / (self.reconstruction.Np * self.reconstruction.dxp)
        # spatial frequency meshgrid
        fx = np.arange(-self.reconstruction.Np // 2, self.reconstruction.Np // 2) * df
        Fx, Fy = np.meshgrid(fx, fx)
        # absolute value of probe and 2D fft
        P = abs(asNumpyArray(self.reconstruction.probe[:, 0, 0, -1, ...]))
        Q = fft2c(P)
        # calculate overlap between positions
        self.reconstruction.areaOverlap = np.mean(
            abs(
                np.sum(
                    abs(Q)**2 * np.exp(-1.0j * 2 * np.pi * (Fx * sx + Fy * sy)),
                    axis=(-1, -2),
                )
            )
            / np.sum(abs(Q) ** 2, axis=(-1, -2)),
            axis=0,
        )
        return (
            self.reconstruction.linearOverlap,
            self.reconstruction.areaOverlap,
        )

    def getErrorMetrics(self):
        r"""
        Compute the normalized reconstruction error for the current iteration.

        For each scan position $j$, the detector-domain error is summed over all
        detector pixels:

        $$
        e_j = \sum_{x,y} E_j(x,y)
        $$

        where $E_j(x,y)$ is the absolute difference between measured and estimated
        detector intensities. If `FourierMaskSwitch` is enabled, the detector error
        is weighted by the Fourier mask $W$:

        $$
        e_j = \sum_{x,y} E_j(x,y)W(x,y)
        $$

        The error at each scan position is normalized by the measured diffraction
        energy:

        $$
        \tilde{e}_j = \frac{e_j}{E_j^{\mathrm{meas}} + 10^{-20}}
        $$

        The total error for the current iteration is then

        $$
        e_{\mathrm{iter}} = \sum_j \tilde{e}_j
        $$

        and is appended to `reconstruction.error`.

        Notes:
            If `saveMemory` is enabled, the per-position errors are accumulated
            during the reconstruction loop instead of storing the full detector
            error array.
        """
        if not self.params.saveMemory:
            # Calculate mean error for all positions (make separate function for all of that)
            if self.params.FourierMaskSwitch:
                self.reconstruction.errorAtPos = np.sum(
                    np.abs(self.reconstruction.detectorError) * self.experimentalData.W,
                    axis=(-1, -2),
                )
            else:
                self.reconstruction.errorAtPos = np.sum(
                    np.abs(self.reconstruction.detectorError), axis=(-1, -2)
                )
        self.reconstruction.errorAtPos = asNumpyArray(
            self.reconstruction.errorAtPos
        ) / asNumpyArray(self.experimentalData.energyAtPos + 1e-20)
        eAverage = np.sum(self.reconstruction.errorAtPos)

        # append to error vector (for plotting error as function of iteration)
        self.reconstruction.error = np.append(self.reconstruction.error, eAverage)

    def getRMSD(self, positionIndex):
        r"""
        Compute the detector-domain intensity error for one scan position.

        The pixel-wise detector error is defined as

        $$
        E(x,y) = \left|I_{\mathrm{measured}}(x,y) - I_{\mathrm{estimated}}(x,y)\right|.
        $$

        If `saveMemory` is disabled, the full detector-error map is stored in
        `reconstruction.detectorError`.

        If `saveMemory` is enabled, the detector error is reduced immediately to
        a per-position scalar:

        $$
        e_j = \sum_{x,y} E_j(x,y).
        $$

        When `FourierMaskSwitch` is enabled, the masked error is used instead:

        $$
        e_j = \sum_{x,y} E_j(x,y)W(x,y).
        $$

        Args:
            positionIndex (int):
                Index of the current scan position.

        Raises:
            NotImplementedError:
                If `saveMemory`, `FourierMaskSwitch`, and `CPSCswitch` are all
                enabled simultaneously.

        Notes:
            Despite its name, this method does not currently compute a root mean
            square deviation. It computes an absolute detector-intensity
            difference.
        """
        # find out wether or not to use the GPU
        xp = getArrayModule(self.reconstruction.Iestimated)
        self.currentDetectorError = abs(
            self.reconstruction.Imeasured - self.reconstruction.Iestimated
        )

        # todo saveMemory implementation
        if self.params.saveMemory:
            if self.params.FourierMaskSwitch and not self.params.CPSCswitch:
                self.reconstruction.errorAtPos[positionIndex] = xp.sum(
                    self.currentDetectorError * self.experimentalData.W
                )
            elif self.params.FourierMaskSwitch and self.params.CPSCswitch:
                raise NotImplementedError
            else:
                self.reconstruction.errorAtPos[positionIndex] = asNumpyArray(
                    xp.sum(self.currentDetectorError)
                )
        else:
            self.reconstruction.detectorError[positionIndex] = self.currentDetectorError

    def intensityProjection(self, positionIndex):
        r"""
        Apply the detector-plane intensity constraint for one scan position.

        The current exit surface wave is first propagated to the detector plane.
        The estimated intensity is calculated from the propagated field as

        $$
        I_{\mathrm{estimated}}(x,y)=\sum_m |\Psi_m(x,y)|^2.
        $$

        For the standard intensity constraint, the detector-plane field is scaled
        by

        $$
        f(x,y)=\sqrt{\frac{I_{\mathrm{measured}}(x,y)}{I_{\mathrm{estimated}}(x,y)+\epsilon}},
        $$

        so that

        $$
        \Psi_{\mathrm{updated}}(x,y)=\Psi(x,y)f(x,y).
        $$

        Alternative projection rules are selected through
        ``params.intensityConstraint``. The current implementation supports
        ``"standard"``, ``"fluctuation"``, ``"exponential"``,
        ``"poisson"``, and ``"interferometric"``.

        Depending on the active parameters, the method may additionally apply
        constrained-pixel-sum decompression, adaptive denoising, Fourier masking,
        background estimation, or an interferometric reference update.

        The constrained detector-plane field is finally propagated back to the
        object plane.

        Args:
            positionIndex (int):
                Index of the current diffraction frame.

        Raises:
            ValueError:
                If ``intensityConstraint`` is not a supported value.
        """
        # figure out whether or not to use the GPU
        xp = getArrayModule(self.reconstruction.esw)
        # zero division mitigator
        gimmel = 1e-10

        # propagate to detector
        self.object2detector()

        # get estimated intensity (2D array, in the case of multislice, only take the last slice)
        if self.params.intensityConstraint == "interferometric":
            self.reconstruction.Iestimated = xp.sum(
                xp.abs(self.reconstruction.ESW + self.reconstruction.reference) ** 2,
                axis=(0, 1, 2),
            )[-1]
        else:
            self.reconstruction.Iestimated = xp.sum(
                xp.abs(self.reconstruction.ESW) ** 2, axis=(0, 1, 2)
            )[-1]
            self.logger.debug(
                f"Estimated intensity: {self.reconstruction.Iestimated.sum()}, Measured: {self.experimentalData.ptychogram[positionIndex].sum()}"
            )
        if self.params.backgroundModeSwitch:
            self.reconstruction.Iestimated += self.reconstruction.background

        # get measured intensity todo implement kPIE
        if self.params.CPSCswitch:
            self.decompressionProjection(positionIndex)
        else:
            self.reconstruction.Imeasured = self.experimentalData.ptychogram[
                positionIndex
            ]

        self.getRMSD(positionIndex)

        # adaptive denoising
        if self.params.adaptiveDenoisingSwitch:
            self.adaptiveDenoising()

        # intensity projection constraints
        if self.params.intensityConstraint == "fluctuation":
            # scaling
            if self.params.FourierMaskSwitch:
                aleph = xp.sum(
                    self.reconstruction.Imeasured
                    * self.reconstruction.Iestimated
                    * self.experimentalData.W
                ) / xp.sum(
                    self.reconstruction.Imeasured
                    * self.reconstruction.Imeasured
                    * self.experimentalData.W
                )
            else:
                aleph = xp.sum(
                    self.reconstruction.Imeasured * self.reconstruction.Iestimated
                ) / xp.sum(
                    self.reconstruction.Imeasured * self.reconstruction.Imeasured
                )
            self.params.intensityScaling[positionIndex] = aleph
            # scaled projection
            frac = (
                (1 + aleph)
                / 2
                * self.reconstruction.Imeasured
                / (self.reconstruction.Iestimated + gimmel)
            )

        elif self.params.intensityConstraint == "exponential":
            x = self.currentDetectorError / (self.reconstruction.Iestimated + gimmel)
            W = xp.exp(-0.05 * x)
            frac = xp.sqrt(
                self.reconstruction.Imeasured
                / (self.reconstruction.Iestimated + gimmel)
            )
            frac = W * frac + (1 - W)

        elif self.params.intensityConstraint == "poisson":
            frac = self.reconstruction.Imeasured / (
                self.reconstruction.Iestimated + gimmel
            )

        elif (
            self.params.intensityConstraint == "standard"
            or self.params.intensityConstraint == "interferometric"
        ):
            frac = xp.sqrt(
                self.reconstruction.Imeasured
                / (self.reconstruction.Iestimated + gimmel)
            )

        else:
            raise ValueError("intensity constraint not properly specified!")

        # apply mask
        if (
            self.params.FourierMaskSwitch
            and self.params.CPSCswitch
            and len(self.reconstruction.error) > 5
        ):
            frac = self.experimentalData.W * frac + (1 - self.experimentalData.W)

        # update ESW
        if self.params.intensityConstraint == "interferometric":
            temp = (
                self.reconstruction.ESW + self.reconstruction.reference
            ) * frac - self.reconstruction.ESW
            self.reconstruction.ESW = (
                self.reconstruction.ESW + self.reconstruction.reference
            ) * frac - self.reconstruction.reference
            self.reconstruction.reference = temp
        else:
            if hasattr(self.params, "intensityMask"):
                if self.params.intensityMask:
                    self.reconstruction.ESW = self.reconstruction.ESW * (
                        frac * (self.reconstruction.intensity_mask)
                        + (self.reconstruction.intensity_mask - 1)
                    )
                else:
                    self.reconstruction.ESW = self.reconstruction.ESW * frac
            else:
                self.reconstruction.ESW = self.reconstruction.ESW * frac

        # update background (see PhD thsis by Peng Li)
        if self.params.backgroundModeSwitch:
            if self.params.FourierMaskSwitch:
                self.reconstruction.background = (
                    self.reconstruction.background
                    * (1 + 1 / self.experimentalData.numFrames * (xp.sqrt(frac) - 1))
                    ** 2
                    * self.experimentalData.W
                )
            else:
                self.reconstruction.background = (
                    self.reconstruction.background
                    * (1 + 1 / self.experimentalData.numFrames * (xp.sqrt(frac) - 1))
                    ** 2
                )

        # back propagate to object plane
        self.detector2object()

    def decompressionProjection(self, positionIndex):
        r"""
            Construct a high-resolution measured intensity for the
        constrained-pixel-sum projection.

        The current high-resolution estimated intensity is divided into blocks
        of size $s \times s$, where $s$ is `CPSCupsamplingFactor`.

        For each detector block, the predicted low-resolution intensity is

        $$
        \hat{I}^{\mathrm{LR}}_{mn} = \sum_{i,j \in \mathrm{block}_{mn}} I_{\mathrm{estimated}}(i,j)
        $$

        A block-wise correction factor is calculated from the measured
        low-resolution diffraction intensity:

        $$
        S_{mn} = \frac{I^{\mathrm{LR}}_{\mathrm{measured},mn}}{\hat{I}^{\mathrm{LR}}_{mn} + \epsilon}
        $$

        The factor is expanded over the corresponding high-resolution block
        and used to construct an effective high-resolution measured intensity:

        $$
        I^{\mathrm{HR}}_{\mathrm{measured}}(i,j) = I_{\mathrm{estimated}}(i,j) S_{mn}
        $$

        This preserves the current estimate of the sub-pixel intensity
        distribution while enforcing that the summed intensity in each
        high-resolution block matches the experimentally measured
        low-resolution pixel intensity.

        Args:
            positionIndex (int):
                Index of the current diffraction frame.

        Notes:
            The generated high-resolution intensity is stored in
            `reconstruction.Imeasured`. When `FourierMaskSwitch` is enabled,
            the CPSC correction is restricted by `experimentalData.W` after
            the first five reconstruction iterations.
        """
        # overwrite the measured intensity (just to have same dimensions as Iestimated)
        xp = getArrayModule(self.reconstruction.Iestimated)

        # determine downsampled fraction (Sl)
        frac = self.experimentalData.ptychogramDownsampled[positionIndex] / (
            xp.sum(
                self.reconstruction.Iestimated.reshape(
                    self.reconstruction.Nd // self.params.CPSCupsamplingFactor,
                    self.params.CPSCupsamplingFactor,
                    self.reconstruction.Nd // self.params.CPSCupsamplingFactor,
                    self.params.CPSCupsamplingFactor,
                ),
                axis=(1, 3),
            )
            + np.finfo(np.float32).eps
        )
        if self.params.FourierMaskSwitch and len(self.reconstruction.error) > 5:
            frac = self.experimentalData.W * frac + (1 - self.experimentalData.W)
        # overwrite up-sampled measured intensity
        self.reconstruction.Imeasured = self.reconstruction.Iestimated * xp.repeat(
            xp.repeat(frac, self.params.CPSCupsamplingFactor, axis=-1),
            self.params.CPSCupsamplingFactor,
            axis=-2,
        )

    def showReconstruction(self, loop):
        """
        Update reconstruction monitoring and optional iteration output.

        The object, probe, reconstruction error, propagation distance, mode
        purities, scan positions, and beam width are sent to the active monitor
        at intervals defined by ``monitor.figureUpdateFrequency``.

        For Fourier ptychography, the object is transformed to real space before
        visualization. When the monitor verbosity is set to ``"high"``, the
        current measured and estimated diffraction intensities, reconstruction
        error, and estimated scan overlap are also displayed.

        If object dumping is enabled, the current reconstructed object is written
        to disk for each iteration.

        Args:
            loop (int):
                Current reconstruction iteration.
        """
        if np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
            if self.experimentalData.operationMode == "FPM":
                object_estimate = np.squeeze(
                    asNumpyArray(
                        fft2c(self.reconstruction.object)[
                            ..., self.monitor.objectROI[0], self.monitor.objectROI[1]
                        ]
                    )
                )
                probe_estimate = np.squeeze(
                    asNumpyArray(
                        self.reconstruction.probe[
                            ..., self.monitor.probeROI[0], self.monitor.probeROI[1]
                        ]
                    )
                )
            else:
                object_estimate = np.squeeze(
                    asNumpyArray(
                        self.reconstruction.object[
                            ..., self.monitor.objectROI[0], self.monitor.objectROI[1]
                        ]
                    )
                )
                probe_estimate = np.squeeze(
                    asNumpyArray(
                        self.reconstruction.probe[
                            ..., self.monitor.probeROI[0], self.monitor.probeROI[1]
                        ]
                    )
                )
            self.monitor.updateObjectProbeErrorMonitor(
                error=self.reconstruction.error,
                object_estimate=object_estimate,
                probe_estimate=probe_estimate,
                zo=self.reconstruction.zo,
                purity_probe=self.reconstruction.purityProbe,
                purity_object=self.reconstruction.purityObject,
                encoder_positions=self.reconstruction.positions,
            )

            self.monitor.writeEngineName(repr(type(self)))

            self.monitor.update_encoder(
                corrected_positions=self.reconstruction.encoder_corrected,
                original_positions=self.experimentalData.encoder,
            )

            self.monitor.updateBeamWidth(*self.getBeamWidth())

            # self.monitor.visualize_probe_engine(self.reconstruction.probe_storage)

            if self.monitor.verboseLevel == "high":
                if self.params.fftshiftSwitch:
                    Iestimated = np.fft.fftshift(
                        asNumpyArray(self.reconstruction.Iestimated)
                    )
                    Imeasured = np.fft.fftshift(
                        asNumpyArray(self.reconstruction.Imeasured)
                    )
                else:
                    Iestimated = asNumpyArray(self.reconstruction.Iestimated)
                    Imeasured = asNumpyArray(self.reconstruction.Imeasured)

                self.monitor.updateDiffractionDataMonitor(
                    Iestimated=Iestimated, Imeasured=Imeasured
                )

                self.getOverlap(0, 1)

                self.pbar.write("")
                self.pbar.write("iteration: %i" % loop)
                self.pbar.write("error: %.1f" % self.reconstruction.error[-1])
                self.pbar.write(
                    "estimated linear overlap: %.1f %%"
                    % (100 * self.reconstruction.linearOverlap)
                )
                self.pbar.write(
                    "estimated area overlap: %.1f %%"
                    % (100 * self.reconstruction.areaOverlap)
                )

                self.monitor.update_overlap(
                    self.reconstruction.areaOverlap, self.reconstruction.linearOverlap
                )
                # self.pbar.write('coherence structure:')

            if self.params.positionCorrectionSwitch:
                # show reconstruction
                return
                if (
                    len(self.reconstruction.error) > self.startAtIteration
                ):  # & (np.mod(loop,
                    # self.monitor.figureUpdateFrequency) == 0):
                    figure, ax = plt.subplots(
                        1, 1, num=102, squeeze=True, clear=True, figsize=(5, 5)
                    )
                    ax.set_title("Estimated scan grid positions")
                    ax.set_xlabel("(um)")
                    ax.set_ylabel("(um)")
                    # ax.set_xscale('symlog')
                    (line1,) = plt.plot(
                        (
                            self.reconstruction.positions0[:, 1]
                            - self.reconstruction.No // 2
                            + self.reconstruction.Np // 2
                        )
                        * self.reconstruction.dxo
                        * 1e6,
                        (
                            self.reconstruction.positions0[:, 0]
                            - self.reconstruction.No // 2
                            + self.reconstruction.Np // 2
                        )
                        * self.reconstruction.dxo
                        * 1e6,
                        "bo",
                        label="before correction",
                    )
                    (line2,) = plt.plot(
                        (
                            self.reconstruction.positions[:, 1]
                            - self.reconstruction.No // 2
                            + self.reconstruction.Np // 2
                        )
                        * self.reconstruction.dxo
                        * 1e6,
                        (
                            self.reconstruction.positions[:, 0]
                            - self.reconstruction.No // 2
                            + self.reconstruction.Np // 2
                        )
                        * self.reconstruction.dxo
                        * 1e6,
                        "yo",
                        label="after correction",
                    )
                    # plt.xlabel('(um))')
                    # plt.ylabel('(um))')
                    # plt.show()
                    plt.legend(handles=[line1, line2])
                    plt.tight_layout()
                    # plt.show(block=False)

                    figure2, ax2 = plt.subplots(
                        1, 1, num=103, squeeze=True, clear=True, figsize=(5, 5)
                    )
                    ax2.set_title("Displacement")
                    ax2.set_xlabel("(um)")
                    ax2.set_ylabel("(um)")
                    plt.plot(
                        self.D[:, 1] * self.reconstruction.dxo * 1e6,
                        self.D[:, 0] * self.reconstruction.dxo * 1e6,
                        "o",
                    )
                    # ax.set_xscale('symlog')
                    plt.tight_layout()
                    # plt.show(block=False)

                    # elif np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
                    figure.show()
                    figure2.show()
                    figure.canvas.draw()
                    figure.canvas.flush_events()
                    figure2.canvas.draw()
                    figure2.canvas.flush_events()
                    # self.showReconstruction(loop)
            # print('iteration:%i' %len(self.reconstruction.error))
            # print('runtime:')
            # print('error:')

        # Dump each iteration the current object
        if self.params.dump_obj:
            folder_path = "dumps"
            if loop == 0:
                if not os.path.exists(folder_path):
                    # Create the folder
                    os.makedirs(folder_path)
                    print(f"Folder '{folder_path}' created.")
                else:
                    print(f"Folder '{folder_path}' already exists.")

            filename = "obj_dump_" + str(loop) + ".h5py"
            import h5py

            file_path = os.path.join(folder_path, filename)
            with h5py.File(file_path, "w") as hdf:
                obj = asNumpyArray(self.reconstruction.object)
                hdf.create_dataset("Object", data=obj)

    def positionCorrection(self, objectPatch, positionIndex, sy, sx):
        r"""
        Estimate the position correction for one scan position.

        The current object patch is compared with the corresponding region of the
        reconstructed object using cross-correlation. For small search radii,
        shifted object patches are evaluated directly. For larger search radii,
        the cross-correlation is calculated in the Fourier domain.

        A correlation-weighted displacement estimate is obtained from the tested
        shifts. Conceptually,

        $$
        g_x = \beta \sum_k \frac{C_k-\bar{C}}{\|O_{\mathrm{patch}}\|_2^2}\Delta x_k
        $$

        and similarly for $g_y$, where $C_k$ is the correlation obtained for the
        candidate shift $(\Delta y_k,\Delta x_k)$.

        The estimated correction is scaled by the position-correction feedback
        factor and accumulated into the position search direction used by
        `positionCorrectionUpdate()`.

        Args:
            objectPatch (ndarray):
                Current reconstructed object patch for the scan position.
            positionIndex (int):
                Index of the current scan position.
            sy (slice):
                Row slice locating the current object patch in the full object.
            sx (slice):
                Column slice locating the current object patch in the full object.

        Returns:
            ndarray:
                Estimated position correction ``[delta_y, delta_x]`` in pixels.
                Returns zeros before position correction becomes active.
        """

        xp = getArrayModule(objectPatch)
        if len(self.reconstruction.error) > self.startAtIteration:
            self.logger.debug("Calculating position correction")
            # position gradients
            # shiftedImages = xp.zeros((self.rowShifts.shape + objectPatch.shape))
            cc = xp.zeros((len(self.rowShifts), 1))

            # use the real-space object (FFT for FPM)
            O = self.reconstruction.object
            Opatch = objectPatch
            if self.experimentalData.operationMode == "FPM":
                O = fft2c(self.reconstruction.object)
                Opatch = fft2c(objectPatch)

            if self.params.positionCorrectionSwitch_radius < 2:
                # do the direct one as it's a bit faster

                for shifts in range(len(self.rowShifts)):
                    tempShift = xp.roll(Opatch, self.rowShifts[shifts], axis=-2)
                    # shiftedImages[shifts, ...] = xp.roll(tempShift, self.colShifts[shifts], axis=-1)
                    shiftedImages = xp.roll(tempShift, self.colShifts[shifts], axis=-1)
                    cc[shifts] = xp.squeeze(
                        xp.sum(shiftedImages.conj() * O[..., sy, sx], axis=(-2, -1))
                    )
                    del tempShift, shiftedImages
                    betaGrad = 1000
                    r = 3
            else:
                # print('doing FT position correction')
                ss = slice(
                    -self.params.positionCorrectionSwitch_radius,
                    self.params.positionCorrectionSwitch_radius + 1,
                )
                rowShifts, colShifts = xp.mgrid[ss, ss]
                self.rowShifts = rowShifts.flatten()
                self.colShifts = colShifts.flatten()
                FT_O = xp.fft.fft2(O[..., sy, sx] - O[..., sy, sx].mean())
                FT_Op = xp.fft.fft2(Opatch - O.mean())
                xcor = xp.fft.ifft2(FT_O * FT_Op.conj())
                xcor = abs(xp.fft.fftshift(xcor))
                N = xcor.shape[-1]
                sy = slice(
                    N // 2 - self.params.positionCorrectionSwitch_radius,
                    N // 2 + self.params.positionCorrectionSwitch_radius + 1,
                )
                xcor = xcor[..., sy, sy]
                cc = xcor.flatten()
                betaGrad = 5
                r = 10
                # dy, dx = xp.unravel_index(xp.argmax(xcor), xcor.shape)
                # dx = dx.get()
            # truncated cross - correlation
            # cc = xp.squeeze(xp.sum(shiftedImages.conj() * self.reconstruction.object[..., sy, sx], axis=(-2, -1)))
            cc = abs(cc)

            normFactor = xp.sum(Opatch.conj() * Opatch, axis=(-2, -1)).real
            grad_x = betaGrad * xp.sum(
                (cc.T - xp.mean(cc)) / normFactor * xp.array(self.colShifts)
            )
            grad_y = betaGrad * xp.sum(
                (cc.T - xp.mean(cc)) / normFactor * xp.array(self.rowShifts)
            )
            # r = np.clip(self.params.positionCorrectionSwitch_radius//5, 3, self.reconstruction.Np//10) # maximum shift in pixels?

            if abs(grad_x) > r:
                grad_x = r * grad_x / abs(grad_x)
            if abs(grad_y) > r:
                grad_y = r * grad_y / abs(grad_y)
            grad_y = asNumpyArray(grad_y)
            grad_x = asNumpyArray(grad_x)
            delta_p = self.daleth * np.array([grad_y, grad_x])
            self.D[positionIndex, :] = delta_p + self.beth * self.D[positionIndex, :]
            return delta_p
        return np.zeros(2)

    def position_update_to_change_in_z(self, loop):
        r"""
        Map the global scaling of corrected scan positions to an update of the
        propagation distance.

        The corrected and original encoder positions are centered and their
        relative spatial scale is estimated as

        $$
        s = \frac{\sigma_{\mathrm{corrected}}}{\sigma_{\mathrm{original}}}
        $$

        A corresponding target propagation distance is estimated as

        $$
        z_{\mathrm{target}} = \frac{z}{s}
        $$

        The distance update is filtered through an Adam optimizer before being
        applied to `reconstruction.zo`. After updating the propagation distance,
        the corrected scan positions are rescaled around their center so that the
        global scale change is transferred from the position correction to the
        propagation distance.

        Args:
            loop (int):
                Current reconstruction iteration.

        Notes:
            This method is used together with position correction and is called
            periodically when `map_position_to_z_change` is enabled.

            The method requires JAX, which is imported when the function is
            called.
        """
        import jax
        from jax.experimental import optimizers

        if not hasattr(self, "optlib"):
            self.i_z_optimizer = 0
            # from itertools import count
            # count
            op_init, op_update, op_get = optimizers.adam(3e-3)
            state = op_init(self.reconstruction.zo)
            self.optlib = {"op_update": op_update, "op_get": op_get, "state": state}
        else:
            state = self.optlib["state"]
            op_get = self.optlib["op_get"]
            op_update = self.optlib["op_update"]

        X0 = self.reconstruction.encoder_corrected
        Y0 = self.experimentalData.encoder
        msqdisplacement = np.linalg.norm(1e6 * X0 - 1e6 * Y0)

        # center both
        X0 = X0 - X0.mean(axis=0, keepdims=True)
        Y0 = Y0 - Y0.mean(axis=0, keepdims=True)

        # now, find the scaling with respect to the original one
        factor = np.std(X0) / np.std(Y0)

        # update z
        new_z = self.reconstruction.zo / factor
        step = new_z - self.reconstruction.zo
        self.logger.info(f"Naive estimate of new z: {new_z:.3f}, stepsize {step:.3f}")
        step = 5 * step
        # check if the thing should be updated.
        if abs(step) < 1e-4:  # if it's too small, just truncate it,
            # it may be that the distance changed due to some other update.
            # Take that into account as if we don't the steps will be super large.
            self.i_z_optimizer += 1
            step = self.reconstruction.zo - op_get(state)
            self.optlib["state"] = op_update(self.i_z_optimizer, -step, state)

            self.logger.info("Skipping update as step is too small")
            # as we're only updating it for sake of good measure, we don't have to update anything else.
            return
        # now, as we're actually updating, we can increase the step
        self.i_z_optimizer += 1
        self.optlib["state"] = op_update(self.i_z_optimizer, -step, state)
        # get the new value
        z_new = float(jax.device_get(op_get(self.optlib["state"])))

        self.logger.info(f"Loop: {loop} step: {step}")
        self.logger.info(
            f"old z: {self.reconstruction.zo:.3f}\n new z calculated: {z_new:.3f}\n diff: {self.reconstruction.zo - z_new}\n"
        )
        # scale the coordinates accordingly
        factor = self.reconstruction.zo / z_new
        self.reconstruction.zo = z_new
        new_encoder = self.reconstruction.encoder_corrected.copy()
        new_encoder -= new_encoder.mean(axis=0, keepdims=True)
        new_encoder /= factor  # this should be the correct one!

        new_encoder += self.experimentalData.encoder.mean(axis=0, keepdims=True)

        msqdisplacement_a = np.linalg.norm(
            1e6 * new_encoder - self.experimentalData.encoder * 1e6
        )

        self.reconstruction.encoder_corrected = new_encoder
        self.logger.info(
            f"Mean square displacement: before: {msqdisplacement:.3f} after: {msqdisplacement_a:.3f}"
        )

    def positionCorrectionUpdate(self):
        r"""
        Apply the accumulated position corrections to the scan coordinates.

        Position corrections estimated by `positionCorrection()` are stored in
        `D` in pixel units and are applied after the position-correction warm-up.

        For conventional ptychography, the corrected encoder coordinates are
        updated as

        $$
        \mathbf{r}_j^{\mathrm{new}} = \mathbf{r}_j^{\mathrm{old}} - \alpha D_j d_{xo}
        $$

        where $\alpha$ is the adaptive step size and $d_{xo}$ is the object-plane
        pixel size. The corrected scan grid is then recentered to the mean of the
        original encoder positions.

        For Fourier ptychography, the corrected Fourier-space positions are
        converted back to illumination coordinates using the inverse FPM
        illumination geometry:

        $$
        \mathbf{r} = \operatorname{sign}(c)\frac{\mathbf{k}z_{\mathrm{LED}}}{\sqrt{c^2-k_x^2-k_y^2}}
        $$

        with

        $$
        c = -\frac{N_p d_{xo}}{\lambda}
        $$

        Notes:
            The method updates `reconstruction.encoder_corrected` in place and is
            called when position correction is enabled.
        """
        
        # fit the scaling out, to put in the z
        if len(self.reconstruction.error) > self.startAtIteration:
            self.logger.info("Updating positions")

            # update positions
            if self.experimentalData.operationMode == "FPM":
                conv = (
                    -(1 / self.reconstruction.wavelength)
                    * self.reconstruction.dxo
                    * self.reconstruction.Np
                )
                z = self.reconstruction.zled
                k = (
                    self.reconstruction.positions
                    - self.adaptStep * self.D
                    - self.reconstruction.No // 2
                    + self.reconstruction.Np // 2
                )
                self.reconstruction.encoder_corrected = (
                    np.sign(conv)
                    * k
                    * z
                    / (np.sqrt(conv**2 - k[:, 0] ** 2 - k[:, 1] ** 2))[..., None]
                )
            else:
                new_encoder = (
                    self.reconstruction.encoder_corrected
                    - self.adaptStep * self.D * self.reconstruction.dxo
                )
                new_encoder = new_encoder - new_encoder.mean(axis=0, keepdims=True)
                new_encoder = new_encoder + self.experimentalData.encoder.mean(
                    axis=0, keepdims=True
                )

                self.reconstruction.encoder_corrected = new_encoder
                self.logger.info(
                    f"Average update size: {abs(self.D).mean():.2f} pixels"
                )

    def applyConstraints(self, loop):
        """
        Apply enabled reconstruction constraints after an iteration.

        This method acts as the central dispatcher for object, probe, position,
        and autofocus constraints selected through `Params`. Depending on the
        active switches, it may apply regularization, probe normalization,
        modal orthogonalization, probe-boundary constraints, smoothing,
        amplitude constraints, spectral coupling, position correction, or
        autofocus updates.

        Constraints are applied sequentially in the order defined by this method,
        so enabling multiple constraints may cause later operations to act on the
        result of earlier ones.

        Args:
            loop (int):
                Current reconstruction iteration.

        Raises:
            NotImplementedError:
                If PSD estimation is enabled.
        """
        # dirks additions, untested
        if self.params.l2reg:
            #     turns down areas that are not updated. Similar to an
            # l2 regularizer
            self.reconstruction.object *= 1 - self.params.l2reg_object_aleph
            self.reconstruction.probe *= 1 - self.params.l2reg_probe_aleph

        # enforce empty beam constraint
        if self.params.modulusEnforcedProbeSwitch:
            self.modulusEnforcedProbe()

        if self.params.orthogonalizationSwitch:
            if np.mod(loop, self.params.orthogonalizationFrequency) == 0:
                self.orthogonalization()

        # probe normalization to measured PSD todo: check for multiwave and multi object states
        if self.params.probePowerCorrectionSwitch:
            self.reconstruction.probe = (
                self.reconstruction.probe
                / np.sqrt(
                    np.sum(self.reconstruction.probe * self.reconstruction.probe.conj())
                )
                * self.experimentalData.maxProbePower
            )
        if self.params.probeSpectralPowerCorrectionSwitch:
            for wl in range(self.reconstruction.probe.shape[0]):
                self.reconstruction.probe[wl, ...] *= (
                    self.experimentalData.maxProbePower
                    * self.experimentalData.spectralPower[wl]
                    / np.sqrt(
                        np.sum(
                            self.reconstruction.probe[wl, ...]
                            * self.reconstruction.probe[wl, ...].conj()
                        )
                    )
                )

        if (
            self.params.comStabilizationSwitch is not None
            and self.params.comStabilizationSwitch is not False
        ):
            if loop % int(self.params.comStabilizationSwitch) == 0:
                self.comStabilization()

        if self.params.PSDestimationSwitch:
            raise NotImplementedError()

        if self.params.probeBoundary:
            self.reconstruction.probe *= self.probeWindow

        if self.params.absorbingProbeBoundary:
            if self.experimentalData.operationMode == "FPM":
                self.absorbingProbeBoundaryAleph = 1

            self.reconstruction.probe = (
                (1 - self.params.absorbingProbeBoundaryAleph)
                * self.reconstruction.probe
                + self.params.absorbingProbeBoundaryAleph
                * self.reconstruction.probe
                * self.probeWindow
            )

            # experimental: also apply in fourier space
            # self.reconstruction.probe = ifft2c(fft2c(self.reconstruction.probe)*self.probeWindow)

        # Todo: objectSmoothenessSwitch,probeSmoothenessSwitch,
        if self.params.probeSmoothenessSwitch:
            self.reconstruction.probe = smooth_amplitude(
                self.reconstruction.probe,
                self.params.probeSmoothenessWidth,
                self.params.probeSmoothnessAleph,
            )

        if self.params.objectSmoothenessSwitch:
            self.reconstruction.object = smooth_amplitude(
                self.reconstruction.object,
                self.params.objectSmoothenessWidth,
                self.params.objectSmoothnessAleph,
            )

        if self.params.absObjectSwitch:
            self.reconstruction.object = (
                1 - self.params.absObjectBeta
            ) * self.reconstruction.object + self.params.absObjectBeta * abs(
                self.reconstruction.object
            )

        if self.params.absProbeSwitch:
            self.reconstruction.probe = (
                1 - self.params.absProbeBeta
            ) * self.reconstruction.probe + self.params.absProbeBeta * abs(
                self.reconstruction.probe
            )

        # this is intended to slowly push non-measured object region to abs value lower than
        # the max abs inside object ROI allowing for good contrast when monitoring object
        if self.params.objectContrastSwitch:
            self.reconstruction.object = (
                0.995 * self.reconstruction.object
                + 0.005
                * np.mean(
                    abs(
                        self.reconstruction.object[
                            ..., self.monitor.objectROI[0], self.monitor.objectROI[1]
                        ]
                    )
                )
            )
        if self.params.couplingSwitch and self.reconstruction.nlambda > 1:
            self.reconstruction.probe[0] = (
                1 - self.params.couplingAleph
            ) * self.reconstruction.probe[
                0
            ] + self.params.couplingAleph * self.reconstruction.probe[1]
            for lambdaLoop in np.arange(1, self.reconstruction.nlambda - 1):
                self.reconstruction.probe[lambdaLoop] = (
                    1 - self.params.couplingAleph
                ) * self.reconstruction.probe[
                    lambdaLoop
                ] + self.params.couplingAleph * (
                    self.reconstruction.probe[lambdaLoop + 1]
                    + self.reconstruction.probe[lambdaLoop - 1]
                ) / 2

            self.reconstruction.probe[-1] = (
                1 - self.params.couplingAleph
            ) * self.reconstruction.probe[
                -1
            ] + self.params.couplingAleph * self.reconstruction.probe[-2]
        if self.params.binaryProbeSwitch:
            probePeakAmplitude = np.max(abs(self.reconstruction.probe))
            probeThresholded = self.reconstruction.probe.copy()
            probeThresholded[
                (
                    abs(probeThresholded)
                    < self.params.binaryProbeThreshold * probePeakAmplitude
                )
            ] = 0

            self.reconstruction.probe = (
                (1 - self.params.binaryProbeAleph) * self.reconstruction.probe
                + self.params.binaryProbeAleph * probeThresholded
            )

        if self.params.positionCorrectionSwitch:
            self.positionCorrectionUpdate()

        if (
            self.params.map_position_to_z_change
            and (loop % 5 == 1)
            and self.params.positionCorrectionSwitch
        ):
            self.position_update_to_change_in_z(loop)

        if self.params.TV_autofocus:
            merit, AOI_image, allmerits = self.reconstruction.TV_autofocus(
                self.params, loop=loop
            )
            self.monitor.update_focusing_metric(
                merit,
                AOI_image,
                metric_name=self.params.TV_autofocus_metric,
                allmerits=allmerits,
            )

        # if self.params.OPRP and loop % self.params.OPRP_tsvd_interval == 0:
        #     self.reconstruction.probe_storage.tsvd()

    def orthogonalization(self):
        r"""
        Orthogonalize mixed-state probe or object modes.

        For multiple probe modes, the reconstructed modes are transformed to an
        orthogonal modal basis independently for each wavelength and slice.
        Conceptually, the transformed modes are

        $$
        P'_k(x,y) = \sum_j U_{kj}P_j(x,y)
        $$

        where $U$ is the modal transformation returned by `orthogonalizeModes()`.

        The normalized modal eigenvalues are used to calculate the probe purity:

        $$
        \mu = \sqrt{\sum_k \tilde{\lambda}_k^2}
        $$

        where

        $$
        \tilde{\lambda}_k = \frac{\lambda_k}{\sum_j \lambda_j}
        $$

        are the normalized modal eigenvalues. A purity of one
        corresponds to a single dominant mode, while lower values indicate a
        stronger mixed-state contribution.

        If momentum acceleration is enabled, the same modal transformation is
        applied to the corresponding momentum and buffer arrays so that all
        reconstruction state variables remain expressed in the same modal basis.

        If only multiple object modes are present, the analogous procedure is
        applied to the object modes and `purityObject` is updated.

        Notes:
            Probe modes are orthogonalized independently for each wavelength and
            slice. The resulting probe purity is stored in
            `reconstruction.purityProbe` and appended to
            `reconstruction.purityProbeHist`.
        """
        xp = getArrayModule(self.reconstruction.probe)
        if self.reconstruction.npsm > 1:
            # orthogonalize the probe for each wavelength and each slice
            for id_l in range(self.reconstruction.nlambda):
                for id_s in range(self.reconstruction.nslice):
                    (
                        self.reconstruction.probe[id_l, 0, :, id_s, :, :],
                        self.normalizedEigenvaluesProbe,
                        self.MSPVprobe,
                    ) = orthogonalizeModes(
                        self.reconstruction.probe[id_l, 0, :, id_s, :, :],
                        method="snapShots",
                    )
                    # normalizedEigenvalues can live on either device, and is only
                    # npsm values long -- reducing it on the host is both cheaper
                    # than launching kernels for it and device-independent.
                    eigenvalues = asNumpyArray(self.normalizedEigenvaluesProbe)
                    self.reconstruction.purityProbe = float(
                        np.sqrt(np.sum(eigenvalues**2))
                    )
                    self.reconstruction.purityProbeHist.append(
                        self.reconstruction.purityProbe
                    )
                    # orthogonolize momentum operator
                    if self.params.momentumAcceleration:
                        # orthogonalize probe Buffer
                        p = self.reconstruction.probeBuffer[
                            id_l, 0, :, id_s, :, :
                        ].reshape((self.reconstruction.npsm, self.reconstruction.Np**2))
                        self.reconstruction.probeBuffer[id_l, 0, :, id_s, :, :] = (
                            xp.array(self.MSPVprobe) @ p
                        ).reshape(
                            (
                                self.reconstruction.npsm,
                                self.reconstruction.Np,
                                self.reconstruction.Np,
                            )
                        )
                        # orthogonalize probe momentum
                        p = self.reconstruction.probeMomentum[
                            id_l, 0, :, id_s, :, :
                        ].reshape((self.reconstruction.npsm, self.reconstruction.Np**2))
                        self.reconstruction.probeMomentum[id_l, 0, :, id_s, :, :] = (
                            xp.array(self.MSPVprobe) @ p
                        ).reshape(
                            (
                                self.reconstruction.npsm,
                                self.reconstruction.Np,
                                self.reconstruction.Np,
                            )
                        )

                        # if self.comStabilizationSwitch:
                        #     self.comStabilization()
            # self.reconstruction.probe_storage.push(self.reconstruction.probe, None, len(self.experimentalData.ptychogram), force=True)

        elif self.reconstruction.nosm > 1:
            # orthogonalize the object for each wavelength and each slice
            for id_l in range(self.reconstruction.nlambda):
                for id_s in range(self.reconstruction.nslice):
                    (
                        self.reconstruction.object[id_l, :, 0, id_s, :, :],
                        self.normalizedEigenvaluesObject,
                        self.MSPVobject,
                    ) = orthogonalizeModes(
                        self.reconstruction.object[id_l, :, 0, id_s, :, :],
                        method="snapShots",
                    )
                    eigenvalues = asNumpyArray(self.normalizedEigenvaluesObject)
                    self.reconstruction.purityObject = float(
                        np.sqrt(np.sum(eigenvalues**2))
                    )

                    # orthogonolize momentum operator
                    if self.params.momentumAcceleration:
                        # orthogonalize object Buffer
                        p = self.reconstruction.objectBuffer[
                            id_l, :, 0, id_s, :, :
                        ].reshape((self.reconstruction.nosm, self.reconstruction.No**2))
                        self.reconstruction.objectBuffer[id_l, :, 0, id_s, :, :] = (
                            xp.array(self.MSPVobject) @ p
                        ).reshape(
                            (
                                self.reconstruction.nosm,
                                self.reconstruction.No,
                                self.reconstruction.No,
                            )
                        )
                        # orthogonalize object momentum
                        p = self.reconstruction.objectMomentum[
                            id_l, :, 0, id_s, :, :
                        ].reshape((self.reconstruction.nosm, self.reconstruction.No**2))
                        self.reconstruction.objectMomentum[id_l, :, 0, id_s, :, :] = (
                            xp.array(self.MSPVobject) @ p
                        ).reshape(
                            (
                                self.reconstruction.nosm,
                                self.reconstruction.No,
                                self.reconstruction.No,
                            )
                        )

        else:
            pass

    def comStabilization(self):
        r"""
        Stabilize the probe center of mass to suppress translational drift.

        Ptychographic reconstruction contains a translational ambiguity that can
        allow the reconstructed probe and object to drift within their numerical
        arrays without strongly affecting the data consistency. This method
        estimates the probe center relative to the center of the reconstruction
        window and shifts the reconstruction when the displacement exceeds
        approximately one pixel.

        The current probe center is estimated as

        $$
        x_c = \frac{\sum_{x,y}X_p(x,y)A(x,y)}{d_{xp}\sum_{x,y}A(x,y)}
        $$

        and

        $$
        y_c = \frac{\sum_{x,y}Y_p(x,y)A(x,y)}{d_{xp}\sum_{x,y}A(x,y)}
        $$

        where $A(x,y)$ is the probe amplitude and $d_{xp}$ is the probe-plane
        pixel size. The resulting coordinates are rounded to integer pixel
        shifts.

        If the probe center is displaced by more than approximately one pixel,
        both the probe and object are shifted by `(-yc, -xc)` to recenter the
        reconstruction while preserving their relative spatial registration.

        When momentum acceleration is enabled, the corresponding probe and object
        momentum and buffer arrays are shifted by the same amount.

        Notes:
            For multislice reconstruction, the last probe slice is used to
            estimate the center.
        """
        self.logger.info("Doing probe com stabilization")
        xp = getArrayModule(self.reconstruction.probe)
        # calculate center of mass of the probe (for multislice cases, the probe for the last slice is used)
        P2 = xp.sum(
            abs(self.reconstruction.probe[:, :, :, -1, ...]) ** 2, axis=(0, 1, 2)
        )
        P2 = abs(self.reconstruction.probe[0, 0, 0, -1]) ** 2
        demon = xp.sum(P2) * self.reconstruction.dxp
        xc = int(
            xp.around(xp.sum(xp.array(self.reconstruction.Xp, xp.float32) * P2) / demon)
        )
        yc = int(
            xp.around(xp.sum(xp.array(self.reconstruction.Yp, xp.float32) * P2) / demon)
        )
        # print('Center of mass:', yc, xc)
        # shift only if necessary
        if xc**2 + yc**2 > 1:
            # self.reconstruction.probe_storage._push_hard(self.reconstruction.probe, 100)
            # self.reconstruction.probe_storage.roll(-yc, -xc)

            # shift probe
            self.reconstruction.probe = xp.roll(
                self.reconstruction.probe, (-yc, -xc), axis=(-2, -1)
            )
            # for k in xp.arange(self.reconstruction.npsm):
            #     self.reconstruction.probe[:, :, k, -1, ...] = \
            #         xp.roll(self.reconstruction.probe[:, :, k, -1, ...], (-yc, -xc), axis=(-2, -1))
            #     # for mPIE
            if self.params.momentumAcceleration:
                self.reconstruction.probeMomentum = xp.roll(
                    self.reconstruction.probeMomentum, (-yc, -xc), axis=(-2, -1)
                )
                self.reconstruction.probeBuffer = xp.roll(
                    self.reconstruction.probeBuffer, (-yc, -xc), axis=(-2, -1)
                )

            # shift object
            self.reconstruction.object = xp.roll(
                self.reconstruction.object, (-yc, -xc), axis=(-2, -1)
            )
            # for mPIE
            if self.params.momentumAcceleration:
                self.reconstruction.objectMomentum = xp.roll(
                    self.reconstruction.objectMomentum, (-yc, -xc), axis=(-2, -1)
                )
                self.reconstruction.objectBuffer = xp.roll(
                    self.reconstruction.objectBuffer, (-yc, -xc), axis=(-2, -1)
                )

    def modulusEnforcedProbe(self):
        r"""
        Constrain the reconstructed probe using a measured empty-beam intensity.

        The current probe is propagated to the detector plane, where its
        estimated intensity is compared with `experimentalData.emptyBeam`.

        The detector-plane probe is scaled by

        $$
        f(x,y) = \sqrt{\frac{I_{\mathrm{empty}}(x,y)}{I_{\mathrm{probe}}(x,y)+\epsilon}}
        $$

        and updated as

        $$
        \Psi_{\mathrm{updated}}(x,y) = \Psi(x,y)f(x,y)
        $$

        so that the measured detector-plane amplitude is enforced while the
        current phase estimate is retained.

        The constrained field is then propagated back to the probe plane and
        stored in `reconstruction.probe`.

        If `FourierMaskSwitch` is enabled, the modulus constraint is applied only
        inside the active Fourier mask, while the detector-plane field outside
        the mask is left unchanged.

        Notes:
            This constraint requires `experimentalData.emptyBeam`, containing a
            measured detector intensity of the illumination without the sample.
        """
        xp = getArrayModule(self.reconstruction.esw)
        self.reconstruction.esw = self.reconstruction.probe
        self.object2detector()

        if self.params.FourierMaskSwitch:
            self.reconstruction.ESW = (
            self.reconstruction.ESW
            * xp.sqrt(
                self.experimentalData.emptyBeam
                / (
                    1e-10
                    + xp.sum(
                        xp.abs(self.reconstruction.ESW) ** 2,
                        axis=(0, 1, 2, 3),
                    )
                )
            )
            * self.experimentalData.W
            + self.reconstruction.ESW * (1 - self.experimentalData.W)
            )
        else:
            self.reconstruction.ESW = self.reconstruction.ESW * np.sqrt(
                self.experimentalData.emptyBeam
                / (1e-10 + xp.sum(abs(self.reconstruction.ESW) ** 2, axis=(0, 1, 2, 3)))
            )

        self.detector2object()

        if self.params.OPRP:
            pass
            # self.probes.append(self.reconstruction.esw.reshape(-))

        self.reconstruction.probe = self.reconstruction.esw

    def adaptiveDenoising(self):
        r"""
        Apply an adaptive amplitude-domain noise-floor correction.

        The measured and estimated detector intensities are converted to
        amplitudes.

        A global noise level is estimated from their mean amplitude difference:

        $$
        n = \left|\left\langle A_{\mathrm{meas}} - A_{\mathrm{est}} \right\rangle\right|
        $$

        The estimated noise floor is subtracted from the measured amplitude and
        negative values are clipped to zero:

        $$
        A'_{\mathrm{meas}} = \max(A_{\mathrm{meas}} - n, 0)
        $$

        The corrected measured intensity is then reconstructed as

        $$
        I'_{\mathrm{meas}} = \left(A'_{\mathrm{meas}}\right)^2
        $$

        Notes:
            This method modifies `reconstruction.Imeasured` in place and is
            applied before the detector-plane intensity constraint.
        """
        # figure out wether or not to use the GPU
        xp = getArrayModule(self.reconstruction.esw)

        Ameasured = self.reconstruction.Imeasured**0.5
        Aestimated = xp.abs(self.reconstruction.Iestimated) ** 0.5

        noise = xp.abs(xp.mean(Ameasured - Aestimated))

        Ameasured = Ameasured - noise
        Ameasured[Ameasured < 0] = 0
        self.reconstruction.Imeasured = Ameasured**2

    def z_update(self, stepsize=0.01, roi_bounds=[0.3, 0.7], d=10):
        """
        Update Z based on TV
        :param stepsize:
        :return:
        """
        self.reconstruction.TV_autofocus()

    def objectPatchUpdate_TV(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        r"""
        Apply total-variation regularization to the engine-specific object update.

        The data-driven object update is first calculated using the current
        engine's `objectPatchUpdate()` implementation:

        $$
        O_{\mathrm{data}} = U_{\mathrm{engine}}(O_j,\Delta\Psi_j)
        $$

        where $U_{\mathrm{engine}}$ denotes the object-update rule implemented by
        the active reconstruction engine. For example, `ePIE` uses the standard
        ePIE update, while `mPIE` uses its regularized mPIE/rPIE update.

        A total-variation regularization term is then added:

        $$
        O'_j = O_{\mathrm{data}} + \lambda\beta_O G_{\mathrm{TV}}(O_j)
        $$

        where $G_{\mathrm{TV}}(O_j)$ is calculated by `grad_TV()` using
        `epsilon=1e-2`, $\beta_O$ is the engine's object update step size, and
        $\lambda$ is controlled by `params.objectTVregStepSize`.

        This design keeps the TV regularization independent of the underlying
        reconstruction engine, allowing different engines to retain their native
        object-update rules while sharing the same TV regularizer.

        The activation and application frequency of this update are controlled by
        `params.objectTVregSwitch` and `params.objectTVfreq` in the reconstruction
        loop.

        Args:
            objectPatch (ndarray):
                Current object patch at the active scan position.
            DELTA (ndarray):
                Exit-wave correction, typically
                `reconstruction.eswUpdate - reconstruction.esw`.

        Returns:
            ndarray:
                Engine-specific object update with the additional TV
                regularization term.
        """

        #xp = getArrayModule(objectPatch)
        #frac = self.reconstruction.probe.conj() / xp.max(
        #    xp.sum(xp.abs(self.reconstruction.probe) ** 2, axis=(0, 1, 2, 3))
        #)

        # gradient = xp.gradient(objectPatch, axis=(4, 5))
        #
        # # norm = xp.abs(gradient[0] + gradient[1]) ** 2
        # norm = (gradient[0] + gradient[1]) ** 2
        # temp = [gradient[0] / xp.sqrt(norm + epsilon), gradient[1] / xp.sqrt(norm + epsilon)]
        # TV_update = divergence(temp)
        
        TV_update = grad_TV(objectPatch, epsilon=1e-2)

        updated_object = self.objectPatchUpdate(objectPatch, DELTA)

        lam = self.params.objectTVregStepSize

        return updated_object + lam * self.betaObject * TV_update
