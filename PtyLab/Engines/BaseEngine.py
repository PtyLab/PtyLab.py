import logging
import warnings

import numpy as np
from matplotlib import pyplot as plt

from PtyLab import Operators
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import (
    Reconstruction,
    calculate_pixel_positions,
)
from PtyLab.Regularizers import grad_TV

# PtyLab imports
from PtyLab.utils.gpuUtils import (
    asNumpyArray,
    getArrayModule,
    isGpuArray,
    transfer_fields_to_cpu,
    transfer_fields_to_gpu,
)
from PtyLab.utils.utils import circ, fft2c, ifft2c, orthogonalizeModes

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
    Smooth the amplitude of a field. Optional phase can be smoothed as well.
    Parameters
    ----------
    field
    width
    aleph
    amplitude_only

    Returns
    -------

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
    Common properties that are common for all reconstruction Engines are defined here.

    Unless you are testing the code, there's hardly any need to create this object. For your own implementation,
    inherit from this object

    """

    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        # These statements don't copy any data, they just keep a reference to the object
        self.betaObject = 0.25
        self.reconstruction: Reconstruction = reconstruction
        self.experimentalData = experimentalData
        self.params = params
        self.monitor = monitor
        self.monitor.reconstruction = reconstruction

        # datalogger
        self.logger = logging.getLogger("BaseEngine")

    def _prepareReconstruction(self):
        """
        Initialize everything that depends on user changeable attributes.
        :return:
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
        set constrained-pixel-sum constraint:
        -save measured diffraction patterns into ptychograpmDownsampled
        -pad the probe (useful when having a pre-calibrated probe)
        -update the coordinates
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
        """Update the experimentalData if necessary"""
        self.experimentalData = experimentalData
        if reconstruction is not None:
            self.reconstruction = reconstruction

    def _initializePCParameters(self):
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
        initialize all kinds of errors:
        detectorError is a matrix calculated at each iteration (numFrames,Nd,Nd);
        errorAtPos sums over detectorError at each iteration, (numFrames,1);
        reconstruction.error sums over errorAtPos, one number at each iteration;
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
        if self.params.probePowerCorrectionSwitch:
            self.reconstruction.probe = (
                self.reconstruction.probe
                / np.sqrt(
                    np.sum(self.reconstruction.probe * self.reconstruction.probe.conj())
                )
                * self.experimentalData.maxProbePower
            )

    def _probeWindow(self):
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
        Set object/probe ROI for monitoring
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

                self.monitor.objectROI = [
                    slice(
                        max(0, yc - ry // 2), min(self.reconstruction.No, yc + ry // 2)
                    ),
                    slice(
                        max(0, xc - rx // 2), min(self.reconstruction.No, xc + rx // 2)
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
        checks miscellaneous quantities specific certain Engines
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
        shift arrays to accelerate fft
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
                if hasattr(self.experimentalData, "w"):
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
                self.params.fftshiftFlag = 0

    def _move_data_to_gpu(self):
        """
        Move the data to the GPU, called when the gpuSwitch is on.
        :return:
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
                    "Yd" "dxd",
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
        Move the data to the CPU, called when the gpuSwitch is off.
        :return:
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
        fields_to_transfer = ["theta", "probeWindow"]
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

        if experimentalData is not None:
            if not isinstance(experimentalData, ExperimentalData):
                raise TypeError("Experimental data should be of class ExperimentalData")
            self.experimentalData = experimentalData

    def changeOptimizable(self, optimizable: Reconstruction):

        if optimizable is not None:
            if not isinstance(optimizable, Reconstruction):
                raise TypeError(
                    f"Argument should be an subclass of Reconstruction, but it is {type(optimizable)}"
                )
            self.reconstruction = optimizable

    def convert2single(self):
        """
        Convert the datasets to single precision. Matches: convert2single.m
        :return:
        """
        self.dtype_complex = np.complex64
        self.dtype_real = np.float32
        self._match_dtypes_complex()
        self._match_dtypes_real()

    def _match_dtypes_complex(self):
        raise NotImplementedError()

    def _match_dtypes_real(self):
        raise NotImplementedError()

    def object2detector(self, esw=None):
        """
        Implements object2detector.m. Modifies esw in-place
        :return:
        """
        if esw is None:
            # todo: check this, it seems weird to store it in self.esw
            esw = self.reconstruction.esw
        self.esw, self.reconstruction.ESW = Operators.Operators.object2detector(
            esw, self.params, self.reconstruction
        )

    def detector2object(self, ESW=None):
        """
        Propagate the ESW to the object plane (in-place).

        Matches: detector2object.m
        :return:
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

    def exportOjb(self, extension=".mat"):
        """
        Export the object.

        If extension == '.mat', export to matlab file.
        If extension == '.png', export to a png file (with amplitude-phase)

        Matches: exportObj (except for the PNG)

        :return:
        """
        raise NotImplementedError()

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
        """
        Calculate probe beam width (Full width half maximum)
        :return:
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

        c = 2 * xp.sqrt(
            2 * xp.log(2)
        )  # constant for converting variance to FWHM (see e.g. https://en.wikipedia.org/wiki/Full_width_at_half_maximum)

        self.reconstruction.beamWidthX = asNumpyArray(c * np.sqrt(xVariance))
        self.reconstruction.beamWidthY = asNumpyArray(c * np.sqrt(yVariance))

        return self.reconstruction.beamWidthY, self.reconstruction.beamWidthX

    def getOverlap(self, ind1, ind2):
        """
        Calculate linear and area overlap between two scan positions indexed ind1 and ind2
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
                    Q**2 * np.exp(-1.0j * 2 * np.pi * (Fx * sx + Fy * sy)),
                    axis=(-1, -2),
                )
            )
            / np.sum(abs(Q) ** 2, axis=(-1, -2)),
            axis=0,
        )

    def getErrorMetrics(self):
        """
        matches getErrorMetrics.m
        :return:
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
        """
        Root mean square deviation between ptychogram and intensity estimate
        :param positionIndex:
        :return:
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
        """Compute the projected intensity.
        Barebones, need to implement other methods
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

        elif self.params.intensityConstraint == "poission":
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
        """
        calculate the upsampled Imeasured from downsampled Imeasured that is actually measured.
        :param positionIndex: index for scan positions
        :return:
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
        Show the reconstruction process.
        :param loop: the iteration number
        :return:
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
                            0, ..., self.monitor.probeROI[0], self.monitor.probeROI[1]
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
        # TODO: print info

    def positionCorrection(self, objectPatch, positionIndex, sy, sx):
        """
        Modified from pcPIE. Position correction is done by using positionCorrection and positionCorrectionUpdate
        :param objectPatch:
        :param positionIndex:
        :param sy:
        :param sx:
        :return:
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
        """
        Update the z based on the position updates.
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

    def applyConstraints(self, loop):
        """
        Apply constraints.
        :param loop: loop number
        :return:
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
            ] + self.params.couplingAleph * self.reconstruction.probe[
                1
            ]
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
            ] + self.params.couplingAleph * self.reconstruction.probe[
                -2
            ]
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
        """
        Perform orthogonalization
        :return:
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
                    self.reconstruction.purityProbe = np.sqrt(
                        np.sum(self.normalizedEigenvaluesProbe**2)
                    )

                    # orthogonolize momentum operator
                    if self.params.momentumAcceleration:
                        # orthogonalize probe Buffer
                        p = self.reconstruction.probeBuffer[
                            id_l, 0, :, id_s, :, :
                        ].reshape(
                            (self.reconstruction.npsm, self.reconstruction.Np**2)
                        )
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
                        ].reshape(
                            (self.reconstruction.npsm, self.reconstruction.Np**2)
                        )
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
                    self.reconstruction.purityObject = np.sqrt(
                        np.sum(self.normalizedEigenvaluesObject**2)
                    )

                    # orthogonolize momentum operator
                    if self.params.momentumAcceleration:
                        # orthogonalize object Buffer
                        p = self.reconstruction.objectBuffer[
                            id_l, :, 0, id_s, :, :
                        ].reshape(
                            (self.reconstruction.nosm, self.reconstruction.No**2)
                        )
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
                        ].reshape(
                            (self.reconstruction.nosm, self.reconstruction.No**2)
                        )
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
        """
        Perform center of mass stabilization (center the probe)
        :return:
        """
        self.logger.info("Doing probe com stabilization")
        xp = getArrayModule(self.reconstruction.probe)
        # calculate center of mass of the probe (for multislice cases, the probe for the last slice is used)
        P2 = xp.sum(
            abs(self.reconstruction.probe[:, :, :, -1, ...]) ** 2, axis=(0, 1, 2)
        )
        P2 = abs(self.reconstruction.probe[0, 0, 0, -1])
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
        # propagate probe to detector
        xp = getArrayModule(self.reconstruction.esw)
        self.reconstruction.esw = self.reconstruction.probe
        self.object2detector()

        if self.params.FourierMaskSwitch:
            self.reconstruction.ESW = self.reconstruction.ESW * xp.sqrt(
                self.experimentalData.emptyBeam / 1e-10
                + xp.sum(xp.abs(self.reconstruction.ESW) ** 2, axis=(0, 1, 2, 3))
            ) * self.experimentalData.W + self.reconstruction.ESW * (
                1 - self.experimentalData.W
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
        """
        Use the difference of mean intensities between the low-resolution
        object estimate and the low-resolution raw data to estimate the
        noise floor to be clipped.
        :return:
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
        """
        Update the object patch with a TV regularization.

        :param objectPatch:
        :param DELTA:
        :return:
        """

        xp = getArrayModule(objectPatch)
        frac = self.reconstruction.probe.conj() / xp.max(
            xp.sum(xp.abs(self.reconstruction.probe) ** 2, axis=(0, 1, 2, 3))
        )

        # gradient = xp.gradient(objectPatch, axis=(4, 5))
        #
        # # norm = xp.abs(gradient[0] + gradient[1]) ** 2
        # norm = (gradient[0] + gradient[1]) ** 2
        # temp = [gradient[0] / xp.sqrt(norm + epsilon), gradient[1] / xp.sqrt(norm + epsilon)]
        # TV_update = divergence(temp)
        TV_update = grad_TV(objectPatch, epsilon=1e-2)
        lam = self.params.objectTVregStepSize
        return (
            objectPatch
            + self.betaObject * xp.sum(frac * DELTA, axis=(0, 2, 3), keepdims=True)
            + lam * self.betaObject * TV_update
        )
