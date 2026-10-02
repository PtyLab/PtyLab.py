import numpy as np
from matplotlib import pyplot as plt

try:
    import cupy as cp
except ImportError:
    # print("Cupy not available, will not be able to run GPU based computation")
    # Still define the name, we'll take care of it later but in this way it's still possible
    # to see that gPIE exists for example.
    cp = None

import logging
import sys

import tqdm

from PtyLab.Engines.BaseEngine import BaseEngine
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.Params.Params import Params

# PtyLab imports
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.utils.gpuUtils import asNumpyArray, getArrayModule
from PtyLab.utils.utils import fft2c, ifft2c

from copy import deepcopy

from PtyLab.Monitor.Monitor import DummyMonitor

from PtyLab.Engines.mPIE import mPIE
class purityPIE(mPIE):
    '''
    purityPIE extends mPIE-style reconstruction with purity-based axial calibration.
    '''
    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        super().__init__(
            reconstruction,
            experimentalData,
            params,
            monitor,
        )

        self.logger = logging.getLogger("purityPIE")
        self.logger.info("Successfully created purityPIE engine")
        self.logger.info(
            "Wavelength attribute: %s",
            self.reconstruction.wavelength,
        )

        self.name = "purityPIE"

    def reconstruct(
        self,
        experimentalData=None,
        reconstruction=None,
    ):
        """
        Run purity-based axial calibration.

        Each candidate propagation distance is reconstructed from the same
        initial state using `numIterations` mPIE-style iterations. The distance
        yielding the highest probe purity is selected, and the corresponding
        reconstruction state is retained.

        If `params.purityZScanPlot` is enabled, the resulting purity-versus-z
        curve is displayed after calibration.

        Args:
            experimentalData (ExperimentalData, optional):
                Experimental dataset to use. If provided, it replaces the
                currently attached dataset.
            reconstruction (Reconstruction, optional):
                Reconstruction state to calibrate. If provided, it replaces the
                currently attached reconstruction.

        Returns:
            tuple:
                `(best_z, best_purity)`, where `best_z` is the selected axial
                propagation distance and `best_purity` is the corresponding
                probe purity.
        """

        self.changeExperimentalData(experimentalData)
        self.changeOptimizable(reconstruction)

        self._prepareReconstruction()

        # Reset momentum buffers after preparation
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()

        # Purity-based axial calibration
        best_z, best_purity = self.purityZScan()

        # Optional scan visualization
        if self.params.purityZScanPlot:
            self.plotPurityZScan()

        # Return reconstruction data to CPU
        if self.params.gpuFlag:
            self.logger.info("switch to cpu")
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

        return best_z, best_purity

    def _createCandidateReconstruction(self, z):
        """
        Create an independent reconstruction state for one candidate z.

        The candidate inherits the user's reconstruction settings but is built
        from a fresh geometry at the requested propagation distance.
        """

        # Independent copies so one candidate cannot modify another
        candidate_data = deepcopy(self.experimentalData)
        candidate_params = deepcopy(self.params)

        # Set candidate geometry before creating Reconstruction
        candidate_data.zo = z

        candidate_reconstruction = Reconstruction(
            candidate_data,
            candidate_params,
        )

        # Copy user-defined reconstruction settings
        candidate_reconstruction.npsm = self.reconstruction.npsm
        candidate_reconstruction.nosm = self.reconstruction.nosm
        candidate_reconstruction.nlambda = self.reconstruction.nlambda
        candidate_reconstruction.nslice = self.reconstruction.nslice

        #candidate_reconstruction.No = self.reconstruction.No

        candidate_reconstruction.initialProbe = self.reconstruction.initialProbe
        candidate_reconstruction.initialObject = self.reconstruction.initialObject

        if hasattr(self.reconstruction, "initialProbe_filename"):
            candidate_reconstruction.initialProbe_filename = (
                self.reconstruction.initialProbe_filename
            )

        if hasattr(self.reconstruction, "initialObject_filename"):
            candidate_reconstruction.initialObject_filename = (
                self.reconstruction.initialObject_filename
            )

        # Fresh object/probe for this candidate geometry
        candidate_reconstruction.initializeObjectProbe()

        return (
            candidate_data,
            candidate_reconstruction,
            candidate_params,
        )

    def purityZScan(self):
        """
        Scan a fixed axial range and select the distance that maximizes
        reconstructed probe purity.

        Each candidate z is evaluated using an independent reconstruction
        state and a standard mPIE reconstruction.
        """

        if self.reconstruction.npsm < 2:
            raise ValueError(
                "Purity-based z scanning requires at least two probe modes "
                "(reconstruction.npsm >= 2)."
            )

        # ------------------------------------------------------------------
        # Scan grid
        # ------------------------------------------------------------------

        z0 = self.reconstruction.zo
        self.purityZInitialGuess = z0

        z_values = np.linspace(
            z0 - self.params.purityZScanRange,
            z0 + self.params.purityZScanRange,
            self.params.purityZScanPoints,
        )

        purity_values = []

        best_purity = -np.inf
        best_z = z0

        best_reconstruction = None

        # ------------------------------------------------------------------
        # Scan information
        # ------------------------------------------------------------------

        tqdm.tqdm.write("")
        tqdm.tqdm.write("Starting purity-based z scan")
        tqdm.tqdm.write(
            f"Initial z = {z0 * 1e3:.6f} mm | "
            f"range = ±{self.params.purityZScanRange * 1e6:.1f} µm | "
            f"points = {len(z_values)} | "
            f"iterations/z = {self.numIterations}"
        )
        tqdm.tqdm.write("")

        # ------------------------------------------------------------------
        # Candidate-z loop
        # ------------------------------------------------------------------

        for z_index, z in enumerate(z_values):

            dz_um = (z - z0) * 1e6

            tqdm.tqdm.write(
                f"[{z_index + 1}/{len(z_values)}] "
                f"Reconstructing z = {z * 1e3:.6f} mm "
                f"(Δz = {dz_um:+.2f} µm)"
            )

            # --------------------------------------------------------------
            # Create a completely independent reconstruction for this z
            # --------------------------------------------------------------
            #np.random.seed(0)
            (
                candidate_data,
                candidate_reconstruction,
                candidate_params,
            ) = self._createCandidateReconstruction(z)

            # Candidate reconstructions do not need their own GUI monitor
            candidate_monitor = DummyMonitor()

            # --------------------------------------------------------------
            # Run standard mPIE
            # --------------------------------------------------------------

            candidate_engine = mPIE(
                candidate_reconstruction,
                candidate_data,
                candidate_params,
                candidate_monitor,
            )

            # Match the purityPIE reconstruction settings
            candidate_engine.numIterations = self.numIterations

            candidate_engine.betaProbe = self.betaProbe
            candidate_engine.betaObject = self.betaObject

            candidate_engine.alphaProbe = self.alphaProbe
            candidate_engine.alphaObject = self.alphaObject

            candidate_engine.feedbackM = self.feedbackM
            candidate_engine.frictionM = self.frictionM

            candidate_engine.reconstruct()

            # --------------------------------------------------------------
            # Final modal decomposition and purity evaluation
            # --------------------------------------------------------------

            candidate_engine.orthogonalization()

            purity = float(
                np.asarray(
                    asNumpyArray(candidate_reconstruction.purityProbe)
                ).squeeze()
            )

            purity_values.append(purity)

            tqdm.tqdm.write(
                f"    Probe purity = {purity:.6f}"
            )

            # --------------------------------------------------------------
            # Update main monitor with the completed candidate state
            # --------------------------------------------------------------

            self.reconstruction.zo = z
            self.reconstruction.object = (
                asNumpyArray(candidate_reconstruction.object).copy()
            )
            self.reconstruction.probe = (
                asNumpyArray(candidate_reconstruction.probe).copy()
            )

            self.reconstruction.purityProbe = purity

            if hasattr(candidate_reconstruction, "error"):
                self.reconstruction.error = np.asarray(
                    asNumpyArray(candidate_reconstruction.error)
                ).copy()

            # Display only completed candidate reconstructions
            self.showReconstruction(
                self.numIterations - 1,
                force=True,
            )

            # --------------------------------------------------------------
            # Keep best candidate
            # --------------------------------------------------------------

            if purity > best_purity:

                best_purity = purity
                best_z = z

                # Candidate objects are independent, so keeping this reference
                # preserves the complete best reconstruction state.
                best_reconstruction = candidate_reconstruction

                tqdm.tqdm.write(
                    f"    NEW BEST: z = {best_z * 1e3:.6f} mm, "
                    f"purity = {best_purity:.6f}"
                )

            tqdm.tqdm.write("")

        # ------------------------------------------------------------------
        # Safety check
        # ------------------------------------------------------------------

        if best_reconstruction is None:
            raise RuntimeError(
                "Purity z scan did not produce a valid purity value."
            )

        # ------------------------------------------------------------------
        # Restore the best candidate into the original Reconstruction object
        #
        # Keep the original object identity because external scripts may still
        # hold a reference to self.reconstruction.
        # ------------------------------------------------------------------

        self.reconstruction.zo = best_z

        self.reconstruction.object = (
            asNumpyArray(best_reconstruction.object).copy()
        )

        self.reconstruction.probe = (
            asNumpyArray(best_reconstruction.probe).copy()
        )

        self.reconstruction.purityProbe = best_purity

        if hasattr(best_reconstruction, "encoder_corrected"):
            self.reconstruction.encoder_corrected = (
                asNumpyArray(
                    best_reconstruction.encoder_corrected
                ).copy()
            )

        if hasattr(best_reconstruction, "objectMomentum"):
            self.reconstruction.objectMomentum = (
                asNumpyArray(
                    best_reconstruction.objectMomentum
                ).copy()
            )

        if hasattr(best_reconstruction, "probeMomentum"):
            self.reconstruction.probeMomentum = (
                asNumpyArray(
                    best_reconstruction.probeMomentum
                ).copy()
            )

        if hasattr(best_reconstruction, "objectBuffer"):
            self.reconstruction.objectBuffer = (
                asNumpyArray(
                    best_reconstruction.objectBuffer
                ).copy()
            )

        if hasattr(best_reconstruction, "probeBuffer"):
            self.reconstruction.probeBuffer = (
                asNumpyArray(
                    best_reconstruction.probeBuffer
                ).copy()
            )

        if hasattr(best_reconstruction, "error"):
            self.reconstruction.error = np.asarray(
                asNumpyArray(best_reconstruction.error)
            ).copy()

        # ------------------------------------------------------------------
        # Store scan results
        # ------------------------------------------------------------------

        self.purityZValues = np.asarray(z_values)
        self.purityZMetrics = np.asarray(purity_values)

        self.purityBestZ = best_z
        self.purityBestValue = best_purity

        # Final monitor update with the selected best candidate
        self.showReconstruction(
            self.numIterations - 1,
            force=True,
        )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------

        tqdm.tqdm.write("=" * 60)
        tqdm.tqdm.write("Purity-based z scan finished")
        tqdm.tqdm.write(
            f"Best z      = {best_z * 1e3:.6f} mm"
        )
        tqdm.tqdm.write(
            f"Best Δz     = {(best_z - z0) * 1e6:+.2f} µm"
        )
        tqdm.tqdm.write(
            f"Best purity = {best_purity:.6f}"
        )
        tqdm.tqdm.write("=" * 60)

        return best_z, best_purity


        """
        Reset the reconstruction state for one candidate z.

        This implementation is intended for analytic initialization:
        initialProbe = "circ"
        initialObject = "ones"

        The object/probe are regenerated after setting the candidate z so that
        their physical sampling is consistent with the candidate geometry.
        Momentum and buffers are also reset.
        """
        if self.reconstruction.initialProbe != "circ":
            raise NotImplementedError(
                "Candidate-z reinitialization is currently implemented "
                "only for initialProbe='circ'."
            )

        if self.reconstruction.initialObject != "ones":
            raise NotImplementedError(
                "Candidate-z reinitialization is currently implemented "
                "only for initialObject='ones'."
            )

        # Set candidate geometry first
        self.reconstruction.zo = z

        # Rebuild object/probe on the candidate-z geometry
        self.reconstruction.initializeObjectProbe()

        self._initialProbePowerCorrection()

        # Reset momentum
        self.reconstruction.initializeObjectMomentum()
        self.reconstruction.initializeProbeMomentum()

        # Reset buffers
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()

        self._prepareReconstruction()

    def plotPurityZScan(self, show=True):
        """
        Plot probe purity as a function of axial distance.

        The axial coordinate is shown relative to the initial z guess used
        for the purity scan.

        Args:
            show (bool, optional):
                Display the figure immediately. Default is `True`.

        Returns:
            tuple:
                Matplotlib `(figure, axis)` objects.
        """

        if not hasattr(self, "purityZValues"):
            raise RuntimeError(
                "No purity z-scan results are available. "
                "Run purityZScan() first."
            )

        if not hasattr(self, "purityZInitialGuess"):
            raise RuntimeError(
                "Initial z guess is not available."
            )

        z_relative_um = (
            np.asarray(self.purityZValues)
            - self.purityZInitialGuess
        ) * 1e6

        purity = np.asarray(self.purityZMetrics)

        best_index = np.argmax(purity)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            z_relative_um,
            purity,
            marker="o",
            label="Probe purity",
        )

        # Initial z guess
        ax.axvline(
            0,
            linestyle="--",
            alpha=0.5,
            label="Initial z guess",
        )

        # Best z
        ax.plot(
            z_relative_um[best_index],
            purity[best_index],
            marker="*",
            markersize=12,
            linestyle="None",
            label=(
                f"Best Δz = {z_relative_um[best_index]:.1f} µm"
            ),
        )

        ax.set_xlabel("Axial offset from initial z (µm)")
        ax.set_ylabel("Probe purity")
        ax.set_title("Purity-based axial calibration")

        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax
    
    
        """
        Run one complete mPIE reconstruction iteration.
        """

        # set position order
        self.setPositionOrder()

        self.pbar_pos = tqdm.tqdm(
            self.positionIndices,
            leave=False,
            desc="ptychogram",
            file=sys.stdout,
        )

        for positionLoop, positionIndex in enumerate(self.pbar_pos):

            row, col = self.reconstruction.positions[positionIndex]
            sy = slice(row, row + self.reconstruction.Np)
            sx = slice(col, col + self.reconstruction.Np)

            objectPatch = self.reconstruction.object[..., sy, sx].copy()

            # exit surface wave
            self.reconstruction.esw = (
                objectPatch * self.reconstruction.probe
            )

            # intensity constraint
            self.intensityProjection(positionIndex)

            DELTA = (
                self.reconstruction.eswUpdate
                - self.reconstruction.esw
            )

            # object update
            if (
                self.params.objectTVregSwitch
                and loop % self.params.objectTVfreq == 0
            ):
                object_patch = self.objectPatchUpdate_TV(
                    objectPatch,
                    DELTA,
                )
            else:
                object_patch = self.objectPatchUpdate(
                    objectPatch,
                    DELTA,
                )

            self.reconstruction.object[..., sy, sx] = object_patch

            if self.keepPatches:
                self.patches[positionIndex, ..., sy, sx] = (
                    asNumpyArray(abs(object_patch) ** 2)
                )

            # probe update
            weight = 1
            if self.params.weigh_probe_updates_by_intensity:
                weight = self.experimentalData.relative_intensity(
                    positionIndex
                )

            self.reconstruction.probe = self.probeUpdate(
                objectPatch,
                DELTA,
                weight,
            )

            # position correction
            if self.params.positionCorrectionSwitch:
                self.positionCorrection(
                    objectPatch,
                    positionIndex,
                    sy,
                    sx,
                )

            # momentum updates
            if np.random.rand(1) > 0.95:
                self.objectMomentumUpdate()
                self.probeMomentumUpdate()

        self.getErrorMetrics()
        self.applyConstraints(loop)

        r"""
        Apply the mPIE momentum update to the reconstructed probe. Similar to `objectMomentumUpdate()`.

        See Also:
            `objectMomentumUpdate`
                Equivalent momentum update applied to the reconstructed object.
        """
        gradient = self.reconstruction.probeBuffer - self.reconstruction.probe
        self.reconstruction.probeMomentum = (
            gradient + self.frictionM * self.reconstruction.probeMomentum
        )
        self.reconstruction.probe = (
            self.reconstruction.probe
            - self.feedbackM * self.reconstruction.probeMomentum
        )
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()

 