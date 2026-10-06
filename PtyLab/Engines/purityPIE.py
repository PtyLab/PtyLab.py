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
        if self.params.purityZAdaptive:
            best_z, best_purity = self.adaptivePurityZScan()
        else:
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

    def _evaluatePurityAtZ(self, z):
        """
        Evaluate probe purity for one candidate propagation distance.

        A fresh reconstruction is created at the requested z value and
        reconstructed using the standard mPIE engine. Probe modes are
        orthogonalized after reconstruction and the resulting probe purity
        is returned together with the candidate reconstruction state.

        Parameters
        ----------
        z : float
            Candidate propagation distance in meters.

        Returns
        -------
        purity : float
            Reconstructed probe purity at the candidate distance.

        candidate_reconstruction : Reconstruction
            Reconstruction state obtained at the candidate distance.
        """

        # ------------------------------------------------------------------
        # Create independent candidate reconstruction
        # ------------------------------------------------------------------

        (
            candidate_data,
            candidate_reconstruction,
            candidate_params,
        ) = self._createCandidateReconstruction(z)

        candidate_monitor = DummyMonitor()

        # ------------------------------------------------------------------
        # Run standard mPIE
        # ------------------------------------------------------------------

        candidate_engine = mPIE(
            candidate_reconstruction,
            candidate_data,
            candidate_params,
            candidate_monitor,
        )

        # Match the purityPIE engine settings
        candidate_engine.numIterations = self.numIterations

        candidate_engine.betaProbe = self.betaProbe
        candidate_engine.betaObject = self.betaObject

        candidate_engine.alphaProbe = self.alphaProbe
        candidate_engine.alphaObject = self.alphaObject

        candidate_engine.feedbackM = self.feedbackM
        candidate_engine.frictionM = self.frictionM

        candidate_engine.reconstruct()

        # ------------------------------------------------------------------
        # Final modal decomposition and purity evaluation
        # ------------------------------------------------------------------

        candidate_engine.orthogonalization()

        purity = float(
            np.asarray(
                asNumpyArray(
                    candidate_reconstruction.purityProbe
                )
            ).squeeze()
        )

        return purity, candidate_reconstruction

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
        best_state = None

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
            # Evaluate candidate
            # --------------------------------------------------------------

            purity, candidate_reconstruction = (
                self._evaluatePurityAtZ(z)
            )

            purity_values.append(purity)

            tqdm.tqdm.write(
                f"    Probe purity = {purity:.6f}"
            )

            # --------------------------------------------------------------
            # Update main monitor with completed candidate
            # --------------------------------------------------------------

            self.reconstruction.zo = z

            self.reconstruction.object = (
                asNumpyArray(
                    candidate_reconstruction.object
                ).copy()
            )

            self.reconstruction.probe = (
                asNumpyArray(
                    candidate_reconstruction.probe
                ).copy()
            )

            self.reconstruction.purityProbe = purity

            if hasattr(candidate_reconstruction, "error"):
                self.reconstruction.error = np.asarray(
                    asNumpyArray(
                        candidate_reconstruction.error
                    )
                ).copy()

            self.showReconstruction(
                self.numIterations - 1,
                force=True,
            )

            # --------------------------------------------------------------
            # Keep only the state required from the best candidate
            # --------------------------------------------------------------

            if purity > best_purity:

                best_purity = purity
                best_z = z

                best_state = {
                    "object": asNumpyArray(
                        candidate_reconstruction.object
                    ).copy(),

                    "probe": asNumpyArray(
                        candidate_reconstruction.probe
                    ).copy(),

                    "purityProbe": purity,
                }

                if hasattr(
                    candidate_reconstruction,
                    "encoder_corrected",
                ):
                    best_state["encoder_corrected"] = (
                        asNumpyArray(
                            candidate_reconstruction.encoder_corrected
                        ).copy()
                    )

                if hasattr(
                    candidate_reconstruction,
                    "objectMomentum",
                ):
                    best_state["objectMomentum"] = (
                        asNumpyArray(
                            candidate_reconstruction.objectMomentum
                        ).copy()
                    )

                if hasattr(
                    candidate_reconstruction,
                    "probeMomentum",
                ):
                    best_state["probeMomentum"] = (
                        asNumpyArray(
                            candidate_reconstruction.probeMomentum
                        ).copy()
                    )

                if hasattr(
                    candidate_reconstruction,
                    "objectBuffer",
                ):
                    best_state["objectBuffer"] = (
                        asNumpyArray(
                            candidate_reconstruction.objectBuffer
                        ).copy()
                    )

                if hasattr(
                    candidate_reconstruction,
                    "probeBuffer",
                ):
                    best_state["probeBuffer"] = (
                        asNumpyArray(
                            candidate_reconstruction.probeBuffer
                        ).copy()
                    )

                if hasattr(
                    candidate_reconstruction,
                    "error",
                ):
                    best_state["error"] = np.asarray(
                        asNumpyArray(
                            candidate_reconstruction.error
                        )
                    ).copy()

                tqdm.tqdm.write(
                    f"    NEW BEST: z = {best_z * 1e3:.6f} mm, "
                    f"purity = {best_purity:.6f}"
                )

            tqdm.tqdm.write("")

        # ------------------------------------------------------------------
        # Safety check
        # ------------------------------------------------------------------

        if best_state is None:
            raise RuntimeError(
                "Purity z scan did not produce a valid purity value."
            )

        # ------------------------------------------------------------------
        # Restore best candidate state
        # ------------------------------------------------------------------

        self.reconstruction.zo = best_z

        self.reconstruction.object = (
            best_state["object"]
        )

        self.reconstruction.probe = (
            best_state["probe"]
        )

        self.reconstruction.purityProbe = (
            best_state["purityProbe"]
        )

        if "encoder_corrected" in best_state:
            self.reconstruction.encoder_corrected = (
                best_state["encoder_corrected"]
            )

        if "objectMomentum" in best_state:
            self.reconstruction.objectMomentum = (
                best_state["objectMomentum"]
            )

        if "probeMomentum" in best_state:
            self.reconstruction.probeMomentum = (
                best_state["probeMomentum"]
            )

        if "objectBuffer" in best_state:
            self.reconstruction.objectBuffer = (
                best_state["objectBuffer"]
            )

        if "probeBuffer" in best_state:
            self.reconstruction.probeBuffer = (
                best_state["probeBuffer"]
            )

        if "error" in best_state:
            self.reconstruction.error = (
                best_state["error"]
            )

        # ------------------------------------------------------------------
        # Store scan results
        # ------------------------------------------------------------------

        self.purityZValues = np.asarray(
            z_values
        )

        self.purityZMetrics = np.asarray(
            purity_values
        )

        self.purityBestZ = best_z
        self.purityBestValue = best_purity

        # Final monitor update
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
            f"Best Δz     = "
            f"{(best_z - z0) * 1e6:+.2f} µm"
        )

        tqdm.tqdm.write(
            f"Best purity = {best_purity:.6f}"
        )

        tqdm.tqdm.write("=" * 60)

        return best_z, best_purity

    def adaptivePurityZScan(self):
        """
        Adaptively search for the propagation distance that maximizes
        reconstructed probe purity.

        The search first determines the uphill direction, then expands the
        axial step while purity keeps increasing. Once the maximum is bracketed,
        the step size is reduced and the search is refined around the current
        best position.
        """

        if self.reconstruction.npsm < 2:
            raise ValueError(
                "Purity-based z scanning requires at least two probe modes "
                "(reconstruction.npsm >= 2)."
            )

        z0 = self.reconstruction.zo
        self.purityZInitialGuess = z0

        step = self.params.purityZInitialStep
        growth = self.params.purityZStepGrowth
        shrink = self.params.purityZStepShrink
        min_step = self.params.purityZMinStep
        purity_tol = self.params.purityZPurityTolerance
        max_evaluations = self.params.purityZMaxEvaluations

        evaluated = {}
        reconstruction_states = {}

        best_z = None
        best_purity = -np.inf
        best_state = None

        z_history = []
        purity_history = []

        def evaluate(z):
            nonlocal best_z
            nonlocal best_purity
            nonlocal best_state

            # Reuse already evaluated points
            key = float(z)

            if key in evaluated:
                return evaluated[key]

            purity, candidate_reconstruction = (
                self._evaluatePurityAtZ(z)
            )

            evaluated[key] = purity

            z_history.append(z)
            purity_history.append(purity)

            tqdm.tqdm.write(
                f"z = {z * 1e3:.6f} mm | "
                f"purity = {purity:.6f}"
            )

            # Store best candidate state
            if purity > best_purity:

                best_purity = purity
                best_z = z

                best_state = {
                    "object": asNumpyArray(
                        candidate_reconstruction.object
                    ).copy(),

                    "probe": asNumpyArray(
                        candidate_reconstruction.probe
                    ).copy(),

                    "purityProbe": purity,
                }

                if hasattr(
                    candidate_reconstruction,
                    "encoder_corrected",
                ):
                    best_state["encoder_corrected"] = (
                        asNumpyArray(
                            candidate_reconstruction.encoder_corrected
                        ).copy()
                    )

                if hasattr(
                    candidate_reconstruction,
                    "objectMomentum",
                ):
                    best_state["objectMomentum"] = (
                        asNumpyArray(
                            candidate_reconstruction.objectMomentum
                        ).copy()
                    )

                if hasattr(
                    candidate_reconstruction,
                    "probeMomentum",
                ):
                    best_state["probeMomentum"] = (
                        asNumpyArray(
                            candidate_reconstruction.probeMomentum
                        ).copy()
                    )

                if hasattr(
                    candidate_reconstruction,
                    "objectBuffer",
                ):
                    best_state["objectBuffer"] = (
                        asNumpyArray(
                            candidate_reconstruction.objectBuffer
                        ).copy()
                    )

                if hasattr(
                    candidate_reconstruction,
                    "probeBuffer",
                ):
                    best_state["probeBuffer"] = (
                        asNumpyArray(
                            candidate_reconstruction.probeBuffer
                        ).copy()
                    )

                if hasattr(
                    candidate_reconstruction,
                    "error",
                ):
                    best_state["error"] = np.asarray(
                        asNumpyArray(
                            candidate_reconstruction.error
                        )
                    ).copy()

                tqdm.tqdm.write(
                    f"NEW BEST: z = {best_z * 1e3:.6f} mm | "
                    f"purity = {best_purity:.6f}"
                )

            return purity

        tqdm.tqdm.write("")
        tqdm.tqdm.write("Starting adaptive purity-based z scan")
        tqdm.tqdm.write(
            f"Initial z = {z0 * 1e3:.6f} mm | "
            f"initial step = {step * 1e6:.1f} µm"
        )
        tqdm.tqdm.write("")

        # --------------------------------------------------------------
        # 1. Initial three-point evaluation
        # --------------------------------------------------------------

        p0 = evaluate(z0)

        z_left = z0 - step
        z_right = z0 + step

        p_left = evaluate(z_left)
        p_right = evaluate(z_right)

        evaluations = len(evaluated)

        # --------------------------------------------------------------
        # 2. Determine initial uphill direction
        # --------------------------------------------------------------

        if p_left > p0 and p_left >= p_right:
            direction = -1

        elif p_right > p0 and p_right > p_left:
            direction = +1

        else:
            direction = 0

        # --------------------------------------------------------------
        # 3. Expansion phase
        # --------------------------------------------------------------

        if direction != 0:

            current_z = z0 + direction * step
            current_purity = evaluate(current_z)

            while evaluations < max_evaluations:

                step *= growth

                next_z = current_z + direction * step
                next_purity = evaluate(next_z)

                evaluations = len(evaluated)

                improvement = next_purity - current_purity

                if improvement > purity_tol:

                    current_z = next_z
                    current_purity = next_purity
                    continue

                # Purity stopped increasing:
                # maximum should now be bracketed.
                break

        # --------------------------------------------------------------
        # 4. Refinement phase
        # --------------------------------------------------------------

        step *= shrink

        while (
            step >= min_step
            and len(evaluated) < max_evaluations
        ):

            left_z = best_z - step
            right_z = best_z + step

            left_purity = evaluate(left_z)

            if len(evaluated) >= max_evaluations:
                break

            right_purity = evaluate(right_z)

            # If neither side significantly improves the best result,
            # reduce the search step.
            if (
                left_purity <= best_purity + purity_tol
                and right_purity <= best_purity + purity_tol
            ):
                step *= shrink
                continue

            # A better point was found. Keep the current step and
            # continue refining around the new best position.

        # --------------------------------------------------------------
        # Safety check
        # --------------------------------------------------------------

        if best_state is None:
            raise RuntimeError(
                "Adaptive purity z scan did not produce a valid result."
            )

        # --------------------------------------------------------------
        # Restore best candidate
        # --------------------------------------------------------------

        self.reconstruction.zo = best_z
        self.reconstruction.object = best_state["object"]
        self.reconstruction.probe = best_state["probe"]
        self.reconstruction.purityProbe = best_state["purityProbe"]

        if "encoder_corrected" in best_state:
            self.reconstruction.encoder_corrected = (
                best_state["encoder_corrected"]
            )

        if "objectMomentum" in best_state:
            self.reconstruction.objectMomentum = (
                best_state["objectMomentum"]
            )

        if "probeMomentum" in best_state:
            self.reconstruction.probeMomentum = (
                best_state["probeMomentum"]
            )

        if "objectBuffer" in best_state:
            self.reconstruction.objectBuffer = (
                best_state["objectBuffer"]
            )

        if "probeBuffer" in best_state:
            self.reconstruction.probeBuffer = (
                best_state["probeBuffer"]
            )

        if "error" in best_state:
            self.reconstruction.error = (
                best_state["error"]
            )

        # --------------------------------------------------------------
        # Store results
        # --------------------------------------------------------------

        order = np.argsort(z_history)

        self.purityZValues = np.asarray(z_history)[order]
        self.purityZMetrics = np.asarray(purity_history)[order]

        self.purityBestZ = best_z
        self.purityBestValue = best_purity

        self.showReconstruction(
            self.numIterations - 1,
            force=True,
        )

        # --------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------

        tqdm.tqdm.write("=" * 60)
        tqdm.tqdm.write("Adaptive purity-based z scan finished")
        tqdm.tqdm.write(
            f"Best z      = {best_z * 1e3:.6f} mm"
        )
        tqdm.tqdm.write(
            f"Best Δz     = {(best_z - z0) * 1e6:+.2f} µm"
        )
        tqdm.tqdm.write(
            f"Best purity = {best_purity:.6f}"
        )
        tqdm.tqdm.write(
            f"Evaluations = {len(evaluated)}"
        )
        tqdm.tqdm.write("=" * 60)

        return best_z, best_purity

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
    
    

    