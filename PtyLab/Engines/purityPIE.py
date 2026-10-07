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

#from PtyLab.Engines.BaseEngine import BaseEngine
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
    r"""
    Purity-based axial self-calibration for ptychographic reconstruction.

    `purityPIE` extends `mPIE` by estimating the sample-to-detector
    axial distance from the reconstructed probe purity.[^liu2025]

    In mixed-state ptychography, the measured diffraction intensity at
    scan position $j$ is modeled as an incoherent sum over multiple probe
    modes:

    $$
    I_j(\mathbf{q}) = \sum_m \left| \mathcal{F}\left[P_m(\mathbf{r}) O(\mathbf{r}-\mathbf{r}_j)\right] \right|^2
    $$

    where $P_m$ denotes probe mode $m$, $O$ is the object transmission
    function, and $\mathcal{F}$ denotes propagation to the detector plane.

    The mutual intensity of the reconstructed probe is

    $$
    J(\mathbf{r}_1,\mathbf{r}_2) = \sum_m P_m^*(\mathbf{r}_1)P_m(\mathbf{r}_2)
    $$

    and can be decomposed into orthonormal coherent modes as

    $$
    J(\mathbf{r}_1,\mathbf{r}_2) = \sum_m \lambda_m \phi_m^*(\mathbf{r}_1)\phi_m(\mathbf{r}_2)
    $$

    where $\lambda_m$ represents the intensity contribution, or relative
    power content, of the corresponding orthonormal mode.

    The probe purity is defined as

    $$
    \mu = \frac{\sqrt{\sum_m \lambda_m^2}}{\sum_m \lambda_m}
    $$

    with $\mu \in [0,1]$.

    If the assumed axial distance is incorrect, a mismatch is introduced
    in the diffraction integral connecting real and reciprocal space.
    This produces stretching deformation and resolution degradation in
    the reconstructed object and probe, which appears as a reduction in
    reconstructed purity.

    The calibrated axial distance is obtained by maximizing the
    reconstructed probe purity:

    $$
    z_{\mathrm{opt}} = \operatorname*{arg\,max}_z \mu(z)
    $$

    For each candidate axial distance, `purityPIE` creates an independent
    reconstruction initialized with the corresponding geometry, performs
    an `mPIE` reconstruction, orthogonalizes the reconstructed probe
    modes, and evaluates the resulting purity. The candidate with the
    highest purity is retained as the calibrated reconstruction.

    Two search strategies are available:

    - A fixed-grid scan evaluates uniformly spaced axial positions defined
    by `params.purityZScanRange` and `params.purityZScanPoints`.

    - An adaptive scan expands the axial step while purity improves and
    subsequently refines the search around the best candidate with
    progressively smaller steps. It is enabled with
    `params.purityZAdaptive = True`.

    Purity-based axial calibration requires at least two reconstructed
    probe modes, i.e. `reconstruction.npsm >= 2`.

    [^liu2025]: C. Liu, W. Eschen, D. S. Penagos Molina,
            L. Licht, M. Abdelaal, and J. Rothhardt,
            "Purity-based self-calibration in ptychography,"
            Opt. Lett. 50, 1581-1584 (2025).
            https://doi.org/10.1364/OL.543865

    See Also:
        `mPIE`
            Momentum-accelerated reconstruction engine used for each
            candidate axial distance.
    """
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
        Run purity-based axial self-calibration.

        The reconstruction first prepares the current object, probe, geometry,
        constraints, and optional GPU state using the standard `mPIE` setup.

        The axial calibration is then performed using one of two search strategies:

        - If `params.purityZAdaptive = False`, `purityZScan()` evaluates a fixed
        grid of candidate axial distances.

        - If `params.purityZAdaptive = True`, `adaptivePurityZScan()` searches the
        axial distance using adaptive step-size expansion and refinement.

        Each candidate distance is evaluated using an independent `mPIE`
        reconstruction. The candidate yielding the highest reconstructed probe
        purity is selected, and its reconstruction state is restored to the
        original `Reconstruction` object.

        If `params.purityZScanPlot` is enabled, the resulting purity-versus-z
        curve is displayed after calibration.

        Args:
            experimentalData (ExperimentalData, optional):
                Experimental dataset to use for the calibration. If provided, it
                replaces the dataset currently attached to the engine.

            reconstruction (Reconstruction, optional):
                Reconstruction state to calibrate. If provided, it replaces the
                reconstruction currently attached to the engine.

        Returns:
            tuple:
                `(best_z, best_purity)`, where `best_z` is the calibrated axial
                propagation distance in meters and `best_purity` is the
                corresponding reconstructed probe purity.
        """

        self.changeExperimentalData(experimentalData)
        self.changeOptimizable(reconstruction)

        self._prepareReconstruction()

        # Reset momentum buffers after preparation
        self.reconstruction.objectBuffer = (
            self.reconstruction.object.copy()
        )

        self.reconstruction.probeBuffer = (
            self.reconstruction.probe.copy()
        )

        target = self.params.purityCalibrationTarget.lower()

        # ------------------------------------------------------------------
        # Axial-distance calibration
        # ------------------------------------------------------------------

        if target == "z":

            if self.params.purityAdaptive:
                best_value, best_purity = (
                    self.adaptivePurityZScan()
                )
            else:
                best_value, best_purity = (
                    self.purityZScan()
                )

        # ------------------------------------------------------------------
        # Wavelength calibration
        # ------------------------------------------------------------------

        elif target == "wavelength":

            if self.params.purityAdaptive:
                best_value, best_purity = (
                    self.adaptivePurityWavelengthScan()
                )
            else:
                best_value, best_purity = (
                    self.purityWavelengthScan()
                )

        # ------------------------------------------------------------------
        # Unsupported target
        # ------------------------------------------------------------------

        else:

            raise ValueError(
                "Unsupported purity calibration target: "
                f"{self.params.purityCalibrationTarget!r}. "
                "Supported targets are 'z' and 'wavelength'."
            )

        if self.params.purityScanPlot:
            self.plotPurityScan()

        # ------------------------------------------------------------------
        # Return reconstruction data to CPU
        # ------------------------------------------------------------------

        if self.params.gpuFlag:
            self.logger.info("switch to cpu")
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

        return best_value, best_purity

    def _createCandidateReconstruction(
        self,
        target,
        value,
    ):
        """
        Create an independent reconstruction for one candidate calibration value.
        """

        candidate_data = deepcopy(
            self.experimentalData
        )

        candidate_params = deepcopy(
            self.params
        )

        # --------------------------------------------------------------
        # Set candidate parameter before creating Reconstruction
        # --------------------------------------------------------------

        if target == "z":
            candidate_data.zo = value

        elif target == "wavelength":
            candidate_data.wavelength = value
            
            if self.reconstruction.nlambda == 1:
                candidate_data.spectralDensity = np.atleast_1d(value)
        else:
            raise ValueError(
                f"Unsupported purity calibration target: {target!r}"
            )

        # Fresh geometry for this candidate
        candidate_reconstruction = Reconstruction(
            candidate_data,
            candidate_params,
        )

        # --------------------------------------------------------------
        # Copy reconstruction settings
        # --------------------------------------------------------------

        candidate_reconstruction.npsm = (
            self.reconstruction.npsm
        )

        candidate_reconstruction.nosm = (
            self.reconstruction.nosm
        )

        candidate_reconstruction.nlambda = (
            self.reconstruction.nlambda
        )

        candidate_reconstruction.nslice = (
            self.reconstruction.nslice
        )

        candidate_reconstruction.initialProbe = (
            self.reconstruction.initialProbe
        )

        candidate_reconstruction.initialObject = (
            self.reconstruction.initialObject
        )

        if hasattr(
            self.reconstruction,
            "initialProbe_filename",
        ):
            candidate_reconstruction.initialProbe_filename = (
                self.reconstruction.initialProbe_filename
            )

        if hasattr(
            self.reconstruction,
            "initialObject_filename",
        ):
            candidate_reconstruction.initialObject_filename = (
                self.reconstruction.initialObject_filename
            )

        candidate_reconstruction.initializeObjectProbe()

        return (
            candidate_data,
            candidate_reconstruction,
            candidate_params,
        )

    def _evaluatePurityAtZ(self, z):
        return self._evaluatePurityAtParameter(
            "z",
            z,
        )

    def _evaluatePurityAtWavelength(
        self,
        wavelength,
    ):
        return self._evaluatePurityAtParameter(
            "wavelength",
            wavelength,
        )

    def _evaluatePurityAtParameter(
        self,
        target,
        value,
    ):
        """
        Evaluate reconstructed probe purity for one candidate calibration value.
        """

        (
            candidate_data,
            candidate_reconstruction,
            candidate_params,
        ) = self._createCandidateReconstruction(
            target,
            value,
        )

        candidate_monitor = DummyMonitor()

        candidate_engine = mPIE(
            candidate_reconstruction,
            candidate_data,
            candidate_params,
            candidate_monitor,
        )

        candidate_engine.numIterations = (
            self.numIterations
        )

        candidate_engine.betaProbe = (
            self.betaProbe
        )

        candidate_engine.betaObject = (
            self.betaObject
        )

        candidate_engine.alphaProbe = (
            self.alphaProbe
        )

        candidate_engine.alphaObject = (
            self.alphaObject
        )

        candidate_engine.feedbackM = (
            self.feedbackM
        )

        candidate_engine.frictionM = (
            self.frictionM
        )

        candidate_engine.reconstruct()

        candidate_engine.orthogonalization()

        purity = float(
            np.asarray(
                asNumpyArray(
                    candidate_reconstruction.purityProbe
                )
            ).squeeze()
        )

        return (
            purity,
            candidate_reconstruction,
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
        error_values = []

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

            candidate_error = float(
                np.asarray(
                    asNumpyArray(
                        candidate_reconstruction.error
                    )
                ).reshape(-1)[-1]
            )

            purity_values.append(purity)
            error_values.append(candidate_error)

            tqdm.tqdm.write(
                f"    Probe purity = {purity:.6f} | "
                f"reconstruction error = {candidate_error:.6e}"
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

        self.purityZErrors = np.asarray(
            error_values
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
        r"""
        Perform adaptive purity-based axial calibration.

        The adaptive scan searches for the axial distance that maximizes the
        reconstructed probe purity while reducing the number of candidate
        reconstructions compared with a dense fixed-grid scan.

        The search starts from the current propagation distance
        `reconstruction.zo` and evaluates three initial positions:

        $$
        z_0-\Delta z,\quad z_0,\quad z_0+\Delta z
        $$

        where the initial step size $\Delta z$ is defined by
        `params.purityZInitialStep`.

        If the purity increases toward one side, the search continues in that
        direction and progressively expands the axial step:

        $$
        \Delta z_{k+1} = g\,\Delta z_k
        $$

        where the expansion factor $g$ is set by `params.purityZStepGrowth`.
        Expansion continues while the increase in purity between consecutive
        candidate positions exceeds `params.purityZPurityTolerance`.

        Once the purity stops increasing, the search enters a refinement phase.
        The axial step is progressively reduced according to

        $$
        \Delta z_{k+1} = s\,\Delta z_k
        $$

        where the reduction factor $s$ is set by `params.purityZStepShrink`.
        Candidate positions on both sides of the current best distance are then
        evaluated until the requested axial resolution is reached.

        The adaptive search is controlled by the following parameters:

        - `params.purityZInitialStep`:
        Initial axial step size used for the three-point search around the
        starting distance.

        - `params.purityZStepGrowth`:
        Multiplicative factor used to increase the step size while moving
        toward higher purity.

        - `params.purityZStepShrink`:
        Multiplicative factor used to reduce the step size during refinement.

        - `params.purityZMinStep`:
        Minimum axial step size. Refinement stops once the current step becomes
        smaller than this value.

        - `params.purityZPurityTolerance`:
        Minimum purity improvement considered significant when determining
        whether the search should continue in the current direction.

        - `params.purityZMaxEvaluations`:
        Maximum number of candidate axial distances that may be reconstructed
        during one adaptive scan.

        Candidate distances that have already been evaluated are cached and reused,
        avoiding repeated `mPIE` reconstructions at the same axial position.

        Each new candidate is evaluated independently using
        `_evaluatePurityAtZ()`. The candidate with the highest reconstructed probe
        purity is retained as the calibrated axial distance, and its reconstruction
        state is restored to the original `Reconstruction` object.

        The adaptive scan results are stored in:

        - `purityZValues`: evaluated axial distances in meters, sorted by z.
        - `purityZMetrics`: reconstructed probe purity corresponding to each
        evaluated distance.
        - `purityBestZ`: axial distance corresponding to the highest purity.
        - `purityBestValue`: maximum reconstructed probe purity.
        - `purityZInitialGuess`: axial distance used as the starting point.

        Returns:
            tuple:
                `(best_z, best_purity)`, where `best_z` is the calibrated axial
                propagation distance in meters and `best_purity` is the maximum
                reconstructed probe purity.

        Raises:
            ValueError:
                If fewer than two probe modes are reconstructed
                (`reconstruction.npsm < 2`).

            RuntimeError:
                If no valid candidate reconstruction produces a purity value.

        See Also:
            `purityZScan`
                Fixed-grid purity-based axial calibration.
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
        max_evaluations = self.params.purityMaxEvaluations

        evaluated = {}

        best_z = None
        best_purity = -np.inf
        best_state = None

        z_history = []
        purity_history = []
        error_history = []

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
            candidate_error = float(
                np.asarray(
                    asNumpyArray(
                        candidate_reconstruction.error
                    )
                ).reshape(-1)[-1]
            )
            
            evaluated[key] = purity

            z_history.append(z)
            purity_history.append(purity)
            error_history.append(candidate_error)
            tqdm.tqdm.write(
                f"z = {z * 1e3:.6f} mm | "
                f"purity = {purity:.6f} | "
                f"error = {candidate_error:.6e}"
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
        self.purityZErrors = np.asarray(error_history)[order]

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
  
    def purityWavelengthScan(self):


        """
        Scan a fixed wavelength range and select the wavelength that maximizes
        reconstructed probe purity.

        Each candidate wavelength is evaluated using an independent reconstruction
        state and a standard mPIE reconstruction.
        """

        if self.reconstruction.npsm < 2:
            raise ValueError(
                "Purity-based wavelength scanning requires at least two probe modes "
                "(reconstruction.npsm >= 2)."
            )

        wavelength0 = float(
            self.reconstruction.wavelength
        )

        self.purityWavelengthInitialGuess = (
            wavelength0
        )

        wavelength_values = np.linspace(
            wavelength0
            - self.params.purityWavelengthScanRange,
            wavelength0
            + self.params.purityWavelengthScanRange,
            self.params.purityWavelengthScanPoints,
        )

        purity_values = []
        error_values = []

        best_purity = -np.inf
        best_wavelength = wavelength0
        best_state = None

        tqdm.tqdm.write("")
        tqdm.tqdm.write(
            "Starting purity-based wavelength scan"
        )

        tqdm.tqdm.write(
            f"Initial wavelength = "
            f"{wavelength0 * 1e9:.6f} nm | "
            f"range = ±"
            f"{self.params.purityWavelengthScanRange * 1e9:.4f} nm | "
            f"points = {len(wavelength_values)} | "
            f"iterations/wavelength = {self.numIterations}"
        )

        tqdm.tqdm.write("")

        for wavelength_index, wavelength in enumerate(
            wavelength_values
        ):

            delta_wavelength_nm = (
                wavelength - wavelength0
            ) * 1e9

            tqdm.tqdm.write(
                f"[{wavelength_index + 1}/"
                f"{len(wavelength_values)}] "
                f"Reconstructing wavelength = "
                f"{wavelength * 1e9:.6f} nm "
                f"(Δλ = {delta_wavelength_nm:+.4f} nm)"
            )

            (
                purity,
                candidate_reconstruction,
            ) = self._evaluatePurityAtWavelength(
                wavelength
            )

            candidate_error = float(
                np.asarray(
                    asNumpyArray(
                        candidate_reconstruction.error
                    )
                ).reshape(-1)[-1]
            )

            purity_values.append(
                purity
            )

            error_values.append(
                candidate_error
            )

            tqdm.tqdm.write(
                f"    Probe purity = {purity:.6f} | "
                f"reconstruction error = {candidate_error:.6e}"
            )

            # Update monitor with current candidate
            self.reconstruction.wavelength = (
                wavelength
            )

            self.reconstruction.dxp = (
                candidate_reconstruction.dxp
            )

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

            self.reconstruction.purityProbe = (
                purity
            )

            if hasattr(
                candidate_reconstruction,
                "error",
            ):
                self.reconstruction.error = (
                    np.asarray(
                        asNumpyArray(
                            candidate_reconstruction.error
                        )
                    ).copy()
                )

            self.showReconstruction(
                self.numIterations - 1,
                force=True,
            )

            if purity > best_purity:

                best_purity = purity
                best_wavelength = wavelength

                best_state = {
                    "object": asNumpyArray(
                        candidate_reconstruction.object
                    ).copy(),

                    "probe": asNumpyArray(
                        candidate_reconstruction.probe
                    ).copy(),

                    "purityProbe": purity,

                    "dxp": float(
                        candidate_reconstruction.dxp
                    ),
                }

                if hasattr(
                    candidate_reconstruction,
                    "spectralDensity",
                ):
                    best_state["spectralDensity"] = (
                        np.asarray(
                            candidate_reconstruction.spectralDensity
                        ).copy()
                    )

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
                    best_state["error"] = (
                        np.asarray(
                            asNumpyArray(
                                candidate_reconstruction.error
                            )
                        ).copy()
                    )

                tqdm.tqdm.write(
                    f"    NEW BEST: wavelength = "
                    f"{best_wavelength * 1e9:.6f} nm | "
                    f"purity = {best_purity:.6f}"
                )

            tqdm.tqdm.write("")

        if best_state is None:
            raise RuntimeError(
                "Purity wavelength scan did not produce "
                "a valid purity value."
            )

        # Restore best candidate
        self.reconstruction.wavelength = (
            best_wavelength
        )

        self.reconstruction.dxp = (
            best_state["dxp"]
        )

        if "spectralDensity" in best_state:
            self.reconstruction.spectralDensity = (
                best_state["spectralDensity"]
            )

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

        # Store scan results
        self.purityWavelengthValues = (
            np.asarray(
                wavelength_values
            )
        )

        self.purityWavelengthMetrics = (
            np.asarray(
                purity_values
            )
        )

        self.purityWavelengthErrors = (
            np.asarray(
                error_values
            )
        )

        self.purityBestWavelength = (
            best_wavelength
        )

        self.purityBestWavelengthValue = (
            best_purity
        )

        self.showReconstruction(
            self.numIterations - 1,
            force=True,
        )

        tqdm.tqdm.write("=" * 60)
        tqdm.tqdm.write(
            "Purity-based wavelength scan finished"
        )

        tqdm.tqdm.write(
            f"Best wavelength = "
            f"{best_wavelength * 1e9:.6f} nm"
        )

        tqdm.tqdm.write(
            f"Best Δλ         = "
            f"{(best_wavelength - wavelength0) * 1e9:+.4f} nm"
        )

        tqdm.tqdm.write(
            f"Best purity     = "
            f"{best_purity:.6f}"
        )

        tqdm.tqdm.write("=" * 60)

        return (
            best_wavelength,
            best_purity,
        )

    def adaptivePurityWavelengthScan(self):

        """
        Adaptively search for the wavelength that maximizes reconstructed
        probe purity.

        The search first determines the uphill direction, then expands the
        wavelength step while purity keeps increasing. Once the maximum is
        bracketed, the step size is reduced and the search is refined around
        the current best wavelength.
        """

        if self.reconstruction.npsm < 2:
            raise ValueError(
                "Purity-based wavelength scanning requires at least two probe modes "
                "(reconstruction.npsm >= 2)."
            )

        wavelength0 = float(
            self.reconstruction.wavelength
        )

        self.purityWavelengthInitialGuess = (
            wavelength0
        )

        step = self.params.purityWavelengthInitialStep
        growth = self.params.purityWavelengthStepGrowth
        shrink = self.params.purityWavelengthStepShrink
        min_step = self.params.purityWavelengthMinStep
        purity_tol = self.params.purityWavelengthPurityTolerance
        max_evaluations = self.params.purityMaxEvaluations

        evaluated = {}

        best_wavelength = None
        best_purity = -np.inf
        best_state = None

        wavelength_history = []
        purity_history = []
        error_history = []

        def evaluate(wavelength):

            nonlocal best_wavelength
            nonlocal best_purity
            nonlocal best_state

            if wavelength <= 0:
                return -np.inf

            key = float(wavelength)

            if key in evaluated:
                return evaluated[key]

            (
                purity,
                candidate_reconstruction,
            ) = self._evaluatePurityAtWavelength(
                wavelength
            )

            candidate_error = float(
                np.asarray(
                    asNumpyArray(
                        candidate_reconstruction.error
                    )
                ).reshape(-1)[-1]
            )

            evaluated[key] = purity

            wavelength_history.append(
                wavelength
            )

            purity_history.append(
                purity
            )

            error_history.append(
                candidate_error
            )

            tqdm.tqdm.write(
                f"wavelength = {wavelength * 1e9:.6f} nm | "
                f"purity = {purity:.6f} | "
                f"error = {candidate_error:.6e}"
            )

            if purity > best_purity:

                best_purity = purity
                best_wavelength = wavelength

                best_state = {
                    "object": asNumpyArray(
                        candidate_reconstruction.object
                    ).copy(),

                    "probe": asNumpyArray(
                        candidate_reconstruction.probe
                    ).copy(),

                    "purityProbe": purity,

                    "dxp": float(
                        candidate_reconstruction.dxp
                    ),
                }

                if hasattr(
                    candidate_reconstruction,
                    "spectralDensity",
                ):
                    best_state["spectralDensity"] = (
                        np.asarray(
                            candidate_reconstruction.spectralDensity
                        ).copy()
                    )

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
                    best_state["error"] = (
                        np.asarray(
                            asNumpyArray(
                                candidate_reconstruction.error
                            )
                        ).copy()
                    )

                tqdm.tqdm.write(
                    f"NEW BEST: wavelength = "
                    f"{best_wavelength * 1e9:.6f} nm | "
                    f"purity = {best_purity:.6f}"
                )

            return purity

        tqdm.tqdm.write("")
        tqdm.tqdm.write(
            "Starting adaptive purity-based wavelength scan"
        )

        tqdm.tqdm.write(
            f"Initial wavelength = "
            f"{wavelength0 * 1e9:.6f} nm | "
            f"initial step = {step * 1e9:.4f} nm"
        )

        tqdm.tqdm.write("")

        # ------------------------------------------------------------------
        # 1. Initial three-point evaluation
        # ------------------------------------------------------------------

        p0 = evaluate(
            wavelength0
        )

        wavelength_left = (
            wavelength0 - step
        )

        wavelength_right = (
            wavelength0 + step
        )

        p_left = evaluate(
            wavelength_left
        )

        p_right = evaluate(
            wavelength_right
        )

        # ------------------------------------------------------------------
        # 2. Determine uphill direction
        # ------------------------------------------------------------------

        if (
            p_left > p0
            and p_left >= p_right
        ):
            direction = -1

        elif (
            p_right > p0
            and p_right > p_left
        ):
            direction = +1

        else:
            direction = 0

        # ------------------------------------------------------------------
        # 3. Expansion phase
        # ------------------------------------------------------------------

        if direction != 0:

            current_wavelength = (
                wavelength0
                + direction * step
            )

            current_purity = evaluate(
                current_wavelength
            )

            while (
                len(evaluated)
                < max_evaluations
            ):

                step *= growth

                next_wavelength = (
                    current_wavelength
                    + direction * step
                )

                if next_wavelength <= 0:
                    break

                next_purity = evaluate(
                    next_wavelength
                )

                improvement = (
                    next_purity
                    - current_purity
                )

                if improvement > purity_tol:

                    current_wavelength = (
                        next_wavelength
                    )

                    current_purity = (
                        next_purity
                    )

                    continue

                break

        # ------------------------------------------------------------------
        # 4. Refinement phase
        # ------------------------------------------------------------------

        step *= shrink

        while (
            step >= min_step
            and len(evaluated)
            < max_evaluations
        ):

            left_wavelength = (
                best_wavelength - step
            )

            right_wavelength = (
                best_wavelength + step
            )

            if left_wavelength > 0:
                left_purity = evaluate(
                    left_wavelength
                )
            else:
                left_purity = -np.inf

            if (
                len(evaluated)
                >= max_evaluations
            ):
                break

            right_purity = evaluate(
                right_wavelength
            )

            if (
                left_purity
                <= best_purity + purity_tol
                and right_purity
                <= best_purity + purity_tol
            ):
                step *= shrink
                continue

        if best_state is None:
            raise RuntimeError(
                "Adaptive purity wavelength scan did not "
                "produce a valid result."
            )

        # ------------------------------------------------------------------
        # Restore best candidate
        # ------------------------------------------------------------------

        self.reconstruction.wavelength = (
            best_wavelength
        )

        self.reconstruction.dxp = (
            best_state["dxp"]
        )

        if "spectralDensity" in best_state:
            self.reconstruction.spectralDensity = (
                best_state["spectralDensity"]
            )

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
        # Store results
        # ------------------------------------------------------------------

        order = np.argsort(
            wavelength_history
        )

        self.purityWavelengthValues = (
            np.asarray(
                wavelength_history
            )[order]
        )

        self.purityWavelengthMetrics = (
            np.asarray(
                purity_history
            )[order]
        )

        self.purityWavelengthErrors = (
            np.asarray(
                error_history
            )[order]
        )

        self.purityBestWavelength = (
            best_wavelength
        )

        self.purityBestWavelengthValue = (
            best_purity
        )

        self.showReconstruction(
            self.numIterations - 1,
            force=True,
        )

        tqdm.tqdm.write("=" * 60)
        tqdm.tqdm.write(
            "Adaptive purity-based wavelength scan finished"
        )

        tqdm.tqdm.write(
            f"Best wavelength = "
            f"{best_wavelength * 1e9:.6f} nm"
        )

        tqdm.tqdm.write(
            f"Best Δλ         = "
            f"{(best_wavelength - wavelength0) * 1e9:+.4f} nm"
        )

        tqdm.tqdm.write(
            f"Best purity     = "
            f"{best_purity:.6f}"
        )

        tqdm.tqdm.write(
            f"Evaluations     = "
            f"{len(evaluated)}"
        )

        tqdm.tqdm.write("=" * 60)

        return (
            best_wavelength,
            best_purity,
        )

    def plotPurityScan_X(
    self,
    show=True,
    fit=True,
    fit_points=5,
):
        """
        Plot probe purity and reconstruction error for the selected
        purity-calibration target.

        Optionally fit a quadratic curve locally around the measured purity
        maximum and estimate the fitted calibration optimum.

        Args:
            show (bool, optional):
                Display the figure immediately. Default is `True`.

            fit (bool, optional):
                If `True`, perform a local quadratic fit around the measured
                purity maximum. Default is `True`.

            fit_points (int, optional):
                Number of measured points used for the local quadratic fit.
                Default is 5.

        Returns:
            tuple:
                Matplotlib `(figure, purity_axis, error_axis)` objects.
        """

        target = (
            self.params.purityCalibrationTarget.lower()
        )

        # ------------------------------------------------------------------
        # Select calibration results
        # ------------------------------------------------------------------

        if target == "z":

            if not hasattr(
                self,
                "purityZValues",
            ):
                raise RuntimeError(
                    "No purity z-scan results are available."
                )

            values = (
                np.asarray(
                    self.purityZValues
                )
                * 1e3
            )

            purity = np.asarray(
                self.purityZMetrics
            )

            errors = (
                np.asarray(
                    self.purityZErrors
                )
                if hasattr(
                    self,
                    "purityZErrors",
                )
                else None
            )

            initial_value = (
                self.purityZInitialGuess
                * 1e3
            )

            xlabel = (
                "Sample-to-detector distance z (mm)"
            )

            title = (
                "Purity-based axial calibration"
            )

            value_symbol = "z"
            value_unit = "mm"
            value_format = ".3f"

        elif target == "wavelength":

            if not hasattr(
                self,
                "purityWavelengthValues",
            ):
                raise RuntimeError(
                    "No purity wavelength-scan results are available."
                )

            values = (
                np.asarray(
                    self.purityWavelengthValues
                )
                * 1e9
            )

            purity = np.asarray(
                self.purityWavelengthMetrics
            )

            errors = (
                np.asarray(
                    self.purityWavelengthErrors
                )
                if hasattr(
                    self,
                    "purityWavelengthErrors",
                )
                else None
            )

            initial_value = (
                self.purityWavelengthInitialGuess
                * 1e9
            )

            xlabel = (
                "Wavelength λ (nm)"
            )

            title = (
                "Purity-based wavelength calibration"
            )

            value_symbol = "λ"
            value_unit = "nm"
            value_format = ".4f"

        else:

            raise ValueError(
                "Unsupported purity calibration target: "
                f"{self.params.purityCalibrationTarget!r}"
            )

        # ------------------------------------------------------------------
        # Best measured purity
        # ------------------------------------------------------------------

        best_purity_index = np.argmax(
            purity
        )

        fig, ax_purity = plt.subplots(
            figsize=(7, 4.5)
        )

        ax_purity.plot(
            values,
            purity,
            marker="o",
            label="Probe purity",
        )

        ax_purity.plot(
            values[
                best_purity_index
            ],
            purity[
                best_purity_index
            ],
            marker="*",
            markersize=12,
            linestyle="None",
            label=(
                f"Measured maximum: "
                f"{value_symbol} = "
                f"{format(values[best_purity_index], value_format)} "
                f"{value_unit}"
            ),
        )

        # ------------------------------------------------------------------
        # Local quadratic fitting
        # ------------------------------------------------------------------

        if fit:

            if fit_points < 3:
                raise ValueError(
                    "`fit_points` must be at least 3."
                )

            if fit_points > len(values):
                fit_points = len(values)

            half_window = (
                fit_points // 2
            )

            start = max(
                0,
                best_purity_index
                - half_window,
            )

            stop = min(
                len(values),
                start + fit_points,
            )

            start = max(
                0,
                stop - fit_points,
            )

            fit_x = values[
                start:stop
            ]

            fit_y = purity[
                start:stop
            ]

            if len(fit_x) >= 3:

                coefficients = np.polyfit(
                    fit_x,
                    fit_y,
                    deg=2,
                )

                a, b, c = coefficients

                if a < 0:

                    fitted_best_value = (
                        -b / (2 * a)
                    )

                    fitted_best_purity = (
                        np.polyval(
                            coefficients,
                            fitted_best_value,
                        )
                    )

                    fit_curve_x = np.linspace(
                        fit_x.min(),
                        fit_x.max(),
                        200,
                    )

                    fit_curve_y = np.polyval(
                        coefficients,
                        fit_curve_x,
                    )

                    ax_purity.plot(
                        fit_curve_x,
                        fit_curve_y,
                        linestyle="--",
                        label="Quadratic fit",
                    )

                    ax_purity.plot(
                        fitted_best_value,
                        fitted_best_purity,
                        marker="X",
                        markersize=9,
                        linestyle="None",
                        label=(
                            f"Fitted maximum: "
                            f"{value_symbol} = "
                            f"{format(fitted_best_value, value_format)} "
                            f"{value_unit}"
                        ),
                    )

                    if target == "z":
                        self.purityBestZFitted = (
                            fitted_best_value
                            * 1e-3
                        )

                        self.purityBestZFittedValue = (
                            fitted_best_purity
                        )

                    elif target == "wavelength":
                        self.purityBestWavelengthFitted = (
                            fitted_best_value
                            * 1e-9
                        )

                        self.purityBestWavelengthFittedValue = (
                            fitted_best_purity
                        )

                else:

                    self.logger.warning(
                        "Quadratic fit does not produce a local maximum "
                        "(fitted curvature is non-negative)."
                    )

        ax_purity.set_xlabel(
            xlabel
        )

        ax_purity.set_ylabel(
            "Probe purity"
        )

        # ------------------------------------------------------------------
        # Reconstruction error
        # ------------------------------------------------------------------

        ax_error = ax_purity.twinx()

        if errors is not None:

            ax_error.plot(
                values,
                errors,
                marker="s",
                linestyle="--",
                label="Reconstruction error",
            )

            best_error_index = np.argmin(
                errors
            )

            ax_error.plot(
                values[
                    best_error_index
                ],
                errors[
                    best_error_index
                ],
                marker="*",
                markersize=12,
                linestyle="None",
                label=(
                    f"Minimum error: "
                    f"{value_symbol} = "
                    f"{format(values[best_error_index], value_format)} "
                    f"{value_unit}"
                ),
            )

            ax_error.set_ylabel(
                "Reconstruction error"
            )

        # ------------------------------------------------------------------
        # Initial guess
        # ------------------------------------------------------------------

        ax_purity.axvline(
            initial_value,
            linestyle="--",
            alpha=0.5,
            label=(
                f"Initial {value_symbol}: "
                f"{format(initial_value, value_format)} "
                f"{value_unit}"
            ),
        )

        # ------------------------------------------------------------------
        # Combined legend
        # ------------------------------------------------------------------

        purity_handles, purity_labels = (
            ax_purity.get_legend_handles_labels()
        )

        error_handles, error_labels = (
            ax_error.get_legend_handles_labels()
        )

        ax_purity.legend(
            purity_handles + error_handles,
            purity_labels + error_labels,
            loc="best",
        )

        ax_purity.set_title(
            title
        )

        ax_purity.grid(
            alpha=0.3
        )

        fig.tight_layout()

        if show:
            plt.show()

        return (
            fig,
            ax_purity,
            ax_error,
        )


    def plotPurityScan(
        self,
        show=True,
        fit=True,
        fit_range=None,
        fit_fraction=0.25,
    ):
        """
        Plot probe purity and reconstruction error for the selected
        purity-calibration target.

        A local quadratic fit can optionally be performed around the measured
        purity maximum. The fitting window can either be specified explicitly
        through `fit_range` or determined automatically as a fraction of the
        scanned parameter range.

        Args:
            show (bool, optional):
                Display the figure immediately. Default is `True`.

            fit (bool, optional):
                Perform a local quadratic fit around the measured purity maximum.
                Default is `True`.

            fit_range (float, optional):
                Half-width of the fitting window in the displayed unit
                (mm for z and nm for wavelength). If `None`, the fitting window
                is determined from `fit_fraction`.

            fit_fraction (float, optional):
                Fraction of the total scanned parameter range used as the
                fitting half-width when `fit_range` is `None`.
                Default is `0.25`.

        Returns:
            tuple:
                Matplotlib `(figure, purity_axis, error_axis)` objects.
        """

        target = (
            self.params.purityCalibrationTarget.lower()
        )

        # ------------------------------------------------------------------
        # Select calibration results
        # ------------------------------------------------------------------

        if target == "z":

            if not hasattr(
                self,
                "purityZValues",
            ):
                raise RuntimeError(
                    "No purity z-scan results are available."
                )

            values = (
                np.asarray(
                    self.purityZValues
                )
                * 1e3
            )

            purity = np.asarray(
                self.purityZMetrics
            )

            errors = (
                np.asarray(
                    self.purityZErrors
                )
                if hasattr(
                    self,
                    "purityZErrors",
                )
                else None
            )

            initial_value = (
                self.purityZInitialGuess
                * 1e3
            )

            xlabel = (
                "Sample-to-detector distance z (mm)"
            )

            title = (
                "Purity-based axial calibration"
            )

            value_symbol = "z"
            value_unit = "mm"
            value_format = ".3f"

        elif target == "wavelength":

            if not hasattr(
                self,
                "purityWavelengthValues",
            ):
                raise RuntimeError(
                    "No purity wavelength-scan results are available."
                )

            values = (
                np.asarray(
                    self.purityWavelengthValues
                )
                * 1e9
            )

            purity = np.asarray(
                self.purityWavelengthMetrics
            )

            errors = (
                np.asarray(
                    self.purityWavelengthErrors
                )
                if hasattr(
                    self,
                    "purityWavelengthErrors",
                )
                else None
            )

            initial_value = (
                self.purityWavelengthInitialGuess
                * 1e9
            )

            xlabel = (
                "Wavelength λ (nm)"
            )

            title = (
                "Purity-based wavelength calibration"
            )

            value_symbol = "λ"
            value_unit = "nm"
            value_format = ".4f"

        else:

            raise ValueError(
                "Unsupported purity calibration target: "
                f"{self.params.purityCalibrationTarget!r}"
            )

        # ------------------------------------------------------------------
        # Measured maximum
        # ------------------------------------------------------------------

        best_purity_index = np.argmax(
            purity
        )

        measured_best_value = (
            values[
                best_purity_index
            ]
        )

        measured_best_purity = (
            purity[
                best_purity_index
            ]
        )

        fig, ax_purity = plt.subplots(
            figsize=(7, 4.5)
        )

        ax_purity.plot(
            values,
            purity,
            marker="o",
            label="Probe purity",
        )

        ax_purity.plot(
            measured_best_value,
            measured_best_purity,
            marker="*",
            markersize=12,
            linestyle="None",
            label=(
                f"Measured maximum: "
                f"{value_symbol} = "
                f"{format(measured_best_value, value_format)} "
                f"{value_unit}"
            ),
        )

        # ------------------------------------------------------------------
        # Local quadratic fit
        # ------------------------------------------------------------------

        if fit:

            if fit_range is None:

                total_range = (
                    values.max()
                    - values.min()
                )

                fit_half_range = (
                    fit_fraction
                    * total_range
                )

            else:

                fit_half_range = (
                    float(fit_range)
                )

            fit_mask = (
                np.abs(
                    values
                    - measured_best_value
                )
                <= fit_half_range
            )

            fit_x = values[
                fit_mask
            ]

            fit_y = purity[
                fit_mask
            ]

            if len(fit_x) < 3:

                self.logger.warning(
                    "Quadratic fit skipped because fewer than "
                    "three scan points fall inside the fitting window."
                )

            else:

                coefficients = np.polyfit(
                    fit_x,
                    fit_y,
                    deg=2,
                )

                a, b, c = coefficients

                fitted_y = np.polyval(
                    coefficients,
                    fit_x,
                )

                residual_sum = np.sum(
                    (
                        fit_y
                        - fitted_y
                    )
                    ** 2
                )

                total_sum = np.sum(
                    (
                        fit_y
                        - np.mean(fit_y)
                    )
                    ** 2
                )

                if total_sum > 0:

                    fit_r2 = (
                        1
                        - residual_sum
                        / total_sum
                    )

                else:

                    fit_r2 = np.nan

                fitted_best_value = (
                    -b
                    / (2 * a)
                    if a != 0
                    else np.nan
                )

                valid_fit = (
                    a < 0
                    and np.isfinite(
                        fitted_best_value
                    )
                    and fit_x.min()
                    <= fitted_best_value
                    <= fit_x.max()
                )

                if valid_fit:

                    fitted_best_purity = (
                        np.polyval(
                            coefficients,
                            fitted_best_value,
                        )
                    )

                    fit_curve_x = np.linspace(
                        fit_x.min(),
                        fit_x.max(),
                        300,
                    )

                    fit_curve_y = np.polyval(
                        coefficients,
                        fit_curve_x,
                    )

                    ax_purity.plot(
                        fit_curve_x,
                        fit_curve_y,
                        linestyle="--",
                        label=(
                            f"Quadratic fit "
                            f"($R^2$ = {fit_r2:.3f})"
                        ),
                    )

                    ax_purity.plot(
                        fitted_best_value,
                        fitted_best_purity,
                        marker="X",
                        markersize=9,
                        linestyle="None",
                        label=(
                            f"Fitted maximum: "
                            f"{value_symbol} = "
                            f"{format(fitted_best_value, value_format)} "
                            f"{value_unit}"
                        ),
                    )

                    if target == "z":

                        self.purityBestZFitted = (
                            fitted_best_value
                            * 1e-3
                        )

                        self.purityBestZFittedValue = (
                            fitted_best_purity
                        )

                        self.purityBestZFitR2 = (
                            fit_r2
                        )

                    elif target == "wavelength":

                        self.purityBestWavelengthFitted = (
                            fitted_best_value
                            * 1e-9
                        )

                        self.purityBestWavelengthFittedValue = (
                            fitted_best_purity
                        )

                        self.purityBestWavelengthFitR2 = (
                            fit_r2
                        )

                else:

                    self.logger.warning(
                        "Quadratic fit did not produce a valid local maximum. "
                        "The fitted curvature may be non-negative or the fitted "
                        "maximum may lie outside the fitting window."
                    )

        ax_purity.set_xlabel(
            xlabel
        )

        ax_purity.set_ylabel(
            "Probe purity"
        )

        # ------------------------------------------------------------------
        # Reconstruction error
        # ------------------------------------------------------------------

        ax_error = ax_purity.twinx()

        if errors is not None:

            ax_error.plot(
                values,
                errors,
                marker="s",
                linestyle="--",
                label="Reconstruction error",
            )

            best_error_index = np.argmin(
                errors
            )

            ax_error.plot(
                values[
                    best_error_index
                ],
                errors[
                    best_error_index
                ],
                marker="*",
                markersize=12,
                linestyle="None",
                label=(
                    f"Minimum error: "
                    f"{value_symbol} = "
                    f"{format(values[best_error_index], value_format)} "
                    f"{value_unit}"
                ),
            )

            ax_error.set_ylabel(
                "Reconstruction error"
            )

        # ------------------------------------------------------------------
        # Initial guess
        # ------------------------------------------------------------------

        ax_purity.axvline(
            initial_value,
            linestyle="--",
            alpha=0.5,
            label=(
                f"Initial {value_symbol}: "
                f"{format(initial_value, value_format)} "
                f"{value_unit}"
            ),
        )

        # ------------------------------------------------------------------
        # Combined legend
        # ------------------------------------------------------------------

        purity_handles, purity_labels = (
            ax_purity.get_legend_handles_labels()
        )

        error_handles, error_labels = (
            ax_error.get_legend_handles_labels()
        )

        ax_purity.legend(
            purity_handles + error_handles,
            purity_labels + error_labels,
            loc="best",
        )

        ax_purity.set_title(
            title
        )

        ax_purity.grid(
            alpha=0.3
        )

        fig.tight_layout()

        if show:
            plt.show()

        return (
            fig,
            ax_purity,
            ax_error,
        )