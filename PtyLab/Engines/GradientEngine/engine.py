"""Automatic differentiation engine based on the optional torch dependency for single-mode CPM.

Import explicitly with ``from PtyLab.Engines.GradientEngine import GradientEngine``.
The forward model and Adam updates stay in Torch; Reconstruction and Monitor
receive independent NumPy snapshots after each iteration. Changes to those
snapshots during a run do not change the Torch parameters.
"""

from numbers import Integral

import numpy as np
import tqdm

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise ImportError(
        "GradientEngine requires PyTorch. Install with: pip install torch"
    ) from exc

from PtyLab.Engines.BaseEngine import BaseEngine

from .losses import amplitude_loss
from .models import SingleSliceModel
from .propagation import KernelPropagator


class GradientEngine(BaseEngine):
    """Optimize complex object and probe using a normalized amplitude loss.

    Uses the standard ``(reconstruction, experimentalData, params, monitor)``
    constructor and ``reconstruct()`` API. Set ``numIterations``,
    ``learningRateObject``, ``learningRateProbe`` and ``device`` on the engine.
    ``device`` defaults to ``"cpu"``; use ``"cuda"`` for Torch GPU execution,
    independently of CuPy. Inputs are NumPy arrays; keep
    ``params.gpuSwitch = False``.

    One iteration accumulates gradients over all scan positions, then takes one
    Adam step. Set ``batchSize`` (default 1) to process several frames together,
    e.g. ``engine.batchSize = 8``. Larger batches retain more intermediate fields
    and use more memory; each batch's graph is released after backward.
    ``reconstruction.error`` records the pre-update objective, including any
    regularizers. By default this is the sum of squared amplitude residuals
    divided by total measured intensity (not the PIE error metric).
    Each call starts a fresh optimizer from the current Reconstruction arrays.

    Components can be assigned before reconstruct(): model (validate + callable),
    lossFunction(field, measuredAmplitude, totalPower), and a list of regularizers
    taking (objectTensor, probeTensor). Defaults reproduce the unregularized
    amplitude objective. Custom models currently must return a 2D complex field;
    trainable model parameters must be registered through parameterGroups().
    Batching custom models or overriding forward requires a forwardBatch override
    returning [batch, height, width]. Custom losses must sum normalized frame
    losses across the batch, rather than average them.

    Subclass hooks: initializeParameters / parameterGroups for extra trainable
    tensors, createOptimizer for the update rule, forward / propagate for the
    physics, computeLoss / regularizationLoss for the objective, and afterStep
    for in-place constraints. Use Torch operations throughout these hooks.
    Override _sync_reconstruction too if extra parameters need NumPy outputs.
    Remove a setting from unsupportedSettings only after implementing it.

    Initially supports one wavelength, object mode, probe mode and slice, fixed
    integer positions, Fraunhofer/Fresnel/ASP/scaledASP propagation and no PIE
    constraints. Geometry kernels are fixed during each run; gradients flow
    through object and probe, not propagation distance or pixel spacing.
    """

    def __init__(self, reconstruction, experimentalData, params, monitor):
        """Attach PtyLab state and set default model, loss and optimizer settings."""
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.numIterations = 100
        self.batchSize = 1
        self.learningRateObject = 0.03
        self.learningRateProbe = 0.01
        self.device = "cpu"
        self.model = SingleSliceModel()
        self.lossFunction = amplitude_loss
        self.regularizers = []

    unsupportedSettings = (
        "objectTVregSwitch",
        "FourierMaskSwitch",
        "CPSCswitch",
        "orthogonalizationSwitch",
        "objectSmoothenessSwitch",
        "absObjectSwitch",
        "objectContrastSwitch",
        "probeSmoothenessSwitch",
        "probeBoundary",
        "absorbingProbeBoundary",
        "probePowerCorrectionSwitch",
        "probeSpectralPowerCorrectionSwitch",
        "modulusEnforcedProbeSwitch",
        "absProbeSwitch",
        "couplingSwitch",
        "binaryProbeSwitch",
        "backgroundModeSwitch",
        "comStabilizationSwitch",
        "PSDestimationSwitch",
        "positionCorrectionSwitch",
        "adaptiveDenoisingSwitch",
        "l2reg",
        "TV_autofocus",
        "OPRP",
        "SHG_probe",
        "momentumAcceleration",
        "adaptiveMomentumAcceleration",
        "weigh_probe_updates_by_intensity",
    )

    supportedPropagators = ("Fraunhofer", "Fresnel", "ASP", "scaledASP")

    def validateSettings(self):
        """Reject incompatible data modes, batching and legacy engine settings.

        Model-specific shape checks are delegated to model.validate().
        Subclasses can extend supportedPropagators and unsupportedSettings
        after implementing the corresponding behavior.
        """
        if (
            isinstance(self.batchSize, (bool, np.bool_))
            or not isinstance(self.batchSize, Integral)
            or self.batchSize < 1
        ):
            raise ValueError("batchSize must be a positive integer.")
        if self.batchSize > 1:
            uses_default_batch = (
                getattr(self.forwardBatch, "__func__", None)
                is GradientEngine.forwardBatch
            )
            if uses_default_batch:
                has_custom_forward = (
                    type(self.model) is not SingleSliceModel
                    or getattr(self.forward, "__func__", None)
                    is not GradientEngine.forward
                )
                if has_custom_forward:
                    raise NotImplementedError(
                        "Override forwardBatch to use batchSize > 1 with a custom model or forward."
                    )
        r, p = self.reconstruction, self.params
        if self.experimentalData.operationMode != "CPM":
            raise NotImplementedError("GradientEngine currently supports CPM only.")
        if p.propagatorType.lower() not in [
            name.lower() for name in self.supportedPropagators
        ]:
            raise NotImplementedError(
                f"Supported propagators: {self.supportedPropagators}"
            )
        self.model.validate(r)
        if p.gpuSwitch or getattr(p, "gpuFlag", False):
            raise ValueError(
                "Set params.gpuSwitch = False; select Torch's device via engine.device."
            )
        if p.fftshiftSwitch or p.fftshiftFlag:
            raise NotImplementedError(
                "GradientEngine requires centered data and fftshiftSwitch = False."
            )
        if p.intensityConstraint != "standard":
            raise NotImplementedError(
                "GradientEngine uses the standard amplitude loss only."
            )
        if p.objectUpdateStart != 1 or p.probeUpdateStart != 1:
            raise NotImplementedError(
                "GradientEngine updates both object and probe from iteration 1."
            )
        # These operations mutate NumPy/CuPy state in BaseEngine and must not
        # silently be applied to detached copies of the Torch parameters.
        enabled = [name for name in self.unsupportedSettings if getattr(p, name, False)]
        if enabled:
            raise NotImplementedError(
                "GradientEngine does not support: " + ", ".join(enabled)
            )
        if self.numIterations < 1:
            raise ValueError("numIterations must be positive.")

    # Keep the Fourier helpers available to subclasses.
    fft2c = staticmethod(KernelPropagator.fft2c)
    ifft2c = staticmethod(KernelPropagator.ifft2c)

    def preparePropagation(self):
        """Create the field propagator once per run from the current geometry.

        Override this hook to provide a different propagation callable. Learnable
        geometry requires kernels built in Torch during each forward pass.
        """
        self.propagation = KernelPropagator(
            self.reconstruction, self.params.propagatorType, self.device
        )

    def propagate(self, object_patch, probe):
        """Return D[O_patch * P], preserving any leading batch dimensions.

        O_patch and P are complex object/probe tensors; multiplication is
        pointwise and D is the prepared propagation callable.
        """
        return self.propagation(object_patch * probe)

    def _sync_reconstruction(self):
        """Copy detached object/probe estimates into NumPy reconstruction state."""
        # copy() is essential on CPU: numpy() alone would share Torch storage.
        self.reconstruction.object = self.objectTensor.detach().cpu().numpy().copy()
        self.reconstruction.probe = self.probeTensor.detach().cpu().numpy().copy()

    def prepareReconstruction(self):
        """Validate data and initialize tensors, propagation, optimizer and monitor.

        Diffraction data must have shape [frames, Np, Np], finite nonnegative
        intensities and positive total power. Scan patches must fit the object.
        Measured amplitudes use float32; trainable fields use complex64.
        Resets the recorded objective history for this run.
        """
        self.validateSettings()
        r = self.reconstruction
        self.positions = np.asarray(r.positions)
        intensity = np.asarray(self.experimentalData.ptychogram)
        if intensity.shape != (len(self.positions), r.Np, r.Np) or not len(
            self.positions
        ):
            raise ValueError(
                "Expected one probe-sized diffraction frame per scan position."
            )
        if (
            not np.isfinite(intensity).all()
            or np.any(intensity < 0)
            or not intensity.sum() > 0
        ):
            raise ValueError(
                "Diffraction intensities must be finite, nonnegative and have positive power."
            )
        if np.any(self.positions < 0) or np.any(self.positions + r.Np > r.No):
            raise ValueError("Scan positions place a probe patch outside the object.")

        # a_j = sqrt(y_j); P_total = sum_{j,pixel} y_j over the entire scan.
        self.measuredAmplitudes = torch.tensor(
            intensity, dtype=torch.float32, device=self.device
        ).sqrt()
        self.totalPower = self.measuredAmplitudes.square().sum()
        if self.batchSize > 1:
            self.positionTensor = torch.tensor(
                self.positions, dtype=torch.long, device=self.device
            )
            self.patchOffsets = torch.arange(r.Np, device=self.device)
        self.initializeParameters()
        self.preparePropagation()
        self.optimizer = self.createOptimizer()
        r.error = []
        self._sync_reconstruction()
        self._setObjectProbeROI(update=True)
        self._showInitialGuesses()

    def initializeParameters(self):
        """Create trainable tensors. Extend this to add learnable parameters."""
        r = self.reconstruction
        self.objectTensor = torch.nn.Parameter(
            torch.tensor(r.object, dtype=torch.complex64, device=self.device)
        )
        self.probeTensor = torch.nn.Parameter(
            torch.tensor(r.probe, dtype=torch.complex64, device=self.device)
        )

    def parameterGroups(self):
        """Return optimizer groups; subclasses can append their own parameters."""
        return [
            {"params": [self.objectTensor], "lr": self.learningRateObject},
            {"params": [self.probeTensor], "lr": self.learningRateProbe},
        ]

    def createOptimizer(self):
        """Create Adam with separate object/probe learning rates.

        Override to use another Torch optimizer with the same parameter groups.
        """
        return torch.optim.Adam(self.parameterGroups())

    def forward(self, positionIndex):
        """Predict a complex detector field for one scan position, entirely in Torch."""
        return self.model(
            self.objectTensor,
            self.probeTensor,
            self.positions[positionIndex],
            self.propagate,
        )

    def forwardBatch(self, indices):
        """Predict [batch, height, width] fields for device-resident scan indices.

        Override alongside custom models/forward implementations when batching.
        """
        # O_j[y, x] = O[row_j + y, col_j + x]; the probe broadcasts over j.
        positions = self.positionTensor[indices]
        rows = positions[:, 0, None, None] + self.patchOffsets[None, :, None]
        cols = positions[:, 1, None, None] + self.patchOffsets[None, None, :]
        obj = self.objectTensor.reshape(self.reconstruction.No, self.reconstruction.No)
        probe = self.probeTensor.reshape(self.reconstruction.Np, self.reconstruction.Np)
        return self.propagate(obj[rows, cols], probe)

    def computeLoss(self, detectorWave, measuredAmplitude):
        """Return a real scalar loss for one frame or a batch of frames.

        Pass the complex predicted field, measured amplitude and full-scan
        power to lossFunction. A custom loss must sum over its input frames
        to keep the full-scan objective independent of batch size.
        """
        return self.lossFunction(detectorWave, measuredAmplitude, self.totalPower)

    def regularizationLoss(self):
        """Return R(O, P) = sum_k R_k(O, P), or None when no penalties are set.

        Each configured callable receives the trainable object and probe and
        returns a real scalar tensor. Its weight belongs inside the callable.
        """
        if not self.regularizers:
            return None
        return sum(
            penalty(self.objectTensor, self.probeTensor)
            for penalty in self.regularizers
        )

    def afterStep(self, iteration):
        """Optional tensor constraints after an optimizer step, under torch.no_grad.

        Update parameters in place to preserve the optimizer's references.
        """
        pass

    def runIteration(self, iteration):
        """Take one optimizer step using all frames and return the pre-step loss.

        J(O, P) = sum_b L_b(O, P) + R(O, P), with b indexing scan batches.
        Each L_b uses the full-scan normalization. Backward calls accumulate
        gradients at the same parameter values before the single update.
        The diffraction monitor receives the last processed pre-update frame.
        iteration is the zero-based index passed to the afterStep hook.
        """
        self.setPositionOrder()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=self.device)
        if self.batchSize > 1:
            indices = torch.tensor(
                self.positionIndices, dtype=torch.long, device=self.device
            )
        for start in range(0, len(self.positionIndices), self.batchSize):
            if self.batchSize == 1:
                index = self.positionIndices[start]
                detector_wave = self.forward(index)
                measured = self.measuredAmplitudes[index]
            else:
                batch = indices[start : start + self.batchSize]
                detector_wave = self.forwardBatch(batch)
                measured = self.measuredAmplitudes[batch]
            loss = self.computeLoss(detector_wave, measured)
            # grad J = sum_b grad L_b + grad R; free each batch graph here.
            loss.backward()
            total_loss += loss.detach()
            # Retain only the final frame, without its graph or batch storage.
            if start + self.batchSize >= len(self.positionIndices):
                last_wave = detector_wave if self.batchSize == 1 else detector_wave[-1]
                last_wave = last_wave.detach().clone()
            del loss, detector_wave, measured

        penalty = self.regularizationLoss()
        if penalty is not None:
            penalty.backward()
            total_loss += penalty.detach()

        self.optimizer.step()
        with torch.no_grad():
            self.afterStep(iteration)

        # Keep the last frame for the existing diffraction monitor.
        self.reconstruction.Iestimated = last_wave.abs().square().cpu().numpy().copy()
        self.reconstruction.Imeasured = (
            self.measuredAmplitudes[self.positionIndices[-1]]
            .square()
            .cpu()
            .numpy()
            .copy()
        )
        return total_loss.item()

    def _replace_experimental_data(self, experimentalData):
        """Replace measurements and positions after checking grid compatibility."""
        previous = self.reconstruction.data
        geometry_fields = (
            "operationMode",
            "wavelength",
            "dxd",
            "zo",
            "theta",
            "spectralDensity",
            "entrancePupilDiameter",
        )
        changed = [
            name
            for name in geometry_fields
            if not np.array_equal(
                getattr(previous, name), getattr(experimentalData, name)
            )
        ]
        if previous.ptychogram.shape[1:] != experimentalData.ptychogram.shape[1:]:
            changed.append("detector shape")
        if changed:
            raise ValueError(
                "Replacement data changes acquisition geometry ("
                + ", ".join(changed)
                + "); create a new Reconstruction."
            )
        self.experimentalData = experimentalData
        self.reconstruction.data = experimentalData
        self.reconstruction.reset_positioncorrection()

    def reconstruct(self, experimentalData=None):
        """Run numIterations from the current estimates with a fresh optimizer.

        experimentalData optionally replaces measurements and scan coordinates,
        resetting corrected positions to a copy of its encoder. Acquisition
        geometry must match the previous dataset; changed geometry requires a
        new Reconstruction. Without replacement, corrected positions are kept.
        Record each pre-update objective and publish NumPy estimates for the
        monitor. A final synchronization also runs if reconstruction raises.
        Results are stored in self.reconstruction; this method returns None.
        """
        if experimentalData is not None:
            self._replace_experimental_data(experimentalData)
        self.prepareReconstruction()
        try:
            with tqdm.trange(self.numIterations, desc=type(self).__name__) as self.pbar:
                for iteration in self.pbar:
                    loss = self.runIteration(iteration)
                    self.reconstruction.error.append(loss)
                    self._sync_reconstruction()
                    self.showReconstruction(iteration)
                    self.pbar.set_postfix(loss=loss)
        finally:
            self._sync_reconstruction()
