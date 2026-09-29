"""Automatic-differentiation ptychography with optional PyTorch that subclasses from the `BaseEngine`"""

from collections.abc import Mapping
from numbers import Integral, Real

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
from .models import PtychographyModel, SharedProbe
from .propagation import KernelPropagator


class GradientEngine(BaseEngine):
    """Estimate selected physical parameters from a full ptychographic scan.

    The standard PtyLab constructor and ``reconstruct()`` entry point are
    preserved. Set ``parameters`` to a mapping from model parameter names to
    optimizer options::

        engine.parameters = {
            "object": {"lr": 0.03},
            "probe.field": {"lr": 0.01},
        }

    Omitted parameters remain fixed. ``parameters=None`` selects object and
    probe with ``learningRateObject`` and ``learningRateProbe`` for compatibility.
    Every reconstruction call creates a fresh optimizer while retaining the
    component values from the previous call. ``reset()`` explicitly imports the
    detached arrays currently stored on the Reconstruction.

    ``optimizerType = "preconditioned"`` replaces Adam with PIE-style gradient
    descent: each gradient is divided by the regularized curvature map from
    ``model.gradient_preconditioners`` before a plain step. Its learning rates
    are dimensionless; about 0.3 suits the amplitude loss. ``updatePerBatch``
    takes one optimizer step per batch instead of one per full scan.

    ``batchSize = None`` (the default) forwards the whole scan as one tensor;
    set an integer only to bound memory. For Fraunhofer data
    ``measuredIntensities`` is stored in unshifted FFT order, see
    ``KernelPropagator.intensity_field``.
    """

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

    def __init__(self, reconstruction, experimentalData, params, monitor):
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.numIterations = 100
        self.batchSize = None
        self.learningRateObject = 0.03
        self.learningRateProbe = 0.01
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parameters = None
        self.model = None
        self.lossFunction = amplitude_loss
        self.regularizers = []
        self.optimizerType = "adam"
        self.preconditionerAlpha = 0.1
        self.updatePerBatch = False

    def validateSettings(self):
        """Reject inputs outside the Stage 1 CPM and single-state contract."""
        if self.batchSize is not None and (
            isinstance(self.batchSize, (bool, np.bool_))
            or not isinstance(self.batchSize, Integral)
            or self.batchSize < 1
        ):
            raise ValueError("batchSize must be a positive integer or None.")
        r, p = self.reconstruction, self.params
        if self.experimentalData.operationMode != "CPM":
            raise NotImplementedError("GradientEngine currently supports CPM only.")
        if p.propagatorType.lower() not in {
            name.lower() for name in self.supportedPropagators
        }:
            raise NotImplementedError(
                f"Supported propagators: {self.supportedPropagators}"
            )
        validator = (
            self.model.validate
            if self.model is not None
            else PtychographyModel.validate
        )
        validator(r)
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
                "GradientEngine requires the standard detector constraint setting."
            )
        if p.objectUpdateStart != 1 or p.probeUpdateStart != 1:
            raise NotImplementedError(
                "Use engine.parameters to select object and probe estimation."
            )
        enabled = [name for name in self.unsupportedSettings if getattr(p, name, False)]
        if enabled:
            raise NotImplementedError(
                "GradientEngine does not support: " + ", ".join(enabled)
            )
        if self.numIterations < 1:
            raise ValueError("numIterations must be positive.")
        if self.optimizerType not in ("adam", "preconditioned"):
            raise ValueError('optimizerType must be "adam" or "preconditioned".')
        alpha = self.preconditionerAlpha
        if (
            isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, Real)
            or not 0 < alpha <= 1
        ):
            raise ValueError("preconditionerAlpha must lie in (0, 1].")

    fft2c = staticmethod(KernelPropagator.fft2c)
    ifft2c = staticmethod(KernelPropagator.ifft2c)

    def preparePropagation(self):
        """Create fixed propagation kernels on the selected Torch device."""
        self.propagation = KernelPropagator(
            self.reconstruction, self.params.propagatorType, self.device
        )

    def propagate(self, exit_wave):
        """Propagate an exit wave while preserving all leading dimensions."""
        return self.propagation(exit_wave)

    def createModel(self):
        """Create the default object and shared-probe component model."""
        r = self.reconstruction
        return PtychographyModel(r.object, SharedProbe(r.probe), r.positions)

    def initializeParameters(self):
        """Create components once, or move retained components to the run device."""
        if self.model is None:
            self.model = self.createModel()
        self.model.to(self.device)
        self.model.set_positions(self.positions)
        self.model.set_propagation(
            self.propagation, getattr(self.propagation, "intensity_field", None)
        )

    def reset(self):
        """Import current Reconstruction object/probe arrays into the components."""
        PtychographyModel.validate(self.reconstruction)
        if self.model is None:
            self.model = self.createModel()
        else:
            if not hasattr(self.model, "reset_from_reconstruction"):
                raise TypeError(
                    "A custom model must implement reset_from_reconstruction()."
                )
            self.model.to(self.device)
            self.model.reset_from_reconstruction(self.reconstruction)
        self.model.to(self.device)
        return self

    def snapshot(self):
        """Return detached CPU NumPy copies of the current physical parameters."""
        if self.model is None:
            return {
                "object": np.array(self.reconstruction.object, copy=True),
                "probe.field": np.array(self.reconstruction.probe, copy=True),
            }
        return {
            "object": self.model.object.detach().cpu().numpy().copy(),
            "probe.field": self.model.probe.field.detach().cpu().numpy().copy(),
        }

    def _sync_reconstruction(self):
        """Publish detached object/probe snapshots for monitoring and persistence."""
        state = self.snapshot()
        self.reconstruction.object = state["object"]
        self.reconstruction.probe = state["probe.field"]

    def _selection(self):
        if self.parameters is None:
            return {
                "object": {"lr": self.learningRateObject},
                "probe.field": {"lr": self.learningRateProbe},
            }
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping or None.")
        if not self.parameters:
            raise ValueError("parameters must select at least one model parameter.")
        return self.parameters

    def parameterGroups(self):
        """Validate the public selection and return optimizer parameter groups."""
        named_items = list(self.model.named_parameters(remove_duplicate=False))
        named = dict(named_items)
        selection = self._selection()
        unknown = sorted(set(selection) - set(named))
        if unknown:
            available = ", ".join(named)
            raise KeyError(
                f"Unknown parameter name(s): {', '.join(unknown)}. Available: {available}."
            )

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        groups = []
        owners = {}
        for name, options in selection.items():
            if not isinstance(options, Mapping):
                raise TypeError(f"Configuration for {name!r} must be a mapping.")
            extra = set(options) - {"lr"}
            if extra:
                raise KeyError(
                    f"Unsupported option(s) for {name!r}: {', '.join(sorted(extra))}."
                )
            if "lr" not in options:
                raise KeyError(f"Parameter {name!r} requires an 'lr' value.")
            rate = options["lr"]
            if (
                isinstance(rate, (bool, np.bool_))
                or not isinstance(rate, Real)
                or not np.isfinite(rate)
                or rate <= 0
            ):
                raise ValueError(
                    f"Learning rate for {name!r} must be positive and finite."
                )
            parameter = named[name]
            owner = owners.get(id(parameter))
            if owner is not None:
                raise ValueError(
                    f"Parameters {owner!r} and {name!r} refer to the same tensor."
                )
            owners[id(parameter)] = name
            parameter.requires_grad_(True)
            groups.append({"params": [parameter], "lr": float(rate), "name": name})
        return groups

    def createOptimizer(self):
        """Create a fresh optimizer of ``optimizerType`` for the selected parameters."""
        if self.optimizerType == "preconditioned":
            return torch.optim.SGD(self.parameterGroups(), foreach=False)
        return torch.optim.Adam(self.parameterGroups(), foreach=False)

    def preconditionGradients(self, indices):
        """Divide gradients by regularized curvature maps of the stepped frames.

        With curvature ``d`` the denominator is ``(1 - a) d + a max(d)`` for
        ``a = preconditionerAlpha``, as in rPIE, divided by the full-scan power
        that normalizes every loss.
        """
        curvature = self.model.gradient_preconditioners(indices)
        alpha = self.preconditionerAlpha
        for group in self.optimizer.param_groups:
            (parameter,) = group["params"]
            if parameter.grad is None:
                continue
            if group["name"] not in curvature:
                raise KeyError(
                    f"No preconditioner for parameter {group['name']!r}; "
                    "extend model.gradient_preconditioners."
                )
            values = curvature[group["name"]]
            denominator = (1 - alpha) * values + alpha * values.max()
            parameter.grad.div_(denominator / self.totalPower)

    def _validate_data(self):
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
        return intensity

    def prepareReconstruction(self):
        """Prepare data, retained components, propagation, and a fresh optimizer."""
        self.validateSettings()
        intensity = self._validate_data()
        self.preparePropagation()
        self.fftOrderIntensity = getattr(self.propagation, "fftOrderIntensity", False)
        self.measuredIntensities = torch.tensor(
            intensity, dtype=torch.float32, device=self.device
        )
        if self.fftOrderIntensity:
            self.measuredIntensities = torch.fft.ifftshift(
                self.measuredIntensities, dim=(-2, -1)
            )
        self.totalPower = self.measuredIntensities.sum()
        self.initializeParameters()
        self.optimizer = self.createOptimizer()
        self.reconstruction.error = []
        self._sync_reconstruction()
        self._setObjectProbeROI(update=True)
        self._showInitialGuesses()

    def forward(self, indices):
        """Predict detector intensities for one device-resident index batch."""
        return self.model(indices)

    def computeLoss(self, predictedIntensity, measuredIntensity):
        """Evaluate the configured full-scan-normalized data objective."""
        return self.lossFunction(predictedIntensity, measuredIntensity, self.totalPower)

    def regularizationLoss(self):
        """Return the sum of penalties receiving the current model state."""
        if not self.regularizers:
            return None
        return sum(penalty(self.model) for penalty in self.regularizers)

    def afterStep(self, iteration):
        """Apply optional in-place component constraints under ``torch.no_grad``."""
        pass

    def runIteration(self, iteration):
        """Accumulate the full-scan objective and take one optimizer step.

        With ``updatePerBatch`` every batch takes its own step, and each step
        carries the regularization penalty weighted by its share of the frames.
        ``afterStep`` then runs after every step.
        """
        self.setPositionOrder()
        total_loss = torch.zeros((), device=self.device)
        indices = torch.as_tensor(
            self.positionIndices, dtype=torch.long, device=self.device
        )
        batch_size = len(indices) if self.batchSize is None else int(self.batchSize)
        batches = list(torch.split(indices, batch_size))
        steps = [[batch] for batch in batches] if self.updatePerBatch else [batches]
        for step in steps:
            self.optimizer.zero_grad(set_to_none=True)
            for batch in step:
                predicted = self.forward(batch)
                measured = self.measuredIntensities[batch]
                loss = self.computeLoss(predicted, measured)
                loss.backward()
                total_loss += loss.detach()
                if batch is batches[-1]:
                    last_intensity = predicted[-1].detach().clone()
                del loss, predicted, measured

            step_indices = torch.cat(step)
            penalty = self.regularizationLoss()
            if penalty is not None:
                if len(step_indices) != len(indices):
                    penalty = penalty * (len(step_indices) / len(indices))
                penalty.backward()
                total_loss += penalty.detach()

            if self.optimizerType == "preconditioned":
                self.preconditionGradients(step_indices)
            self.optimizer.step()
            with torch.no_grad():
                self.afterStep(iteration)

        self.reconstruction.Iestimated = self.centeredIntensity(last_intensity)
        self.reconstruction.Imeasured = self.centeredIntensity(
            self.measuredIntensities[self.positionIndices[-1]]
        )
        return total_loss.item()

    def centeredIntensity(self, intensity):
        """Return a detached NumPy detector image in PtyLab's centered order."""
        if self.fftOrderIntensity:
            intensity = torch.fft.fftshift(intensity, dim=(-2, -1))
        return intensity.detach().cpu().numpy().copy()

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
        """Run from retained component values with a fresh optimizer."""
        if experimentalData is not None:
            self._replace_experimental_data(experimentalData)
        try:
            self.prepareReconstruction()
            with tqdm.trange(self.numIterations, desc=type(self).__name__) as self.pbar:
                for iteration in self.pbar:
                    loss = self.runIteration(iteration)
                    self.reconstruction.error.append(loss)
                    self._sync_reconstruction()
                    self.showReconstruction(iteration)
                    self.pbar.set_postfix(loss=loss)
        finally:
            if self.model is not None:
                self._sync_reconstruction()
