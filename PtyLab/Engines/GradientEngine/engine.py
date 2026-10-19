"""
Differentiable ptychography engine with PyTorch, built on
`BaseEngine`.
"""

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
        "GradientEngine requires PyTorch. Install with: uv pip install 'ptylab[torch]'"
    ) from exc

from PtyLab.Engines.BaseEngine import BaseEngine

from .batching import (
    MeasurementLoader,
    data_device,
    iteration_peak,
    largest_batch_within,
    split_indices,
)
from .losses import amplitude_loss
from .models import PtychographyModel, SharedProbe, probe_scale
from .phases import _SETTINGS, Phase
from .propagation import KernelPropagator


def _is_number(value, kind=Real):
    """Return whether `value` is a `kind` number; booleans are not numbers here."""
    return isinstance(value, kind) and not isinstance(value, bool)


class GradientEngine(BaseEngine):
    r"""
    Estimate selected physical parameters from a full ptychographic scan by
    automatic differentiation.

    The standard PtyLab constructor and `reconstruct()` entry point are
    preserved. The model predicts the detector intensity of every frame,

    $$
    I_j = \left|\mathcal{D}\left\lbrace O(\mathbf{r} + \mathbf{x}_j)\, P(\mathbf{r})\right\rbrace \right|^2 + b,
    $$

    and an optimizer minimizes `lossFunction` between $I_j$ and the measured
    frames, summed over the scan, plus any `regularizers`.

    Set `parameters` to a mapping from model parameter names to optimizer
    options:

    ```python
    engine.parameters = {
        "object": {"lr": 0.03},
        "probe.field": {"lr": 0.01},
    }
    ```

    Omitted parameters remain fixed. `parameters = None` selects object and
    probe with `learningRateObject` and `learningRateProbe`.

    The probe is optimized scale-free: `probe.field` stores $F = P/s$, with $s$
    set from the measured photons per frame at every preparation and `reset()`
    (see `models.probe_scale`). Probe learning rates are therefore
    dimensionless, like the object's; an older Adam rate `lr` for the physical
    probe corresponds to `lr / s`. `snapshot()`, `reconstruction.probe` and
    `model.probe(indices)` stay in physical units.

    Every reconstruction call creates a fresh optimizer while retaining the
    component values from the previous call. `reset()` explicitly imports the
    detached arrays currently stored on the `Reconstruction`. `run(phases)`
    runs a sequence of `Phase` settings, each with a fresh optimizer;
    `reconstruct()` is a single phase with the engine's current settings.

    NumPy is used only at PtyLab boundaries: reading `ExperimentalData` and
    `Reconstruction` arrays, building propagation kernels, and publishing
    detached results. Measurements, backgrounds and all per-iteration work are
    tensors on `device`.

    Args:
        reconstruction (Reconstruction):
            Mutable reconstruction state containing the current object, probe,
            geometry, and reconstruction results.

        experimentalData (ExperimentalData):
            Experimental diffraction data and acquisition geometry.

        params (Params):
            Reconstruction parameters. Switches the engine does not support
            are listed in `unsupportedSettings` and rejected.

        monitor (Monitor):
            Monitor used to visualize and report reconstruction progress.

    Attributes:
        numIterations (int):
            Full-scan iterations per `reconstruct()` call. Initialized to 100.

        parameters (Mapping or None):
            Selected model parameters and their optimizer options. Initialized
            to None.

        learningRateObject (float):
            Object learning rate when `parameters` is None. Initialized to
            0.03.

        learningRateProbe (float):
            Scale-free probe learning rate when `parameters` is None.
            Initialized to 0.1.

        lossFunction (callable):
            Data objective from `losses`, optionally wrapped in
            `functools.partial`. Initialized to `amplitude_loss`.

        regularizers (list):
            Penalties called with the model, added once per full scan.
            Initialized to an empty list.

        optimizerType (str):
            `"adam"`, or `"preconditioned"` for PIE-style gradient descent:
            each gradient is divided by the regularized curvature map from
            `model.gradient_preconditioners` before a plain step. Its learning
            rates are dimensionless; about 0.3 suits the amplitude loss.
            Initialized to `"adam"`.

        preconditionerAlpha (float):
            Regularization $\alpha$ of the preconditioner, see
            `preconditionGradients`. Initialized to 0.1.

        updatePerBatch (bool):
            Take one optimizer step per batch instead of one per full scan.
            With it, `params.positionOrder = "random"` reshuffles the frames
            every iteration, so each batch samples the whole scan. Initialized
            to False.

        constraints (list):
            In-place operations applied after every optimizer step, each with
            its own schedule (see `constraints`), for example
            `[ProbeCentering(), ObjectAmplitudeClamp(1.0)]`. `afterStep` still
            runs after them for subclasses. Initialized to an empty list.

        batchSize (int or None):
            Frames forwarded per batch. None forwards the whole scan as one
            tensor; set an integer only to bound memory. The step stays one
            per full scan, with gradients accumulated over batches, so the
            result does not depend on it. Initialized to None.

        memoryBudget (float or None):
            CUDA only: bytes of allocated CUDA memory of the process (other
            tensors included) that one iteration may use. The largest batch
            within it is picked, see `suggestBatchSize`. Initialized to None.

        dataDevice (str or None):
            Where the measurements live. None keeps them on `device` unless
            they would take more than half of the free GPU memory; `"cpu"`
            keeps them in pinned host memory and copies each batch over while
            the previous one is computed. Initialized to None.

        background (float or array_like or None):
            Incoherent detector background $b$ that the model adds to every
            predicted frame, so the data are compared with
            $|\mathcal{D}\lbrace \psi\rbrace |^2 + b$ instead of being
            background-subtracted. Every loss counts $b$ as photons, with
            shot-noise variance $g\,b$, so $b$ is for Poisson backgrounds such
            as stray light. A noise-free offset (electronic bias, the mean of
            dark frames) is subtracted without clipping and kept non-negative.
            A nonnegative scalar or one frame-shaped image in PtyLab's centered
            order, as a NumPy array or tensor. It becomes the model parameter
            `"background"`: fixed by default, estimated (and kept nonnegative)
            when selected in `parameters`. Assigning it or calling `reset()`
            imports it again; otherwise a learned value is retained between runs.
            Initialized to None.

        readoutSigma (float or array_like or None):
            Readout noise $\sigma$ handed to the loss as `read_out_sigma`, in
            the loss's units (detector units for `poisson_loss` and
            `mixed_poisson_gaussian_loss`, photon counts for `amplitude_loss`
            with `anscombe_offset_switch=True`). A nonnegative scalar, used
            for every detector pixel, or one $(N_p, N_p)$ map
            $\sigma(\mathbf q)$ in PtyLab's centered order, for example the
            temporal standard deviation of any number of dark frames whose
            mean is subtracted from the data (it then also holds the
            dark-current shot noise). None leaves the loss's own
            `read_out_sigma`. Initialized to None.

        compile (bool):
            Evaluate the loss through `torch.compile`. Only the real-valued
            loss is compiled: TorchInductor generates no code for complex
            operations, so compiling the complex forward model made it
            slower. Each loss function is compiled once per engine, so a run
            of phases recompiles only when a phase changes the loss. On the
            512/128 CPM fixture the gain is within the noise. Initialized to
            False.

        device (str):
            Torch device of the model. Initialized to `"cuda"` when available,
            else `"cpu"`.
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
        self.memoryBudget = None
        self.dataDevice = None
        self.compile = False
        self.learningRateObject = 0.03
        self.learningRateProbe = 0.1  # scale-free, see probe_scale
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parameters = None
        self.model = None
        self.lossFunction = amplitude_loss
        self.regularizers = []
        self.optimizerType = "adam"
        self.preconditionerAlpha = 0.1
        self.updatePerBatch = False
        self.constraints = []
        self.background = None
        self.readoutSigma = None

    @property
    def background(self):
        """Detector background imported into the model at the next run."""
        return self._background

    @background.setter
    def background(self, value):
        self._background = value
        self._backgroundPending = True

    def validateSettings(self):
        """
        Reject inputs outside the supported CPM contract.

        Raises:
            ValueError:
                If a setting is unsupported or inconsistent.

            NotImplementedError:
                If the data, the propagator or the state counts are not supported
                in this version.
        """
        if self.batchSize is not None and (
            not _is_number(self.batchSize, Integral) or self.batchSize < 1
        ):
            raise ValueError("batchSize must be a positive integer or None.")
        if self.memoryBudget is not None and (
            not _is_number(self.memoryBudget) or not self.memoryBudget > 0
        ):
            raise ValueError("memoryBudget must be a positive number of bytes or None.")
        if self.dataDevice is not None and torch.device(self.dataDevice).type not in (
            "cpu",
            torch.device(self.device).type,
        ):
            raise ValueError('dataDevice must be None, "cpu" or the engine device.')
        if not isinstance(self.compile, (bool, np.bool_)):
            raise ValueError("compile must be True or False.")
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
        # functools.partial keeps bound arguments in .keywords; the engine's
        # read_out_sigma would silently override the loss's.
        if self.readoutSigma is not None and "read_out_sigma" in getattr(
            self.lossFunction, "keywords", {}
        ):
            raise ValueError(
                "Set read_out_sigma in the loss or engine.readoutSigma, not both."
            )
        if self.numIterations < 1:
            raise ValueError("numIterations must be positive.")
        if self.optimizerType not in ("adam", "preconditioned"):
            raise ValueError('optimizerType must be "adam" or "preconditioned".')
        if isinstance(self.constraints, (str, bytes)) or not all(
            callable(constraint) for constraint in self.constraints
        ):
            raise TypeError("constraints must be a list of callables.")
        alpha = self.preconditionerAlpha
        if not _is_number(alpha) or not 0 < alpha <= 1:
            raise ValueError("preconditionerAlpha must lie in (0, 1].")

    fft2c = staticmethod(KernelPropagator.fft2c)
    ifft2c = staticmethod(KernelPropagator.ifft2c)

    def preparePropagation(self):
        """Create fixed propagation kernels on the selected Torch device."""
        self.propagation = KernelPropagator(
            self.reconstruction, self.params.propagatorType, self.device
        )

    def propagate(self, exit_wave):
        """
        Propagate an exit wave while preserving all leading dimensions.

        Args:
            exit_wave (torch.Tensor):
                Complex exit wave with spatial axes last.

        Returns:
            torch.Tensor:
                Complex detector field.
        """
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
        self.model.set_propagation(self.propagation)

    def reset(self):
        """Import the current `Reconstruction` object, probe and background."""
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
        frames = np.asarray(self.experimentalData.ptychogram)
        self._set_probe_scale(frames.sum(dtype=np.float64) / len(frames))
        self._backgroundPending = True
        return self

    def _set_probe_scale(self, photons_per_frame):
        """Rescale a scale-free probe for the data, keeping the physical probe."""
        probe = getattr(self.model, "probe", None)
        if hasattr(probe, "set_scale"):
            probe.set_scale(probe_scale(photons_per_frame, self.reconstruction.Np**2))

    def _physical_probe(self):
        probe = self.model.probe
        if hasattr(probe, "physical_field"):
            return probe.physical_field()
        return probe.field

    def snapshot(self):
        """
        Return detached CPU NumPy copies of the current physical parameters.

        Returns:
            dict:
                Arrays keyed by parameter name. `"probe.field"` is the physical
                probe, not the scale-free parameter. A model background is
                included as `"background"` in centered order.
        """
        if self.model is None:
            return {
                "object": np.array(self.reconstruction.object, copy=True),
                "probe.field": np.array(self.reconstruction.probe, copy=True),
            }
        state = {
            "object": self.model.object.detach().cpu().numpy().copy(),
            "probe.field": self._physical_probe().detach().cpu().numpy().copy(),
        }
        if getattr(self.model, "background", None) is not None:
            state["background"] = self.centeredIntensity(self.model.background)
        return state

    def _sync_reconstruction(self):
        """Publish detached object and probe snapshots for monitoring and saving."""
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
        """
        Validate the public parameter selection and return optimizer groups.

        Returns:
            list:
                One optimizer parameter group per selected parameter.

        Raises:
            KeyError:
                If the selection names unknown parameters or options, or lacks
                an `"lr"`.

            TypeError:
                If the options of a parameter are not a mapping.

            ValueError:
                If a learning rate is not positive and finite, or two names
                refer to the same tensor.
        """
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
            if not _is_number(rate) or not np.isfinite(rate) or rate <= 0:
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
        """Create a fresh optimizer of `optimizerType` for the selected parameters."""
        if self.optimizerType == "preconditioned":
            return torch.optim.SGD(self.parameterGroups(), foreach=False)
        return torch.optim.Adam(self.parameterGroups(), foreach=False)

    def preconditionGradients(self, indices):
        r"""
        Divide gradients by regularized curvature maps of the stepped frames.

        With the curvature map $D$ from `model.gradient_preconditioners`, each
        gradient is divided by

        $$
        \frac{(1 - \alpha)\,D + \alpha\max D}{P},
        $$

        with $\alpha$ = `preconditionerAlpha`, as in rPIE, and $P$ the
        full-scan power that normalizes every loss.

        Args:
            indices (torch.Tensor):
                Frame indices of the step.

        Raises:
            KeyError:
                If the model has no curvature map for a selected parameter.
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
        r"""
        Return the measurements as a tensor after checking them where they live.

        Negative values are allowed: raw offset-subtracted frames with readout
        noise contain them, and losses that need $y \geq 0$ clamp internally.
        """
        r = self.reconstruction
        self.positions = np.asarray(r.positions)
        # A copy, so the engine never aliases ExperimentalData's array.
        frames = np.asarray(self.experimentalData.ptychogram)
        intensity = torch.tensor(
            frames,
            dtype=torch.float32,
            device=data_device(self.device, self.dataDevice, frames.size * 4),
        )
        if intensity.shape != (len(self.positions), r.Np, r.Np) or not len(
            self.positions
        ):
            raise ValueError(
                "Expected one probe-sized diffraction frame per scan position."
            )
        if not torch.isfinite(intensity).all() or not intensity.sum() > 0:
            raise ValueError(
                "Diffraction intensities must be finite and have positive total power."
            )
        if np.any(self.positions < 0) or np.any(self.positions + r.Np > r.No):
            raise ValueError("Scan positions place a probe patch outside the object.")
        return intensity

    def _background_tensor(self):
        """Return `background` as a device tensor."""
        background = torch.as_tensor(self.background, device=self.device)
        if background.dtype == torch.bool or background.is_complex():
            raise TypeError("background must be a real scalar or image.")
        background = background.to(torch.float32)
        frame_shape = self.measuredIntensities.shape[-2:]
        if background.ndim not in (0, 2) or (
            background.ndim == 2 and background.shape != frame_shape
        ):
            raise ValueError(
                f"background must be a scalar or an image of shape {tuple(frame_shape)}."
            )
        if not torch.isfinite(background).all() or (background < 0).any():
            raise ValueError("background must be finite and nonnegative.")
        return background

    def _import_background(self):
        """Copy a newly assigned `background` into the model."""
        if not self._backgroundPending:
            return
        if self.background is None:
            if getattr(self.model, "background", None) is not None:
                self.model.set_background(None)
        elif not hasattr(self.model, "set_background"):
            raise TypeError("A custom model must implement set_background().")
        else:
            self.model.set_background(self._background_tensor())
        self._backgroundPending = False

    def _readout_sigma_tensor(self):
        """Return `readoutSigma` as None, a float, or an `(Np, Np)` device map."""
        sigma = self.readoutSigma
        if sigma is None or np.ndim(sigma) == 0:
            # A scalar stays a Python float, exactly as if bound in the loss.
            return None if sigma is None else float(sigma)
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=self.device)
        # One value per detector pixel; it broadcasts against (batch, Np, Np).
        if sigma.shape != self.measuredIntensities.shape[-2:]:
            raise ValueError("readoutSigma must be a scalar or an (Np, Np) map.")
        if not torch.isfinite(sigma).all() or (sigma < 0).any():
            raise ValueError("readoutSigma must be finite and nonnegative.")
        return sigma

    def prepareReconstruction(self):
        """Prepare data, retained components, propagation and a fresh optimizer."""
        self.validateSettings()
        self.measuredIntensities = self._validate_data()
        self.preparePropagation()
        self._loader = MeasurementLoader(self.measuredIntensities, self.device)
        self.measuredIntensities = data = self._loader.data
        # Summed in double precision, so it is the same wherever the data live.
        self.totalPower = data.sum(dtype=torch.float64).to(self.device, torch.float32)
        self.initializeParameters()
        self._set_probe_scale(self.totalPower.item() / len(self.measuredIntensities))
        self._import_background()
        self._readoutSigma = self._readout_sigma_tensor()
        self.optimizer = self.createOptimizer()
        self.reconstruction.error = []
        self.effectiveBatchSize = self.batchSize
        if self.memoryBudget is not None and torch.device(self.device).type == "cuda":
            self.effectiveBatchSize = self.suggestBatchSize(self.memoryBudget)[
                "batch size"
            ]
        self._sync_reconstruction()
        self._setObjectProbeROI(update=True)
        self._showInitialGuesses()

    def measuredBatch(self, indices):
        """
        Return the measured frames of a batch on `device`.

        Args:
            indices (torch.Tensor):
                Frame indices of the batch.

        Returns:
            torch.Tensor:
                Measured frames, centered.
        """
        return next(self.measuredBatches([indices]))

    def measuredBatches(self, batches):
        """
        Yield the measured frames of each index batch on `device`, in order.

        Host-resident data for a CUDA run are copied while the previous batch
        is computed, see `batching.MeasurementLoader`.

        Args:
            batches (Sequence):
                Index tensors, one per batch.

        Returns:
            Iterator:
                Measured frames of each batch.
        """
        return self._loader(batches)

    def batchIndices(self, indices):
        """
        Split device indices into consecutive batches of `effectiveBatchSize`.

        Args:
            indices (torch.Tensor):
                Frame indices of the scan.

        Returns:
            tuple:
                Index tensors, one per batch.
        """
        return split_indices(indices, self.effectiveBatchSize)

    def suggestBatchSize(self, budget):
        """
        Return the largest batch whose measured iteration peak fits a memory budget.

        The peak of `torch.cuda.max_memory_allocated` over one iteration (data,
        parameters, gradients, optimizer state and temporaries) is measured by
        `batching.iteration_peak` at the sizes `batching.largest_batch_within`
        tries: doubling, then bisection.

        Args:
            budget (float):
                Memory budget in bytes.

        Returns:
            dict:
                `"batch size"`, its `"peak bytes"`, and `"measurements"`, the
                peak of every size tried.

        Raises:
            RuntimeError:
                If `device` is not CUDA.

            MemoryError:
                If one frame per batch already exceeds the budget.

        Notes:
            Each measurement runs two iterations and restores parameters and
            optimizer state. Constraints and `afterStep` do run, and a random
            position order draws from NumPy's global generator.
        """
        if torch.device(self.device).type != "cuda":
            raise RuntimeError(
                "suggestBatchSize measures CUDA memory; device is not CUDA."
            )
        return largest_batch_within(
            budget, len(self.positions), lambda size: iteration_peak(self, size)
        )

    def forward(self, indices):
        """
        Predict the detector intensities of one batch.

        Args:
            indices (torch.Tensor):
                Device-resident frame indices.

        Returns:
            torch.Tensor:
                Predicted intensity of shape `(batch, Np, Np)`.
        """
        return self.model(indices)

    def computeLoss(self, predictedIntensity, measuredIntensity):
        """
        Evaluate the configured data objective, normalized by the full-scan power.

        With `readoutSigma` set, the loss also gets it as `read_out_sigma`:
        the scalar, or the `(Np, Np)` map.

        Args:
            predictedIntensity (torch.Tensor):
                Predicted intensity of a batch.

            measuredIntensity (torch.Tensor):
                Measured intensity of the same batch.

        Returns:
            torch.Tensor:
                Real scalar loss.
        """
        loss = self.lossFunction
        if self.compile:
            cache = self.__dict__.setdefault("_compiledLosses", {})
            if loss not in cache:
                cache[loss] = torch.compile(loss)
            loss = cache[loss]
        sigma = getattr(self, "_readoutSigma", None)
        if sigma is None:
            return loss(predictedIntensity, measuredIntensity, self.totalPower)
        return loss(
            predictedIntensity, measuredIntensity, self.totalPower, read_out_sigma=sigma
        )

    def regularizationLoss(self):
        """
        Return the sum of the regularization penalties of the current model.

        Returns:
            torch.Tensor or None:
                Sum of the `regularizers`, or None if there are none.
        """
        if not self.regularizers:
            return None
        return sum(penalty(self.model) for penalty in self.regularizers)

    def afterStep(self, iteration):
        """
        Hook for subclasses, called after every optimizer step and constraint.

        Args:
            iteration (int):
                Iteration index within the current `reconstruct()` call.
        """
        pass

    def runIteration(self, iteration):
        """
        Accumulate the full-scan objective and take one optimizer step.

        With `updatePerBatch` every batch takes its own step, and each step
        carries the regularization penalty weighted by its share of the frames.
        Constraints and `afterStep` run after every step.

        Args:
            iteration (int):
                Iteration index within the current `reconstruct()` call.

        Returns:
            float:
                Full-scan loss of the iteration.
        """
        self.setPositionOrder()
        total_loss = torch.zeros((), device=self.device)
        indices = torch.as_tensor(
            self.positionIndices, dtype=torch.long, device=self.device
        )
        batches = self.batchIndices(indices)
        steps = [[batch] for batch in batches] if self.updatePerBatch else [batches]
        measured_batches = self.measuredBatches(batches)
        for step in steps:
            self.optimizer.zero_grad(set_to_none=True)
            for batch in step:
                predicted = self.forward(batch)
                measured = next(measured_batches)
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
                background = getattr(self.model, "background", None)
                if background is not None and background.requires_grad:
                    background.clamp_(min=0)
                for constraint in self.constraints:
                    constraint(self, iteration)
                self.afterStep(iteration)

        self.reconstruction.Iestimated = self.centeredIntensity(last_intensity)
        self.reconstruction.Imeasured = self.centeredIntensity(
            self.measuredIntensities[self.positionIndices[-1]]
        )
        return total_loss.item()

    def centeredIntensity(self, intensity):
        """
        Return a detached NumPy copy of a detector image or stack.

        Intensities, measurements, backgrounds and `readoutSigma` maps all use
        PtyLab's centered pixel order, so no reordering is needed. The method
        is kept so existing code that converts for plotting keeps working.

        Args:
            intensity (torch.Tensor):
                Image or stack.

        Returns:
            np.ndarray:
                Copy on the host.
        """
        return intensity.detach().cpu().numpy().copy()

    def _replace_experimental_data(self, experimentalData):
        """
        Replace measurements and positions after checking grid compatibility.

        Raises:
            ValueError:
                If the new data change the acquisition geometry.
        """
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
        """
        Run `numIterations` from the retained component values with a fresh optimizer.

        Args:
            experimentalData (ExperimentalData, optional):
                Replacement data with the same acquisition geometry. If None,
                the current data are used.
        """
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

    def run(self, phases):
        """
        Run phases in order, each with a fresh optimizer.

        Each `Phase` sets its non-None fields on the engine for its own
        iterations, starting from the estimates the previous phase left. Its
        None fields take the engine's own settings, not those of an earlier
        phase. The engine's settings are restored afterwards, and
        `reconstruction.error` holds the errors of all phases in order.

        Args:
            phases (Sequence):
                Nonempty sequence of `Phase` records, for example from
                `blind_recipe`.

        Returns:
            list:
                One dict per phase with its `"name"` and `"error"` history.

        Raises:
            TypeError:
                If `phases` is empty or contains anything but `Phase` records.
        """
        phases = list(phases)
        if not phases or not all(isinstance(phase, Phase) for phase in phases):
            raise TypeError("run() needs a nonempty sequence of Phase records.")
        # Every attribute a phase can set, so the engine's own values come back.
        names = (*_SETTINGS.values(), "numIterations")
        saved = {name: getattr(self, name) for name in names}
        history, errors = [], []
        try:
            for phase in phases:
                # A field left at None means the engine's own value, not the
                # value an earlier phase set.
                for name, value in {**saved, **phase.settings()}.items():
                    setattr(self, name, value)
                self.numIterations = phase.iterations
                try:
                    self.reconstruct()
                finally:
                    # Also keeps the errors of an interrupted phase.
                    history.append(
                        {"name": phase.name, "error": list(self.reconstruction.error)}
                    )
                    errors.extend(self.reconstruction.error)
        finally:
            for name, value in saved.items():
                setattr(self, name, value)
            self.reconstruction.error = errors
        return history
