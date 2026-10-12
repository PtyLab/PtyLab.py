# Differentiable Ptychography

`PtyLab.Engines.GradientEngine` implements gradient-based ptychographic reconstruction using the automatic differentiation of PyTorch. A forward model predicts the detector intensity of every scan frame, a loss compares it with the measurements, and an optimizer updates the parameters you select.

!!! warning
    The module is a work in progress. The API described here should stay stable, but this is not guaranteed yet. This page would also update with new changes.

## Scope

- Conventional ptychography (CPM) only.
- Currently does not support multiwavelength, multistate, multislice yet. However this is being implemented in stages. The six-axis object and probe shapes of PtyLab are kept for overall consistency. 
- Propagators: `Fraunhofer`, `Fresnel`, `ASP` and `scaledASP`.
- Centered data (`params.fftshiftSwitch = False`) and the standard intensity constraint.
- PtyLab switches that have no gradient-based counterpart yet (position correction, TV regularization, OPR, momentum and others) are rejected at the start of a run. The full list is `GradientEngine.unsupportedSettings`.

## Installation

PyTorch is an optional dependency, installed with the `torch` extra (see [Installation](../getting-started/installation.md#optional-differentiable-ptychography)):

```bash
uv pip install "ptylab[gpu,torch]" --torch-backend=cu126
```

## Quick start

```python
import PtyLab
from PtyLab.Engines.GradientEngine import GradientEngine

experimentalData, reconstruction, params, monitor, _ = PtyLab.easyInitialize(
    "example:simulation_cpm", operationMode="CPM"
)
params.gpuSwitch = False  # Torch selects its own device, see engine.device

engine = GradientEngine(reconstruction, experimentalData, params, monitor)
engine.numIterations = 200
engine.reconstruct()

reconstruction.saveResults("result.hdf5")
```

The constructor and `reconstruct()` are the same as for the other engines, and the results are written back to `reconstruction.object`, `reconstruction.probe` and `reconstruction.error` as NumPy arrays. Unlike the classic engines, `reconstruct()` runs to completion and does not yield.

The engine runs on `"cuda"` when it is available and on `"cpu"` otherwise. Set `engine.device` to override this. Leave `params.gpuSwitch` off: it controls CuPy, not PyTorch.

## Selecting parameters

`engine.parameters` maps model parameter names to optimizer options. Parameters that are not listed stay fixed.

```python
engine.parameters = {
    "object": {"lr": 0.03},
    "probe.field": {"lr": 0.1},
}
```

| Name | Quantity |
|---|---|
| `"object"` | Complex object |
| `"probe.field"` | Complex probe, scale-free (see below) |
| `"background"` | Detector background, if `engine.background` is set |

`parameters = None` (the default) estimates object and probe with `learningRateObject` (0.03) and `learningRateProbe` (0.1).

The probe is optimized scale-free. `probe.field` stores the probe divided by a fixed power of two that is set from the measured photons per frame, so one probe learning rate works across photon counts. `reconstruction.probe`, `engine.snapshot()` and `model.probe(indices)` stay in physical units. A learning rate `lr` that was tuned for the physical probe corresponds to `lr / engine.model.probe.scale.item()`.

Every call to `reconstruct()` starts a fresh optimizer from the current estimates. If you edit `reconstruction.object` or `reconstruction.probe` between runs, call `engine.reset()` so that the model imports them.

## Optimizers

`engine.optimizerType` is `"adam"` (default) or `"preconditioned"`. The preconditioned optimizer is a PIE-style gradient descent: each gradient is divided by a regularized curvature map (`preconditionerAlpha`, default 0.1) before a plain step. Its learning rates are dimensionless; about 0.3 suits the amplitude loss.

## Losses

Set `engine.lossFunction` to one of the functions in `losses`. Options are passed with `functools.partial`.

```python
from functools import partial
from PtyLab.Engines.GradientEngine.losses import poisson_loss

engine.lossFunction = partial(poisson_loss, read_out_sigma=2.0)
```

| Loss | Noise model | Options |
|---|---|---|
| `amplitude_loss` (default) | Conventional amplitude objective; with `anscombe_offset_switch=True`, the generalized Anscombe transform | `anscombe_offset_switch`, `read_out_sigma` |
| `intensity_loss` | Gaussian with constant variance | |
| `poisson_loss` | Poisson, optionally with Gaussian readout noise | `gain`, `read_out_sigma` |
| `mixed_poisson_gaussian_loss` | Gaussian approximation of Poisson plus readout noise | `gain`, `read_out_sigma` |

All losses expect intensities in detector units, sum over pixels and frames, and are normalized by the total measured power. Measured frames may contain negative values (offset-subtracted raw data). A detector background belongs in the model, not in the data: set `engine.background` to a scalar or a frame-shaped image in PtyLab's centered order. It is fixed by default and estimated (kept nonnegative) when `"background"` is selected in `parameters`.

`poisson_loss` can be negative. The default `Monitor` currently fails to plot a negative error; use `easyInitialize(..., dummyMonitor=True)` or `DummyMonitor()` with this loss.

## Regularizers

`engine.regularizers` is a list of functions that take the model and return a scalar penalty. They are added once per iteration.

```python
from functools import partial
from PtyLab.Engines.GradientEngine.regularizers import object_smoothness

engine.regularizers = [partial(object_smoothness, weight=1e-4)]
```

## Constraints

`engine.constraints` is a list of operations applied in place after every optimizer step, outside the gradient computation.

```python
from PtyLab.Engines.GradientEngine import (
    ObjectAmplitudeClamp,
    PhaseRampRemoval,
    ProbeCentering,
)

engine.constraints = [
    ProbeCentering(threshold=2),
    ObjectAmplitudeClamp(maximum=1.0),
    PhaseRampRemoval(),
]
```

| Constraint | Effect |
|---|---|
| `ProbeCentering` | Rolls probe and object so that the probe's centre of mass stays at the window centre. Leaves Fraunhofer intensities unchanged. |
| `ObjectAmplitudeClamp` | Limits the object amplitude to `[minimum, maximum]`, keeping the phase. |
| `PhaseRampRemoval` | Moves a linear phase ramp and a global phase from the object to the probe. Leaves all predicted intensities unchanged. |

Each constraint takes the schedule arguments `start`, `step`, `end` (iterations) and `relax` (fraction of the correction that is not applied). Constraints that move a parameter also move its optimizer state. Any callable `constraint(engine, iteration)` is accepted, and subclasses of `Constraint` get the schedule for free. Subclasses of `GradientEngine` can still override `afterStep(iteration)`, which runs after the constraints.

## Reconstruction Phases

A reconstruction often needs different settings at different times. `engine.run(phases)` runs a list of `Phase` records in order. Each phase starts a fresh optimizer from the estimates of the previous one, and fields left at None keep the engine's settings.

```python
from PtyLab.Engines.GradientEngine import Phase

engine.run([
    Phase(200, {"object": {"lr": 0.03}}, name="object only"),
    Phase(800, {"object": {"lr": 0.03}, "probe.field": {"lr": 0.1}}),
])
```

`Phase` sets `parameters`, `optimizerType`, `loss` and `constraints`. `run` restores the engine's settings afterwards, writes the errors of all phases to `reconstruction.error`, and returns one history per phase.

`blind_recipe(iterations)` builds two phases for recovering object and probe from a flat start: 20 % of the iterations with the preconditioned optimizer on the amplitude loss, then Adam on the engine's loss.

```python
from PtyLab.Engines.GradientEngine import ProbeCentering, blind_recipe

engine.run(blind_recipe(iterations=1000, constraints=[ProbeCentering()]))
```

## Batching and memory

By default the whole scan is forwarded as one tensor and the optimizer takes one step per iteration.

- `batchSize` limits how many frames are forwarded at once. Gradients are accumulated over batches, so the result does not depend on it; it only bounds memory.
- `memoryBudget` (CUDA only) is a number of bytes. The engine measures and picks the largest batch whose iteration fits within it. `engine.suggestBatchSize(budget)` runs the same search on its own.
- `dataDevice` sets where the measurements are kept. None keeps them on `device` unless they would take more than half of the free GPU memory; `"cpu"` keeps them in pinned host memory and copies each batch while the previous one is computed.
- `updatePerBatch = True` takes one step per batch instead. Combine it with `params.positionOrder = "random"` so that each batch samples the whole scan.
- `compile = True` evaluates the loss through `torch.compile`. Only the real-valued loss is compiled, and the gain is small; it is off by default.

## Custom models

The default model is a `PtychographyModel` that owns the object and a `SharedProbe`. To change the physics, subclass `GradientEngine` and override `createModel()`:

```python
from PtyLab.Engines.GradientEngine import GradientEngine, PtychographyModel, SharedProbe

class MyEngine(GradientEngine):
    def createModel(self):
        r = self.reconstruction
        return MyModel(r.object, SharedProbe(r.probe), r.positions)
```

A custom model is a `torch.nn.Module`. Its `forward(indices)` returns the predicted intensity of the selected frames, and its parameters are selected by their `named_parameters()` names. To support `engine.reset()`, it must implement `reset_from_reconstruction(reconstruction)`. Subclassing `PtychographyModel` and overriding `detector_fields` or `forward` is the simplest way to keep positions, propagation and background handling.

Model tensors have these shapes:

| Quantity | Shape |
|---|---|
| Object | `(nlambda, nosm, 1, nslice, No, No)` |
| Probe | `(nlambda, 1, npsm, 1, Np, Np)` |
| Predicted intensity | `(batch, Np, Np)` |

For the Fraunhofer propagator, predicted intensities and `engine.measuredIntensities` are in unshifted FFT pixel order, which saves two shifts per frame. `engine.centeredIntensity(intensity)` converts to PtyLab's centered order. A custom loss with per-pixel weights or masks must use the same order as the prediction.

## Modules

| Module | Contents |
|---|---|
| `engine` | `GradientEngine` |
| `models` | `PtychographyModel`, `SharedProbe`, `probe_scale` |
| `propagation` | `KernelPropagator`: fixed Fraunhofer, Fresnel, ASP and scaled-ASP kernels |
| `losses` | Data objectives |
| `regularizers` | Penalties on the model |
| `constraints` | `Constraint`, `ProbeCentering`, `ObjectAmplitudeClamp`, `PhaseRampRemoval` |
| `phases` | `Phase`, `blind_recipe` |

The docstrings of each module give the equations and all options.
