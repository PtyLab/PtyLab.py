# Differentiable Ptychography

`PtyLab.Engines.GradientEngine` implements differentiable ptychography that implements gradient-based optimization for phase-retrieval. This module leverages the automatic differentiation capability of PyTorch. A forward model predicts the detector intensity of every scan frame, a loss compares it with the measurements, and an optimizer updates the parameters you select.

!!! warning
    The module is a work in progress. The API described here should stay stable, but this is not guaranteed yet. This page would also update with new changes.

## Scope

- Conventional ptychography (CPM) only.
- Mixed states: any number of object states (`nosm`) and probe states (`npsm`), see [Mixed states](#mixed-states).
- Currently does not support multiwavelength or multislice yet. However this is being implemented in stages. The unified array layout of PtyLab is maintained for overall consistency in implementation.
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

## Mixed states

Set `reconstruction.nosm` and `reconstruction.npsm` before `initializeObjectProbe()`. Every object state $O_a$ meets every probe state $P_m$, and the intensities of all pairs add:

$$
I_j = \sum_{a}\sum_{m} \left|\mathcal{D}\{O_a(\mathbf{r} + \mathbf{x}_j)\,P_m(\mathbf{r})\}\right|^2 + b .
$$

```python
reconstruction.npsm = 4
reconstruction.initializeObjectProbe()
engine = GradientEngine(reconstruction, data, params, monitor)
engine.reconstruct()
engine.model.probe.mode_powers()  # (nlambda, npsm): sum of |P_m|^2 per mode
```

All probe states share one learning rate and one probe scale. Start the states different from each other: exactly identical states receive identical gradients and stay identical. `initializeObjectProbe()` already adds a little noise to each state, which is enough. A unitary rotation of the states leaves every intensity unchanged, so the stored modes are one basis among many. Add `ProbeModeOrthogonalization` (see [Constraints](#constraints)) to keep them orthogonal and sorted by power, so that `mode_powers()` can be compared between runs:

```python
from PtyLab.Engines.GradientEngine import ProbeCentering, ProbeModeOrthogonalization

engine.constraints = [ProbeCentering(), ProbeModeOrthogonalization(step=10)]
```

PtyLab's `params.orthogonalizationSwitch` is not used by this engine; the constraint replaces it.

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

### Readout noise per pixel

`read_out_sigma` can be bound in the loss as one number. When the noise differs between pixels (hot pixels, detector glow/amp-glow), it is preferable to give it to the engine instead as `engine.readoutSigma`, in the same units as the loss's `read_out_sigma`: either a scalar, used for every detector pixel, or one frame-shaped map $\sigma(\mathbf q)$ in PtyLab's centered order. The map is typically the temporal standard deviation of dark frames recorded with the scan's exposure; any number of dark frames can be used.

```python
dark = ...                                  # (n_dark, N, N) dark frames, centered order
engine.lossFunction = partial(poisson_loss, gain=g)
engine.readoutSigma = dark.std(axis=0)      # one sigma per detector pixel
```

The engine passes the map to the loss as `read_out_sigma`. The loss must have a `read_out_sigma` argument, and must not bind its own: setting it in both places raises an error. `amplitude_loss` takes a map only with `anscombe_offset_switch=True`. A per-pixel standard deviation from $n$ dark frames has a relative error of about $1/\sqrt{2(n-1)}$, so use tens of frames.

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
| `ProbeModeOrthogonalization` | Rotates the probe modes into orthogonal modes, strongest first, by the same snapshot method as PtyLab's `orthogonalizeModes`. Leaves all predicted intensities unchanged. |
| `AbsorbingProbeBoundary` | Damps the probe near the edges of its window by the same super-Gaussian window as PtyLab's `absorbingProbeBoundary`: $P \leftarrow [(1 - a) + aW]P$ with `aleph` $= a$ (default 0.05). Keeps weak modes from drifting into the window corners, where they light object regions nothing else constrains. |

Each constraint takes the schedule arguments `start`, `step`, `end` (iterations) and `relax` (fraction of the correction that is not applied). Constraints that move or scale a parameter also move or scale its optimizer state. PtyLab's `absorbingProbeBoundary` switch stays in `unsupportedSettings`; add `AbsorbingProbeBoundary()` to the list instead. Any callable `constraint(engine, iteration)` is accepted, and subclasses of `Constraint` get the schedule for free. Subclasses of `GradientEngine` can still override `afterStep(iteration)`, which runs after the constraints.

## Reconstruction Phases

A reconstruction often needs different settings at different times. `engine.run(phases)` runs a list of `Phase` records in order. Each phase starts a fresh optimizer from the estimates of the previous one, and fields left at None keep the engine's settings.

```python
from PtyLab.Engines.GradientEngine import Phase

engine.run([
    Phase(200, {"object": {"lr": 0.03}}, name="object only"),
    Phase(800, {"object": {"lr": 0.03}, "probe.field": {"lr": 0.1}}),
])
```

`Phase` sets `parameters`, `optimizerType`, `loss`, `constraints` and `readoutSigma`. A field left at None takes the engine's own setting, not that of an earlier phase. `run` restores the engine's settings afterwards, writes the errors of all phases to `reconstruction.error`, and returns one history per phase.

`blind_recipe(iterations)` builds two phases for recovering object and probe from a flat start: 20 % of the iterations with the preconditioned optimizer on the amplitude loss, then Adam on the engine's loss.

```python
from PtyLab.Engines.GradientEngine import ProbeCentering, blind_recipe

engine.run(blind_recipe(iterations=1000, constraints=[ProbeCentering()]))
```

For raw frames with the detector background in the model, pass the readout noise to the recipe as a scalar or a per-pixel map. It is used by the Adam phase; the start always runs the plain amplitude loss without readout noise:

```python
engine.background = dark.mean(axis=0)
engine.run(
    blind_recipe(
        iterations=1000,
        loss=poisson_loss,
        read_out_sigma=dark.std(axis=0),
        constraints=[ProbeCentering()],
    )
)
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

Every detector-plane array uses PtyLab's centered pixel order: predicted intensities, `engine.measuredIntensities`, `engine.background`, `engine.readoutSigma` and any per-pixel weights or masks in a custom loss. `engine.centeredIntensity(intensity)` returns a NumPy copy, for plotting.

## Modules

| Module | Contents |
|---|---|
| `engine` | `GradientEngine` |
| `models` | `PtychographyModel`, `SharedProbe`, `probe_scale` |
| `propagation` | `KernelPropagator`: fixed Fraunhofer, Fresnel, ASP and scaled-ASP kernels |
| `losses` | Data objectives |
| `regularizers` | Penalties on the model |
| `constraints` | `Constraint`, `ProbeCentering`, `ObjectAmplitudeClamp`, `PhaseRampRemoval`, `ProbeModeOrthogonalization`, `AbsorbingProbeBoundary` |
| `phases` | `Phase`, `blind_recipe` |

The docstrings of each module give the equations and all options; they are rendered in the [Gradient Engine API reference](../api/gradient-engine.md).
