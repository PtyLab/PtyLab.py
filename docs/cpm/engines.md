# Engines

Engines implement the iterative reconstruction algorithms. All engines inherit from `BaseEngine` and share a common interface.

## Available engines

| Engine | Full Name | Best For |
|--------|-----------|----------|
| `ePIE` | Extended PIE | General-purpose starting point, simple and robust |
| `mPIE` | Momentum PIE | Faster convergence than ePIE; recommended default |
| `mqNewton` | Momentum quasi-Newton | High-quality results with faster convergence |
| `qNewton` | Quasi-Newton | Newton-step updates without momentum |
| `multiPIE` | Multi-mode PIE | Multiple probe or object modes simultaneously |
| `mPIE_tv` | mPIE + Total Variation | TV-regularized object reconstruction |
| `ePIE_TV` | ePIE + Total Variation | TV regularization with ePIE updates |
| `zPIE` | Defocus-correcting PIE | Unknown or uncertain sample-detector distance |
| `aPIE` | Angle-correcting PIE | Reflection geometry with uncertain tilt angle |
| `pcPIE` | Position-correcting PIE | Corrects scan position errors during reconstruction |
| `e3PIE` | Enhanced ePIE | Multislice (thick sample) reconstruction |
| `OPR` | Orthogonal Probe Relaxation | Spatially varying probe (e.g. aberrations, drift) |

## Common interface

All engines share the same constructor and iteration pattern:

```python
from PtyLab import Engines

engine = Engines.mPIE(reconstruction, experimentalData, params, monitor)
engine.numIterations = 50
for loop, posLoop in engine.reconstruct():
    pass
```

The `reconstruct()` method is a generator that yields `(loop, posLoop)` at each scan position, allowing you to insert custom logic mid-reconstruction.

## Engine-specific parameters

### ePIE / mPIE

```python
engine.betaObject = 0.25   # object update step size (0 < beta ≤ 1)
engine.betaProbe = 0.25    # probe update step size (0 < beta ≤ 1)
```

### mqNewton / qNewton

These engines use quasi-Newton updates and generally require fewer iterations than PIE-based methods.

### pcPIE (position correction)

```python
params.positionCorrectionSwitch = True
params.positionCorrectionSwitch_radius = 1  # search radius in pixels
```

See [Position Correction](../advanced/position-correction.md) for a full guide.

### e3PIE (multislice)

```python
reconstruction.nslice = 3  # number of depth slices
engine = Engines.e3PIE(reconstruction, experimentalData, params, monitor)
```

See [Multislice](../advanced/multislice.md) for details.

### OPR (Orthogonal Probe Relaxation)

```python
params.OPR_modes = [0]        # which probe modes to relax
params.OPR_subspace = 4       # subspace dimension
params.OPR_alpha = 0.05       # feedback strength
params.OPR_orthogonalize_modes = True
```

## Engine selection guide

```
Is the probe spatially varying across the scan?
  Yes → OPR
  No  → continue

Is the sample thick (multislice)?
  Yes → e3PIE
  No  → continue

Are scan positions unreliable?
  Yes → pcPIE
  No  → continue

Do you want the fastest convergence?
  Yes → mqNewton or mPIE
  No  → ePIE (simplest, most interpretable)
```

## Chaining engines

You can run multiple engines sequentially on the same `Reconstruction` object. A common pattern is to warm up with `ePIE` then refine with `mqNewton`:

```python
# Phase 1: rough convergence
engine1 = Engines.ePIE(reconstruction, experimentalData, params, monitor)
engine1.numIterations = 30
for loop, posLoop in engine1.reconstruct():
    pass

# Phase 2: fine convergence
engine2 = Engines.mqNewton(reconstruction, experimentalData, params, monitor)
engine2.numIterations = 30
for loop, posLoop in engine2.reconstruct():
    pass

reconstruction.saveResults("result.hdf5")
```
