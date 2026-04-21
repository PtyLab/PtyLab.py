# Configuration Reference

The `Params` object holds all shared configuration for the reconstruction. It is created once and passed to both the `Reconstruction` and the engine.

```python
from PtyLab import Params

params = Params()
params.propagatorType = "Fraunhofer"
params.positionOrder = "random"
```

---

## Propagation

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `propagatorType` | `"Fraunhofer"` | `"Fraunhofer"`, `"Fresnel"`, `"ASP"`, `"scaledASP"` | Wave propagation method from sample to detector |
| `fftshiftSwitch` | `False` | `bool` | Pre-shift FFT for slight performance gain; changes coordinate convention |

!!! note
    `"Fraunhofer"` (far-field) is the standard choice for most experiments. Use `"ASP"` (Angular Spectrum) for near-field or when the Fraunhofer approximation breaks down.

---

## Update control

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `probeUpdateStart` | `1` | `int` | Iteration at which probe updates begin |
| `objectUpdateStart` | `1` | `int` | Iteration at which object updates begin |
| `positionOrder` | `"random"` | `"random"`, `"sequential"`, `"NA"` | Order in which scan positions are visited each iteration |

---

## Momentum acceleration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `momentumAcceleration` | `False` | Enable momentum-based acceleration (used by mPIE/mqNewton) |
| `adaptiveMomentumAcceleration` | `False` | Automatically adjust momentum strength during reconstruction |

---

## Object regularization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `objectSmoothenessSwitch` | `False` | Apply smoothness regularization to the object |
| `objectSmoothenessWidth` | `2` | Smoothing kernel width (pixels) |
| `objectSmoothnessAleph` | `1e-2` | Smoothness regularization strength |
| `absObjectSwitch` | `False` | Constrain object to be real-valued (absorption only) |
| `absObjectBeta` | `1e-2` | Relaxation parameter for abs-only constraint |
| `objectTVregSwitch` | `False` | Enable total variation regularization on the object |
| `objectTVfreq` | `5` | How often (iterations) to apply TV update |
| `objectTVregStepSize` | `1e-3` | TV gradient step size |
| `l2reg` | `False` | Enable L2 regularization |
| `l2reg_object_aleph` | `0.001` | L2 regularization strength for the object |

---

## Probe regularization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `probeSmoothenessSwitch` | `False` | Apply smoothness regularization to the probe |
| `probeSmoothenessWidth` | `3` | Smoothing kernel width (pixels) |
| `probeSmoothenessAleph` | `5e-2` | Smoothness regularization strength |
| `probeBoundary` | `False` | Apply a soft boundary constraint to the probe |
| `absorbingProbeBoundary` | `False` | Zero the probe outside its support (hard boundary) |
| `absorbingProbeBoundaryAleph` | `5e-2` | Boundary strength |
| `probePowerCorrectionSwitch` | `False` | Normalize probe power to match measured intensity |
| `modulusEnforcedProbeSwitch` | `False` | Enforce probe modulus from an empty-beam measurement |
| `absProbeSwitch` | `False` | Constrain probe to be real-valued |
| `absProbeBeta` | `1e-2` | Relaxation parameter for abs-only probe constraint |
| `binaryProbeSwitch` | `False` | Enforce a binary probe support |
| `l2reg_probe_aleph` | `0.01` | L2 regularization strength for the probe |

---

## Probe orthogonalization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `orthogonalizationSwitch` | `False` | Orthogonalize incoherent probe modes during reconstruction |
| `orthogonalizationFrequency` | `10` | How often (iterations) to orthogonalize |

---

## Orthogonal Probe Relaxation (OPR)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPR_modes` | `[0]` | Which probe modes to include in OPR relaxation |
| `OPR_subspace` | `4` | Number of subspace vectors for OPR |
| `OPR_alpha` | `0.05` | OPR feedback strength |
| `OPR_orthogonalize_modes` | `True` | Orthogonalize all modes within OPR |

---

## Intensity constraint

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `intensityConstraint` | `"standard"` | `"standard"`, `"sigmoid"` | How to apply the Fourier modulus constraint |

---

## Autofocusing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TV_autofocus` | `False` | Enable TV-based autofocusing |
| `TV_autofocus_metric` | `"TV"` | Focus metric: `"TV"`, `"std"`, or `"min_std"` |
| `TV_autofocus_stepsize` | `5` | Step size (µm) between focus planes |
| `TV_autofocus_range_dof` | `11` | Number of planes to test (should be odd) |
| `TV_autofocus_roi` | `[0.4, 0.6]` | Region of interest (fractional) for metric computation |
| `TV_autofocus_run_every` | `3` | Run autofocus every N iterations |

---

## Other constraints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `couplingSwitch` | `False` | Couple adjacent wavelengths in polychromatic mode |
| `couplingAleph` | `0.5` | Coupling strength |
| `backgroundModeSwitch` | `False` | Estimate and subtract a background term |
| `comStabilizationSwitch` | `False` | Stabilize center of mass of the probe |
| `adaptiveDenoisingSwitch` | `False` | Adaptive noise clipping during intensity projection |
| `CPSCswitch` | `False` | Constrained Pixel Sum Correction |
| `CPSCupsamplingFactor` | — | Upsampling factor for CPSC |

---

## GPU

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpuSwitch` | `False` | Move all arrays to GPU (requires CuPy and a CUDA GPU) |
| `saveMemory` | `False` | Reduce GPU memory usage at slight performance cost |

See [GPU Acceleration](../advanced/gpu.md) for setup and usage.

---

## Output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dump_obj` | `False` | Save the object estimate to disk at every iteration |
