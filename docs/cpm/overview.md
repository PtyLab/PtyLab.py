# CPM Workflow Overview

## What is conventional ptychography?

Conventional Ptychographic Microscopy (CPM) is a computational imaging technique where a coherent probe beam is scanned across a sample at overlapping positions. At each position, a diffraction pattern (intensity only) is recorded. An iterative algorithm then recovers both the complex-valued object (amplitude and phase) and the illumination probe from these intensity measurements alone.

## Data flow

```
HDF5 File
    |
    v
ExperimentalData   (loads diffraction patterns, geometry, scan positions)
    |
    v
Reconstruction     (holds mutable state: object, probe, positions)
    |
    v
Engine.reconstruct()   (iterative phase retrieval: ePIE, mPIE, mqNewton, ...)
    |
    v
Monitor            (real-time visualization of object, probe, error)
```

## Step-by-step reconstruction workflow

### 1. Load experimental data

Diffraction patterns and experimental geometry are loaded from an HDF5 file into an `ExperimentalData` object. The required fields are: `ptychogram` (diffraction intensities), `wavelength`, `encoder` (scan positions), `dxd` (detector pixel size), and `zo` (sample-to-detector distance).

See [Data Format](data-format.md) for the full HDF5 specification.

### 2. Initialize reconstruction state

A `Reconstruction` object is created from the experimental data. It computes coordinate grids and allocates arrays for the object and probe:

- **Object**: complex-valued 2D array representing the sample's transmission function
- **Probe**: complex-valued 2D array representing the illumination beam

Initialization types:

| Type | Description |
|------|-------------|
| `"circ"` | Circular aperture with random noise |
| `"circ_smooth"` | Smoothed circular aperture |
| `"ones"` | All-ones array with random noise |
| `"rand"` | Fully random initialization |
| `"gaussian"` | Gaussian distribution |
| `"upsampled"` | Upsampled from diffraction patterns |

### 3. Configure parameters

The `Params` object holds shared settings: propagator type, constraint switches, regularization options, position ordering, and more. See [Configuration Reference](configuration.md) for all options.

### 4. Choose and configure an engine

An engine implements a specific reconstruction algorithm. All engines share the same interface:

```python
engine = Engines.mPIE(reconstruction, experimentalData, params, monitor)
engine.numIterations = 50
for loop, posLoop in engine.reconstruct():
    pass
```

See [Engines](engines.md) for the full list and selection guide.

### 5. Iterate

During `engine.reconstruct()`, the algorithm loops through scan positions. At each position:

1. Extract the object patch at the current scan position
2. Compute the exit surface wave (ESW): `probe × object_patch`
3. Propagate the ESW to the detector plane
4. Apply the intensity constraint (replace computed amplitude with measured amplitude)
5. Back-propagate to the sample plane
6. Update the object and probe estimates

The `Monitor` shows live plots of reconstruction progress.

### 6. Save results

```python
reconstruction.saveResults("result.hdf5")
```

## Reconstructed output

The reconstructed output is a 6D array of shape `(nlambda, nosm, npsm, nslice, No, No)`:

| Dimension | Meaning |
|-----------|---------|
| `nlambda` | Number of wavelengths (polychromatic reconstruction) |
| `nosm` | Object state mixture (incoherent object modes) |
| `npsm` | Probe state mixture (incoherent probe modes) |
| `nslice` | Depth slices (multislice for thick samples) |
| `No` | Output frame size (pixels) |

## Key concepts

| Term | Description |
|------|-------------|
| **Object** | Complex-valued transmission function of the sample |
| **Probe** | Complex-valued illumination beam at the sample plane |
| **Ptychogram** | Stack of 2D diffraction intensity patterns, one per scan position |
| **Encoder / Scan positions** | Physical (x, y) coordinates where the probe illuminates the sample |
| **Exit surface wave (ESW)** | Product of probe and object patch — the field immediately behind the sample |
| **Propagator** | Method to compute wave propagation between sample and detector planes |
| **Intensity constraint** | Replacing computed amplitude with measured amplitude while retaining the phase |

## Multi-modal capabilities

PtyLab.py supports advanced reconstruction modes by adjusting dimension parameters on the `Reconstruction` object:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `npsm` | 1 | Number of incoherent probe modes (partial coherence) |
| `nosm` | 1 | Number of incoherent object modes |
| `nlambda` | 1 | Number of wavelengths (polychromatic) |
| `nslice` | 1 | Number of object slices (multislice, thick samples) |

!!! note
    Polychromatic reconstruction requires `spectralDensity` in the experimental data. Multislice reconstruction requires setting `nslice > 1` and using the `e3PIE` engine. See [Advanced Topics](../advanced/multislice.md).
