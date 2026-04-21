# FPM Workflow Overview

## What is Fourier ptychography?

Fourier Ptychographic Microscopy (FPM) uses an LED array to illuminate a sample from different angles. Each LED illuminates the sample with a plane wave at a specific angle, producing a low-resolution image on a camera. By combining many such images from different angles, FPM computationally synthesizes a high-resolution, wide-field image — recovering both amplitude and phase — without mechanically scanning the sample.

## Data flow

```
HDF5 File
    |
    v
ExperimentalData   (loads intensity images, LED positions, geometry)
    |
    v
IlluminationCalibration   (optional: refine LED positions/angles)
    |
    v
Reconstruction     (holds mutable state: object, probe/pupil)
    |
    v
Engine.reconstruct()   (iterative phase retrieval in Fourier domain)
    |
    v
Monitor            (real-time visualization)
```

## FPM vs CPM

| | CPM | FPM |
|---|---|---|
| Illumination | Coherent focused probe scanned across sample | LED array: angular illumination scanning |
| Scanning | Sample plane (real space) | Fourier plane (pupil function) |
| Output | Complex object + probe in real space | Complex object + pupil function |
| Resolution | Limited by probe overlap | Limited by LED array NA |

## Step-by-step FPM workflow

### 1. Initialize with `easyInitialize`

```python
import PtyLab
from PtyLab import Engines

data, recon, params, monitor, engine, calib = PtyLab.easyInitialize(
    "fpm_data.hdf5",
    engine=Engines.mqNewton,
    operationMode="FPM",
)
```

`easyInitialize` returns a **6-tuple** for FPM — note the additional `IlluminationCalibration` object:

| Object | Description |
|--------|-------------|
| `experimentalData` | FPM diffraction data and geometry |
| `reconstruction` | Mutable state: object array, pupil |
| `params` | Shared configuration |
| `monitor` | Real-time visualization |
| `engine` | Reconstruction algorithm |
| `calib` | LED calibration helper (`IlluminationCalibration`) |

### 2. Calibrate LED positions (optional but recommended)

Real LED arrays often have small positioning errors. The `IlluminationCalibration` object fits a model to correct these:

```python
calib.fit_mode = "Translation"
calib.runCalibration()
```

See [LED Calibration](calibration.md) for details.

### 3. Reconstruct

```python
recon.initializeObjectProbe()
engine.numIterations = 50
for loop, posLoop in engine.reconstruct():
    pass

recon.saveResults("fpm_result.hdf5")
```

## FPM required data fields

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `ptychogram` | `float32 (N, Nd, Nd)` | counts | Stack of N intensity images |
| `wavelength` | `float64` | m | Illumination wavelength |
| `encoder` | `float64 (N, 2)` | m | LED (x, y) positions |
| `dxd` | `float64` | m | Camera pixel size (in the object plane after magnification) |
| `zled` | `float64` | m | LED-array to sample distance |
| `magnification` | `float64` | — | Microscope magnification |

## FPM optional data fields

| Field | Type | Description |
|-------|------|-------------|
| `NA` | `float64` | Numerical aperture of the objective |

## Recommended engines for FPM

| Engine | Notes |
|--------|-------|
| `mqNewton` | Recommended — fast quasi-Newton updates |
| `mPIE` | Good all-purpose choice with momentum |
| `ePIE` | Simplest; use for debugging |
