# LED Calibration

## Why calibrate?

Physical LED arrays are manufactured with small tolerances — actual LED positions deviate from their nominal values by tens of microns. These errors translate into incorrect illumination angles, which degrade reconstruction quality. The `IlluminationCalibration` class fits a geometric model to correct LED positions before reconstruction.

## Getting the calibration object

`IlluminationCalibration` is returned automatically by `easyInitialize` in FPM mode:

```python
import PtyLab
from PtyLab import Engines

data, recon, params, monitor, engine, calib = PtyLab.easyInitialize(
    "fpm_data.hdf5",
    engine=Engines.mqNewton,
    operationMode="FPM",
)
```

## Running calibration

```python
calib.fit_mode = "Translation"
calib.runCalibration()
```

After `runCalibration()`, the corrected LED positions are propagated into the `ExperimentalData` and `Reconstruction` objects automatically.

## Calibration modes

| Mode | Description |
|------|-------------|
| `"Translation"` | Fits a global (dx, dy) offset to all LED positions |

!!! note
    The `"Translation"` mode corrects for a uniform shift of the entire LED array (e.g. from physical misalignment). For more complex distortions, more advanced calibration may be needed.

## When to skip calibration

If you are working with synthetic (simulated) data or have independently verified your LED positions, you can skip calibration:

```python
# Skip calibration, go straight to reconstruction
recon.initializeObjectProbe()
engine.numIterations = 50
for loop, posLoop in engine.reconstruct():
    pass
```

## Inspecting corrected positions

After calibration, the corrected encoder values are stored on the `Reconstruction` object:

```python
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(*data.encoder.T, label="Original", alpha=0.5)
plt.scatter(*recon.encoder_corrected.T, label="Corrected", alpha=0.5)
plt.legend()
plt.axis("equal")
plt.show()
```
