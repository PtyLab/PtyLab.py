# Data Format

PtyLab.py loads experimental data from **HDF5** files, with all fields at the root of the file.

## CPM required fields

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `ptychogram` | `float32 (N, Nd, Nd)` | counts | Stack of N diffraction intensity patterns |
| `wavelength` | `float64` | m | Illumination wavelength |
| `encoder` | `float64 (N, 2)` | m | Scan positions (x, y) in the sample plane |
| `dxd` | `float64` | m | Detector pixel size |
| `zo` | `float64` | m | Sample-to-detector distance |

## CPM optional fields

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `entrancePupilDiameter` | `float64` | m | Diameter of the probe aperture |
| `spectralDensity` | `float64 (nlambda,)` | m | Wavelengths of a polychromatic beam, one per reconstructed wavelength |
| `theta` | `float64` | rad | Tilt angle for reflection ptychography |
| `emptyBeam` | `float32 (Nd, Nd)` | counts | Reference probe image (for modulus-enforced probe) |

## FPM fields

Required:

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `ptychogram` | `float32 (N, Nd, Nd)` | counts | Stack of N intensity images |
| `wavelength` | `float64` | m | Illumination wavelength |
| `encoder` | `float64 (N, 2)` | m | LED (x, y) positions |
| `dxd` | `float64` | m | Camera pixel size; the object-plane pixel size is `dxd / magnification` |
| `zled` | `float64` | m | LED-array to sample distance |
| `magnification` | `float64` | — | Microscope magnification |

Optional:

| Field | Type | Description |
|-------|------|-------------|
| `NA` | `float64` | Numerical aperture of the objective |

## Loading data

```python
from PtyLab import ExperimentalData

data = ExperimentalData("path/to/data.hdf5", operationMode="CPM")   # or "FPM"
```

Or pass the filename directly to `easyInitialize`:

```python
import PtyLab
experimentalData, reconstruction, params, monitor, engine = PtyLab.easyInitialize(
    "path/to/data.hdf5",
    engine=...,
    operationMode="CPM",
)
```

## Example datasets

Instead of a path, both accept these names:

| Name | Description |
|------|-------------|
| `"example:simulation_cpm"` | Synthetic CPM ptychogram with known object and probe (`simu.hdf5`) |
| `"example:simulation_fpm"` | Measured FPM dataset of a lung carcinoma sample (`LungCarcinomaSmallFPM.hdf5`) |
| `"test:nodata"` | Empty object for unit testing (no diffraction data) |

```python
data = ExperimentalData("example:simulation_cpm", operationMode="CPM")
```

## Your own data

Write your measurements to an HDF5 file with the fields above, using any HDF5 library. All distances are in meters, and `ptychogram` holds intensities (counts), not amplitudes.

## Saving results

`reconstruction.saveResults("result.hdf5")` writes the reconstructed object, probe and related state to an HDF5 file. `type="object"` or `type="probe"` saves only one of them.
