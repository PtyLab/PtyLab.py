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

## Creating an HDF5 file

Write your own measurements to a compatible file with `h5py`:

```python
import h5py
import numpy as np

ptychogram = ...   # shape (N, Nd, Nd), dtype float32
encoder = ...      # shape (N, 2), dtype float64, units: meters

with h5py.File("mydata.hdf5", "w") as f:
    f.create_dataset("ptychogram", data=ptychogram.astype(np.float32))
    f.create_dataset("wavelength", data=np.float64(532e-9))   # 532 nm
    f.create_dataset("encoder", data=encoder.astype(np.float64))
    f.create_dataset("dxd", data=np.float64(6.5e-6))           # 6.5 µm pixels
    f.create_dataset("zo", data=np.float64(50e-3))             # 50 mm distance
```

!!! note
    All distances are in **meters**. The `ptychogram` holds intensities (counts), not amplitudes. Negative values and NaNs are not supported.
