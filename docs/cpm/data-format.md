# Data Format

PtyLab.py loads experimental data from **HDF5** files. All fields are stored at the root level of the file.

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
| `spectralDensity` | `float64 (nlambda,)` | — | Relative spectral weights for polychromatic reconstruction |
| `theta` | `float64` | rad | Tilt angle for reflection ptychography |
| `emptyBeam` | `float32 (Nd, Nd)` | counts | Reference probe image (for modulus-enforced probe) |

## Loading data

```python
from PtyLab import ExperimentalData

data = ExperimentalData("path/to/data.hdf5", operationMode="CPM")
data.loadData("path/to/data.hdf5")
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

## Built-in example datasets

PtyLab.py ships with simulation datasets for testing and tutorials. Use these special filename strings:

| Name | Description |
|------|-------------|
| `"example:simulation_cpm"` | Synthetic CPM ptychogram with known object and probe |
| `"example:simulation_fpm"` | Synthetic FPM ptychogram |
| `"test:nodata"` | Empty object for unit testing (no diffraction data) |

```python
data = ExperimentalData(operationMode="CPM")
data.loadData("example:simulation_cpm")
```

## Creating an HDF5 file

Use `h5py` to create a compatible data file from your own measurements:

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
    All distances must be in **meters**. The `ptychogram` should contain raw intensity values (counts), not amplitude. Negative values and NaNs are not supported.
