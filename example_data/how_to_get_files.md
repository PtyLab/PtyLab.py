How to get the example datasets
===============================

## Synthetic CPM dataset (`simu.hdf5`)

This file is generated automatically — you do not need to run any script manually.

**From a notebook:** The tutorials `02 – CPM Example Data` and `04 – CPM Position Correction`
generate it on first run via:

```python
from PtyLab.io import getExampleDataFolder, generate_simu_hdf5

filePath = getExampleDataFolder() / "simu.hdf5"
if not filePath.exists():
    generate_simu_hdf5(filePath)
```

**From a script or the REPL:**

```python
from PtyLab.io import getExampleDataFolder, generate_simu_hdf5

generate_simu_hdf5(getExampleDataFolder() / "simu.hdf5")
```

**From the test suite:** `tests/conftest.py` generates `simu.hdf5` automatically before
any test that depends on `"example:simulation_cpm"`.

## Other datasets

Additional datasets (e.g. `LungCarcinomaSmallFPM.hdf5`, `TwoLayer_bin4.hdf5`) are downloaded
automatically by the corresponding tutorials on first run. Place any `.hdf5` files you download
manually into this folder to make them available to the loaders.
