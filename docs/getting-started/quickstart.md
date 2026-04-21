# Quick Start

## Using `easyInitialize`

The fastest way to run a reconstruction is with `easyInitialize()`, which wires up all components from an HDF5 data file:

```python
import PtyLab
from PtyLab import Engines

experimentalData, reconstruction, params, monitor, engine = PtyLab.easyInitialize(
    "path/to/data.hdf5",
    engine=Engines.mPIE,
    operationMode="CPM",
)

engine.numIterations = 50
for loop, posLoop in engine.reconstruct():
    pass

reconstruction.saveResults("result.hdf5")
```

`easyInitialize` returns a 5-tuple for CPM:

| Object | Type | Description |
|--------|------|-------------|
| `experimentalData` | `ExperimentalData` | Diffraction data and geometry from the HDF5 file |
| `reconstruction` | `Reconstruction` | Mutable state: object array, probe array, scan positions |
| `params` | `Params` | Shared configuration (propagator type, constraints, switches) |
| `monitor` | `Monitor` | Real-time visualization during reconstruction |
| `engine` | `BaseEngine` | The reconstruction algorithm instance (e.g. `mPIE`) |

## Using built-in example data

To get started without your own dataset, use the bundled simulation data:

```python
experimentalData, reconstruction, params, monitor, engine = PtyLab.easyInitialize(
    "example:simulation_cpm",
    engine=Engines.ePIE,
    operationMode="CPM",
)
```

Available example datasets:

| Name | Description |
|------|-------------|
| `"example:simulation_cpm"` | Synthetic CPM dataset |
| `"example:simulation_fpm"` | Synthetic FPM dataset |

## Headless mode

For batch processing or server environments without a display:

```python
experimentalData, reconstruction, params, monitor, engine = PtyLab.easyInitialize(
    "data.hdf5",
    engine=Engines.mPIE,
    dummyMonitor=True,
)
```

## Manual initialization

When you need more control (e.g. custom initial probe, multiple modes), skip `easyInitialize` and set up each component explicitly:

```python
from PtyLab import ExperimentalData, Reconstruction, Params, Monitor, Engines

# Load data
experimentalData = ExperimentalData("data.hdf5", operationMode="CPM")

# Configure parameters
params = Params()
params.propagatorType = "ASP"
params.positionOrder = "random"

# Set up visualization
monitor = Monitor()
monitor.figureUpdateFrequency = 5
monitor.objectPlot = "complex"

# Set up reconstruction state
reconstruction = Reconstruction(experimentalData, params)
reconstruction.npsm = 2   # two incoherent probe modes
reconstruction.nosm = 1
reconstruction.nlambda = 1
reconstruction.nslice = 1
reconstruction.initializeObjectProbe()

# Create and configure engine
engine = Engines.mPIE(reconstruction, experimentalData, params, monitor)
engine.numIterations = 50
engine.betaObject = 0.25
engine.betaProbe = 0.25

for loop, posLoop in engine.reconstruct():
    pass

reconstruction.saveResults("result.hdf5")
```

## Saving and loading

```python
# Save the full reconstruction state
reconstruction.saveResults("result.hdf5")

# Reload probe or object from a previous run
reconstruction.load_probe("previous_result.hdf5")
reconstruction.load_object("previous_result.hdf5")
```

## Next steps

- [CPM Workflow Overview](../cpm/overview.md) — understand the full reconstruction pipeline
- [Engines](../cpm/engines.md) — choose the right reconstruction algorithm
- [Configuration Reference](../cpm/configuration.md) — all available `Params` options
- [FPM Workflow](../fpm/overview.md) — Fourier ptychography with LED arrays
- [Tutorial Notebooks](../tutorials/tutorial_CPM_sim.ipynb) — worked examples end to end
