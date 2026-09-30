# Quick Start
PtyLab supports both **conventional ptychography (CPM)** and **Fourier ptychography (FPM)**. This quick start uses CPM as the main example.

If you are new to PtyLab, a good place to start a ptychography reconstruction is the example script:
```text
example_scripts/exampleReconstructionCPM.py
```
The script shows a complete CPM reconstruction workflow and can be used as a reference while reading the sections below.

To get started without your own dataset, use the bundled simulation data `simu.hdf5`. If it has not yet been generated， run
```text
example_scripts/simulationData.py
``` 
for example data generation.

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
The alias `"example:simulation_cpm"` resolves to the synthetic CPM dataset `simu.hdf5`.

Available example datasets:

| Name | Description |
|------|-------------|
| `"example:simulation_cpm"` | Synthetic CPM dataset |
| `"example:simulation_fpm"` | Synthetic FPM dataset |


`easyInitialize` returns a 5-tuple for CPM:

| Object | Type | Description |
|--------|------|-------------|
| `experimentalData` | `ExperimentalData` | Diffraction data and experimental geometry from the HDF5 file |
| `reconstruction` | `Reconstruction` | Mutable reconstruction state, including object, probe, and scan positions |
| `params` | `Params` | Shared configuration (propagator type, constraints, switches) |
| `monitor` | `Monitor` | Real-time visualization during reconstruction |
| `engine` | `BaseEngine` | The reconstruction algorithm instance (e.g. `mPIE`) |

A minimal reconstruction can then be run as:

```python
engine.numIterations = 50

for loop, posLoop in engine.reconstruct():
    pass

reconstruction.saveResults("result.hdf5")
```

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
- [Configuration Reference](../cpm/configuration.md) — all available `Params` options
- [Engines](../cpm/engines.md) — choose the right reconstruction algorithm
- [Tutorial Notebooks](../tutorials/tutorial_CPM_sim.ipynb) — worked examples end to end
- [FPM Workflow](../fpm/overview.md) — Fourier ptychography with LED arrays
