# Position Correction

## Why correct positions?

Scan position errors are common in practice — mechanical stage imperfections, thermal drift, and vibration all introduce deviations between the nominal encoder positions and the true sample positions. Even small position errors (a fraction of a pixel) degrade reconstruction quality. Position-correcting ptychography estimates and corrects these errors as part of the reconstruction.

## Using `pcPIE`

The `pcPIE` engine extends the standard PIE update with a cross-correlation based position correction step:

```python
import PtyLab
from PtyLab import Engines

data, recon, params, monitor, engine = PtyLab.easyInitialize(
    "data.hdf5",
    engine=Engines.pcPIE,
    operationMode="CPM",
)

params.positionCorrectionSwitch = True
params.positionCorrectionSwitch_radius = 2   # search radius in pixels

engine.numIterations = 100
for loop, posLoop in engine.reconstruct():
    pass

recon.saveResults("corrected_result.hdf5")
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `params.positionCorrectionSwitch` | `False` | Enable position correction |
| `params.positionCorrectionSwitch_radius` | `1` | Search radius (pixels) around each nominal position |

!!! tip
    Start with a small radius (1–2 pixels). Large radii slow down reconstruction and can cause instabilities early in convergence. Run a few iterations without position correction first to let the object and probe stabilize.

## Visualizing position corrections

After reconstruction, inspect how much each position was corrected:

```python
recon.make_alignment_plot(saveit=False)
```

This opens an interactive Bokeh plot comparing the original encoder positions to the corrected positions, with arrows showing the displacement magnitude and direction.

To save the plot to a file:

```python
recon.make_alignment_plot(saveit=True)
```

## Resetting corrections

To discard all position corrections and revert to the original encoder positions:

```python
recon.reset_positioncorrection()
```

This restores `reconstruction.positions` from the original encoder values.

## Corrected positions

The corrected positions are stored in:

```python
recon.encoder_corrected   # corrected physical positions (meters)
recon.positions           # corrected pixel positions
recon.positions0          # original pixel positions (before correction)
```

## Related

- [Engines](../cpm/engines.md) — `pcPIE` and other specialized engines
- [Configuration Reference](../cpm/configuration.md) — full parameter list
