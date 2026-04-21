# Multislice Reconstruction

## What is multislice?

Standard ptychography treats the sample as a thin, 2D transmission object. For thick samples — where the beam diffracts significantly as it propagates *through* the sample — this approximation breaks down. **Multislice ptychography** models the sample as a stack of thin slices separated by free-space propagation, allowing accurate reconstruction of thick 3D objects.

## When to use multislice

Use multislice when:

- The sample thickness is comparable to or larger than the depth of focus of the probe
- You observe a systematic defocus-like artifact in reconstruction that cannot be corrected by `zPIE`
- You need 3D (depth-resolved) information about the sample

## Setup

Set `nslice > 1` on the `Reconstruction` object and use the `e3PIE` engine:

```python
import PtyLab
from PtyLab import Engines

data, recon, params, monitor, engine = PtyLab.easyInitialize(
    "data.hdf5",
    engine=Engines.e3PIE,
    operationMode="CPM",
)

# Configure multislice
recon.nslice = 4           # number of depth slices
recon.initializeObjectProbe()

engine.numIterations = 100
for loop, posLoop in engine.reconstruct():
    pass

recon.saveResults("multislice_result.hdf5")
```

## Output

With `nslice = N`, the reconstructed object has shape `(nlambda, nosm, npsm, N, No, No)`. Each slice `[:, :, :, i, :, :]` represents one depth layer of the sample.

## Notes

!!! note
    Multislice reconstruction requires more iterations to converge than single-slice. Start with `nslice = 2` and increase if needed.

!!! tip
    The `e3PIE` engine internally handles free-space propagation between slices using the Angular Spectrum propagator. Setting `params.propagatorType` has no effect on inter-slice propagation.

## Related

- [Engines](../cpm/engines.md) — `e3PIE` engine details
- [Configuration Reference](../cpm/configuration.md) — parameter overview
