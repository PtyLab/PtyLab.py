# Examples

Worked CPM and FPM reconstructions on simulated and real data.


- [**01 – CPM Simulation**](example_CPM_sim.md) — Simulate a CPM dataset and reconstruct it end-to-end.
- [**02 – CPM Example Data**](example_CPM.md) — Reconstruct a real CPM dataset using `easyInitialize`.
- [**03 – FPM Example Data**](example_FPM.md) — Reconstruct a Fourier ptychography dataset from LED-array images.
- [**04 – CPM Mixed States**](example_mixedStates.md) — Reconstruct partially coherent X-ray data with several probe modes.
- [**05 – CPM Multi-Slice**](example_multiSlice.md) — Reconstruct the layers of a thick sample separately with `e3PIE`.
- [**06 – CPM Multi-Wavelength**](example_CPM_multiWavelength.md) — Handle datasets acquired with multiple illumination wavelengths.
- [**07 – CPM Position Correction**](example_LateralPositionCorrection.md) — Correct lateral probe-position errors during reconstruction.
- [**08 – CPM Purity-Based Calibration**](example_purityCalibration.md) — Calibrate the sample-to-detector distance or the wavelength from the probe purity. *Long run: GPU recommended.*
- [**09 – CPM Reflection**](example_reflectionPtychography.md) — Correct the tilted plane, calibrate the tilt angle with aPIE, and reconstruct reflection data. *Long run: GPU recommended.*
- [**10 – CPM Interferometric Intensity**](example_interferometricIntensityConstraint.md) — Apply an interferometric intensity constraint during reconstruction.
- [**11 – Scan Grids**](example_scanGrids.md) — Generate scan grids, check their step size and optimize the scan path.
- [**12 – CPM Orthogonal Probe Relaxation**](example_OPR.md) — Let the probe vary across the scan to handle a drifting beam.
- [**13 – CPM Differentiable Ptychography**](example_differentiablePtychography.md) — Reconstruct with the PyTorch `GradientEngine`: forward model, loss, optimizer and mixed states.

!!! tip "Stuck?"
    The [User Guide](../cpm/overview.md) explains the [data format](../cpm/data-format.md), the [engines](../cpm/engines.md) and every [`Params` setting](../cpm/configuration.md). The [API Reference](../api/index.md) documents each class and function.